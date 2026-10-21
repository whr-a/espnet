"""Real Miles actor contract with eight native CPU ranks and frozen rewards.

The worker initializer is bypassed because Ray GPU placement is tested on H100;
the unchanged Miles base class is imported and its ESPnet train method executes
native distributed scoring, global reward preparation and real Adam updates.
"""

import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils import distributed_utils, object_store
from torch import nn
from torch.nn import functional as F

import espnet2.rl.miles_native_actor as actor_module
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.miles_native_actor import BehaviorSnapshot, ESPnetTrainRayActor
from espnet2.rl.rollout import sha256
from espnet2.rl.scoring import score_native_batch
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores

_REWARD_CALLS = 0


def counting_reward(source, generation):
    global _REWARD_CALLS
    _REWARD_CALLS += 1
    assert not dist.is_initialized() or dist.get_rank() == 0
    if source.get("bad_reward"):
        raise ValueError("deliberate reward failure")
    return float(generation["candidate"] % 3 - 1)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.02, -0.04, 0.01]))
        self.register_buffer("frozen", torch.tensor([0.5]))
        self.scored = 0
        self.backward_calls = 0
        self.weight.register_hook(self._backward)
        # Fail once this many candidates have been scored without gradients.
        self.fail = None

    def _backward(self, gradient):
        self.backward_calls += 1
        return gradient

    def checkpoint_state_dict(self):
        return self.state_dict()

    def forward(self, batch):
        if self.fail is not None and self.scored >= self.fail:
            raise ValueError("deliberate local scorer failure")
        if not torch.is_grad_enabled():
            self.scored += len(batch)
        coefficients = torch.tensor(
            [source["coefficient"] for source in batch.non_tensor_batch["source"]]
        )[:, None]
        logits = (coefficients * self.weight).unsqueeze(1).expand(-1, 3, -1)
        logp = F.log_softmax(logits, -1)
        mask = batch.batch["response_mask"]
        selected = logp.gather(-1, batch.batch["responses"].unsqueeze(-1)).squeeze(-1)
        return TokenScores(selected * mask, -(logp.exp() * logp).sum(-1) * mask, mask)


def config(root, mode="disagg_async"):
    return ESPnetRLConfig(
        execution_mode=mode,
        prompts_per_step=32,
        group_size=8,
        # Eight training ranks, one declared device each, as the worker spawns.
        train_gpus="0,1,2,3,4,5,6,7",
        device="cpu",
        optimizer_device="cpu",
        reference_device="cpu",
        distributed_strategy="native_allreduce",
        allreduce_chunk_bytes=16,
        output_dir=str(root),
        micro_batch_size=8,
        learning_rate=0.002,
        steps=2,
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        reward_function=f"{__name__}:counting_reward",
    )


def batch_for(checkpoint, digest):
    sources = [{"id": f"p{i}", "coefficient": 0.1 + i / 20} for i in range(32)]
    generations = [
        {
            "source_id": source["id"],
            "candidate": candidate,
            "prompt_token_ids": [1],
            "token_ids": [candidate % 3, (candidate + 1) % 3, 0],
            "action_mask": [1, 0, 1],
            "log_probs": [0.0, 0.0, 0.0],
            "behavior_policy_version": 0,
            "behavior_checkpoint_sha256": digest,
        }
        for source in sources
        for candidate in range(8)
    ]
    batch = collate_rollouts(sources, generations, 8)
    with torch.no_grad():
        initial = Policy()
        initial.load_state_dict(torch.load(checkpoint, weights_only=True))
        batch.batch["rollout_log_probs"] = initial(batch).log_probs
    return batch


class Store:
    def __init__(self, data):
        self.data = data
        self.calls = 0
        self.fail = False

    def get(self, ref):
        assert not dist.is_initialized() or dist.get_rank() == 0
        self.calls += 1
        if self.fail:
            raise ValueError("deliberate object store read failure")
        return object_store.ObjectStoreGetResult(
            copy.deepcopy(self.data), lambda value: None
        )


class FrozenRollout:
    def __init__(self):
        self.policy = None
        self.calls = 0

    def sync_weights(self, policy, step):
        assert dist.get_rank() == 0
        self.policy = copy.deepcopy(policy)

    def generate(self, sources, step):
        assert dist.get_rank() == 0
        self.calls += 1
        generations = [
            {
                "source_id": source["id"],
                "candidate": candidate,
                "prompt_token_ids": [1],
                "token_ids": [candidate % 3, (candidate + 1) % 3, 0],
                "action_mask": [1, 0, 1],
                "log_probs": [0.0, 0.0, 0.0],
            }
            for source in sources
            for candidate in range(8)
        ]
        temporary = collate_rollouts(sources, generations, 8)
        with torch.no_grad():
            values = self.policy(temporary).log_probs
        for generation, value in zip(generations, values, strict=True):
            generation["log_probs"] = value.tolist()
        return generations


def _actor_worker(rank, root):
    global _REWARD_CALLS
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    distributed_utils.GLOO_GROUP = dist.group.WORLD
    try:
        cfg = config(root)
        checkpoint = Path(root) / "behavior.pth"
        if rank == 0:
            torch.save(Policy().state_dict(), checkpoint)
        dist.barrier()
        digest = sha256(checkpoint)
        data = {
            "batch": batch_for(checkpoint, digest),
            "behavior_policy_version": 0,
            "behavior_checkpoint_sha256": digest,
        }
        store = Store(data)
        object_store._INSTANCE = store
        actor = ESPnetTrainRayActor.__new__(ESPnetTrainRayActor)
        actor._rank, actor._world_size = rank, 8
        actor.config = cfg
        actor.trainer = NativeGRPOTrainer(Policy(), cfg)
        actor.trainer.source_count = 32
        actor.version, actor.last_end, actor.last_metrics = 0, None, None
        actor.snapshot = BehaviorSnapshot(actor.trainer.policy)
        # Rank zero exported the initial policy as version 0.
        actor.exported = {0: digest} if rank == 0 else {}

        def rebuild(*args):
            raise AssertionError("The behavior audit never rebuilds a model")

        actor_module.build_policy = rebuild
        initial = copy.deepcopy(actor.trainer.policy.state_dict())

        def update(step):
            actor.trainer.policy.scored = 0
            actor.trainer.reference.scored = 0
            result = actor.train(step, "rollout")
            assert result.outcome == TrainStepOutcome.NORMAL
            assert actor.version == step + 1
            # A stale batch is scored again by the learner's own module,
            # holding the snapshot of the version that generated it.
            assert actor.trainer.policy.scored == (32 if step == 0 else 64)
            assert actor.trainer.reference.scored == 32
            assert actor.trainer.policy.backward_calls == (step + 1) * 4
            for value in actor.trainer.policy.parameters():
                copies = [torch.empty_like(value) for _ in range(8)]
                dist.all_gather(copies, value.detach())
                for other in copies:
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
            for name, value in actor.trainer.reference.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
            actor.save_model(step)

        update(0)
        # The first update kept the weights it started from as version 0.
        assert actor.snapshot.version == 0
        (saved,) = actor.snapshot.values
        torch.testing.assert_close(saved, initial["weight"], rtol=0, atol=0)

        # The stale update fails on every rank and leaves the learner and the
        # snapshot as they were, wherever the failure happens.
        for failure in (
            "read",
            "hash",
            "snapshot",
            "scorer",
            "behavior",
            "numerical",
            "reward",
        ):
            previous = copy.deepcopy(actor.trainer.policy.state_dict())
            store.data = copy.deepcopy(data)
            store.fail = failure == "read"
            actor.snapshot.version = 7 if failure == "snapshot" else 0
            actor.trainer.policy.scored = 0
            # The behavior pass begins after the 32 current candidates.
            actor.trainer.policy.fail = (
                {"scorer": 0, "behavior": 32}.get(failure) if rank == 3 else None
            )
            if failure == "hash":
                store.data["behavior_checkpoint_sha256"] = "0" * 64
            elif failure == "numerical":
                store.data["batch"].batch["rollout_log_probs"][0, 0] += 1
            elif failure == "reward":
                store.data["batch"].non_tensor_batch["source"][0]["bad_reward"] = True
            try:
                actor.train(1, "rollout")
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"Expected all-rank failure for {failure}")
            assert actor.version == 1
            for name, value in actor.trainer.policy.state_dict().items():
                torch.testing.assert_close(value, previous[name], rtol=0, atol=0)
            torch.testing.assert_close(saved, initial["weight"], rtol=0, atol=0)
            actor.trainer.policy.fail = None
        store.data, store.fail, actor.snapshot.version = data, False, 0
        _REWARD_CALLS = 0
        update(1)
        if rank == 0:
            assert _REWARD_CALLS == 256
            metrics = actor.last_metrics
            assert metrics["rollout/global_candidate_batch_size"] == 256
            # The snapshot reproduces the generating policy exactly.
            assert metrics["rollout/logprob_max_error"] == 0
            assert metrics["rollout/logprob_mean_error"] == 0
            assert metrics["rollout/logprob_p99_error"] == 0
            assert metrics["rollout/weight_staleness"] == 1
            assert metrics["rollout/current_behavior_logprob_max_difference"] > 0
        else:
            assert _REWARD_CALLS == 0 and store.calls == 0
        assert actor.snapshot.version == 1

        # The synchronous recipe likewise shards native scores, and invokes
        # generation/reward once per global batch before broadcasting preparation.
        _REWARD_CALLS = 0
        sync_root = Path(root) / "sync"
        if rank == 0:
            sync_root.mkdir()
        dist.barrier()
        sync_config = config(sync_root, mode="sync")
        sync_rollout = FrozenRollout() if rank == 0 else None
        synchronous = NativeGRPOTrainer(Policy(), sync_config, rollout=sync_rollout)
        sources = [{"id": f"p{i}", "coefficient": 0.1 + i / 20} for i in range(32)]
        synchronous.train(sources)
        assert synchronous.policy.scored == 64
        assert synchronous.reference.scored == 64
        assert synchronous.policy.backward_calls == 8
        assert _REWARD_CALLS == (512 if rank == 0 else 0)
        # Miles PPO now shares WEST's complete collective checkpoint contract.
        full_path = sync_root / "step-000002.pt"
        full = torch.load(full_path, map_location="cpu", weights_only=False)
        assert full["scheduler"]["last_epoch"] == 2
        state = full["trainer_state"]
        assert state["next_update"] == 2 and state["consumed_prompts"] == 64
        assert len(state["rank_rng_states"]) == 8
        restored_config = copy.deepcopy(sync_config)
        restored_config.output_dir = str(sync_root / "restored")
        restored = NativeGRPOTrainer(Policy(), restored_config)
        restored.source_count = 32
        restored.load_training_checkpoint(full_path)
        assert restored.completed_updates == synchronous.completed_updates == 2
        for name, value in restored.policy.state_dict().items():
            torch.testing.assert_close(
                value, synchronous.policy.state_dict()[name], rtol=0, atol=0
            )
        assert restored.scheduler.state_dict() == synchronous.scheduler.state_dict()
        if rank == 0:
            assert sync_rollout.calls == 2
        (Path(root) / f"rank-{rank}.passed").write_text("passed\n")
    finally:
        object_store._INSTANCE = None
        distributed_utils.GLOO_GROUP = None
        dist.destroy_process_group()


class TestMilesNativeActor(unittest.TestCase):
    def test_prepared_native_scores_do_not_rescore_and_require_detached_identity(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "behavior.pth"
            torch.save(Policy().state_dict(), checkpoint)
            digest = sha256(checkpoint)
            cfg = config(root)
            trainer = NativeGRPOTrainer(Policy(), cfg)
            behavior = Policy().requires_grad_(False)
            batch = batch_for(checkpoint, digest)
            native = batch.batch["rollout_log_probs"].clone()
            trainer.prepare_async_batch(
                batch,
                behavior,
                behavior_version=0,
                learner_version=1,
                behavior_checkpoint_sha256=digest,
                native_scores=(native - 0.1, native, native.clone()),
            )
            self.assertEqual(trainer.policy.scored, 0)
            self.assertEqual(trainer.reference.scored, 0)
            self.assertEqual(behavior.scored, 0)
            with self.assertRaisesRegex(ValueError, "detached"):
                trainer.prepare_async_batch(
                    batch,
                    behavior,
                    behavior_version=0,
                    learner_version=1,
                    behavior_checkpoint_sha256=digest,
                    native_scores=(native.clone().requires_grad_(), native, native),
                )
            with self.assertRaisesRegex(ValueError, "differ from learner"):
                trainer.prepare_async_batch(
                    batch,
                    behavior_version=0,
                    learner_version=0,
                    behavior_checkpoint_sha256=digest,
                    native_scores=(native - 0.1, native, native),
                )

    def test_eight_rank_actor_updates_global_rewards_and_collective_failures(self):
        with tempfile.TemporaryDirectory() as root:
            multiprocessing.start_processes(
                _actor_worker,
                args=(root,),
                nprocs=8,
                start_method="spawn",
                join=True,
            )
            self.assertEqual(len(list(Path(root).glob("rank-*.passed"))), 8)
            checkpoint = torch.load(Path(root) / "step-000002.pt", weights_only=False)
            self.assertEqual(checkpoint["step"], 2)
            self.assertTrue(checkpoint["optimizer"]["state"])
            for state in checkpoint["optimizer"]["state"].values():
                self.assertEqual(state["step"].item(), 2)
            saved = json.loads(
                (Path(root) / "metrics.jsonl").read_text().splitlines()[-1]
            )
            self.assertEqual(saved["runtime/train_world_size"], 8)


class TestBehaviorSnapshot(unittest.TestCase):
    def test_weight_swap_audit_equals_a_separately_loaded_behavior_model(self):
        class Scorer(Policy):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor([0.3, 0.0, -0.3]))
                self.bias.requires_grad_(False)

            def forward(self, batch):
                scores = super().forward(batch)
                return TokenScores(
                    scores.log_probs + self.bias[batch.batch["responses"]],
                    scores.entropy,
                    scores.response_mask,
                )

        with tempfile.TemporaryDirectory() as root:
            cfg = config(root)
            learner = Scorer()
            trainer = NativeGRPOTrainer(learner, cfg)
            snapshot = BehaviorSnapshot(learner)
            snapshot.take(0)
            behavior = Scorer()
            behavior.load_state_dict(learner.state_dict())
            # The learner moves on to version 1.
            with torch.no_grad():
                learner.weight.add_(torch.tensor([0.5, -0.2, 0.1]))
            current = copy.deepcopy(learner.state_dict())
            checkpoint = Path(root) / "rollout.pth"
            torch.save(Policy().state_dict(), checkpoint)
            batch = batch_for(checkpoint, "a" * 64)
            with torch.no_grad():
                expected = trainer._score_chunks(behavior, batch)
                expected_current = trainer._score_chunks(learner, batch)
            old, audited, _ = score_native_batch(trainer, batch, snapshot)
            torch.testing.assert_close(audited, expected, rtol=0, atol=0)
            torch.testing.assert_close(old, expected_current, rtol=0, atol=0)
            self.assertFalse(torch.equal(old, audited))
            # The learner holds its own weights again, and the snapshot only
            # the trainable ones, ready for the next audit.
            for name, value in learner.state_dict().items():
                torch.testing.assert_close(value, current[name], rtol=0, atol=0)
            (saved,) = snapshot.values
            torch.testing.assert_close(saved, behavior.weight, rtol=0, atol=0)
            # A failure while it holds the snapshot still gives them back.
            learner.fail = learner.scored
            with self.assertRaisesRegex(ValueError, "deliberate"):
                trainer._score_chunks(snapshot, batch)
            for name, value in learner.state_dict().items():
                torch.testing.assert_close(value, current[name], rtol=0, atol=0)
            torch.testing.assert_close(saved, behavior.weight, rtol=0, atol=0)

    def test_snapshot_values_are_views_of_one_buffer_per_dtype(self):
        model = nn.Module()
        model.a = nn.Parameter(torch.randn(3, 2))
        model.b = nn.Parameter(torch.randn(4).to(torch.bfloat16))
        model.c = nn.Parameter(torch.randn(5))
        model.frozen = nn.Parameter(torch.randn(2), requires_grad=False)
        snapshot = BehaviorSnapshot(model)
        snapshot.take(3)
        a, b, c = snapshot.values
        storage = a.untyped_storage()
        self.assertEqual(storage.data_ptr(), c.untyped_storage().data_ptr())
        self.assertEqual(storage.nbytes(), (6 + 5) * 4)
        self.assertEqual(b.untyped_storage().nbytes(), 4 * 2)
        for value, parameter in zip(snapshot.values, (model.a, model.b, model.c)):
            torch.testing.assert_close(value, parameter.detach(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
