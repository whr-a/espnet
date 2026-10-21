"""Real two-rank delay controls for rollout group order and object timeouts."""

import copy
import json
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn import functional as F

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


def _timeout_control(rank, root, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=30),
    )
    stage = "new_group"
    started = time.monotonic()
    try:
        # Both original failure modes use real additional Gloo groups on CPU.
        # No NCCL/backend mock or timeout exception stub is involved.
        dist.barrier()
        if mode == "late_group" and rank == 0:
            time.sleep(3)
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=1))
        stage = "broadcast"
        if mode == "short_broadcast" and rank == 0:
            time.sleep(3)
        payload = [{"batch": list(range(256))} if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0, group=group)
    except (RuntimeError, dist.DistError) as error:
        receipt = {
            "rank": rank,
            "mode": mode,
            "failed_stage": stage,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
        }
        (Path(root) / f"rank-{rank}.json").write_text(json.dumps(receipt))
    else:
        raise AssertionError(f"Real small-timeout control unexpectedly passed: {mode}")
    finally:
        dist.destroy_process_group()


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.02, -0.04, 0.01]))

    def checkpoint_state_dict(self):
        return self.state_dict()

    def forward(self, batch):
        logits = self.weight.expand(len(batch), 1, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        mask = batch.batch["response_mask"]
        chosen = log_probs.gather(-1, batch.batch["responses"].unsqueeze(-1)).squeeze(
            -1
        )
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return TokenScores(chosen * mask, entropy * mask, mask)


class _DelayedRollout:
    def __init__(self):
        self.trainer = None
        self.policy = None

    def sync_weights(self, policy, step):
        assert hasattr(self.trainer, "_rollout_object_group")
        self.policy = copy.deepcopy(policy)

    def generate(self, sources, step):
        assert dist.get_rank() == 0
        time.sleep(3)
        log_probs = F.log_softmax(self.policy.weight.detach(), dim=-1).tolist()
        return [
            {
                "source_id": source["id"],
                "prompt_token_ids": [1],
                "token_ids": [candidate % 3],
                "log_probs": [log_probs[candidate % 3]],
                "text": "yes" if candidate % 3 == 0 else "no",
            }
            for source in sources
            for candidate in range(8)
        ]


def _delayed_training(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=30),
    )
    try:
        cfg = ESPnetRLConfig(
            device="cpu",
            reference_device="cpu",
            distributed_strategy="native_allreduce",
            distributed_timeout_minutes=0.2,
            output_dir=root,
            steps=1,
            prompts_per_step=32,
            group_size=8,
            # Two training ranks, one declared device each.
            train_gpus="0,1",
            micro_batch_size=32,
            adapter_factory="unused:factory",
            exporter="unused:exporter",
            sglang_engine="unused:Engine",
        )
        rollout = _DelayedRollout() if rank == 0 else None
        trainer = NativeGRPOTrainer(_Policy(), cfg, rollout=rollout)
        if rollout:
            rollout.trainer = trainer
        group = trainer._object_collective_group()
        assert group is not None and group is not dist.group.WORLD
        assert trainer._object_collective_group() is group
        original_broadcast = dist.broadcast_object_list
        original_gather = dist.all_gather_object
        phases = []

        def record(kind, operation, args, kwargs, caller):
            assert kwargs.get("group") is group
            assert dist.get_backend(group) == "gloo"
            phases.append((kind, caller))
            return operation(*args, **kwargs)

        def broadcast(*args, **kwargs):
            return record(
                "broadcast",
                original_broadcast,
                args,
                kwargs,
                sys._getframe(1).f_code.co_name,
            )

        def gather(*args, **kwargs):
            return record(
                "gather",
                original_gather,
                args,
                kwargs,
                sys._getframe(1).f_code.co_name,
            )

        # Rank zero also delays reward/prepare, while the other rank waits on
        # the same explicit group used for generation and native score objects.
        original_prepare = trainer.prepare_batch

        def prepare(*args, **kwargs):
            assert rank == 0
            time.sleep(3)
            return original_prepare(*args, **kwargs)

        trainer.prepare_batch = prepare
        sources = [{"id": f"p{i:02d}", "reference": "yes"} for i in range(32)]
        initial = trainer.policy.weight.detach().clone()
        with (
            patch.object(dist, "broadcast_object_list", broadcast),
            patch.object(dist, "all_gather_object", gather),
        ):
            trainer.train(sources)
        assert not torch.equal(initial, trainer.policy.weight.detach())
        assert phases == [
            ("broadcast", "train"),
            ("gather", "score_native_batch"),
            ("broadcast", "train_batch"),
            ("broadcast", "record_metrics"),
            ("gather", "save_training_checkpoint"),
            ("broadcast", "save_training_checkpoint"),
        ]
        (Path(root) / f"rank-{rank}.passed").write_text("passed\n")
    finally:
        dist.destroy_process_group()


class TestRolloutObjectCollectives(unittest.TestCase):
    def test_real_late_group_and_short_broadcast_negative_controls(self):
        for mode, stage in (
            ("late_group", "new_group"),
            ("short_broadcast", "broadcast"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                multiprocessing.start_processes(
                    _timeout_control,
                    args=(root, mode),
                    nprocs=2,
                    start_method="spawn",
                    join=True,
                )
                receipts = [
                    json.loads((Path(root) / f"rank-{rank}.json").read_text())
                    for rank in range(2)
                ]
                self.assertTrue(all(item["failed_stage"] == stage for item in receipts))
                self.assertTrue(all(item["error"] for item in receipts))
                print(json.dumps({"control": mode, "receipts": receipts}), flush=True)

    def test_eager_group_actual_32x8_update_with_delayed_rollout_and_reward(self):
        with tempfile.TemporaryDirectory() as root:
            multiprocessing.start_processes(
                _delayed_training,
                args=(root,),
                nprocs=2,
                start_method="spawn",
                join=True,
            )
            self.assertEqual(len(list(Path(root).glob("rank-*.passed"))), 2)
            metrics = json.loads((Path(root) / "metrics.jsonl").read_text())
            self.assertEqual(metrics["step"], 1)
            checkpoint = torch.load(
                Path(root) / "step-000001.pt", map_location="cpu", weights_only=False
            )
            state = checkpoint["trainer_state"]
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(checkpoint["scheduler"]["last_epoch"], 1)
            self.assertEqual(state["world_size"], 2)
            self.assertEqual(state["completed_updates"], 1)
            self.assertEqual(state["next_update"], 1)
            self.assertEqual(state["consumed_prompts"], 32)
            self.assertEqual(state["source_count"], 32)
            self.assertEqual(len(state["rank_rng_states"]), 2)
            for rng in state["rank_rng_states"]:
                self.assertEqual(set(rng), {"python", "numpy", "torch_cpu"})
            self.assertTrue(checkpoint["optimizer"]["state"])
            for adam in checkpoint["optimizer"]["state"].values():
                self.assertEqual(adam["step"].item(), 1)
                self.assertTrue(torch.isfinite(adam["exp_avg"]).all())
                self.assertTrue(torch.isfinite(adam["exp_avg_sq"]).all())
            native = torch.load(
                Path(root) / "step-000001.native.pth", weights_only=True
            )
            self.assertEqual(set(native), set(checkpoint["model"]))
            for name, value in native.items():
                torch.testing.assert_close(
                    value, checkpoint["model"][name], rtol=0, atol=0
                )

    def test_timeout_is_finite_and_configured(self):
        cfg = ESPnetRLConfig(
            adapter_factory="unused:factory",
            exporter="unused:exporter",
            sglang_engine="unused:Engine",
        )
        self.assertEqual(cfg.distributed_timeout_minutes, 4320)
        for invalid in (0, -1, float("nan"), float("inf"), True, "4320"):
            cfg.distributed_timeout_minutes = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                cfg.validate()
