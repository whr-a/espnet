"""CPU checkpoint continuation and two-rank pre-update failure handling."""

import asyncio
import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing

import espnet2.rl.miles_native_actor as actor_module
import espnet2.rl.trainer as trainer_module
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.failure_checkpoint import preserve_preupdate_failure
from espnet2.rl.miles_components import NativeTrainerGroup
from espnet2.rl.miles_native_actor import ESPnetTrainRayActor
from espnet2.rl.trainer import NativeGRPOTrainer

ERROR = "Rollout/native max logprob disagreement 2.84811 > 1.5"


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([0.25, -0.125], dtype=torch.bfloat16)
        )

    def checkpoint_state_dict(self):
        return self.state_dict()


def config(root):
    return ESPnetRLConfig(
        family="bagpiper",
        device="cpu",
        reference_device="cpu",
        dtype="bfloat16",
        optimizer_device="cpu",
        execution_mode="disagg_sync",
        distributed_strategy="native_allreduce",
        distributed_timeout_minutes=1,
        train_gpus="0,1",
        max_weight_staleness=0,
        prompts_per_step=32,
        group_size=8,
        steps=1000,
        learning_rate=1e-6,
        lr_scheduler="linear",
        shuffle_train_data=True,
        training_objective="west_grpo",
        loss_agg_mode="seq-mean-token-mean",
        kl_type="k3",
        output_dir=str(root),
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )


def completed_trainer(root):
    trainer = NativeGRPOTrainer(TinyPolicy(), config(root))
    trainer.source_count = 64
    # Reach a real Adam/scheduler state without model scoring or inference.
    for _ in range(15):
        optimizer_update(trainer)
    trainer.completed_updates = 15
    return trainer


def optimizer_update(trainer):
    trainer.policy.weight.grad = torch.full_like(trainer.policy.weight, 0.5)
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def actor_for(trainer, rank=0, world_size=1):
    actor = ESPnetTrainRayActor.__new__(ESPnetTrainRayActor)
    actor._rank, actor._world_size = rank, world_size
    actor.config, actor.trainer = trainer.config, trainer
    actor.version, actor.snapshot = 15, None
    actor.exported = {15: "a" * 64} if rank == 0 else {}
    return actor


def test_full_failure_checkpoint_resumes_masters_moments_rng_cursor_and_lr(tmp_path):
    trainer = completed_trainer(tmp_path)
    native = trainer.policy.weight.detach().clone()
    masters = trainer.optimizer._shadow[0].detach().clone()
    rng = torch.get_rng_state().clone()
    result = preserve_preupdate_failure(actor_for(trainer), 15, ERROR)
    assert result["status"] == "Saved"
    assert trainer.completed_updates == trainer.scheduler.last_epoch == 15
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(9.85e-7)
    assert torch.equal(trainer.policy.weight, native)
    assert torch.equal(trainer.optimizer._shadow[0], masters)
    checkpoint = torch.load(
        result["checkpoint"], map_location="cpu", weights_only=False
    )
    state = checkpoint["trainer_state"]
    assert (
        checkpoint["step"] == state["completed_updates"] == state["next_update"] == 15
    )
    assert state["consumed_prompts"] == 480
    assert state["data_order"] == "torch_randperm_seed_plus_epoch_v1"
    assert state["rank_rng_states"][0]["torch_cpu"].equal(rng)
    assert checkpoint["optimizer"]["espnet_master_weights_version"] == 1
    assert checkpoint["optimizer"]["espnet_master_weights"][0].equal(masters)
    assert all(
        value["step"] == 15 for value in checkpoint["optimizer"]["state"].values()
    )
    restored_config = replace(trainer.config, resume_checkpoint=result["checkpoint"])
    restored = NativeGRPOTrainer(TinyPolicy(), restored_config)
    restored.source_count = 64
    restored.load_training_checkpoint(result["checkpoint"])
    assert restored.completed_updates == 15
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(9.85e-7)
    optimizer_update(trainer)
    optimizer_update(restored)
    assert torch.equal(trainer.policy.weight, restored.policy.weight)
    assert torch.equal(trainer.optimizer._shadow[0], restored.optimizer._shadow[0])
    assert trainer.scheduler.state_dict() == restored.scheduler.state_dict()
    assert not torch.cuda.is_initialized()


def test_existing_checkpoint_and_receipt_are_preserved(tmp_path):
    trainer = completed_trainer(tmp_path)
    actor = actor_for(trainer)
    saved = preserve_preupdate_failure(actor, 15, ERROR)
    checkpoint = Path(saved["checkpoint"])
    receipt = checkpoint.parent / "preservation.json"
    original_checkpoint, original_receipt = (
        checkpoint.read_bytes(),
        receipt.read_bytes(),
    )
    rejected = preserve_preupdate_failure(actor, 15, ERROR)
    assert rejected["status"] == "NotSaved"
    assert checkpoint.read_bytes() == original_checkpoint
    assert receipt.read_bytes() == original_receipt


def test_receipt_failure_remains_secondary(tmp_path, monkeypatch):
    trainer = completed_trainer(tmp_path)

    def fail_receipt(*args, **kwargs):
        raise OSError("Synthetic receipt write failure")

    monkeypatch.setattr("espnet2.rl.failure_checkpoint.json.dump", fail_receipt)
    result = preserve_preupdate_failure(actor_for(trainer), 15, ERROR)
    assert result["original_error"] == ERROR
    assert (
        "Synthetic receipt write failure" in result["rank_receipt_status"][0]["error"]
    )


def _install_prepare_failure(actor):
    data = {
        "batch": SimpleNamespace(
            non_tensor_batch={"generation": [{"behavior_hash_kind": "native-file/v1"}]}
        ),
        "behavior_policy_version": 15,
        "behavior_checkpoint_sha256": "a" * 64,
    }
    actor_module.object_store.get_instance = lambda: SimpleNamespace(
        get=lambda ref: nullcontext(data)
    )
    actor_module.score_native_batch = lambda *args: None

    def reject(*args):
        raise RuntimeError(ERROR)

    def forbidden_update(*args):
        raise AssertionError("The rejected batch must never reach update_actor")

    actor._prepare_synchronous_batch = reject
    actor.trainer.update_actor = forbidden_update


def _failure_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    original_capture = trainer_module.capture_rng_state
    original_serializer = trainer_module.save_checkpoint
    try:
        trainer = completed_trainer(root)
        actor = actor_for(trainer, rank, 2)
        _install_prepare_failure(actor)
        previous = copy.deepcopy(trainer.policy.state_dict())
        torch.manual_seed(100 + rank)
        for fault in ("normal", "rank1-cursor", "rank1-capture", "owner-disk"):
            directory = Path(root, fault)
            if rank == 0:
                directory.mkdir()
            dist.barrier()
            actor.config.output_dir = str(directory)
            trainer.completed_updates = (
                14 if fault == "rank1-cursor" and rank == 1 else 15
            )

            def capture(device):
                if fault == "rank1-capture" and rank == 1:
                    raise RuntimeError("Synthetic rank1 RNG capture failure")
                return original_capture(device)

            def serialize(*args, **kwargs):
                if fault == "owner-disk":
                    raise OSError("Synthetic owner disk full")
                return original_serializer(*args, **kwargs)

            trainer_module.capture_rng_state = capture
            trainer_module.save_checkpoint = serialize
            try:
                actor.train(15, "rollout")
            except RuntimeError as error:
                assert str(error) == "RuntimeError: " + ERROR
                assert error.__notes__
            else:
                raise AssertionError("The original prepared-batch error must propagate")
            assert actor.version == trainer.scheduler.last_epoch == 15
            assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(9.85e-7)
            assert not Path(directory, "metrics.jsonl").exists()
            for name, value in trainer.policy.state_dict().items():
                torch.testing.assert_close(value, previous[name], rtol=0, atol=0)
            dist.barrier()
            if rank == 0:
                receipt = next(
                    directory.glob("failure-checkpoints/*/preservation.json")
                )
                value = json.loads(receipt.read_text())
                assert value["status"] == ("Saved" if fault == "normal" else "NotSaved")
                if fault == "normal":
                    checkpoint = torch.load(value["checkpoint"], weights_only=False)
                    state = checkpoint["trainer_state"]
                    assert state["completed_updates"] == state["next_update"] == 15
                    assert state["consumed_prompts"] == 480
                    assert len(state["rank_rng_states"]) == 2
                    assert not torch.equal(
                        state["rank_rng_states"][0]["torch_cpu"],
                        state["rank_rng_states"][1]["torch_cpu"],
                    )
                else:
                    assert not list(
                        directory.glob("failure-checkpoints/*/full-state.pt")
                    )
        Path(root, f"rank-{rank}.passed").write_text("passed\n")
    finally:
        trainer_module.capture_rng_state = original_capture
        trainer_module.save_checkpoint = original_serializer
        dist.destroy_process_group()


def test_two_rank_prepare_failure_checkpoint_and_error_collectives(tmp_path):
    multiprocessing.start_processes(
        _failure_worker,
        args=(str(tmp_path),),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    assert len(list(tmp_path.glob("rank-*.passed"))) == 2


def test_driver_waits_for_all_rank_errors_before_cleanup():
    events = []
    original = RuntimeError(ERROR)

    class Worker:
        def __init__(self, rank):
            async def run():
                if rank:
                    await asyncio.sleep(0.03)
                    events.append("peer-finished-preservation")
                    raise RuntimeError("peer original error")
                raise original

            self.train = SimpleNamespace(remote=lambda *args, **kwargs: run())

    async def verify():
        group = NativeTrainerGroup([Worker(0), Worker(1)])
        with pytest.raises(RuntimeError) as failure:
            await group.call("train", 15, None)
        assert failure.value is original
        events.append("cleanup-may-start")

    asyncio.run(verify())
    assert events == ["peer-finished-preservation", "cleanup-may-start"]
