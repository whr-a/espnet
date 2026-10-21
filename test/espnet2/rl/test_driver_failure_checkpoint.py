"""A central synchronous driver preserves two real CPU/Gloo learner states."""

import asyncio
import json
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing

import espnet2.rl.miles_components as components_module
import espnet2.rl.trainer as trainer_module
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.miles_async_driver import run_synchronous_loop
from espnet2.rl.miles_components import NativeTrainerGroup
from espnet2.rl.miles_native_actor import ESPnetTrainRayActor
from espnet2.rl.trainer import NativeGRPOTrainer

ERROR = "Synthetic SGLang unsupported finish_reason: length after 2048 frames"


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


def optimizer_update(trainer):
    trainer.policy.weight.grad = torch.full_like(trainer.policy.weight, 0.5)
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def _learner(rank, root, fault, connection):
    """Only transport is replaced: actor/helper/trainer/Gloo are real."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    original_capture = trainer_module.capture_rng_state
    original_save = trainer_module.save_checkpoint
    try:
        trainer = NativeGRPOTrainer(TinyPolicy(), config(root))
        trainer.source_count = 64
        for _ in range(15):
            optimizer_update(trainer)
        trainer.completed_updates = 15
        actor = ESPnetTrainRayActor.__new__(ESPnetTrainRayActor)
        actor._rank, actor._world_size = rank, 2
        actor.config, actor.trainer, actor.version = trainer.config, trainer, 15
        initial_native = trainer.policy.weight.detach().clone()
        initial_master = (
            trainer.optimizer._shadow[0].detach().clone() if rank == 0 else None
        )
        torch.manual_seed(100 + rank)
        connection.send({"status": "Ready", "rank": rank})
        preserved = False
        while True:
            name, args, kwargs = connection.recv()
            if name == "preserve_rollout_failure":
                if fault == "rank1-cursor" and rank == 1:
                    trainer.completed_updates = 14
                if fault == "rank1-capture" and rank == 1:
                    trainer_module.capture_rng_state = lambda device: (
                        _ for _ in ()
                    ).throw(RuntimeError("Synthetic rank1 RNG capture failure"))
                if fault == "owner-disk":
                    trainer_module.save_checkpoint = lambda *a, **k: (
                        _ for _ in ()
                    ).throw(OSError("Synthetic disk full"))
                result = actor.preserve_rollout_failure(*args, **kwargs)
                preserved = True
                assert actor.version == trainer.scheduler.last_epoch == 15
                assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(9.85e-7)
                assert torch.equal(trainer.policy.weight, initial_native)
                if rank == 0:
                    assert torch.equal(trainer.optimizer._shadow[0], initial_master)
                Path(root, f"rank-{rank}.preserved").write_text(json.dumps(result))
                if fault == "rpc-error" and rank == 0:
                    connection.send(
                        {"error": "Synthetic reply failure after preservation"}
                    )
                else:
                    if fault == "rpc-error" and rank == 1:
                        time.sleep(0.08)
                        Path(root, "peer-reply-finished").write_text("finished\n")
                    connection.send({"result": result})
            elif name == "dispose":
                assert preserved, "Driver disposed learner before failure preservation"
                assert all(
                    Path(root, f"rank-{peer}.preserved").is_file() for peer in (0, 1)
                )
                if fault == "rpc-error":
                    assert Path(root, "peer-reply-finished").exists()
                if fault == "normal":
                    # Expected uninterrupted update16 is computed only after
                    # checkpoint15 has completed and disposal has begun.
                    expected_random = torch.rand(4)
                    optimizer_update(trainer)
                    torch.save(
                        {
                            "model": trainer.policy.state_dict(),
                            "optimizer": (
                                trainer.optimizer.state_dict() if rank == 0 else None
                            ),
                            "scheduler": trainer.scheduler.state_dict(),
                            "random": expected_random,
                        },
                        Path(root, f"expected-rank-{rank}.pt"),
                    )
                assert not torch.cuda.is_initialized()
                dist.destroy_process_group()
                connection.send({"result": None})
                break
            else:
                raise AssertionError(
                    f"Rejected generation must not call learner {name}"
                )
    finally:
        trainer_module.capture_rng_state = original_capture
        trainer_module.save_checkpoint = original_save
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class PipeWorker:
    """Awaitable Ray-shaped RPCs to real CPU learner processes."""

    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        async def invoke(*args, **kwargs):
            self.connection.send((name, args, kwargs))
            value = await asyncio.to_thread(self.connection.recv)
            if "error" in value:
                raise RuntimeError(value["error"])
            return value["result"]

        return SimpleNamespace(remote=invoke)


class DriverComponents:
    start_rollout_id = 15

    def __init__(self, group, error, stage):
        self.group, self.error, self.stage = group, error, stage

    def initialize(self, args, *, disposer):
        pass

    async def create_rollout_components(self, args):
        return self, self, None

    async def create_training_models(self, args, rollout):
        return self.group, None

    async def prepare_rollout(self, rollout_id):
        assert rollout_id == 15
        if self.stage == "prepare":
            raise self.error

    async def get(self, rollout_id):
        assert rollout_id == 15
        raise self.error

    async def update_weights(self, *args, **kwargs):
        pass

    async def dispose(self):
        pass


def _resume_worker(rank, root, checkpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/resume-rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        cfg = replace(config(root), resume_checkpoint=checkpoint)
        trainer = NativeGRPOTrainer(TinyPolicy(), cfg)
        trainer.source_count = 64
        state = trainer.load_training_checkpoint(checkpoint)
        assert trainer.completed_updates == trainer.scheduler.last_epoch == 15
        assert state["consumed_prompts"] == 480 and state["next_update"] == 15
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(9.85e-7)
        expected = torch.load(
            Path(root, f"expected-rank-{rank}.pt"), weights_only=False
        )
        assert torch.equal(torch.rand(4), expected["random"])
        optimizer_update(trainer)
        for name, value in trainer.policy.state_dict().items():
            assert torch.equal(value, expected["model"][name])
        assert trainer.scheduler.state_dict() == expected["scheduler"]
        if rank == 0:
            actual = trainer.optimizer.state_dict()
            for value, saved in zip(
                actual["espnet_master_weights"],
                expected["optimizer"]["espnet_master_weights"],
                strict=True,
            ):
                assert torch.equal(value, saved)
            for key, value in actual["state"].items():
                for name, tensor in value.items():
                    assert torch.equal(
                        tensor, expected["optimizer"]["state"][key][name]
                    )
        Path(root, f"rank-{rank}.resumed").write_text("passed\n")
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "fault,stage",
    [
        ("normal", "get"),
        ("normal", "prepare"),
        ("rank1-cursor", "get"),
        ("rank1-capture", "get"),
        ("owner-disk", "get"),
        ("rpc-error", "get"),
    ],
)
def test_central_driver_preserves_real_two_rank_state_before_disposal(
    tmp_path, monkeypatch, fault, stage
):
    from miles.utils.async_utils import with_disposer

    context = multiprocessing.get_context("spawn")
    connections, processes = [], []
    for rank in range(2):
        parent, child = context.Pipe()
        process = context.Process(
            target=_learner, args=(rank, str(tmp_path), fault, child)
        )
        process.start()
        child.close()
        connections.append(parent)
        processes.append(process)
    try:
        for connection in connections:
            assert connection.poll(60) and connection.recv()["status"] == "Ready"
        # No Ray runtime is started. Only Ray kill/transport is replaced; the
        # actual NativeTrainerGroup dispatch/all-reply await remains exercised.
        monkeypatch.setattr(components_module.ray, "kill", lambda *args, **kwargs: None)
        group = NativeTrainerGroup(
            [PipeWorker(connection) for connection in connections]
        )
        original = RuntimeError(ERROR)
        components = DriverComponents(group, original, stage)
        args = SimpleNamespace(
            start_rollout_id=0,
            num_rollout=1000,
            native_eval_interval=100,
            save_interval=100,
        )

        async def execute():
            async def driver(arguments, *, disposer):
                await run_synchronous_loop(arguments, components, disposer=disposer)

            with pytest.raises(RuntimeError) as failure:
                await with_disposer(driver, args)
            assert failure.value is original
            assert original.__notes__
            if fault == "rpc-error":
                assert "Synthetic reply failure" in original.__notes__[0]

        asyncio.run(execute())
        assert group.disposed and args.start_rollout_id == 15
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        receipt_path = next(tmp_path.glob("failure-checkpoints/*/preservation.json"))
        receipt = json.loads(receipt_path.read_text())
        expected_status = "Saved" if fault in ("normal", "rpc-error") else "NotSaved"
        assert receipt["status"] == expected_status
        assert receipt["failure_stage"] == "rollout_generation_before_update"
        assert receipt["original_error"] == "RuntimeError: " + ERROR
        assert receipt["completed_updates"] == receipt["next_update"] == 15
        assert not receipt["optimizer_or_scheduler_stepped"]
        if expected_status == "Saved":
            checkpoint = torch.load(receipt["checkpoint"], weights_only=False)
            state = checkpoint["trainer_state"]
            assert checkpoint["step"] == state["completed_updates"] == 15
            assert state["failure_stage"] == "rollout_generation_before_update"
            assert state["consumed_prompts"] == 480 and state["data_seed"] == 42
            assert state["source_count"] == 64 and len(state["rank_rng_states"]) == 2
            assert checkpoint["optimizer"]["espnet_master_weights_version"] == 1
            assert all(
                value["step"] == 15
                for value in checkpoint["optimizer"]["state"].values()
            )
            if fault == "normal" and stage == "get":
                multiprocessing.start_processes(
                    _resume_worker,
                    args=(str(tmp_path), receipt["checkpoint"]),
                    nprocs=2,
                    start_method="spawn",
                    join=True,
                )
                assert len(list(tmp_path.glob("rank-*.resumed"))) == 2
        else:
            assert not list(tmp_path.glob("failure-checkpoints/*/full-state.pt"))
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


@pytest.mark.parametrize(
    "field,value",
    [("execution_mode", "disagg_async"), ("training_objective", "miles_ppo")],
)
def test_driver_failure_rpc_rejects_other_training_modes(tmp_path, field, value):
    trainer = NativeGRPOTrainer(TinyPolicy(), config(tmp_path))
    actor = ESPnetTrainRayActor.__new__(ESPnetTrainRayActor)
    actor._rank, actor._world_size = 0, 1
    actor.config, actor.trainer, actor.version = trainer.config, trainer, 0
    setattr(actor.config, field, value)
    result = actor.preserve_rollout_failure(0, ERROR)
    assert result["status"] == "NotSaved"
    assert "requires synchronous WEST" in result["rank_admission"][0]["error"]
    assert not list(tmp_path.glob("failure-checkpoints/*/full-state.pt"))
