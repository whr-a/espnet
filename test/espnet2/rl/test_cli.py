"""Canonical entry validation, launch geometry and worker routing, without
model or Ray startup."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import yaml

from espnet2.bin import rl_train as cli
from espnet2.bin.rl_train import launch_command, parser
from espnet2.rl.config import ESPnetRLConfig


def config_file(tmp_path, **overrides):
    config = ESPnetRLConfig(
        **{
            "adapter_factory": "unused:factory",
            "exporter": "unused:exporter",
            "sglang_engine": "unused:Engine",
            "manifest": "original.jsonl",
            **overrides,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    return str(path)


def test_validation_applies_overrides_once_without_launching(
    tmp_path, monkeypatch, capsys
):
    observed = []

    def validate(config, args):
        observed.append((config.manifest, config.checkpoint, args.data_admission))
        return [{"id": "one"}], {"commit": "pinned"}, {"verified": True}

    monkeypatch.setattr(cli, "validate_inputs", validate)
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected launch")
    )
    cli.main(
        [
            "--config",
            config_file(tmp_path),
            "--manifest",
            "override.jsonl",
            "--checkpoint",
            "checkpoint.pt",
            "--data-admission",
            "admission.json",
            "--validate-only",
        ]
    )
    assert observed == [("override.jsonl", "checkpoint.pt", "admission.json")]
    assert json.loads(capsys.readouterr().out)["backend"] == "local"


def test_worker_marker_prevents_recursive_launch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "validate_inputs", lambda *args: ([], {}, None))
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("Recursive launch")
    )
    from espnet2.rl.trainer import local as trainer

    monkeypatch.setattr(trainer, "run", lambda config: calls.append(config.manifest))
    cli.main(["--config", config_file(tmp_path), "--worker", "local"])
    assert calls == ["original.jsonl"]


def test_local_validation_rejects_a_device_off_train_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "validate_inputs", lambda *args: pytest.fail("Unexpected input access")
    )
    path = config_file(tmp_path, device="cuda:0", train_gpus="3", rollout_gpus="3")
    with pytest.raises(ValueError, match="set device: cuda:3"):
        cli.main(["--config", path, "--validate-only"])


def test_ray_validation_takes_train_gpus_as_role_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "validate_inputs", lambda *args: ([], {}, None))
    path = config_file(
        tmp_path,
        family="bagpiper",
        training_objective="west_grpo",
        loss_agg_mode="seq-mean-token-mean",
        top_k=50,
        device="cuda:0",
        train_gpus="1",
        rollout_gpus="0",
    )
    selectors = ["--ray-address", "a", "--train-node", "n", "--rollout-node", "n"]
    cli.main(["--config", path, *selectors, "--validate-only"])


def test_incompatible_worker_fails_before_input_or_runtime_access(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cli, "validate_inputs", lambda *args: pytest.fail("Unexpected input access")
    )
    with pytest.raises(ValueError, match="Worker backend"):
        cli.main(["--config", config_file(tmp_path), "--worker", "ray"])


def test_external_torchrun_enters_worker_without_nested_launch(tmp_path, monkeypatch):
    for name, value in {
        "TORCHELASTIC_RUN_ID": "existing-launch",
        "LOCAL_WORLD_SIZE": "2",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "RANK": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "validate_inputs", lambda *args: ([], {}, None))
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("Nested torchrun")
    )
    from espnet2.rl.trainer import local as trainer

    observed = []
    monkeypatch.setattr(
        trainer, "run", lambda config: observed.append(config.train_gpus)
    )
    cli.main(["--config", config_file(tmp_path, train_gpus="0,1")])
    assert observed == ["0,1"]


def run_until_frozen_config(tmp_path, monkeypatch, device, train_gpus="0"):
    """Run the trainer up to its frozen-config check and return set_device calls."""
    from espnet2.rl.trainer import local as trainer

    calls = []
    monkeypatch.setattr(trainer, "read_manifest", lambda path, config: [])
    monkeypatch.setattr(trainer, "engine_class", lambda config: None)
    monkeypatch.setattr(trainer, "verify_miles", lambda config: {})
    monkeypatch.setattr(trainer, "set_all_random_seed", lambda seed: None)
    monkeypatch.setattr(
        trainer, "get_torch_device", lambda: SimpleNamespace(set_device=calls.append)
    )
    monkeypatch.setattr(trainer.distributed, "init_process_group", lambda *a, **k: None)
    # run() turns off TF32 for float32 runs; keep the flags for later tests.
    monkeypatch.setattr(
        torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32
    )
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32
    )
    # An existing frozen config stops the run right after device selection.
    (tmp_path / "frozen_config.json").write_text("{}")
    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        manifest="unused.jsonl",
        output_dir=str(tmp_path),
        device=device,
        train_gpus=train_gpus,
        rollout_gpus=train_gpus,
    )
    with pytest.raises(FileExistsError, match="fresh output_dir"):
        trainer.run(config)
    return calls, config


# A bare "cuda" is the GPU train_gpus names, as validate() resolves it.
@pytest.mark.parametrize(
    "device,train_gpus,selected",
    [
        ("cuda:3", "3", [torch.device("cuda", 3)]),
        ("cuda", "3", [torch.device("cuda", 3)]),
        ("cpu", "0", []),
    ],
)
def test_single_rank_makes_its_configured_gpu_current(
    tmp_path, monkeypatch, device, train_gpus, selected
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    calls, _ = run_until_frozen_config(tmp_path, monkeypatch, device, train_gpus)
    assert calls == selected


def test_each_torchrun_rank_still_takes_its_local_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    calls, config = run_until_frozen_config(tmp_path, monkeypatch, "cuda:0", "0,1")
    assert calls == [1]
    assert config.device == "cuda:1"


@pytest.mark.parametrize("train_gpus,world_size", [("0", "2"), ("0,1", "4")])
def test_external_torchrun_rejects_wrong_geometry(
    tmp_path, monkeypatch, train_gpus, world_size
):
    for name, value in {
        "TORCHELASTIC_RUN_ID": "existing-launch",
        "LOCAL_WORLD_SIZE": "2",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": world_size,
        "RANK": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        cli, "validate_inputs", lambda *args: pytest.fail("Unexpected worker startup")
    )
    with pytest.raises(ValueError, match="geometry"):
        cli.main(["--config", config_file(tmp_path, train_gpus=train_gpus)])


@pytest.mark.parametrize(
    "caller_value,symmetric_memory,expected",
    [
        (None, False, "0"),
        ("0", False, "0"),
        ("1", False, "1"),
        (None, True, "1"),
        ("0", True, "1"),
        ("1", True, "1"),
    ],
)
def test_ray_roles_share_nccl_mode_before_actor_creation(
    tmp_path, monkeypatch, caller_value, symmetric_memory, expected
):
    pytest.importorskip("ray")
    import ray

    from espnet2.rl.miles import driver as miles_async_driver
    from espnet2.rl.rollout.process import subprocess_environment

    if caller_value is None:
        monkeypatch.delenv("NCCL_CUMEM_ENABLE", raising=False)
    else:
        monkeypatch.setenv("NCCL_CUMEM_ENABLE", caller_value)
    node_id = "1" * 56
    events, job_environment = [], {}

    def initialize(*, address, runtime_env):
        assert address == "unused-address"
        job_environment.update(runtime_env["env_vars"])
        # Importing Miles' whole trainer environment would disable Ray's GPU
        # isolation. Only the allocation policy belongs in this shared contract.
        assert job_environment == {"NCCL_CUMEM_ENABLE": expected}
        events.append("init")

    class Worker:
        def __getattr__(self, name):
            async def call(*args, **kwargs):
                if name == "propose_master_addr_and_port":
                    return "host", 12345
                if name == "get_training_version":
                    return 0
                if name == "get_device_census":
                    return {"rank": 0, "ray_node_id": node_id, "ray_gpu_ids": [0]}

            return SimpleNamespace(remote=call)

    class ActorFactory:
        def __init__(self, cls):
            self.cls = cls

        def options(self, **kwargs):
            assert "runtime_env" not in kwargs
            return self

        def remote(self, *args, **kwargs):
            assert events[0] == "init"
            if self.cls.__name__ == "RolloutActor":
                with monkeypatch.context() as context:
                    context.setenv(
                        "NCCL_CUMEM_ENABLE", job_environment["NCCL_CUMEM_ENABLE"]
                    )
                    child = subprocess_environment(ESPnetRLConfig(**args[0]), "1")
                assert child["NCCL_CUMEM_ENABLE"] == expected
                assert child["CUDA_VISIBLE_DEVICES"] == "1"
            events.append((self.cls.__name__, dict(job_environment)))
            return Worker()

    monkeypatch.setattr(ray, "init", initialize)
    monkeypatch.setattr(ray, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(ray, "remote", lambda **kwargs: ActorFactory)
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": node_id,
                "NodeManagerAddress": "host",
                "Resources": {"GPU": 2},
            }
        ],
    )

    def load_driver(config, components):
        async def driver(args, *, disposer):
            await components.create_rollout_components(args)
            await components.create_training_models(args, None)

        return driver, {}

    monkeypatch.setattr(miles_async_driver, "load_async_loop", load_driver)
    config = ESPnetRLConfig(
        execution_mode="disagg_async",
        train_gpus="0",
        rollout_gpus="1",
        output_dir=str(tmp_path),
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        sglang_engine_kwargs={
            "enable_symm_mem": symmetric_memory,
            "enable_memory_saver": True,
        },
    )
    args = SimpleNamespace(
        ray_address="unused-address", train_node=node_id, rollout_node=node_id
    )
    miles_async_driver.run_ray(config, args, [], None)
    assert events == [
        "init",
        ("RolloutActor", {"NCCL_CUMEM_ENABLE": expected}),
        ("ESPnetTrainRayActor", {"NCCL_CUMEM_ENABLE": expected}),
        "shutdown",
    ]


def frozen_config(tmp_path, mode, *, small=False, name=None, **overrides):
    """Write a recipe yaml whose geometry is only its two device lists.

    The small profile is the one-GPU recipe: a single training rank sharing
    its device with the rollout engine. The full profiles keep the original
    four collocated training ranks with four rollout replicas, and eight of
    each for the disaggregated async node pair.
    """
    devices = (
        dict(train_gpus="0", rollout_gpus="0")
        if small
        else dict(
            train_gpus="0,1,2,3,4,5,6,7" if mode == "disagg_async" else "0,1,2,3",
            rollout_gpus=("0,1,2,3,4,5,6,7" if mode == "disagg_async" else "4,5,6,7"),
        )
    )
    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        execution_mode=mode,
        prompts_per_step=2 if small else 32,
        group_size=4 if small else 8,
        steps=5,
        warmup_steps=0 if small else 1,
        reference_device="cpu",
        manifest="train32.jsonl",
        **{**devices, **overrides},
    )
    path = tmp_path / f"{name or mode}.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    return path


def test_shared_launcher_routes_async_and_collocate_and_keeps_small_sync(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    sync = frozen_config(tmp_path, "sync")
    args = parser().parse_args(
        ["--config", str(sync), "--data-admission", "audit.json"]
    )
    command, environment = launch_command(args, python="pinned-python")
    assert command[:7] == [
        "pinned-python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "espnet2.bin.rl_train",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    args.validate_only = True
    assert launch_command(args, python="pinned-python")[0][:3] == [
        "pinned-python",
        "-m",
        "espnet2.bin.rl_train",
    ]
    async_path = frozen_config(tmp_path, "disagg_async")
    args = parser().parse_args(
        [
            "--config",
            str(async_path),
            "--data-admission",
            "audit.json",
            "--ray-address",
            "ray-address",
            "--train-node",
            "train-node",
            "--rollout-node",
            "rollout-node",
        ]
    )
    assert launch_command(args, python="pinned-python")[0][:3] == [
        "pinned-python",
        "-m",
        "espnet2.bin.rl_train",
    ]
    args.rollout_node = None
    with pytest.raises(ValueError, match="rollout-node"):
        launch_command(args)
    # A one-GPU recipe trains in this process and keeps the pod's device
    # list, so its own `device` selects the GPU.
    small = frozen_config(tmp_path, "sync", small=True, name="small")
    args = parser().parse_args(["--config", str(small)])
    command, environment = launch_command(args, python="pinned-python")
    assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
    assert "torch.distributed.run" not in command
    assert environment.get("CUDA_VISIBLE_DEVICES") == os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )


@pytest.mark.parametrize(
    "train_gpus, rollout_gpus, ranks",
    [
        ("0", "0", 1),
        ("3", "3", 1),
        ("0", "1", 1),
        ("0,1", "2,3", 2),
        ("0,1,2,3", "4,5,6,7", 4),
        ("0,1,2,3,4,5,6,7", "0,1,2,3,4,5,6,7", 8),
    ],
)
def test_launcher_geometry_follows_the_recipe_train_gpus(
    tmp_path, monkeypatch, train_gpus, rollout_gpus, ranks
):
    """One training rank per declared train GPU; torchrun only above one."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    path = frozen_config(
        tmp_path, "sync", train_gpus=train_gpus, rollout_gpus=rollout_gpus
    )
    assert ESPnetRLConfig.load(path).train_ranks == ranks
    args = parser().parse_args(["--config", str(path)])
    command, environment = launch_command(args, python="pinned-python")
    if ranks == 1:
        assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
        assert "torch.distributed.run" not in command
        assert environment.get("CUDA_VISIBLE_DEVICES") == os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
    else:
        assert environment["CUDA_VISIBLE_DEVICES"] == train_gpus
        assert command[:7] == [
            "pinned-python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            "-m",
            "espnet2.bin.rl_train",
        ]


def test_single_rank_launches_inside_a_platform_distributed_environment(
    tmp_path, monkeypatch
):
    """A pytorch job pod exports WORLD_SIZE; one rank must still launch.

    The stale variables are dropped from the child environment, so only a
    nested torchrun is refused.
    """
    for name, value in (
        ("WORLD_SIZE", "2"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "10.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        monkeypatch.setenv(name, value)
    one = frozen_config(tmp_path, "sync", train_gpus="0", rollout_gpus="0", name="one")
    command, environment = launch_command(
        parser().parse_args(["--config", str(one)]), python="pinned-python"
    )
    assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
    assert "torch.distributed.run" not in command
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert name not in environment
    # Several ranks still launch: this toolkit's torchrun builds its own
    # rendezvous from the scrubbed environment and pins the devices.
    many = frozen_config(
        tmp_path, "sync", train_gpus="0,1", rollout_gpus="0", name="many"
    )
    command, environment = launch_command(
        parser().parse_args(["--config", str(many)]), python="pinned-python"
    )
    assert "--nproc-per-node=2" in command
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    for name in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert name not in environment
