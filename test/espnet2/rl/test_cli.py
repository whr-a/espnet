"""Canonical entry validation and worker routing without model or Ray startup."""

import json
from types import SimpleNamespace

import pytest

from espnet2.rl import cli
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
    from espnet2.rl import trainer

    monkeypatch.setattr(trainer, "run", lambda config: calls.append(config.manifest))
    cli.main(["--config", config_file(tmp_path), "--worker", "local"])
    assert calls == ["original.jsonl"]


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
    from espnet2.rl import trainer

    observed = []
    monkeypatch.setattr(
        trainer, "run", lambda config: observed.append(config.train_gpus)
    )
    cli.main(["--config", config_file(tmp_path, train_gpus="0,1")])
    assert observed == ["0,1"]


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
    import ray

    from espnet2.rl import miles_async_driver
    from espnet2.rl.rollout import subprocess_environment

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
    cli.run_ray(config, args, [], None)
    assert events == [
        "init",
        ("RolloutActor", {"NCCL_CUMEM_ENABLE": expected}),
        ("ESPnetTrainRayActor", {"NCCL_CUMEM_ENABLE": expected}),
        "shutdown",
    ]
