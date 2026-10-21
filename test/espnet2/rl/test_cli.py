"""Canonical entry validation and worker routing without model or Ray startup."""

import json

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
