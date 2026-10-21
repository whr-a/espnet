"""Reject a reward/prompt mismatch before model or distributed initialization."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from espnet2.rl import data
from espnet2.rl.config import ESPnetRLConfig
from espnet2.speechlm.model.speechlm.rl.bagpiper.data import qa_prompt, qa_system_prompt

REWARD = "espnet2.speechlm.model.speechlm.rl.bagpiper.rewards:qa_bagpiper_v2_reward"


def native_source():
    question, choices = "Which animal is audible?", ["Dog", "Cat"]
    return {
        "id": "qa-1",
        "task": "qa",
        "qa_template": "bagpiper",
        "question": question,
        "choices": choices,
        "reference": "Dog",
        "system_prompt": qa_system_prompt("bagpiper"),
        "prompt": qa_prompt(question, choices, "bagpiper"),
    }


def manifest(tmp_path, records):
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def forbidden(*args, **kwargs):
    raise AssertionError("Initialization must not occur before source validation")


def test_generic_optional_reward_hook_runs_only_when_configured(tmp_path, monkeypatch):
    path = manifest(tmp_path, [{"id": "one"}])
    observed = []

    def reward(source, generation):
        return 0

    reward.validate_sources = observed.append
    monkeypatch.setattr(data, "resolve_callable", lambda name: reward)
    records = data.read_manifest(path, config=SimpleNamespace(reward_function="hook"))
    assert observed == [records]
    monkeypatch.setattr(data, "resolve_callable", forbidden)
    assert data.read_manifest(path) == records
    assert (
        data.read_manifest(path, config=SimpleNamespace(reward_function=None))
        == records
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"qa_template": "default"},
        {"system_prompt": None},
        {"system_prompt": "You are a helpful assistant."},
        {"prompt": "Answer only in <answer> tags."},
        {"task": "tts_asr_wer"},
        {"question": None},
        {"choices": ["Dog"]},
        {"reference": "Neither option"},
    ],
)
def test_native_reward_rejects_mismatched_source_without_rewriting(tmp_path, updates):
    record = {**native_source(), **updates}
    path = manifest(tmp_path, [record])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="regenerate.*--qa-template bagpiper"):
        data.read_manifest(path, config=SimpleNamespace(reward_function=REWARD))
    assert path.read_bytes() == before


def test_native_passes_and_west_reward_keeps_its_existing_contract(tmp_path):
    record = native_source()
    path = manifest(tmp_path, [record])
    assert data.read_manifest(path, config=SimpleNamespace(reward_function=REWARD)) == [
        record
    ]
    west = copy.deepcopy(record)
    west.pop("system_prompt")
    west["qa_template"] = "default"
    west["prompt"] = qa_prompt(west["question"], west["choices"])
    path = manifest(tmp_path, [west])
    config = SimpleNamespace(
        reward_function=REWARD.replace(
            "qa_bagpiper_v2_reward", "qa_west_default_reward"
        )
    )
    assert data.read_manifest(path, config=config) == [west]


def config_for(path, mode):
    return ESPnetRLConfig(
        family="bagpiper",
        device="cpu",
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        execution_mode=mode,
        manifest=str(path),
        train_gpus="0",
        rollout_gpus="1" if mode == "disagg_async" else "0",
        reward_function=REWARD,
    )


@pytest.mark.parametrize("mode", ["sync", "disagg_async"])
@pytest.mark.parametrize("validate_only", [False, True])
def test_cli_rejects_native_mismatch_before_engines_or_ray(
    tmp_path, monkeypatch, mode, validate_only
):
    from espnet2.bin import rl_train, rl_train_async
    from espnet2.rl import cli as shared_cli
    from espnet2.rl import miles_library, rollout_worker

    cli = rl_train if mode == "sync" else rl_train_async
    path = manifest(tmp_path, [{**native_source(), "qa_template": "default"}])
    config = config_for(path, mode)
    monkeypatch.setattr(shared_cli.ESPnetRLConfig, "load", lambda path: config)
    monkeypatch.setattr(miles_library, "verify_miles", forbidden)
    monkeypatch.setattr(rollout_worker, "engine_class", forbidden)
    monkeypatch.setitem(sys.modules, "ray", None)
    argv = ["rl_train", "--config", "unused.yaml"]
    if mode == "disagg_async":
        argv += [
            "--ray-address",
            "unused",
            "--train-node",
            "unused",
            "--rollout-node",
            "unused",
        ]
    if validate_only:
        argv.append("--validate-only")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="regenerate.*--qa-template bagpiper"):
        cli.main()


@pytest.mark.parametrize("mode", ["sync", "disagg_async"])
def test_validate_only_accepts_native_sources_without_ray(tmp_path, monkeypatch, mode):
    from espnet2.bin import rl_train, rl_train_async
    from espnet2.rl import cli as shared_cli
    from espnet2.rl import miles_library, rollout_worker

    cli = rl_train if mode == "sync" else rl_train_async
    path = manifest(tmp_path, [native_source(), {**native_source(), "id": "qa-2"}])
    config = config_for(path, mode)
    monkeypatch.setattr(shared_cli.ESPnetRLConfig, "load", lambda path: config)
    monkeypatch.setattr(miles_library, "verify_miles", lambda config: {"checked": True})
    monkeypatch.setattr(rollout_worker, "engine_class", lambda config: None)
    monkeypatch.setitem(sys.modules, "ray", None)
    argv = ["rl_train", "--config", "unused.yaml", "--validate-only"]
    if mode == "disagg_async":
        argv += [
            "--ray-address",
            "unused",
            "--train-node",
            "unused",
            "--rollout-node",
            "unused",
        ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()


def test_direct_trainer_rejects_before_device_or_model_setup(tmp_path, monkeypatch):
    from espnet2.rl import trainer

    path = manifest(tmp_path, [{**native_source(), "system_prompt": None}])
    config = config_for(path, "sync")
    monkeypatch.setattr(trainer, "get_device_name", forbidden)
    monkeypatch.setattr(trainer, "engine_class", forbidden)
    monkeypatch.setattr(trainer, "build_policy", forbidden)
    monkeypatch.setattr(trainer.distributed, "init_process_group", forbidden)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="regenerate.*--qa-template bagpiper"):
        trainer.run(config)
