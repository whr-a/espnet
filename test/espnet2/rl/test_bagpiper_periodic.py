"""Periodic evaluation preserves failures and reuses its actor's reservation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from espnet2.rl.rollout import sha256
from espnet2.speechlm.model.speechlm.rl.bagpiper import periodic


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path, monkeypatch, fail=None):
    plan = tmp_path / "plan.json"
    dump(
        plan,
        {
            "expected_steps": list(range(100, 1001, 100)),
            "input_sha256": {},
            "manifest": "frozen1000.jsonl",
        },
    )
    native = tmp_path / "training/step-000100.native.pth"
    native.parent.mkdir()
    native.write_bytes(b"immutable-native-checkpoint")
    config = SimpleNamespace(
        output_dir=str(native.parent),
        rollout_python=sys.executable,
        periodic_eval_options={
            "plan": str(plan),
            "plan_sha256": sha256(plan),
            "output_dir": str(tmp_path / "evaluations"),
            "resource_preflight_command": ["resource-admission"],
            "evaluation_config": "frozen-baseline.yaml",
            "audit_script": "audit-periodic.py",
        },
    )
    policy = {"step": 100, "native_sha256": "f" * 64}
    identity = {"replica": 0, "gpu_ids": [0], "gpu": "0"}
    events = []
    monkeypatch.setattr(
        periodic,
        "subprocess_environment",
        lambda config, gpu: {"CUDA_VISIBLE_DEVICES": gpu},
    )

    def run(command, *, environment, log, receipt):
        if command[0] == "resource-admission":
            phase = "restore-admission" if "restore_" in str(log) else "eval-admission"
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
            dump(command[command.index("--output") + 1], {"status": "Accepted"})
        elif command[1] == "-m":
            phase = "decode"
            assert environment["CUDA_VISIBLE_DEVICES"] == "0"
            assert command[command.index("--policy-step") + 1] == "100"
            out = Path(command[command.index("--output") + 1])
            for name in ("responses.json", "protocol.json", "summary.json"):
                dump(out / name, {"fixture": name})
        else:
            phase = "audit"
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
            out = Path(command[command.index("--output") + 1])
            dump(out / "completion.json", {"status": "Succeeded"})
            dump(out / "comparison.json", {"format_bonus": False})
        events.append(phase)
        dump(receipt, {"exit_code": 7 if fail == phase else 0})
        if fail == phase:
            raise RuntimeError("deliberate " + phase + " failure")

    monkeypatch.setattr(periodic, "_run", run)

    def release():
        events.append("release")
        return {"closed": True, "pid": 123, "exit_code": 0}

    def restore():
        events.append("restore")
        return {"version": 100, "gpu": "0", "pid": 456}

    return config, policy, identity, events, release, restore


def test_completed_step_is_verified_and_not_decoded_twice(tmp_path, monkeypatch):
    config, policy, identity, events, release, restore = setup(tmp_path, monkeypatch)
    result = periodic.run_periodic_evaluation(
        config, policy, identity, release_engine=release, restore_engine=restore
    )
    assert events == [
        "release",
        "eval-admission",
        "decode",
        "audit",
        "restore-admission",
        "restore",
    ]
    assert result["status"] == "Succeeded" and result["new_gpu_reservations"] == 0
    assert result["count"] == 1000 and result["missing"] == 0
    second = periodic.run_periodic_evaluation(
        config, policy, identity, release_engine=release, restore_engine=restore
    )
    assert second == result and len(events) == 6
    # Reuse also verifies raw generation evidence, not just its completion label.
    directory = Path(config.periodic_eval_options["output_dir"], "step-000100")
    (directory / "after/responses.json").write_text("tampered")
    with pytest.raises(ValueError, match="without verified successful evidence"):
        periodic.run_periodic_evaluation(
            config, policy, identity, release_engine=release, restore_engine=restore
        )
    assert len(events) == 6


@pytest.mark.parametrize("stage", ["eval-admission", "decode", "audit"])
def test_failed_step_restores_engine_and_preserves_failure(
    tmp_path, monkeypatch, stage
):
    config, policy, identity, events, release, restore = setup(
        tmp_path, monkeypatch, stage
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        periodic.run_periodic_evaluation(
            config, policy, identity, release_engine=release, restore_engine=restore
        )
    assert events[-2:] == ["restore-admission", "restore"]
    directory = Path(config.periodic_eval_options["output_dir"], "step-000100")
    assert periodic._read(directory / "completion.json")["status"] == "Failed"
    count = len(events)
    with pytest.raises(ValueError, match="without verified successful evidence"):
        periodic.run_periodic_evaluation(
            config, policy, identity, release_engine=release, restore_engine=restore
        )
    assert len(events) == count


def test_failed_restore_admission_never_restarts_a_training_engine(
    tmp_path, monkeypatch
):
    config, policy, identity, events, release, restore = setup(
        tmp_path, monkeypatch, "restore-admission"
    )
    with pytest.raises(RuntimeError, match="deliberate restore-admission"):
        periodic.run_periodic_evaluation(
            config, policy, identity, release_engine=release, restore_engine=restore
        )
    assert events[-1] == "restore-admission" and "restore" not in events


def test_actual_subprocess_exit_is_recorded_even_on_failure(tmp_path):
    receipt = tmp_path / "exit.json"
    with pytest.raises(ValueError, match="exited 7"):
        periodic._run(
            [sys.executable, "-c", "raise SystemExit(7)"],
            environment={"CUDA_VISIBLE_DEVICES": ""},
            log=tmp_path / "child.log",
            receipt=receipt,
        )
    value = periodic._read(receipt)
    assert value["exit_code"] == 7 and value["pid"] > 0
