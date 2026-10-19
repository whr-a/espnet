# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Matched MMAU-Mini evaluation at a synchronous native training safe point.

The caller retains the four existing Ray actor reservations. Replica zero's
training engine stops for a separate BF16 evaluator on that same GPU; learner
weights, Adam masters, scheduler, source cursor and learner RNG remain resident.
No training/evaluation overlap, extra GPU reservation or automatic retry occurs.
"""

import fcntl
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from espnet2.rl.rollout import sha256, subprocess_environment


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _run(command, *, environment, log, receipt):
    with Path(log).open("x") as stream:
        child = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
        code = child.wait()
    value = {
        "command": command,
        "pid": child.pid,
        "exit_code": code,
        "finished_utc": _stamp(),
    }
    _write(receipt, value)
    _require(code == 0, f"Periodic evaluation subprocess exited {code}: {log}")
    return value


def run_periodic_evaluation(
    config, policy, identity, *, release_engine, restore_engine
):
    """Family callback for ``RolloutActor.evaluate``; failures stop the driver.

    A verified completed step is idempotently reused. Incomplete attempts remain
    immutable and require an explicit recovery with a new output directory.
    The root-reviewed plan and resource admission helper freeze all paths and
    allow reuse only of the assigned .147 physical GPU after its engine exits.
    """
    options = config.periodic_eval_options
    step = policy["step"]
    plan_path = Path(options["plan"])
    _require(sha256(plan_path) == options["plan_sha256"], "Periodic plan changed")
    plan = _read(plan_path)
    _require(step in plan["expected_steps"], "Unexpected periodic evaluation step")
    _require(identity["replica"] == 0, "Only rollout replica0 may evaluate")
    _require(
        len(identity["gpu_ids"]) == 1
        and identity["gpu"] == str(identity["gpu_ids"][0]),
        "Periodic evaluator requires exactly its actor's assigned GPU",
    )
    # Hash the already-saved immutable native checkpoint, never live parameters.
    checkpoint = Path(config.output_dir, f"step-{step:06d}.native.pth")
    _require(
        checkpoint.is_file(),
        "Periodic evaluation requires the already-saved native milestone. "
        "A recovery into a new output directory must copy its original native "
        "checkpoint bytes; reserializing restored tensors changes its identity.",
    )
    checkpoint_hash = sha256(checkpoint)
    root = Path(options["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"step-{step:06d}"
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if directory.exists():
            complete = _read(directory / "completion.json")
            _require(
                complete["status"] == "Succeeded"
                and complete["step"] == step
                and complete["checkpoint_sha256"] == checkpoint_hash
                and complete["plan_sha256"] == options["plan_sha256"]
                and all(
                    sha256(directory / name) == digest
                    for name, digest in complete["artifact_sha256"].items()
                ),
                "Periodic step already exists without verified successful evidence; "
                "automatic evaluation retries are disabled",
            )
            return complete
        directory.mkdir()
        state = {
            "status": "Prepared",
            "step": step,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "published_training_native_sha256": policy["native_sha256"],
            "plan_sha256": options["plan_sha256"],
            "identity": identity,
            "created_utc": _stamp(),
            "automatic_retry": False,
            "new_gpu_reservations": 0,
        }
        _write(directory / "state.json", state)
        released = False
        error = None
        started = time.monotonic()
        try:
            for path, expected in plan["input_sha256"].items():
                _require(sha256(path) == expected, "Frozen eval input changed: " + path)
            released = True  # Restore is attempted even if close reports an error.
            state["closed_training_engine"] = release_engine()
            _require(
                state["closed_training_engine"]["closed"],
                "Training inference process did not close",
            )
            state["status"] = "AdmittingResources"
            _write(directory / "state.json", state)
            environment = subprocess_environment(config, identity["gpu"])
            environment.update(
                OMP_NUM_THREADS="8",
                TOKENIZERS_PARALLELISM="false",
                PYTHONDONTWRITEBYTECODE="1",
            )
            admission_environment = {**environment, "CUDA_VISIBLE_DEVICES": ""}
            admission_command = list(options["resource_preflight_command"]) + [
                "--gpu",
                identity["gpu"],
                "--output",
                str(directory / "resource_preflight.json"),
            ]
            _run(
                admission_command,
                environment=admission_environment,
                log=directory / "resource_preflight.log",
                receipt=directory / "resource_preflight_exit.json",
            )
            admission = _read(directory / "resource_preflight.json")
            _require(admission["status"] == "Accepted", "GPU admission rejected")
            command = [
                config.rollout_python,
                "-m",
                "espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate",
                "--config",
                options["evaluation_config"],
                "--manifest",
                plan["manifest"],
                "--checkpoint",
                str(checkpoint),
                "--policy-step",
                str(step),
                "--output",
                str(directory / "after"),
                "--qa-protocol",
                "bagpiper_native",
                "--temperature",
                "0",
                "--samples",
                "1",
                "--max-new-tokens",
                "4096",
                "--audio-seconds",
                "60",
                "--batch-size",
                "256",
            ]
            state.update(status="Evaluating", command=command, started_utc=_stamp())
            _write(directory / "state.json", state)
            _run(
                command,
                environment=environment,
                log=directory / "evaluator.log",
                receipt=directory / "evaluator_exit.json",
            )
            _run(
                [
                    config.rollout_python,
                    options["audit_script"],
                    "--plan",
                    str(plan_path),
                    "--after",
                    str(directory / "after"),
                    "--step",
                    str(step),
                    "--checkpoint-sha256",
                    checkpoint_hash,
                    "--exit-record",
                    str(directory / "evaluator_exit.json"),
                    "--output",
                    str(directory / "audit"),
                ],
                environment=admission_environment,
                log=directory / "audit.log",
                receipt=directory / "audit_exit.json",
            )
            _require(
                _read(directory / "audit/completion.json")["status"] == "Succeeded",
                "Periodic1000-item audit did not succeed",
            )
        except BaseException as failure:
            error = failure
            state.update(error=str(failure), traceback=traceback.format_exc())
        finally:
            if released:
                try:
                    # The BF16 evaluator may take long enough for policy or
                    # capacity evidence to expire. Re-read both before starting
                    # the original training engine on the now-idle same GPU.
                    if "environment" not in locals():
                        environment = subprocess_environment(config, identity["gpu"])
                        admission_environment = {
                            **environment,
                            "CUDA_VISIBLE_DEVICES": "",
                        }
                    restore_admission = list(options["resource_preflight_command"]) + [
                        "--gpu",
                        identity["gpu"],
                        "--output",
                        str(directory / "restore_resource_preflight.json"),
                    ]
                    _run(
                        restore_admission,
                        environment=admission_environment,
                        log=directory / "restore_resource_preflight.log",
                        receipt=directory / "restore_resource_preflight_exit.json",
                    )
                    _require(
                        _read(directory / "restore_resource_preflight.json")["status"]
                        == "Accepted",
                        "Training engine restoration admission rejected",
                    )
                    state["restored_training_engine"] = restore_engine()
                    _require(
                        state["restored_training_engine"]["version"] == step,
                        "Restored training engine has the wrong policy",
                    )
                except BaseException as failure:
                    state["restoration_error"] = str(failure)
                    state["restoration_traceback"] = traceback.format_exc()
                    error = error or failure
            state.update(
                status="Failed" if error else "Succeeded",
                finished_utc=_stamp(),
                duration_seconds=time.monotonic() - started,
            )
            _write(directory / "state.json", state)
        if error:
            _write(directory / "completion.json", state)
            raise error
        complete = {
            **state,
            "count": 1000,
            "missing": 0,
            "artifact_sha256": {
                name: sha256(directory / name)
                for name in (
                    "state.json",
                    "evaluator_exit.json",
                    "audit_exit.json",
                    "resource_preflight.json",
                    "restore_resource_preflight.json",
                    "audit/completion.json",
                    "audit/comparison.json",
                    "after/responses.json",
                    "after/protocol.json",
                    "after/summary.json",
                )
            },
        }
        _write(directory / "completion.json", complete)
        return complete
