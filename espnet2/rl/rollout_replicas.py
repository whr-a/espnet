"""Complete prompt groups on isolated SGLang subprocesses and GPUs."""

import copy
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .rollout import sha256


def _process_start_ticks(pid):
    try:
        # comm may contain spaces; fields after the closing parenthesis start at 3.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def terminate_owned_process(directory):
    """Signal only a child session whose PID identity matches our receipt."""
    path = Path(directory, "process.json")
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    pid = receipt["pid"]
    if (
        receipt["start_ticks"] is None
        or _process_start_ticks(pid) != receipt["start_ticks"]
    ):
        return False
    try:
        if os.getpgid(pid) != pid or receipt["pgid"] != pid:
            raise ValueError(
                "Rollout child process group differs from its ownership receipt"
            )
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _process_start_ticks(pid) != receipt["start_ticks"]:
            return True
        time.sleep(0.05)
    if _process_start_ticks(pid) == receipt["start_ticks"]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def run_owned_subprocess(command, *, environment, directory, stream):
    """A separate child session and immutable PID identity enable exact cleanup."""
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    receipt = {
        "pid": process.pid,
        "pgid": process.pid,
        "start_ticks": _process_start_ticks(process.pid),
        "command": command,
    }
    try:
        Path(directory, "process.json").write_text(json.dumps(receipt) + "\n")
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise


def run_replica(
    config,
    policy_path,
    sources,
    rollout_id,
    replica,
    gpu,
    expected_hash,
    *,
    policy_version=None,
):
    from .rollout import CheckpointRollout

    config = copy.deepcopy(config)
    directory = Path(
        config.output_dir,
        "rollout_batches",
        f"batch-{rollout_id:06d}",
        f"replica-{replica}",
    )
    directory.mkdir(parents=True, exist_ok=False)
    policy_path = Path(policy_path)
    if sha256(policy_path / "native.pt") != expected_hash:
        raise ValueError("Rollout policy checkpoint hash mismatch")
    model = directory / "hf"
    model.mkdir()
    for source in (policy_path / "hf").iterdir():
        if source.is_dir():
            shutil.copytree(source, model / source.name, symlinks=True)
        elif source.suffix == ".json":
            shutil.copyfile(source, model / source.name)
        else:
            (model / source.name).symlink_to(source.resolve())
    (directory / "native.pt").symlink_to((policy_path / "native.pt").resolve())
    requests, responses = directory / "requests.json", directory / "responses.json"
    requests.write_text(json.dumps(sources) + "\n")
    config.seed += replica * 1000000
    if policy_version is None:
        policy_version = rollout_id
    frozen = directory / "config.json"
    frozen.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    environment = CheckpointRollout(config)._environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.time()
    with (directory / "generation.log").open("w") as stream:
        run_owned_subprocess(
            [
                config.rollout_python,
                "-m",
                "espnet2.rl.rollout_worker",
                "--config",
                str(frozen),
                "--model",
                str(model),
                "--requests",
                str(requests),
                "--output",
                str(responses),
                "--step",
                str(rollout_id),
                "--policy-version",
                str(policy_version),
            ],
            environment=environment,
            directory=directory,
            stream=stream,
        )
    result = json.loads(responses.read_text())
    if (
        result["native_sha256"] != expected_hash
        or result["policy_step"] != policy_version
        or result["rollout_id"] != rollout_id
    ):
        raise ValueError("Rollout worker identity mismatch")
    return {
        "generations": result["generations"],
        "start": start,
        "end": time.time(),
        "replica": replica,
        "gpu": str(gpu),
        "seed": config.seed,
        "behavior_version": policy_version,
        "rollout_id": rollout_id,
    }


def merge_generations(sources, results, group_size):
    grouped = {}
    expected = {source["id"] for source in sources}
    for result in results:
        for generation in result["generations"]:
            identity = generation["source_id"]
            if identity not in expected:
                raise ValueError("Rollout replica returned an unknown source")
            grouped.setdefault(identity, []).append(generation)
    if set(grouped) != expected or any(
        len(group) != group_size for group in grouped.values()
    ):
        raise ValueError("Rollout replicas must return every complete prompt group")
    return [generation for source in sources for generation in grouped[source["id"]]]
