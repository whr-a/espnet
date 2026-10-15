"""Complete prompt groups on isolated SGLang subprocesses and GPUs."""

import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from .rollout import sha256


def run_replica(config, policy_path, sources, rollout_id, replica, gpu, expected_hash):
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
    frozen = directory / "config.json"
    frozen.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    environment = CheckpointRollout(config)._environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.time()
    with (directory / "generation.log").open("w") as stream:
        subprocess.run(
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
            ],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = json.loads(responses.read_text())
    if result["native_sha256"] != expected_hash or result["policy_step"] != rollout_id:
        raise ValueError("Rollout worker identity mismatch")
    return {
        "generations": result["generations"],
        "start": start,
        "end": time.time(),
        "replica": replica,
        "gpu": str(gpu),
        "seed": config.seed,
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
