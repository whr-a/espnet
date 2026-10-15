"""Shared batch, placement, and frozen-data gates for the 32×8 profiles."""

import json
import os
from pathlib import Path

from .data import read_manifest
from .rollout import sha256


def validate_short_run(config):
    config.validate()
    if (
        config.prompts_per_step != 32
        or config.group_size != 8
        or config.steps > 5
        or config.ppo_epochs != 1
        or config.warmup_steps != 1
    ):
        raise ValueError(
            "The short-run profile requires 32×8, at most 5 updates, and first update warmup"
        )
    expected = 8 if config.execution_mode == "disagg_async" else 4
    if config.rollout_backend != "sglang" or config.rollout_num_replicas != expected:
        raise ValueError(
            f"This profile requires {expected} independent SGLang replicas"
        )


def validate_collocate_geometry(config, *, running):
    validate_short_run(config)
    if config.execution_mode != "sync":
        raise ValueError("The collocate launcher requires execution_mode=sync")
    if config.rollout_gpus.split(",") != ["4", "5", "6", "7"]:
        raise ValueError("Collocate rollout uses physical GPUs 4–7")
    if running and (
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") != ["0", "1", "2", "3"]
        or int(os.environ.get("WORLD_SIZE", "1")) != 4
        or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 4
        or int(os.environ.get("LOCAL_RANK", "-1")) not in range(4)
    ):
        raise ValueError("Launch four torchrun training ranks on physical GPUs 0–3")


def verify_data_admission(config, path):
    """Bind live manifests/media to the registered protection audit receipt."""
    admission = json.loads(Path(path).read_text())
    if (
        admission.get("schema") != "espnet-short-run-data-v1"
        or admission.get("training_ready") is not True
    ):
        raise ValueError("Data has not passed the complete training admission")
    if admission.get("scoped_component_admission") != "VERIFIED_REGISTERED_PROTECTION":
        raise ValueError(
            "All registered protected and heldout components must be excluded"
        )
    if admission.get("exact_key_gate") != "PASSED":
        raise ValueError("Registered protected identities must be excluded")
    scope = admission.get("verified_protection_scope", {})
    if (
        scope.get("specialist_records") != 1389
        or scope.get("registered_audio_count", 0) <= 0
        or scope.get("registered_audio_decoded") != scope.get("registered_audio_count")
    ):
        raise ValueError(
            "The complete registered 945+444 protection scope must be verified"
        )
    bindings = admission.get("protection_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Data admission must bind registered protection metadata")
    for protected, expected in bindings.items():
        if sha256(protected) != expected:
            raise ValueError("Protected metadata changed after data preparation")
    manifest_hashes = admission.get("manifest_sha256", {})
    if manifest_hashes.get("train32") != sha256(config.manifest):
        raise ValueError("Training manifest differs from its frozen admission")
    if config.eval_manifest is None or manifest_hashes.get("heldout") != sha256(
        config.eval_manifest
    ):
        raise ValueError("Heldout manifest differs from its frozen admission")
    train, heldout = read_manifest(config.manifest), read_manifest(config.eval_manifest)
    if (
        len(train) != 32
        or not heldout
        or admission.get("counts") != {"train": len(train), "heldout": len(heldout)}
    ):
        raise ValueError("Manifest counts differ from the frozen 32-prompt admission")
    hashes = []
    for rows in (train, heldout):
        media = set()
        for row in rows:
            if (
                not row.get("audio")
                or not row.get("audio_sha256")
                or sha256(row["audio"]) != row["audio_sha256"]
            ):
                raise ValueError("Admitted audio changed or lacks a frozen SHA256")
            if row["audio_sha256"] in media:
                raise ValueError("Independent prompts cannot duplicate admitted audio")
            media.add(row["audio_sha256"])
        hashes.append(media)
    if hashes[0] & hashes[1]:
        raise ValueError("Training and heldout audio are not independent")
    return admission
