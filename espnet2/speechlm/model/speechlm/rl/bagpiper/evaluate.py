# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Independent Bagpiper evaluation with complete native/action identity gates."""

import argparse
import json
import math
import subprocess
import traceback
from pathlib import Path

import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts, read_manifest
from espnet2.rl.rollout import (
    CheckpointRollout,
    RolloutProcess,
    read_json,
    sha256,
    write_json,
)
from espnet2.rl.types import validate_token_scores
from espnet2.rl.utils import build_policy, resolve_callable
from espnet2.speechlm.rl.evaluate import validate_export_files


def validate_manifests(training, heldout):
    if not training or not heldout:
        raise ValueError("Training and held-out manifests must be nonempty")
    for records in (training, heldout):
        if len({row["id"] for row in records}) != len(records):
            raise ValueError("Manifest identities must be unique")
    if {row["id"] for row in training} & {row["id"] for row in heldout}:
        raise ValueError("Training and held-out utterance IDs overlap")
    tasks = {row["task"] for row in heldout}
    if len(tasks) != 1 or not tasks <= {"qa", "tts_asr_wer"}:
        raise ValueError("Held-out manifest must contain one supported task")


def validate_responses(result, sources, config, native_hash, policy_step):
    if (
        result.get("native_sha256") != native_hash
        or result.get("policy_step") != policy_step
        or result.get("rollout_backend") != "sglang"
    ):
        raise ValueError(
            "Response checkpoint, policy version or backend identity differs"
        )
    generations = result["generations"]
    batch = collate_rollouts(sources, generations, config.group_size)
    for index, generation in enumerate(generations):
        source = sources[index // config.group_size]
        if (
            "prompt_token_ids" in source
            and source["prompt_token_ids"] != generation["prompt_token_ids"]
        ):
            raise ValueError("Saved prompt differs from the prepared held-out manifest")
    return generations, batch


@torch.inference_mode()
def verify_native_log_probs(policy, batch, tolerance):
    errors = []
    for index in range(len(batch)):
        item = batch.select_idxs([index])
        scores = policy(item)
        mask = item.batch["response_mask"]
        validate_token_scores(scores, mask)
        expected = item.batch["rollout_log_probs"].to(scores.log_probs.device)
        active = mask.to(scores.log_probs.device).bool()
        error = float((scores.log_probs - expected).abs()[active].max())
        if not math.isfinite(error) or error > tolerance:
            raise ValueError(f"Native/SGLang logprob mismatch: {error} > {tolerance}")
        errors.append(error)
    return errors


def verify_export(checkpoint, directory):
    """Verify all export files and exact native tensors, including frozen codec."""
    from safetensors import safe_open

    directory = Path(directory).resolve()
    files = {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    validate_export_files(directory, files)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("module", state)
    expected = {key for key in state if key != "vocab_weight"}
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if set(mapping) != expected:
        raise ValueError(
            "Inference export tensor keys differ from the complete native state"
        )
    for key, shard in mapping.items():
        if shard not in files:
            raise ValueError(
                f"Inference tensor shard is outside the audited file set: {shard}"
            )
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            if not torch.equal(state[key], handle.get_tensor(key)):
                raise ValueError(
                    f"Inference export tensor differs from native checkpoint: {key}"
                )
    return {
        "native_sha256": sha256(checkpoint),
        "hf_files": files,
        "exact_native_export_tensors": len(mapping),
        "excluded_nonpolicy": [key for key in state if key not in expected],
    }


def summarize(sources, generations, config):
    from espnet2.rl.rewards import words

    from .rewards import ASRWERReward, mmau_match

    metric = "wer" if sources[0]["task"] == "tts_asr_wer" else "mmau_official_accuracy"
    callback = (
        ASRWERReward(config)
        if metric == "wer"
        else resolve_callable(config.reward_function)
    )
    rows, values = [], []
    for index, generation in enumerate(generations):
        source = sources[index // config.group_size]
        reward = float(callback(source, generation))
        value = -reward if metric == "wer" else float(mmau_match(source, generation))
        if not math.isfinite(reward) or not math.isfinite(value):
            raise ValueError("Every held-out reward and metric must be finite")
        rows.append(
            {
                "source_id": source["id"],
                "sample_index": index % config.group_size,
                "reward": reward,
                metric: value,
                "text": generation["text"],
                "reference": source["reference"],
                "reference_words": len(words(source["reference"])),
            }
        )
        values.append(value)
    return {
        "metric": metric,
        "value": sum(values) / len(values),
        "sample_count": len(values),
        "prompt_count": len(sources),
        "group_size": config.group_size,
        "source_ids": [source["id"] for source in sources],
        "sampling": "unit temperature, native recorded support, all group candidates",
        "denominator": len(values),
        "metric_definition": (
            "mean completion WER"
            if metric == "wer"
            else "official MMAU token-set matches / all configured completions"
        ),
        "dataset": sources[0]["dataset"],
        "generations": rows,
    }


def run(args):
    cfg = ESPnetRLConfig.load(args.config)
    if cfg.family != "bagpiper" or cfg.rollout_backend != "sglang":
        raise ValueError("Bagpiper evaluation requires its native SGLang configuration")
    if args.manifest:
        cfg.eval_manifest = args.manifest
    if not cfg.eval_manifest:
        raise ValueError("Set a separate held-out eval_manifest before evaluation")
    training = read_manifest(cfg.manifest)
    sources = read_manifest(cfg.eval_manifest)
    validate_manifests(training, sources)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    cfg.checkpoint = str(checkpoint)
    cfg.device = args.device
    cfg.output_dir = str(Path(args.output).resolve())
    cfg.validate()
    directory = Path(cfg.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "native.pt").symlink_to(checkpoint)
    (directory / "frozen_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n"
    )
    (directory / "requests.json").write_text(json.dumps(sources, indent=2) + "\n")
    evidence = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest": str(Path(cfg.eval_manifest).resolve()),
        "manifest_sha256": sha256(cfg.eval_manifest),
        "training_manifest": str(Path(cfg.manifest).resolve()),
        "training_manifest_sha256": sha256(cfg.manifest),
        "config_sha256": sha256(args.config),
        "policy_step": args.policy_step,
        "distinct_train_ids": True,
        "native_device": cfg.device,
    }
    (directory / "protocol.json").write_text(json.dumps(evidence, indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if args.responses:
            result = read_json(args.responses)
            evidence.update(
                saved_responses=str(Path(args.responses).resolve()),
                saved_responses_sha256=sha256(args.responses),
            )
            if args.export_dir:
                export = verify_export(checkpoint, args.export_dir)
                (directory / "export_audit.json").write_text(
                    json.dumps(export, indent=2) + "\n"
                )
                evidence["exact_native_export_tensors"] = export[
                    "exact_native_export_tensors"
                ]
        else:
            if args.export_dir:
                raise ValueError(
                    "--export-dir is only for read-only saved-response auditing"
                )
            rollout = CheckpointRollout(cfg)
            export_command = [
                cfg.rollout_python,
                "-m",
                "espnet2.rl.export_worker",
                "--config",
                str(directory / "frozen_config.json"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(directory / "hf"),
            ]
            with (directory / "export.log").open("w") as handle:
                subprocess.run(
                    export_command,
                    env=rollout._environment(),
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            export = verify_export(checkpoint, directory / "hf")
            (directory / "export_audit.json").write_text(
                json.dumps(export, indent=2) + "\n"
            )
            evidence["exact_native_export_tensors"] = export[
                "exact_native_export_tensors"
            ]
            process = RolloutProcess(
                cfg, cfg.rollout_gpus, rollout.directory / "replica-0", 0
            )
            try:
                process.load(directory / "hf", args.policy_step)
                generations = process.generate(sources, args.policy_step)
            finally:
                process.close()
            result = {
                "policy_step": args.policy_step,
                "native_sha256": evidence["checkpoint_sha256"],
                "rollout_backend": "sglang",
                "generations": generations,
            }
            write_json(directory / "responses.json", result, indent=2)
        generations, batch = validate_responses(
            result, sources, cfg, evidence["checkpoint_sha256"], args.policy_step
        )
        if args.responses:
            write_json(directory / "responses.json", result, indent=2)
        batch.save_to_disk(directory / "rollout.pkl")
        policy = build_policy(cfg, cfg.device)
        policy.requires_grad_(False)
        errors = verify_native_log_probs(policy, batch, cfg.score_tolerance)
        del policy
        metrics = {
            "status": "passed",
            **evidence,
            **summarize(sources, generations, cfg),
            "native_logprob_max_error": max(errors),
            "completion_logprob_errors": errors,
            "generation": (
                "audited existing responses" if args.responses else "actual SGLang"
            ),
        }
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (directory / "completion.json").write_text(
            json.dumps({"status": "Succeeded", "metrics": "metrics.json"}) + "\n"
        )
        print(
            json.dumps(
                {key: value for key, value in metrics.items() if key != "generations"},
                indent=2,
            )
        )
    except Exception as error:
        (directory / "failure.txt").write_text(traceback.format_exc())
        (directory / "completion.json").write_text(
            json.dumps({"status": "Failed", "error": str(error)}) + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Effective frozen JSON or YAML with original training manifest",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Native model state, not the combined optimizer checkpoint",
    )
    parser.add_argument(
        "--output", required=True, help="New immutable evaluation directory"
    )
    parser.add_argument("--manifest", help="Override the prepared held-out manifest")
    parser.add_argument(
        "--device", default="cpu", help="Native probability verification device"
    )
    parser.add_argument("--policy-step", type=int, default=-1)
    parser.add_argument(
        "--responses", help="Audit existing real responses without new generation"
    )
    parser.add_argument(
        "--export-dir",
        help="Existing HF directory for exact native tensor verification with --responses",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
