# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a native OpusLM checkpoint on a separate prepared audio manifest."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts, read_manifest
from espnet2.rl.engine import fit_context
from espnet2.rl.rewards import edit_distance, words
from espnet2.rl.rollout import CheckpointRollout, sha256
from espnet2.rl.types import drift_gate
from espnet2.rl.utils import build_policy


def summarize_wer(sources, generations, group_size):
    """Audit every group; report sample mean and word-weighted corpus WER."""
    # Reuse the training contract for completeness, order and action evidence.
    collate_rollouts(sources, generations, group_size)
    rows = []
    for index, generation in enumerate(generations):
        source = sources[index // group_size]
        if (
            "prompt_token_ids" in source
            and generation["prompt_token_ids"] != source["prompt_token_ids"]
        ):
            raise ValueError("Saved native prompt differs from the heldout manifest")
        reference, hypothesis = words(source["reference"]), words(generation["text"])
        errors = edit_distance(reference, hypothesis)
        rows.append(
            {
                "source_id": source["id"],
                "sample_index": index % group_size,
                "reference": source["reference"],
                "hypothesis": generation["text"],
                "reference_words": len(reference),
                "word_errors": errors,
                "sample_wer_denominator": max(1, len(reference)),
                "wer": errors / max(1, len(reference)),
            }
        )
    total_words = sum(row["reference_words"] for row in rows)
    errors = sum(row["word_errors"] for row in rows)
    return {
        "samples": len(sources),
        "completions": len(rows),
        "group_size": group_size,
        "source_ids": [source["id"] for source in sources],
        "wer_mean": sum(row["wer"] for row in rows) / len(rows),
        "wer_corpus": errors / total_words if total_words else None,
        "total_word_errors": errors,
        "total_reference_words": total_words,
        "empty_reference_completions": sum(not row["reference_words"] for row in rows),
        "normalization": "NFKC, casefold, punctuation removed, whitespace tokens",
        "mean_definition": "mean over every completion of errors/max(1,reference_words)",
        "corpus_definition": "sum(errors)/sum(reference_words), repeated for every completion; null when denominator is zero",
        "records": rows,
    }


@torch.inference_mode()
def verify_log_probs(policy, batch, config):
    """Compare actual saved SGLang actions with the final native checkpoint.

    Returns every completion's largest difference and the training drift
    gate's statistics over all of them, which pass its tolerances.
    """
    errors, native = [], []
    for index in range(len(batch)):
        item = batch.select_idxs([index])
        scores = policy(item)
        expected = item.batch["rollout_log_probs"].to(scores.log_probs.device)
        mask = item.batch["response_mask"].to(scores.log_probs.device)
        active = scores.response_mask.to(mask.device)
        if (
            scores.log_probs.shape != expected.shape
            or active.shape != mask.shape
            or not torch.isfinite(active).all()
            or not ((active == 0) | (active == 1)).all()
            or not torch.equal(active, mask)
            or not torch.isfinite(scores.log_probs[mask.bool()]).all()
        ):
            raise ValueError("Native action scores/masks differ from the sampled batch")
        errors.append(float((scores.log_probs - expected).abs()[mask.bool()].max()))
        native.append(scores.log_probs.cpu())
    drift = drift_gate(
        torch.cat(native),
        batch.batch["rollout_log_probs"],
        batch.batch["response_mask"],
        config,
        "Native/SGLang",
        "the evaluated checkpoint must have generated these responses",
    )
    return errors, drift


def validate_export_files(directory, recorded):
    """Require a complete hash list of regular files contained in this export."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir() or not isinstance(recorded, dict) or not recorded:
        raise ValueError("Existing inference export has no recorded files")
    paths = {}
    for relative, expected in recorded.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"Invalid recorded inference export path/hash: {relative}")
        path = root / relative
        if any(
            parent.is_symlink()
            for parent in (path, *path.parents)
            if parent != root and parent.is_relative_to(root)
        ):
            raise ValueError(f"Inference export contains a symlink: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(
                f"Inference export file is outside its directory: {relative}"
            )
        paths[relative] = resolved
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"Inference export contains a symlink: {path.relative_to(root)}"
            )
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if set(paths) != actual:
        raise ValueError(
            "Recorded inference export files differ from its complete file set"
        )
    for relative, path in paths.items():
        if sha256(path) != recorded[relative]:
            raise ValueError(f"Existing inference export file changed: {relative}")


def export_checkpoint(rollout, checkpoint, step, existing_export=None):
    """Use a read-only native symlink; optionally reuse an audited HF export."""
    version = rollout.directory / f"policy-{step:06d}"
    version.mkdir()
    (version / "native.pt").symlink_to(checkpoint)
    if existing_export:
        existing = Path(existing_export).resolve()
        if sha256(existing / "native.pt") != sha256(checkpoint):
            raise ValueError(
                "Existing inference export belongs to a different native checkpoint"
            )
        sync = json.loads((existing / "sync.json").read_text())
        if sync["native_sha256"] != sha256(checkpoint):
            raise ValueError(
                "Existing export native hash differs from its synchronization record"
            )
        validate_export_files(existing / "hf", sync.get("hf_files"))
        (version / "hf").symlink_to(existing / "hf", target_is_directory=True)
    else:
        args = [
            rollout.config.rollout_python,
            "-m",
            "espnet2.rl.export_worker",
            "--config",
            str(Path(rollout.config.output_dir) / "frozen_config.json"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(version / "hf"),
        ]
        with (version / "export.log").open("w") as stream:
            subprocess.run(
                args,
                env=rollout._environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        sync = {
            "native_sha256": sha256(checkpoint),
            "hf_files": {
                str(path.relative_to(version / "hf")): sha256(path)
                for path in sorted((version / "hf").rglob("*"))
                if path.is_file()
            },
        }
        if not sync["hf_files"]:
            raise RuntimeError("Exporter produced no inference model files")
    sync = {**sync, "step": step, "native_symlink": str(checkpoint)}
    (version / "sync.json").write_text(json.dumps(sync, indent=2) + "\n")
    rollout.version = version
    return sync


def run(args):
    config = ESPnetRLConfig.load(args.config)
    if config.family not in ("opuslm", "opuslm_dialogue"):
        raise ValueError("This evaluator supports OpusLM Plain and Dialogue")
    if config.rollout_backend != "sglang":
        raise ValueError("OpusLM evaluation requires the SGLang backend")
    training = read_manifest(config.manifest)
    sources = read_manifest(args.manifest)
    if {row["id"] for row in training} & {row["id"] for row in sources}:
        raise ValueError("Heldout and training utterance IDs overlap")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config.checkpoint = str(checkpoint)
    config.eval_manifest = str(Path(args.manifest).resolve())
    config.output_dir = str(Path(args.output_dir).resolve())
    if args.device:
        config.device = args.device
    if args.rollout_gpus:
        config.rollout_gpus = args.rollout_gpus
    if args.group_size is not None:
        config.group_size = args.group_size
    prompts = [
        len(row["prompt_token_ids"]) for row in sources if "prompt_token_ids" in row
    ]
    if prompts:
        # Held-out prompts may be longer than the recipe's training prompts.
        fit_context(config, {}, max(prompts) + config.max_tokens + 2)
    config.validate()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n"
    )
    evidence = {
        "checkpoint": str(checkpoint),
        "native_sha256": sha256(checkpoint),
        "manifest": config.eval_manifest,
        "manifest_sha256": sha256(args.manifest),
        "training_manifest": str(Path(config.manifest).resolve()),
        "training_manifest_sha256": sha256(config.manifest),
        "config_sha256": sha256(args.config),
        "policy_step": args.policy_step,
        "distinct_train_ids": True,
    }
    (output / "protocol.json").write_text(json.dumps(evidence, indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if args.responses:
            result = json.loads(Path(args.responses).read_text())
            if (
                result["native_sha256"] != evidence["native_sha256"]
                or result["policy_step"] != args.policy_step
            ):
                raise ValueError(
                    "Saved responses belong to a different policy checkpoint/version"
                )
            generations = result["generations"]
            evidence.update(
                saved_responses=str(Path(args.responses).resolve()),
                saved_responses_sha256=sha256(args.responses),
            )
        else:
            rollout = CheckpointRollout(config)
            export_checkpoint(rollout, checkpoint, args.policy_step, args.export_dir)
            try:
                rollout.load(rollout.version, args.policy_step)
                generations = rollout.generate(sources, args.policy_step)
            finally:
                rollout.close()
        (output / "responses.json").write_text(
            json.dumps({**evidence, "generations": generations}, indent=2) + "\n"
        )
        batch = collate_rollouts(sources, generations, config.group_size)
        batch.save_to_disk(output / "rollout.pkl")
        policy = build_policy(config, config.device)
        policy.requires_grad_(False)
        errors, drift = verify_log_probs(policy, batch, config)
        summary = {
            "status": "passed",
            **evidence,
            **summarize_wer(sources, generations, config.group_size),
            **{f"logprob_{name}_error": value for name, value in drift.items()},
            "completion_logprob_errors": errors,
            "generation": (
                "audited existing responses" if args.responses else "actual SGLang"
            ),
            "scope": "nine-stream audio conditions to stream-zero text WER; no codec-output TTS evaluation",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "completion.json").write_text(
            json.dumps({"status": "Succeeded", "summary": "summary.json"}) + "\n"
        )
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "records"},
                indent=2,
            )
        )
    except Exception as error:
        (output / "completion.json").write_text(
            json.dumps({"status": "Failed", "error": str(error)}) + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Effective frozen_config.json or training YAML, including original training manifest",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Native .pth state, not the combined optimizer .pt",
    )
    parser.add_argument(
        "--manifest", required=True, help="Separately prepared heldout JSONL"
    )
    parser.add_argument(
        "--output-dir", required=True, help="New immutable evaluation directory"
    )
    parser.add_argument("--policy-step", type=int, default=0)
    parser.add_argument("--device", help="Native scoring device; CPU is supported")
    parser.add_argument("--rollout-gpus", help="SGLang subprocess CUDA_VISIBLE_DEVICES")
    parser.add_argument("--group-size", type=int)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--export-dir",
        help="Audited existing policy directory with native.pt, hf and sync.json",
    )
    existing.add_argument(
        "--responses", help="Audit saved real responses without generating again"
    )
    args = parser.parse_args()
    if args.policy_step < 0:
        parser.error("--policy-step must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
