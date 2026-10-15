# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Held-out Bagpiper evaluation through the family's rollout engine.

The evaluated checkpoint is exported and loaded into the recipe's
``sglang_engine``, which decodes every held-out prompt. MMAU follows WEST's
decode_mmau.py: greedy decoding of up to 2,048 new tokens, every clip cut to
its first 30 s, and the official MMAU matcher applied to WEST's extraction and
to the v1 reward's. TTS prompts are scored with the training reward's ASR WER.
A sampled evaluation (temperature 1) can also verify every action's probability
against the native checkpoint; a greedy decode is not a sampled policy.
"""

import argparse
import json
import math
import os
import subprocess
import traceback
from collections import Counter
from pathlib import Path

import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts, read_manifest
from espnet2.rl.rewards import edit_distance, words
from espnet2.rl.rollout import read_json, sha256, subprocess_environment, write_json
from espnet2.rl.types import validate_token_scores
from espnet2.rl.utils import build_policy, resolve_callable
from espnet2.speechlm.rl.evaluate import validate_export_files

from .rewards import ANSWER, ASRWERReward, mmau_answer, mmau_match, qa_answer


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


def validate_responses(result, sources, samples, native_hash, policy_step):
    """The generations of this checkpoint: ``samples`` per source, in order."""
    if (
        result.get("native_sha256") != native_hash
        or result.get("policy_step") != policy_step
        or result.get("rollout_backend") != "sglang"
    ):
        raise ValueError(
            "Response checkpoint, policy version or backend identity differs"
        )
    generations = result["generations"]
    if len(generations) != len(sources) * samples:
        raise ValueError("Incomplete held-out generations")
    for index, generation in enumerate(generations):
        source = sources[index // samples]
        if generation["source_id"] != source["id"]:
            raise ValueError("Held-out generations are out of manifest order")
        if (
            "prompt_token_ids" in source
            and source["prompt_token_ids"] != generation["prompt_token_ids"]
        ):
            raise ValueError("Saved prompt differs from the prepared held-out manifest")
    return generations


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


def clip_audio(sources, directory, seconds):
    """Cut every longer clip to its first ``seconds``, as decode_mmau.py does.

    The cut is written as float WAV, so the frontend reads exactly the samples
    it would have read from the original file. Returns the number cut.
    """
    import soundfile as sf

    clipped = 0
    for index, source in enumerate(sources):
        info = sf.info(source["audio"])
        limit = int(seconds * info.samplerate)
        if info.frames > limit:
            audio, rate = sf.read(
                source["audio"], frames=limit, dtype="float32", always_2d=True
            )
            Path(directory).mkdir(exist_ok=True)
            path = Path(directory) / f"{index:06d}.wav"
            sf.write(str(path), audio, rate, subtype="FLOAT")
            source.update(audio=str(path), clipped_from=source["audio"])
            clipped += 1
    return clipped


def generate(config, sources, directory, sampling, batch_size):
    """Decode every source with the family engine built from ``directory/hf``.

    Each ``batch_size`` sources form one rollout, which bounds what the engine
    holds per rollout (prepared sources, audio encodings).
    """
    (directory / "engine").mkdir()
    engine = resolve_callable(config.sglang_engine)(
        config, directory / "hf", directory=directory / "engine", replica=0
    )
    try:
        generations = []
        for rollout_id, start in enumerate(range(0, len(sources), batch_size)):
            chunk = sources[start : start + batch_size]
            generations += engine.generate(chunk, rollout_id, sampling=sampling)
    finally:
        engine.close()
    return generations


def _accuracy(items):
    count = len(items)
    west = sum(item["west_correct"] for item in items)
    v1 = sum(item["v1_correct"] for item in items)
    return {
        "count": count,
        "west_correct": west,
        "west_accuracy": west / count,
        "v1_correct": v1,
        "v1_accuracy": v1 / count,
    }


def score_mmau(sources, generations, samples):
    """Official MMAU matches of WEST's and of the v1 reward's extracted answer,
    overall and for every category the manifest records."""
    items = []
    for index, generation in enumerate(generations):
        source = sources[index // samples]
        text = generation["text"]
        west = mmau_answer(text)
        answer, path = qa_answer(text)
        items.append(
            {
                "source_id": source["id"],
                "sample": index % samples,
                "categories": source.get("categories", {}),
                "reference": source["reference"],
                "choices": source["choices"],
                "text": text,
                "finish_reason": generation.get("finish_reason"),
                "answer_blocks": len(ANSWER.findall(text)),
                "west_answer": west,
                "west_correct": mmau_match(source, west),
                "v1_answer": answer,
                "v1_path": path,
                "v1_correct": mmau_match(source, answer),
            }
        )
    groups = {}
    for item in items:
        for name, value in item["categories"].items():
            groups.setdefault(name, {}).setdefault(str(value), []).append(item)
    summary = {
        "metric": "mmau_official_accuracy",
        **_accuracy(items),
        "one_answer_block": sum(item["answer_blocks"] == 1 for item in items),
        "v1_paths": dict(Counter(item["v1_path"] for item in items)),
        "truncated": sum(item["finish_reason"] == "length" for item in items),
        "by_category": {
            name: {value: _accuracy(rows) for value, rows in sorted(values.items())}
            for name, values in sorted(groups.items())
        },
    }
    return items, summary


def score_tts(reward, sources, generations, samples):
    """The training reward's ASR transcript and WER of every candidate."""
    repeated = [sources[index // samples] for index in range(len(generations))]
    items = []
    for index, row in enumerate(reward.transcribe(repeated, generations)):
        reference = words(row["reference"])
        errors = (
            max(1, len(reference))
            if row["hypothesis"] is None
            else edit_distance(reference, words(row["hypothesis"]))
        )
        items.append(
            {
                **row,
                "sample": index % samples,
                "finish_reason": generations[index].get("finish_reason"),
                "errors": errors,
                "reference_words": len(reference),
            }
        )
    summary = {
        "metric": "wer",
        "count": len(items),
        "wer_mean": sum(item["wer"] for item in items) / len(items),
        "corpus_wer": sum(item["errors"] for item in items)
        / max(1, sum(item["reference_words"] for item in items)),
        "statuses": dict(Counter(item["status"] for item in items)),
        "truncated": sum(item["finish_reason"] == "length" for item in items),
    }
    return items, summary


def _without_features(generation):
    # The manifest identifies the native audio features; keep only their hash.
    prepared = generation.get("prepared_native_source")
    if not prepared:
        return generation
    return {
        **generation,
        "prepared_native_source": {
            key: value for key, value in prepared.items() if key != "data"
        },
    }


def run(args):
    cfg = ESPnetRLConfig.load(args.config)
    if cfg.family != "bagpiper" or cfg.rollout_backend != "sglang":
        raise ValueError("Bagpiper evaluation requires its native SGLang configuration")
    manifest = args.manifest or cfg.eval_manifest
    if not manifest:
        raise ValueError("Set a separate held-out eval_manifest or --manifest")
    if args.verify_native and args.temperature != 1.0:
        raise ValueError("Only sampled actions have native probabilities to verify")
    sources = read_manifest(manifest)
    validate_manifests(read_manifest(cfg.manifest), sources)
    task = sources[0]["task"]
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cfg.checkpoint, cfg.output_dir = str(checkpoint), str(directory)
    cfg.eval_manifest = str(Path(manifest).resolve())
    cfg.device = args.device
    cfg.adapter_options = {**cfg.adapter_options, "reward_device": args.device}
    # What the engine decodes: ``samples`` candidates per prompt (one for a
    # greedy decode, so this record is not a valid training config) within
    # the evaluation's token budget.
    cfg.group_size, cfg.max_tokens = args.samples, args.max_new_tokens
    sampling = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    (directory / "frozen_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n"
    )
    (directory / "requests.json").write_text(json.dumps(sources, indent=2) + "\n")
    protocol = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest": cfg.eval_manifest,
        "manifest_sha256": sha256(manifest),
        "training_manifest": str(Path(cfg.manifest).resolve()),
        "training_manifest_sha256": sha256(cfg.manifest),
        "config_sha256": sha256(args.config),
        "policy_step": args.policy_step,
        "task": task,
        "sampling": sampling,
        "samples_per_prompt": args.samples,
        "audio_seconds": args.audio_seconds if task == "qa" else None,
    }
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if args.responses:
            result = read_json(args.responses)
            protocol.update(
                saved_responses=str(Path(args.responses).resolve()),
                saved_responses_sha256=sha256(args.responses),
            )
        else:
            # The engine runs in this process, which is never a training rank.
            environment = subprocess_environment(
                cfg, os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            )
            os.environ.clear()
            os.environ.update(environment)
            with (directory / "export.log").open("w") as handle:
                subprocess.run(
                    [
                        cfg.rollout_python,
                        "-m",
                        "espnet2.rl.export_worker",
                        "--config",
                        str(directory / "frozen_config.json"),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(directory / "hf"),
                    ],
                    env=environment,
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            export = verify_export(checkpoint, directory / "hf")
            (directory / "export_audit.json").write_text(
                json.dumps(export, indent=2) + "\n"
            )
            protocol["exact_native_export_tensors"] = export[
                "exact_native_export_tensors"
            ]
            if task == "qa":
                protocol["clipped_audio"] = clip_audio(
                    sources, directory / "audio", args.audio_seconds
                )
            generations = generate(cfg, sources, directory, sampling, args.batch_size)
            result = {
                "policy_step": args.policy_step,
                "native_sha256": protocol["checkpoint_sha256"],
                "rollout_backend": "sglang",
                "sampling": sampling,
                "generations": (
                    generations
                    if args.verify_native
                    else [_without_features(item) for item in generations]
                ),
            }
            write_json(directory / "responses.json", result)
            result["generations"] = generations
        (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        generations = validate_responses(
            result,
            sources,
            args.samples,
            protocol["checkpoint_sha256"],
            args.policy_step,
        )
        if task == "qa":
            items, summary = score_mmau(sources, generations, args.samples)
        else:
            reward = ASRWERReward(cfg, audits=None)
            items, summary = score_tts(reward, sources, generations, args.samples)
            del reward
        if args.verify_native:
            batch = collate_rollouts(sources, generations, args.samples)
            batch.save_to_disk(directory / "rollout.pkl")
            policy = build_policy(cfg, cfg.device).requires_grad_(False)
            errors = verify_native_log_probs(policy, batch, cfg.score_tolerance)
            summary["native_logprob_max_error"] = max(errors)
        with (directory / "items.jsonl").open("w") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary = {"status": "passed", **protocol, **summary}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (directory / "completion.json").write_text(
            json.dumps({"status": "Succeeded", "summary": "summary.json"}) + "\n"
        )
        print(json.dumps(summary, indent=2))
        return summary
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
    parser.add_argument("--policy-step", type=int, default=-1)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="0 decodes greedily"
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--samples", type=int, default=1, help="Candidates decoded per prompt"
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=30.0,
        help="QA clips are cut to their first seconds",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Prompts per engine rollout"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="ASR reward and native verification"
    )
    parser.add_argument(
        "--verify-native",
        action="store_true",
        help="Check every sampled action's probability against the checkpoint",
    )
    parser.add_argument(
        "--responses", help="Score existing responses without new generation"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
