# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Prepare LibriTTS, AVQA and user Kaldi manifests for Bagpiper RL stages."""

import argparse
import hashlib
import importlib.util
import io
import json
import random
import re
from functools import lru_cache
from pathlib import Path

# Option letters such as "A. " or "(A) " that some MMAU/AVQA exports prepend.
# The official MMAU references and choices carry none.
OPTION_LABEL = re.compile(r"^(?:[A-Z]\.|\([A-Z]\))\s*")
QA_TEMPLATE = "{question} Please choose the answer from the following options: {choices}. Output the final answer in <answer> </answer>."
TTS_TEMPLATE = "Please read the following text aloud verbatim: {text}"
QA_THINK_TEMPLATE = "{question} Please choose the answer from the following options: {choices}. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
# MMAU's own groups of an item, kept for per-group held-out accuracy.
CATEGORIES = ("task", "category", "sub-category", "difficulty")


@lru_cache(maxsize=1)
def _native_qa_contract():
    # Keep standalone corpus preparation independent of model-package imports.
    spec = importlib.util.spec_from_file_location(
        "bagpiper_qa_contract", Path(__file__).with_name("qa_contract.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qa_prompt(question, choices, template="default"):
    if template == "bagpiper":
        return _native_qa_contract().native_qa_prompt(question, choices)
    selected = {"default": QA_TEMPLATE, "think": QA_THINK_TEMPLATE}[template]
    return selected.format(question=question, choices=choices)


def qa_template_description(template):
    """Bind both native system conditioning and user rendering in data receipts."""
    if template == "bagpiper":
        return (
            qa_system_prompt(template)
            + "\n\nquestion: {question}\n Options: {semicolon-separated choices}"
        )
    return {"default": QA_TEMPLATE, "think": QA_THINK_TEMPLATE}[template]


def qa_system_prompt(template):
    if template == "bagpiper":
        return _native_qa_contract().BAGPIPER_QA_SYSTEM_PROMPT
    return None


def validate_native_qa_sources(records):
    """Reject mismatched native QA conditioning before training starts.

    Existing manifests are read-only: regenerate them with the native template
    rather than silently changing the prompt under an admitted experiment.
    """
    contract = _native_qa_contract()
    for record in records:
        choices = record.get("choices")
        question = record.get("question")
        valid = (
            record.get("task") == "qa"
            and record.get("qa_template") == "bagpiper"
            and record.get("system_prompt") == contract.BAGPIPER_QA_SYSTEM_PROMPT
            and isinstance(question, str)
            and bool(question.strip())
            and isinstance(choices, list)
            and len(choices) >= 2
            and all(isinstance(choice, str) and choice.strip() for choice in choices)
            and record.get("prompt") == contract.native_qa_prompt(question, choices)
            and isinstance(record.get("reference"), str)
            and contract.match_choice(record.get("reference", ""), choices)[0]
            is not None
        )
        if not valid:
            raise ValueError(
                f"Native Bagpiper QA source {record.get('id')!r} requires matching "
                "task, qa_template=bagpiper, system, question/options and reference; "
                "regenerate the manifest with --qa-template bagpiper "
                "(run.sh: --rl-qa-template bagpiper). Existing manifests are not rewritten."
            )


def _kaldi(path):
    values = {}
    for row in Path(path).read_text().splitlines():
        if not row.strip():
            continue
        key, value = row.split(maxsplit=1)
        if key in values:
            raise ValueError(f"Duplicate utterance {key}")
        values[key] = value
    return values


def prepare_kaldi(wav_scp, text, task, questions=None, choices=None, output_dir=None):
    from espnet2.rl.prepare import materialize_kaldi_audio

    if output_dir is None:
        raise ValueError("An output directory is required to freeze wav.scp audio")
    records = materialize_kaldi_audio(
        wav_scp, text, Path(output_dir) / "audio", channel="first"
    )
    question = _kaldi(questions) if questions else {}
    choice = json.loads(Path(choices).read_text()) if choices else {}
    for record in records:
        key = record["id"]
        record.update(task=task, dataset="user_wav_scp")
        if task == "tts_asr_wer":
            record["prompt"] = TTS_TEMPLATE.format(text=record["reference"])
        else:
            if key not in question:
                raise ValueError(f"Missing QA question for {key}")
            record.update(
                question=question[key],
                choices=choice.get(key, []),
                prompt=QA_TEMPLATE.format(
                    question=question[key], choices=choice.get(key, [])
                ),
            )
    return records


def prepare_libritts(directory):
    records = []
    for wav in sorted(Path(directory).rglob("*.wav")):
        text = wav.with_suffix(".normalized.txt")
        if not text.is_file():
            continue
        reference = text.read_text().strip()
        records.append(
            {
                "id": wav.stem,
                "audio": str(wav.resolve()),
                "reference": reference,
                "prompt": TTS_TEMPLATE.format(text=reference),
                "task": "tts_asr_wer",
                "dataset": "LibriTTS",
            }
        )
    if not records:
        raise ValueError("No LibriTTS WAV/normalized transcript pairs")
    return records


def prepare_parquet(path, dataset, output_dir):
    import pandas as pd
    import soundfile as sf

    records = []
    frame = pd.read_parquet(path)
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for index, row in frame.iterrows():
        source = row.to_dict()
        if "extra_info" in source:  # Existing cached prepare_avqa/prepare_mmau.
            extra = attributes = source["extra_info"]
            audio = extra["audio_path"]
            question = extra["question"]
            choices = list(extra.get("choices", []))
            answer = source["reward_model"]["ground_truth"]
            key = str(extra.get("id", f"{dataset}-{index:06d}"))
        else:
            value = source.get("audio", source.get("context"))
            data = value.get("bytes") if isinstance(value, dict) else value
            if data:
                waveform, rate = sf.read(io.BytesIO(data))
                audio = str((audio_dir / f"{dataset}-{index:06d}.wav").resolve())
                sf.write(audio, waveform, rate)
            else:
                audio = value["path"]
            question = source.get("question", source.get("instruction"))
            choices = list(source.get("choices", []))
            if dataset == "avqa" and "\nChoices:\n" in question:
                question, option = question.split("\nChoices:\n", 1)
                choices = [
                    re.sub(r"^[A-Z]\.\s*", "", item.strip())
                    for item in option.splitlines()
                    if item.strip()
                ]
            answer = source["answer"]
            key = str(source.get("id", f"{dataset}-{index:06d}"))
            attributes = json.loads(source.get("other_attributes") or "{}")
        if dataset == "avqa":
            question = question.replace("video", "audio")
        # Strip option letters from the answer and every choice alike, so the
        # prompt and the official matcher see the same label-free options.
        answer = OPTION_LABEL.sub("", answer).strip()
        choices = [OPTION_LABEL.sub("", str(choice)).strip() for choice in choices]
        record = {
            "id": key,
            "audio": audio,
            "reference": answer,
            "question": question,
            "choices": choices,
            "prompt": QA_TEMPLATE.format(question=question, choices=choices),
            "task": "qa",
            "dataset": dataset,
        }
        categories = {
            name: attributes[name] for name in CATEGORIES if name in attributes
        }
        if categories:
            record["categories"] = categories
        records.append(record)
    return records


def select(records, max_words=None, shuffle_seed=None, limit=None):
    """References of at most ``max_words`` WER words, in the source order or
    shuffled by ``random.Random(shuffle_seed)``, then the first ``limit``.

    Prompts are taken in manifest order, so a shuffled manifest is how a run
    sees a random sample of the corpus without repeating a prompt.
    """
    from espnet2.rl.rewards import words

    if max_words is not None:
        records = [row for row in records if len(words(row["reference"])) <= max_words]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)
    return records[:limit] if limit else records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("libritts", "avqa", "mmau", "wav_scp"), required=True
    )
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wav-scp")
    parser.add_argument("--text")
    parser.add_argument("--questions")
    parser.add_argument("--choices")
    parser.add_argument("--task", choices=("tts_asr_wer", "qa"), default="qa")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Shuffle with random.Random(seed) before --limit (default: source order)",
    )
    parser.add_argument(
        "--max-words", type=int, help="Keep TTS references of at most this many words"
    )
    parser.add_argument("--hf-revision")
    parser.add_argument(
        "--qa-template", choices=("default", "think", "bagpiper"), default="default"
    )
    args = parser.parse_args()
    if args.max_words is not None and args.task != "tts_asr_wer":
        parser.error("--max-words filters TTS references")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen manifest {output}")
    if args.dataset == "wav_scp":
        records = prepare_kaldi(
            args.wav_scp,
            args.text,
            args.task,
            args.questions,
            args.choices,
            output.parent,
        )
    elif args.dataset == "libritts":
        records = prepare_libritts(args.input)
    else:
        source = args.input
        if source is None:
            if args.dataset == "mmau":
                raise ValueError(
                    "MMAU preparation requires its own official parquet via --input"
                )
            if not args.hf_revision:
                raise ValueError(
                    "Remote AVQA download needs --hf-revision to freeze provenance"
                )
            from datasets import Audio, load_dataset

            ds = load_dataset(
                "gijs/avqa-processed", revision=args.hf_revision, split="train"
            ).cast_column("audio", Audio(decode=False))
            source = output.parent / "avqa-source.parquet"
            ds.to_parquet(str(source))
        records = prepare_parquet(source, args.dataset, output.parent)
    if args.task == "qa":
        for record in records:
            record["prompt"] = qa_prompt(
                record["question"], record["choices"], args.qa_template
            )
            record["qa_template"] = args.qa_template
            if args.qa_template == "bagpiper":
                record["system_prompt"] = qa_system_prompt(args.qa_template)
    prepared = len(records)
    records = select(records, args.max_words, args.shuffle_seed, args.limit)
    if not records:
        raise ValueError("No RL records prepared")
    encoded = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    output.write_text(encoded)
    (output.with_suffix(output.suffix + ".audit.json")).write_text(
        json.dumps(
            {
                "count": len(records),
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "input": args.input,
                "dataset": args.dataset,
                "prepared": prepared,
                "max_words": args.max_words,
                "shuffle_seed": args.shuffle_seed,
                "limit": args.limit,
                "hf_revision": args.hf_revision,
                "qa_template": args.qa_template,
                "native_qa_contract_sha256": (
                    hashlib.sha256(
                        Path(__file__).with_name("qa_contract.py").read_bytes()
                    ).hexdigest()
                    if args.qa_template == "bagpiper"
                    else None
                ),
                "template": (
                    qa_template_description(args.qa_template)
                    if args.task == "qa"
                    else TTS_TEMPLATE
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
