# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Prepare LibriTTS, AVQA and user Kaldi manifests for Bagpiper RL stages."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

# Option letters such as "A. " or "(A) " that some MMAU/AVQA exports prepend.
# The official MMAU references and choices carry none.
OPTION_LABEL = re.compile(r"^(?:[A-Z]\.|\([A-Z]\))\s*")
QA_TEMPLATE = "{question} Please choose the answer from the following options: {choices}. Output the final answer in <answer> </answer>."
TTS_TEMPLATE = "Please read the following text aloud verbatim: {text}"
QA_THINK_TEMPLATE = "{question} Please choose the answer from the following options: {choices}. Output the thinking process in <think> </think> and final answer in <answer> </answer>."


def qa_prompt(question, choices, template="default"):
    selected = {"default": QA_TEMPLATE, "think": QA_THINK_TEMPLATE}[template]
    return selected.format(question=question, choices=choices)


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
            extra = source["extra_info"]
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
        if dataset == "avqa":
            question = question.replace("video", "audio")
        # Strip option letters from the answer and every choice alike, so the
        # prompt and the official matcher see the same label-free options.
        answer = OPTION_LABEL.sub("", answer).strip()
        choices = [OPTION_LABEL.sub("", str(choice)).strip() for choice in choices]
        records.append(
            {
                "id": key,
                "audio": audio,
                "reference": answer,
                "question": question,
                "choices": choices,
                "prompt": QA_TEMPLATE.format(question=question, choices=choices),
                "task": "qa",
                "dataset": dataset,
            }
        )
    return records


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
    parser.add_argument("--hf-revision")
    parser.add_argument(
        "--qa-template", choices=("default", "think"), default="default"
    )
    args = parser.parse_args()
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
    if args.limit:
        records = records[: args.limit]
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
                "hf_revision": args.hf_revision,
                "qa_template": args.qa_template,
                "template": (
                    (
                        QA_TEMPLATE
                        if args.qa_template == "default"
                        else QA_THINK_TEMPLATE
                    )
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
