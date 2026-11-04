# Copyright 2025 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""ASR manifests from Kaldi ``wav.scp`` and ``text``, joined by utterance ID."""

from pathlib import Path

from espnet2.rl.utils import sha256

from .kaldi import audio_directory, materialize_kaldi_audio, write_manifest


def prepare_kaldi_manifest(wav_scp, text, output, task="asr_wer", **kwargs):
    """Materialize pipe audio, normalize rate, and join references by utterance ID."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifest: {output}")
    sample_rate = int(kwargs.get("sample_rate", 16000))
    records = materialize_kaldi_audio(
        wav_scp, text, audio_directory(output), sample_rate
    )
    for record in records:
        record["reward_task"] = task
    return write_manifest(
        output,
        records,
        {
            "task": task,
            "sample_rate": sample_rate,
            "wav_scp": str(Path(wav_scp).resolve()),
            "text": str(Path(text).resolve()),
            "wav_scp_sha256": sha256(wav_scp),
            "text_sha256": sha256(text),
        },
    )


def add_arguments(parser):
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reward-task",
        default="asr_wer",
        help="The reward_task stamped into every record",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)


def run(args):
    return prepare_kaldi_manifest(
        args.wav_scp,
        args.text,
        args.output,
        args.reward_task,
        sample_rate=args.sample_rate,
    )
