# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Long-form ASR from Kaldi inputs, with an enforced reference-length floor."""

import hashlib
from pathlib import Path

from espnet2.rl.rewards.text import words
from espnet2.rl.utils import sha256

from .kaldi import audio_directory, input_files, materialize_kaldi_audio, write_manifest


def audio_window_seconds(train_config):
    """The model's audio window in seconds, from its own training YAML.

    OWSM carries it as ``preprocessor_conf.speech_length``; a family that
    declares no window returns None, and the caller must then say what the
    limit is rather than assume there is none.
    """
    import yaml

    recipe = yaml.safe_load(Path(train_config).read_text()) or {}
    for section in ("preprocessor_conf", "frontend_conf"):
        value = (recipe.get(section) or {}).get("speech_length")
        if value is not None:
            return float(value)
    return None


def session_of(identity):
    """The recording a Kaldi utterance id belongs to: everything but its index."""
    return identity.rsplit("-", 1)[0]


def prepare_longform_manifest(
    wav_scp,
    text,
    output,
    *,
    min_reference_words=200,
    concatenate=0,
    sample_rate=16000,
    max_audio_seconds=None,
    train_config=None,
):
    """Long-form ASR from Kaldi inputs, with an enforced reference-length floor.

    ``concatenate`` joins that many consecutive utterances of one session into
    a single item, which is how a long-form split is simulated from an
    utterance-segmented corpus; 0 keeps the corpus items as they are. Every
    written record declares ``min_reference_words``, and the reward's
    ``validate_sources`` rejects the manifest if any reference is shorter, so
    a short-utterance set cannot be trained as long-form by mistake.

    The audio window is the second half of that contract, and this preparer
    requires it: pass ``train_config`` (the model's own training YAML, whose
    ``preprocessor_conf.speech_length`` is the window in seconds) or
    ``max_audio_seconds`` outright. A default of "unknown" is what let a
    caller build a set no model could read, so there is no such default. A reference-length floor alone does not make a
    manifest trainable: a family whose frontend takes a fixed window rejects
    audio longer than it outright. OWSM is one — its ``load_audio`` raises
    ``"Audio has N samples, exceeding the OWSM window of M"``
    (``espnet2/s2t/rl/policy.py``) rather than truncating, because long inputs
    need explicit offline segmentation. Pass the window and this preparer
    fails closed here, naming the longest item, instead of letting every
    sample die one by one at the first read of training. For OWSM v4 the
    window is ``preprocessor_conf.speech_length`` seconds from its training
    YAML (30 s at 16 kHz), which caps a packed item at roughly 84 words, so a
    200-word floor and a 30-second window cannot both be met: pack by
    duration against the window and set the floor from the result.
    """
    import numpy as np
    import soundfile as sf

    records = materialize_kaldi_audio(
        wav_scp, text, audio_directory(output), sample_rate
    )
    if concatenate > 1:
        grouped = {}
        for record in records:
            grouped.setdefault(session_of(record["id"]), []).append(record)
        joined = []
        directory = audio_directory(output)
        for session, items in sorted(grouped.items()):
            for start in range(0, len(items) - concatenate + 1, concatenate):
                chunk = items[start : start + concatenate]
                waveform = np.concatenate(
                    [sf.read(item["audio"], dtype="float32")[0] for item in chunk]
                )
                identity = f"{session}-cat{start:04d}x{concatenate}"
                target = directory / (
                    hashlib.sha256(identity.encode()).hexdigest()[:20] + ".wav"
                )
                sf.write(str(target), waveform, sample_rate, subtype="FLOAT")
                joined.append(
                    {
                        "id": identity,
                        "audio": str(target),
                        "reference": " ".join(item["reference"] for item in chunk),
                        "sample_rate": sample_rate,
                        "audio_sha256": sha256(target),
                        "concatenated_ids": [item["id"] for item in chunk],
                        "concatenated_sha256": [item["audio_sha256"] for item in chunk],
                    }
                )
        if not joined:
            raise ValueError(
                f"No session has {concatenate} consecutive utterances to concatenate"
            )
        records = joined
    if max_audio_seconds is None and train_config is not None:
        max_audio_seconds = audio_window_seconds(train_config)
    if max_audio_seconds is None:
        raise ValueError(
            "Long-form preparation needs the model's audio window: pass "
            "train_config (its preprocessor_conf.speech_length) or "
            "max_audio_seconds. A fixed-window frontend rejects longer audio "
            "outright rather than truncating, so a manifest built without the "
            "window can fail on every single sample at the first read of "
            "training."
        )
    if max_audio_seconds <= 0:
        raise ValueError("max_audio_seconds must be positive")
    budget = float(max_audio_seconds)
    over = []
    for record in records:
        seconds = sf.info(record["audio"]).frames / float(
            record.get("sample_rate", sample_rate)
        )
        record["audio_seconds"] = seconds
        record["max_audio_seconds"] = budget
        if seconds > budget:
            over.append((record["id"], round(seconds, 2)))
    if over:
        over.sort(key=lambda row: -row[1])
        raise ValueError(
            f"{len(over)} item(s) exceed the declared {budget} s audio window, "
            f"longest first: {over[:3]}. A fixed-window frontend rejects these "
            f"outright rather than truncating, so lower --concatenate or pack by "
            f"duration against the window instead of by utterance count."
        )
    short = [
        (record["id"], len(words(record["reference"])))
        for record in records
        if len(words(record["reference"])) < min_reference_words
    ]
    if short:
        raise ValueError(
            f"{len(short)} item(s) are shorter than the declared long-form floor "
            f"{min_reference_words} words, e.g. {short[:3]}. Raise --concatenate or "
            f"lower the floor deliberately."
        )
    for record in records:
        record.update(reward_task="longform_wer", min_reference_words=min_reference_words)
    return write_manifest(
        output,
        records,
        {
            "task": "longform_wer",
            "min_reference_words": min_reference_words,
            "concatenate": concatenate,
            "max_audio_seconds": max_audio_seconds,
            "sample_rate": sample_rate,
            **input_files(wav_scp=wav_scp, text=text),
            "reference_words": {
                record["id"]: len(words(record["reference"])) for record in records
            },
        },
    )


def add_arguments(parser):
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-reference-words", type=int, default=200)
    parser.add_argument("--concatenate", type=int, default=0)
    parser.add_argument(
        "--train-config",
        help="The model's training YAML; its preprocessor_conf.speech_length is "
        "the audio window. Required unless --max-audio-seconds is given.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        help="The model's audio window. A fixed-window frontend rejects longer "
        "audio outright, so declaring it fails the manifest closed here. OWSM v4 "
        "is 30 (preprocessor_conf.speech_length).",
    )


def run(args):
    return prepare_longform_manifest(
        args.wav_scp,
        args.text,
        args.output,
        min_reference_words=args.min_reference_words,
        concatenate=args.concatenate,
        sample_rate=args.sample_rate,
        max_audio_seconds=args.max_audio_seconds,
        train_config=args.train_config,
    )
