# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Two-speaker mixtures and their per-speaker references, for cpWER."""

import hashlib
import random

from espnet2.rl.utils import sha256

from .kaldi import (
    audio_directory,
    input_files,
    materialize_kaldi_audio,
    read_kaldi,
    write_manifest,
)

MULTISPEAKER_PROMPT = (
    "Two people speak in this audio. Transcribe each speaker separately. "
    "Write {tags} before that speaker's words, each tag exactly once."
)
#: The instruction a captioner-pretrained policy needs before cpWER can
#: separate anything. Measured on 256 mixtures at 0.3 overlap with one
#: unchanged Bagpiper policy: under MULTISPEAKER_PROMPT every candidate
#: clipped to the same reward -- 1 distinct value over the whole set, cpWER
#: clipped mean exactly 1.0, no mixture below cpWER 1 -- so the GRPO group
#: advantage is identically zero and no number of updates can move the
#: policy. Under this instruction the same policy and the same mixtures give
#: 29 distinct rewards and put 11.7% of mixtures under cpWER 1. It asks for a
#: transcript rather than a description and forbids the commentary a captioner
#: adds, which is what moves 95.8 words per response out of the unattributed
#: preamble.
MULTISPEAKER_VERBATIM_PROMPT = (
    "Write the verbatim transcript of this two-speaker audio, one line per "
    "speaker, each line starting with that speaker's tag. Use {tag0} for one "
    "speaker and {tag1} for the other. Write only the transcribed words. No "
    "description, no commentary, no quotation marks."
)
#: Instruction texts ``--prompt-template`` may name. A template may use
#: ``{tags}`` for the tags joined with "and", or ``{tag0}``/``{tag1}`` for one
#: tag each.
MULTISPEAKER_PROMPTS = {
    "default": MULTISPEAKER_PROMPT,
    "verbatim_tagged": MULTISPEAKER_VERBATIM_PROMPT,
}


def mix_two_speakers(first, second, overlap, sample_rate):
    """Sum two waveforms with the second delayed, then scale to avoid clipping.

    ``overlap`` is the share of the shorter utterance that plays at the same
    time as the first one: 0 concatenates, 1 starts both together. The peak is
    rescaled only when the sum would clip, so a non-overlapped region keeps
    its original amplitude.
    """
    import numpy as np

    if not 0 <= overlap <= 1:
        raise ValueError("overlap must lie in [0, 1]")
    delay = int(round(len(first) * (1 - overlap)))
    length = max(len(first), delay + len(second))
    mixture = np.zeros(length, dtype="float32")
    mixture[: len(first)] += first
    mixture[delay : delay + len(second)] += second
    peak = float(np.abs(mixture).max())
    if peak > 1.0:
        mixture /= peak
    return mixture, delay / sample_rate


def prepare_multispeaker_manifest(
    wav_scp,
    text,
    utt2spk,
    output,
    *,
    overlap=0.3,
    seed=42,
    sample_rate=16000,
    tags=("<spk0>", "<spk1>"),
    prompt=None,
):
    """Two-speaker mixtures and their per-speaker references, for cpWER.

    Utterances are paired across *different* speakers in a seeded order, so a
    pair never mixes one speaker with themselves. The references stay separate
    per speaker and in tag order; the mixture is the only model input.
    """
    import soundfile as sf

    speakers = read_kaldi(utt2spk)
    base = materialize_kaldi_audio(
        wav_scp, text, audio_directory(output), sample_rate
    )
    by_speaker = {}
    for record in base:
        by_speaker.setdefault(speakers[record["id"]], []).append(record)
    if len(by_speaker) < 2:
        raise ValueError("A two-speaker mixture needs utterances from two speakers")
    generator = random.Random(seed)
    pools = {name: list(items) for name, items in sorted(by_speaker.items())}
    for items in pools.values():
        generator.shuffle(items)
    order = sorted(pools)
    directory = audio_directory(output)
    records, index = [], 0
    while sum(len(items) for items in pools.values()) >= 2:
        # Always draw from the two fullest pools, so one speaker's surplus
        # never pairs with itself and the mixture count is maximal.
        first_name, second_name = sorted(
            order, key=lambda name: (-len(pools[name]), name)
        )[:2]
        if not pools[second_name]:
            break
        first, second = pools[first_name].pop(), pools[second_name].pop()
        waves = [sf.read(item["audio"], dtype="float32")[0] for item in (first, second)]
        mixture, delay = mix_two_speakers(*waves, overlap, sample_rate)
        identity = f"mix{index:04d}-{first['id']}-{second['id']}"
        target = directory / (hashlib.sha256(identity.encode()).hexdigest()[:20] + ".wav")
        sf.write(str(target), mixture, sample_rate, subtype="FLOAT")
        records.append(
            {
                "id": identity,
                "audio": str(target),
                "sample_rate": sample_rate,
                "audio_sha256": sha256(target),
                "reward_task": "multispeaker_cpwer",
                "references": [first["reference"], second["reference"]],
                "speakers": [first_name, second_name],
                "speaker_tags": list(tags),
                "source_ids": [first["id"], second["id"]],
                "source_audio_sha256": [first["audio_sha256"], second["audio_sha256"]],
                "overlap": overlap,
                "second_onset_seconds": delay,
                "prompt": (prompt or MULTISPEAKER_PROMPT).format(
                    tags=" and ".join(tags),
                    **{f"tag{position}": tag for position, tag in enumerate(tags)},
                ),
            }
        )
        index += 1
    if not records:
        raise ValueError("No cross-speaker pair could be formed")
    return write_manifest(
        output,
        records,
        {
            "task": "multispeaker_cpwer",
            "overlap": overlap,
            "seed": seed,
            "sample_rate": sample_rate,
            "tags": list(tags),
            "speakers": len(by_speaker),
            **input_files(wav_scp=wav_scp, text=text, utt2spk=utt2spk),
            "prompt_template": prompt or MULTISPEAKER_PROMPT,
        },
    )


def add_arguments(parser):
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument(
        "--prompt-template",
        choices=tuple(MULTISPEAKER_PROMPTS),
        default="default",
        help="Instruction text. Use verbatim_tagged when the starting policy is "
        "a captioner: with the default instruction such a policy scores every "
        "candidate at the cpWER clip, which leaves the GRPO group no advantage "
        "to learn from.",
    )
    parser.add_argument("--seed", type=int, default=42)


def run(args):
    return prepare_multispeaker_manifest(
        args.wav_scp,
        args.text,
        args.utt2spk,
        args.output,
        overlap=args.overlap,
        seed=args.seed,
        sample_rate=args.sample_rate,
        prompt=MULTISPEAKER_PROMPTS[args.prompt_template],
    )
