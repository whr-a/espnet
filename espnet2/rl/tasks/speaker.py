# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Closed-set speaker identification from Kaldi ``wav.scp`` and ``utt2spk``."""

import random

from .kaldi import (
    audio_directory,
    input_files,
    materialize_kaldi_audio,
    read_kaldi,
    write_manifest,
)

SPEAKER_PROMPT = (
    "Listen to the audio and identify who is speaking. "
    "Choose exactly one of these speakers: {choices}. "
    "Put the chosen speaker label inside <answer></answer>."
)


def speaker_choice_set(speaker, inventory, size, seed, key=None):
    """The true speaker plus deterministic distractors, in a seeded order.

    ``key`` is what the draw is keyed on besides the seed, and it decides what
    the candidate list can leak. Keyed on the speaker (the default), every
    utterance of one speaker is offered the same list in the same order; that
    is harmless while training and heldout hold *different* speakers, but a
    closed-set split that keeps one speaker inventory and divides it by
    utterance then makes the list a function of the answer, so a model can
    learn "this list means that label" and never listen to the audio. Pass the
    utterance id to key the draw per utterance and close that channel; the
    shuffle uses the same generator, so the order stops leaking too.
    """
    others = sorted(set(inventory) - {speaker})
    if len(others) + 1 < size:
        raise ValueError(
            f"A {size}-way speaker choice set needs {size} distinct speakers; the "
            f"inventory has {len(others) + 1}"
        )
    generator = random.Random(f"{seed}:{speaker if key is None else key}")
    choices = [speaker, *generator.sample(others, size - 1)]
    generator.shuffle(choices)
    return choices


def prepare_speaker_id_manifest(
    wav_scp,
    utt2spk,
    output,
    *,
    choices=4,
    seed=42,
    sample_rate=16000,
    prompt=None,
    choice_key="utterance",
):
    """Closed-set speaker identification from Kaldi ``wav.scp`` and ``utt2spk``.

    ``choice_key`` defaults to ``"utterance"`` because that is the safe choice
    in both split designs and costs nothing. Keying on the speaker makes the
    candidate list a per-speaker constant, which leaks the answer whenever
    training and heldout share one speaker inventory -- measured on this
    corpus, 2674 records collapsed to 40 distinct lists and list-to-label was
    injective, so a model could memorize 40 prompts and never listen. That
    mode stays reachable for reproducing an older manifest, but it has to be
    asked for.

    ``choice_key`` selects what the candidate draw is keyed on: ``"speaker"``
    repeats one list for all of that speaker's utterances, ``"utterance"``
    draws a fresh list per utterance. Use ``"utterance"`` whenever training and
    heldout share the speaker inventory, or the candidate list identifies the
    answer on its own -- see ``speaker_choice_set``.
    """
    if choice_key not in ("speaker", "utterance"):
        raise ValueError("choice_key must be speaker or utterance")
    speakers = read_kaldi(utt2spk)
    records = materialize_kaldi_audio(
        wav_scp, utt2spk, audio_directory(output), sample_rate
    )
    inventory = sorted(set(speakers.values()))
    template = prompt or SPEAKER_PROMPT
    for record in records:
        speaker = speakers[record["id"]]
        options = speaker_choice_set(
            speaker,
            inventory,
            choices,
            seed,
            None if choice_key == "speaker" else record["id"],
        )
        # materialize_kaldi_audio joined utt2spk as its reference column; the
        # speaker label is supervision, so name it as such and drop the alias.
        record.pop("reference")
        record.update(
            reward_task="speaker_id",
            speaker=speaker,
            speaker_choices=options,
            # The validator reads this to tell an acknowledged per-speaker draw
            # from a manifest that leaks the answer by accident.
            choice_key=choice_key,
            prompt=template.format(choices=", ".join(options)),
        )
    return write_manifest(
        output,
        records,
        {
            "task": "speaker_id",
            "choices": choices,
            "choice_key": choice_key,
            "seed": seed,
            "sample_rate": sample_rate,
            "speakers": len(inventory),
            **input_files(wav_scp=wav_scp, utt2spk=utt2spk),
            "prompt_template": template,
        },
    )


def add_arguments(parser):
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--speaker-choices", type=int, default=4)
    parser.add_argument(
        "--speaker-choice-key",
        choices=("speaker", "utterance"),
        default="utterance",
        help="Key the candidate draw per utterance (default, safe in both split "
        "designs) or per speaker. Per speaker makes the candidate list a "
        "per-speaker constant, which leaks the answer when train and heldout "
        "share one speaker inventory; it is only for reproducing an older "
        "manifest",
    )
    parser.add_argument("--seed", type=int, default=42)


def run(args):
    return prepare_speaker_id_manifest(
        args.wav_scp,
        args.utt2spk,
        args.output,
        choices=args.speaker_choices,
        seed=args.seed,
        sample_rate=args.sample_rate,
        choice_key=args.speaker_choice_key,
    )
