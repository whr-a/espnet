# Copyright 2026 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auditable manifests for the task rewards in :mod:`espnet2.rl.rewards`.

Three preparers read ordinary Kaldi recipe inputs and write a manifest whose
records carry exactly the supervision their reward declares:

* ``speaker_id`` from ``wav.scp`` and ``utt2spk``, with a deterministic
  closed choice set per utterance;
* ``longform_wer`` from ``wav.scp`` and ``text``, optionally concatenating
  consecutive same-session utterances into genuinely long references;
* ``multispeaker_cpwer`` from single-speaker inputs, mixing two speakers at a
  declared overlap so cpWER has real overlapped speech to score.

``derive_task_manifest`` is the fourth entry: it attaches a new task's
supervision to an **already prepared** manifest without touching its audio
condition, so a family whose prompt is a prepared token matrix (OpusLM) can
reuse an audited condition and change only the reward. Every output records
the inputs it came from and their hashes, as :mod:`espnet2.rl.prepare` does.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .prepare import _kaldi, materialize_kaldi_audio
from .rewards.text import words

SPEAKER_PROMPT = (
    "Listen to the audio and identify who is speaking. "
    "Choose exactly one of these speakers: {choices}. "
    "Put the chosen speaker label inside <answer></answer>."
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


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(output, records, audit):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifest: {output}")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(
            {**audit, "rows": len(records), "manifest_sha256": _digest(output)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return records


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
    speakers = _kaldi(utt2spk)
    records = materialize_kaldi_audio(
        wav_scp, utt2spk, Path(output).parent / (Path(output).stem + "_audio"), sample_rate
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
    return _write(
        output,
        records,
        {
            "task": "speaker_id",
            "choices": choices,
            "choice_key": choice_key,
            "seed": seed,
            "sample_rate": sample_rate,
            "speakers": len(inventory),
            "wav_scp": str(Path(wav_scp).resolve()),
            "utt2spk": str(Path(utt2spk).resolve()),
            "wav_scp_sha256": _digest(wav_scp),
            "utt2spk_sha256": _digest(utt2spk),
            "prompt_template": template,
        },
    )



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
        wav_scp, text, Path(output).parent / (Path(output).stem + "_audio"), sample_rate
    )
    if concatenate > 1:
        grouped = {}
        for record in records:
            grouped.setdefault(session_of(record["id"]), []).append(record)
        joined = []
        directory = Path(output).parent / (Path(output).stem + "_audio")
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
                        "audio_sha256": _digest(target),
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
    return _write(
        output,
        records,
        {
            "task": "longform_wer",
            "min_reference_words": min_reference_words,
            "concatenate": concatenate,
            "max_audio_seconds": max_audio_seconds,
            "sample_rate": sample_rate,
            "wav_scp": str(Path(wav_scp).resolve()),
            "text": str(Path(text).resolve()),
            "wav_scp_sha256": _digest(wav_scp),
            "text_sha256": _digest(text),
            "reference_words": {
                record["id"]: len(words(record["reference"])) for record in records
            },
        },
    )


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

    speakers = _kaldi(utt2spk)
    base = materialize_kaldi_audio(
        wav_scp, text, Path(output).parent / (Path(output).stem + "_audio"), sample_rate
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
    directory = Path(output).parent / (Path(output).stem + "_audio")
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
                "audio_sha256": _digest(target),
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
    return _write(
        output,
        records,
        {
            "task": "multispeaker_cpwer",
            "overlap": overlap,
            "seed": seed,
            "sample_rate": sample_rate,
            "tags": list(tags),
            "speakers": len(by_speaker),
            "wav_scp": str(Path(wav_scp).resolve()),
            "text": str(Path(text).resolve()),
            "utt2spk": str(Path(utt2spk).resolve()),
            "wav_scp_sha256": _digest(wav_scp),
            "text_sha256": _digest(text),
            "utt2spk_sha256": _digest(utt2spk),
            "prompt_template": prompt or MULTISPEAKER_PROMPT,
        },
    )


def append_text_instruction(prompt_token_ids, text_ids, text_token_start):
    """Append tokenized instruction text to a prepared multi-stream prompt.

    A prepared OpusLM prompt is ``[frames, nq]``; its primary stream carries
    native control and text ids while the secondary streams carry codec/SSL
    conditions. An instruction is text, so it occupies the primary stream only
    and pads the rest, exactly as ``_native_prompt`` builds a text prompt.
    """
    if not prompt_token_ids or not all(
        len(frame) == len(prompt_token_ids[0]) for frame in prompt_token_ids
    ):
        raise ValueError("A prepared prompt must be a nonempty rectangular [P, nq]")
    width = len(prompt_token_ids[0])
    return [list(frame) for frame in prompt_token_ids] + [
        [int(token) + int(text_token_start)] + [0] * (width - 1) for token in text_ids
    ]


def derive_task_manifest(
    source_manifests,
    output,
    *,
    task,
    supervision,
    instruction_tokens=None,
    text_token_start=None,
    drop=(),
):
    """Attach a new task's supervision to already prepared records.

    ``supervision`` maps a record id to the fields that task's reward needs.
    The audio condition — ``prompt_token_ids``, ``prompt_codec_mask``,
    ``native_preparation`` — is copied unchanged, so the derived manifest
    reuses an audited preparation instead of re-encoding audio. When
    ``instruction_tokens`` is given, those text ids are appended to each
    prompt and the original prompt hash is kept beside the new one, because
    the prompt is no longer the one ``native_preparation`` describes.
    """
    records, seen = [], set()
    for path in source_manifests:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                raise ValueError(f"Duplicate source record {record['id']!r}")
            seen.add(record["id"])
            fields = supervision.get(record["id"])
            if fields is None:
                raise ValueError(f"No {task} supervision for record {record['id']!r}")
            for name in drop:
                record.pop(name, None)
            if instruction_tokens is not None:
                if text_token_start is None:
                    raise ValueError(
                        "Appending an instruction needs the export's text_token_start"
                    )
                original = record["prompt_token_ids"]
                record["prompt_token_ids"] = append_text_instruction(
                    original, instruction_tokens, text_token_start
                )
                record["prompt_codec_mask"] = list(
                    record.get("prompt_codec_mask", [False] * len(original))
                ) + [False] * len(instruction_tokens)
                record["derived_prompt"] = {
                    "appended_text_tokens": list(instruction_tokens),
                    "text_token_start": int(text_token_start),
                    "original_frames": len(original),
                    "frames": len(record["prompt_token_ids"]),
                }
            # A mixed manifest's supervision names each record's own task, so
            # its reward_task wins over the one this call was given.
            record.update({"reward_task": task, **fields})
            records.append(record)
    if not records:
        raise ValueError("No source record was derived")
    return _write(
        output,
        records,
        {
            "task": task,
            "derived_from": [str(Path(path).resolve()) for path in source_manifests],
            "derived_from_sha256": [_digest(path) for path in source_manifests],
            "instruction_tokens": (
                None if instruction_tokens is None else list(instruction_tokens)
            ),
            "text_token_start": text_token_start,
            "dropped_fields": list(drop),
            "audio_condition": "copied unchanged from the source manifest",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=("speaker_id", "longform_wer", "multispeaker_cpwer"),
    )
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text")
    parser.add_argument("--utt2spk")
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
    parser.add_argument("--min-reference-words", type=int, default=200)
    parser.add_argument("--concatenate", type=int, default=0)
    parser.add_argument(
        "--train-config",
        help="The model's training YAML; its preprocessor_conf.speech_length is "
        "the audio window. Required for longform_wer unless --max-audio-seconds "
        "is given.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        help="The model's audio window. A fixed-window frontend rejects longer "
        "audio outright, so declaring it fails the manifest closed here. OWSM v4 "
        "is 30 (preprocessor_conf.speech_length).",
    )
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument(
        "--prompt-template",
        choices=tuple(MULTISPEAKER_PROMPTS),
        default="default",
        help="Instruction text for multispeaker_cpwer. Use verbatim_tagged "
        "when the starting policy is a captioner: with the default "
        "instruction such a policy scores every candidate at the cpWER clip, "
        "which leaves the GRPO group no advantage to learn from.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task == "speaker_id":
        if not args.utt2spk:
            parser.error("speaker_id requires --utt2spk")
        prepare_speaker_id_manifest(
            args.wav_scp,
            args.utt2spk,
            args.output,
            choices=args.speaker_choices,
            seed=args.seed,
            sample_rate=args.sample_rate,
            choice_key=args.speaker_choice_key,
        )
    elif args.task == "longform_wer":
        if not args.text:
            parser.error("longform_wer requires --text")
        prepare_longform_manifest(
            args.wav_scp,
            args.text,
            args.output,
            min_reference_words=args.min_reference_words,
            concatenate=args.concatenate,
            sample_rate=args.sample_rate,
            max_audio_seconds=args.max_audio_seconds,
            train_config=args.train_config,
        )
    else:
        if not args.text or not args.utt2spk:
            parser.error("multispeaker_cpwer requires --text and --utt2spk")
        prepare_multispeaker_manifest(
            args.wav_scp,
            args.text,
            args.utt2spk,
            args.output,
            overlap=args.overlap,
            seed=args.seed,
            sample_rate=args.sample_rate,
            prompt=MULTISPEAKER_PROMPTS[args.prompt_template],
        )


if __name__ == "__main__":
    main()
