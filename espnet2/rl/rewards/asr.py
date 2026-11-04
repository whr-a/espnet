# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Speech recognition rewards: WER, its clipped form and long-form WER."""

from . import require_fields, require_text_mode
from .text import word_error_rate, words


def word_error_reward(source, generation):
    """Negative WER of the sampled text: the reward of a recipe that names none."""
    require_text_mode(source, "wer")
    if "reference" not in source:
        raise ValueError("WER reward requires a reference in every manifest record")
    return -word_error_rate(source["reference"], generation["text"])


def asr_wer_reward(source, generation):
    """Negative WER of the sampled text, clipped at a silent response's -1."""
    require_text_mode(source, "asr_wer")
    rate = word_error_rate(source["reference"], generation["text"])
    return {"reward": -min(rate, 1.0), "wer": rate}


asr_wer_reward.validate_sources = lambda records: require_fields(
    records, "asr_wer", ("reference",)
)


def repeated_word_fraction(hypothesis, window=8):
    """Share of words that repeat a ``window``-gram seen earlier.

    Long-form decoding fails by looping, which WER alone reports only as
    insertions. This is a diagnostic column, not part of the reward.
    """
    tokens = words(hypothesis)
    if len(tokens) <= window:
        return 0.0
    seen, repeats = set(), 0
    for start in range(len(tokens) - window + 1):
        gram = tuple(tokens[start : start + window])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / (len(tokens) - window + 1)


def longform_wer_reward(source, generation):
    """Negative WER on a long reference, with early-stop and loop diagnostics.

    The reward is ``-min(WER, 1)``: a response can never score below a silent
    one, so inserting words is bounded the same way Bagpiper's TTS reward
    bounds its ASR. ``length_ratio`` and ``truncated`` separate the two
    long-form failure modes that a single WER number hides: stopping early
    (ratio well below 1, deletion-dominated WER) and looping (ratio above 1
    with a high ``repeat_rate``).

    ``min_reference_words`` in the record is the floor this task was prepared
    with; it is checked once per manifest so a short-utterance set cannot be
    trained as long-form by pointing a config at it.
    """
    require_text_mode(source, "longform_wer")
    reference, hypothesis = source["reference"], generation["text"]
    reference_words, hypothesis_words = words(reference), words(hypothesis)
    rate = word_error_rate(reference, hypothesis)
    reason = generation.get("finish_reason")
    if isinstance(reason, dict):
        reason = reason.get("type")
    return {
        "reward": -min(rate, 1.0),
        "wer": rate,
        "truncated": float(reason == "length"),
        "length_ratio": len(hypothesis_words) / max(1, len(reference_words)),
        "repeat_rate": repeated_word_fraction(hypothesis),
        "empty": float(not hypothesis_words),
    }


def _validate_longform(records):
    """Reject a long-form manifest on CPU, before any model or GPU starts.

    Two things make a long-form item trainable, and a reference-length floor
    is only one of them. The other is the model's audio window: a family whose
    frontend takes a fixed window rejects longer audio outright — OWSM's
    ``load_audio`` raises rather than truncating — so a manifest packed by
    utterance count can die on every single sample at the first read of
    training. ``espnet2.rl.tasks.longform.prepare_longform_manifest`` therefore
    records the window it was built against, and this checks the audio against
    it here, where the error costs seconds instead of a GPU allocation.

    A record that declares no window is still accepted, so manifests prepared
    before that field existed keep working; those only fail at the first read.
    """
    require_fields(records, "longform_wer", ("reference", "min_reference_words"))
    over = []
    for record in records:
        floor = int(record["min_reference_words"])
        actual = len(words(record["reference"]))
        if floor < 1:
            raise ValueError("min_reference_words must be positive")
        if actual < floor:
            raise ValueError(
                f"Record {record['id']!r} has {actual} reference words, below the "
                f"declared long-form floor {floor}. Prepare a long-form split "
                f"rather than relabeling short utterances."
            )
        budget = record.get("max_audio_seconds")
        if budget is None:
            continue
        if float(budget) <= 0:
            raise ValueError(
                f"Record {record['id']!r} declares a nonpositive audio window"
            )
        seconds = record.get("audio_seconds")
        if seconds is None and record.get("audio"):
            import soundfile

            seconds = soundfile.info(record["audio"]).frames / float(
                record.get("sample_rate", 16000)
            )
        if seconds is not None and float(seconds) > float(budget):
            over.append((record["id"], round(float(seconds), 2), float(budget)))
    if over:
        over.sort(key=lambda row: -row[1])
        raise ValueError(
            f"{len(over)} record(s) exceed the audio window their own manifest "
            f"declares, longest first: {over[:3]}. A fixed-window frontend "
            f"rejects these at the first read rather than truncating, so repack "
            f"by duration against the window."
        )


longform_wer_reward.validate_sources = _validate_longform
