# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Closed-set speaker identification and same/different speaker verification."""

from . import require_fields, require_text_mode
from .text import ANSWER, answer_span, normalized


def match_speaker(prediction, choices):
    """The index of the single closed-set speaker label the prediction states.

    A label matches when its normalized words appear as a contiguous run in
    the normalized prediction, so "the speaker is 1263" selects "1263".
    Returns ``(index, "unique" | "ambiguous" | "none")``; several distinct
    matching labels earn nothing, because the response named no one speaker.
    """
    predicted = normalized(prediction).split()
    hits = []
    for index, choice in enumerate(choices):
        label = normalized(str(choice)).split()
        if not label:
            continue
        if any(
            predicted[start : start + len(label)] == label
            for start in range(len(predicted) - len(label) + 1)
        ):
            hits.append(index)
    if not hits:
        return None, "none"
    if len({normalized(str(choices[i])) for i in hits}) > 1:
        return None, "ambiguous"
    return hits[0], "unique"


def speaker_id_reward(source, generation):
    """Closed-set speaker identification accuracy plus a terminal format bonus.

    ``speaker`` is the true label and ``speaker_choices`` the candidate labels
    the prompt offered. Accuracy is 1 when the response names exactly that
    label and no other candidate; the 0.1 format bonus requires exactly one
    answer block, so reasoning before it is allowed. Scale and weights match
    the native QA protocol ``qa_bagpiper_v2_reward`` already in this tree.
    """
    require_text_mode(source, "speaker_id")
    choices = list(source["speaker_choices"])
    truth, truth_path = match_speaker(source["speaker"], choices)
    if truth is None:
        raise ValueError(
            f"Record {source['id']!r} speaker label is not exactly one of its "
            f"speaker_choices ({truth_path})"
        )
    span, path = answer_span(generation["text"])
    predicted, status = match_speaker(span, choices)
    accuracy = float(predicted is not None and predicted == truth)
    answer_format = float(len(ANSWER.findall(generation["text"])) == 1)
    return {
        "reward": accuracy + 0.1 * answer_format,
        "acc": accuracy,
        "format": answer_format,
        "parsed": float(status == "unique"),
        "ambiguous": float(status == "ambiguous"),
        **{f"path_{name}": float(path == name) for name in ("tag", "line", "text")},
    }


def _declared_choice_key(record):
    """What draw a speaker-id record says it was built with, or None.

    ``choice_key`` is what espnet2.rl.tasks.speaker stamps. ``choice_draw`` with a
    ``keyed_on`` entry is what a preparer written before that field existed
    records, and "utterance_id" there means the same thing as "utterance" here,
    so both normalize to the same answer.
    """
    key = record.get("choice_key")
    if key is None:
        draw = record.get("choice_draw")
        if isinstance(draw, dict):
            key = draw.get("keyed_on")
    if key in ("utterance_id", "utterance"):
        return "utterance"
    return key


def _validate_speaker_id(records):
    """Reject a speaker-id manifest whose candidate list leaks the answer.

    The contract used to check only that the fields were present and that the
    label was one of the choices. It missed the failure that actually voids a
    result: if the candidate list is a per-speaker constant, then in a
    closed-set split that keeps one speaker inventory and divides it by
    utterance, the list becomes a function of the label and a model can learn
    "this list means that label" without listening. Measured on LibriSpeech
    dev-clean, 2674 records collapsed to 40 distinct lists with an injective
    list-to-label map.

    No reward function can catch this, because a reward sees the model's answer
    and not how the data was built. So it is checked here, where
    ``read_manifest`` runs it before any model, GPU or Ray actor starts.

    A manifest that explicitly declares a per-speaker draw asked for it
    knowingly and is allowed through; one that does not declare it and still
    shows per-speaker-constant lists is the accidental case. Two spellings of
    that declaration are accepted, because preparers written before this check
    existed already use the second one: a top-level ``choice_key``, or a
    ``choice_draw`` dict with a ``keyed_on`` entry.
    """
    require_fields(records, "speaker_id", ("speaker", "speaker_choices"))
    declared = {_declared_choice_key(record) for record in records}
    if declared == {"speaker"}:
        return
    lists = {}
    for record in records:
        lists.setdefault(record["speaker"], set()).add(
            tuple(record["speaker_choices"])
        )
    constant = [name for name, seen in lists.items() if len(seen) == 1]
    repeated = [name for name in constant if sum(
        1 for record in records if record["speaker"] == name) > 1]
    # One speaker cannot be leaked: with a single label there is nothing for
    # the list to distinguish.
    if len(lists) > 1 and repeated and len(repeated) == len(lists):
        raise ValueError(
            f"Every utterance of each of the {len(repeated)} speakers is offered "
            f"the same candidate list, so the list identifies the answer and a "
            f"model can score without listening. Regenerate with "
            f"espnet2.rl.tasks.speaker's default choice_key=\"utterance\", or, if a "
            f"per-speaker draw is intended, say so by stamping "
            f"choice_key=\"speaker\" (or choice_draw={{\"keyed_on\": \"speaker\"}}) "
            f"into the records."
        )


speaker_id_reward.validate_sources = _validate_speaker_id


#: The two verdicts a verification prompt offers.
SPEAKER_VERDICTS = ("same", "different")


def speaker_verification_reward(source, generation):
    """Binary same/different-speaker accuracy plus a terminal format bonus.

    One item is one binary decision over two segments of one audio.
    ``verdict`` is the true label and ``verdict_choices`` the two labels the
    prompt offered. The prediction is read with exactly the matching rule
    :func:`speaker_id_reward` uses -- :func:`match_speaker` over the offered
    labels, via the first ``<answer>`` block else the last nonempty line -- and
    kept on the same scale, accuracy plus a 0.1 format bonus, so a
    verification run and an N-way identification run are directly comparable.
    Naming both labels earns nothing: a response that says "not the same,
    these are different speakers" named no one verdict.

    What a binary task adds is per-class bookkeeping. A reward that logs only
    mean accuracy cannot tell a policy that discriminates from a policy that
    answers one label every time, because on a label-balanced set both score
    0.5. ``true_same``/``hit_same`` and ``true_different``/``hit_different``
    are logged as means, so

        TPR = mean(hit_same) / mean(true_same)
        TNR = mean(hit_different) / mean(true_different)
        balanced accuracy = (TPR + TNR) / 2

    are recoverable per training step, and balanced accuracy is exactly 0.5
    for every constant policy.
    """
    require_text_mode(source, "speaker_verification")
    choices = [str(choice) for choice in source["verdict_choices"]]
    truth, truth_path = match_speaker(source["verdict"], choices)
    if truth is None:
        raise ValueError(
            f"Record {source['id']!r} verdict is not exactly one of its "
            f"verdict_choices ({truth_path})"
        )
    span, path = answer_span(generation["text"])
    predicted, status = match_speaker(span, choices)
    accuracy = float(predicted is not None and predicted == truth)
    answer_format = float(len(ANSWER.findall(generation["text"])) == 1)
    positive = float(choices[truth] == "same")
    return {
        "reward": accuracy + 0.1 * answer_format,
        "acc": accuracy,
        "format": answer_format,
        "parsed": float(status == "unique"),
        "ambiguous": float(status == "ambiguous"),
        # Per-class counters. Their batch means reconstruct TPR, TNR and
        # balanced accuracy, which no constant policy can push off 0.5.
        "true_same": positive,
        "true_different": 1.0 - positive,
        "hit_same": accuracy * positive,
        "hit_different": accuracy * (1.0 - positive),
        "said_same": float(
            predicted is not None
            and status == "unique"
            and choices[predicted] == "same"
        ),
        **{f"path_{name}": float(path == name) for name in ("tag", "line", "text")},
    }


def _validate_speaker_verification(records):
    """Reject a manifest that cannot supervise verification, before any GPU."""
    require_fields(records, "speaker_verification", ("verdict", "verdict_choices"))
    for record in records:
        choices = [str(choice) for choice in record["verdict_choices"]]
        if sorted(choices) != sorted(SPEAKER_VERDICTS):
            raise ValueError(
                f"Record {record.get('id')!r} offers {choices}, not "
                f"{list(SPEAKER_VERDICTS)}"
            )
        if str(record["verdict"]) not in choices:
            raise ValueError(
                f"Record {record.get('id')!r} verdict {record['verdict']!r} is not "
                f"one of its own choices"
            )


speaker_verification_reward.validate_sources = _validate_speaker_verification
