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
        lists.setdefault(record["speaker"], set()).add(tuple(record["speaker_choices"]))
    constant = [name for name, seen in lists.items() if len(seen) == 1]
    repeated = [
        name
        for name in constant
        if sum(1 for record in records if record["speaker"] == name) > 1
    ]
    # One speaker cannot be leaked: with a single label there is nothing for
    # the list to distinguish.
    if len(lists) > 1 and repeated and len(repeated) == len(lists):
        raise ValueError(
            f"Every utterance of each of the {len(repeated)} speakers is offered "
            f"the same candidate list, so the list identifies the answer and a "
            f"model can score without listening. Regenerate with "
            f'espnet2.rl.tasks.speaker\'s default choice_key="utterance", or, if a '
            f"per-speaker draw is intended, say so by stamping "
            f'choice_key="speaker" (or choice_draw={{"keyed_on": "speaker"}}) '
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


# How far a conditioning variable may shift P(same) before it counts as a
# channel through which the label can be predicted without comparing voices.
LABEL_LEAK_TOLERANCE = 0.10
LABEL_BALANCE_TOLERANCE = 0.05
# Speakers seen fewer times than this carry too much sampling noise to judge.
MIN_SPEAKER_APPEARANCES = 3


def _leak_free_corpus(records):
    """Reject a verification corpus whose CONSTRUCTION gives away the label.

    Record-local validity is not enough for this task. A pair corpus can have
    every field right and still be solvable without listening, and the closed-set
    round showed that happens in practice: there the candidate list was drawn per
    speaker, so the list identified the answer and a model could score 100% from
    the prompt alone. The analogous channels here are structural, so they are
    checked over the whole corpus rather than per record.

    Each check below corresponds to one closure in the pair preparer and to
    one measured line in its ``.audit.json``.
    They are asserted here so the property survives a refactor of the preparer:
    a reader of this function can see what the corpus is required to guarantee
    without reading the generator.

    1. A same-speaker pair never takes both segments from one recording. If it
       did, "did the channel change mid-file" would separate the classes.
    2. Every speaker that appears in a different-speaker pair also appears in a
       same-speaker pair. Otherwise recognizing that speaker is a free point,
       because it can only ever occur on one side.
    3. Every speaker's own P(same | this speaker appears) is the SAME, so which
       speakers are present carries no differential signal. Note it is
       UNIFORMITY that matters, not agreement with the label share: a
       same-speaker pair spends one speaker slot and a different-speaker pair
       spends two, so with equal item counts every speaker sits at 1/3 while
       the labels are at 1/2. Comparing each speaker against the label share
       would reject a correctly built corpus.
    4. The labels are near balanced, so "always answer the majority" cannot
       beat chance.
    5. No prompt string and no total duration predicts the label. These are the
       two non-acoustic fields the model can read; the preparer holds them
       constant, but the requirement is the weaker and more general one that
       they carry no label information.
    """
    positives = [r for r in records if str(r.get("verdict")) == "same"]
    negatives = [r for r in records if str(r.get("verdict")) == "different"]
    if not positives or not negatives:
        raise ValueError(
            "A verification corpus needs both same-speaker and different-speaker "
            f"pairs; saw {len(positives)} same and {len(negatives)} different"
        )

    def chapters(record):
        return [s.get("chapter") for s in record.get("segments", [])]

    def speakers(record):
        return {s.get("speaker") for s in record.get("segments", [])}

    # 1. same-speaker pairs must be cross-recording
    within = [r for r in positives if len(set(chapters(r))) < 2]
    if within:
        raise ValueError(
            f"{len(within)} same-speaker pairs take both segments from one "
            f"recording (e.g. {within[0].get('id')!r}). Draw them across "
            "different recordings, or the channel alone separates the classes."
        )

    # 2. both arms must draw from one speaker pool
    positive_speakers = {s for r in positives for s in speakers(r)}
    negative_only = {s for r in negatives for s in speakers(r)} - positive_speakers
    if negative_only:
        raise ValueError(
            f"{len(negative_only)} speakers appear only in different-speaker "
            f"pairs (e.g. {sorted(negative_only)[:3]}). Recognizing them is a "
            "free point; restrict both arms to one speaker pool."
        )

    # 3. every speaker must carry the SAME label prior (uniformity, not 1/2)
    share = len(positives) / len(records)
    per_speaker = {}
    for record in records:
        for speaker in speakers(record):
            same, total = per_speaker.get(speaker, (0, 0))
            per_speaker[speaker] = (
                same + (str(record.get("verdict")) == "same"),
                total + 1,
            )
    priors = {
        speaker: same / total
        for speaker, (same, total) in per_speaker.items()
        if total >= MIN_SPEAKER_APPEARANCES
    }
    if priors and max(priors.values()) - min(priors.values()) > LABEL_LEAK_TOLERANCE:
        low = min(priors, key=priors.get)
        high = max(priors, key=priors.get)
        raise ValueError(
            f"P(same | speaker appears) ranges from {priors[low]:.3f} ({low!r}) "
            f"to {priors[high]:.3f} ({high!r}), a spread above "
            f"{LABEL_LEAK_TOLERANCE}. Recognizing the high-prior speakers then "
            "predicts the label; give every speaker the same ratio of "
            "same-speaker to different-speaker appearances."
        )

    # 4. labels near balanced
    if abs(share - 0.5) > LABEL_BALANCE_TOLERANCE:
        raise ValueError(
            f"Labels are {share:.3f} same, further than "
            f"{LABEL_BALANCE_TOLERANCE} from balanced; a constant answer would "
            "then beat chance."
        )

    # 5. no readable non-acoustic field may predict the label
    for field in ("prompt", "total_seconds"):
        groups = {}
        for record in records:
            if field not in record:
                continue
            same, total = groups.get(record[field], (0, 0))
            groups[record[field]] = (
                same + (str(record.get("verdict")) == "same"),
                total + 1,
            )
        for value, (same, total) in groups.items():
            if total >= 10 and abs(same / total - share) > LABEL_LEAK_TOLERANCE:
                raise ValueError(
                    f"{field} value {str(value)[:60]!r} carries label "
                    f"information: P(same) is {same / total:.3f} over {total} "
                    f"records against the corpus share {share:.3f}. The model "
                    "reads this field without listening."
                )


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
        # The corpus gates below read speaker and chapter per segment; a record
        # without them cannot be audited, so it is rejected rather than skipped.
        segments = record.get("segments")
        if not isinstance(segments, list) or len(segments) != 2:
            raise ValueError(
                f"Record {record.get('id')!r} must carry exactly two segments; "
                "regenerate the manifest with a pair preparer that records each "
                "segment's speaker and chapter."
            )
        for segment in segments:
            if not segment.get("speaker") or not segment.get("chapter"):
                raise ValueError(
                    f"Record {record.get('id')!r} has a segment without a speaker "
                    "or chapter, so its construction cannot be audited."
                )
    _leak_free_corpus(records)


speaker_verification_reward.validate_sources = _validate_speaker_verification
