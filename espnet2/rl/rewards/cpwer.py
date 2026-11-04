# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Multi-talker ASR: concatenated minimum-permutation WER, plain and shaped."""

import re
from itertools import permutations

from . import require_fields, require_text_mode
from .text import edit_distance, words

# A speaker tag the prompt asked for, e.g. "<spk0>", "<spk1>" or "<spk:alice>".
SPEAKER_TAG = re.compile(r"<spk(?::[^<>]{1,64}|\d{1,3})>")


def split_speaker_streams(text, tags):
    """Concatenate the response's segments per requested speaker tag.

    Text before the first tag is not attributed to any speaker and is
    returned separately, so a preamble never silently joins speaker one.
    Returns ``(streams, preamble, stated_tags)``.
    """
    positions = [
        (match.start(), match.end(), match.group(0))
        for match in SPEAKER_TAG.finditer(text)
    ]
    streams = {tag: [] for tag in tags}
    stated = []
    preamble = text[: positions[0][0]] if positions else text
    for index, (_, end, tag) in enumerate(positions):
        stop = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        stated.append(tag)
        if tag in streams:
            streams[tag].append(text[end:stop].strip())
    return (
        {
            tag: " ".join(part for part in parts if part)
            for tag, parts in streams.items()
        },
        preamble.strip(),
        stated,
    )


def concatenated_wer(references, hypotheses):
    """cpWER: the speaker permutation minimizing total word errors.

    Errors and reference words are summed across speakers before dividing,
    as CHiME's concatenated minimum-permutation WER does, so one speaker's
    short turn cannot dominate the score. ``hypotheses`` is padded with empty
    strings when the response named fewer speakers than the reference.
    """
    expected = [words(reference) for reference in references]
    total = max(1, sum(map(len, expected)))
    padded = list(hypotheses) + [""] * max(0, len(expected) - len(hypotheses))
    best = None
    for order in permutations(range(len(padded)), len(expected)):
        errors = sum(
            edit_distance(expected[index], words(padded[position]))
            for index, position in enumerate(order)
        )
        best = errors if best is None else min(best, errors)
    return (0.0 if best is None else best) / total


def multispeaker_cpwer_reward(source, generation):
    """Negative concatenated minimum-permutation WER for multi-talker ASR.

    ``references`` holds one transcript per speaker and ``speaker_tags`` the
    tags the prompt asked the model to emit. The response is split on those
    tags, each speaker's segments are concatenated in order, and cpWER takes
    the best assignment between response streams and references. The reward is
    ``-min(cpWER, 1)``. ``tag_format`` reports whether the response used each
    requested tag exactly once, as a diagnostic, not as part of the reward:
    cpWER already penalizes a missing speaker as whole-stream deletions.
    """
    require_text_mode(source, "multispeaker_cpwer")
    references, tags = list(source["references"]), list(source["speaker_tags"])
    streams, preamble, stated = split_speaker_streams(generation["text"], tags)
    hypotheses = [streams[tag] for tag in tags]
    rate = concatenated_wer(references, hypotheses)
    return {
        "reward": -min(rate, 1.0),
        "cpwer": rate,
        "tag_format": float(sorted(stated) == sorted(tags)),
        "tags_stated": float(len(stated)),
        "unassigned_words": float(len(words(preamble))),
        "empty_streams": float(sum(not words(part) for part in hypotheses)),
    }


#: cpWER searches every assignment of response streams to references, so its
#: cost is factorial in the speaker count and it runs once per candidate. Four
#: speakers is 24 assignments; eight would be 40,320 per candidate, which would
#: make the reward the slowest part of an update. A larger mixture needs a
#: different matcher, not a larger search.
MAX_CPWER_SPEAKERS = 4


def _validate_multispeaker(records):
    require_fields(records, "multispeaker_cpwer", ("references", "speaker_tags"))
    for record in records:
        references, tags = record["references"], record["speaker_tags"]
        if len(references) > MAX_CPWER_SPEAKERS:
            raise ValueError(
                f"Record {record['id']!r} has {len(references)} speakers; cpWER "
                f"searches every assignment, so this reward accepts at most "
                f"{MAX_CPWER_SPEAKERS}. A larger mixture needs a different matcher."
            )
        if len(references) != len(tags) or len(references) < 2:
            raise ValueError(
                f"Record {record['id']!r} needs at least two speakers and one tag "
                f"per reference; got {len(references)} references and {len(tags)} tags"
            )
        if len(set(tags)) != len(tags) or any(
            not SPEAKER_TAG.fullmatch(tag) for tag in tags
        ):
            raise ValueError(
                f"Record {record['id']!r} speaker_tags must be distinct and match "
                f"{SPEAKER_TAG.pattern}"
            )
        if any(not str(reference).strip() for reference in references):
            raise ValueError(f"Record {record['id']!r} has an empty speaker reference")


multispeaker_cpwer_reward.validate_sources = _validate_multispeaker


#: How much of the reward the two shaping terms may move. cpWER spans 1.0, so
#: 0.25 each keeps a real transcription gain worth more than any amount of
#: tagging or brevity: a policy cannot buy a better reward by labelling an
#: unchanged caption.
CPWER_SHAPING_WEIGHT = 0.25


def multispeaker_cpwer_shaped_reward(source, generation):
    """cpWER plus the two unsaturated terms a non-compliant policy needs.

    ``multispeaker_cpwer_reward`` returns ``-min(cpWER, 1)``, which is flat
    everywhere the response is worse than silence. That is where an audio
    *captioner* policy starts: it describes the recording instead of
    transcribing it, so every candidate is worse than silence, every reward is
    exactly -1, and the GRPO group advantage, which subtracts the group mean
    and divides by the group standard deviation, is identically zero: the
    policy gradient vanishes and no number of updates can move the policy.

    This reward keeps cpWER as the objective and adds two terms that still
    rank a group of uniformly bad answers: ``tag_format``, which is 1 only
    when the response stated each requested tag exactly once, and a brevity
    term against words that no speaker was credited with, measured against the
    reference length. Both are bounded by ``CPWER_SHAPING_WEIGHT`` so neither
    can outrank a genuine cpWER improvement. The components are reported
    separately so "learned to tag without transcribing better" is visible
    rather than hidden inside one scalar.
    """
    base = multispeaker_cpwer_reward(source, generation)
    expected = max(1, sum(len(words(reference)) for reference in source["references"]))
    verbosity = min(1.0, base["unassigned_words"] / expected)
    return {
        **base,
        "reward": (
            -min(base["cpwer"], 1.0)
            + CPWER_SHAPING_WEIGHT * base["tag_format"]
            - CPWER_SHAPING_WEIGHT * verbosity
        ),
        "cpwer_reward": -min(base["cpwer"], 1.0),
        "format_term": CPWER_SHAPING_WEIGHT * base["tag_format"],
        "verbosity_term": -CPWER_SHAPING_WEIGHT * verbosity,
        "unassigned_word_ratio": verbosity,
    }


multispeaker_cpwer_shaped_reward.validate_sources = _validate_multispeaker
