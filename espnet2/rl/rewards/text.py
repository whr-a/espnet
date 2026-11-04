# Copyright 2025 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Text normalization, word error rate and answer extraction shared by rewards
and the families' held-out evaluators."""

import re
import unicodedata

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
DECORATION = " \t*_`\"'“”‘’.,;:!?"


def edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        updated = [i]
        for j, actual in enumerate(hypothesis, 1):
            updated.append(
                min(updated[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual))
            )
        row = updated
    return row[-1]


def words(text):
    """Words the WER compares: NFKC, casefolded, punctuation removed.

    LibriSpeech-style references are unpunctuated while models such as OWSM
    write punctuated text, so raw tokens would count punctuation as word
    errors and a WER reward would mostly teach the model to drop it.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w\s]", " ", normalized).split()


def word_error_rate(reference, hypothesis):
    expected = words(reference)
    return edit_distance(expected, words(hypothesis)) / max(1, len(expected))


def normalized(text):
    """The comparison form of a label: NFKC, casefolded, punctuation dropped."""
    return " ".join(words(text))


def answer_span(text):
    """The text a reward scores, and where it came from.

    The first answer tag, else the last nonempty line, else the whole
    response: ``(span, path)`` with ``path`` one of ``"tag"``, ``"line"`` and
    ``"text"``. A single explicit convention keeps every task reward here
    comparable; family-specific parsers stay in their own model packages.
    """
    match = ANSWER.search(text)
    if match:
        return match.group(1).strip(), "tag"
    lines = [line for line in text.splitlines() if line.strip(DECORATION)]
    if lines:
        return lines[-1].strip(DECORATION), "line"
    return text.strip(DECORATION), "text"
