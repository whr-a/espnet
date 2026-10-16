# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Historical Bagpiper choice mapping for comparable native QA evaluation.

Vendored from ``bagpiper/eval/prompt_common.py``; its source SHA256 is
``c146ecf2b2a09b5eaf8b49aec90cf62341c6aaef74889f5d49adc13a1efa7eb6``.
The parsing and mapping functions are unchanged. This preserves its historical
substring, Jaccard and unique-whole-response fallbacks for named evaluation
only. The stricter training contract lives in ``qa_contract.py``. Neither
extractor receives a reference answer.
"""

import re
import unicodedata

ANSWER_MARKER = "the correct option is:"

_WORD = re.compile(r"\b\w+\b")
# Strip only a genuine option *label*: "(A)", "(a)", "A)", "A.", "A:" — but NOT a
# bare "A <word>", which would wreck music answers like "a minor" / "b flat".
_LETTER_PREFIX = re.compile(r"^\s*(?:\(([A-Ha-h])\)|([A-Ha-h])[\.\):])\s*")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    # drop a leading option letter like "(a) " / "a. " / "b) "
    s = _LETTER_PREFIX.sub("", s)
    s = s.strip().strip('"').strip("'")
    # audio-event labels use underscores ("dog_barking"); treat as spaces so a
    # model output of "dog barking" still maps to the canonical choice.
    s = s.replace("_", " ")
    # strip trailing punctuation
    s = re.sub(r"[\s\.\,\;\:\!\?]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokens(s: str):
    # split on underscore too (official scorer keeps it, but for fuzzy mapping
    # we want "dog_barking" ~ "dog barking")
    return set(re.findall(r"[a-z0-9]+", str(s).lower().replace("_", " ")))


def extract_raw_answer(text: str):
    """Return the model's stated option text (after the marker), or None."""
    if not text:
        return None
    # Prefer the LAST occurrence of the marker.
    idxs = [m.end() for m in re.finditer(re.escape(ANSWER_MARKER), text, re.I)]
    if idxs:
        tail = text[idxs[-1] :].strip()
        # take the first non-empty line after the marker
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        tail = lines[0].strip() if lines else ""
        tail = tail.strip().strip('"').strip("'")
        tail = re.sub(r"[\s\.]+$", "", tail)
        if tail:
            return tail
    return None


def map_to_choice(raw_answer, choices, full_text=None):
    """Map the model output to exactly one of `choices`.

    Returns (matched_choice or None, method). Strategy:
      1. exact normalized match of raw_answer to a choice
      2. one choice's normalized form is a substring of raw_answer (longest)
      3. token-Jaccard argmax (>=0.5) over raw_answer
      4. fall back to scanning full_text for a unique choice mention
    """
    norm_choices = [_norm(c) for c in choices]

    def _try(text):
        if not text:
            return None, None
        nt = _norm(text)
        # 1 exact
        for c, nc in zip(choices, norm_choices):
            if nt == nc:
                return c, "exact"
        # 2 substring (prefer the longest choice contained in the answer)
        subs = [(c, nc) for c, nc in zip(choices, norm_choices) if nc and nc in nt]
        if subs:
            subs.sort(key=lambda x: len(x[1]), reverse=True)
            return subs[0][0], "substring"
        # 3 token Jaccard
        at = _tokens(nt)
        best, bestj = None, 0.0
        for c, nc in zip(choices, norm_choices):
            ct = _tokens(nc)
            if not ct:
                continue
            j = len(at & ct) / max(1, len(at | ct))
            if j > bestj:
                best, bestj = c, j
        if best is not None and bestj >= 0.5:
            return best, f"jaccard{bestj:.2f}"
        return None, None

    m, meth = _try(raw_answer)
    if m is not None:
        return m, meth

    # 4 scan full text: choice mentioned exactly once near the end
    if full_text:
        nt = _norm(full_text)
        hits = [(c, nc) for c, nc in zip(choices, norm_choices) if nc and nc in nt]
        if len(hits) == 1:
            return hits[0][0], "fulltext-unique"
    return None, None


def parse_prediction(text, choices):
    """Convenience: returns (matched_choice|None, raw_answer|None, method)."""
    raw = extract_raw_answer(text)
    m, meth = map_to_choice(raw, choices, full_text=text)
    return m, raw, meth
