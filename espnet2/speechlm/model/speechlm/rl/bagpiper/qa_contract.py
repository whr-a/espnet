# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Reference-independent final answers for Bagpiper's historical native QA format.

The system and user prompts reproduce ``bagpiper/eval/prompt_common.py``.
Choice mapping intentionally omits that historical evaluator's substring,
Jaccard and whole-caption fallbacks. Parsing never receives a reference.
"""

import re
import string
import unicodedata

BAGPIPER_QA_SYSTEM_PROMPT = (
    "You are an expert audio understanding assistant. You will be given an "
    "audio clip, a question related to the audio's details, and four options. "
    "Your task is to choose the correct option. For every request, start by "
    "generating a detailed caption that describes the audio and includes some "
    "details related to the question. Then provide a thinking trace of getting "
    "the correct answer. Finally, print the correct option."
)
BAGPIPER_QA_ANSWER_MARKER = "the correct option is:"
_NATIVE_CUE = re.compile(re.escape(BAGPIPER_QA_ANSWER_MARKER), re.I)
_OPTION_HEADER = re.compile(
    r"^[ \t]*[*# ]*(?:the[ \t]+)?correct[ \t]+option[* \t]*:[* \t]*",
    re.I | re.M,
)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
_ANSWER_START = re.compile(r"<\s*/?\s*answer\b", re.I)
_ANSWER_OPEN = re.compile(r"<answer>", re.I)
_ANSWER_CLOSE = re.compile(r"</answer>", re.I)
_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)
_LABEL = re.compile(r"^(?:\(([A-Za-z])\)|([A-Za-z])[.):])\s*")
_BARE_LABEL = re.compile(r"^[A-Za-z]$")
_DECORATION = " \t\r\n*_`\"'“”‘’.,;:!?"


def native_qa_prompt(question, choices):
    """Render historical native QA user text without imposing an answer-tag style."""
    question = (question or "").strip()
    options = "; ".join(choice.strip() for choice in choices)
    return f"question: {question}\n Options: {options}"


def _clean(text):
    return unicodedata.normalize("NFKC", str(text)).strip(_DECORATION)


def _labelled(text):
    text = _clean(text)
    match = _LABEL.match(text)
    if match:
        return (match.group(1) or match.group(2)).upper(), _clean(text[match.end() :])
    return None, text


def normalize_choice(text):
    """Normalize decoration, Unicode, case, underscores and spacing only.

    Labels are handled separately. In particular, ``a minor`` and ``b flat``
    retain their first word, and sharps and hyphens remain significant.
    """
    return " ".join(_clean(text).replace("_", " ").casefold().split())


def _catalog(choices):
    """Keep literal choices and map identical normalized bodies to their first index."""
    if isinstance(choices, (str, bytes)):
        return None
    try:
        choices = tuple(choices)
    except TypeError:
        return None
    if not choices or len(choices) > len(string.ascii_uppercase):
        return None
    if any(not isinstance(choice, str) for choice in choices):
        return None
    parts = [_labelled(choice) for choice in choices]
    explicit_labels = False
    if all(label and body for label, body in parts):
        supplied = [label for label, _ in parts]
        positions = sorted(ord(label) for label in supplied)
        coherent = len(set(supplied)) == len(choices) and positions == list(
            range(positions[0], positions[0] + len(choices))
        )
        # A whole catalogue must establish labels. An isolated name initial
        # such as F. Scott Fitzgerald must retain its literal option text.
        # Parentheses explicitly identify labels, including invalid repeats;
        # bare labels must form a coherent A-D, E-H, or other contiguous set.
        parenthesized = all(_clean(choice).startswith("(") for choice in choices)
        explicit_labels = parenthesized or coherent
    texts, labels = [], {}
    for index, choice in enumerate(choices):
        label, body = parts[index] if explicit_labels else (None, choice)
        normalized = normalize_choice(body)
        label = label or string.ascii_uppercase[index]
        if not normalized or label in labels:
            return None
        canonical = texts.index(normalized) if normalized in texts else index
        texts.append(normalized)
        labels[label] = canonical
    return choices, texts, labels, [normalize_choice(choice) for choice in choices]


def match_choice(answer, choices):
    """Return ``(canonical_index, method)`` for an exact option or option label.

    An explicit label and its option text must agree. Bare one-letter answers
    refer to the supplied order unless choices themselves carry labels.
    Identical normalized option bodies share their first index, so duplicated
    distractors never invalidate an otherwise well-formed question. Repeated
    explicit labels remain invalid. There is no substring, fuzzy,
    reference-dependent or whole-completion search.
    """
    catalog = _catalog(choices)
    if catalog is None:
        return None, "invalid_choices"
    _, texts, labels, literals = catalog
    literal = normalize_choice(answer)
    if literal in literals:
        return texts.index(texts[literals.index(literal)]), "exact"
    label, body = _labelled(answer)
    normalized = normalize_choice(body)
    exact = texts.index(normalized) if normalized in texts else None
    if label:
        labelled = labels.get(label)
        if not body:
            return labelled, "label" if labelled is not None else "unknown_label"
        if exact is not None and exact == labelled:
            return exact, "label_text"
        return None, "contradictory_label" if exact is not None else "unknown_answer"
    if _BARE_LABEL.fullmatch(body):
        labelled = labels.get(body.upper())
        if exact is not None and labelled is not None and exact != labelled:
            return None, "ambiguous_answer"
        if exact is not None:
            return exact, "exact"
        return labelled, "label" if labelled is not None else "unknown_label"
    return exact, "exact" if exact is not None else "unknown_answer"


def _line(tail):
    """The first nonempty decorated line and everything following that line."""
    offset = 0
    for line in tail.splitlines(keepends=True):
        offset += len(line)
        if _clean(line):
            return _clean(line), tail[offset:]
    return "", ""


def parse_native_qa_answer(text, choices):
    """Parse a terminal native cue, option header, answer tag or exact option.

    Returns a JSON-safe dict with ``path``, ``status``, ``raw_answer``,
    ``predicted_choice``, ``predicted_index``, ``match_method``, ``format`` and
    ``format_reason``. A native marker, line-anchored ``Correct Option:``
    header or final answer block earns the structural flag independently of
    answer correctness. Exact plain or post-think options can be correct without
    the structural flag. No renderer
    control tokens are discarded here; callers must decode text consistently.

    A completed think prefix may omit its opening marker after a renderer
    prefill. An unterminated, repeated or reversed think structure is rejected.
    Answer tags must be a single complete, nonnested final-region block.
    Multiple terminal declarations must resolve to the same option; explicit
    contradictory or malformed declarations are never rescued by a fallback.
    """
    result = {
        "path": "none",
        "status": "empty",
        "raw_answer": "",
        "predicted_choice": None,
        "predicted_index": None,
        "match_method": "none",
        "format": False,
        "format_reason": "missing_answer",
    }

    def reject(status, path="none"):
        return {**result, "path": path, "status": status, "format_reason": status}

    catalog = _catalog(choices)
    if catalog is None:
        return reject("invalid_choices")
    choices = catalog[0]
    if not isinstance(text, str) or not text.strip():
        return result
    openings = list(_THINK_OPEN.finditer(text))
    closings = list(_THINK_CLOSE.finditer(text))
    if (
        len(openings) > 1
        or len(closings) > 1
        or (openings and not closings)
        or (openings and openings[0].start() > closings[0].start())
    ):
        return reject("malformed_reasoning")
    boundary = closings[0].end() if closings else 0
    region = text[boundary:]
    starts = list(_ANSWER_START.finditer(text))
    opens = list(_ANSWER_OPEN.finditer(text))
    closes = list(_ANSWER_CLOSE.finditer(text))
    blocks = list(_ANSWER.finditer(text))
    if starts and (
        len(starts) != 2 or len(opens) != 1 or len(closes) != 1 or len(blocks) != 1
    ):
        return reject("malformed_answer_tags", "answer_tag")
    block = blocks[0] if blocks else None
    if block and block.start() < boundary:
        return reject("answer_inside_reasoning", "answer_tag")
    candidates = []
    if block:
        raw = block.group(1).strip()
        if not _clean(raw):
            return reject("empty_answer_tag", "answer_tag")
        candidates.append((raw, "answer_tag", text[block.end() :]))
    for pattern, path in (
        (_NATIVE_CUE, "native_cue"),
        (_OPTION_HEADER, "option_header"),
    ):
        for cue in pattern.finditer(text, boundary):
            if block and block.start() <= cue.start() < block.end():
                continue
            raw, suffix = _line(text[cue.end() :])
            # A declaration may introduce the single final answer block directly.
            if (
                block
                and cue.end() <= block.start()
                and not text[cue.end() : block.start()].strip(_DECORATION)
            ):
                continue
            if not raw:
                return reject(f"empty_{path}", path)
            candidates.append((raw, path, suffix))
    if not candidates:
        candidates = [(region.strip(), "post_think" if boundary else "plain", "")]
    mapped = [
        (match_choice(raw, choices), raw, path, tail) for raw, path, tail in candidates
    ]
    indices = {match[0] for match, _, _, _ in mapped if match[0] is not None}
    if len(indices) > 1:
        return reject("conflicting_answers", mapped[-1][2])
    if len(mapped) > 1 and any(match[0] is None for match, _, _, _ in mapped):
        return reject("ambiguous_answers", mapped[-1][2])
    (index, method), raw, path, tail = mapped[-1]
    # An additional exact option on a later line contradicts the declaration.
    for trailing in tail.splitlines():
        trailing_index, _ = match_choice(trailing, choices)
        if index is not None and trailing_index is not None and trailing_index != index:
            return reject("conflicting_answers", path)
    structured = path in ("answer_tag", "native_cue", "option_header")
    terminal = not tail.strip(_DECORATION)
    valid_format = structured and terminal and len(mapped) == 1
    if index is None and method in ("contradictory_label", "ambiguous_answer"):
        valid_format = False
    return {
        **result,
        "path": path,
        "status": "ok" if index is not None else "unmapped_answer",
        "raw_answer": raw,
        "predicted_choice": choices[index] if index is not None else None,
        "predicted_index": index,
        "match_method": method,
        "format": valid_format,
        "format_reason": (
            "terminal_answer" if valid_format else "unstructured_or_nonterminal"
        ),
    }
