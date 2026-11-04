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
"""Model-agnostic task rewards for speech RL beyond single-speaker ASR WER.

Every reward here scores the *text* a rollout produced, so it works for any
family whose engine emits text: OWSM, OpusLM, OpusLM Dialogue and Bagpiper.
Each returns its value as ``"reward"`` next to the components it was made of,
which :func:`espnet2.rl.rewards.outcome_rewards` logs as ``reward/<name>``.

Each reward also exposes ``validate_sources(records)``, which
:func:`espnet2.rl.data.read_manifest` calls before any model, GPU or Ray actor
starts. A manifest that does not carry a reward's required supervision is
rejected there rather than silently scored as a failure during training.

Rewards never read a manifest field they did not declare, and none of them
loads a model, so they all run on CPU.
"""

import re
from itertools import permutations

from .rewards import edit_distance, word_error_rate, words
from .utils import resolve_callable

# Rollout modes whose sampled actions are text. Audio-output modes must use a
# task reward that decodes codec actions, such as Bagpiper's ``tts_asr_wer``.
TEXT_MODES = ("", "audio_text", "text_only", "text_text", "text_dialogue")

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
# A speaker tag the prompt asked for, e.g. "<spk0>", "<spk1>" or "<spk:alice>".
SPEAKER_TAG = re.compile(r"<spk(?::[^<>]{1,64}|\d{1,3})>")
DECORATION = " \t*_`\"'“”‘’.,;:!?"


def require_text_mode(source, task):
    mode = source.get("mode", "")
    if mode not in TEXT_MODES:
        raise ValueError(
            f"{task} scores generated text; rollout mode {mode!r} emits other actions"
        )


def require_fields(records, task, fields):
    """Reject a manifest that cannot supervise ``task`` before training starts."""
    for record in records:
        missing = [name for name in fields if record.get(name) in (None, "", [])]
        if missing:
            raise ValueError(
                f"{task} requires {', '.join(fields)} in every manifest record; "
                f"record {record.get('id')!r} is missing {', '.join(missing)}. "
                f"Regenerate the manifest with the matching espnet2.rl.task_data "
                f"preparer."
            )
        declared = record.get("reward_task")
        if declared is not None and declared != task:
            raise ValueError(
                f"Record {record.get('id')!r} declares reward_task {declared!r} but "
                f"the configured reward is {task}. Point reward_function at the "
                f"declared task, or regenerate the manifest."
            )


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


# ----------------------------------------------------------------- speaker id


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


speaker_id_reward.validate_sources = lambda records: require_fields(
    records, "speaker_id", ("speaker", "speaker_choices")
)


# ---------------------------------------------------------------- long-form


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
    require_fields(records, "longform_wer", ("reference", "min_reference_words"))
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


longform_wer_reward.validate_sources = _validate_longform


# ------------------------------------------------------- multi-speaker cpWER


def split_speaker_streams(text, tags):
    """Concatenate the response's segments per requested speaker tag.

    Text before the first tag is not attributed to any speaker and is
    returned separately, so a preamble never silently joins speaker one.
    Returns ``(streams, preamble, stated_tags)``.
    """
    positions = [
        (match.start(), match.end(), match.group(0)) for match in SPEAKER_TAG.finditer(text)
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
        {tag: " ".join(part for part in parts if part) for tag, parts in streams.items()},
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
        if len(set(tags)) != len(tags) or any(not SPEAKER_TAG.fullmatch(tag) for tag in tags):
            raise ValueError(
                f"Record {record['id']!r} speaker_tags must be distinct and match "
                f"{SPEAKER_TAG.pattern}"
            )
        if any(not str(reference).strip() for reference in references):
            raise ValueError(f"Record {record['id']!r} has an empty speaker reference")


multispeaker_cpwer_reward.validate_sources = _validate_multispeaker


# ----------------------------------------------------------- mixed-task mix


def asr_wer_reward(source, generation):
    """Negative WER of the sampled text, as the single-task ASR reward scores it."""
    require_text_mode(source, "asr_wer")
    rate = word_error_rate(source["reference"], generation["text"])
    return {"reward": -min(rate, 1.0), "wer": rate}


asr_wer_reward.validate_sources = lambda records: require_fields(
    records, "asr_wer", ("reference",)
)

#: Task names a mixed manifest may declare without a module path. Each value
#: is scored by the reward of the same name in this module.
BUILTIN_TASKS = {
    "asr_wer": asr_wer_reward,
    "longform_wer": longform_wer_reward,
    "speaker_id": speaker_id_reward,
    "multispeaker_cpwer": multispeaker_cpwer_reward,
}


def resolve_task(name):
    """The reward for a declared task: a built-in name or ``module:callable``."""
    if name in BUILTIN_TASKS:
        return BUILTIN_TASKS[name]
    if ":" not in str(name):
        raise ValueError(
            f"Unknown reward_task {name!r}: use one of "
            f"{', '.join(sorted(BUILTIN_TASKS))} or an explicit module:callable"
        )
    return resolve_callable(name)


def score_task(reward, sources, generations):
    """Score one task's candidates, honoring a batched component reward."""
    if getattr(reward, "batched", False):
        rows = list(reward(sources, generations))
    else:
        rows = [
            reward(source, generation)
            for source, generation in zip(sources, generations, strict=True)
        ]
    if len(rows) != len(sources):
        raise ValueError("A component reward must score every candidate")
    return [row if isinstance(row, dict) else {"reward": row} for row in rows]


def mixed_task_reward(sources, generations):
    """One RL run over several tasks: each record is scored by its own reward.

    Every record declares ``reward_task``; its candidates are scored by that
    task's reward and keep its value as ``reward``. Mixing scales is safe
    because GRPO normalizes within a prompt group: ``espnet2/rl/trainer.py``
    passes the per-prompt ``uid`` to ``compute_grpo_outcome_advantage``
    (``espnet2/rl/miles_library.py``), which groups candidates by that id
    before Miles normalizes. Two tasks therefore never share a baseline.

    ``outcome_rewards`` requires one fixed set of logged components, so this
    reward reports, for every task present in the manifest, ``share_<task>``
    (1 on that task's candidates) and ``sum_<task>`` (its reward there, 0
    elsewhere). The batch mean reward of one task is
    ``reward/sum_<task> / reward/share_<task>``; a task's own components stay
    in its single-task runs.
    """
    tasks = sorted({str(source["reward_task"]) for source in sources})
    rows = [None] * len(sources)
    for task in tasks:
        indices = [i for i, source in enumerate(sources) if str(source["reward_task"]) == task]
        scored = score_task(
            resolve_task(task),
            [sources[i] for i in indices],
            [generations[i] for i in indices],
        )
        for index, row in zip(indices, scored, strict=True):
            value = float(row["reward"])
            rows[index] = {
                "reward": value,
                **{f"share_{name}": float(name == task) for name in tasks},
                **{f"sum_{name}": value if name == task else 0.0 for name in tasks},
            }
    return rows


mixed_task_reward.batched = True


def _validate_mixed(records):
    missing = [r.get("id") for r in records if not r.get("reward_task")]
    if missing:
        raise ValueError(
            f"mixed_task_reward needs reward_task in every record; missing for "
            f"{', '.join(map(repr, missing[:5]))}"
        )
    grouped = {}
    for record in records:
        grouped.setdefault(str(record["reward_task"]), []).append(record)
    for task, subset in grouped.items():
        validator = getattr(resolve_task(task), "validate_sources", None)
        if validator is not None:
            validator(subset)


mixed_task_reward.validate_sources = _validate_mixed
