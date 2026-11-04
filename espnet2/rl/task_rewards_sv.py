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
"""Speaker verification: one binary decision over two segments of one audio.

This lives beside :mod:`espnet2.rl.task_rewards` rather than inside it so that
a branch editing the rewards already there cannot conflict with this one.

The reward keeps the closed-set speaker-id scale -- accuracy plus a 0.1
terminal format bonus -- so a verification run and a 10-way identification run
are directly comparable. What it adds is the per-class bookkeeping a binary
task needs: a reward that only logs mean accuracy cannot tell a policy that
discriminates from a policy that answers one label every time, because on a
label-balanced set both score 0.5. ``true_same``/``hit_same`` and
``true_different``/``hit_different`` are logged as means, so

    TPR = mean(hit_same) / mean(true_same)
    TNR = mean(hit_different) / mean(true_different)
    balanced accuracy = (TPR + TNR) / 2

are recoverable per training step, and balanced accuracy is exactly 0.5 for
every constant policy.
"""

from .task_rewards import ANSWER, answer_span, match_speaker, require_text_mode

LABELS = ("same", "different")


def speaker_verification_reward(source, generation):
    """Binary same/different-speaker accuracy plus a terminal format bonus.

    ``verdict`` is the true label and ``verdict_choices`` the two labels the
    prompt offered. The prediction is read with exactly the matching rule the
    closed-set speaker-id reward uses (:func:`match_speaker` over the offered
    labels, via the first ``<answer>`` block else the last nonempty line), so
    the two tasks' ``parsed``/``format``/``path_*`` columns mean the same
    thing. Naming both labels earns nothing: a response that says "not the
    same, these are different speakers" named no one verdict.
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
            predicted is not None and status == "unique" and choices[predicted] == "same"
        ),
        **{f"path_{name}": float(path == name) for name in ("tag", "line", "text")},
    }


def _validate(records):
    """Reject a manifest that cannot supervise verification, before any GPU."""
    for record in records:
        for field in ("verdict", "verdict_choices"):
            if not record.get(field):
                raise ValueError(
                    f"speaker_verification requires verdict and verdict_choices in "
                    f"every manifest record; record {record.get('id')!r} is missing "
                    f"{field}. Regenerate the manifest with scripts/build_pairs.py."
                )
        declared = record.get("reward_task")
        if declared is not None and declared != "speaker_verification":
            raise ValueError(
                f"Record {record.get('id')!r} declares reward_task {declared!r} but "
                f"the configured reward is speaker_verification."
            )
        choices = [str(choice) for choice in record["verdict_choices"]]
        if sorted(choices) != sorted(LABELS):
            raise ValueError(
                f"Record {record.get('id')!r} offers {choices}, not {list(LABELS)}"
            )
        if str(record["verdict"]) not in choices:
            raise ValueError(
                f"Record {record.get('id')!r} verdict {record['verdict']!r} is not "
                f"one of its own choices"
            )


speaker_verification_reward.validate_sources = _validate
