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
"""The speaker-verification reward and the per-class columns it must log."""

import pytest

from espnet2.rl.task_rewards_sv import speaker_verification_reward


def record(verdict="same", identity="a__b__4s"):
    return {
        "id": identity,
        "verdict": verdict,
        "verdict_choices": ["same", "different"],
        "reward_task": "speaker_verification",
    }


def score(text, verdict="same"):
    return speaker_verification_reward(record(verdict), {"text": text})


def test_tagged_correct_answer_earns_accuracy_and_format():
    scores = score("The voices match. <answer>same</answer>", "same")
    assert scores["acc"] == 1.0
    assert scores["format"] == 1.0
    assert scores["parsed"] == 1.0
    assert scores["reward"] == pytest.approx(1.1)
    assert scores["path_tag"] == 1.0


def test_tagged_wrong_answer_keeps_only_the_format_bonus():
    scores = score("<answer>different</answer>", "same")
    assert scores["acc"] == 0.0
    assert scores["format"] == 1.0
    assert scores["reward"] == pytest.approx(0.1)


def test_naming_both_labels_is_ambiguous_and_earns_no_accuracy():
    scores = score("<answer>not the same, they are different</answer>", "different")
    assert scores["ambiguous"] == 1.0
    assert scores["parsed"] == 0.0
    assert scores["acc"] == 0.0


def test_the_last_line_is_the_fallback_path_without_a_tag():
    scores = score("A long caption about the recording.\nThe speakers are different.", "different")
    assert scores["path_line"] == 1.0
    assert scores["format"] == 0.0
    assert scores["acc"] == 1.0
    assert scores["reward"] == pytest.approx(1.0)


def test_two_answer_blocks_lose_the_format_bonus():
    scores = score("<answer>same</answer> or maybe <answer>same</answer>", "same")
    assert scores["format"] == 0.0
    assert scores["acc"] == 1.0


def test_per_class_columns_reconstruct_tpr_tnr_and_balanced_accuracy():
    # A policy that always answers "same": accuracy 0.5 on a balanced set, but
    # balanced accuracy exactly 0.5 -- which is the point of these columns.
    batch = [score("<answer>same</answer>", verdict) for verdict in
             ["same"] * 4 + ["different"] * 4]
    mean = lambda name: sum(row[name] for row in batch) / len(batch)
    assert mean("acc") == pytest.approx(0.5)
    tpr = mean("hit_same") / mean("true_same")
    tnr = mean("hit_different") / mean("true_different")
    assert tpr == pytest.approx(1.0)
    assert tnr == pytest.approx(0.0)
    assert (tpr + tnr) / 2 == pytest.approx(0.5)
    assert mean("said_same") == pytest.approx(1.0)


def test_a_discriminating_policy_moves_balanced_accuracy_off_one_half():
    batch = [
        score("<answer>same</answer>", "same"),
        score("<answer>same</answer>", "same"),
        score("<answer>different</answer>", "different"),
        score("<answer>same</answer>", "different"),
    ]
    mean = lambda name: sum(row[name] for row in batch) / len(batch)
    tpr = mean("hit_same") / mean("true_same")
    tnr = mean("hit_different") / mean("true_different")
    assert (tpr + tnr) / 2 == pytest.approx(0.75)


def test_validate_sources_rejects_a_manifest_without_verdicts():
    with pytest.raises(ValueError, match="verdict"):
        speaker_verification_reward.validate_sources([{"id": "x"}])


def test_validate_sources_rejects_a_foreign_reward_task():
    rows = [{**record(), "reward_task": "speaker_id"}]
    with pytest.raises(ValueError, match="declares reward_task"):
        speaker_verification_reward.validate_sources(rows)


def test_validate_sources_rejects_choices_that_are_not_the_two_labels():
    rows = [{**record(), "verdict_choices": ["yes", "no"]}]
    with pytest.raises(ValueError, match="not"):
        speaker_verification_reward.validate_sources(rows)


def test_validate_sources_accepts_the_real_manifest_shape():
    speaker_verification_reward.validate_sources(
        [record("same", "a__b__4s"), record("different", "c__d__4s")]
    )


def test_a_verdict_outside_its_own_choices_is_rejected_at_scoring_time():
    bad = {**record(), "verdict": "maybe"}
    with pytest.raises(ValueError, match="not exactly one of"):
        speaker_verification_reward(bad, {"text": "<answer>same</answer>"})
