"""The synchronous actor rejects stale/mixed groups before reward learning."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.miles_native_actor import ESPnetTrainRayActor


class Batch:
    def __init__(self):
        self.non_tensor_batch = {
            "uid": np.array([f"q{i}" for i in range(32) for _ in range(8)]),
            "generation": [
                {"behavior_policy_version": 7, "behavior_checkpoint_sha256": "f" * 64}
                for _ in range(256)
            ],
        }

    def __len__(self):
        return len(self.non_tensor_batch["uid"])


def actor_and_batch():
    actor = ESPnetTrainRayActor.__new__(ESPnetTrainRayActor)
    actor.version = 7
    actor.config = ESPnetRLConfig(
        execution_mode="disagg_sync", prompts_per_step=32, group_size=8
    )
    calls = []

    def prepare(batch, native_scores):
        calls.append(native_scores)
        return {"reward/mean": 0.5}

    actor.trainer = SimpleNamespace(prepare_batch=prepare)
    data = {"behavior_version": 7, "behavior_checkpoint_sha256": "f" * 64}
    sampled = torch.full((256, 2), -1.0)
    raw_reference = torch.full((256, 2), -2.0)
    return actor, Batch(), data, (sampled, sampled, raw_reference), calls


def test_sync_batch_uses_current_rollout_and_raw_reference_without_async_loss():
    actor, batch, data, scores, calls = actor_and_batch()
    result = actor._prepare_synchronous_batch(batch, data, scores)
    assert calls == [(scores[0], scores[2])]
    assert result["rollout/weight_staleness"] == 0
    assert result["rollout/global_prompt_batch_size"] == 32
    assert result["rollout/global_candidate_batch_size"] == 256


@pytest.mark.parametrize("corruption", ["stale", "partial", "mixed", "duplicate"])
def test_sync_batch_rejects_invalid_learning_inputs(corruption):
    actor, batch, data, scores, calls = actor_and_batch()
    if corruption == "stale":
        data["behavior_version"] = 6
    elif corruption == "partial":
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"][:-1]
    elif corruption == "mixed":
        batch.non_tensor_batch["generation"][8]["behavior_policy_version"] = 6
    else:
        batch.non_tensor_batch["uid"][8:16] = "q0"
    with pytest.raises(ValueError, match="Synchronous"):
        actor._prepare_synchronous_batch(batch, data, scores)
    assert calls == []


def test_sync_batch_rejects_an_independent_behavior_score():
    actor, batch, data, scores, calls = actor_and_batch()
    with pytest.raises(ValueError, match="behavior and current scores differ"):
        actor._prepare_synchronous_batch(
            batch, data, (scores[0], scores[0] - 0.1, scores[2])
        )
    assert calls == []
