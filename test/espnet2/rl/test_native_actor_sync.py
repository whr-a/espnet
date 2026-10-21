"""The synchronous actor rejects stale/mixed groups before reward learning."""

import numpy as np
import pytest
import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.miles_native_actor import ESPnetTrainRayActor
from espnet2.rl.trainer import NativeGRPOTrainer


class NoRescorePolicy(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)

    def forward(self, batch):
        raise AssertionError("Already audited native scores must not be recomputed")


def actor_and_batch():
    actor = ESPnetTrainRayActor.__new__(ESPnetTrainRayActor)
    actor.version = 7
    actor.config = ESPnetRLConfig(
        family="bagpiper",
        execution_mode="disagg_sync",
        device="cpu",
        training_objective="west_grpo",
        loss_agg_mode="seq-mean-token-mean",
        prompts_per_step=32,
        group_size=8,
        rollout_gpus="1",
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )
    actor.trainer = NativeGRPOTrainer(NoRescorePolicy(), actor.config)
    calls = []
    real_prepare = actor.trainer.prepare_batch

    def prepare(batch, native_scores):
        calls.append(native_scores)
        return real_prepare(batch, native_scores=native_scores)

    actor.trainer.prepare_batch = prepare
    sources = [{"id": f"q{i}", "reference": "correct"} for i in range(32)]
    generations = [
        {
            "source_id": source["id"],
            "prompt_token_ids": [1],
            "token_ids": [2, 3],
            "log_probs": [-1.0, -1.0],
            "text": "correct" if candidate % 2 == 0 else "wrong",
            "behavior_policy_version": 7,
            "behavior_checkpoint_sha256": "f" * 64,
        }
        for source in sources
        for candidate in range(8)
    ]
    batch = collate_rollouts(sources, generations, 8)
    data = {"behavior_version": 7, "behavior_checkpoint_sha256": "f" * 64}
    sampled = torch.full((256, 2), -1.0)
    raw_reference = torch.full((256, 2), -2.0)
    return actor, batch, data, (sampled, sampled, raw_reference), calls


def test_sync_batch_uses_current_rollout_and_raw_reference_without_async_loss():
    actor, batch, data, scores, calls = actor_and_batch()
    result = actor._prepare_synchronous_batch(batch, data, scores)
    assert calls == [(scores[0], scores[2])]
    assert result["rollout/weight_staleness"] == 0
    assert result["rollout/global_prompt_batch_size"] == 32
    assert result["rollout/global_candidate_batch_size"] == 256
    assert result["reward/mean"] == -0.5
    torch.testing.assert_close(batch.batch["old_log_probs"], scores[0], rtol=0, atol=0)
    torch.testing.assert_close(batch.batch["ref_log_probs"], scores[2], rtol=0, atol=0)
    assert batch.batch["advantages"].abs().gt(0).all()


@pytest.mark.parametrize("corruption", ["stale", "partial", "mixed", "duplicate"])
def test_sync_batch_rejects_invalid_learning_inputs(corruption):
    actor, batch, data, scores, calls = actor_and_batch()
    messages = {
        "stale": "current complete batch",
        "partial": "current complete batch",
        "mixed": "Mixed or unrecorded behavior checkpoint",
        "duplicate": "complete unique prompt groups",
    }
    if corruption == "stale":
        data["behavior_version"] = 6
    elif corruption == "partial":
        batch = batch.select_idxs(np.arange(255))
    elif corruption == "mixed":
        batch.non_tensor_batch["generation"][8]["behavior_policy_version"] = 6
    else:
        batch.non_tensor_batch["uid"][8:16] = "q0"
    with pytest.raises(ValueError, match=messages[corruption]):
        actor._prepare_synchronous_batch(batch, data, scores)
    assert calls == []


def test_sync_batch_rejects_an_independent_behavior_score():
    actor, batch, data, scores, calls = actor_and_batch()
    with pytest.raises(ValueError, match="behavior and current scores differ"):
        actor._prepare_synchronous_batch(
            batch, data, (scores[0], scores[0] - 0.1, scores[2])
        )
    assert calls == []
