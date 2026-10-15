"""Independent probability and gradient oracles for stale GRPO batches."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.miles_library import compute_policy_loss_vanilla, kl_penalty
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


def configuration(**kwargs):
    return ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        **kwargs,
    )


def test_prompt_batch_geometry_and_frozen_roundtrip(tmp_path):
    # prompts_per_step and group_size are the only names for the per-update
    # geometry now, and they carry their own defaults.
    plain = configuration()
    assert (plain.prompts_per_step, plain.group_size) == (2, 4)
    config = configuration(
        execution_mode="disagg_async",
        prompts_per_step=32,
        group_size=8,
        steps=5,
        warmup_steps=1,
    )
    config.validate()
    assert config.prompts_per_step == 32
    assert config.group_size == 8
    assert config.global_candidate_batch_size == 256
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert ESPnetRLConfig.load(path).to_dict() == config.to_dict()
    # The former aliases cannot name the geometry under any value any more.
    for alias in ("train_batch_size", "n_samples_per_prompt"):
        with pytest.raises(TypeError):
            configuration(**{alias: 32})


@pytest.mark.parametrize(
    "change",
    [
        {"rollout_backend": "vllm"},
        {"update_weights_interval": 2},
        {"max_weight_staleness": 2},
        {"tis_clip_low": 1.1},
        {"tis_clip": float("nan")},
        {"warmup_steps": 5},
        {"rollout_mem_fraction": 1.0},
        {"rollout_mem_fraction": 0.0},
    ],
)
def test_invalid_async_profiles(change):
    with pytest.raises(ValueError):
        configuration(execution_mode="disagg_async", steps=5, **change).validate()


@pytest.mark.parametrize(
    "change",
    [
        {"score_tolerance": None},
        {"score_tolerance": float("nan")},
        {"score_mean_tolerance": -0.1},
        {"score_mean_tolerance": float("inf")},
        {"score_mean_tolerance": True},
    ],
)
def test_invalid_drift_gate_settings(change):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        configuration(**change).validate()


@pytest.mark.parametrize("change", [{"tis_clip_low": 0.0}, {"tis_clip": 0.9}])
def test_icepop_bounds_hold_in_sync_too(change):
    with pytest.raises(ValueError, match="IcePop"):
        configuration(**change).validate()


def test_icepop_loss_gradient_and_independent_kl_with_forced_frames():
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    old = torch.zeros_like(mask, requires_grad=True)
    behavior = -torch.tensor([[1.5, 999.0, 0.4], [2.1, 0.75, 999.0]]).log()
    behavior.requires_grad_()
    current = torch.zeros_like(mask, requires_grad=True)
    advantage = torch.tensor([[1.0, 99.0, -2.0], [3.0, -4.0, 99.0]])
    config = SimpleNamespace(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        tis_clip_low=0.5,
        tis_clip=2.0,
        global_batch_info={"batch_num_tokens": 4, "dp_size": 1},
    )
    loss, metrics = compute_policy_loss_vanilla(
        old,
        current,
        advantage,
        mask,
        "token-mean",
        config,
        rollout_log_probs=behavior,
    )
    # Valid ratios 1.5 and .75 contribute; .4 and 2.1 are popped.
    torch.testing.assert_close(loss, torch.tensor(0.375))
    loss.backward()
    torch.testing.assert_close(
        current.grad, torch.tensor([[-0.375, 0.0, 0.0], [0.0, 0.75, 0.0]])
    )
    assert behavior.grad is None
    assert metrics["actor/tis_clipfrac"] == 0.5
    assert metrics["actor/tis"] == pytest.approx(1.1875)
    # KL continues to cover all sampled actions, including popped PG tokens.
    kl_current = torch.zeros_like(mask, requires_grad=True)
    ref = torch.full_like(mask, -0.2)
    kl = (kl_penalty(kl_current, ref, "k3") * mask).sum() / 4
    kl.backward()
    assert kl_current.grad[0, 2] != 0
    assert kl_current.grad[1, 0] != 0
    assert kl_current.grad[0, 1] == 0


def test_async_global_denominator_survives_microbatches():
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    old = torch.zeros_like(mask)
    behavior = -torch.tensor([[1.5, 1.0], [0.75, 1.0]]).log()
    advantages = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    config = SimpleNamespace(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        tis_clip_low=0.5,
        tis_clip=2.0,
        global_batch_info={"batch_num_tokens": 3, "dp_size": 1},
    )
    current = torch.zeros_like(mask, requires_grad=True)
    whole, _ = compute_policy_loss_vanilla(
        old, current, advantages, mask, "token-mean", config, rollout_log_probs=behavior
    )
    whole.backward()
    chunks = torch.zeros_like(mask, requires_grad=True)
    split = sum(
        compute_policy_loss_vanilla(
            old[i : i + 1],
            chunks[i : i + 1],
            advantages[i : i + 1],
            mask[i : i + 1],
            "token-mean",
            config,
            rollout_log_probs=behavior[i : i + 1],
        )[0]
        for i in range(2)
    )
    split.backward()
    torch.testing.assert_close(split, whole)
    torch.testing.assert_close(chunks.grad, current.grad)


class Scorer(torch.nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(probability))

    def forward(self, batch):
        mask = batch.batch["response_mask"]
        return TokenScores(self.weight.expand_as(mask), torch.zeros_like(mask), mask)


def async_batch(config):
    sources = [
        {"id": f"p{i}", "reference": "hello"} for i in range(config.prompts_per_step)
    ]
    generations = [
        {
            "source_id": source["id"],
            "prompt_token_ids": [1],
            "token_ids": [2, 0],
            "log_probs": [-0.2, 0.0],
            "action_mask": [1, 0],
            "text": "hello",
            "behavior_policy_version": 0,
            "behavior_checkpoint_sha256": "a" * 64,
        }
        for source in sources
        for _ in range(config.group_size)
    ]
    return collate_rollouts(sources, generations, config.group_size)


@pytest.mark.parametrize("mode", ["sync", "disagg_async"])
def test_drift_gate_statistics_and_the_mean_gate_on_an_offset(mode):
    config = configuration(execution_mode=mode, prompts_per_step=32, group_size=8)
    assert config.score_mean_tolerance is None
    trainer = NativeGRPOTrainer(Scorer(-0.2), config, reference=Scorer(-0.3))

    def prepare(errors):
        # Every candidate's one sampled action; its forced frame stays out.
        batch = async_batch(config)
        batch.batch["rollout_log_probs"][:, 0] -= errors
        if mode == "sync":
            return trainer.prepare_batch(batch)
        return trainer.prepare_async_batch(
            batch,
            behavior_version=0,
            learner_version=0,
            behavior_checkpoint_sha256="a" * 64,
        )

    # A small offset on every action: well under the maximum's bound.
    errors = torch.linspace(0.0, 0.02, 256)
    metrics = prepare(errors)
    assert metrics["rollout/logprob_max_error"] == pytest.approx(0.02, abs=1e-6)
    assert metrics["rollout/logprob_mean_error"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["rollout/logprob_p99_error"] == pytest.approx(
        np.quantile(errors.numpy(), 0.99), abs=1e-6
    )
    config.score_mean_tolerance = 0.005
    with pytest.raises(RuntimeError, match="mean logprob disagreement 0.01"):
        prepare(errors)
    config.score_mean_tolerance = 0.05
    prepare(errors)
    with pytest.raises(RuntimeError, match="max logprob disagreement 0.04"):
        prepare(torch.full((256,), 0.04))


def test_synchronous_update_applies_icepop_and_equal_logprobs_weigh_one():
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    old = torch.tensor([[-0.5, -1.2, 0.0], [-0.3, 0.0, -2.0]])
    advantages = torch.tensor([[1.0, 1.0, 0.0], [-2.0, 0.0, -2.0]])
    config = SimpleNamespace(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        tis_clip_low=0.5,
        tis_clip=2.0,
        global_batch_info={"batch_num_tokens": 4, "dp_size": 1},
    )
    plain_current = (old + 0.1).requires_grad_()
    current = (old + 0.1).requires_grad_()
    plain, _ = compute_policy_loss_vanilla(
        old, plain_current, advantages, mask, "token-mean", config
    )
    corrected, metrics = compute_policy_loss_vanilla(
        old, current, advantages, mask, "token-mean", config, rollout_log_probs=old
    )
    plain.backward()
    corrected.backward()
    # An engine that agrees with the learner leaves the update unchanged.
    torch.testing.assert_close(corrected, plain, rtol=0, atol=0)
    torch.testing.assert_close(current.grad, plain_current.grad, rtol=0, atol=0)
    assert (metrics["actor/tis"], metrics["actor/tis_clipfrac"]) == (1.0, 0.0)
    assert metrics["actor/tis_abs"] == 0.0
    # The synchronous trainer hands the engine's logprobs to the same loss.
    sync = configuration(prompts_per_step=32, group_size=8, micro_batch_size=64)
    trainer = NativeGRPOTrainer(Scorer(-0.2), sync, reference=Scorer(-0.3))
    batch = async_batch(sync)
    trainer.prepare_batch(batch)
    metrics = trainer.update_actor(batch)
    assert metrics["actor/tis"] == pytest.approx(1.0)
    assert metrics["actor/tis_clipfrac"] == 0.0


def test_behavior_numerical_audit_accepts_real_learner_drift():
    config = configuration(
        execution_mode="disagg_async", prompts_per_step=32, group_size=8
    )
    trainer = NativeGRPOTrainer(Scorer(-0.7), config, reference=Scorer(-0.3))
    batch = async_batch(config)
    metrics = trainer.prepare_async_batch(
        batch,
        Scorer(-0.2),
        behavior_version=0,
        learner_version=1,
        behavior_checkpoint_sha256="a" * 64,
    )
    assert metrics["rollout/global_candidate_batch_size"] == 256
    assert metrics["rollout/logprob_max_error"] == 0
    assert metrics["rollout/current_behavior_logprob_max_difference"] == 0.5
    assert not batch.batch["old_log_probs"].requires_grad
    assert not batch.batch["behavior_native_log_probs"].requires_grad


@pytest.mark.parametrize(
    "failure", ["probabilities", "version", "groups", "staleness", "current"]
)
def test_behavior_audit_rejects_contract_violation(failure):
    config = configuration(
        execution_mode="disagg_async", prompts_per_step=32, group_size=8
    )
    trainer = NativeGRPOTrainer(Scorer(-0.7), config, reference=Scorer(-0.3))
    batch = async_batch(config)
    behavior = Scorer(-0.2)
    version = 1
    if failure == "probabilities":
        behavior = Scorer(-0.4)
    elif failure == "version":
        batch.non_tensor_batch["generation"][0]["behavior_policy_version"] = 1
    elif failure == "groups":
        batch.non_tensor_batch["uid"][8:16] = "p0"
    elif failure == "staleness":
        version = 2
    elif failure == "current":
        behavior = trainer.policy
    with pytest.raises((ValueError, RuntimeError)):
        trainer.prepare_async_batch(
            batch,
            behavior,
            behavior_version=0,
            learner_version=version,
            behavior_checkpoint_sha256="a" * 64,
        )


def test_async_cannot_enter_synchronous_driver():
    trainer = NativeGRPOTrainer(
        Scorer(-0.2), configuration(execution_mode="disagg_async")
    )
    with pytest.raises(ValueError, match="Miles async driver"):
        trainer.train([])


def test_replica_merge_keeps_groups_and_rejects_partial_or_duplicate_results():
    from espnet2.rl.rollout import merge_generations

    sources = [{"id": f"p{i}"} for i in range(32)]
    results = [
        [
            {"source_id": source["id"], "sample": sample}
            for source in sources[replica::8]
            for sample in range(8)
        ]
        for replica in range(8)
    ]
    merged = merge_generations(sources, results, 8)
    assert len(merged) == 256
    assert [row["source_id"] for row in merged] == [
        source["id"] for source in sources for _ in range(8)
    ]
    with pytest.raises(ValueError, match="complete"):
        merge_generations(sources, results[:-1], 8)
    with pytest.raises(ValueError, match="complete"):
        merge_generations(sources, results + [results[0]], 8)
