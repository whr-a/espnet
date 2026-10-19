"""Opt-in WEST mean-only guard; unrelated profiles retain their bounds."""

import json

import pytest
import torch
import yaml

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.types import drift_gate


def configuration(**changes):
    return ESPnetRLConfig(
        **{
            "adapter_factory": "unused:factory",
            "exporter": "unused:exporter",
            "sglang_engine": "unused:Engine",
            "training_objective": "west_grpo",
            "loss_agg_mode": "seq-mean-token-mean",
            "execution_mode": "disagg_sync",
            "score_tolerance": None,
            "score_mean_tolerance": 0.02,
            **changes,
        }
    )


@pytest.mark.parametrize("mode", ["sync", "disagg_sync"])
def test_west_mean_only_roundtrip(tmp_path, mode):
    config = configuration(execution_mode=mode)
    config.validate()
    for name, text in (
        ("config.json", json.dumps(config.to_dict())),
        ("config.yaml", yaml.safe_dump(config.to_dict())),
    ):
        path = tmp_path / name
        path.write_text(text)
        restored = ESPnetRLConfig.load(path)
        restored.validate()
        assert restored.score_tolerance is None
        assert restored.score_mean_tolerance == 0.02


@pytest.mark.parametrize(
    "mean", [None, 0, -0.01, float("nan"), float("inf"), True, "0.02"]
)
def test_mean_only_requires_positive_finite_numeric_mean(mean):
    with pytest.raises(ValueError, match="finite and positive"):
        configuration(score_mean_tolerance=mean).validate()


@pytest.mark.parametrize("mode", ["sync", "disagg_async"])
def test_other_objectives_cannot_disable_maximum(mode):
    with pytest.raises(
        ValueError, match="score_tolerance must be finite and nonnegative"
    ):
        configuration(
            training_objective="miles_ppo",
            loss_agg_mode="token-mean",
            execution_mode=mode,
        ).validate()


@pytest.mark.parametrize("maximum", [0, 0.03, 1.5])
@pytest.mark.parametrize("mean", [None, 0, 0.02])
def test_existing_numeric_bound_combinations_unchanged(maximum, mean):
    config = configuration(
        training_objective="miles_ppo",
        loss_agg_mode="token-mean",
        execution_mode="sync",
        score_tolerance=maximum,
        score_mean_tolerance=mean,
    )
    config.validate()


@pytest.mark.parametrize("field", ["score_tolerance", "score_mean_tolerance"])
@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True, "0.02"])
def test_invalid_numeric_bounds_stay_rejected(field, invalid):
    changes = {"score_tolerance": 0.03, "score_mean_tolerance": 0.02, field: invalid}
    with pytest.raises(ValueError, match="finite and nonnegative"):
        configuration(**changes).validate()


def test_mean_only_retains_maximum_telemetry_and_rejects_systematic_error():
    config = configuration()
    config.validate()
    native = torch.zeros(1, 1000, dtype=torch.float64)
    rollout = native.clone()
    rollout[0, 0] = -3.0
    mask = torch.ones_like(native, dtype=torch.bool)
    result = drift_gate(native, rollout, mask, config, "fixture", "CPU test")
    assert result["max"] == 3.0
    assert result["mean"] == pytest.approx(0.003)
    assert result["p99"] == 0.0
    with pytest.raises(RuntimeError, match="mean logprob disagreement"):
        drift_gate(native, native - 0.025, mask, config, "fixture", "CPU test")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_mean_only_still_rejects_active_nonfinite_values(invalid):
    native = torch.tensor([[0.0, invalid]])
    with pytest.raises(RuntimeError, match="max logprob disagreement"):
        drift_gate(
            native,
            torch.zeros_like(native),
            torch.ones_like(native, dtype=torch.bool),
            configuration(),
            "fixture",
            "CPU test",
        )


def test_existing_maximum_guard_remains_active():
    config = configuration(score_tolerance=0.03)
    with pytest.raises(RuntimeError, match="max logprob disagreement"):
        drift_gate(
            torch.zeros(1, 1000),
            torch.tensor([[-0.04] + [0.0] * 999]),
            torch.ones(1, 1000, dtype=torch.bool),
            config,
            "fixture",
            "CPU test",
        )
