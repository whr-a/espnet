"""Token scoring contract shared by native ESPnet architectures."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TokenScores:
    log_probs: torch.Tensor
    entropy: torch.Tensor
    response_mask: torch.Tensor
    stream_log_probs: torch.Tensor | None = None


def validate_binary_mask(mask, name):
    """Reject weights and nonfinite values where sampled-action identity is required."""
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"{name} must contain only finite binary values")


def validate_token_scores(scores, expected_mask):
    """Ensure native scoring preserves the inference worker's sampled actions."""
    validate_binary_mask(scores.response_mask, "Native response_mask")
    if not torch.equal(scores.response_mask.to(expected_mask.device), expected_mask):
        raise ValueError("Native response_mask differs from the sampled-action mask")
    for name in ("log_probs", "entropy"):
        values = getattr(scores, name)
        if values.shape != expected_mask.shape or not torch.isfinite(values).all():
            raise ValueError(f"Native {name} must be finite and match the action shape")


def tempered(logits, temperature):
    """The logits actions are sampled from at ``temperature``: logits / T.

    At unit temperature ``logits`` itself, so its numerics stay exact.
    Otherwise divided by a tensor, as SGLang's sampler divides: CUDA divides
    by a Python number as a product with its rounded reciprocal, which
    differs in the last bit for about one value in ten.
    """
    if temperature == 1:
        return logits
    return logits / logits.new_tensor(temperature)


class NativeESPnetPolicy(nn.Module):
    def forward(self, batch) -> TokenScores:
        raise NotImplementedError

    def checkpoint_state_dict(self):
        return self.model.state_dict()


def drift_gate(native, rollout, mask, config, source, hint):
    """Check native against rollout log probabilities of the sampled actions.

    The maximum catches a gross error in any one action; the optional mean
    (score_mean_tolerance) catches a small systematic offset that a
    low-precision engine's per-action noise would hide under any usable
    maximum. Returns the max, mean and 99th percentile of the differences.
    """
    errors = (native - rollout).abs()[mask.bool()].double()
    statistics = {
        "max": errors.max().item(),
        "mean": errors.mean().item(),
        "p99": np.quantile(errors.numpy(), 0.99).item(),
    }
    for name, tolerance in (
        ("max", config.score_tolerance),
        ("mean", config.score_mean_tolerance),
    ):
        value = statistics[name]
        if not np.isfinite(value) or (tolerance is not None and value > tolerance):
            raise RuntimeError(
                f"{source} {name} logprob disagreement {value:.6g} > "
                f"{tolerance}; {hint}"
            )
    return statistics


def support_counts(policy):
    """Frames a policy scored under a recorded support its own logits do not
    give, and all frames it scored, in no-grad passes so far. Only a policy
    that scores under the supports its rollout recorded (Bagpiper) counts."""
    return [
        getattr(policy, "support_mismatches", 0),
        getattr(policy, "support_frames", 0),
    ]


def support_mismatch_rate(policy, counts, name, gate=True):
    """The mismatch rate of one scoring pass's ``counts`` (None if it counted
    nothing). When gated, a rate above the policy's support_mismatch_rate
    fails: occasional flips of a near tie are tolerated, systematic ones not."""
    mismatches, frames = counts
    if not frames:
        return None
    if gate and mismatches > policy.support_mismatch_rate * frames:
        raise RuntimeError(
            f"The {name} policy gives {mismatches} of {frames} frames another "
            f"support than their rollout recorded (rate above "
            f"{policy.support_mismatch_rate})"
        )
    return mismatches / frames
