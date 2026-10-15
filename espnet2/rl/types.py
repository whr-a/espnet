"""Token scoring contract shared by native ESPnet architectures."""

from dataclasses import dataclass

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


class NativeESPnetPolicy(nn.Module):
    def forward(self, batch) -> TokenScores:
        raise NotImplementedError

    def checkpoint_state_dict(self):
        return self.model.state_dict()


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
