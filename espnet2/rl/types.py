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
