"""Token scoring contract shared by native ESPnet architectures."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TokenScores:
    log_probs: torch.Tensor
    entropy: torch.Tensor
    response_mask: torch.Tensor


class NativeESPnetPolicy(nn.Module):
    def forward(self, batch) -> TokenScores:
        raise NotImplementedError

    def checkpoint_state_dict(self):
        return self.model.state_dict()

