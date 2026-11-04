# Copyright 2025 Individual Contributor: Haoran Wang
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
"""Differentiable categorical scores of sampled actions."""

import torch


def selected_log_probs(logits, labels, mask, temperature=1.0, top_k=0):
    """Return categorical log probability and entropy without weighting actions.

    ``mask`` distinguishes sampled actions from deterministic environment tokens.
    An audio stream's SFT loss weight must not multiply its policy log probability.
    Masked labels may be outside the vocabulary, but active labels may not be.
    """
    if logits.shape[:-1] != labels.shape or labels.shape != mask.shape:
        raise ValueError("logits, labels, and mask have incompatible shapes")
    if not logits.is_floating_point() or mask.dtype != torch.bool:
        raise ValueError("logits must be floating point and action mask must be bool")
    if temperature <= 0:
        raise ValueError("RL requires a stochastic policy with temperature > 0")
    active = mask.reshape(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = labels.reshape(-1)
    valid_labels = flat_labels[active]
    if ((valid_labels < 0) | (valid_labels >= logits.shape[-1])).any():
        raise ValueError("An active action is outside its stream vocabulary")
    # Avoid NaNs from ignored rows with no permitted token.
    selected_logits = flat_logits[active] / temperature
    if (
        selected_logits.numel()
        and not torch.isfinite(selected_logits).any(dim=-1).all()
    ):
        raise ValueError("An active action has empty categorical support")
    if top_k > 0 and top_k < logits.shape[-1]:
        values, indices = selected_logits.topk(top_k, dim=-1)
        selected_logits = torch.full_like(selected_logits, -torch.inf).scatter(
            -1, indices, values
        )
    log_dist = selected_logits.log_softmax(dim=-1)
    logp_active = log_dist.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(logp_active).all():
        raise ValueError("A sampled action is outside the declared sampling support")
    probabilities = log_dist.exp()
    entropy_active = -(
        probabilities * log_dist.masked_fill(~torch.isfinite(log_dist), 0)
    ).sum(-1)
    # index_copy is differentiable and does not evaluate invalid inactive labels.
    # Nonfinite masked logits must not contaminate inactive scores.
    zeros = flat_logits.new_zeros(flat_labels.shape)
    logp = zeros.index_copy(0, active.nonzero().squeeze(-1), logp_active)
    entropy = zeros.index_copy(0, active.nonzero().squeeze(-1), entropy_active)
    return logp.reshape(labels.shape), entropy.reshape(labels.shape)
