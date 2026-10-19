# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Finite open-interval Gumbel sampling for Bagpiper's stateless QA profile.

The pinned SGLang sampler maps UINT32_MAX to one, making its Gumbel noise
infinite. Adding that noise to a filtered -inf log probability yields NaN,
which argmax can select outside top-k. Reuse the same Murmur hash and seeds,
map each hash to the midpoint of its uniform bin, and preserve hard masks.
The SGLang repository and its pinned source remain unchanged.
"""

import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32


@torch.compile(dynamic=True)
def safe_multinomial_with_seed(logprobs, seed, positions):
    """Choose one finite-support action per row using reproducible hash noise.

    Each row must contain at least one finite log probability. Computation in
    float64 makes both uint32 endpoints strictly interior to (0, 1). The
    caller's log probabilities, seed and position tensors are not modified.
    """
    columns = torch.arange(logprobs.shape[1], device=logprobs.device)
    hashed = murmur_hash32(seed.to(torch.uint64), positions, columns)
    uniform = (hashed.to(torch.float64) + 0.5) / 4294967296.0
    scores = -torch.log(-torch.log(uniform)) + logprobs.to(torch.float64)
    scores = scores.masked_fill(~torch.isfinite(logprobs), -torch.inf)
    return torch.argmax(scores, dim=1, keepdim=True)


def install_safe_stateless_sampler():
    """Install only inside an opted-in Bagpiper SGLang worker process.

    The Bagpiper model constructor must call this only when its explicit
    stateless QA configuration flag is true. Standard or evaluation profiles
    without that flag retain SGLang's original sampler.
    """
    from sglang.srt.layers import sampler

    if sampler.multinomial_with_seed is safe_multinomial_with_seed:
        return
    current = getattr(
        sampler.multinomial_with_seed, "__wrapped__", sampler.multinomial_with_seed
    )
    if (
        current.__module__ != sampler.__name__
        or current.__name__ != "multinomial_with_seed"
    ):
        raise RuntimeError("Unexpected SGLang stateless sampler implementation")
    sampler.multinomial_with_seed = safe_multinomial_with_seed
