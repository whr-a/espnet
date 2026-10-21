"""Matched output projection precision without changing parameter storage."""

import torch
from torch.nn import functional as F


def fp32_lm_head_enabled(config):
    """Use SGLang's existing option for both native and rollout computation."""
    enabled = getattr(config, "sglang_engine_kwargs", {}).get(
        "enable_fp32_lm_head", False
    )
    if type(enabled) is not bool:
        raise ValueError("enable_fp32_lm_head must be a boolean")
    return enabled


def project_logits(hidden, weight, bias=None, *, fp32=False):
    """Project in FP32 when opted in, retaining gradients to original weights."""
    if not fp32:
        return F.linear(hidden, weight, bias)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(
            hidden.float(), weight.float(), None if bias is None else bias.float()
        )
