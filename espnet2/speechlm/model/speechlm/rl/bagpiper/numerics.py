# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Opt-in SGLang operations matching Bagpiper's native BF16 boundaries.

The native HF policy rounds residual sums, normalized activations and SiLU
outputs to BF16 before the following operation. Fused kernels can remove
these boundaries. These instance-local replacements retain the native
operations. Decoder inverse frequencies and the audio sinusoidal table are
initialized on CPU in FP32, matching the native model's construction before
its device move. Parameters, attention and sampling settings are unchanged.
End-to-end numerical agreement still requires an actual rollout check.
"""

import hashlib
import json
import logging
from types import MethodType

import torch

logger = logging.getLogger(__name__)


def _native_rmsnorm(self, x, residual=None, post_residual_addition=None):
    """HF RMSNorm with the preceding BF16 residual addition made explicit."""
    if x.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16:
        raise TypeError("Bagpiper native BF16 norms require BF16 inputs and weights")
    if residual is not None:
        x = x + residual
        if post_residual_addition is not None:
            x = x + post_residual_addition
        residual = x
    elif post_residual_addition is not None:
        raise ValueError("A post-residual addition requires a residual")
    normalized = x.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
    result = self.weight * normalized.to(x.dtype)
    return result if residual is None else (result, residual)


def _inverse_frequency(module):
    """The default HF inverse frequencies, initialized on CPU in FP32."""
    return 1.0 / (
        module.base
        ** (
            torch.arange(0, module.rotary_dim, 2, dtype=torch.float32, device="cpu")
            / module.rotary_dim
        )
    )


def _native_rope_cache(module):
    """Build FP32 phases from CPU-initialized inverse frequencies, as HF does."""
    if (
        not module.is_neox_style
        or module.rotary_dim != module.head_size
        or hasattr(module, "sin_cos_cache")
    ):
        raise ValueError(
            "Bagpiper native BF16 RoPE requires default full-head Neox RoPE"
        )
    inverse = _inverse_frequency(module)
    device = module.cos_sin_cache.device
    inverse = inverse.to(device)
    positions = torch.arange(
        module.max_position_embeddings, device=device, dtype=torch.float32
    )
    # Keep inverse frequencies and phases in FP32. The rotary application
    # casts cos/sin to the activation dtype before its separate products.
    with torch.autocast(device_type=device.type, enabled=False):
        phases = positions[:, None] * inverse[None, :]
        return torch.cat((phases.cos(), phases.sin()), dim=-1)


def _buffer_receipt(tensor):
    """Describe a precision buffer without hashing model parameters."""
    value = tensor.detach().cpu().contiguous()

    def digest(data):
        return hashlib.sha256(
            data.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "raw_sha256": digest(value),
        "fp32_value_sha256": digest(value.float()),
        "effective_bf16_value_sha256": digest(value.bfloat16()),
    }


def _restore_audio_position_table(model):
    """Use the native HF CPU initialization for this model's audio positions."""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        SinusoidsPositionEmbedding,
    )

    position = model.audio_tower.positional_embedding
    if type(position) is not SinusoidsPositionEmbedding:
        raise ValueError("Bagpiper native BF16 audio requires the HF sinusoidal table")
    original = position.positional_embedding
    # This pinned HF class explicitly computes its timescales and trig in
    # FP32, including under SGLang's BF16 default dtype. Avoid a global dtype
    # change and rebuild only this nonpersistent buffer, not the encoder.
    with torch.device("cpu"):
        replacement = SinusoidsPositionEmbedding(
            position.length, position.channels, position.max_timescale
        ).positional_embedding
    if replacement.dtype != torch.float32:
        raise TypeError("Native HF sinusoidal initialization must remain FP32")
    position.positional_embedding = replacement.to(
        device=original.device, dtype=original.dtype
    )


def install_native_bf16_numerics(model):
    """Select native arithmetic only for this opted-in Bagpiper model.

    Supports the unquantized, all-BF16 Qwen3 backbone with default RoPE.
    SGLang's mixed-precision FSDP target is a different contract and is
    rejected. The HF audio encoder keeps SDPA and receives the same CPU
    sinusoidal initialization as the native model.
    """
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

    if model.lm_head.weight.dtype != torch.bfloat16:
        raise ValueError("Bagpiper native BF16 compatibility requires BF16 weights")
    modules = list(model.model.modules())
    for module in modules:
        if isinstance(module, RMSNorm):
            if (
                module.weight.dtype != torch.bfloat16
                or module.fp32_residual
                or module.override_orig_dtype is not None
                or module.variance_size_override is not None
            ):
                raise ValueError("Unsupported mixed-precision Bagpiper RMSNorm")
        elif (
            isinstance(module, RotaryEmbedding) and type(module) is not RotaryEmbedding
        ):
            raise ValueError("Bagpiper native BF16 compatibility requires default RoPE")
    _restore_audio_position_table(model)
    for module in modules:
        if isinstance(module, RMSNorm):
            module._forward_method = MethodType(_native_rmsnorm, module)
        elif isinstance(module, SiluAndMul):
            module._forward_method = module.forward_native
        elif isinstance(module, RotaryEmbedding):
            module.cos_sin_cache = _native_rope_cache(module)
            module._apply_rotary_emb_wrapped = apply_rotary_emb
            module._forward_method = module.forward_native
    # One startup receipt under this explicit profile only. This HF audio
    # encoder has a sinusoidal table rather than rotary frequency buffers.
    # Its BF16 digest permits comparison after the table's forward cast.
    buffers = {
        name: _buffer_receipt(value)
        for name, value in model.named_buffers()
        if name.rsplit(".", 1)[-1]
        in ("inv_freq", "original_inv_freq", "positional_embedding")
    }
    frequencies = {
        name: _buffer_receipt(_inverse_frequency(module))
        for name, module in model.model.named_modules()
        if type(module) is RotaryEmbedding
    }
    logger.info(
        "Bagpiper native BF16 precision receipt: %s",
        json.dumps(
            {"buffers": buffers, "rope_cpu_inverse_frequencies": frequencies},
            sort_keys=True,
        ),
    )
