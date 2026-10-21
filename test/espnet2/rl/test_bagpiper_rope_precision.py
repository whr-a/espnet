"""Actual HF rotary phases survive the native BF16 factory conversion on CPU."""

import copy

import pytest
import torch
from torch import nn

from espnet2.speechlm.model.speechlm.rl.bagpiper.policy import move_native_model

pytest.importorskip("transformers.models.qwen3.modeling_qwen3")


def native_model():
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
    )
    config._attn_implementation = "sdpa"
    core = nn.Module()
    core.model = Qwen3Model(config).eval()
    # A separate frontend buffer must retain the ordinary dtype conversion.
    core.audio_encoder = copy.deepcopy(core.model.rotary_emb)
    return core


@pytest.mark.parametrize("wrapped", [False, True])
def test_exact_fp32_rope_phases_and_forward_leave_bf16_parameters_unchanged(wrapped):
    torch.manual_seed(19)
    core = native_model()
    core.model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
    original = core.model.rotary_emb.inv_freq.detach().clone()
    expected = copy.deepcopy(core).to(dtype=torch.bfloat16)
    # Independent oracle: original FP32 frequencies and normally cast weights.
    expected.model.rotary_emb.inv_freq = original.clone()
    expected.model.rotary_emb.original_inv_freq = original.clone()
    if wrapped:
        model = nn.Module()
        model.corelm = core
    else:
        model = core
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    requires_grad = {
        name: value.requires_grad for name, value in core.named_parameters()
    }
    assert move_native_model(
        model, "cpu", torch.bfloat16, preserve_decoder_rope_fp32=True
    ) is model
    assert parameter_ids == {
        name: id(value) for name, value in model.named_parameters()
    }
    for name, value in core.named_parameters():
        assert value.dtype == torch.bfloat16
        assert value.requires_grad == requires_grad[name]
        torch.testing.assert_close(
            value, dict(expected.named_parameters())[name], rtol=0, atol=0
        )
    rotary = core.model.rotary_emb
    for name in ("inv_freq", "original_inv_freq"):
        assert getattr(rotary, name).dtype == torch.float32
        torch.testing.assert_close(getattr(rotary, name), original, rtol=0, atol=0)
        assert getattr(core.audio_encoder, name).dtype == torch.bfloat16
    assert not torch.equal(original, original.bfloat16().float())
    positions = torch.tensor([[1000, 1024, 2048]])
    inputs = torch.randn(1, 3, 128, dtype=torch.bfloat16)
    cos, sin = rotary(inputs, positions)
    phases = positions.float()[..., None] * original[None, None, :]
    phases = torch.cat([phases, phases], dim=-1)
    torch.testing.assert_close(cos, phases.cos().bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(sin, phases.sin().bfloat16(), rtol=0, atol=0)
    with torch.no_grad():
        actual_hidden = core.model(
            inputs_embeds=inputs, position_ids=positions, use_cache=False
        ).last_hidden_state
        expected_hidden = expected.model(
            inputs_embeds=inputs, position_ids=positions, use_cache=False
        ).last_hidden_state
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    assert torch.isfinite(actual_hidden).all()


def test_default_retains_legacy_conversion_and_rounded_buffers_cannot_be_recovered():
    core = native_model()
    original = core.model.rotary_emb.inv_freq.detach().clone()
    move_native_model(core, "cpu", torch.bfloat16)
    assert core.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert not torch.equal(original, core.model.rotary_emb.inv_freq.float())
    with pytest.raises(ValueError, match="before casting"):
        move_native_model(
            core, "cpu", torch.bfloat16, preserve_decoder_rope_fp32=True
        )


def test_opt_in_requires_a_decoder_frequency_buffer():
    core = nn.Module()
    core.model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no original RoPE"):
        move_native_model(
            core, "cpu", torch.bfloat16, preserve_decoder_rope_fp32=True
        )
