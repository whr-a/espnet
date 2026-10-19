"""Native BF16 boundaries and safe stateless sampling without a GPU engine."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")
pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    SinusoidsPositionEmbedding,
)

from espnet2.speechlm.model.speechlm.rl.bagpiper import sampling
from espnet2.speechlm.model.speechlm.rl.bagpiper.numerics import (
    install_native_bf16_numerics,
)


@pytest.fixture(autouse=True)
def no_engine_runtime(monkeypatch):
    from sglang.srt.layers import activation
    from sglang.srt.layers.rotary_embedding import base

    execution = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None))
    monkeypatch.setattr(activation, "get_exec", lambda: execution)
    monkeypatch.setattr(base, "get_exec", lambda: execution)
    # The pinned runtime defaults to a CUDA target even with CUDA hidden.
    # Select its native CPU constructor path without initializing a device.
    monkeypatch.setattr(base, "_is_cpu", True)


def backbone():
    model = nn.Module()
    model.lm_head = nn.Linear(128, 32, bias=False).bfloat16()
    model.model = nn.ModuleList(
        [
            RMSNorm(128).bfloat16(),
            SiluAndMul(),
            RotaryEmbedding(128, 128, 2048, 1000000.0, True, torch.bfloat16),
        ]
    )
    model.audio_tower = nn.Module()
    model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(64, 32)
    return model


def test_native_bf16_rmsnorm_matches_hf_and_preserves_rounded_residual():
    model = backbone()
    norm = model.model[0]
    generator = torch.Generator().manual_seed(42)
    norm.weight.data.copy_((1 + 0.3 * torch.randn(128, generator=generator)).bfloat16())
    hf = Qwen3RMSNorm(128).bfloat16()
    hf.weight.data.copy_(norm.weight)
    x = torch.randn(64, 128, generator=generator).bfloat16()
    residual = torch.randn(64, 128, generator=generator).bfloat16()
    original_x, original_residual = x.clone(), residual.clone()
    install_native_bf16_numerics(model)
    torch.testing.assert_close(norm(x), hf(x), rtol=0, atol=0)
    actual, saved = norm(x, residual)
    torch.testing.assert_close(actual, hf(x + residual), rtol=0, atol=0)
    torch.testing.assert_close(saved, x + residual, rtol=0, atol=0)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)
    extra = torch.randn(64, 128, generator=generator).bfloat16()
    actual, saved = norm(x, residual, extra)
    torch.testing.assert_close(actual, hf((x + residual) + extra), rtol=0, atol=0)
    torch.testing.assert_close(saved, (x + residual) + extra, rtol=0, atol=0)


def test_native_bf16_rope_and_silu_match_actual_hf_operations():
    model = backbone()
    original_parameters = {
        key: value.clone() for key, value in model.state_dict().items()
    }
    install_native_bf16_numerics(model)
    config = Qwen3Config(
        hidden_size=128,
        num_attention_heads=1,
        head_dim=128,
        rope_theta=1000000.0,
        max_position_embeddings=2048,
    )
    hf = Qwen3RotaryEmbedding(config)
    generator = torch.Generator().manual_seed(43)
    positions = torch.tensor([0, 250, 1000, 1274, 2047])
    q = torch.randn(1, 2, len(positions), 128, generator=generator).bfloat16()
    k = torch.randn(1, 1, len(positions), 128, generator=generator).bfloat16()
    cos, sin = hf(q, positions[None, :])
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
    rope = model.model[2]
    assert rope.cos_sin_cache.dtype == torch.float32
    actual_q, actual_k = rope(
        positions, q[0].transpose(0, 1).reshape(len(positions), -1), k[0, 0]
    )
    torch.testing.assert_close(
        actual_q.reshape(len(positions), 2, 128).transpose(0, 1)[None],
        expected_q,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual_k[None, None], expected_k, rtol=0, atol=0)
    gates = torch.randn(64, 256, generator=generator).bfloat16()
    torch.testing.assert_close(
        model.model[1](gates),
        torch.nn.functional.silu(gates[:, :128]) * gates[:, 128:],
        rtol=0,
        atol=0,
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_parameters[key], rtol=0, atol=0)


def test_native_bf16_profile_rejects_mixed_precision():
    model = backbone()
    model.model[0].weight.data = model.model[0].weight.float()
    with pytest.raises(ValueError, match="mixed-precision"):
        install_native_bf16_numerics(model)


def test_native_audio_table_matches_full_size_hf_cpu_initialization():
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        expected = SinusoidsPositionEmbedding(1500, 1280).positional_embedding
        torch.set_default_dtype(torch.bfloat16)
        model = backbone()
        position = model.audio_tower.positional_embedding = SinusoidsPositionEmbedding(
            1500, 1280
        )
        position.positional_embedding.add_(0.01)
        parameters = {name: value.clone() for name, value in model.named_parameters()}
        install_native_bf16_numerics(model)
        actual = position.positional_embedding
        assert actual.dtype == torch.float32
        assert actual.device.type == "cpu"
        assert torch.get_default_dtype() == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.bfloat16(), expected.bfloat16(), rtol=0, atol=0
        )
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("width", [3, 128])
def test_safe_sampling_endpoints_and_hard_support(monkeypatch, width):
    count = min(50, width - 1)
    probabilities = torch.full((4, width), -torch.inf, dtype=torch.float64)
    probabilities[:, :count] = torch.linspace(-5, 0, count)
    hashed = torch.full((4, width), 2**32 - 1, dtype=torch.int64)
    hashed[:, :count] = torch.arange(count)[None, :] * 70000001
    monkeypatch.setattr(sampling, "murmur_hash32", lambda *unused: hashed)
    function = sampling.safe_multinomial_with_seed.__wrapped__
    original = probabilities.clone()
    seeds, positions = torch.arange(4), torch.arange(4) + 200
    actual = function(probabilities, seeds, positions)
    assert torch.equal(actual, function(probabilities, seeds, positions))
    assert (actual < count).all()
    assert torch.isfinite(probabilities.gather(1, actual)).all()
    assert torch.equal(probabilities, original)
    uniform = (hashed.double() + 0.5) / 2**32
    assert ((uniform > 0) & (uniform < 1)).all()
    assert torch.isfinite(-torch.log(-torch.log(uniform))).all()


def test_safe_sampler_installer_is_explicit_and_idempotent(monkeypatch):
    from sglang.srt.layers import sampler as engine_sampler

    original = engine_sampler.multinomial_with_seed
    monkeypatch.setattr(engine_sampler, "multinomial_with_seed", original)
    assert engine_sampler.multinomial_with_seed is original
    sampling.install_safe_stateless_sampler()
    assert engine_sampler.multinomial_with_seed is sampling.safe_multinomial_with_seed
    sampling.install_safe_stateless_sampler()
    assert engine_sampler.multinomial_with_seed is sampling.safe_multinomial_with_seed
