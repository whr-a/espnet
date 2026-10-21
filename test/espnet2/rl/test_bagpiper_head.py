"""Matched BF16-storage/FP32-compute head values, gradients and publication."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from espnet2.rl.config import ESPnetRLConfig
from espnet2.speechlm.model.speechlm.rl.bagpiper import integration, policy
from espnet2.speechlm.model.speechlm.rl.bagpiper.head import (
    fp32_lm_head_enabled,
    project_logits,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.sglang_model import (
    ESPnetBagpiperForCausalLM,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.tokens import CODEC, PHASES, TEXT

GEOMETRY = dict(
    num_stream=4,
    eos_token_id=2,
    eot_token_id=3,
    assistant_token_id=6,
    text_token_id=7,
    audio_token_id=8,
    text_token_offset=16,
    text_token_end=48,
    codec_base_offset=48,
    codec_layer_size=8,
)


def native_core():
    core = nn.Module()
    core.model = nn.Identity()
    core._embed = lambda *args: None
    core.num_stream = 4
    core.stream_emb = nn.Embedding(4, 13)
    core.lm_head = nn.Linear(13, 80, bias=False)
    return core.bfloat16()


@pytest.mark.parametrize("fp32", [False, True])
def test_head_values_gradients_and_native_sg_protocol_agreement(fp32):
    torch.manual_seed(43)
    core = native_core()
    native = policy.NativeSpeechLMPolicy(
        core, GEOMETRY, temperature=0.7, west_grpo=True, fp32_lm_head=fp32
    )
    reference = copy.deepcopy(native)
    assert reference.fp32_lm_head == fp32
    hidden = torch.randn(5, 13).bfloat16().requires_grad_()
    expected_hidden = hidden.detach().clone().requires_grad_()
    expected_weight = core.lm_head.weight.detach().clone().requires_grad_()
    labels = torch.tensor([[18, 0, 0, 0]] * 5)
    active = torch.tensor([[True, False, False, False]] * 5)
    supports = torch.full((5,), TEXT)
    empty = torch.empty(0, dtype=torch.long)
    projected = F.linear(
        expected_hidden.float() if fp32 else expected_hidden,
        expected_weight.float() if fp32 else expected_weight,
    )
    protocol = (
        projected.float().masked_fill(~native.allowed[supports], -torch.inf) / 0.7
    )
    expected_protocol = protocol.log_softmax(-1)[:, 18]
    raw_projection = F.linear(
        expected_hidden.float() if fp32 else expected_hidden,
        expected_weight.float() if fp32 else expected_weight,
    )
    expected_raw = raw_projection.float().log_softmax(-1)[:, 18]
    measured_protocol = native._head(hidden, labels, active, supports, empty)[0][:, 0]
    measured_raw = native._raw_head(hidden, labels, active, supports, empty)[0][:, 0]
    torch.testing.assert_close(measured_protocol, expected_protocol, rtol=0, atol=0)
    torch.testing.assert_close(measured_raw, expected_raw, rtol=0, atol=0)
    (measured_protocol.sum() + measured_raw.sum()).backward()
    (expected_protocol.sum() + expected_raw.sum()).backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        core.lm_head.weight.grad, expected_weight.grad, rtol=0, atol=0
    )
    assert core.lm_head.weight.dtype == core.lm_head.weight.grad.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in native.parameters())

    sg = ESPnetBagpiperForCausalLM.__new__(ESPnetBagpiperForCausalLM)
    nn.Module.__init__(sg)
    sg.lm_head, sg.geometry = core.lm_head, GEOMETRY
    sg.vocab, sg.fp32_lm_head, sg.codec = 80, fp32, False
    sg.temperature, sg.masks = 0.7, native.allowed
    sg.phase = torch.full((5,), PHASES.index("text"))
    sg.text_audio = torch.zeros(5, dtype=torch.bool)
    result = sg._head(hidden.detach(), torch.arange(5))
    assert all(row == [TEXT, 1] for row in result.customized_info["espnet_bagpiper"])
    torch.testing.assert_close(
        result.next_token_logits, protocol.detach(), rtol=0, atol=0
    )


@pytest.mark.parametrize("fp32", [False, True])
def test_secondary_stream_projection_matches_native(fp32):
    torch.manual_seed(3)
    core = native_core()
    native = policy.NativeSpeechLMPolicy(
        core, GEOMETRY, temperature=0.7, fp32_lm_head=fp32
    )
    sg = ESPnetBagpiperForCausalLM.__new__(ESPnetBagpiperForCausalLM)
    nn.Module.__init__(sg)
    sg.geometry, sg.stream_emb = GEOMETRY, core.stream_emb
    sg.temperature, sg.fp32_lm_head = 0.7, fp32
    hidden = torch.randn(3, 13).bfloat16()
    ids, sg_scores = sg._secondary(hidden, core.lm_head.weight)
    labels = torch.cat((torch.full((3, 1), 48), ids), -1)
    active = torch.ones(3, 4, dtype=torch.bool)
    columns = native._head(
        hidden, labels, active, torch.full((3,), CODEC), torch.arange(3)
    )[0]
    torch.testing.assert_close(columns[:, 1:], sg_scores, rtol=0, atol=0)


def test_fp32_projection_preserves_storage_and_overrides_autocast():
    torch.manual_seed(0)
    hidden, weight = torch.randn(3, 13).bfloat16(), torch.randn(80, 13).bfloat16()
    bias = torch.randn(80).bfloat16()
    before = [x.clone() for x in (hidden, weight, bias)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        measured = project_logits(hidden, weight, bias, fp32=True)
    expected = F.linear(hidden.float(), weight.float(), bias.float())
    assert measured.dtype == torch.float32
    torch.testing.assert_close(measured, expected, rtol=0, atol=0)
    assert not torch.equal(measured, F.linear(hidden, weight, bias).float())
    for value, original in zip((hidden, weight, bias), before, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_one_config_option_reaches_factory_export_and_reference(
    tmp_path, monkeypatch, enabled
):
    core = native_core()
    checkpoint = tmp_path / "native.pth"
    recipe, assets = tmp_path / "native.yaml", tmp_path / "assets"
    torch.save(core.state_dict(), checkpoint)
    recipe.write_text("job_type: speechlm\n")
    assets.mkdir()
    (assets / "config.json").write_text(
        json.dumps({"text_config": {}, "eos_token_id": 2, "bos_token_id": 1})
    )
    config = ESPnetRLConfig(
        family="bagpiper",
        checkpoint=str(checkpoint),
        train_config=str(recipe),
        dtype="bfloat16",
        sglang_engine_kwargs={"enable_fp32_lm_head": enabled},
    )
    monkeypatch.setattr(policy, "_build_current_native_model", lambda *args: core)
    monkeypatch.setattr(policy, "_bagpiper_geometry", lambda model: GEOMETRY)
    native = policy.factory(config, "cpu")
    assert native.fp32_lm_head == copy.deepcopy(native).fp32_lm_head == enabled
    config.adapter_options = {"reference_assets": str(assets)}
    monkeypatch.setattr(integration, "_native_export_geometry", lambda *args: None)
    integration.exporter(config, checkpoint, tmp_path / "export")
    metadata = json.loads((tmp_path / "export/config.json").read_text())
    assert metadata.get("espnet_bagpiper_fp32_lm_head", False) == enabled
    if not enabled:
        assert "espnet_bagpiper_fp32_lm_head" not in metadata
    assert all(
        value.dtype == torch.bfloat16
        for value in native.checkpoint_state_dict().values()
    )


def test_fp32_head_sets_rollout_ieee_before_engine_initialization(
    tmp_path, monkeypatch
):
    import os

    from espnet2.rl.engine import SGLangRolloutEngine
    from espnet2.speechlm.model.speechlm.rl.bagpiper.rollout import BagpiperEngine

    class BeforeEngine(Exception):
        pass

    def constructor(*args, **kwargs):
        assert os.environ["TRITON_F32_DEFAULT"] == "ieee"
        assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
        raise BeforeEngine

    monkeypatch.setenv("TRITON_F32_DEFAULT", "tf32")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    monkeypatch.setattr(SGLangRolloutEngine, "__init__", constructor)
    config = ESPnetRLConfig(
        dtype="bfloat16", sglang_engine_kwargs={"enable_fp32_lm_head": True}
    )
    with pytest.raises(BeforeEngine):
        BagpiperEngine(config, tmp_path, directory=tmp_path)


def test_fp32_head_option_requires_boolean():
    with pytest.raises(ValueError, match="must be a boolean"):
        fp32_lm_head_enabled(
            SimpleNamespace(sglang_engine_kwargs={"enable_fp32_lm_head": "false"})
        )
