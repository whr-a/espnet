"""CPU contracts for real tiny HF Qwen3 and Qwen3 audio inference caches.

Run explicitly with pytest; no model downloads, Engine, CUDA or Ray startup.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from .inference_cache import CacheContractError, CandidateInferenceCache
from .sglang_model import ESPnetBagpiperForCausalLM


def tiny_model(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    geometry = {
        "text_config": {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "attention_dropout": 0.0,
            "max_position_embeddings": 128,
        },
        "audio_config": {
            "num_mel_bins": 8,
            "encoder_layers": 2,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "d_model": 32,
            "output_dim": 24,
            "max_source_positions": 64,
            "n_window": 8,
            "n_window_infer": 32,
            "conv_chunksize": 8,
            "downsample_hidden_size": 4,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        "adaptor_input_dim": 24,
        "num_stream": 9,
        "text_token_id": 4,
        "audio_token_id": 5,
        "assistant_token_id": 1,
        "eos_token_id": 2,
        "eot_token_id": 3,
        "codec_base_offset": 56,
        "codec_layer_size": 8,
        "text_token_offset": 8,
        "text_token_end": 56,
    }
    model = (
        ESPnetBagpiperForCausalLM(
            SimpleNamespace(
                espnet_bagpiper=geometry,
                espnet_rl_payload_path=str(tmp_path / "payload.json"),
                espnet_rl_cache_max_frames=128,
            )
        )
        .float()
        .eval()
    )
    native = {
        "continuous_audio_feats": torch.randn(1, 24, 8).tolist(),
        "continuous_audio_lengths": [24],
        "continuous_audio_indices": [[0, 2, 3]],
    }
    # Include every secondary stream and its native zero padding behavior.
    sequence = torch.randint(8, 128, (8, 9))
    sequence[:, 1::2] = 0
    payload = {
        "source_id": "tiny-audio",
        "sample": 0,
        "step": 0,
        "mode": "text_audio",
        "phase": "text",
        "position": 0,
        "cache_reset": True,
        "native_model_inputs": native,
        "sequence": sequence.tolist(),
    }
    return model, payload


def forward(model, payload):
    import json

    model.payload_path.write_text(json.dumps(payload))
    ids = torch.tensor([row[0] for row in payload["sequence"]])
    return model(ids, torch.arange(len(ids)), SimpleNamespace(batch_size=1))


def test_real_tiny_audio_qwen_full_vs_hf_kv_hidden_and_audio_calls(tmp_path):
    model, payload = tiny_model(tmp_path)
    full = copy.deepcopy(model)
    full.inference_cache_enabled = False
    audio_cached = []
    audio_full = []
    model.audio_tower.register_forward_hook(lambda *args: audio_cached.append(1))
    full.audio_tower.register_forward_hook(lambda *args: audio_full.append(1))
    for position in range(13):
        payload["position"] = position
        payload["cache_reset"] = position == 0
        expected = full._hidden(payload, torch.device("cpu"))
        actual = model._hidden(payload, torch.device("cpu"))
        torch.testing.assert_close(expected, actual, atol=3e-6, rtol=0)
        for stream in range(9):
            before = (
                expected if stream == 0 else expected + model.stream_emb.weight[stream]
            )
            after = actual if stream == 0 else actual + model.stream_emb.weight[stream]
            before = torch.log_softmax(before @ model.lm_head.weight.T, -1)
            after = torch.log_softmax(after @ model.lm_head.weight.T, -1)
            torch.testing.assert_close(before, after, atol=1e-5, rtol=0)
        payload["sequence"].append(
            [8 + position, *[0 if i % 2 else 56 + i for i in range(8)]]
        )
    assert len(audio_cached) == 1 and len(audio_full) == 13
    snapshot = model.inference_cache.snapshot()
    assert snapshot["counters"]["prefills"] == 1
    assert snapshot["counters"]["decodes"] == 12
    assert snapshot["kv_frames"] == 20


def test_actual_forward_support_secondary_rng_and_tokens(tmp_path):
    model, payload = tiny_model(tmp_path)
    # Two equal model instances: paired uncached forward must not clear the KV
    # belonging to the cached model.
    uncached = copy.deepcopy(model)
    uncached.inference_cache_enabled = False
    for position in range(16):
        payload["position"] = position
        payload["cache_reset"] = position == 0
        payload["phase"] = ("text", "transition", "audio", "audio_stop")[position % 4]
        torch.manual_seed(42 + position)
        full = forward(uncached, payload)
        full_trace = full.customized_info["espnet_token_trace"][0]
        full_primary = int(
            torch.multinomial(torch.softmax(full.next_token_logits[0], -1), 1)
        )
        state = torch.get_rng_state().clone()
        torch.manual_seed(42 + position)
        actual = forward(model, payload)
        trace = actual.customized_info["espnet_token_trace"][0]
        primary = int(
            torch.multinomial(torch.softmax(actual.next_token_logits[0], -1), 1)
        )
        assert torch.equal(state, torch.get_rng_state())
        assert (
            primary == full_primary
            and trace["secondary_ids"] == full_trace["secondary_ids"]
        )
        assert trace["secondary_mask"] == full_trace["secondary_mask"]
        assert trace["primary_active"] == full_trace["primary_active"]
        assert trace["sampling_contracts"] == full_trace["sampling_contracts"]
        assert torch.equal(
            torch.isfinite(actual.next_token_logits),
            torch.isfinite(full.next_token_logits),
        )
        support = torch.isfinite(actual.next_token_logits)
        torch.testing.assert_close(
            actual.next_token_logits[support],
            full.next_token_logits[support],
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.tensor(trace["secondary_log_probs"]),
            torch.tensor(full_trace["secondary_log_probs"]),
            atol=2e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.log_softmax(actual.next_token_logits.float(), -1)[0, primary],
            torch.log_softmax(full.next_token_logits.float(), -1)[0, primary],
            atol=2e-5,
            rtol=0,
        )
        payload["sequence"].append([primary, *trace["secondary_ids"]])


@pytest.mark.parametrize(
    "field", ["source_id", "sample", "step", "mode", "media", "weights"]
)
def test_identity_boundaries_require_fresh_position_and_kv(tmp_path, field):
    model, payload = tiny_model(tmp_path)
    model._hidden(payload, torch.device("cpu"))
    old = model.inference_cache.state.past_key_values
    changed = copy.deepcopy(payload)
    if field == "source_id":
        changed[field] += "-different"
    elif field in ("sample", "step"):
        changed[field] += 1
    elif field == "mode":
        changed[field] = "text_only"
    elif field == "media":
        changed["native_model_inputs"]["continuous_audio_feats"][0][0][0] += 1
    else:
        with torch.no_grad():
            model.audio_adaptor.weight.add_(0.001)
    model._hidden(changed, torch.device("cpu"))
    assert model.inference_cache.state.past_key_values is not old
    changed["position"] = 1
    changed["cache_reset"] = False
    changed["sequence"].append([8, *[0] * 8])
    changed["step"] += 1
    with pytest.raises(CacheContractError, match="position zero"):
        model._hidden(changed, torch.device("cpu"))
    assert model.inference_cache.state is None


@pytest.mark.parametrize(
    "violation", ["primary", "secondary", "repeat", "skip", "extra", "kv_length"]
)
def test_complete_extension_contract_rejects_changed_and_out_of_order_requests(
    tmp_path, violation
):
    model, payload = tiny_model(tmp_path)
    model._hidden(payload, torch.device("cpu"))
    payload["cache_reset"] = False
    payload["position"] = 1
    payload["sequence"].append([8, *[0] * 8])
    if violation in ("primary", "secondary"):
        payload["sequence"][0][0 if violation == "primary" else 8] += 1
    elif violation == "repeat":
        payload["position"] = 0
    elif violation == "skip":
        payload["position"] = 2
    elif violation == "extra":
        payload["sequence"].append([9, *[0] * 8])
    else:
        model.inference_cache.state.past_key_values.crop(7)
    with pytest.raises(CacheContractError):
        model._hidden(payload, torch.device("cpu"))
    assert model.inference_cache.state is None


def test_one_candidate_bounds_and_terminal_weight_device_cleanup(tmp_path):
    model, payload = tiny_model(tmp_path)
    for sample in range(20):
        payload["sample"] = sample
        model._hidden(payload, torch.device("cpu"))
        assert model.inference_cache.snapshot()["active_candidates"] == 1
    model.load_weights(
        (name, value.detach().clone()) for name, value in model.named_parameters()
    )
    assert model.inference_cache.state is None
    model._hidden(payload, torch.device("cpu"))
    model.to(dtype=torch.float64)
    assert model.inference_cache.state is None
    model.to(dtype=torch.float32)
    payload["cache_release"] = True
    forward(model, payload)
    assert model.inference_cache.state is None
    model.inference_cache.max_frames = 4
    result = forward(model, payload)
    assert model.inference_cache.state is None
    assert (
        result.customized_info["espnet_token_trace"][0]["inference_cache"]["operation"]
        == "full_prefix"
    )
    model.inference_cache = CandidateInferenceCache(128, 1)
    payload["cache_release"] = False
    model._hidden(payload, torch.device("cpu"))
    assert model.inference_cache.state.audio_features is None
    payload["position"] = 1
    payload["cache_reset"] = False
    payload["sequence"].append([8, *[0] * 8])
    model._hidden(payload, torch.device("cpu"))
    assert model.inference_cache.counters["audio_encoder_calls"] == 1


def test_failed_hf_forward_never_reuses_partially_mutated_cache(tmp_path, monkeypatch):
    model, payload = tiny_model(tmp_path)
    model._hidden(payload, torch.device("cpu"))
    payload["position"] = 1
    payload["cache_reset"] = False
    payload["sequence"].append([8, *[0] * 8])
    original = model.model.forward

    def fail_after_mutation(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("fixture partial HF failure")

    monkeypatch.setattr(model.model, "forward", fail_after_mutation)
    with pytest.raises(RuntimeError, match="partial HF failure"):
        model._hidden(payload, torch.device("cpu"))
    assert model.inference_cache.state is None


def test_native_and_sglang_contract_failure_also_clears_cache(tmp_path):
    model, payload = tiny_model(tmp_path)
    model._hidden(payload, torch.device("cpu"))
    ids = torch.tensor([row[0] for row in payload["sequence"]])
    with pytest.raises(ValueError, match="one request"):
        model(ids, torch.arange(len(ids)), SimpleNamespace(batch_size=2))
    assert model.inference_cache.state is None
    model._hidden(payload, torch.device("cpu"))
    import json

    model.payload_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prompt mismatch"):
        model(ids + 1, torch.arange(len(ids)), SimpleNamespace(batch_size=1))
    assert model.inference_cache.state is None
    model._hidden(payload, torch.device("cpu"))
    payload["sequence"][0] = [1]
    with pytest.raises(CacheContractError, match="multi-stream"):
        model._hidden(payload, torch.device("cpu"))
    assert model.inference_cache.state is None
