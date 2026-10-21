"""WEST raw loss and filtered rollout-integrity domains on CPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from espnet2.rl.batch import RolloutBatch
from espnet2.rl.scoring import score_native_batch
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores
from espnet2.speechlm.model.speechlm.rl.bagpiper import rollout as rollout_module
from espnet2.speechlm.model.speechlm.rl.bagpiper.policy import NativeSpeechLMPolicy
from espnet2.speechlm.model.speechlm.rl.bagpiper.tokens import DETECT, TEXT

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


def tiny_policy_and_batch():
    core = nn.Module()
    core.model = nn.Identity()
    core._embed = lambda *args: None
    core.num_stream = 4
    core.stream_emb = nn.Embedding(4, 80)
    core.lm_head = nn.Linear(80, 80, bias=False)
    with torch.no_grad():
        core.lm_head.weight.copy_(torch.eye(80))
    policy = NativeSpeechLMPolicy(
        core, GEOMETRY, temperature=0.7, top_k=50, west_grpo=True
    )
    hidden = torch.linspace(-1.0, 1.0, 320).reshape(1, 4, 80).requires_grad_()
    with torch.no_grad():
        hidden[0, 0, 7] = hidden[0, 1, 18] = hidden[0, 2, 2] = 5.0
    policy._hidden = lambda sequence, sources: hidden
    source = {
        "id": "qa",
        "mode": "text_only",
        "support": [DETECT, TEXT, TEXT],
        "sampling_contract": {"temperature": 0.7, "top_k": 50},
    }
    prompts = np.empty(1, dtype=object)
    prompts[0] = [[6, 0, 0, 0]]
    batch = RolloutBatch(
        {
            "responses": torch.tensor([[[7, 0, 0, 0], [18, 0, 0, 0], [2, 0, 0, 0]]]),
            "response_mask": torch.tensor([[0.0, 1.0, 1.0]]),
            "response_stream_mask": torch.tensor(
                [
                    [
                        [False] * 4,
                        [True, False, False, False],
                        [True, False, False, False],
                    ]
                ]
            ),
            "response_lengths": torch.tensor([3]),
        },
        {"prompt_token_ids": prompts, "source": np.array([source], dtype=object)},
    )
    return policy, batch, hidden


def test_raw_west_values_gradients_and_forced_frame_mask():
    policy, batch, hidden = tiny_policy_and_batch()
    raw = policy(batch, score_raw=True)
    expected = hidden[0, :3].float().log_softmax(-1)
    selected = expected[torch.arange(3), torch.tensor([7, 18, 2])]
    selected = selected * batch.batch["response_mask"][0]
    torch.testing.assert_close(raw.log_probs[0], selected)
    torch.testing.assert_close(raw.response_mask, batch.batch["response_mask"].bool())
    assert raw.stream_log_probs[..., 1:].eq(0).all()
    torch.testing.assert_close(policy.score_raw(batch).log_probs, raw.log_probs)
    loss = -raw.log_probs.sum()
    loss.backward()
    analytical = expected.detach().exp()
    analytical[torch.arange(3), torch.tensor([7, 18, 2])] -= 1
    analytical[0] = 0
    torch.testing.assert_close(hidden.grad[0, :3], analytical)
    # Raw normalization includes vocabulary outside native text support.
    assert hidden.grad[0, 1, 60] > 0
    assert hidden.grad[0, 0].eq(0).all()
    assert hidden.grad[0, 3].eq(0).all()
    assert policy.core.lm_head.weight.grad is not None
    assert torch.isfinite(policy.core.lm_head.weight.grad).all()
    with torch.no_grad():
        integrity = policy(batch)
    assert not torch.allclose(integrity.log_probs, raw.log_probs)
    assert policy.support_mismatches == 0


@pytest.mark.parametrize("change", ["text_audio", "secondary", "contract", "legacy"])
def test_raw_west_rejects_wrong_mode_or_sampling_contract(change):
    policy, batch, _ = tiny_policy_and_batch()
    if change == "text_audio":
        batch.non_tensor_batch["source"][0]["mode"] = "text_audio"
    elif change == "secondary":
        batch.batch["response_stream_mask"][0, 1, 1] = True
    elif change == "contract":
        batch.non_tensor_batch["source"][0]["sampling_contract"]["top_k"] = 4
    else:
        policy.west_grpo = False
    with pytest.raises(ValueError, match="WEST|sampling contract"):
        policy(batch, score_raw=True)


class Recorder:
    def __init__(self):
        self.calls = []

    def generate(self, **request):
        self.calls.append(request)
        return [
            {
                "output_ids": [18, 2],
                "meta_info": {
                    "id": rid,
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.2, 18, None], [-0.3, 2, None]],
                    "espnet_bagpiper": [[TEXT, 1], [TEXT, 1]],
                },
            }
            for rid in request["rid"]
        ]


def engine(tmp_path, monkeypatch, *, west=True, stateless=True):
    obj = rollout_module.BagpiperEngine.__new__(rollout_module.BagpiperEngine)
    obj.config = SimpleNamespace(
        temperature=0.7,
        max_tokens=1024,
        group_size=8,
        training_objective="west_grpo" if west else "miles_ppo",
        top_k=50 if west else -1,
        seed=42,
        adapter_options={"stateless_sampling": stateless} if stateless else {},
    )
    obj.model_path, obj.directory, obj.batches = tmp_path, tmp_path, 0
    obj.geometry, obj.engine = dict(GEOMETRY), Recorder()
    obj.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "final")
    obj.check_request = lambda *args: None
    monkeypatch.setattr(
        rollout_module,
        "request_factory",
        lambda config, model, sources: (sources, [[[6, 0, 0, 0]]] * len(sources)),
    )
    monkeypatch.setattr(rollout_module, "prepared_native_metadata", lambda source: {})
    return obj


def test_west_top_k_trace_and_stateless_seed_survive_engine_restart(
    tmp_path, monkeypatch
):
    obj = engine(tmp_path, monkeypatch)
    sources = [
        {"id": "audio-1", "mode": "text_only"},
        {"id": "audio-2", "mode": "text_only"},
    ]
    generations = obj.generate(sources, 100)
    params = obj.engine.calls[-1]["sampling_params"]
    seeds = [p["sampling_seed"] for p in params]
    assert len(set(seeds)) == 16 and all(0 <= seed < 2**32 for seed in seeds)
    assert {p["top_k"] for p in params} == {50}
    assert {p["temperature"] for p in params} == {1.0}
    assert [g["sampling_seed"] for g in generations] == seeds
    assert generations[0]["sampling_contract"] == {
        "temperature": 0.7,
        "top_k": 50,
        "log_prob_domain": "tempered_protocol_before_top_k",
    }
    stored = torch.load(tmp_path / "sources.pt", weights_only=True)
    assert stored["temperature"] == 0.7 and stored["stateless_sampling"] is True
    # Engine batch counters and partitioning do not affect an absolute rollout seed.
    obj.batches = 876
    obj.generate(sources[:1], 100)
    assert [
        p["sampling_seed"] for p in obj.engine.calls[-1]["sampling_params"]
    ] == seeds[:8]
    obj.generate(sources[:1], 101)
    assert [
        p["sampling_seed"] for p in obj.engine.calls[-1]["sampling_params"]
    ] != seeds[:8]


def test_legacy_sampling_flags_and_payload_stay_unchanged(tmp_path, monkeypatch):
    obj = engine(tmp_path, monkeypatch, west=False, stateless=False)
    result = obj.generate([{"id": "audio", "mode": "text_audio"}], 1)
    params = obj.engine.calls[-1]["sampling_params"]
    assert {p["top_k"] for p in params} == {-1}
    assert not any("sampling_seed" in p for p in params)
    assert not any("sampling_seed" in g for g in result)
    assert result[0]["sampling_contract"] == {"temperature": 0.7, "top_k": 0}
    stored = torch.load(tmp_path / "sources.pt", weights_only=True)
    assert "stateless_sampling" not in stored


def test_stateless_sampling_rejects_secondary_audio(tmp_path, monkeypatch):
    obj = engine(tmp_path, monkeypatch, west=False)
    with pytest.raises(ValueError, match="text_only"):
        obj.generate([{"id": "tts", "mode": "text_audio"}], 1)


def test_stateless_sampling_requires_actual_engine_seed_support(tmp_path, monkeypatch):
    obj = engine(tmp_path, monkeypatch)
    obj.replica, obj.config.dtype = 0, "bfloat16"
    options = {"attention_backend": "fa3"}
    monkeypatch.setattr(
        rollout_module, "engine_kwargs", lambda *args, **kwargs: dict(options)
    )
    assert obj.settings(tmp_path)["enable_deterministic_inference"] is True
    options["enable_deterministic_inference"] = False
    with pytest.raises(ValueError, match="deterministic inference"):
        obj.settings(tmp_path)
    obj.config.adapter_options = {}
    assert obj.settings(tmp_path)["enable_deterministic_inference"] is False
    options.clear()
    assert "enable_deterministic_inference" not in obj.settings(tmp_path)


class DomainOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, batch, *, score_raw=False):
        self.calls.append(score_raw)
        mask = batch.batch["response_mask"]
        value = torch.full_like(mask, -9.0 if score_raw else -0.5) * mask
        return TokenScores(value, torch.zeros_like(mask), mask)


def test_distributed_preparation_keeps_raw_reference_separate():
    from espnet2.rl.data import collate_rollouts

    sources = [{"id": "q"}]
    generations = [
        {
            "source_id": "q",
            "prompt_token_ids": [1],
            "token_ids": [2, 3],
            "log_probs": [-0.5, -0.5],
        }
    ]
    batch = collate_rollouts(sources, generations, 1)
    trainer = NativeGRPOTrainer.__new__(NativeGRPOTrainer)
    trainer.rank, trainer.world_size = 0, 1
    trainer.config = SimpleNamespace(
        training_objective="west_grpo", micro_batch_size=1, micro_batch_tokens=None
    )
    trainer.policy, trainer.reference = DomainOracle(), DomainOracle()
    old, behavior, reference = score_native_batch(trainer, batch)
    assert old.eq(-0.5).all() and torch.equal(old, behavior)
    assert reference.eq(-9.0).all()
    assert trainer.policy.calls == [False] and trainer.reference.calls == [True]
