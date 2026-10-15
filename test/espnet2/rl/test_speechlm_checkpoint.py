"""Check recomputation against the ordinary native scores and gradients."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from espnet2.speechlm.model.speechlm.rl.bagpiper.policy import (
    NativeSpeechLMPolicy,
    checkpoint_native_layers,
    offload_saved_activations,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.tokens import (
    ASSISTANT,
    CODEC,
    EOS,
    TEXT,
)


class ScoreCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()
        self._embed = lambda *args: None
        self.num_stream = 4
        self.stream_emb = nn.Embedding(4, 7)
        self.lm_head = nn.Linear(7, 64, bias=False)


def test_chunk_checkpoint_preserves_joint_distribution_and_gradients():
    torch.manual_seed(13)
    core = ScoreCore()
    with torch.no_grad():
        core.lm_head.weight.normal_(std=0.01)
        core.stream_emb.weight.normal_(std=0.001)
        for time, target in enumerate((20, 20, 20, 40, 40, 40, 20)):
            core.lm_head.weight[target, time] = 3
    geometry = dict(
        text_token_offset=12,
        text_token_end=40,
        codec_base_offset=40,
        codec_layer_size=4,
        text_token_id=7,
        audio_token_id=8,
        eot_token_id=3,
        assistant_token_id=6,
        eos_token_id=2,
    )
    labels = torch.tensor(
        [
            [
                [20, 0, 0, 0],
                [3, 0, 0, 0],
                [6, 0, 0, 0],
                [41, 45, 49, 53],
                [42, 46, 50, 54],
                [3, 45, 49, 53],
                [2, 0, 0, 0],
            ]
        ]
    )
    masks = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ]
        ],
        dtype=torch.bool,
    )
    source = dict(
        mode="text_audio",
        sampling_extra_args=dict(phase="text"),
        sampling_contracts=[
            [dict(temperature=1.0, top_k=64) for _ in range(4)] for _ in range(7)
        ],
        support=[TEXT, TEXT, ASSISTANT, CODEC, CODEC, CODEC, EOS],
    )
    results = []
    for recompute in (False, True):
        model = copy.deepcopy(core)
        policy = NativeSpeechLMPolicy(
            model,
            token_chunk_size=2,
            bagpiper_config=geometry,
            score_checkpointing=recompute,
        )
        hidden = torch.eye(7)[None].requires_grad_()
        lp, entropy, per_stream = policy._score_frames(
            hidden, labels, masks, source, True
        )
        (lp.sum() + 0.03 * entropy.sum()).backward()
        results.append(
            (
                lp.detach(),
                entropy.detach(),
                per_stream.detach(),
                hidden.grad,
                {n: p.grad for n, p in model.named_parameters()},
            )
        )
    for plain, recomputed in zip(results[0][:-1], results[1][:-1], strict=True):
        torch.testing.assert_close(plain, recomputed, rtol=1e-6, atol=1e-6)
    for name in results[0][-1]:
        torch.testing.assert_close(
            results[0][-1][name], results[1][-1][name], rtol=1e-6, atol=1e-6
        )


def test_native_layer_checkpoint_preserves_eval_and_backbone_gradient():
    torch.manual_seed(17)
    model = Qwen3Model(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            attention_dropout=0.3,
        )
    )
    model.eval()
    results = []
    for recompute in (False, True):
        replica = copy.deepcopy(model)
        embeddings = torch.randn(
            1, 9, 16, generator=torch.Generator().manual_seed(19)
        ).requires_grad_()
        with checkpoint_native_layers(replica, recompute):
            assert not replica.training and all(
                not layer.training for layer in replica.layers
            )
            output = replica(
                inputs_embeds=embeddings, use_cache=False
            ).last_hidden_state
        assert all("forward" not in layer.__dict__ for layer in replica.layers)
        output.square().sum().backward()
        results.append(
            (
                output.detach(),
                embeddings.grad,
                {n: p.grad for n, p in replica.named_parameters()},
            )
        )
    for plain, recomputed in zip(results[0][:-1], results[1][:-1], strict=True):
        torch.testing.assert_close(plain, recomputed, rtol=1e-5, atol=1e-5)
    for name, grad in results[0][-1].items():
        if grad is not None:
            torch.testing.assert_close(grad, results[1][-1][name], rtol=1e-5, atol=1e-5)


def text_only_policy():
    core = ScoreCore()
    with torch.no_grad():
        core.lm_head.weight.zero_()
        core.stream_emb.weight.zero_()
        core.lm_head.weight[20, 0] = 3
    geometry = dict(
        text_token_offset=12,
        text_token_end=40,
        codec_base_offset=40,
        codec_layer_size=4,
        text_token_id=7,
        audio_token_id=8,
        eot_token_id=3,
        assistant_token_id=6,
        eos_token_id=2,
    )
    policy = NativeSpeechLMPolicy(
        core, token_chunk_size=2, bagpiper_config=geometry, score_checkpointing=False
    )
    source = dict(
        mode="text_only",
        sampling_contracts=[[dict(temperature=1.0, top_k=64) for _ in range(4)]],
        support=[TEXT],
    )
    hidden = torch.zeros(1, 1, 7, requires_grad=True)
    hidden.data[..., 0] = 1
    labels = torch.tensor([[[20, 0, 0, 0]]])
    masks = torch.tensor([[[True, False, False, False]]])
    return policy, source, hidden, labels, masks


def test_inactive_codec_stream_preserves_zero_gradient_and_adam_state():
    policy, source, hidden, labels, masks = text_only_policy()
    before = policy.core.stream_emb.weight.detach().clone()
    logp, entropy = policy._score_frames(hidden, labels, masks, source)
    (logp.sum() + 0.03 * entropy.sum()).backward()
    assert policy.core.stream_emb.weight.grad is not None
    assert torch.count_nonzero(policy.core.stream_emb.weight.grad) == 0
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    optimizer.step()
    assert torch.equal(policy.core.stream_emb.weight, before)
    state = optimizer.state[policy.core.stream_emb.weight]
    assert state["step"] == 1
    assert torch.count_nonzero(state["exp_avg"]) == 0


@pytest.mark.parametrize("setting,value", [("temperature", 0.5), ("top_k", 3)])
def test_inactive_codec_stream_rejects_changed_sampling_contract(setting, value):
    policy, source, hidden, labels, masks = text_only_policy()
    source["sampling_contracts"][0][1][setting] = value
    with pytest.raises(ValueError, match="untruncated unit-temperature"):
        policy._score_frames(hidden, labels, masks, source)


def test_text_only_codec_argmax_keeps_the_native_text_support():
    policy, source, hidden, labels, masks = text_only_policy()
    with torch.no_grad():
        policy.core.lm_head.weight[41, 0] = 4
        policy._score_frames(hidden, labels, masks, source)
    assert (policy.support_mismatches, policy.support_frames) == (0, 1)
    labels[0, 0, 0] = 41
    with pytest.raises(ValueError, match="zero likelihood"):
        policy._score_frames(hidden, labels, masks, source)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_all_forced_block_preserves_zero_head_gradient(checkpoint):
    policy, source, hidden, labels, masks = text_only_policy()
    policy.score_checkpointing = checkpoint
    policy.core.lm_head.bias = nn.Parameter(torch.zeros(64))
    source.update(
        mode="text_audio",
        sampling_extra_args=dict(phase="transition"),
        support=[ASSISTANT],
    )
    labels[0, 0, 0] = 6
    masks[:] = False
    logp, entropy = policy._score_frames(hidden, labels, masks, source)
    (logp.sum() + 0.03 * entropy.sum()).backward()
    for parameter in (
        policy.core.lm_head.weight,
        policy.core.lm_head.bias,
        policy.core.stream_emb.weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA storage-view test")
def test_activation_offload_keeps_parameter_storage_views_on_cuda():
    model = nn.Linear(3, 4, bias=False).cuda()
    active = torch.ones(1, 3, device="cuda", requires_grad=True)
    observed = {}

    class SavedViews(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, weight):
            ctx.save_for_backward(value, weight.T)
            return value.sum()

        @staticmethod
        def backward(ctx, gradient):
            value, weight_view = ctx.saved_tensors
            observed["activation_device"] = value.device
            observed["weight_storage"] = weight_view.untyped_storage().data_ptr()
            observed["weight_device"] = weight_view.device
            return gradient.expand_as(value), None

    with offload_saved_activations(model, True):
        result = SavedViews.apply(active, model.weight)
    result.backward()
    assert observed["activation_device"] == active.device
    assert observed["weight_device"] == model.weight.device
    assert observed["weight_storage"] == model.weight.untyped_storage().data_ptr()
    assert torch.equal(active.grad, torch.ones_like(active))
