"""Analytical native action probabilities including secondary audio conditions."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from espnet2.rl.batch import RolloutBatch
from espnet2.speechlm.rl.model import OriginalOpusARDelay
from espnet2.speechlm.rl.policy import NativeSpeechLMPolicy


class IdentityDecoder(nn.Module):
    def forward(self, inputs_embeds, use_cache=False):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def test_secondary_audio_condition_and_unsampled_heads_gradient():
    core = OriginalOpusARDelay.__new__(OriginalOpusARDelay)
    nn.Module.__init__(core)
    core.nq, core.pad_id = 9, 0
    core.emb = nn.Embedding(4, 2)
    core.lm_head = nn.Linear(2, 4, bias=False)
    core.head_emb = nn.Embedding(12, 2)
    with torch.no_grad():
        core.emb.weight.copy_(
            torch.tensor([[9.0, 9.0], [1.0, 0.0], [0.0, 2.0], [0.5, -0.5]])
        )
        core.lm_head.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
        )
        core.head_emb.weight.zero_()
        core.head_emb.weight[0].copy_(torch.tensor([0.25, 0.5]))
    core.decoders = nn.Module()
    core.decoders.model_type = "huggingface"
    core.decoders.model = IdentityDecoder()
    model = nn.Module()
    model.corelm = core
    policy = NativeSpeechLMPolicy(model)
    contract = [
        [{"temperature": 1.0, "top_k": 0, "allowed_ranges": [[1, 4]]}] + [{}] * 8
    ]
    source = {"sampling_contracts": contract}
    prefixes = np.empty(1, dtype=object)
    prefixes[0] = [[1, 2] + [0] * 7]
    batch = RolloutBatch(
        {
            "responses": torch.tensor([[[2] + [0] * 8]]),
            "response_mask": torch.ones((1, 1)),
            "response_stream_mask": torch.tensor([[[True] + [False] * 8]]),
            "response_lengths": torch.tensor([1]),
        },
        {"prompt_token_ids": prefixes, "source": np.array([source], dtype=object)},
    )
    result = policy(batch)
    # All pad embeddings are removed. The secondary input changes the sampled
    # primary distribution; unsampled output heads remain absent from the loss.
    hidden = torch.tensor([1.25, 2.5])
    logits = torch.tensor([2.5, 3.75, -1.25])
    expected = logits[1] - logits.logsumexp(0)
    torch.testing.assert_close(result.log_probs[0, 0], expected)
    (-result.log_probs.sum()).backward()
    probabilities = logits.softmax(0)
    allowed_heads = core.lm_head.weight.detach()[1:]
    analytical = (probabilities[:, None] * allowed_heads).sum(0) - allowed_heads[1]
    torch.testing.assert_close(core.emb.weight.grad[1], analytical)
    torch.testing.assert_close(core.emb.weight.grad[2], analytical)
    torch.testing.assert_close(core.emb.weight.grad[0], torch.zeros(2))
    torch.testing.assert_close(core.head_emb.weight.grad[0], analytical)
    assert not core.head_emb.weight.grad[1:].any()
    assert result.stream_log_probs[0, 0, 1:].eq(0).all()
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_released_delay_layout_and_lengths():
    core = OriginalOpusARDelay.__new__(OriginalOpusARDelay)
    nn.Module.__init__(core)
    core.nq = 3
    source = torch.tensor([[[1, 10, 100], [2, 20, 200]]])
    output, lengths = core.delay_interleave(source, torch.tensor([2]), pad=-1)
    expected = torch.tensor([[[1, -1, -1], [2, 10, -1], [-1, 20, 100], [-1, -1, 200]]])
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(lengths, torch.tensor([4]))


def test_half_precision_backbone_config_still_loads_checkpoints_exactly():
    from transformers import LlamaConfig

    geometry = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    # As AutoConfig reads it from a released bfloat16 backbone.
    geometry.dtype = torch.bfloat16
    core = OriginalOpusARDelay(geometry, ["<pad>", "<sos/eos>", "a", "b"], 9)
    assert {parameter.dtype for parameter in core.parameters()} == {torch.float32}
    state = {key: torch.randn_like(value) for key, value in core.state_dict().items()}
    core.load_state_dict(state, strict=True)
    # A saved policy reloads bit for bit, as the rollout engine loads it.
    assert all(torch.equal(core.state_dict()[key], state[key]) for key in state)
