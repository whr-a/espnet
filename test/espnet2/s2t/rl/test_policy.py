"""Scoring contracts for the native OWSM post-training actor."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from espnet2.s2t.rl.policy import OWSMPolicy, build_policy, load_audio


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, memory, memory_lengths, inputs, lengths):
        vocabulary = torch.arange(7, device=inputs.device).float()
        logits = -((vocabulary - inputs[..., None]).abs()) + self.weight
        return logits, lengths


def test_native_alignment_uses_last_prompt_position():
    model = nn.Module()
    model.decoder = Decoder()
    model.token_list = []
    model.ctc = None
    model.eos = 0
    policy = OWSMPolicy(model)
    batch = SimpleNamespace(
        batch={
            "responses": torch.tensor([[3, 2]]),
            "response_mask": torch.ones(1, 2),
            "response_lengths": torch.tensor([2]),
        },
        non_tensor_batch={
            "source": [{"encoder_features": np.zeros((2, 1))}],
            "prompt_token_ids": [[1, 3]],
        },
    )
    scores = policy(batch)
    expected = torch.log_softmax(-torch.abs(torch.arange(7).float() - 3), 0)
    assert scores.log_probs[0, 0].item() == pytest.approx(expected[3].item())
    assert scores.log_probs[0, 1].item() == pytest.approx(expected[2].item())


def test_factory_does_not_recursively_dispatch_itself(monkeypatch):
    sentinel = nn.Linear(1, 1)
    monkeypatch.setattr(OWSMPolicy, "from_config", lambda config, device: sentinel)
    config = SimpleNamespace(adapter_factory="espnet2.s2t.rl.policy:build_policy")
    assert build_policy(config, "cpu") is sentinel
    assert not sentinel.training


def test_audio_window_rejects_truncation():
    with pytest.raises(ValueError, match="exceeding"):
        load_audio({"audio_array": [1.0, 2.0], "sample_rate": 16000}, 16000, 1)
    padded = load_audio({"audio_array": [1.0], "sample_rate": 16000}, 16000, 3)
    np.testing.assert_array_equal(padded, [1.0, 0.0, 0.0])
