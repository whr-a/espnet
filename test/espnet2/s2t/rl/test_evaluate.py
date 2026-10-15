"""Batched greedy heldout decoding matches one utterance at a time."""

from types import SimpleNamespace

import torch
from torch import nn

from espnet2.s2t.rl.evaluate import greedy_decode

EOS = 0


class CountingDecoder(nn.Module):
    """Counts up from the last token until the utterance's target length."""

    def forward(self, memory, memory_lengths, tokens, lengths):
        targets = memory[:, 0, 0].long()
        positions = torch.arange(tokens.size(1))
        following = tokens % 9 + 1
        finished = positions[None, :] + 1 >= targets[:, None]
        chosen = torch.where(finished, torch.full_like(tokens, EOS), following)
        return nn.functional.one_hot(chosen, 10).float(), lengths


class CountingPolicy(nn.Module):
    sample_rate = 16000
    window_samples = 2

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(decoder=CountingDecoder(), eos=EOS)

    def _encode(self, speech, lengths):
        memory = speech[:, :1, None].expand(-1, 3, 1)
        return memory, torch.full_like(lengths, 3)

    def build_prefix(self, source, config):
        return [1]


def test_batched_greedy_matches_single_utterances():
    policy = CountingPolicy()
    sources = [
        {"audio_array": [length], "sample_rate": 16000} for length in (4, 2, 6, 3)
    ]
    batched = greedy_decode(policy, sources, None, max_tokens=8, batch_size=3)
    single = greedy_decode(policy, sources, None, max_tokens=8, batch_size=1)
    assert batched == single
    assert batched == [[2, 3, 4, EOS], [2, EOS], [2, 3, 4, 5, 6, EOS], [2, 3, EOS]]


def test_greedy_stops_at_the_token_budget():
    policy = CountingPolicy()
    sources = [{"audio_array": [9], "sample_rate": 16000}]
    assert greedy_decode(policy, sources, None, max_tokens=3) == [[2, 3, 4]]
