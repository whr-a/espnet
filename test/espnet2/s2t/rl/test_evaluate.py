"""Batched greedy heldout decoding, and the sampled decode beside it."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from espnet2.s2t.rl.evaluate import (
    greedy_decode,
    restrict,
    row_seed,
    sampled_decode,
    score_samples,
    select_subset,
)

EOS = 0
VOCAB = 10


class CountingDecoder(nn.Module):
    """Counts up from the last token until the utterance's target length."""

    def forward(self, memory, memory_lengths, tokens, lengths):
        targets = memory[:, 0, 0].long()
        positions = torch.arange(tokens.size(1))
        following = tokens % 9 + 1
        finished = positions[None, :] + 1 >= targets[:, None]
        chosen = torch.where(finished, torch.full_like(tokens, EOS), following)
        return nn.functional.one_hot(chosen, VOCAB).float(), lengths


class SpreadDecoder(nn.Module):
    """Three unequal continuations, then EOS at the utterance's target length.

    Unlike CountingDecoder its next-token mass is spread, so a sampled decode
    genuinely branches while argmax stays pinned to token 2.
    """

    def forward(self, memory, memory_lengths, tokens, lengths):
        targets = memory[:, 0, 0].long()
        positions = torch.arange(tokens.size(1))
        finished = positions[None, :] + 1 >= targets[:, None]
        logits = torch.zeros(*tokens.shape, VOCAB)
        logits[..., 2], logits[..., 3] = 2.0, 1.0
        stop = torch.full((VOCAB,), -20.0)
        stop[EOS] = 10.0
        return torch.where(finished[..., None], stop, logits), lengths


class CountingPolicy(nn.Module):
    sample_rate = 16000
    window_samples = 2
    decoder_class = CountingDecoder

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(decoder=self.decoder_class(), eos=EOS)

    def _encode(self, speech, lengths):
        memory = speech[:, :1, None].expand(-1, 3, 1)
        return memory, torch.full_like(lengths, 3)

    def build_prefix(self, source, config):
        return [1]


class SpreadPolicy(CountingPolicy):
    decoder_class = SpreadDecoder


def utterances(lengths):
    return [{"audio_array": [length], "sample_rate": 16000} for length in lengths]


def test_batched_greedy_matches_single_utterances():
    policy = CountingPolicy()
    sources = utterances((4, 2, 6, 3))
    batched = greedy_decode(policy, sources, None, max_tokens=8, batch_size=3)
    single = greedy_decode(policy, sources, None, max_tokens=8, batch_size=1)
    assert batched == single
    assert batched == [[2, 3, 4, EOS], [2, EOS], [2, 3, 4, 5, 6, EOS], [2, 3, EOS]]


def test_greedy_stops_at_the_token_budget():
    policy = CountingPolicy()
    sources = utterances((9,))
    assert greedy_decode(policy, sources, None, max_tokens=3) == [[2, 3, 4]]


@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_zero_temperature_reproduces_greedy_token_for_token(batch_size):
    """The sampled path at temperature 0 is the existing greedy evidence path."""
    policy = CountingPolicy()
    sources = utterances((4, 2, 6, 3))
    expected = greedy_decode(policy, sources, None, max_tokens=8, batch_size=32)
    assert (
        sampled_decode(
            policy,
            sources,
            None,
            max_tokens=8,
            batch_size=batch_size,
            temperature=0.0,
            samples=1,
        )
        == expected
    )


def test_zero_temperature_repeats_the_same_draw_for_every_sample():
    policy = CountingPolicy()
    sources = utterances((4, 2))
    greedy = greedy_decode(policy, sources, None, max_tokens=8)
    sampled = sampled_decode(
        policy, sources, None, max_tokens=8, temperature=0.0, samples=3
    )
    # Rows are source-major: three identical draws per utterance.
    assert sampled == [greedy[0]] * 3 + [greedy[1]] * 3


def test_unit_temperature_is_reproducible_and_seed_dependent():
    policy = SpreadPolicy()
    sources = utterances((6, 6, 6, 6))
    draw = dict(max_tokens=8, temperature=1.0, samples=4)
    first = sampled_decode(policy, sources, None, seed=7, **draw)
    assert sampled_decode(policy, sources, None, seed=7, **draw) == first
    assert sampled_decode(policy, sources, None, seed=8, **draw) != first
    # A spread distribution must actually leave the greedy mode somewhere.
    assert first != sampled_decode(
        policy, sources, None, temperature=0.0, samples=4, max_tokens=8
    )


@pytest.mark.parametrize("batch_size", [1, 2, 5, 64])
def test_sampling_does_not_depend_on_the_batch_shape(batch_size):
    """Each row owns its stream, so re-running at another batch size agrees."""
    policy = SpreadPolicy()
    sources = utterances((5, 7, 4, 6, 5))
    reference = sampled_decode(
        policy,
        sources,
        None,
        max_tokens=8,
        batch_size=16,
        temperature=1.0,
        samples=2,
        seed=3,
    )
    assert (
        sampled_decode(
            policy,
            sources,
            None,
            max_tokens=8,
            batch_size=batch_size,
            temperature=1.0,
            samples=2,
            seed=3,
        )
        == reference
    )


def test_every_prompt_gets_the_requested_number_of_draws():
    policy = SpreadPolicy()
    sources = utterances((5, 7, 4))
    rows = sampled_decode(
        policy, sources, None, max_tokens=8, temperature=1.0, samples=5, seed=1
    )
    assert len(rows) == len(sources) * 5


def test_row_seeds_are_distinct_per_prompt_and_draw():
    seeds = {row_seed(0, index, draw) for index in range(16) for draw in range(4)}
    assert len(seeds) == 64
    assert row_seed(0, 1, 0) != row_seed(1, 0, 0) != row_seed(0, 0, 1)


def test_sampling_needs_at_least_one_draw():
    with pytest.raises(ValueError, match="at least one draw"):
        sampled_decode(SpreadPolicy(), utterances((4,)), None, samples=0)


def test_top_k_one_collapses_sampling_onto_the_mode():
    policy = SpreadPolicy()
    sources = utterances((5, 6))
    greedy = sampled_decode(
        policy, sources, None, max_tokens=8, temperature=0.0, samples=2
    )
    assert (
        sampled_decode(
            policy,
            sources,
            None,
            max_tokens=8,
            temperature=1.0,
            samples=2,
            seed=5,
            top_k=1,
        )
        == greedy
    )


def test_restrict_keeps_the_full_support_at_the_training_settings():
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    assert torch.equal(restrict(probabilities, 1.0, -1), probabilities)


def test_restrict_drops_the_tail_outside_top_p_and_top_k():
    probabilities = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    assert torch.equal(
        restrict(probabilities, 0.7, -1), torch.tensor([[0.5, 0.3, 0.0, 0.0]])
    )
    assert torch.equal(
        restrict(probabilities, 1.0, 2), torch.tensor([[0.5, 0.3, 0.0, 0.0]])
    )


def test_sampled_draws_follow_the_decoder_distribution():
    """Many draws of one token land near softmax over the spread logits."""
    policy = SpreadPolicy()
    # Length 9 exceeds the budget, so no row stops and token 1 is never EOS.
    rows = sampled_decode(
        policy,
        utterances((9,)) * 50,
        None,
        max_tokens=1,
        temperature=1.0,
        samples=20,
        seed=11,
    )
    first = [row[0] for row in rows]
    total = math.exp(2) + math.exp(1) + 8
    assert len(first) == 1000
    assert abs(first.count(2) / len(first) - math.exp(2) / total) < 0.05
    assert abs(first.count(3) / len(first) - math.exp(1) / total) < 0.05


def scored(samples, texts, eos=None):
    sources = [
        {"id": "a", "reference": "one two three four"},
        {"id": "b", "reference": "five six"},
    ]
    generations = [[1]] * (len(sources) * samples)
    if eos is None:
        eos = [True] * len(generations)
    return score_samples(sources, texts, generations, eos, samples)


def test_score_samples_counts_draws_denominator_and_failures():
    rows, prompts, totals = scored(
        2,
        [
            "one two three four",  # a, draw 0: perfect
            "one two three",  # a, draw 1: one deletion
            "five six",  # b, draw 0: perfect
            "nine ten",  # b, draw 1: two substitutions
        ],
        eos=[True, False, True, True],
    )
    assert [row["id"] for row in rows] == ["a", "a", "b", "b"]
    assert [row["draw"] for row in rows] == [0, 1, 0, 1]
    assert [row["errors"] for row in rows] == [0, 1, 0, 2]
    assert [prompt["errors"] for prompt in prompts] == [[0, 1], [0, 2]]
    assert [prompt["best_errors"] for prompt in prompts] == [0, 0]
    assert [prompt["worst_errors"] for prompt in prompts] == [1, 2]
    assert [prompt["mean_errors"] for prompt in prompts] == [0.5, 1.0]
    assert [prompt["eos_failures"] for prompt in prompts] == [1, 0]
    assert totals["prompt_count"] == 2
    assert totals["samples_per_prompt"] == 2
    assert totals["sample_count"] == 4
    # Six reference words per pass, counted once per draw in the pooled WER.
    assert totals["reference_words_per_draw"] == 6
    assert totals["total_reference_words"] == 12
    assert totals["total_errors"] == 3
    assert totals["eos_failures"] == 1
    assert totals["corpus_wer"] == pytest.approx(3 / 12)
    assert totals["corpus_wer_best"] == pytest.approx(0 / 6)
    assert totals["corpus_wer_worst"] == pytest.approx(3 / 6)


def test_pooled_sampled_wer_equals_the_mean_over_draws():
    _, _, totals = scored(
        4, ["one two three four", "one two", "one", "one two three"] + ["five six"] * 4
    )
    assert totals["corpus_wer"] == pytest.approx(
        totals["total_errors"] / totals["total_reference_words"]
    )


def test_one_draw_reduces_to_the_single_pass_denominator():
    _, _, totals = scored(1, ["one two three", "five six"])
    assert totals["total_reference_words"] == totals["reference_words_per_draw"] == 6
    assert totals["corpus_wer"] == totals["corpus_wer_best"] == pytest.approx(1 / 6)


def test_select_subset_strides_and_caps_reproducibly():
    sources = [{"id": str(index)} for index in range(10)]
    assert [s["id"] for s in select_subset(sources, 3, 0)] == ["0", "3", "6", "9"]
    assert [s["id"] for s in select_subset(sources, 3, 2)] == ["0", "3"]
    assert select_subset(sources, 1, 0) == sources
    with pytest.raises(ValueError, match="at least 1"):
        select_subset(sources, 0, 0)
    with pytest.raises(ValueError, match="no prompts"):
        select_subset([], 1, 0)
