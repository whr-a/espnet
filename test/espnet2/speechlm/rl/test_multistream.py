"""OpusLM's nine-stream codec rollout, on the CPU.

Covers the frame geometry both halves of the loop must agree on (delay ramp,
per-stream supports), that forced pad positions score exactly zero, that the
engine's record satisfies collate_rollouts' multistream checks, and that a
secondary stream really conditions the next frame's primary probability with
the gradient that implies.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.speechlm.rl.multistream import (
    active_streams,
    block_finished,
    codec_supports,
    deinterleave_codec,
)
from espnet2.speechlm.rl.policy import NativeSpeechLMPolicy
from espnet2.speechlm.rl.tokens import selected_log_probs

NQ, SIZE = 9, 4
# A miniature of the released layout: SSL, then eight codec slices, then text.
METADATA = {
    "rl_inference_semantics": "native_pad_cleared",
    "nq": NQ,
    "pad_token_id": 0,
    "eos_token_id": 5,
    "codec_ssl_start_end_token_id": 7,
    "text_bpe_start_end_token_id": 8,
    "ssl_token_start": 10,
    "ssl_token_end": 20,
    "codec_token_start": 20,
    "codec_per_stream_size": SIZE,
    "codec_token_end": 20 + (NQ - 1) * SIZE,
    "codec_per_stream_count": NQ - 1,
    "text_token_start": 52,
    "text_token_end": 60,
    "vocab_size": 60,
    "codec_ssl_plain_tts_task_token_id": 82 % 60,
}
VOCAB = METADATA["vocab_size"]


# --------------------------------------------------------------------------
# frame geometry
# --------------------------------------------------------------------------


def test_delay_ramp_brings_one_stream_online_per_frame():
    for step in range(NQ):
        assert active_streams(NQ, step, None) == [True] * (step + 1) + [False] * (
            NQ - step - 1
        )
    # Past the ramp every stream is real.
    assert active_streams(NQ, NQ + 5, None) == [True] * NQ


def test_each_stream_flushes_exactly_its_own_delay_after_the_block_closes():
    closed = 20
    # At the closing frame stream 0 is spent but every codec stream still
    # carries the code of an audio frame that exists.
    assert active_streams(NQ, closed, closed) == [False] + [True] * (NQ - 1)
    for offset in range(1, NQ - 1):
        active = active_streams(NQ, closed + offset, closed)
        assert active[0] is False
        # Stream q survives while offset < q.
        assert active[1:] == [offset < stream for stream in range(1, NQ)]
        assert not block_finished(NQ, closed + offset, closed)
    assert active_streams(NQ, closed + NQ - 1, closed) == [False] * NQ
    assert block_finished(NQ, closed + NQ - 1, closed)


def test_supports_tile_the_codec_range_one_slice_per_stream():
    supports = codec_supports(METADATA)
    assert len(supports) == NQ
    # Stream 0 is the SSL semantic token plus the marker that closes a block.
    assert supports[0]["allowed_ranges"] == [[10, 20]]
    assert supports[0]["allowed_token_ids"] == [7]
    covered = []
    for stream in range(1, NQ):
        ((low, high),) = supports[stream]["allowed_ranges"]
        assert high - low == SIZE
        assert "allowed_token_ids" not in supports[stream]
        covered.append((low, high))
    assert covered[0][0] == METADATA["codec_token_start"]
    assert covered[-1][1] == METADATA["codec_token_end"]
    # Slices are contiguous and disjoint.
    assert all(covered[i][1] == covered[i + 1][0] for i in range(len(covered) - 1))


def test_a_layout_whose_slices_do_not_tile_the_codec_range_is_refused():
    broken = dict(METADATA, codec_token_end=METADATA["codec_token_end"] + 1)
    with pytest.raises(ValueError, match="tile"):
        codec_supports(broken)


# --------------------------------------------------------------------------
# de-interleaving for the reward side
# --------------------------------------------------------------------------


def delayed_block(length, codes=None):
    """A complete sampled block: `length` audio frames plus the delay flush."""
    base = METADATA["codec_token_start"]
    frames = [[0] * NQ for _ in range(length + NQ - 1)]
    for audio in range(length):
        frames[audio][0] = METADATA["ssl_token_start"] + audio % 10
        for stream in range(1, NQ):
            value = (
                codes[stream - 1][audio] if codes is not None else (audio + stream) % SIZE
            )
            frames[audio + stream][stream] = base + (stream - 1) * SIZE + value
    frames[length][0] = METADATA["codec_ssl_start_end_token_id"]
    return frames


def test_deinterleave_recovers_the_codes_the_delay_hid():
    codes = [[(audio * 3 + stream) % SIZE for audio in range(6)] for stream in range(NQ - 1)]
    recovered, status = deinterleave_codec(delayed_block(6, codes), METADATA)
    assert status == "ok"
    assert recovered.shape == (NQ - 1, 6)
    assert recovered.tolist() == codes


def test_a_truncated_block_yields_only_its_complete_audio_frames():
    # Cut off while the block was still open: nothing closed it, so the audio
    # is as long as the slowest stream got.
    frames = delayed_block(20)[:12]
    recovered, status = deinterleave_codec(frames, METADATA)
    assert status == "ok"
    assert recovered.shape == (NQ - 1, 12 - (NQ - 1))


def test_a_block_cut_off_mid_flush_keeps_the_frames_that_did_arrive():
    """Stream 0 closed at frame 6, declaring six audio frames, but the request
    ended before the flush delivered them. Only the arrived ones are real."""
    frames = delayed_block(6)[:10]
    recovered, status = deinterleave_codec(frames, METADATA)
    assert status == "ok"
    assert recovered.shape == (NQ - 1, 10 - (NQ - 1))


def test_a_complete_block_uses_the_length_its_close_marker_declares():
    frames = delayed_block(6)
    recovered, status = deinterleave_codec(frames, METADATA)
    assert status == "ok"
    assert recovered.shape == (NQ - 1, 6)


def test_a_block_shorter_than_the_ramp_has_no_audio():
    assert deinterleave_codec(delayed_block(6)[:NQ - 2], METADATA)[1] == "no_audio"


def test_a_code_outside_its_own_layer_is_never_clamped():
    frames = delayed_block(6)
    # Put stream 2's id where stream 1's belongs.
    frames[1][1] = METADATA["codec_token_start"] + SIZE
    assert deinterleave_codec(frames, METADATA)[1] == "invalid_codec"


# --------------------------------------------------------------------------
# forced positions score zero
# --------------------------------------------------------------------------


def test_an_inactive_stream_scores_exactly_zero_whatever_its_logits():
    logits = torch.randn(1, 3, VOCAB, generator=torch.Generator().manual_seed(0))
    labels = torch.zeros(1, 3, dtype=torch.long)  # pad id
    mask = torch.tensor([[True, False, False]])
    logp, entropy = selected_log_probs(logits, labels, mask)
    assert logp[0, 1] == 0.0 and logp[0, 2] == 0.0
    assert entropy[0, 1] == 0.0 and entropy[0, 2] == 0.0
    # The active position is a real score.
    assert logp[0, 0] != 0.0


def test_a_fully_forced_frame_does_not_poison_the_other_frames():
    # Every stream masked out: no NaN, no exception, all zero.
    logits = torch.randn(1, 2, VOCAB, generator=torch.Generator().manual_seed(1))
    logits = logits.masked_fill(
        ~torch.zeros(VOCAB, dtype=torch.bool), -torch.inf
    )
    labels = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    logp, entropy = selected_log_probs(logits, labels, mask)
    assert torch.equal(logp, torch.zeros(1, 2))
    assert torch.isfinite(entropy).all()


# --------------------------------------------------------------------------
# the learner's contract
# --------------------------------------------------------------------------


def multistream_generation(source_id, length=4):
    """An engine record of a closed codec block, in the rollout schema."""
    frames = delayed_block(length)
    supports = codec_supports(METADATA)
    masks, stream_scores, scores, contracts = [], [], [], []
    closed = length
    for step, frame in enumerate(frames):
        active = active_streams(NQ, step, None if step < closed else closed)
        if step == closed:
            # Stream 0's closing marker is a real action, recorded with the
            # mask computed before the block closed.
            active = active_streams(NQ, step, None)
        per_stream = [
            -0.25 * (stream + 1) if active[stream] else 0.0 for stream in range(NQ)
        ]
        masks.append(active)
        stream_scores.append(per_stream)
        scores.append(sum(per_stream))
        contracts.append([dict(spec) for spec in supports])
    return {
        "source_id": source_id,
        "token_ids": frames,
        "log_probs": scores,
        "stream_log_probs": stream_scores,
        "stream_mask": masks,
        "action_mask": [any(mask) for mask in masks],
        "codec_frame_mask": [bool(mask[1]) for mask in masks],
        "sampling_contracts": contracts,
        "prompt_token_ids": [[METADATA["eos_token_id"]] + [0] * (NQ - 1)],
        "text": "",
    }


def test_collate_accepts_the_engine_record_and_keeps_every_stream():
    sources = [{"id": "a"}, {"id": "b"}]
    generations = [
        multistream_generation(source["id"]) for source in sources for _ in range(2)
    ]
    batch = collate_rollouts(sources, generations, 2)
    responses = batch.batch["responses"]
    assert responses.ndim == 3 and responses.shape[-1] == NQ
    stream_mask = batch.batch["response_stream_mask"]
    assert stream_mask.shape == responses.shape
    # collate's own invariant, restated: an active frame is one with a stream.
    assert torch.equal(
        batch.batch["response_mask"].bool(), stream_mask.bool().any(-1)
    )
    # The per-frame behaviour probability is the sum over streams.
    expected = torch.tensor(generations[0]["log_probs"])
    assert torch.allclose(
        batch.batch["rollout_log_probs"][0, : len(expected)], expected, atol=1e-6
    )
    # The contracts travel to the learner on the source.
    contracts = batch.non_tensor_batch["source"][0]["sampling_contracts"]
    assert len(contracts) == len(generations[0]["token_ids"])
    assert all(len(row) == NQ for row in contracts)


def test_collate_rejects_a_mask_that_disagrees_with_the_frames():
    generation = multistream_generation("a")
    generation["action_mask"][0] = False
    with pytest.raises(ValueError, match="action_mask differs"):
        collate_rollouts([{"id": "a"}], [generation], 1)


def test_collate_rejects_a_multistream_rollout_without_a_stream_mask():
    generation = multistream_generation("a")
    generation["stream_mask"] = None
    with pytest.raises(ValueError, match="sampling mask"):
        collate_rollouts([{"id": "a"}], [generation], 1)


# --------------------------------------------------------------------------
# a secondary stream really conditions the primary, with the right gradient
# --------------------------------------------------------------------------


class TinyCore(nn.Module):
    """The released ARDelay contract in miniature: pad-cleared stream sum in,
    per-stream head_emb offset on the way out."""

    def __init__(self, hidden=6):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.nq, self.pad_id = NQ, METADATA["pad_token_id"]
        self.emb = nn.Embedding(VOCAB, hidden)
        self.lm_head = nn.Linear(hidden, VOCAB, bias=False)
        self.head_emb = nn.Embedding(12, hidden)
        for parameter in self.parameters():
            with torch.no_grad():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator) * 0.3
                )
        self.decoders = nn.Module()
        self.decoders.model_type = "huggingface"
        self.decoders.model = _Causal(hidden, generator)

    def process_embedding(self, sequence):
        embeddings = self.emb(sequence)
        return embeddings.masked_fill((sequence == self.pad_id)[..., None], 0).sum(2)


class _Causal(nn.Module):
    """A causal map: frame t sees the prefix sum up to t, so a change at t
    moves every later frame and no earlier one."""

    def __init__(self, hidden, generator):
        super().__init__()
        self.mix = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.mix.weight.copy_(torch.randn(hidden, hidden, generator=generator) * 0.3)

    def forward(self, inputs_embeds=None, use_cache=False, **kwargs):
        cumulative = inputs_embeds.cumsum(1)
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(cumulative)))

    def train(self, mode=True):
        return self


def tiny_policy():
    model = nn.Module()
    model.corelm = TinyCore()
    policy = NativeSpeechLMPolicy(model)
    return policy


def primary_logp_of_last_frame(policy, frames, prompt):
    """log P(primary of the last frame | everything before it)."""
    sequence = torch.cat(
        (torch.tensor(prompt, dtype=torch.long), torch.tensor(frames, dtype=torch.long))
    )[None]
    hidden = policy._hidden(sequence)[:, len(prompt) - 1 : -1]
    core = policy.core
    logits = torch.nn.functional.linear(
        hidden[:, -1] + core.head_emb.weight[0], core.lm_head.weight
    )
    mask = torch.zeros(VOCAB, dtype=torch.bool)
    mask[METADATA["ssl_token_start"] : METADATA["ssl_token_end"]] = True
    logits = logits.masked_fill(~mask, -torch.inf)
    return logits.log_softmax(-1)[0, frames[-1][0]]


def test_a_secondary_stream_changes_the_next_primary_probability():
    policy = tiny_policy()
    prompt = [[METADATA["codec_ssl_start_end_token_id"]] + [0] * (NQ - 1)]
    base = METADATA["codec_token_start"]
    frames = [
        [METADATA["ssl_token_start"] + 1] + [0] * (NQ - 1),
        [METADATA["ssl_token_start"] + 2] + [base + 0] + [0] * (NQ - 2),
        [METADATA["ssl_token_start"] + 3] + [base + 1] + [0] * (NQ - 2),
    ]
    first = primary_logp_of_last_frame(policy, frames, prompt)
    # Change only stream 1 of the middle frame.
    moved = [list(row) for row in frames]
    moved[1][1] = base + 2
    second = primary_logp_of_last_frame(policy, moved, prompt)
    assert not torch.isclose(first, second, atol=1e-6), (
        "the secondary stream is not reaching the primary's input: the engine's "
        "pending-sum would be unverifiable"
    )


def test_the_secondary_stream_gradient_matches_a_finite_difference():
    """The pad-cleared sum is the only path from a codec embedding to the
    primary's score, so perturbing that embedding must move the score by the
    analytic gradient."""
    policy = tiny_policy()
    core = policy.core
    prompt = [[METADATA["codec_ssl_start_end_token_id"]] + [0] * (NQ - 1)]
    base = METADATA["codec_token_start"]
    frames = [
        [METADATA["ssl_token_start"] + 1] + [0] * (NQ - 1),
        [METADATA["ssl_token_start"] + 2] + [base + 0] + [0] * (NQ - 2),
        [METADATA["ssl_token_start"] + 3] + [base + 1] + [0] * (NQ - 2),
    ]
    token = base + 0  # the codec id whose embedding we perturb
    direction = torch.zeros_like(core.emb.weight)
    generator = torch.Generator().manual_seed(11)
    direction[token] = torch.randn(core.emb.weight.shape[1], generator=generator)

    core.zero_grad()
    score = primary_logp_of_last_frame(policy, frames, prompt)
    score.backward()
    analytic = float((core.emb.weight.grad * direction).sum())

    epsilon = 1e-3
    with torch.no_grad():
        saved = core.emb.weight.detach().clone()
        core.emb.weight.copy_(saved + epsilon * direction)
        plus = float(primary_logp_of_last_frame(policy, frames, prompt))
        core.emb.weight.copy_(saved - epsilon * direction)
        minus = float(primary_logp_of_last_frame(policy, frames, prompt))
        core.emb.weight.copy_(saved)
    numeric = (plus - minus) / (2 * epsilon)
    assert analytic != 0.0
    assert math.isclose(analytic, numeric, rel_tol=2e-3, abs_tol=1e-5), (
        f"analytic {analytic} vs finite difference {numeric}"
    )


def test_pad_positions_contribute_no_gradient_to_the_primary_score():
    """A pad entry is cleared before the sum, so the pad embedding must be
    unreachable from the score -- otherwise the learner and the engine would
    disagree about what the environment contributed."""
    policy = tiny_policy()
    core = policy.core
    prompt = [[METADATA["codec_ssl_start_end_token_id"]] + [0] * (NQ - 1)]
    frames = [
        [METADATA["ssl_token_start"] + 1] + [0] * (NQ - 1),
        [METADATA["ssl_token_start"] + 2] + [0] * (NQ - 1),
    ]
    core.zero_grad()
    primary_logp_of_last_frame(policy, frames, prompt).backward()
    pad = core.emb.weight.grad[METADATA["pad_token_id"]]
    assert torch.equal(pad, torch.zeros_like(pad))


def test_scores_reads_each_streams_own_support_from_the_contract():
    """_scores must mask stream q to stream q's slice. If it used one shared
    support, a code legal for stream 2 would score finite under stream 1."""
    policy = tiny_policy()
    prompt = [[METADATA["codec_ssl_start_end_token_id"]] + [0] * (NQ - 1)]
    supports = codec_supports(METADATA)
    base = METADATA["codec_token_start"]
    frames = torch.tensor(
        [[METADATA["ssl_token_start"] + 1] + [base + 0] + [0] * (NQ - 2)]
    )
    active = torch.tensor([[True, True] + [False] * (NQ - 2)])[None]
    sequence = torch.cat((torch.tensor(prompt), frames))[None]
    hidden = policy._hidden(sequence)[:, len(prompt) - 1 : -1]
    logp, _, stream_logp = policy._scores(
        hidden, frames[None], active, [[dict(s) for s in supports]]
    )
    assert torch.isfinite(stream_logp).all()
    # Stream 1's score is the log of a distribution over SIZE codes only.
    assert stream_logp[0, 0, 1] >= math.log(1e-12)
    assert stream_logp[0, 0, 1] > -math.log(VOCAB) * 3
    # Inactive streams stay at zero, and the frame total is the stream sum.
    assert torch.equal(stream_logp[0, 0, 2:], torch.zeros(NQ - 2))
    assert torch.allclose(logp[0, 0], stream_logp[0, 0].sum(), atol=1e-6)


def test_an_action_outside_the_declared_support_is_refused():
    """Fail closed: if the engine recorded a support the action violates, the
    learner must not quietly score it."""
    policy = tiny_policy()
    prompt = [[METADATA["codec_ssl_start_end_token_id"]] + [0] * (NQ - 1)]
    supports = codec_supports(METADATA)
    # Stream 1 holding one of stream 2's ids.
    frames = torch.tensor(
        [[METADATA["ssl_token_start"] + 1] + [METADATA["codec_token_start"] + SIZE]
         + [0] * (NQ - 2)]
    )
    active = torch.tensor([[True, True] + [False] * (NQ - 2)])[None]
    sequence = torch.cat((torch.tensor(prompt), frames))[None]
    hidden = policy._hidden(sequence)[:, len(prompt) - 1 : -1]
    with pytest.raises(ValueError, match="outside the declared sampling support"):
        policy._scores(hidden, frames[None], active, [[dict(s) for s in supports]])
