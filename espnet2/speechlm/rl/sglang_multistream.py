# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haoran Wang
"""OpusLM's nine-stream ARDelay policy inside SGLang, sampling real codec.

The text-only interface in sglang_model.py hands the engine a prompt embedding
and masks every logit to the text range, so it can only ever emit stream 0.
Codec output needs the engine to know what the secondary streams sampled,
because the next frame's input is the pad-cleared sum of all nine streams
(model.py:process_embedding) -- an embedding the caller cannot precompute.

So this model takes primary input_ids and builds its own embeddings, as
Bagpiper does: it reads the batch's prepared nine-stream prompts from the
torch.save file named by ``espnet_rollout_sources``, keeps the summed secondary
embedding of the last frame in a GPU table indexed by the SGLang request-pool
slot, and samples streams 1..nq-1 itself. SGLang samples stream 0 from the
logits returned here, masked to that frame's support. Every frame's actions and
log probabilities leave through ``customized_info["espnet_opuslm"]``.

Unlike Bagpiper, every OpusLM stream reads the whole lm_head and is
distinguished only by head_emb.weight[stream] (policy.py:_scores), so the
support must be masked explicitly rather than taken from a head slice.

TP=1; the engine freezes radix cache, chunked prefill, CUDA graphs and overlap
scheduling off, so each extend row is one whole new prompt and each decode row
one frame.
"""

from pathlib import Path

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.olmo2 import Olmo2ForCausalLM
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, Olmo2Config

from espnet2.speechlm.rl.multistream import active_streams
from espnet2.speechlm.rl.sglang_model import _backbone_config
from espnet2.speechlm.rl.sglang_model import _CompleteLoad as _TextCompleteLoad
from espnet2.speechlm.rl.sglang_model import _native_fp32_norms


class _CompleteLoad(_TextCompleteLoad):
    """The text model's complete-load check, which also drops the prompt
    embeddings computed with the previous weights."""

    def load_weights(self, weights):
        super().load_weights(weights)
        self.batch = None


class _OpusMultiStream:
    """The frame state machine and secondary sampler, mixed into a backbone."""

    def setup(self, config):
        geometry = self.geometry = {
            key: int(config.to_dict()[key])
            for key in (
                "nq",
                "vocab_size",
                "pad_token_id",
                "eos_token_id",
                "ssl_token_start",
                "ssl_token_end",
                "codec_token_start",
                "codec_per_stream_size",
                "codec_ssl_start_end_token_id",
                "text_token_start",
                "text_token_end",
                "text_bpe_start_end_token_id",
            )
        }
        self.nq = geometry["nq"]
        self.vocab = geometry["vocab_size"]
        self.head_emb = nn.Embedding(12, config.hidden_size)
        self.register_buffer(
            "supports", self._support_masks(geometry), persistent=False
        )
        self.sources_path = Path(config.espnet_rollout_sources)
        self.batch = self.sources = None
        self.temperature = 1.0
        # Per-slot decode state, grown on demand.
        self.step = self.closed = self.pending = None
        _native_fp32_norms(self)

    @staticmethod
    def _support_masks(geometry):
        """[nq + 2, vocab] allowed ids per stream, plus stream 0's two tails.

        Rows 0..nq-1 are the streams' codec-frame supports and must equal
        multistream.codec_supports, which the learner scores against. Rows
        OPEN_STREAM0 and DONE_STREAM0 are what stream 0 may emit once the block
        closed: SGLang keeps sampling stream 0 from whatever this model
        returns, so the only way to make it emit the environment's pad is to
        leave pad as the single allowed id (log probability 0, matching the
        learner's 0 for an inactive stream), and the only way to end the
        request is to leave just EOS.
        """
        nq, vocab = geometry["nq"], geometry["vocab_size"]
        size = geometry["codec_per_stream_size"]
        base = geometry["codec_token_start"]
        masks = torch.zeros(nq + 2, vocab, dtype=torch.bool)
        masks[0, geometry["ssl_token_start"] : geometry["ssl_token_end"]] = True
        masks[0, geometry["codec_ssl_start_end_token_id"]] = True
        for stream in range(1, nq):
            low = base + (stream - 1) * size
            masks[stream, low : low + size] = True
        masks[nq, geometry["pad_token_id"]] = True
        masks[nq + 1, geometry["eos_token_id"]] = True
        return masks

    @property
    def flush_support(self):
        return self.supports[self.nq]

    @property
    def done_support(self):
        return self.supports[self.nq + 1]

    def _load(self, rids):
        """Read the prepared sources of the batch named by the request IDs."""
        batches = {rid.split(":", 1)[0] for rid in rids}
        if len(batches) != 1:
            raise RuntimeError("One SGLang batch mixes OpusLM rollout batches")
        (batch,) = batches
        if batch == self.batch:
            return
        record = torch.load(self.sources_path, map_location="cpu", weights_only=True)
        if str(record["batch"]) != batch:
            raise RuntimeError(
                f"{self.sources_path} holds batch {record['batch']}, not {batch}"
            )
        self.batch, self.sources = batch, record["sources"]
        self.temperature = float(record["temperature"])
        self.seen = set()

    def _reserve(self, slots):
        size = int(slots.max()) + 1
        if self.step is not None and len(self.step) >= size:
            return
        old = (self.step, self.closed, self.pending)
        size = max(size, 2 * len(old[0]) if old[0] is not None else 64)
        self.step = slots.new_zeros(size, dtype=torch.long)
        self.closed = slots.new_full((size,), -1, dtype=torch.long)
        self.pending = slots.new_zeros(
            (size, self.config.hidden_size), dtype=torch.float32
        )
        if old[0] is not None:
            for table, value in zip((self.step, self.closed, self.pending), old):
                table[: len(value)] = value

    def _embed_frames(self, frames):
        """Native ARDelay embedding: zero every pad entry, then sum streams."""
        rows = self.model.embed_tokens(frames)
        return (
            rows.float()
            .masked_fill((frames == self.geometry["pad_token_id"])[..., None], 0)
            .sum(-2)
        )

    def _extend(self, input_ids, batch):
        """Nine-stream prompt embeddings of new requests; reset their state."""
        self._load(batch.rids)
        pieces, indices, start = [], [], 0
        for rid, length, prefix in zip(
            batch.rids,
            batch.extend_seq_lens_cpu,
            batch.extend_prefix_lens_cpu,
            strict=True,
        ):
            index = int(rid.split(":")[1])
            prompt = self.sources[index]["prompt"]
            if prefix or length != len(prompt) or rid in self.seen:
                # A retracted, chunked or prefix-cached request would be
                # recomputed from primary ids without its secondary streams.
                raise RuntimeError(
                    f"SGLang extended OpusLM request {rid} by {length} tokens "
                    f"after {prefix}; every request must be prefilled once, whole"
                )
            self.seen.add(rid)
            frames = prompt.to(input_ids.device)
            if not torch.equal(frames[:, 0], input_ids[start : start + length]):
                raise RuntimeError("SGLang input ids differ from the native prompts")
            pieces.append(self._embed_frames(frames))
            indices.append(index)
            start += length
        embeds = torch.cat(pieces)
        slots = batch.req_pool_indices
        self._reserve(slots)
        self.step[slots] = 0
        self.closed[slots] = -1
        self.pending[slots] = 0
        lengths = torch.tensor(batch.extend_seq_lens_cpu, device=input_ids.device)
        return embeds.to(self.model.embed_tokens.weight.dtype), lengths.cumsum(0) - 1

    def _decode(self, input_ids, slots):
        """The next input is the pad-cleared sum of the last frame's streams."""
        primary = self.model.embed_tokens(input_ids)
        # Stream 0 is pad once the block closed; its embedding must not be added.
        live = (self.closed[slots] < 0).float()[:, None]
        summed = primary.float() * live + self.pending[slots]
        return summed.to(primary.dtype)

    def _secondary(self, hidden, slots, codec):
        """Sample streams 1..nq-1 for every row, under their own supports."""
        ids = hidden.new_zeros((len(hidden), self.nq - 1), dtype=torch.long)
        scores = hidden.new_zeros((len(hidden), self.nq - 1), dtype=torch.float32)
        for stream in range(1, self.nq):
            live = codec[:, stream]
            if not bool(live.any()):
                continue
            rows = live.nonzero().squeeze(-1)
            logits = F.linear(
                hidden[rows] + self.head_emb.weight[stream].to(hidden.dtype),
                self.lm_head.weight[: self.vocab],
            ).float()
            logits = logits.masked_fill(~self.supports[stream], -torch.inf)
            if self.temperature == 0:
                choice = logits.argmax(-1)
            else:
                logits = logits / self.temperature
                choice = torch.multinomial(logits.softmax(-1), 1)[:, 0]
            ids[rows, stream - 1] = choice
            scores[rows, stream - 1] = logits.log_softmax(-1).gather(
                -1, choice[:, None]
            )[:, 0]
        return ids, scores

    def _head(self, hidden, slots):
        step, closed = self.step[slots], self.closed[slots]
        # Which streams carry a real action at this frame, per row.
        active = torch.stack(
            [
                torch.as_tensor(
                    active_streams(self.nq, int(s), None if int(c) < 0 else int(c)),
                    device=hidden.device,
                )
                for s, c in zip(step.tolist(), closed.tolist())
            ]
        )
        raw = F.linear(
            hidden + self.head_emb.weight[0].to(hidden.dtype),
            self.lm_head.weight[: self.vocab],
        )
        # Stream 0 samples SSL ids while the block is open, is forced to pad
        # during the delay flush, and is forced to EOS once the flush ended.
        flushing = closed >= 0
        done = flushing & (step - closed >= self.nq - 1)
        support = torch.where(
            done[:, None],
            self.done_support,
            torch.where(flushing[:, None], self.flush_support, self.supports[0]),
        )
        # A fresh tensor: SGLang's sampler softmaxes it in place.
        logits = raw.float().masked_fill(~support, -torch.inf)
        if self.temperature:  # 0 decodes greedily
            logits = logits / self.temperature
        ids, scores = self._secondary(hidden, slots, active)
        # Carry this frame's secondary embedding into the next input.
        secondary = torch.where(
            active[:, 1:, None],
            self.model.embed_tokens(ids).float(),
            torch.zeros((), device=hidden.device, dtype=torch.float32),
        ).sum(-2)
        self.pending[slots] = secondary
        self.step[slots] = step + 1
        rows = [
            [int(a) for a in row] + sampled + score
            for row, sampled, score in zip(
                active.tolist(), ids.tolist(), scores.tolist()
            )
        ]
        return LogitsProcessorOutput(
            next_token_logits=logits, customized_info={"espnet_opuslm": rows}
        )

    def observe_primary(self, slots, tokens):
        """Record that stream 0 closed the block, after SGLang sampled it.

        SGLang samples stream 0 from the logits _head returned, so the model
        only learns the sampled id on the next decode call, in input_ids.
        """
        close = self.geometry["codec_ssl_start_end_token_id"]
        closing = (tokens == close) & (self.closed[slots] < 0)
        if bool(closing.any()):
            # step was already advanced past the closing frame by _head.
            self.closed[slots] = torch.where(
                closing, self.step[slots] - 1, self.closed[slots]
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        mode = forward_batch.forward_mode
        if mode == ForwardMode.IDLE:
            return LogitsProcessorOutput(
                next_token_logits=input_ids.new_empty(
                    (0, self.vocab), dtype=torch.float
                )
            )
        if input_embeds is not None or get_embedding:
            raise ValueError("OpusLM builds its own native input embeddings")
        slots = forward_batch.req_pool_indices
        if mode == ForwardMode.EXTEND:
            embeds, last = self._extend(input_ids, forward_batch)
        elif mode == ForwardMode.DECODE:
            self.observe_primary(slots, input_ids)
            embeds = self._decode(input_ids, slots)
            last = None
        else:
            raise RuntimeError(f"OpusLM rollout cannot run SGLang {mode.name}")
        hidden = self.model(input_ids, positions, forward_batch, input_embeds=embeds)
        if last is not None:
            hidden = hidden[last]
        return self._head(hidden, slots)


class OpusLMMultiStreamForConditionalGeneration(
    _CompleteLoad, _OpusMultiStream, Olmo2ForCausalLM
):
    def __init__(self, config, quant_config=None, prefix="", **kwargs):
        super().__init__(
            _backbone_config(config, Olmo2Config), quant_config, prefix, **kwargs
        )
        self.config = config
        self.setup(config)


class OpusLMDialogueMultiStreamForConditionalGeneration(
    _CompleteLoad, _OpusMultiStream, LlamaForCausalLM
):
    def __init__(self, config, quant_config=None, prefix="", **kwargs):
        super().__init__(
            _backbone_config(config, LlamaConfig), quant_config, prefix, **kwargs
        )
        self.config = config
        self.setup(config)


EntryClass = [
    OpusLMMultiStreamForConditionalGeneration,
    OpusLMDialogueMultiStreamForConditionalGeneration,
]
