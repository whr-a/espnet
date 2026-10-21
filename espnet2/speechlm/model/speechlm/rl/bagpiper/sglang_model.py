# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Bagpiper's native multi-stream policy on SGLang's own Qwen3 layers.

RadixAttention over the paged KV pool, many requests in flight. Every stream's
logits are divided by the batch's temperature here; SGLang samples the primary
stream from them, masked to each frame's support, at its own temperature 1,
and this model samples the seven secondary codec streams, with the categorical
rules the learner scores. Request IDs are ``batch:source:sample``; the batch's
prepared sources (native prompt frames, mode, mel features, temperature) are
read from the ``torch.save`` file named by ``espnet_rollout_sources``, which
BagpiperEngine rewrites before every batch. Per-request decoding state lives in
GPU tables indexed by the SGLang request-pool slot. Every frame returns its
support, primary activity and secondary actions as
``customized_info["espnet_bagpiper"]``: ``[support, primary_active]``, followed
by the seven secondary ids and their log probabilities when the secondary
streams are active. TP=1; the engine freezes radix cache, chunked prefill, CUDA
graphs and overlap scheduling off, so every extend row is one complete new
prompt and every decode row one frame.

WEST QA retains this temperature-scaled protocol path for integrity traces;
SGLang applies top-k afterwards. Its separate native objective scores raw
full-vocabulary logits, so these traces are never used as WEST loss ratios.
"""

from pathlib import Path

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)

from espnet2.rl.types import tempered

from .tokens import frame_supports, next_phase, support_masks

# Native names of the modules Bagpiper adds to its Qwen3 backbone.
RENAMES = {
    "multimodal_io_dict.continuous_audio.model.audio_tower.": "audio_tower.",
    "adaptor.continuous_audio.": "audio_adaptor.",
}
# The frozen Xcodec decoder serves the TTS reward only.
CODEC_MODEL = "multimodal_io_dict.discrete_audio.codec_model."
# SGLang fuses the q/k/v and gate/up projections into one parameter each.
STACKED = (
    ("q_proj", "qkv_proj", "q"),
    ("k_proj", "qkv_proj", "k"),
    ("v_proj", "qkv_proj", "v"),
    ("gate_proj", "gate_up_proj", 0),
    ("up_proj", "gate_up_proj", 1),
)


class ESPnetBagpiperForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("Bagpiper RL requires unquantized weights")
        super().__init__(config, None, prefix)
        self.setup(config)

    def setup(self, config):
        """Add Bagpiper's streams, audio encoder and rollout state to Qwen3."""
        geometry = self.geometry = config.espnet_bagpiper
        self.hidden, self.vocab = config.hidden_size, config.vocab_size
        self.stream_emb = nn.Embedding(geometry["num_stream"], self.hidden)
        audio = Qwen3OmniMoeAudioEncoderConfig(
            **{
                key: value
                for key, value in geometry["audio_config"].items()
                if key != "model_type"
            }
        )
        audio._attn_implementation = "sdpa"
        self.audio_tower = Qwen3OmniMoeAudioEncoder(audio)
        self.audio_adaptor = nn.Linear(geometry["adaptor_input_dim"], self.hidden)
        self.register_buffer(
            "masks", support_masks(geometry, self.vocab), persistent=False
        )
        self.sources_path = Path(config.espnet_rollout_sources)
        self.batch = self.phase = self.text_audio = self.pending = None
        self.codec = False
        if getattr(config, "espnet_bagpiper_match_native_bf16", False):
            from .numerics import install_native_bf16_numerics

            install_native_bf16_numerics(self)
        if getattr(config, "espnet_bagpiper_stateless_sampling", False):
            from .sampling import install_safe_stateless_sampler

            install_safe_stateless_sampler()
        if self.lm_head.weight.dtype == torch.float32:
            # sgl_kernel's RMSNorm has no float32 kernel; SGLang's native norm
            # is the full-precision reference (as for OpusLM).
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    module._forward_method = module.forward_native

    def load_weights(self, weights):
        """Load one complete native policy: every parameter exactly once."""
        from espnet2.rl.weight_transport import verify_weights

        weights = verify_weights(
            weights, self.sources_path.with_suffix(".weights.json")
        )
        params = dict(self.named_parameters())
        shards = {fused: [] for _, fused, _ in STACKED}
        for _, fused, shard in STACKED:
            shards[fused].append(shard)
        expected = {
            (name, shard)
            for name in params
            for shard in shards.get(name.split(".")[-2], [None])
        }
        loaded = []
        for original, tensor in weights:
            if original.startswith(CODEC_MODEL):
                continue
            name, shard = original, None
            for before, after in RENAMES.items():
                if name.startswith(before):
                    name = after + name[len(before) :]
            for part, fused, index in STACKED if name.startswith("model.") else ():
                if f".{part}." in name:
                    name, shard = name.replace(f".{part}.", f".{fused}."), index
                    break
            if name not in params:
                raise ValueError(f"Unexpected Bagpiper native weight {original}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor) if shard is None else loader(param, tensor, shard)
            loaded.append((name, shard))
        if len(set(loaded)) != len(loaded) or set(loaded) != expected:
            missing = sorted(str(item) for item in expected - set(loaded))
            raise ValueError(
                f"Bagpiper weights are incomplete or repeated; missing {missing[:4]}"
            )
        # Audio features computed with the previous weights must not survive.
        self.batch = None
        return set(params)

    def _load(self, rids):
        """Read the prepared sources of the batch named by the request IDs."""
        batches = {rid.split(":", 1)[0] for rid in rids}
        if len(batches) != 1:
            raise RuntimeError("One SGLang batch mixes Bagpiper rollout batches")
        (batch,) = batches
        if batch == self.batch:
            return
        record = torch.load(self.sources_path, map_location="cpu", weights_only=True)
        if str(record["batch"]) != batch:
            raise RuntimeError(
                f"{self.sources_path} holds batch {record['batch']}, not {batch}"
            )
        self.batch, self.sources = batch, record["sources"]
        self.temperature = record["temperature"]
        self.codec = any(source["text_audio"] for source in self.sources)
        if record.get("stateless_sampling", False) and self.codec:
            raise ValueError("Stateless Bagpiper sampling requires text_only QA")
        self.audio, self.seen = {}, set()

    def _audio(self, index, like):
        """The adapted audio encoder output of one source, shared by its samples."""
        cached = self.audio.get(index)
        if cached is None:
            audio = self.sources[index]["audio"]
            features = audio["feats"].to(device=like.device, dtype=like.dtype)
            encoded = self.audio_tower(
                features[0].T, feature_lens=audio["lengths"].to(like.device)
            )
            encoded = getattr(encoded, "last_hidden_state", encoded)
            cached = self.audio[index] = [
                self.audio_adaptor(encoded),
                self.sources[index]["samples"],
            ]
        cached[1] -= 1
        if not cached[1]:
            del self.audio[index]
        return cached[0]

    def _reserve(self, slots):
        """Grow the state tables to cover these request-pool slots."""
        size = int(slots.max()) + 1
        if self.phase is not None and len(self.phase) >= size:
            return
        old = (self.phase, self.text_audio, self.pending)
        size = max(size, 2 * len(old[0]) if old[0] is not None else 64)
        self.phase = slots.new_zeros(size, dtype=torch.int8)
        self.text_audio = slots.new_zeros(size, dtype=torch.bool)
        self.pending = slots.new_zeros((size, self.hidden), dtype=torch.float32)
        if old[0] is not None:
            for table, value in zip((self.phase, self.text_audio, self.pending), old):
                table[: len(value)] = value

    def _extend(self, input_ids, batch):
        """Native prompt embeddings of new requests; reset their state rows."""
        self._load(batch.rids)
        embeds = self.model.embed_tokens(input_ids)
        indices, start = [], 0
        for rid, length, prefix in zip(
            batch.rids,
            batch.extend_seq_lens_cpu,
            batch.extend_prefix_lens_cpu,
            strict=True,
        ):
            index = int(rid.split(":")[1])
            source = self.sources[index]
            if prefix or length != len(source["prompt"]) or rid in self.seen:
                # A retracted, chunked or prefix-cached request would be
                # recomputed from primary ids without its secondary streams.
                raise RuntimeError(
                    f"SGLang extended Bagpiper request {rid} by {length} tokens "
                    f"after {prefix}; every request must be prefilled once, whole"
                )
            self.seen.add(rid)
            rows = embeds[start : start + length]
            secondary = source["prompt"][:, 1:]
            if secondary.any():
                # Sum of all streams, padding (id 0) cleared in streams >= 1.
                secondary = secondary.to(input_ids.device)
                extra = self.model.embed_tokens(secondary).float()
                extra = extra.masked_fill((secondary == 0)[..., None], 0).sum(1)
                rows.copy_((rows.float() + extra).to(rows.dtype))
            if source["audio"] is not None:
                audio = source["audio"]
                rows[audio["start"] : audio["start"] + audio["length"]] = self._audio(
                    index, rows
                )
            indices.append(index)
            start += length
        prompts = torch.cat([self.sources[i]["prompt"][:, 0] for i in indices])
        if not torch.equal(prompts.to(input_ids), input_ids):
            raise RuntimeError("SGLang input ids differ from the native prompts")
        slots = batch.req_pool_indices
        self._reserve(slots)
        self.phase[slots] = 0
        self.text_audio[slots] = torch.tensor(
            [self.sources[i]["text_audio"] for i in indices], device=slots.device
        )
        self.pending[slots] = 0
        lengths = torch.tensor(batch.extend_seq_lens_cpu, device=input_ids.device)
        return embeds, lengths.cumsum(0) - 1

    def _decode(self, input_ids, slots):
        """The next input is the sum of every stream of the last frame."""
        self.phase[slots] = next_phase(
            self.phase[slots],
            input_ids,
            self.text_audio[slots],
            self.geometry["eot_token_id"],
        )
        embeds = self.model.embed_tokens(input_ids)
        if self.codec:
            # One rounding of the FP32 sum, as the learner's stream sum.
            embeds = (embeds.float() + self.pending[slots]).to(embeds.dtype)
        return embeds

    def _secondary(self, hidden, weight):
        """Sample streams 1..7 from their own codec layers for every row."""
        geometry = self.geometry
        size, base = geometry["codec_layer_size"], geometry["codec_base_offset"]
        ids, log_probs = [], []
        for stream in range(1, geometry["num_stream"]):
            lo = base + stream * size
            logits = F.linear(
                hidden + self.stream_emb.weight[stream].to(hidden.dtype),
                weight[lo : lo + size],
            ).float()
            if self.temperature == 0:
                choice = logits.argmax(-1)
            else:
                logits = tempered(logits, self.temperature)
                choice = torch.multinomial(logits.softmax(-1), 1)[:, 0]
            ids.append(choice + lo)
            log_probs.append(logits.log_softmax(-1).gather(-1, choice[:, None])[:, 0])
        return torch.stack(ids, -1), torch.stack(log_probs, -1)

    def _head(self, hidden, slots):
        weight = self.lm_head.weight[: self.vocab]
        raw = F.linear(hidden, weight)
        support, codec = frame_supports(
            raw, self.phase[slots], self.text_audio[slots], self.geometry
        )
        allowed = self.masks[support]
        active = (allowed & torch.isfinite(raw)).sum(-1) > 1
        # A fresh tensor: SGLang's sampler softmaxes it in place.
        logits = raw.float().masked_fill(~allowed, -torch.inf)
        if self.temperature:  # 0 decodes greedily
            logits = tempered(logits, self.temperature)
        flags = torch.stack((support, active.long(), codec.long()), -1).tolist()
        rows = [row[:2] for row in flags]
        if self.codec:
            ids, log_probs = self._secondary(hidden, weight)
            self.pending[slots] = torch.where(
                codec[:, None], self.model.embed_tokens(ids).float().sum(1), 0
            )
            if any(row[2] for row in flags):
                for row, flag, sampled, scores in zip(
                    rows, flags, ids.tolist(), log_probs.tolist()
                ):
                    if flag[2]:
                        row += sampled + scores
        return LogitsProcessorOutput(
            next_token_logits=logits, customized_info={"espnet_bagpiper": rows}
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
            raise ValueError("Bagpiper builds its own native input embeddings")
        if mode == ForwardMode.EXTEND:
            embeds, last = self._extend(input_ids, forward_batch)
        elif mode == ForwardMode.DECODE:
            embeds = self._decode(input_ids, forward_batch.req_pool_indices)
            last = None
        else:
            raise RuntimeError(f"Bagpiper rollout cannot run SGLang {mode.name}")
        hidden = self.model(input_ids, positions, forward_batch, input_embeds=embeds)
        if last is not None:
            hidden = hidden[last]
        return self._head(hidden, forward_batch.req_pool_indices)


EntryClass = ESPnetBagpiperForCausalLM
