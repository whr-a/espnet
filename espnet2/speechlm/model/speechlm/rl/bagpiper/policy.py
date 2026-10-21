# Copyright 2025 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher-forced RL scores from the original ESPnet model components.

The current SpeechLM training forward returns fused scalar CE, which is not a
per-action RL interface. This wrapper retains its embeddings, audio encoder,
transformer, stream offsets and output projection, and exposes differentiable
log probabilities of the actions a Bagpiper rollout sampled, every frame under
the support the rollout recorded and every stream at its sampling temperature.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from espnet2.rl.types import TokenScores, tempered

from .tokens import SUPPORTS, frame_supports, support_masks


@contextmanager
def checkpoint_native_layers(model, enabled):
    """Recompute native decoder layers while retaining their evaluation mode.

    HF's built-in layer checkpointing is gated by training mode. RL scoring
    disables dropout, so temporarily wrap the original layer forwards instead.
    Restore every attribute before returning; checkpoint and reference copies
    retain the original native module names and methods.
    """
    if not enabled or not torch.is_grad_enabled():
        yield
        return
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Native layer checkpointing requires an HF decoder layer list")
    restored = []
    try:
        for layer in layers:
            prior = layer.__dict__.get("forward")
            original = layer.forward

            def recompute(*args, _original=original, **kwargs):
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)

            restored.append((layer, prior))
            layer.forward = recompute
        yield
    finally:
        for layer, prior in restored:
            if prior is None:
                layer.__dict__.pop("forward", None)
            else:
                layer.forward = prior


@contextmanager
def offload_saved_activations(model, enabled):
    """Losslessly store checkpoint boundary activations on CPU for backward.

    Parameter and buffer storage remains on its original device, including any
    transposed/shared views saved by autograd. Nested checkpoint implementations
    may own additional saved tensors; this only controls this hook's boundary.
    """
    if not enabled or not torch.is_grad_enabled():
        yield
        return
    protected = {
        tensor.untyped_storage().data_ptr()
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.device.type == "cuda"
    }

    def pack(tensor):
        if (
            tensor.device.type != "cuda"
            or tensor.untyped_storage().data_ptr() in protected
        ):
            return tensor
        return tensor.device, tensor.detach().to("cpu", copy=True)

    def unpack(saved):
        if isinstance(saved, tuple):
            device, tensor = saved
            return tensor.to(device)
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def move_native_model(model, device, dtype, *, preserve_decoder_rope_fp32=False):
    """Move native weights while retaining the decoder's exact RoPE frequencies.

    ``Module.to(dtype=...)`` also rounds nonpersistent floating-point buffers.
    HF computes rotary phases in FP32, but casting an already rounded frequency
    back to FP32 cannot recover it. Save the original decoder buffers before
    the move; audio frontends and every parameter follow the normal conversion.
    """
    frequencies = []
    if preserve_decoder_rope_fp32:
        decoder = getattr(getattr(model, "corelm", model), "model", None)
        if decoder is None:
            raise ValueError(
                "FP32 decoder RoPE preservation requires a native decoder"
            )
        for module in decoder.modules():
            if not hasattr(module, "inv_freq"):
                continue
            for name in ("inv_freq", "original_inv_freq"):
                value = getattr(module, name, None)
                if value is None:
                    continue
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        "Preserve decoder RoPE before casting its original FP32 buffers"
                    )
                frequencies.append((module, name, value.detach().clone()))
        if not frequencies:
            raise ValueError("Native decoder exposes no original RoPE frequency buffers")
    model = model.to(device=device, dtype=dtype)
    for module, name, original in frequencies:
        setattr(module, name, original.to(device=getattr(module, name).device))
    return model


class NativeSpeechLMPolicy(nn.Module):
    """Score Bagpiper rollouts as right-padded batches with one vectorised head.

    Batch contract follows espnet2.rl.batch.RolloutBatch, with the actions and
    sampling masks of every stream (responses, response_stream_mask [B,T,Q]).
    Each source carries its rollout's per-frame ``support``, its uniform
    ``sampling_contract`` (every stream's whole support at ``temperature``)
    and the recipe's prompt features (``native_model_inputs``). Prompts must
    have exactly the same multimodal token expansion as SGLang; silently
    re-tokenizing the audio would change the policy.

    WEST text-only QA additionally exposes ``score_raw=True``: unfiltered
    full-primary-vocabulary probabilities at temperature one for its loss.
    The default forward remains the tempered protocol integrity pass even
    when WEST sampling truncates the engine distribution to its top 50.
    """

    def __init__(
        self,
        model,
        bagpiper_config,
        token_chunk_size=32,
        activation_checkpointing=False,
        score_checkpointing=True,
        activation_cpu_offload=False,
        support_mismatch_rate=1e-3,
        temperature=1.0,
        top_k=-1,
        west_grpo=False,
    ):
        super().__init__()
        self.model = model
        core = self.core
        required = ("model", "_embed", "stream_emb", "lm_head", "num_stream")
        if any(getattr(core, name, None) is None for name in required):
            raise ValueError(
                "Bagpiper scoring requires a complete native parallel model"
            )
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        # Frames per candidate in one head block: the block of a batch of B
        # candidates holds token_chunk_size * B frames.
        self.token_chunk_size = token_chunk_size
        self.num_stream = int(core.num_stream)
        self.bagpiper_config = bagpiper_config
        self.activation_checkpointing = bool(activation_checkpointing)
        self.score_checkpointing = bool(score_checkpointing)
        self.activation_cpu_offload = bool(activation_cpu_offload)
        # Frames scored under a recorded support other than the one this
        # policy's own logits give, and all frames scored, in no-grad passes.
        # A scoring pass of the current or behavior policy fails above this
        # rate (espnet2.rl.scoring); the reference only counts.
        self.support_mismatch_rate = support_mismatch_rate
        self.support_mismatches = self.support_frames = 0
        # Every stream is sampled from softmax(logits / temperature).
        self.temperature = temperature
        self.top_k = top_k
        self.west_grpo = bool(west_grpo)
        # The primary ids every support allows, [len(SUPPORTS), vocab].
        weight = core.lm_head.weight
        self.register_buffer(
            "allowed",
            support_masks(bagpiper_config, weight.shape[0], weight.device),
            persistent=False,
        )

    @property
    def core(self):
        return getattr(self.model, "corelm", self.model)

    def checkpoint_state_dict(self):
        return self.model.state_dict()

    def forward(self, batch, *, score_raw=False):
        if score_raw and not self.west_grpo:
            raise ValueError("Raw Bagpiper scoring requires the WEST QA objective")
        with offload_saved_activations(self.model, self.activation_cpu_offload):
            return self._forward(batch, score_raw=score_raw)

    def score_raw(self, batch):
        """WEST objective probabilities, distinct from rollout integrity scores."""
        return self(batch, score_raw=True)

    def _forward(self, batch, *, score_raw=False):
        responses = batch.batch["responses"].cpu()
        masks = batch.batch["response_mask"].cpu().bool()
        streams = batch.batch.get("response_stream_mask")
        if responses.ndim != 3 or responses.shape[-1] != self.num_stream:
            raise ValueError("Bagpiper responses must hold every stream [B,T,Q]")
        if streams is None:
            raise ValueError("Multistream RL requires an explicit response_stream_mask")
        streams = streams.cpu().bool()
        if streams.shape != responses.shape or masks.shape != responses.shape[:2]:
            raise ValueError("Response mask shape mismatch")
        rows, width = masks.shape
        lengths = batch.batch["response_lengths"].tolist()
        sources = batch.non_tensor_batch["source"]
        prompts, supports, modes = [], [], []
        for length, prompt, source in zip(
            lengths, batch.non_tensor_batch["prompt_token_ids"], sources, strict=True
        ):
            prompt = torch.as_tensor(prompt, dtype=torch.long)
            if (
                prompt.ndim != 2
                or prompt.shape[1] != self.num_stream
                or not prompt.shape[0]
            ):
                raise ValueError("Expected a nonempty prompt [P,Q]")
            if not 0 <= length <= width:
                raise ValueError("Invalid response length")
            # One contract covers every stream of every frame.
            contract = source.get("sampling_contract") or {}
            top_k = contract.get("top_k", 0)
            full_support = top_k <= 0 or top_k >= self.allowed.shape[1]
            allowed_top_k = (
                top_k == self.top_k
                if self.west_grpo and self.top_k > 0
                else full_support
            )
            if contract.get("temperature") != self.temperature or not allowed_top_k:
                raise ValueError(
                    "Bagpiper rollout sampling contract differs from its policy: "
                    f"temperature={self.temperature}, top_k={self.top_k}"
                )
            support = source.get("support")
            if support is None or len(support) != length:
                raise ValueError(
                    "Bagpiper scoring requires the support its rollout recorded "
                    "for every frame"
                )
            extra = source.get("sampling_extra_args", {})
            mode = source.get("mode", extra.get("mode", "text_only"))
            if (
                mode not in ("text_only", "text_audio")
                or extra.get("phase", "text") != "text"
            ):
                raise ValueError(
                    "Bagpiper rollouts decode text_only or text_audio from text"
                )
            prompts.append(prompt)
            supports.append(torch.as_tensor(support, dtype=torch.long))
            modes.append(mode == "text_audio")
        if score_raw and (any(modes) or streams[..., 1:].any()):
            raise ValueError("Raw WEST Bagpiper scoring requires text_only QA actions")
        support = torch.cat(supports)
        if ((support < 0) | (support >= len(SUPPORTS))).any():
            raise ValueError("Unknown Bagpiper support")
        # Right padding: causal attention keeps it from every real position.
        span = max(len(prompt) + length for prompt, length in zip(prompts, lengths))
        sequence = torch.zeros((rows, span, self.num_stream), dtype=torch.long)
        hidden_index, frame_index = [], []
        for row, (prompt, length) in enumerate(zip(prompts, lengths)):
            size = len(prompt)
            sequence[row, :size] = prompt
            sequence[row, size : size + length] = responses[row, :length]
            # hidden[P-1+t] predicts frame t; the final frame's hidden is unused.
            hidden_index.append(torch.arange(length) + row * span + size - 1)
            frame_index.append(torch.arange(length) + row * width)
        hidden_index, frame_index = torch.cat(hidden_index), torch.cat(frame_index)
        labels = responses.reshape(-1, self.num_stream)[frame_index]
        active = (streams & masks[..., None]).reshape(-1, self.num_stream)[frame_index]
        sampled = torch.zeros(rows * width, dtype=torch.bool)
        sampled[frame_index] = active.any(-1)
        counts = torch.tensor(lengths, dtype=torch.long)
        text_audio = torch.repeat_interleave(torch.tensor(modes), counts)
        phase = self._phases(labels[:, 0], counts, text_audio)
        device = self.core.lm_head.weight.device
        hidden = self._hidden(sequence.to(device), sources)
        hidden = hidden.reshape(rows * span, -1)[hidden_index.to(device)]
        logp, entropy, stream_logp = self._score_frames(
            hidden,
            *(item.to(device) for item in (labels, active, support, phase, text_audio)),
            block=self.token_chunk_size * rows,
            score_raw=score_raw,
        )
        frame_index = frame_index.to(device)

        def place(values):
            """Frame values [N,...] at their candidates' positions [B,T,...]."""
            shape = values.shape[1:]
            padded = values.new_zeros((rows * width, *shape))
            return padded.index_copy(0, frame_index, values).view(rows, width, *shape)

        return TokenScores(
            place(logp),
            place(entropy),
            sampled.view(rows, width).to(device),
            place(stream_logp),
        )

    def _phases(self, primary, lengths, text_audio):
        """Every frame's decoding phase, an index into tokens.PHASES.

        text_audio: text -(eot)-> transition (forced assistant) -> audio
        -(eot)-> audio_stop (forced eos); text_only stays in text. The phase of
        a frame follows from the eots its own rollout sampled before it.
        """
        eot = (primary == self.bagpiper_config["eot_token_id"]).long()
        filled = lengths > 0
        starts = (lengths.cumsum(0) - lengths)[filled]
        before = eot.cumsum(0) - eot
        before = before - torch.repeat_interleave(before[starts], lengths[filled])
        follows = torch.zeros_like(eot)
        follows[1:] = eot[:-1]
        follows[starts] = 0
        phase = torch.where(
            before == 0, 0, torch.where(before == 1, torch.where(follows == 1, 1, 2), 3)
        )
        return torch.where(text_audio, phase, 0)

    def _hidden(self, sequence, sources):
        """Final hidden states of right-padded native sequences [B,S,Q].

        Each prompt's continuous features go through the native encoder and
        adaptor once per feature array: the candidates of one prompt share it.
        """
        core = self.core
        core.eval()  # Disable dropout without disabling autograd.
        embeds = core._embed(sequence, {})
        encoded = {}
        for row, source in enumerate(sources):
            native = dict(source.get("native_model_inputs") or {})
            for name, handler in core.multimodal_io_dict.items():
                if handler.is_discrete or native.get(name + "_feats") is None:
                    continue
                feats = native.pop(name + "_feats")
                lengths = native.pop(name + "_lengths")
                key = name, id(feats), tuple(lengths)
                if key not in encoded:
                    # A copy: arrays from the object store are read-only.
                    values = torch.tensor(
                        feats, dtype=torch.float32, device=embeds.device
                    )
                    parameter = next(handler.parameters(), None)
                    if parameter is not None:
                        # Frontend features stay fp32 in the manifest; convert
                        # only at the native encoder boundary.
                        values = values.to(parameter.dtype)
                    encoded[key] = [
                        core.adaptor[name](feat)
                        for feat in handler.encode_batch(
                            values,
                            torch.as_tensor(lengths, device=embeds.device),
                        )
                    ]
                for feat, (index, start, length) in zip(
                    encoded[key], native.pop(name + "_indices"), strict=True
                ):
                    if index != 0:
                        raise ValueError(
                            "Native prompt features index their own prompt"
                        )
                    embeds[row, start : start + length] = feat
            if native:
                raise ValueError(f"Unsupported native model inputs {sorted(native)}")
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None]
        with checkpoint_native_layers(core.model, self.activation_checkpointing):
            output = core.model(
                inputs_embeds=embeds, position_ids=positions, use_cache=False
            )
        return output.last_hidden_state

    def _score_frames(
        self,
        hidden,
        labels,
        active,
        support,
        phase,
        text_audio,
        block=None,
        *,
        score_raw=False,
    ):
        """Log probabilities of frames under the supports their rollout recorded.

        Every argument is flat over frames: ``hidden`` [N,H] predicts the
        actions ``labels`` [N,Q], sampled where ``active`` [N,Q]; ``support``
        [N] is the recorded support and ``phase``/``text_audio`` [N] the
        decoding state the current policy's own support needs. Returns the
        joint log probability and entropy [N] and the per-stream log
        probabilities [N,Q] from head blocks of ``block`` frames, recomputed in
        backward with score_checkpointing. A scoring pass (no gradient) counts
        the frames whose own support differs from the recorded one.
        """
        cfg = self.bagpiper_config
        frames, streams = labels.shape
        secondary = active[:, 1:].any(-1)
        if score_raw and (text_audio.any() or secondary.any()):
            raise ValueError("Raw WEST Bagpiper scoring requires text_only QA actions")
        if (active[:, 1:] != secondary[:, None]).any():
            # The seven secondary codec streams are sampled together or not.
            raise ValueError(
                "Bagpiper rollout stream mask differs from its recorded support"
            )
        vocab, primary = self.allowed.shape[1], labels[:, 0]
        valid = (primary >= 0) & (primary < vocab)
        valid &= self.allowed[support, primary.clamp(0, vocab - 1)]
        size = cfg["codec_layer_size"]
        codes = labels[:, 1:] - (
            cfg["codec_base_offset"]
            + size * torch.arange(1, streams, device=labels.device)
        )
        valid &= ((codes >= 0) & (codes < size)).all(-1) | ~secondary
        if not valid.all():
            raise ValueError(
                "Bagpiper rollout action has zero likelihood under its recorded support"
            )
        counting = not torch.is_grad_enabled()
        codec_rows = secondary.nonzero()[:, 0].cpu()
        block = block or self.token_chunk_size
        parts = []
        head = self._raw_head if score_raw else self._head
        for start in range(0, frames, block):
            end = min(frames, start + block)
            first, last = torch.searchsorted(
                codec_rows, torch.tensor([start, end])
            ).tolist()
            args = (
                hidden[start:end],
                labels[start:end],
                active[start:end],
                support[start:end],
                (codec_rows[first:last] - start).to(hidden.device),
            )
            if counting:
                parts.append(head(*args, phase[start:end], text_audio[start:end]))
            elif self.score_checkpointing:
                parts.append(checkpoint(head, *args, use_reentrant=False))
            else:
                parts.append(head(*args))
        stream_logp = torch.cat([part[0] for part in parts]) if parts else None
        if stream_logp is None:
            stream_logp = hidden.new_zeros((0, streams), dtype=torch.float32)
            entropy, expected = stream_logp[:, 0], active[:, 0]
        else:
            entropy = torch.cat([part[1] for part in parts])
            expected = torch.cat([part[2] for part in parts])
        if not torch.equal(expected, active[:, 0]):
            raise ValueError(
                "Bagpiper rollout stream mask differs from its recorded support"
            )
        if counting:
            self.support_mismatches += int(sum(part[3] for part in parts))
            self.support_frames += frames
        logp = stream_logp.sum(-1)
        if torch.is_grad_enabled():
            # Unsampled codec streams keep zero rather than absent offset
            # gradients, so optimizer state and weight decay stay unchanged.
            logp = logp + self.core.stream_emb.weight.sum() * 0
        return logp, entropy, stream_logp

    def _raw_head(
        self, hidden, labels, active, support, codec_rows, phase=None, text_audio=None
    ):
        """Raw full-vocabulary log probabilities for the literal WEST QA loss.

        The sampled-action mask still excludes forced native control frames.
        Temperature and protocol support masking belong only to the separate
        rollout-integrity pass; top-k is never part of this loss distribution.
        """
        if len(codec_rows) or active[:, 1:].any():
            raise ValueError("Raw WEST Bagpiper scoring requires text_only QA actions")
        core = self.core
        raw = F.linear(hidden, core.lm_head.weight, core.lm_head.bias)
        dist = raw.float().log_softmax(-1)
        sampled = active[:, 0]
        primary = torch.where(sampled, dist.gather(-1, labels[:, :1])[:, 0], 0.0)
        columns = torch.cat(
            (primary[:, None], primary.new_zeros((len(primary), labels.shape[1] - 1))),
            -1,
        )
        entropy = -(dist.exp() * dist).sum(-1)
        entropy = torch.where(sampled, entropy, 0.0)
        allowed = self.allowed[support] & torch.isfinite(raw)
        mismatches = None
        if phase is not None:
            own, codec = frame_supports(raw, phase, text_audio, self.bagpiper_config)
            mismatches = ((own != support) | codec).sum()
        return columns, entropy, allowed.sum(-1) > 1, mismatches

    def _head(
        self, hidden, labels, active, support, codec_rows, phase=None, text_audio=None
    ):
        """One block of frames under their recorded supports.

        Returns per-stream log probabilities, the joint entropy, the primary
        activity the supports give (more than one allowed finite logit) and,
        with the decoding state, how many frames the current logits give
        another support. The secondary streams are projected on their own
        codec layers for the rows that sampled them only.
        """
        cfg, core = self.bagpiper_config, self.core
        weight, bias = core.lm_head.weight, core.lm_head.bias
        raw = F.linear(hidden, weight, bias)
        logits = tempered(
            raw.float().masked_fill(~self.allowed[support], -torch.inf),
            self.temperature,
        )
        finite = torch.isfinite(logits)
        dist = logits.log_softmax(-1)
        sampled = active[:, 0]
        columns = [torch.where(sampled, dist.gather(-1, labels[:, :1])[:, 0], 0.0)]
        entropy = -(dist.exp() * dist.masked_fill(~finite, 0)).sum(-1)
        entropy = torch.where(sampled, entropy, 0.0)
        size = cfg["codec_layer_size"]
        codec = hidden[codec_rows]
        for stream in range(1, labels.shape[1]):
            column = torch.zeros_like(columns[0])
            if len(codec_rows):
                lo = cfg["codec_base_offset"] + stream * size
                scores = F.linear(
                    codec + core.stream_emb.weight[stream].to(codec.dtype),
                    weight[lo : lo + size],
                    None if bias is None else bias[lo : lo + size],
                )
                scores = tempered(scores.float(), self.temperature).log_softmax(-1)
                target = labels[codec_rows, stream : stream + 1] - lo
                column = column.index_copy(
                    0, codec_rows, scores.gather(-1, target)[:, 0]
                )
                entropy = entropy.index_add(
                    0, codec_rows, -(scores.exp() * scores).sum(-1)
                )
            columns.append(column)
        mismatches = None
        if phase is not None:
            own, codec_argmax = frame_supports(raw, phase, text_audio, cfg)
            recorded = (
                active[:, 1] if labels.shape[1] > 1 else torch.zeros_like(sampled)
            )
            mismatches = ((own != support) | (codec_argmax != recorded)).sum()
        return torch.stack(columns, -1), entropy, finite.sum(-1) > 1, mismatches


def _build_current_native_model(recipe, options):
    """Build the original ParallelLLM without importing a PP/TorchTitan trainer.

    Weights are loaded strictly immediately afterwards, so original pretrained
    text-row initialization is unnecessary. Vocabulary/IO construction follows
    SpeechLMJobTemplate._build_vocabulary in the pinned stable ESPnet release.
    """
    from transformers import AutoConfig

    from espnet2.speechlm.model.speechlm.lm.parallel import (
        build_parallel_hf_class,
        configure_moe_model,
    )
    from espnet2.speechlm.model.speechlm.multimodal_io.audio import (
        ContinuousAudioIO,
        DiscreteAudioIO,
    )
    from espnet2.speechlm.model.speechlm.multimodal_io.text import HuggingFaceTextIO

    choices = {
        "text": HuggingFaceTextIO,
        "discrete_audio": DiscreteAudioIO,
        "continuous_audio": ContinuousAudioIO,
    }
    io = {}
    for name, kwargs in recipe["multimodal_io"].items():
        if (
            name == "discrete_audio"
            and kwargs.get("codec_choice") == "Xcodec"
            and kwargs.get("ssl_choice") is None
        ):
            # Native checkpoints contain the entire codec, including its
            # semantic/acoustic submodels. Rebuild from geometry so the
            # existing strict loader supplies the final weights once.
            from transformers import XcodecModel

            tag = options.get("codec_config_path", kwargs["codec_hf_model_tag"])
            geometry = AutoConfig.from_pretrained(
                tag, revision=options.get("codec_config_revision")
            )
            handler = DiscreteAudioIO.__new__(DiscreteAudioIO)
            nn.Module.__init__(handler)
            handler.modality, handler.is_discrete = "audio", True
            handler.codec_choice = "Xcodec"
            handler.codec_hf_model_tag = kwargs["codec_hf_model_tag"]
            handler.codec_max_token_per_frame = kwargs.get(
                "codec_max_token_per_frame", 8
            )
            handler.codec_model = XcodecModel(geometry)
            handler.codec_n_streams = min(
                geometry.num_quantizers, handler.codec_max_token_per_frame
            )
            handler.codec_vocab_size = [
                geometry.codebook_size
            ] * handler.codec_n_streams
            handler.codec_sample_rate, handler.codec_frame_shift = (
                geometry.sample_rate,
                geometry.hop_length,
            )
            handler.codec_frame_per_second = geometry.frame_rate
            handler.codec_bandwidth = min(
                geometry.target_bandwidths,
                key=lambda item: abs(
                    item - handler.codec_n_streams * geometry.frame_rate * 10 / 1000
                ),
            )
            handler.use_codec, handler.use_ssl = True, False
            handler.ssl_choice = handler.ssl_hf_model_tag = None
            handler.delay_interleave = kwargs.get("delay_interleave", False)
            handler.stream_weights = kwargs.get("stream_weights")
            handler.device = "cpu"
            handler._init_ssl(None, None)
            handler._init_sanity_check()
            io[name] = handler
        elif (
            name == "continuous_audio"
            and kwargs.get("encoder_hf_model_tag") == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        ):
            # Every audio-tower parameter is supplied by the native checkpoint.
            # Construct only that exact tower, preserving the original module
            # names and get_audio_features implementation; no 30B LM download.
            from transformers import AutoFeatureExtractor
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeAudioEncoder,
                Qwen3OmniMoeThinkerForConditionalGeneration,
            )

            handler = ContinuousAudioIO.__new__(ContinuousAudioIO)
            nn.Module.__init__(handler)
            handler.modality, handler.is_discrete = "audio", False
            handler.device = "cpu"
            handler.encoder_choice = kwargs.get("encoder_choice", "huggingface")
            handler.encoder_hf_model_tag = kwargs["encoder_hf_model_tag"]
            handler.attn_implementation = options.get("native_attention", "sdpa")
            handler.dtype_str = kwargs.get("dtype", "bfloat16")
            handler.dtype = getattr(torch, handler.dtype_str)
            tag = options.get("audio_config_path", handler.encoder_hf_model_tag)
            geometry = AutoConfig.from_pretrained(
                tag, revision=options.get("audio_config_revision")
            ).thinker_config.audio_config
            geometry._attn_implementation = handler.attn_implementation

            class TensorAudioThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
                def get_audio_features(self, *args, **kwargs):
                    result = super().get_audio_features(*args, **kwargs)
                    if isinstance(result, torch.Tensor):
                        return result
                    # Transformers 5 returns a ModelOutput while the original
                    # ESPnet handler splits the flattened Tensor. Preserve its
                    # old get_audio_features interface and autograd graph.
                    features = getattr(result, "last_hidden_state", None)
                    if features is None:
                        features = getattr(result, "pooler_output", None)
                    if not isinstance(features, torch.Tensor):
                        raise TypeError("Qwen audio encoder returned no feature Tensor")
                    return features

            holder = TensorAudioThinker.__new__(TensorAudioThinker)
            nn.Module.__init__(holder)
            holder.audio_tower = Qwen3OmniMoeAudioEncoder(geometry)
            handler.model = holder
            processor_source = options.get(
                "audio_processor_path", handler.encoder_hf_model_tag
            )
            handler.processor = AutoFeatureExtractor.from_pretrained(
                processor_source, revision=options.get("audio_processor_revision")
            )
            handler.d_model = geometry.output_dim
            handler.sample_rate = handler.processor.sampling_rate
            handler.hop_length = handler.processor.hop_length
            io[name] = handler
        else:
            io[name] = choices[name](**kwargs)
    vocab = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|eot|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|text|>",
        "<|audio|>",
        "<|image|>",
        "<|video|>",
        "<|toolcall|>",
    ]
    vocab += [f"<|unused_{index}|>" for index in range(len(vocab), 256)]
    intervals, weights = {"special_token": [(0, 256)]}, [torch.ones(256)]
    mm_start, mm_end, num_stream = None, None, 1
    for name, handler in io.items():
        if not handler.is_discrete:
            continue
        start = len(vocab)
        vocab += handler.get_vocabulary()
        if name != "text":
            mm_start = start if mm_start is None else min(mm_start, start)
            mm_end = len(vocab) if mm_end is None else max(mm_end, len(vocab))
        intervals[name] = []
        for (lo, hi), weight in zip(
            handler.get_stream_interval(), handler.get_stream_weight(), strict=True
        ):
            intervals[name].append((start + lo, start + hi))
            weights.append(torch.ones(hi - lo) * weight)
        num_stream = max(num_stream, handler.num_stream())
    # Apply the current recipe's recorded-layout check without constructing
    # its supervised trainer or loading the model a second time.
    from types import SimpleNamespace

    from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

    SpeechLMJobTemplate._check_recorded_layout(
        SimpleNamespace(config=recipe), intervals
    )
    model_config = recipe["model"]
    if model_config.get("model_choice") != "parallel":
        raise ValueError(
            "Initial native backend requires a complete parallel model, not a pruned PP stage"
        )
    tag = model_config["model_hf_tag"]
    geometry = AutoConfig.from_pretrained(tag)
    geometry.vocab_size = len(vocab)
    geometry.tie_word_embeddings = model_config.get("model_conf", {}).get(
        "tie_word_embeddings", False
    )
    geometry._attn_implementation = options.get(
        "native_attention",
        model_config.get("model_conf", {}).get("attn_implementation", "sdpa"),
    )
    model = configure_moe_model(build_parallel_hf_class(tag)(geometry))
    # Native SpeechLM rebuilds the expanded embedding with padding_idx=0.
    # Preserve that gradient rule when constructing directly from geometry.
    model.model.embed_tokens.padding_idx = 0
    model.num_stream = num_stream
    model.stream_emb = nn.Embedding(num_stream, geometry.hidden_size)
    model.multimodal_io_dict = nn.ModuleDict(io)
    model.adaptor = nn.ModuleDict(
        {
            name: nn.Linear(handler.feature_dim(), geometry.hidden_size)
            for name, handler in io.items()
            if not handler.is_discrete
        }
    )
    model.vocab, model.vocab_intervals = vocab, intervals
    model.multimodal_vocab_range = (mm_start, mm_end)
    model.register_buffer("vocab_weight", torch.cat(weights).float())
    model.z_loss_weight = model_config.get("model_conf", {}).get("z_loss_weight", 0.0)
    return model


def factory(config, device):
    """Build Bagpiper/current SpeechLM with its original ESPnet recipe.

    Legacy OpusLM uses its pinned speechlm-branch S2T-independent task loader;
    load it in a separate environment so Python cannot mix two espnet2 packages.
    """
    import yaml

    if getattr(config, "family", None) != "bagpiper":
        raise ValueError("This adapter scores Bagpiper's deployed policy")
    path = Path(config.train_config)
    recipe = yaml.safe_load(path.read_text())
    if recipe.get("job_type") == "speechlm":
        model = _build_current_native_model(recipe, config.adapter_options)
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        state = state.get("module", state)
        model.load_state_dict(state, strict=True)
        frozen = recipe.get("trainer", {}).get("freeze_param", [])
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) for prefix in frozen):
                parameter.requires_grad_(False)
    else:
        raise ValueError("Bagpiper adapter requires its current native SpeechLM recipe")
    if config.adapter_options.get("reference_assets"):
        import json

        exported = json.loads(
            (
                Path(config.adapter_options["reference_assets"]) / "config.json"
            ).read_text()
        )
        geometry = _bagpiper_geometry(model)
        for field in (
            "text_token_offset",
            "text_token_end",
            "codec_base_offset",
            "codec_layer_size",
        ):
            if geometry[field] != exported[field]:
                raise ValueError(
                    f"Native recipe modality order differs from frozen token assets: {field} {geometry[field]} != {exported[field]}; preserve original YAML mapping order"
                )
    west_grpo = getattr(config, "training_objective", "miles_ppo") == "west_grpo"
    model = move_native_model(
        model,
        device,
        getattr(torch, config.dtype),
        preserve_decoder_rope_fp32=config.adapter_options.get(
            "native_decoder_rope_fp32", west_grpo
        ),
    ).eval()
    if recipe.get("model", {}).get("activation_checkpointing", False):
        model.model.gradient_checkpointing_enable()
    return NativeSpeechLMPolicy(
        model,
        _bagpiper_geometry(model),
        token_chunk_size=config.adapter_options.get("score_chunk_size", 32),
        activation_checkpointing=recipe.get("model", {}).get(
            "activation_checkpointing", False
        ),
        score_checkpointing=config.adapter_options.get("score_checkpointing", True),
        activation_cpu_offload=config.adapter_options.get(
            "activation_cpu_offload", False
        ),
        support_mismatch_rate=config.adapter_options.get("support_mismatch_rate", 1e-3),
        temperature=config.temperature,
        top_k=config.top_k,
        west_grpo=west_grpo,
    )


def _bagpiper_geometry(model):
    """Derive categorical layout from the actual native recipe vocabulary."""
    core = getattr(model, "corelm", model)
    text = core.vocab_intervals["text"]
    audio = core.vocab_intervals["discrete_audio"]
    if (
        len(text) != 1
        or len(audio) != core.num_stream
        or len({hi - lo for lo, hi in audio}) != 1
    ):
        raise ValueError(
            "Bagpiper recipe has incompatible per-stream vocabulary intervals"
        )
    if any(
        lo != audio[0][0] + index * (audio[0][1] - audio[0][0])
        for index, (lo, _) in enumerate(audio)
    ):
        raise ValueError("Bagpiper codec stream intervals must be contiguous")
    return {
        "text_token_offset": text[0][0],
        "text_token_end": text[0][1],
        "codec_base_offset": audio[0][0],
        "codec_layer_size": audio[0][1] - audio[0][0],
        **{
            key + "_token_id": core.vocab.index("<|" + key + "|>")
            for key in ("text", "audio", "eos", "eot", "assistant")
        },
    }
