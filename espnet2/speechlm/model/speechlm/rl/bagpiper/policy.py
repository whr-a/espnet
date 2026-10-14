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

The current SpeechLM training forward returns fused scalar CE; released OpusLM
ARDelay training forward also interleaves its input. Neither is a per-action RL
interface. This wrapper retains their embeddings, encoders, transformer, stream
offsets, and output projection, and exposes differentiable selected logprobs.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .tokens import selected_log_probs


from espnet2.rl.types import TokenScores


class NativeSpeechLMPolicy(nn.Module):
    """Score actual, already delayed SGLang actions; do not interleave them twice.

    Batch contract follows espnet2.rl.batch.RolloutBatch.
    Multistream rollouts additionally supply response_stream_mask [B,T,Q].
    ``source['native_model_inputs']`` carries the recipe's feature tensors and
    indices for its prompt. Prompts must have exactly the same multimodal token
    expansion as SGLang; silently re-tokenizing the audio would change the policy.
    """

    def __init__(
        self,
        model,
        layout="auto",
        token_chunk_size=32,
        stream_vocab_ranges=None,
        inference_semantics="native",
        legacy_codec_pad_embeddings=False,
        bagpiper_config=None,
    ):
        super().__init__()
        self.model = model
        core = getattr(model, "corelm", model)
        if layout == "auto":
            layout = "parallel" if hasattr(core, "stream_emb") else "ar_delay"
        if layout not in ("parallel", "ar_delay"):
            raise ValueError(f"Unsupported native SpeechLM layout {layout!r}")
        required = (
            ("model", "_embed", "stream_emb", "lm_head", "num_stream")
            if layout == "parallel"
            else ("process_embedding", "decoders", "head_emb", "lm_head", "nq")
        )
        if any(not hasattr(core, name) for name in required):
            raise ValueError(f"Model does not implement native {layout} components")
        if layout == "parallel" and getattr(core, "lm_head", None) is None:
            raise ValueError("A pruned pipeline stage is not a complete scoring policy")
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        self.layout = layout
        self.token_chunk_size = token_chunk_size
        self.num_stream = int(core.num_stream if layout == "parallel" else core.nq)
        self.stream_vocab_ranges = stream_vocab_ranges
        if inference_semantics not in ("native", "vllm_legacy_pad_bias"):
            raise ValueError("Unknown SpeechLM inference semantics")
        if layout != "ar_delay" and inference_semantics != "native":
            raise ValueError("Legacy pad-bias semantics applies only to OpusLM ARDelay")
        self.inference_semantics = inference_semantics
        self.legacy_codec_pad_embeddings = bool(legacy_codec_pad_embeddings)
        self.bagpiper_config = bagpiper_config
        if bagpiper_config is not None and layout != "parallel":
            raise ValueError("Bagpiper deployed likelihood requires a parallel model")

    @property
    def core(self):
        return getattr(self.model, "corelm", self.model)

    def checkpoint_state_dict(self):
        return self.model.state_dict()

    def _hidden(self, sequence, native_inputs):
        core = self.core
        core.eval()  # Disable dropout without disabling autograd.
        if self.layout == "parallel":
            kwargs = {"seqs": sequence, **native_inputs}
            for name, handler in core.multimodal_io_dict.items():
                features = kwargs.get(name + "_feats")
                parameter = next(handler.parameters(), None)
                if features is not None and parameter is not None and features.is_floating_point():
                    # Frontend artifacts remain fp32 in the manifest; convert
                    # only at the trainable native encoder boundary.
                    kwargs[name + "_feats"] = features.to(dtype=parameter.dtype)
            embeds = core._embed(sequence, kwargs)
            positions = kwargs.get("position_ids")
            if positions is None:
                positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
            # One utterance per forward avoids padding leakage and packed-example
            # attention without imposing a different native model backend.
            output = core.model(
                inputs_embeds=embeds,
                position_ids=positions,
                use_cache=False,
                **kwargs.get("attn_args", {}),
            )
            return output.last_hidden_state
        # Bypass ARDelayLM.forward: it would delay the actual rollout a second time.
        embeds = core.process_embedding(sequence, conti_feats=native_inputs.get("conti_feats"))
        if self.inference_semantics == "vllm_legacy_pad_bias":
            codec_mask = native_inputs.get("_rl_legacy_codec_mask")
            if codec_mask is None or codec_mask.shape != sequence.shape[:2]:
                raise ValueError("Legacy SGLang semantics needs exact codec_frame_mask")
            # Current published SGLang retains stream-zero emb(0) in flush frames
            # and adds (Q-1)*emb(0) to text positions. Native ARDelay clears
            # every pad embedding. Keep this policy choice explicit/frozen.
            pad_embed = core.emb.weight[core.pad_id]
            primary_pad = sequence[..., 0] == core.pad_id
            embeds = embeds + primary_pad[..., None] * pad_embed
            embeds = embeds + (~codec_mask)[..., None] * (self.num_stream - 1) * pad_embed
            if self.legacy_codec_pad_embeddings:
                # Dialogue's current SGLang buffer embeds padding in secondary
                # codec streams too; plain OpusLM clears these buffer pads.
                count_pad = (sequence[..., 1:] == core.pad_id).sum(-1)
                embeds = embeds + (codec_mask * count_pad)[..., None] * pad_embed
        decoder = core.decoders
        if getattr(decoder, "model_type", None) == "huggingface":
            # The legacy wrapper retains a KV cache across calls. Independent
            # teacher-forced utterances must use fresh attention state.
            return decoder.model(inputs_embeds=embeds, use_cache=False).last_hidden_state
        decoder.kv_cache = None
        return decoder(embeds)

    def _stream_weight(self, stream):
        core = self.core
        offsets = core.stream_emb.weight if self.layout == "parallel" else core.head_emb.weight
        if self.layout == "parallel" and stream == 0:
            return offsets[stream] * 0
        return offsets[stream]

    def _score_frames(self, hidden, labels, stream_mask, source, return_streams=False):
        b, t, q = labels.shape
        frame_logp = hidden.new_zeros((b, t), dtype=torch.float32)
        frame_entropy = hidden.new_zeros((b, t), dtype=torch.float32)
        stream_log_probs = []
        # Optional per-frame contracts are emitted by SGLang's trace hook. They
        # retain phase masks and sampling settings, including EOS suppression.
        contracts = source.get("sampling_contracts")
        phases = None
        if self.bagpiper_config is not None:
            if contracts is None:
                raise ValueError("Bagpiper deployed likelihood requires actual rollout contracts")
            mode = source.get("mode", source.get("sampling_extra_args", {}).get("mode", "text_only"))
            if mode not in ("text_only", "text_audio"):
                raise ValueError("Bagpiper deployed likelihood supports text_only/text_audio")
            phase = source.get("sampling_extra_args", {}).get("phase", "text")
            phases = []
            for token in labels[0, :, 0].tolist():
                if phase not in ("text", "transition", "audio", "audio_stop"):
                    raise ValueError("Unknown Bagpiper decoding phase")
                phases.append(phase)
                if mode == "text_audio":
                    if phase == "text" and token == self.bagpiper_config["eot_token_id"]:
                        phase = "transition"
                    elif phase == "transition":
                        phase = "audio"
                    elif phase == "audio" and token == self.bagpiper_config["eot_token_id"]:
                        phase = "audio_stop"
        for stream in range(q):
            logp_chunks, entropy_chunks = [], []
            for start in range(0, t, self.token_chunk_size):
                end = min(t, start + self.token_chunk_size)
                h = hidden[:, start:end] + self._stream_weight(stream).to(hidden.dtype)
                logits = F.linear(h, self.core.lm_head.weight, self.core.lm_head.bias)
                primary_logits = None
                if phases is not None:
                    primary_logits = F.linear(hidden[:, start:end], self.core.lm_head.weight, self.core.lm_head.bias)
                targets = labels[:, start:end, stream]
                active = stream_mask[:, start:end, stream]
                if contracts is not None:
                    chunks, ents = [], []
                    for j in range(end - start):
                        spec = contracts[start + j][stream]
                        row = logits[:, j : j + 1]
                        if phases is not None:
                            # SGLang normalizes an unlimited internal sampler's
                            # top_k to the full vocabulary size. That value
                            # keeps every logit, including the current modality
                            # mask, and is not a truncated policy distribution.
                            top_k = spec.get("top_k", 0)
                            if spec.get("temperature", 1.0) != 1.0 or 0 < top_k < row.shape[-1]:
                                raise ValueError("Bagpiper native RL requires untruncated unit-temperature sampling")
                            row = self._bagpiper_deployed_row(
                                row,
                                primary_logits[:, j : j + 1],
                                stream,
                                phases[start + j],
                                mode,
                                targets[:, j : j + 1],
                                active[:, j : j + 1],
                            )
                        elif spec.get("allowed_ranges") is not None:
                            allowed = torch.zeros(row.shape[-1], device=row.device, dtype=torch.bool)
                            for lo, hi in spec["allowed_ranges"]:
                                allowed[lo:hi] = True
                            for token in spec.get("allowed_token_ids", []):
                                allowed[token] = True
                            row = row.masked_fill(~allowed, -torch.inf)
                        for token in spec.get("excluded_token_ids", []):
                            row = row.clone()
                            row[..., token] = -torch.inf
                        lp, ent = selected_log_probs(
                            row,
                            targets[:, j : j + 1],
                            active[:, j : j + 1],
                            spec.get("temperature", 1.0),
                            spec.get("top_k", 0),
                        )
                        chunks.append(lp)
                        ents.append(ent)
                    logp, entropy = torch.cat(chunks, 1), torch.cat(ents, 1)
                else:
                    # This is an unwarped native policy distribution. Supply
                    # contracts to reproduce restricted/temperature sampling.
                    if self.stream_vocab_ranges is not None:
                        lo, hi = self.stream_vocab_ranges[stream]
                        allowed = torch.arange(logits.shape[-1], device=logits.device)
                        logits = logits.masked_fill((allowed < lo) | (allowed >= hi), -torch.inf)
                    logp, entropy = selected_log_probs(logits, targets, active)
                logp_chunks.append(logp)
                entropy_chunks.append(entropy)
            if logp_chunks:
                stream_logp = torch.cat(logp_chunks, 1)
                frame_logp = frame_logp + stream_logp
                frame_entropy = frame_entropy + torch.cat(entropy_chunks, 1)
                stream_log_probs.append(stream_logp)
            else:
                stream_log_probs.append(hidden.new_zeros((b, t), dtype=torch.float32))
        if return_streams:
            return frame_logp, frame_entropy, torch.stack(stream_log_probs, -1)
        return frame_logp, frame_entropy

    def _bagpiper_deployed_row(self, logits, raw_primary, stream, phase, mode, target, recorded_active):
        """Recompute deployed support from current logits, never old support.

        A changed categorical support gives zero likelihood to an excluded
        action. The current finite-logprob GRPO backend cannot optimize that
        boundary; reject it explicitly rather than substituting the old mask.
        """
        cfg = self.bagpiper_config
        tentative = int(raw_primary[0, 0].argmax())
        detect = tentative in (cfg["text_token_id"], cfg["audio_token_id"])
        audio = cfg["codec_base_offset"] <= tentative < logits.shape[-1] and not detect
        allowed = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
        if stream:
            lo = cfg["codec_base_offset"] + stream * cfg["codec_layer_size"]
            allowed[lo : lo + cfg["codec_layer_size"]] = True
            expected_active = audio
        else:
            if mode == "text_audio" and phase in ("transition", "audio_stop"):
                forced = cfg["assistant_token_id"] if phase == "transition" else cfg["eos_token_id"]
                allowed[forced] = True
            elif detect:
                allowed[cfg["text_token_id"]] = True
                if mode == "text_audio" and phase == "audio":
                    allowed[cfg["audio_token_id"]] = True
            elif audio:
                lo = cfg["codec_base_offset"]
                allowed[lo : lo + cfg["codec_layer_size"]] = True
                allowed[cfg["eos_token_id"]] = allowed[cfg["eot_token_id"]] = True
            else:
                allowed[cfg["text_token_offset"] : cfg["text_token_end"]] = True
                allowed[cfg["eos_token_id"]] = allowed[cfg["eot_token_id"]] = True
            expected_active = int((allowed & torch.isfinite(logits[0, 0])).sum()) > 1
        if bool(recorded_active[0, 0]) != expected_active:
            raise ValueError("Bagpiper current policy changed the sampled-stream action layout")
        if not stream or expected_active:
            if not 0 <= int(target[0, 0]) < len(allowed) or not allowed[int(target[0, 0])]:
                raise ValueError("Bagpiper rollout action has zero likelihood under current deployed support")
        return logits.masked_fill(~allowed, -torch.inf)

    def forward(self, batch):
        device = next(self.model.parameters()).device
        responses = batch.batch["responses"].to(device)
        if responses.ndim == 2:
            if self.num_stream != 1:
                raise ValueError("Multistream policy requires responses [B,T,Q], including all sampled codebooks")
            responses = responses.unsqueeze(-1)
        if responses.ndim != 3 or responses.shape[-1] != self.num_stream:
            raise ValueError("Response stream count differs from native model")
        masks = batch.batch["response_mask"].to(device).bool()
        lengths = batch.batch["response_lengths"].tolist()
        stream_masks = batch.batch.get("response_stream_mask")
        if stream_masks is None:
            if self.num_stream > 1:
                raise ValueError("Multistream RL requires an explicit response_stream_mask")
            stream_masks = masks.unsqueeze(-1)
        stream_masks = stream_masks.to(device).bool()
        if stream_masks.shape != responses.shape or masks.shape != responses.shape[:2]:
            raise ValueError("Response mask shape mismatch")
        lp_rows, entropy_rows, mask_rows, stream_rows = [], [], [], []
        for i, length in enumerate(lengths):
            if length < 0 or length > responses.shape[1]:
                raise ValueError("Invalid response length")
            prefix = torch.as_tensor(
                batch.non_tensor_batch["prompt_token_ids"][i],
                device=device,
                dtype=torch.long,
            )
            if prefix.ndim == 1:
                prefix = torch.cat(
                    (
                        prefix[:, None],
                        torch.zeros(
                            prefix.shape[0],
                            self.num_stream - 1,
                            device=device,
                            dtype=torch.long,
                        ),
                    ),
                    -1,
                )
            if prefix.ndim != 2 or prefix.shape[1] != self.num_stream or not prefix.shape[0]:
                raise ValueError("Expected a nonempty prompt [P,Q]")
            source = batch.non_tensor_batch["source"][i]
            native = _native_model_inputs(source.get("native_model_inputs", {}), device)
            sequence = torch.cat((prefix, responses[i, :length]), 0)[None].clone()
            if self.inference_semantics == "vllm_legacy_pad_bias":
                codec_frames = source.get("codec_frame_mask")
                if codec_frames is None or len(codec_frames) != length:
                    raise ValueError("Legacy SGLang scoring requires recorded codec_frame_mask")
                prompt_codec = source.get("prompt_codec_mask", [False] * prefix.shape[0])
                if len(prompt_codec) != prefix.shape[0]:
                    raise ValueError("prompt_codec_mask must cover the native prompt")
                native["_rl_legacy_codec_mask"] = torch.as_tensor(
                    [*prompt_codec, *codec_frames], device=device, dtype=torch.bool
                )[None]
            if "position_ids" in native and native["position_ids"].shape[-1] != sequence.shape[1]:
                raise ValueError("position_ids must include the full prompt and response")
            hidden = self._hidden(sequence, native)
            # hidden[P-1] predicts the first response; the final response's
            # hidden is not used. Target rows retain their rollout delay layout.
            hidden = hidden[:, prefix.shape[0] - 1 : prefix.shape[0] + length - 1]
            active = stream_masks[i : i + 1, :length] & masks[i : i + 1, :length, None]
            logp, entropy, stream_logp = self._score_frames(
                hidden, responses[i : i + 1, :length], active, source, return_streams=True
            )
            padding = responses.shape[1] - length
            lp_rows.append(F.pad(logp, (0, padding)))
            entropy_rows.append(F.pad(entropy, (0, padding)))
            mask_rows.append(F.pad(active.any(-1), (0, padding)))
            stream_rows.append(F.pad(stream_logp, (0, 0, 0, padding)))
        return TokenScores(torch.cat(lp_rows), torch.cat(entropy_rows), torch.cat(mask_rows), torch.cat(stream_rows))


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _native_model_inputs(value, device):
    """Decode JSON-compatible native recipe features into actual tensors."""
    result = _to_device(value, device)
    for name, item in result.items():
        if name == "position_ids" or name.endswith(("_indices", "_lengths")):
            result[name] = torch.as_tensor(item, device=device, dtype=torch.long)
        elif name.endswith("_feats") and name != "conti_feats":
            result[name] = torch.as_tensor(item, device=device, dtype=torch.float32)
        elif name == "attn_args":
            result[name] = {
                key: torch.as_tensor(array, device=device, dtype=torch.int32)
                if key.startswith("cu_seq_lens")
                else array
                for key, array in item.items()
            }
        elif name == "conti_feats":
            result[name] = [
                [
                    (torch.as_tensor(feats, device=device, dtype=torch.float32), int(start), int(end), part)
                    for feats, start, end, part in utterance
                ]
                for utterance in item
            ]
    return result


def _build_current_native_model(recipe, options):
    """Build the original ParallelLLM without importing a PP/TorchTitan trainer.

    Weights are loaded strictly immediately afterwards, so original pretrained
    text-row initialization is unnecessary. Vocabulary/IO construction follows
    SpeechLMJobTemplate._build_vocabulary in the pinned stable ESPnet release.
    """
    from espnet2.speechlm.model.speechlm.lm.parallel import (
        build_parallel_hf_class,
        configure_moe_model,
    )
    from espnet2.speechlm.model.speechlm.multimodal_io.audio import (
        ContinuousAudioIO,
        DiscreteAudioIO,
    )
    from espnet2.speechlm.model.speechlm.multimodal_io.text import HuggingFaceTextIO
    from transformers import AutoConfig

    choices = {
        "text": HuggingFaceTextIO,
        "discrete_audio": DiscreteAudioIO,
        "continuous_audio": ContinuousAudioIO,
    }
    io = {}
    for name, kwargs in recipe["multimodal_io"].items():
        if name == "discrete_audio" and kwargs.get("codec_choice") == "Xcodec" and kwargs.get("ssl_choice") is None:
            # Native checkpoints contain the entire codec, including its
            # semantic/acoustic submodels. Rebuild from geometry so the
            # existing strict loader supplies the final weights once.
            from transformers import XcodecModel

            tag = options.get("codec_config_path", kwargs["codec_hf_model_tag"])
            geometry = AutoConfig.from_pretrained(tag, revision=options.get("codec_config_revision"))
            handler = DiscreteAudioIO.__new__(DiscreteAudioIO)
            nn.Module.__init__(handler)
            handler.modality, handler.is_discrete = "audio", True
            handler.codec_choice = "Xcodec"
            handler.codec_hf_model_tag = kwargs["codec_hf_model_tag"]
            handler.codec_max_token_per_frame = kwargs.get("codec_max_token_per_frame", 8)
            handler.codec_model = XcodecModel(geometry)
            handler.codec_n_streams = min(geometry.num_quantizers, handler.codec_max_token_per_frame)
            handler.codec_vocab_size = [geometry.codebook_size] * handler.codec_n_streams
            handler.codec_sample_rate, handler.codec_frame_shift = geometry.sample_rate, geometry.hop_length
            handler.codec_frame_per_second = geometry.frame_rate
            handler.codec_bandwidth = min(
                geometry.target_bandwidths,
                key=lambda item: abs(item - handler.codec_n_streams * geometry.frame_rate * 10 / 1000),
            )
            handler.use_codec, handler.use_ssl = True, False
            handler.ssl_choice = handler.ssl_hf_model_tag = None
            handler.delay_interleave = kwargs.get("delay_interleave", False)
            handler.stream_weights = kwargs.get("stream_weights")
            handler.device = "cpu"
            handler._init_ssl(None, None)
            handler._init_sanity_check()
            io[name] = handler
        elif name == "continuous_audio" and kwargs.get("encoder_hf_model_tag") == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
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
            processor_source = options.get("audio_processor_path", handler.encoder_hf_model_tag)
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
        for (lo, hi), weight in zip(handler.get_stream_interval(), handler.get_stream_weight(), strict=True):
            intervals[name].append((start + lo, start + hi))
            weights.append(torch.ones(hi - lo) * weight)
        num_stream = max(num_stream, handler.num_stream())
    model_config = recipe["model"]
    if model_config.get("model_choice") != "parallel":
        raise ValueError("Initial native backend requires a complete parallel model, not a pruned PP stage")
    tag = model_config["model_hf_tag"]
    geometry = AutoConfig.from_pretrained(tag)
    geometry.vocab_size = len(vocab)
    geometry.tie_word_embeddings = model_config.get("model_conf", {}).get("tie_word_embeddings", False)
    geometry._attn_implementation = options.get(
        "native_attention",
        model_config.get("model_conf", {}).get("attn_implementation", "sdpa"),
    )
    model = configure_moe_model(build_parallel_hf_class(tag)(geometry))
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
        exported=json.loads((Path(config.adapter_options["reference_assets"])/"config.json").read_text())
        geometry=_bagpiper_geometry(model)
        for field in ("text_token_offset","text_token_end","codec_base_offset","codec_layer_size"):
            if geometry[field] != exported[field]:
                raise ValueError(f"Native recipe modality order differs from frozen token assets: {field} {geometry[field]} != {exported[field]}; preserve original YAML mapping order")
    model = model.to(device=device, dtype=getattr(torch, config.dtype)).eval()
    if recipe.get("model", {}).get("activation_checkpointing", False):
        model.model.gradient_checkpointing_enable()
    return NativeSpeechLMPolicy(
        model,
        inference_semantics=config.adapter_options.get("inference_semantics", "native"),
        legacy_codec_pad_embeddings=False,
        bagpiper_config=_bagpiper_geometry(model) if getattr(config, "family", None) == "bagpiper" else None,
    )


def _bagpiper_geometry(model):
    """Derive categorical layout from the actual native recipe vocabulary."""
    core = getattr(model, "corelm", model)
    text = core.vocab_intervals["text"]
    audio = core.vocab_intervals["discrete_audio"]
    if len(text) != 1 or len(audio) != core.num_stream or len({hi - lo for lo, hi in audio}) != 1:
        raise ValueError("Bagpiper recipe has incompatible per-stream vocabulary intervals")
    if any(lo != audio[0][0] + index * (audio[0][1] - audio[0][0]) for index, (lo, _) in enumerate(audio)):
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
