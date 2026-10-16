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
"""Reuse ESPnet's frontend, encoder and teacher-forced decoder for RL."""

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .language import resolve_owsm_language_symbol


def resolve_callable(name: str):
    module, symbol = name.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


from espnet2.rl.types import NativeESPnetPolicy, TokenScores, tempered


def object_array(values):
    # np.array(list_of_equal_length_lists) creates an unwanted extra batch axis.
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def load_audio(
    source: dict[str, Any], sample_rate: int, window_samples: int | None = None
):
    """Use the same deterministic mono, rate and window contract as rollout."""
    if "audio_array" in source:
        waveform = np.asarray(source["audio_array"], dtype=np.float32)
        rate = int(source.get("sample_rate", sample_rate))
    else:
        import soundfile as sf

        waveform, rate = sf.read(
            Path(source["audio"]), dtype="float32", always_2d=False
        )
    if rate != sample_rate:
        raise ValueError(
            f"Audio rate {rate} differs from model rate {sample_rate}; resample the manifest offline"
        )
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    if window_samples is not None:
        # Match the vLLM processor: long inputs need explicit offline segmentation.
        if len(waveform) > window_samples:
            raise ValueError(
                f"Audio has {len(waveform)} samples, exceeding the OWSM window of {window_samples}"
            )
        waveform = np.pad(waveform, (0, window_samples - len(waveform)))
    return waveform


def owsm_symbol_id(token_list, value, *, language=False):
    """Resolve actual OWSM symbols with the same table as the vLLM endpoint."""
    ids = {token: index for index, token in enumerate(token_list)}
    if value in ids:
        return ids[value]
    if language or value.strip().lower().strip("<>").startswith("st_"):
        return ids[resolve_owsm_language_symbol(token_list, value)]
    raise ValueError(f"OWSM checkpoint vocabulary lacks task symbol {value!r}")


class OWSMPolicy(NativeESPnetPolicy):
    def __init__(
        self,
        model: nn.Module,
        args=None,
        train_encoder: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.args = args
        # Actions are sampled from softmax(logits / temperature).
        self.temperature = temperature
        if getattr(model, "decoder", None) is None:
            raise ValueError(
                "OWSM-CTC is a distinct policy; this backend requires an autoregressive decoder"
            )
        preprocessing = getattr(args, "preprocessor_conf", None) or {}
        self.sample_rate = int(preprocessing.get("fs", 16000))
        seconds = preprocessing.get("speech_length")
        self.window_samples = (
            int(seconds * self.sample_rate) if seconds is not None else None
        )
        self.token_list = list(getattr(model, "token_list", []))
        if not train_encoder:
            for name, parameter in self.model.named_parameters():
                if not name.startswith("decoder."):
                    parameter.requires_grad_(False)
        # CTC and auxiliary losses are not part of the AR GRPO objective.
        if getattr(model, "ctc", None) is not None:
            for parameter in model.ctc.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_config(cls, config: Any, device: str):
        from espnet2.tasks.s2t import S2TTask

        with Path(config.train_config).open() as stream:
            args = argparse.Namespace(**yaml.safe_load(stream))
        # build_model_from_file currently loads strict=False and can silently
        # miss old frontend key names. An RL actor must load every trained tensor.
        model = S2TTask.build_model(args)
        state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
        for wrapper in ("module", "state_dict", "model"):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
        target = model.state_dict()
        normalized = {}
        for name, tensor in state.items():
            mapped = name
            if name not in target:
                mapped = mapped.replace(".embed.out.0.", ".embed.out.")
                mapped = mapped.replace(".embed.out.1.", ".embed.pos_enc.")
            if mapped in normalized:
                raise ValueError(
                    f"Duplicate checkpoint key after normalization: {mapped}"
                )
            normalized[mapped] = tensor
        model.load_state_dict(normalized, strict=True)
        # Never cast the whole model to bf16/fp16: restoring frontend buffers
        # to float32 afterwards cannot recover the lost mel/statistic precision.
        model = model.to(device=device)
        for name in ("frontend", "normalize"):
            component = getattr(model, name, None)
            if component is not None:
                component.float()
        for name in ("preencoder", "encoder", "postencoder", "decoder", "ctc"):
            component = getattr(model, name, None)
            if component is not None:
                component.to(dtype=getattr(torch, config.dtype))
        return cls(model, args, config.train_encoder, config.temperature)

    @staticmethod
    def _cast_features(features, component):
        parameter = next(component.parameters(), None)
        return features.to(parameter.dtype) if parameter is not None else features

    def _encode(self, speech, speech_lengths):
        # Preserve ESPnet's native feature/encoder path and encoder autograd,
        # while explicitly crossing its fp32 frontend -> neural dtype boundary.
        # SpecAugment is excluded from all policy scoring, even in train mode.
        with torch.autocast(device_type=speech.device.type, enabled=False):
            features, lengths = self.model._extract_feats(
                speech.float(), speech_lengths
            )
            if self.model.normalize is not None:
                features, lengths = self.model.normalize(features, lengths)
        if self.model.preencoder is not None:
            features, lengths = self.model.preencoder(
                self._cast_features(features, self.model.preencoder), lengths
            )
        features = self._cast_features(features, self.model.encoder)
        if self.model.encoder.interctc_use_conditioning:
            encoded, lengths, _ = self.model.encoder(
                features, lengths, ctc=self.model.ctc
            )
        else:
            encoded, lengths, _ = self.model.encoder(features, lengths)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        if self.model.postencoder is not None:
            encoded, lengths = self.model.postencoder(
                self._cast_features(encoded, self.model.postencoder), lengths
            )
        return encoded, lengths

    def build_prefix(self, source: dict[str, Any], config: Any):
        if "prompt_token_ids" in source:
            prefix = list(source["prompt_token_ids"])
            if not prefix:
                raise ValueError("OWSM decoder prefix must be nonempty")
            return prefix
        ids = {token: index for index, token in enumerate(self.token_list)}
        prefix = [
            self.model.sos,
            owsm_symbol_id(
                self.token_list, source.get("language", config.language), language=True
            ),
            owsm_symbol_id(self.token_list, source.get("task", config.task)),
        ]
        if not source.get("predict_timestamps", config.predict_timestamps):
            prefix.append(ids["<notimestamps>"])
        previous = source.get("previous_token_ids")
        if previous:
            prefix = [self.model.sop, *previous, *prefix]
        return prefix

    def forward(self, batch: Any):
        device = next(self.model.parameters()).device
        responses = batch.batch["responses"].to(device)
        mask = batch.batch["response_mask"].to(device)
        lengths = batch.batch["response_lengths"].tolist()
        if responses.ndim != 2:
            raise ValueError("OWSM actions must be text token IDs [B,T]")
        sources = batch.non_tensor_batch["source"]
        if all("encoder_features" in source for source in sources):
            # Explicit precomputed memory is useful for decoder-only training.
            # The manifest must identify how features were generated.
            memories = [
                torch.as_tensor(s["encoder_features"], device=device) for s in sources
            ]
            memory = pad_sequence(memories, batch_first=True)
            memory_lens = torch.tensor([len(s) for s in memories], device=device)
        else:
            speech = [
                torch.as_tensor(
                    load_audio(s, self.sample_rate, self.window_samples), device=device
                )
                for s in sources
            ]
            speech_lens = torch.tensor([len(s) for s in speech], device=device)
            speech = pad_sequence(speech, batch_first=True)
            memory, memory_lens = self._encode(speech, speech_lens)
        # Also cover fp32 memory supplied explicitly for decoder-only training.
        memory = self._cast_features(memory, self.model.decoder)
        prefixes = batch.non_tensor_batch["prompt_token_ids"]
        decoder_inputs = []
        for row, prefix, length in zip(responses, prefixes, lengths, strict=True):
            prefix = torch.as_tensor(prefix, dtype=torch.long, device=device)
            if not len(prefix):
                raise ValueError("Empty decoder prompt")
            decoder_inputs.append(torch.cat((prefix, row[: length - 1])))
        decoder_lens = torch.tensor([len(row) for row in decoder_inputs], device=device)
        decoder_input = pad_sequence(
            decoder_inputs, batch_first=True, padding_value=self.model.eos
        )
        logits, _ = self.model.decoder(memory, memory_lens, decoder_input, decoder_lens)
        # A prefix of P tokens predicts response[0] at decoder output P-1.
        positions = torch.arange(responses.size(1), device=device).unsqueeze(0)
        offsets = torch.tensor([len(p) - 1 for p in prefixes], device=device).unsqueeze(
            1
        )
        positions = (positions + offsets).clamp_max(logits.size(1) - 1)
        logits = logits.gather(1, positions[..., None].expand(-1, -1, logits.size(-1)))
        distribution = torch.log_softmax(
            tempered(logits.float(), self.temperature), dim=-1
        )
        log_probs = distribution.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
        entropy = -(distribution.exp() * distribution).sum(-1)
        return TokenScores(log_probs * mask, entropy * mask, mask)


def build_policy(config: Any, device: str):
    """Miles factory hook for the native autoregressive OWSM policy."""
    policy = OWSMPolicy.from_config(config, device)
    policy.eval()
    return policy
