# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Differentiable native OpusLM probabilities for exact SGLang actions.

The released ARDelay forward interleaves inputs and returns scalar CE. RL must
score actions in their already delayed rollout layout, using the same original
embeddings, decoder and per-stream output head, without delaying a second time.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.rl.types import TokenScores

from .tokens import selected_log_probs


class NativeSpeechLMPolicy(nn.Module):
    def __init__(self, model, token_chunk_size=32, inference_semantics="native"):
        super().__init__()
        if inference_semantics != "native":
            raise ValueError(
                "This SGLang interface requires the native pad-cleared policy"
            )
        self.model = model
        self.num_stream = int(model.corelm.nq)
        self.token_chunk_size = token_chunk_size
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")

    @property
    def core(self):
        return self.model.corelm

    def checkpoint_state_dict(self):
        return self.model.state_dict()

    def _hidden(self, sequence):
        core = self.core
        core.eval()  # Keep autograd while disabling stochastic dropout.
        if getattr(self, "gradient_checkpointing", False) and torch.is_grad_enabled():
            # HF gates activation checkpointing on training; no model dropout
            # is allowed by the released recipe used for this RL policy.
            core.decoders.model.train()
        embeddings = core.process_embedding(sequence)
        decoder = core.decoders
        if decoder.model_type != "huggingface":
            raise ValueError("Released Opus RL requires its HF-backed native decoder")
        return decoder.model(
            inputs_embeds=embeddings, use_cache=False
        ).last_hidden_state

    def _scores(self, hidden, labels, active, contracts):
        streams, entropies = [], []
        for stream in range(self.num_stream):
            if not bool(active[..., stream].any()):
                streams.append(hidden.new_zeros(labels.shape[:2], dtype=torch.float32))
                entropies.append(
                    hidden.new_zeros(labels.shape[:2], dtype=torch.float32)
                )
                continue
            probabilities, entropy = [], []
            for start in range(0, labels.shape[1], self.token_chunk_size):
                end = min(labels.shape[1], start + self.token_chunk_size)
                logits = F.linear(
                    hidden[:, start:end] + self.core.head_emb.weight[stream],
                    self.core.lm_head.weight,
                )
                for frame in range(start, end):
                    row = logits[:, frame - start : frame - start + 1]
                    spec = contracts[frame][stream]
                    if spec.get("allowed_ranges") is not None:
                        support = torch.zeros(
                            row.shape[-1], device=row.device, dtype=torch.bool
                        )
                        for low, high in spec["allowed_ranges"]:
                            support[low:high] = True
                        support[spec.get("allowed_token_ids", [])] = True
                        row = row.masked_fill(~support, -torch.inf)
                    for token in spec.get("excluded_token_ids", []):
                        row = row.clone()
                        row[..., token] = -torch.inf
                    logp, ent = selected_log_probs(
                        row,
                        labels[:, frame : frame + 1, stream],
                        active[:, frame : frame + 1, stream],
                        spec.get("temperature", 1.0),
                        spec.get("top_k", 0),
                    )
                    probabilities.append(logp)
                    entropy.append(ent)
            streams.append(torch.cat(probabilities, 1))
            entropies.append(torch.cat(entropy, 1))
        stream_logp = torch.stack(streams, -1)
        return stream_logp.sum(-1), torch.stack(entropies, -1).sum(-1), stream_logp

    def forward(self, batch):
        device = next(self.parameters()).device
        responses = batch.batch["responses"].to(device)
        mask = batch.batch["response_mask"].to(device).bool()
        stream_mask = batch.batch["response_stream_mask"].to(device).bool()
        if responses.ndim != 3 or responses.shape[-1] != self.num_stream:
            raise ValueError("Opus responses must retain every delayed stream [B,T,9]")
        if stream_mask.shape != responses.shape or mask.shape != responses.shape[:2]:
            raise ValueError("Actual response action-mask shape mismatch")
        rows, ents, masks, streams = [], [], [], []
        for i, length in enumerate(batch.batch["response_lengths"].tolist()):
            if not 0 < length <= responses.shape[1]:
                raise ValueError("Invalid actual response length")
            prefix = torch.as_tensor(
                batch.non_tensor_batch["prompt_token_ids"][i],
                device=device,
                dtype=torch.long,
            )
            if (
                prefix.ndim != 2
                or prefix.shape[1] != self.num_stream
                or not len(prefix)
            ):
                raise ValueError("Opus prefix must preserve all native streams [P,9]")
            source = batch.non_tensor_batch["source"][i]
            if source.get("native_model_inputs"):
                raise ValueError(
                    "This Opus interface accepts discrete codec/SSL conditions"
                )
            contracts = source["sampling_contracts"]
            if len(contracts) != length or any(
                len(row) != self.num_stream for row in contracts
            ):
                raise ValueError(
                    "Every recorded action stream needs a sampling contract"
                )
            sequence = torch.cat((prefix, responses[i, :length]))[None]
            hidden = self._hidden(sequence)[
                :, len(prefix) - 1 : len(prefix) + length - 1
            ]
            active = stream_mask[i : i + 1, :length] & mask[i : i + 1, :length, None]
            logp, entropy, stream_logp = self._scores(
                hidden, responses[i : i + 1, :length], active, contracts
            )
            padding = responses.shape[1] - length
            rows.append(F.pad(logp, (0, padding)))
            ents.append(F.pad(entropy, (0, padding)))
            masks.append(F.pad(active.any(-1), (0, padding)))
            streams.append(F.pad(stream_logp, (0, 0, 0, padding)))
        return TokenScores(
            torch.cat(rows), torch.cat(ents), torch.cat(masks), torch.cat(streams)
        )


def factory(config, device):
    """Strictly load the released native ARDelay architecture inside ESPnet."""
    import yaml

    path = Path(config.train_config)
    recipe = yaml.safe_load(path.read_text())
    from transformers import AutoConfig

    from .model import OriginalOpusModel

    tag = recipe.get("transformer_conf", {}).get("hf_model_tag") or recipe.get(
        "corelm_conf", {}
    ).get("hf_model_tag")
    if not tag or recipe.get("corelm") != "ar_delay":
        raise ValueError(
            "Opus RL requires the released HF-backed ARDelay configuration"
        )
    geometry = AutoConfig.from_pretrained(tag)
    geometry._attn_implementation = config.adapter_options.get(
        "native_attention", "sdpa"
    )
    tokens = recipe["token_list"]
    if isinstance(tokens, str):
        import json

        tokens = json.loads((path.parent / tokens).read_text())
    model = OriginalOpusModel(
        geometry,
        tokens,
        recipe["codec_token_in_use"],
        recipe.get("corelm_conf", {}).get("share_emb", False),
    )
    state = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("module", state)
    # Criterion weights are not independent inference parameters. Every
    # original corelm parameter remains mandatory and shape checked.
    state = {
        key: value for key, value in state.items() if not key.startswith("criterion.")
    }
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=getattr(torch, config.dtype)).eval()
    policy = NativeSpeechLMPolicy(model, inference_semantics="native")
    policy.gradient_checkpointing = config.adapter_options.get(
        "gradient_checkpointing", False
    )
    if policy.gradient_checkpointing:
        if any(
            getattr(module, "p", 0) > 0
            for module in model.modules()
            if isinstance(module, nn.Dropout)
        ):
            raise ValueError(
                "Activation checkpointing requires a dropout-free released decoder"
            )
        model.corelm.decoders.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return policy
