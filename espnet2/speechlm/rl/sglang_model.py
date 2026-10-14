# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haoran Wang
"""ESPnet-owned SGLang implementations of OpusLM and OpusLM Dialogue.

The first post-training interface emits text while retaining the nine-stream
native audio condition. ESPnet supplies exact pad-cleared, summed codec/SSL
embeddings at prefill. Decode tokens use the native stream-zero embedding, and
all output scores include the released stream-zero head embedding. Codec output
needs a separate frame-state sampler and is deliberately rejected by this API.
"""

import torch
from torch import nn
from transformers import LlamaConfig, Olmo2Config

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.olmo2 import Olmo2ForCausalLM


class _OpusTextLogits(LogitsProcessor):
    def __init__(self, config, head_emb):
        super().__init__(config)
        # Register head_emb only on the enclosing model; duplicate module paths
        # would change loading and checkpoint accounting.
        object.__setattr__(self, "_head_emb", head_emb)
        self.text_start = int(config.text_token_start)
        self.text_end = int(config.text_token_end)
        self.allowed_ids = [int(config.eos_token_id), int(config.text_bpe_start_end_token_id)]

    def _get_logits(self, hidden_states, lm_head, logits_metadata):
        logits = super()._get_logits(hidden_states, lm_head, logits_metadata)
        allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        allowed[self.text_start:self.text_end] = True
        allowed[self.allowed_ids] = True
        return logits.masked_fill(~allowed, -torch.inf)

    def forward(self, input_ids, hidden_states, lm_head, *args, **kwargs):
        return super().forward(
            input_ids, hidden_states + self._head_emb.weight[0], lm_head, *args, **kwargs
        )


def _native_fp32_norms(model):
    # FlashInfer's CUDA norm kernels target half/BF16; retain SGLang's actual
    # built-in FP32 native RMSNorm for the published full-precision RL policy.
    for module in model.modules():
        if isinstance(module, RMSNorm) and module.weight.dtype == torch.float32:
            module._forward_method = module.forward_native


def _backbone_config(config, cls):
    data = config.to_dict()
    data.pop("model_type", None)
    return cls(**data)


class OpusLMForConditionalGeneration(Olmo2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix="", **kwargs):
        super().__init__(_backbone_config(config, Olmo2Config), quant_config, prefix, **kwargs)
        self.head_emb = nn.Embedding(12, config.hidden_size)
        self.logits_processor = _OpusTextLogits(config, self.head_emb)
        _native_fp32_norms(self)


class OpusLMDialogueForConditionalGeneration(LlamaForCausalLM):
    def __init__(self, config, quant_config=None, prefix="", **kwargs):
        super().__init__(_backbone_config(config, LlamaConfig), quant_config, prefix, **kwargs)
        self.head_emb = nn.Embedding(12, config.hidden_size)
        self.logits_processor = _OpusTextLogits(config, self.head_emb)
        _native_fp32_norms(self)


EntryClass = [OpusLMForConditionalGeneration, OpusLMDialogueForConditionalGeneration]
