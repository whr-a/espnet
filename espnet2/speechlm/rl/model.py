# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Released OpusLM ARDelay components as an ESPnet model library.

Keep the checkpoint's original component and parameter names while using the
published HF decoder. This avoids importing an entire archived ESPnet tree or
a DPO trainer merely to expose the released model's inference probabilities.
"""

import torch
from torch import nn


class OriginalOpusARDelay(nn.Module):
    def __init__(self, geometry, token_list, num_streams, share_embeddings=False):
        super().__init__()
        from transformers import AutoModel

        self.emb = nn.Embedding(len(token_list), geometry.hidden_size)
        self.lm_head = nn.Linear(geometry.hidden_size, len(token_list), bias=False)
        if share_embeddings:
            self.lm_head.weight = self.emb.weight
        self.head_emb = nn.Embedding(12, geometry.hidden_size)
        self.nq = int(num_streams)
        self.n_ctx = geometry.max_position_embeddings
        self.sos_eos = token_list.index("<sos/eos>")
        self.pad_id = token_list.index("<pad>")
        self.decoders = nn.Module()
        # A backbone config may declare half precision (SmolLM: bfloat16),
        # which from_config honours: loading an FP32 checkpoint would then
        # round every decoder weight. Build in FP32 so loads are exact; the
        # policy factory casts to the recipe dtype afterwards.
        self.decoders.model = AutoModel.from_config(geometry, dtype=torch.float32)
        self.decoders.model.set_input_embeddings(None)
        self.decoders.model_type = "huggingface"

    def process_embedding(self, sequence):
        # Exact released ARParallelLM/ARDelayLM process_embedding: padding is
        # zero before sum, not nine copies of the pad embedding.
        embeddings = self.emb(sequence)
        return embeddings.masked_fill((sequence == self.pad_id)[..., None], 0).sum(2)

    def delay_interleave(self, dec_seq, dec_seq_lengths=None, pad=0):
        batch, length, streams = dec_seq.shape
        delayed = dec_seq.new_full((batch, length + self.nq - 1, streams), pad)
        for stream in range(streams):
            delayed[:, stream : stream + length, stream] = dec_seq[..., stream]
        if dec_seq_lengths is not None:
            dec_seq_lengths = dec_seq_lengths + self.nq - 1
        return delayed, dec_seq_lengths


class OriginalOpusModel(nn.Module):
    def __init__(self, geometry, token_list, num_streams, share_embeddings=False):
        super().__init__()
        self.corelm = OriginalOpusARDelay(
            geometry, token_list, num_streams, share_embeddings
        )
