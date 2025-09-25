#!/usr/bin/env python3

"""Inference script for ESPnet Universa model."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch.nn as nn

import numpy as np
import torch
import torchaudio.transforms as T
from typeguard import typechecked

from espnet2.universa_ext.infer import load_model
from espnet2.universa_ext.utils import lens2mask, override
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
import torchaudio
from typing import List

class universaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.universa_model, self.config = load_model("vvwangvv/universa-ext_wavlm-base_5metric")
        self.universa_model.to(dtype=getattr(torch, "float32"))
        self.resampler = torchaudio.transforms.Resample(24000, 16000)
        print(self.universa_model)
    @typechecked
    def __call__(self, audio: torch.Tensor, **kwargs):
        self.resampler = self.resampler.to(audio.device)
        audio = audio.squeeze(1)
        audio = self.resampler(audio)

        audio_lengths = torch.tensor([audio.shape[1]] * audio.shape[0], device=audio.device)
        feats, feats_lengths = self.universa_model.feature_extractor(audio, audio_lengths)
        feats = self.universa_model.feat_proj(feats)
        padding_mask = lens2mask(feats_lengths)
        x = self.universa_model.encoder(feats, src_key_padding_mask=(1 - padding_mask).bool()) * padding_mask.unsqueeze(-1)

        pooled = x.sum(dim=1) / feats_lengths.unsqueeze(-1)
        metric_logits = self.universa_model.metric_proj(pooled)
        
        metric2pred: Dict[str, torch.Tensor] = {
            name: self.universa_model.metric2act[name](metric_logits[:, self.universa_model.metric2idx[name]]) for name in self.universa_model.metric2idx
        }
        sum_of_values = sum(metric2pred.values())
        return -sum_of_values.mean()
