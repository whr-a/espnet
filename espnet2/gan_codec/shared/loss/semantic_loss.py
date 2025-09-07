"""
Implementation of this loss function is taken from 
stable-audio-tools library with minor modifications:

https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/training/losses/semantic.py


MIT License

Copyright (c) 2023 Stability AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from typing import List, Optional
import torchaudio

from einops import rearrange
from torch import nn


def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

class HubertLoss(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000, 
                 feature_ids: Optional[List[int]] = None,
                 model_name: str = "HUBERT_LARGE"
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.feature_ids = feature_ids
        self.model_name = model_name
        
        # Load model based on the specified model name
        if self.model_name == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
        elif self.model_name == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
        elif self.model_name == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.bundle_model = bundle.get_model()
        self.model_sample_rate = bundle.sample_rate

        for param in self.bundle_model.parameters():
            param.requires_grad = False
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # We do not want to use unnecessary disk space 
        # when saving checkpoints due to the self supervised
        # model used here
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        keys_to_remove = [k for k in state if k.startswith(prefix + 'bundle_model')]
        for k in keys_to_remove:
            del state[k]
        return state
        
    def forward(self, x, y):
        if self.sample_rate != self.model_sample_rate:
            x = torchaudio.functional.resample(x, self.sample_rate, self.model_sample_rate)
            y = torchaudio.functional.resample(y, self.sample_rate, self.model_sample_rate)

        x = fold_channels_into_batch(x)
        y = fold_channels_into_batch(y)

        conv_features = (
            self.feature_ids is not None and
            len(self.feature_ids) == 1 and
            self.feature_ids[0] == -1)

        # Extract features from conv layers only.
        if conv_features:
            if self.bundle_model.normalize_waveform:
                x = nn.functional.layer_norm(x, x.shape)
                y = nn.functional.layer_norm(y, y.shape)
            x_list, _ = self.bundle_model.model.feature_extractor(x, None)
            y_list, _ = self.bundle_model.model.feature_extractor(y, None)
            x_list = [x_list]
            y_list = [y_list]
        else:
            x_list, _ = self.bundle_model.extract_features(x)
            y_list, _ = self.bundle_model.extract_features(y)

        loss = 0
        denom = 0
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            if self.feature_ids is None or i in self.feature_ids or conv_features:
                loss += nn.functional.l1_loss(x, y) / (y.std() + 1e-5)
                denom += 1

        loss = loss / denom
        return loss
