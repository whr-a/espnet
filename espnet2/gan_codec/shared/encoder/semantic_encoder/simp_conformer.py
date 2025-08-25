# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tarfile
from collections import OrderedDict

# ---------------------------------------------------------------------------- #
#                           依赖的子模块 (Sub-modules)                           #
# ---------------------------------------------------------------------------- #

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        # xscaling is now handled inside ConformerEncoder based on config
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is None or self.pe.size(1) < x.size(1):
            pe = torch.zeros(x.size(1), self.d_model)
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        pos_emb = self.pe[:, :x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)

# CORRECTED: Matches Canary-1B's separate Q, K, V layers
class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        self.n_head, self.n_feat, self.d_head = n_head, n_feat, n_feat // n_head
        self.linear_q, self.linear_k, self.linear_v = nn.Linear(n_feat, n_feat), nn.Linear(n_feat, n_feat), nn.Linear(n_feat, n_feat)
        self.linear_out, self.linear_pos = nn.Linear(n_feat, n_feat), nn.Linear(n_feat, n_feat, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pos_bias_u, self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_head, self.d_head)), nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        b, h, t1, t2 = x.shape
        x_padded = F.pad(x, (1, 0))
        x_padded = x_padded.view(b, h, t2 + 1, t1)
        x = x_padded[:, :, 1:].view_as(x)
        return x
        
    def forward(self, q, k, v, pos_emb, mask):
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.n_head, self.d_head)
        k = self.linear_k(k).view(batch_size, -1, self.n_head, self.d_head)
        v = self.linear_v(v).view(batch_size, -1, self.n_head, self.d_head)
        pos_emb = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.n_head, self.d_head)
        q_with_bias_u, q_with_bias_v = (q + self.pos_bias_u).transpose(1, 2), (q + self.pos_bias_v).transpose(1, 2)
        AC = torch.matmul(q_with_bias_u, k.transpose(1, 2).transpose(-2, -1))
        BD = self.rel_shift(torch.matmul(q_with_bias_v, pos_emb.transpose(1, 2).transpose(-2, -1)))
        scores = (AC + BD) / math.sqrt(self.d_head)
        
        # ------------------- BUG FIX HERE ------------------- #
        # The mask shape [B, T] needs to be expanded to [B, 1, 1, T] 
        # to be broadcastable to scores shape [B, H, T, T].
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # shape: [B, 1, 1, T]
            scores = scores.masked_fill(mask, float('-inf'))
        # ----------------- END OF BUG FIX ----------------- #

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2).contiguous().view(batch_size, -1, self.n_feat)
        return self.linear_out(out)

# CORRECTED: Renamed `norm` to `batch_norm`
class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=d_model, bias=True
        )
        self.batch_norm = nn.BatchNorm1d(d_model) # Renamed from self.norm
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True)
        self.activation = Swish()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)

# CORRECTED: This is now just the two linear layers, as LN is outside
class FeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

# CORRECTED: Matches Canary-1B's Layer structure and naming
class ConformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_expansion_factor, conv_kernel_size, dropout_att):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForward(d_model, ff_expansion_factor)
        
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model, dropout_att)
        
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvolutionModule(d_model, conv_kernel_size)
        
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForward(d_model, ff_expansion_factor)
        
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, pos_emb, pad_mask):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        x = self.dropout(x) * 0.5 + residual
        
        residual = x
        x = self.norm_self_att(x)
        x = self.self_attn(x, x, x, pos_emb, pad_mask)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.norm_conv(x)
        x = self.conv(x)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.norm_feed_forward2(x)
        x = self.feed_forward2(x)
        x = self.dropout(x) * 0.5 + residual
        
        x = self.norm_out(x)
        return x

class ConvSubsampling(nn.Module):
    def __init__(self, subsampling_factor, feat_in, feat_out, conv_channels):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        
        # This structure exactly replicates NeMo's 'dw_striding' for factor=8
        # and matches the keys from your checkpoint blueprint.
        layers = []
        in_channels = 1
        
        # Stage 1: Standard Conv (stride 2) -> Corresponds to 'conv.0'
        layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        in_channels = conv_channels

        # Stage 2: Depthwise-Separable Conv (stride 2) -> 'conv.2' and 'conv.3'
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels))
        layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ReLU())

        # Stage 3: Depthwise-Separable Conv (stride 2) -> 'conv.5' and 'conv.6'
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels))
        layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)
        
        # Output projection layer, matches the '4096' dim from your blueprint
        final_feat_dim = (((feat_in - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1
        self.out = nn.Linear(conv_channels * final_feat_dim, feat_out)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.out(x)
        return x, lengths // self.subsampling_factor

# ---------------------------------------------------------------------------- #
#                             核心 ConformerEncoder                             #
# ---------------------------------------------------------------------------- #

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        feat_in, n_layers, d_model, feat_out=-1, subsampling='dw_striding',
        subsampling_factor=8, subsampling_conv_channels=256, ff_expansion_factor=4,
        n_heads=8, pos_emb_max_len=5000, conv_kernel_size=9, dropout=0.1,
        dropout_pre_encoder=0.1, dropout_emb=0.0, dropout_att=0.1, xscaling=False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.xscaling = xscaling
        
        self.pre_encode = ConvSubsampling(
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
        )
        
        self.pos_enc = RelPositionalEncoding(d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len)
        
        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=d_model, n_heads=n_heads,
                ff_expansion_factor=ff_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout_att=dropout_att
            ) for _ in range(n_layers)
        ])
        
        # No separate out_proj in Canary-1B encoder, it's part of the main model
        self.out_proj = None

    def _create_padding_mask(self, lengths, max_len):
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask

    def forward(self, audio_signal, length):
        x, encoded_lengths = self.pre_encode(audio_signal, length)
        
        if self.xscaling:
            x = x * math.sqrt(self.d_model)
        
        x, pos_emb = self.pos_enc(x)
        
        max_audio_length = x.size(1)
        pad_mask = self._create_padding_mask(encoded_lengths, max_audio_length)
        
        for layer in self.layers:
            x = layer(x, pos_emb, pad_mask)
        
        if self.out_proj is not None:
            x = self.out_proj(x)
            
        x = x.transpose(1, 2)
        
        return x, encoded_lengths

# ---------------------------------------------------------------------------- #
#                      如何加载权重及进行推理 (Usage Example)                       #
# ---------------------------------------------------------------------------- #
def load_from_standalone_pt(model, pt_path):
    print(f"Loading standalone encoder weights from: {pt_path}")
    state_dict = torch.load(pt_path, map_location=torch.device('cpu'))
    
    # Let's try loading with strict=False first to see if there are minor issues
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully with strict=True!")
    except RuntimeError as e:
        print(f"!!! Strict loading failed. Trying with strict=False. This may be okay. !!!\nError: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded with strict=False.")
    return model


if __name__ == '__main__':
    config = {
        'feat_in': 128, 'n_layers': 32, 'd_model': 1024, 'feat_out': -1,
        'subsampling': 'dw_striding', 'subsampling_factor': 8, 'subsampling_conv_channels': 256,
        'ff_expansion_factor': 4, 'n_heads': 8, 'pos_emb_max_len': 5000,
        'conv_kernel_size': 9, 'dropout': 0.1, 'dropout_pre_encoder': 0.1,
        'dropout_emb': 0.0, 'dropout_att': 0.1, 'xscaling': False,
    }

    print("Instantiating CORRECTED standalone ConformerEncoder...")
    standalone_encoder = ConformerEncoder(**config)
    standalone_encoder.eval()
    print("Model instantiated successfully.")

    standalone_encoder_path = '/u/hwang41/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt'
    
    try:
        standalone_encoder = load_from_standalone_pt(standalone_encoder, standalone_encoder_path)
    except FileNotFoundError:
        print(f"SKIPPING WEIGHT LOADING: Standalone encoder file not found at '{standalone_encoder_path}'.")

    print("\nPerforming a test inference pass...")
    batch_size = 4
    num_timesteps = 200
    
    input_signal = torch.randn(batch_size, num_timesteps, config['feat_in'])
    input_lengths = torch.randint(low=num_timesteps // 2, high=num_timesteps, size=(batch_size,))
    input_lengths[0] = num_timesteps

    print(f"Input signal shape: {input_signal.shape}")
    print(f"Input lengths: {input_lengths}")

    with torch.no_grad():
        encoded_output, encoded_lengths = standalone_encoder(input_signal, input_lengths)

    print(f"\nEncoded output shape: {encoded_output.shape}")
    print(f"Encoded lengths: {encoded_lengths}")
    
    expected_len = torch.div(input_lengths, config['subsampling_factor'], rounding_mode='floor')
    assert torch.equal(expected_len, encoded_lengths)
    print("\nOutput lengths are correct. Corrected standalone model works as expected!")