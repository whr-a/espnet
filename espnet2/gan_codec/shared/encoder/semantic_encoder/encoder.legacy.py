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

"""
这个文件定义了主要的 SemanticEncoder 类。

它将以下两个模块的功能整合为一个端到端的流程：
1. 音频预处理器 (from extract.py): 将 24kHz 音频波形转换为 log-Mel 谱图。
2. Conformer 编码器 (from simp_conformer.py): 将谱图编码为深层语义表征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# 假设 simp_conformer.py 和 extract.py 与此文件在同一个目录下
# 这种相对导入是标准的 Python 包内导入方式
from simp_conformer import ConformerEncoder
from extract import get_canary_flash_preprocessor_output


class SemanticEncoder(nn.Module):
    """
    一个端到端的语义编码器，封装了从 24kHz 音频到 Conformer 编码向量的全过程。
    """

    def __init__(self, conformer_config: Optional[Dict] = None):
        """
        初始化 SemanticEncoder。

        Args:
            conformer_config (Dict, optional): 用于初始化 ConformerEncoder 的配置字典。
                                               如果为 None，将使用默认的 Canary-1B 配置。
        """
        super().__init__()

        if conformer_config is None:
            # 使用与 Canary-1B 匹配的默认配置
            conformer_config = {
                'feat_in': 128, 'n_layers': 32, 'd_model': 1024,
                'subsampling_factor': 8, 'subsampling_conv_channels': 256,
                'ff_expansion_factor': 4, 'n_heads': 8, 'conv_kernel_size': 9,
                'dropout_pre_encoder': 0.1, 'dropout_att': 0.1, 'xscaling': False,
            }
        
        # 成员1：Conformer 编码器
        self.encoder = ConformerEncoder(**conformer_config)
        # 注意：预处理器是一个无状态的函数，不需要作为 nn.Module 成员
        # 最终的归一化层 (L2 Norm) 是一个无参数的操作，将在 forward 中通过函数调用实现

    def load_weights(self, encoder_ckpt_path: str, strict: bool = True):
        """
        一个辅助函数，用于加载预训练的独立 encoder 权重。

        Args:
            encoder_ckpt_path (str): 指向 `encoder_standalone.pt` 文件的路径。
            strict (bool): 是否使用严格模式加载权重。默认为 True。
        """
        print(f"Loading standalone encoder weights from: {encoder_ckpt_path}")
        try:
            state_dict = torch.load(encoder_ckpt_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=strict)
            print(f"✅ Encoder weights loaded successfully (strict={strict})!")
        except FileNotFoundError:
            print(f"⚠️ WARNING: Checkpoint file not found at '{encoder_ckpt_path}'. Model is using random weights.")
        except Exception as e:
            print(f"❌ An error occurred during weight loading: {e}")
            import traceback
            traceback.print_exc()

    def forward(
        self, 
        input_signal_24k: torch.Tensor, 
        length_24k: torch.Tensor,
        normalize_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        执行从 24kHz 音频到语义向量的前向传播。

        Args:
            input_signal_24k (torch.Tensor): 形状为 (B, T_wav) 的 24kHz 原始音频波形。
            length_24k (torch.Tensor): 形状为 (B,) 的每个音频的实际长度。
            normalize_output (bool): 是否对最终的输出向量进行 L2 归一化。
                                     这对于 K-Means 等任务很有用。默认为 False。

        Returns:
            Dict[str, torch.Tensor]: 一个包含结果的字典:
                - 'embeddings' (torch.Tensor): Conformer 输出的语义向量, 形状 (B, D, T_feat)。
                - 'lengths' (torch.Tensor): 语义向量序列的实际长度, 形状 (B,)。
                - 'normalized_embeddings' (torch.Tensor, optional): 如果 normalize_output=True,
                  则包含 L2 归一化后的语义向量。
        """
        # 1. 音频预处理：24kHz wav -> 16kHz log-Mel (B, D_mel, T_mel)
        # 调用从 extract.py 导入的函数
        processed_features, processed_lengths = get_canary_flash_preprocessor_output(
            input_signal_24k, length_24k
        )

        # 2. Conformer 编码：log-Mel -> Semantic Embeddings
        # Conformer 模型需要 (B, T, D) 格式的输入，所以需要转置
        processed_features_transposed = processed_features.transpose(1, 2)
        
        # encoder 的输出也是 (B, D, T)
        encoded_output, encoded_lengths = self.encoder(
            audio_signal=processed_features_transposed, 
            length=processed_lengths
        )

        # 3. 准备输出
        output_dict = {
            "embeddings": encoded_output,
            "lengths": encoded_lengths,
        }

        # 4. (可选) 对输出进行 L2 归一化，用于 K-Means
        if normalize_output:
            # 沿着特征维度 (dim=1) 进行归一化
            normalized_embeddings = F.normalize(encoded_output, p=2, dim=1)
            output_dict["normalized_embeddings"] = normalized_embeddings

        return output_dict


# ============================================================================ #
#                            Part 2: 使用示例                                    #
# ============================================================================ #

if __name__ == '__main__':
    # --- 1. 初始化端到端语义编码器 ---
    print("Initializing the end-to-end SemanticEncoder...")
    semantic_encoder = SemanticEncoder()
    semantic_encoder.eval()
    print("SemanticEncoder initialized successfully.")

    # --- 2. 加载预训练的 Conformer 权重 ---
    # !!! 注意: 请确保你的 simp_conformer.py 和 extract.py 就在旁边 !!!
    # !!! 并且这个路径指向你之前生成的独立 encoder 权重 !!!
    encoder_weights_path = '/u/hwang41/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt'
    semantic_encoder.load_weights(encoder_weights_path)
    
    # --- 3. 准备一批模拟的 24kHz 音频输入 ---
    print("\nPreparing dummy 24kHz audio batch...")
    batch_size = 2
    sample_rate_24k = 24000
    seconds = 5
    max_len = sample_rate_24k * seconds
    
    # 创建不同长度的音频
    audio_lengths_24k = torch.tensor([max_len, int(max_len * 0.8)])
    signals_24k = torch.randn(batch_size, max_len)
    
    # 将填充部分置零
    for i in range(batch_size):
        signals_24k[i, audio_lengths_24k[i]:] = 0.0
        
    print(f"Input waveform shape: {signals_24k.shape}")
    print(f"Input waveform lengths: {audio_lengths_24k}")

    # --- 4. 执行完整的前向传播 ---
    print("\nPerforming forward pass...")
    with torch.no_grad():
        # Case 1: 获取标准输出
        outputs = semantic_encoder(signals_24k, audio_lengths_24k, normalize_output=False)
        
        # Case 2: 获取归一化后的输出
        normalized_outputs = semantic_encoder(signals_24k, audio_lengths_24k, normalize_output=True)

    # --- 5. 打印结果 ---
    print("\n--- Standard Output ---")
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    print(f"Output lengths: {outputs['lengths']}")

    print("\n--- Normalized Output (for K-Means) ---")
    print(f"Normalized embeddings shape: {normalized_outputs['normalized_embeddings'].shape}")
    print(f"Output lengths: {normalized_outputs['lengths']}")
    
    # 验证归一化向量的 L2 范数是否为 1
    l2_norms = torch.linalg.norm(normalized_outputs['normalized_embeddings'][:, :, 0], dim=1)
    print(f"L2 norms of first frame of normalized embeddings: {l2_norms}")
    assert torch.allclose(l2_norms, torch.ones_like(l2_norms)), "L2 Normalization failed!"
    print("✅ L2 Normalization verified successfully.")
    print("\nEnd-to-end SemanticEncoder works as expected!")