from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from espnet2.gan_codec.shared.quantizer.modules.core_vq import (
    AutoGroupVectorQuantize,
)


class MultiRateAutoGroupResidualVectorQuantize(nn.Module):
    """
    A residual vector quantizer that supports two distinct bitrate modes.
    It consists of a set of 'base' quantizers for a low-bitrate representation
    and a set of 'advancement' quantizers that quantize the residual from the
    base layer for a higher bitrate representation.

    Args:
        input_dim (int): Dimension of the input tensor.
        
        # Base (Low Bitrate) Config
        n_base_codebooks (int): Number of codebooks for the low-bitrate layer.
        base_codebook_size (int): Size of each codebook in the low-bitrate layer.
        base_codebook_dim (Union[int, list]): Dimension of codebook vectors in the low-bitrate layer.

        # advancement (High Bitrate) Config
        n_advance_codebooks (int): Number of codebooks for the high-bitrate layer.
        advance_codebook_size (int): Size of each codebook in the high-bitrate layer.
        advance_codebook_dim (Union[int, list]): Dimension of codebook vectors in the high-bitrate layer.
        
        quantizer_dropout (float): Probability of dropping the entire advancement layer during training.
        frame_residual_vq (bool): Whether to use frame-level residual VQ.
    """
    def __init__(
        self,
        input_dim: int = 512,
        # Base (Low Bitrate) Config
        n_base_codebooks: int = 3,
        base_codebook_size: int = 2048, # 2^11
        base_codebook_dim: Union[int, list] = 8,
        # advancement (High Bitrate) Config
        n_advance_codebooks: int = 14,
        advance_codebook_size: int = 16384, # 2^14
        advance_codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.1, # Dropout probability for the advancement layer
        frame_residual_vq: bool = False,
    ):
        super().__init__()

        # --- Base Quantizers Initialization ---
        self.n_base_codebooks = n_base_codebooks
        if isinstance(base_codebook_dim, int):
            base_codebook_dim = [base_codebook_dim for _ in range(n_base_codebooks)]
        
        self.base_quantizers = nn.ModuleList([
            AutoGroupVectorQuantize(input_dim, base_codebook_size, base_codebook_dim[i], frame_residual_vq=frame_residual_vq)
            for i in range(n_base_codebooks)
        ])

        # --- advancement Quantizers Initialization ---
        self.n_advance_codebooks = n_advance_codebooks
        if isinstance(advance_codebook_dim, int):
            advance_codebook_dim = [advance_codebook_dim for _ in range(n_advance_codebooks)]

        self.advance_quantizers = nn.ModuleList([
            AutoGroupVectorQuantize(input_dim, advance_codebook_size, advance_codebook_dim[i], frame_residual_vq=frame_residual_vq)
            for i in range(n_advance_codebooks)
        ])

        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, mode: str = 'high'):
        """
        Quantizes the input tensor based on the specified mode.

        Args:
            z (Tensor[B x D x T]): Input tensor.
            mode (str): Inference mode, 'low' or 'high'. Ignored during training.

        Returns:
            dict: A dictionary containing quantized tensor, codes, latents, and losses.
        """
        if not self.training and mode not in ['low', 'high']:
            raise ValueError("Mode must be 'low' or 'high' during inference.")

        # =================================================================
        # 1. Process with Base Quantizers (always active)
        # =================================================================
        z_q_base = 0
        residual = z
        commitment_loss_base = 0
        codebook_loss_base = 0
        
        base_indices = []
        base_latents = []

        for quantizer in self.base_quantizers:
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            z_q_base = z_q_base + z_q_i
            residual = residual - z_q_i
            commitment_loss_base += commitment_loss_i
            codebook_loss_base += codebook_loss_i
            base_indices.append(indices_i)
            base_latents.append(z_e_i)

        # In case of low-bitrate inference, we are done
        if not self.training and mode == 'low':
            codes = torch.stack(base_indices, dim=1)
            latents = torch.cat(base_latents, dim=1)
            return z_q_base, codes, latents, commitment_loss_base, codebook_loss_base

        # =================================================================
        # 2. Process with advancement Quantizers (conditional)
        # =================================================================
        z_q_advance = 0
        commitment_loss_advance = 0
        codebook_loss_advance = 0

        advance_indices = []
        advance_latents = []

        # During training, apply dropout to the entire advancement layer
        if self.training:
            # Create a mask for the batch. 1 = use advancement, 0 = drop advancement
            mask = torch.bernoulli(torch.full((z.shape[0],), 1 - self.quantizer_dropout, device=z.device))
            # Reshape for broadcasting: [B, 1, 1]
            mask_b11 = mask[:, None, None]
        else: # During 'high' mode inference, always use the advancement layer
            mask = torch.ones(z.shape[0], device=z.device)
            mask_b11 = 1.0

        for quantizer in self.advance_quantizers:
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            
            # Apply mask to the output and residual
            z_q_advance = z_q_advance + z_q_i * mask_b11
            residual = residual - z_q_i * mask_b11

            # Apply mask to losses
            commitment_loss_advance += (commitment_loss_i * mask).mean()
            codebook_loss_advance += (codebook_loss_i * mask).mean()

            advance_indices.append(indices_i)
            advance_latents.append(z_e_i)
            
        # =================================================================
        # 3. Combine results
        # =================================================================
        z_q = z_q_base + z_q_advance
        
        # Combine codes and latents
        codes = torch.stack(base_indices + advance_indices, dim=1)
        latents = torch.cat(base_latents + advance_latents, dim=1)
        
        commitment_loss = commitment_loss_base + commitment_loss_advance
        codebook_loss = codebook_loss_base + codebook_loss_advance

        return z_q, codes, latents, commitment_loss, codebook_loss

    def encode(self, x: torch.Tensor, mode: str = 'high') -> torch.Tensor:
        """
        根据指定模式将输入张量编码为码本索引。

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, D, T]。
            mode (str): 编码模式, 'low' 或 'high'。

        Returns:
            torch.Tensor: 离散码本索引, 形状为 [B, N_quantizers, T]。
        """
        if mode not in ['low', 'high']:
            raise ValueError("Mode must be 'low' or 'high'.")

        residual = x
        all_codes = []

        for quantizer in self.base_quantizers:
            _, _, _, indices, z_e_i = quantizer(residual)
            z_q_i = quantizer.out_proj(z_e_i)
            residual = residual - z_q_i
            all_codes.append(indices)
        
        if mode == 'high':
            for quantizer in self.enhance_quantizers:
                _, _, _, indices, z_e_i = quantizer(residual)
                z_q_i = quantizer.out_proj(z_e_i)
                residual = residual - z_q_i
                all_codes.append(indices)
        
        return torch.stack(all_codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        从码本索引解码，自动根据索引数量判断模式。

        Args:
            codes (torch.Tensor): 离散码本索引, 形状为 [B, N_quantizers, T]。

        Returns:
            torch.Tensor: 重建的连续表示, 形状为 [B, D, T]。
        """
        n_quantizers = codes.shape[1]

        is_low_mode = (n_quantizers == self.n_base_codebooks)
        is_high_mode = (n_quantizers == self.n_base_codebooks + self.n_enhance_codebooks)

        if not (is_low_mode or is_high_mode):
            raise ValueError(
                f"Invalid number of codebooks: {n_quantizers}. "
                f"Expected {self.n_base_codebooks} (low mode) or "
                f"{self.n_base_codebooks + self.n_enhance_codebooks} (high mode)."
            )

        z_q = 0.0

        for i in range(self.n_base_codebooks):
            quantizer = self.base_quantizers[i]

            codes_a = codes[:, i, :] // quantizer.codebook_size_b
            codes_b = codes[:, i, :] - codes_a * quantizer.codebook_size_b
            
            z_pa_i = quantizer.decode_code(codes_a, quantizer.codebook_a)
            z_pb_i = quantizer.decode_code(codes_b, quantizer.codebook_b)
            
            z_aq = quantizer.out_proj_a(z_pa_i)
            z_bq = quantizer.out_proj_b(z_pb_i)
            z_q = z_q + torch.cat((z_aq, z_bq), dim=1)

        if is_high_mode:
            for i in range(self.n_base_codebooks, n_quantizers):
                enh_idx = i - self.n_base_codebooks
                quantizer = self.enhance_quantizers[enh_idx]
                
                codes_a = codes[:, i, :] // quantizer.codebook_size_b
                codes_b = codes[:, i, :] - codes_a * quantizer.codebook_size_b

                z_pa_i = quantizer.decode_code(codes_a, quantizer.codebook_a)
                z_pb_i = quantizer.decode_code(codes_b, quantizer.codebook_b)

                z_aq = quantizer.out_proj_a(z_pa_i)
                z_bq = quantizer.out_proj_b(z_pb_i)
                z_q = z_q + torch.cat((z_aq, z_bq), dim=1)

        return z_q