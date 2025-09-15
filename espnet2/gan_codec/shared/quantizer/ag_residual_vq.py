import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from espnet2.gan_codec.shared.quantizer.modules.core_vq import (
    AutoGroupVectorQuantize,
)

class AutoGroupResidualVectorQuantize(nn.Module):

    def __init__(
        self,
        input_dim: int = 160,
        n_codebooks: int = 6,
        codebook_size: int = 32,
        codebook_dim: Union[int, list] = 8,
        target_n_q: list = [1, 6],
        frame_residual_vq: bool = False,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList([
            AutoGroupVectorQuantize(input_dim, codebook_size, codebook_dim[i], frame_residual_vq=frame_residual_vq)
            for i in range(n_codebooks)
        ])
        self.target_n_q = target_n_q

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            # n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            # dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            # n_dropout = int(z.shape[0] * self.quantizer_dropout)
            # n_quantizers[:n_dropout] = dropout[:n_dropout]
            # n_quantizers = n_quantizers.to(z.device)
            n_quantizers = random.choice(self.target_n_q)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """

        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            codes_a = codes[:, i, :] // self.quantizers[i].codebook_size_b
            codes_b = codes[:, i, :] - codes_a * self.quantizers[i].codebook_size_b

            z_pa_i = self.quantizers[i].decode_code(
                codes_a, self.quantizers[i].codebook_a)  # z_q = self.decode_code(indices, codebook_in)
            z_pb_i = self.quantizers[i].decode_code(codes_b, self.quantizers[i].codebook_b)

            z_p.append(torch.cat((z_pa_i, z_pb_i), dim=1))

            # z_q_i = self.quantizers[i].out_proj(z_p_i)

            z_aq = self.quantizers[i].out_proj_a(z_pa_i)
            z_bq = self.quantizers[i].out_proj_b(z_pb_i)
            z_q = z_q + torch.cat((z_aq, z_bq), dim=1)

        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

