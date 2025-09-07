# Copyright 2024 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from https://github.com/facebookresearch/encodec
# Original license as follows:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core vector quantization implementation."""
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.distributed as distributed
from einops import rearrange, repeat, reduce
from torch import nn, einsum

from espnet2.gan_codec.shared.quantizer.modules.distrib import broadcast_tensors


def default(val: Any, d: Any) -> Any:
    return val if val is not None else d


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

    
def noop(*args, **kwargs):
    pass


def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)


def cdist(x, y, eps = 1e-8):
    x2 = reduce(x ** 2, 'n d -> n', 'sum')
    y2 = reduce(y ** 2, 'c d -> c', 'sum')
    xy = einsum('n d, c d -> n c', x, y) * -2
    return (rearrange(x2, 'n -> n 1') + rearrange(y2, 'c -> 1 c') + xy).clamp(min = eps).sqrt()


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    num_probs = len(probs)

    for i, prob in enumerate(probs):
        is_last = i == (num_probs - 1)

        s = torch.binomial(total_count, prob / remainder) if not is_last else total_count
        sample[i] = s
        total_count -= s
        remainder -= prob

    assert total_count == 0, f'invalid total count {total_count}'

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return out


def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = sample_vectors,
    all_reduce_fn = noop
):
    dim, dtype= samples.shape[-1], samples.dtype
    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch
            and use the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at
            initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster size
            less than the specified threshold with randomly selected vector from
            the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: Union[Callable[..., torch.Tensor], Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        use_ddp = distributed.is_available() and \
            distributed.is_initialized() and distributed.get_world_size() > 1
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
        self.sample_fn = sample_vectors_distributed if use_ddp else sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')

        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.update_ema()
        self.inited.data.copy_(torch.Tensor([True]))

    def replace_(self, samples, mask):
        sampled = self.sample_fn(samples, mask.sum().item())
        self.embed.data[mask] = sampled
        self.cluster_size.data[mask] = self.threshold_ema_dead_code
        self.embed_avg.data[mask] = sampled * self.threshold_ema_dead_code

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        
    def update_ema(self):
        cluster_size = laplace_smoothing(
            self.cluster_size, self.codebook_size, self.epsilon) *\
                  self.cluster_size.sum(dim = -1, keepdim = True)

        embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed_normalized)

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        """Codebook Forward with EMA.

        Args:
            x (Tensor): Vector for quantization (B, T, D)

        Return:
            Tensor: Quantized output (B, T, D)
            Tensor: Codebook Index (B, T)
        """
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)  # (BxT, D)

        # Initialize the embedding (only activated for the first time)
        self.init_embed_(x)

        # Quantization Process
        embed_ind = self.quantize(x)  # (BxT)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # (BxT, V)
        embed_ind = self.postprocess_emb(embed_ind, shape)  # (B, T)
        quantize = self.dequantize(embed_ind)  # (B, T, D)

        if self.training:
            # ema update number of frames per cluster
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size, cluster_size, self.decay)

            # Use encoder embedding to update ema with assignments
            embed_sum = x.t() @ embed_onehot  # (D, BxT) @ (BxT, V) -> (D, V)
            self.all_reduce_fn(embed_sum)

            # ema update embedding
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            self.update_ema()

            self.expire_codes_(x)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.

    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified
            dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
            Replace any codes that have an exponential moving average cluster size
            less than the specified threshold with randomly selected vector from
            the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, mask=None):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        if not self.quantizer_dropout:
            loss = torch.tensor([0.0], device=device, requires_grad=self.training)

            if self.training:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss

            quantize = self.project_out(quantize)
            quantize = rearrange(quantize, "b n d -> b d n")
            return quantize, embed_ind, loss
        else:
            commit_loss = torch.tensor(
                [0.0], device=device, requires_grad=self.training
            )
            quant_loss = torch.tensor([0.0], device=device, requires_grad=self.training)
            if self.training:
                if self.quantizer_dropout:
                    _commit_loss = F.mse_loss(
                        quantize.detach(), x, reduction="none"
                    ).mean([1, 2])
                    commit_loss = commit_loss + (_commit_loss * mask).mean()
                    _quant_loss = F.mse_loss(
                        quantize, x.detach(), reduction="none"
                    ).mean([1, 2])
                    quant_loss = quant_loss + (_quant_loss * mask).mean()

                else:
                    _commit_loss = F.mse_loss(quantize.detach(), x)
                    commit_loss = commit_loss + _commit_loss
                    _quant_loss = F.mse_loss(quantize, x.detach(), reduction="none")
                    quant_loss = quant_loss + _quant_loss

            quantize = self.project_out(quantize)
            quantize = rearrange(quantize, "b n d -> b d n")
            return quantize, embed_ind, commit_loss, quant_loss

class AutoGroupVectorQuantize(nn.Module):
    """
    Inspirations:
    The core ideas of our approach are derived from the following papers and concepts:
    1.Grouped Quantization: The concept of grouping from HIFI-CODEC: GROUP-RESIDUAL VECTOR QUANTIZATION FOR HIGH FIDELITY AUDIO CODEC.
    2.Cosine Similarity Search: The technique of performing a codebook search after dimensionality reduction and using L2 normalization to shift the distance metric from Euclidean distance to cosine similarity, as detailed in High-Fidelity Audio Compression with Improved RVQGAN.
    3.Temporal Residual Coding: An idea from traditional codecs, where temporal residual coding is used to reduce the dynamic range of codebook representations for non-initial speech frames.
    Key Steps:
    The core pipeline consists of the following steps:
    1.(Optional) Apply inter-frame residual coding to the input latents along the time dimension.
    2.Simultaneously perform adaptive grouping and dimensionality reduction on the input latents.
    3.Apply intra-frame residual coding to the resulting parallel data after reduction.
    4.Perform the codebook search in parallel across all groups.
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, frame_residual_vq=False):
        super().__init__()

        self.codebook_size_a = codebook_size
        self.codebook_size_b = codebook_size
        self.codebook_dim = codebook_dim
        self.frame_residual_vq = frame_residual_vq
        self.codebook_dim_a = codebook_dim
        self.codebook_dim_b = codebook_dim

        self.in_proj_a = WNConv1d(input_dim, self.codebook_dim_a, kernel_size=1)
        self.out_proj_a = WNConv1d(self.codebook_dim_a, input_dim // 2, kernel_size=1)

        self.in_proj_b = WNConv1d(input_dim, self.codebook_dim_b, kernel_size=1)
        self.out_proj_b = WNConv1d(self.codebook_dim_b, input_dim // 2, kernel_size=1)

        self.codebook_a = nn.Embedding(self.codebook_size_a, self.codebook_dim_a)
        self.codebook_b = nn.Embedding(self.codebook_size_b, self.codebook_dim_b)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space

        if self.frame_residual_vq:
            for frame in range(z.shape[-1] - 1, 0, -1):
                z[..., frame] = z[..., frame] - z[..., frame - 1]

        z_a = self.in_proj_a(z)  # z_a : (B x D x T)
        z_b = self.in_proj_b(z)  # z_b : (B x D x T)

        # z_a = z_a - z_b
        z_aq, indices_a = self.decode_latents(z_a, self.codebook_a)
        z_bq, indices_b = self.decode_latents(z_b, self.codebook_b)

        commitment_loss = F.mse_loss(z_a, z_aq.detach(), reduction="none").mean([1, 2]) + F.mse_loss(
            z_b, z_bq.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_aq, z_a.detach(), reduction="none").mean([1, 2]) + F.mse_loss(
            z_bq, z_b.detach(), reduction="none").mean([1, 2])

        # c
        z_aq = (z_a + (z_aq - z_a).detach()
               )  # noop in forward pass, straight-through gradient estimator in backward pass
        z_bq = (z_b + (z_bq - z_b).detach()
               )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_aq = self.out_proj_a(z_aq)
        z_bq = self.out_proj_b(z_bq)
        z_q = torch.cat((z_aq, z_bq), dim=1)

        if self.frame_residual_vq:
            for frame in range(1, 1, z_q.shape[-1]):
                z_q[..., frame] = z_q[..., frame - 1] + z_q[..., frame]

        indices = indices_a * self.codebook_size_b + indices_b  
        latent = torch.cat((z_a, z_b), dim=1)

        return z_q, commitment_loss, codebook_loss, indices, latent

    def embed_code(self, embed_id, codebook):
        return F.embedding(embed_id, codebook.weight)

    def decode_code(self, embed_id, codebook):
        return self.embed_code(embed_id, codebook).transpose(1, 2)

    def decode_latents(self, latents, codebook_in):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = codebook_in.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() +
                codebook.pow(2).sum(1, keepdim=True).t())
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices, codebook_in)
        # z_q shape [B,dim,T]
        return z_q, indices

class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        self.quantizer_dropout = kwargs.get("quantizer_dropout")

    def forward(self, x, n_q: Optional[int] = None):
        quantized_out = 0.0
        residual = x

        if not self.quantizer_dropout:
            all_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)

            for layer in self.layers[:n_q]:
                quantized, indices, loss = layer(residual)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)
                all_losses.append(loss)

            out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
            return quantized_out, out_indices, out_losses
        else:
            all_commit_losses = []
            all_quant_losses = []
            all_indices = []

            n_q = n_q or len(self.layers)
            if self.training:
                n_q = torch.ones((x.shape[0],)) * len(self.layers) + 1
                dropout = torch.randint(1, len(self.layers) + 1, (x.shape[0],))
                n_dropout = int(x.shape[0] * self.quantizer_dropout)
                n_q[:n_dropout] = dropout[:n_dropout]
                n_q = n_q.to(x.device)

            for i, layer in enumerate(self.layers):
                if self.training is False and i >= n_q:
                    break
                mask = torch.full((x.shape[0],), fill_value=i, device=x.device) < n_q
                quantized, indices, commit_loss, quant_loss = layer(residual, mask)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized * mask[:, None, None]

                all_indices.append(indices)
                all_commit_losses.append(commit_loss)
                all_quant_losses.append(quant_loss)

            out_commit_losses, out_quant_losses, out_indices = map(
                torch.stack, (all_commit_losses, all_quant_losses, all_indices)
            )
            return quantized_out, out_indices, out_commit_losses, out_quant_losses

    def encode(
        self, x: torch.Tensor, n_q: Optional[int] = None, st: Optional[int] = None
    ) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:  # 设置解码的起止layer
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
