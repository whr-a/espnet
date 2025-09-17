"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import tensor, Tensor, int32
from torch.amp import autocast

import einx
from einops import rearrange, pack, unpack

import random

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z):
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
    ):
        super().__init__()

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype = int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype = int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = return_indices

        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    
    def symmetry_preserving_bound(self, z):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """
        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (z.tanh() + 1) / 2.) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z)

        # determine where to add a random offset elementwise
        # if using noise dropout

        if not self.training or noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(int32)

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices

class LayeredFSQ(Module):
    def __init__(
        self,
        dim: int,
        levels: list[int] | tuple[int, ...],
        num_codebooks_base = 1,
        num_codebooks_enhancement = 5,
        skip_fsq: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.codebook_dim = len(levels)

        self.skip_fsq = skip_fsq

        self.project_in = nn.Linear(dim, self.codebook_dim * (num_codebooks_base + num_codebooks_enhancement))
        self.project_out = nn.Linear(self.codebook_dim * (num_codebooks_base + num_codebooks_enhancement), dim)

        self.fsq_base = FSQ(
            levels = levels,
            dim = self.codebook_dim * num_codebooks_base,
            num_codebooks = num_codebooks_base
        )

        self.fsq_enhancement = FSQ(
            levels = levels,
            dim = self.codebook_dim * num_codebooks_enhancement,
            num_codebooks = num_codebooks_enhancement
        )

    def forward(self, z, mode = 6):
        z_projected = self.project_in(z)

        if self.skip_fsq:
            # --- Corrected skip_fsq logic ---
            # High-rate output uses the full projected vector
            output_high_rate = self.project_out(z_projected)

            # Low-rate output uses only the base part of the vector, padding the rest with zeros
            z_base, _ = z_projected.split([self.fsq_base.dim, self.fsq_enhancement.dim], dim=-1)
            zeros_for_enhancement = torch.zeros_like(z_projected[..., self.fsq_base.dim:])
            z_low_rate_projected = torch.cat([z_base, zeros_for_enhancement], dim=-1)
            output_low_rate = self.project_out(z_low_rate_projected)

            return output_low_rate, output_high_rate, None

        z_base, z_enhancement = z_projected.split([self.fsq_base.dim, self.fsq_enhancement.dim], dim = -1)

        q_base, indices_base = self.fsq_base(z_base)
        q_enhancement, indices_enhancement = self.fsq_enhancement(z_enhancement)

        zeros_for_enhancement = torch.zeros_like(q_enhancement)
        q_low_rate_24d = torch.cat([q_base, zeros_for_enhancement], dim = -1)
        output_low_rate = self.project_out(q_low_rate_24d)

        q_high_rate_24d = torch.cat([q_base, q_enhancement], dim = -1)
        output_high_rate = self.project_out(q_high_rate_24d)

        if mode == 1:
            indices = indices_base
        elif mode == 6:
            indices = torch.cat([indices_base.unsqueeze(-1), indices_enhancement], dim = -1)
        else:
            indices = None

        return output_low_rate, output_high_rate, indices
    def decode(self, indices):
        if indices.ndim == 2:
            q_base = self.fsq_base.indices_to_codes(indices)
            
            b, n, _ = q_base.shape
            zeros_enhancement = torch.zeros(b, n, self.fsq_enhancement.dim, device=indices.device, dtype=q_base.dtype)
            
            q_vector = torch.cat([q_base, zeros_enhancement], dim = -1)

        elif indices.ndim == 3:
            indices_base = indices[..., 0]
            indices_enhancement = indices[..., 1:]

            q_base = self.fsq_base.indices_to_codes(indices_base)
            q_enhancement = self.fsq_enhancement.indices_to_codes(indices_enhancement)
            
            q_vector = torch.cat([q_base, q_enhancement], dim = -1)
        else:
            raise ValueError("Invalid indices shape")

        decoded_z = self.project_out(q_vector)
        return decoded_z