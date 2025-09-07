# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# Adapted from seanet.py: 
# https://github.com/espnet/espnet/blob/master/espnet2/gan_codec/shared/decoder/seanet.py

# Adapted from https://github.com/facebookresearch/encodec by Jiatong Shi

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

# Revised by Yusuf Ziya Isik @ Cisco International Ltd 
# for LRAC Challenge baseline system
# providing more control on hyperparameters

"""Encodec SEANet-based decoder implementation."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa
from torch.nn.utils import spectral_norm, weight_norm  # noqa

from espnet2.gan_codec.shared.encoder.seanet import (
    SLSTM,
    SConv1d,
    apply_parametrization_norm,
    get_norm_module,
)
from espnet2.gan_codec.shared.encoder.generic_seanet import (
    GenericSEANetResnetBlock
)
from espnet2.gan_codec.shared.encoder.snake_activation import Snake1d


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv

    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding

    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from
        # `pad_for_conv1d` would be removed at the very end,
        # when keeping only the right length for the output, as
        # removing it here would require also passing the length
        # at the matching layer in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class GenericSEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        input_dimension (int): Input embedding dimension.
        output_dimension (int): Number of output audio channels.
        enable_input_layer (bool): Whether to have an input layer or not.
        n_filters (List[int]): Number of channels per block.
        strides (List[int]): Strides of upsampling conv layers.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization
            used along with the convolution.
        input_kernel_size (int): Kernel size for the initial convolution.
        output_kernel_size (int): Kernel size for the output convolution.
        input_layer_causal (bool): Whether the input layer is causal or not.
        output_layer_causal (bool): Whether the output layer is causal or not.
        upsampling_kernel_sizes (List[int]): kernel sizes for the upsampling layers.
        residual_kernel_sizes (List[List[int]]): Kernel sizes for the residual layers.
        residual_dilations (List[List[int]]): Dilation rates for the residual layers.
        residual_causality_modes (List[List[bool]]): Causality modes for residual layers.
        upsampling_causality_modes (List[bool]): Causality modes for upsampling layers.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple (streamable)
            convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed
            convolution under the causal setup. If equal to 1.0, it means that all
            the trimming is done at the right.
    """

    def __init__(
        self,
        input_dimension: int = 128,
        output_dimension: int = 1,
        enable_input_layer: bool = True,
        n_filters: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: Optional[str] = None,
        final_activation_params: Optional[dict] = None,
        norm: str = "weight_norm",
        norm_params: Dict[str, Any] = None,
        input_kernel_size: int = 7,
        output_kernel_size: int = 7,
        input_layer_causal: bool = True,
        output_layer_causal: bool = True,
        upsampling_kernel_sizes: Optional[List[int]] = None,
        residual_kernel_sizes: Optional[List[List[int]]] = None,
        residual_dilations: Optional[List[List[int]]] = None,
        residual_causality_modes: Optional[List[List[bool]]] = None,
        upsampling_causality_modes: Optional[List[List[bool]]] = None,
        pad_mode: str = "constant",
        true_skip: bool = True,
        compress: int = 1,
        lstm: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        
        n_filters = n_filters or self.get_default_init_params()['n_filters']
        strides = strides or self.get_default_init_params()['strides']
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hop_length = np.prod(strides)
        self.strides = strides

        # Default conv layer parameters
        upsampling_kernel_sizes = upsampling_kernel_sizes or self.get_default_init_params()['upsampling_kernel_sizes']
        upsampling_causality_modes = upsampling_causality_modes or self.get_default_init_params()['upsampling_causality_modes']
        residual_kernel_sizes = residual_kernel_sizes or self.get_default_init_params()['residual_kernel_sizes']
        residual_dilations = residual_dilations or self.get_default_init_params()['residual_dilations']
        residual_causality_modes = residual_causality_modes or self.get_default_init_params()['residual_causality_modes']
        norm_params = norm_params or self.get_default_init_params()['norm_params']

        if activation == "Snake":
            act = Snake1d
        else:
            act = getattr(nn, activation)
        
        if enable_input_layer:
            model: List[nn.Module] = [
                SConv1d(
                    input_dimension,
                    n_filters[0],
                    input_kernel_size,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=input_layer_causal,
                    pad_mode=pad_mode,
                )
            ]
        else:
            model: List[nn.Module] = []

        if lstm:
            model += [SLSTM(n_filters[0], num_layers=lstm)]

        # Upsampling + Residual layers
        for idx, (stride, ks, residual_kernels, dilation_pairs, causality_modes) in enumerate(zip(
            self.strides, upsampling_kernel_sizes, residual_kernel_sizes, 
            residual_dilations, residual_causality_modes)):
            # Add upsampling layers
            model += [
                act(**activation_params),
                SConvTranspose1d(
                    n_filters[idx],
                    n_filters[idx+1],
                    kernel_size=ks,
                    stride=stride,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=upsampling_causality_modes[idx],
                    trim_right_ratio=trim_right_ratio,
                ),
            ]

            # Add residual layers
            for kernels, dilations, causality in zip(residual_kernels, dilation_pairs, causality_modes):
                model += [
                    GenericSEANetResnetBlock(
                        n_filters[idx+1],
                        kernel_sizes=kernels,
                        dilations=dilations,
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causality_modes=causality,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

        # Add final layers
        model += [
            act(**activation_params),
            SConv1d(
                n_filters[-1],
                output_dimension,
                output_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=output_layer_causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    @staticmethod
    def get_default_init_params():
        init = {
            "input_dimension": 128,
            "output_dimension": 1,
            "enable_input_layer": False,
            "n_filters": [64, 32, 16, 8],
            "strides": [5, 4, 4, 3],
            "activation": "ELU",
            "activation_params": {"alpha": 1.0},
            "final_activation": None,
            "final_activation_params": None,
            "norm": "weight_norm",
            "norm_params": {},
            "input_kernel_size": 7,
            "output_kernel_size": 7,
            "input_layer_causal": True,
            "output_layer_causal": True,
            "upsampling_kernel_sizes": [5, 4, 4, 3],
            "residual_kernel_sizes": [[[5, 5]] * 3 for _ in range(4)],
            "residual_dilations": [[[1, 1], [1, 3], [1, 9]] for _ in range(4)],
            "residual_causality_modes": [[[True, True]] * 3 for _ in range(4)],
            "upsampling_causality_modes": [True] * 4,
            "pad_mode": "constant",
            "true_skip": True,
            "compress": 1,
            "lstm": 0,
            "trim_right_ratio": 1.0,
        }
        return init
    
    def forward(self, z):
        y = self.model(z)
        return y
