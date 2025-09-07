# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# Adapted from seanet.py: 
# https://github.com/espnet/espnet/blob/master/espnet2/gan_codec/shared/encoder/seanet.py

# Adapted from https://github.com/facebookresearch/encodec by Jiatong Shi

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

# Revised by Yusuf Ziya Isik @ Cisco International Ltd 
# for LRAC Challenge baseline system
# providing more control on hyperparameters


"""Encodec SEANet-based encoder implementation."""


from typing import List, Dict, Any, Optional
import numpy as np

from torch import nn

from espnet2.gan_codec.shared.encoder.seanet import SConv1d, SLSTM, Snake1d


class GenericSEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    This class gives more control on the hyperparameters
    compared to SEANetResnetBlock.

    Args:
        dim (int): Input/output dimension
        kernel_sizes (list): List of convolution kernel sizes.
        dilations (list): List of dilation rates for convolutions.
        activation (str): Name of activation function (e.g. "ReLU", "ELU").
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization
            used along with the convolution.
        causality_modes (List[bool]): Causality setting per convolution.
        pad_mode (str): Padding mode used in the convolutions.
        compress (int): Dimensionality reduction factor for internal layers.
        true_skip (bool): Use identity or learnable 1x1 conv for residual skip.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        activation: str = "ELU",
        activation_params: Optional[Dict[str, Any]] = None,
        norm: str = "weight_norm",
        norm_params: Optional[Dict[str, Any]] = None,
        causality_modes: Optional[List[bool]] = None,
        pad_mode: str = "zero",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()

        kernel_sizes = kernel_sizes or [3, 1]
        dilations = dilations or [1] * len(kernel_sizes)
        causality_modes = causality_modes or [True] * len(kernel_sizes)
        activation_params = activation_params or {"alpha": 1.0}
        norm_params = norm_params or {}

        assert len(kernel_sizes) == len(dilations), \
            "Number of kernel sizes must match number of dilations"
        assert len(kernel_sizes) == len(causality_modes), \
            "Number of kernel sizes must match number of causality modes"
        
        if activation == "Snake":
            activation_fn = Snake1d
        else:
            activation_fn = getattr(nn, activation)
        
        hidden_dim = dim // compress
        layers = []
        
        for i, (k, d, causal) in enumerate(zip(kernel_sizes, dilations, causality_modes)):
            in_ch = dim if i == 0 else hidden_dim
            out_ch = dim if i == len(kernel_sizes) - 1 else hidden_dim
            layers += [
                activation_fn(**activation_params),
                SConv1d(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    dilation=d,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*layers)
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=True,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class GenericSEANetEncoder(nn.Module):
    """SEANet encoder.
    This class allows more control on the kernel sizes, latency and dilations
    in the convolution layers compared to SEANetEncoder.

    Args:
        input_channels (int): Audio channels.
        output_dimension (int): Output embedding dimension.
        enable_output_layer (bool): Whether to have a final output conv layer or not
        n_filters (List[int]): Number of channels per block.
        strides (List[int]): strides for convolution layers.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization
            used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        input_kernel_size (int): Kernel size for input convolutionlayer
        output_kernel_size (int): Kernel size for the output convolution.
        input_layer_causal (bool): Whether input layer is causal or not.
        output_layer_causal (bool): Whether output layer is causal or not.
        downsampling_kernel_sizes (List[int]): Kernel sizes for the downsampling layers.
        residual_kernel_sizes (List[List[int]]): Kernel sizes for the residual layers.
        residual_dilations (List[List[int]]): Dilation rates for the residual layers.
        residual_causality_modes (List[List[bool]]): Causality modes for residual layers.
        downsampling_causality_modes (List[bool]): Causality modes for downsampling layers.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple (streamable)
            convolution as the skip connection in the residual network blocks.
        compress (int): Dimensionality reduction factor for internal layers.
        lstm (int): Number of LSTM layers at the end of the encoder.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_dimension: int = 128,
        enable_output_layer: bool = True,
        n_filters: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        activation: str = "ELU",
        activation_params: Optional[Dict[str, Any]] = None,
        norm: str = "weight_norm",
        norm_params: Dict[str, Any] = None,
        input_kernel_size: int = 7,
        output_kernel_size: int = 7,
        input_layer_causal: bool = True,
        output_layer_causal: bool = True,
        downsampling_kernel_sizes: Optional[List[int]] = None,
        residual_kernel_sizes: Optional[List[List[int]]] = None,
        residual_dilations: Optional[List[List[int]]] = None,
        residual_causality_modes: Optional[List[List[bool]]] = None,
        downsampling_causality_modes: Optional[List[bool]] = None,
        pad_mode: str = "constant",
        true_skip: bool = True,
        compress: int = 1,
        lstm: int = 0,
    ):
        super().__init__()
        n_filters = n_filters or self.get_default_init_params()['n_filters']
        strides = strides or self.get_default_init_params()['strides']
        activation_params = activation_params or self.get_default_init_params()['activation_params']
        norm_params = norm_params or self.get_default_init_params()['norm_params']
        
        # Default residual parameters
        downsampling_kernel_sizes = downsampling_kernel_sizes or self.get_default_init_params()['downsampling_kernel_sizes']
        downsampling_causality_modes = downsampling_causality_modes or self.get_default_init_params()['downsampling_causality_modes']
        residual_kernel_sizes = residual_kernel_sizes or self.get_default_init_params()['residual_kernel_sizes']
        residual_dilations = residual_dilations or self.get_default_init_params()['residual_dilations']
        residual_causality_modes = residual_causality_modes or self.get_default_init_params()['residual_causality_modes']
        
        self.enable_output_layer = enable_output_layer
        self.input_channels = input_channels
        self.output_dimension = output_dimension
        self.n_filters = n_filters
        self.strides = strides
        self.hop_length = int(np.prod(self.strides))

        if activation == "Snake":
            activation_fn = Snake1d
        else:
            activation_fn = getattr(nn, activation)
        
        if not self.enable_output_layer:
            assert self.output_dimension == n_filters[-1], \
                "Without an output layer, the output dimension should be the same as the last filter dimension!!"
            
        model: List[nn.Module] = [
            SConv1d(
                input_channels, n_filters[0], input_kernel_size,
                norm=norm, norm_kwargs=norm_params,
                causal=input_layer_causal, pad_mode=pad_mode,
            )
        ]
        # Residual + downsampling layers
        for idx, (stride, ks, residual_kernels, dilation_pairs, causality_modes) in enumerate(zip(
            self.strides, downsampling_kernel_sizes, residual_kernel_sizes, 
            residual_dilations, residual_causality_modes)):
            # Add residual layers
            for kernels, dilations, causality in zip(residual_kernels, dilation_pairs, causality_modes):
                model += [
                    GenericSEANetResnetBlock(
                        self.n_filters[idx],
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

            # Add downsampling layers
            model += [
                activation_fn(**activation_params),
                SConv1d(
                    n_filters[idx],
                    n_filters[idx+1],
                    kernel_size=ks,
                    stride=stride,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=downsampling_causality_modes[idx],
                    pad_mode=pad_mode,
                ),
            ]

        if lstm:
            model += [SLSTM(n_filters[-1], num_layers=lstm)]

        if self.enable_output_layer:
            model += [
                activation_fn(**activation_params),
                SConv1d(
                    n_filters[-1],
                    self.output_dimension,
                    output_kernel_size,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=output_layer_causal,
                    pad_mode=pad_mode,
                ),
            ]

        self.model = nn.Sequential(*model)

    @staticmethod
    def get_default_init_params():
        init_params = {
            "input_channels": 1,
            "output_dimension": 128,
            "enable_output_layer": False,
            "n_filters": [8, 16, 32, 64, 128],
            "strides": [3, 4, 4, 5],
            "activation": "ELU",
            "activation_params": {},
            "norm": "weight_norm",
            "norm_params": {},
            "input_kernel_size": 7,
            "output_kernel_size": 7,
            "input_layer_causal": True,
            "output_layer_causal": True,
            "downsampling_kernel_sizes": [6, 8, 8, 10],
            "residual_kernel_sizes": [[[5, 5]] * 3 for _ in range(4)],
            "residual_dilations": [[[1, 1], [1, 3], [1, 9]] for _ in range(4)],
            "residual_causality_modes": [[[True, True]] * 3 for _ in range(4)],
            "downsampling_causality_modes": [True] * 4,
            "pad_mode": "constant",
            "true_skip": True,
            "compress": 1,
            "lstm": 0,
        }
        return init_params
        
    def forward(self, x):
        return self.model(x)
