# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Lrac with DeepFilterNet Enhancement - Batch Processing Version."""
import functools
import math
import random
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_codec.shared.encoder.generic_seanet import GenericSEANetEncoder
from espnet2.gan_codec.shared.decoder.generic_seanet import GenericSEANetDecoder
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer

# DeepFilterNet imports
from df.checkpoint import load_model as load_model_cp
from df.config import config
from df.model import ModelParams
from df.utils import as_complex, as_real, get_norm_alpha
from libdf import DF, erb, erb_norm, unit_norm


class LracGeneratorWithDF(nn.Module):
    """Lrac Generator with DeepFilterNet Enhancement (Batch Processing)."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        encoder_params: Dict[str, Any] = {},
        decoder_params: Dict[str, Any] = {},
        quantizer_params: Dict[str, Any] = {},
        preload: bool = False,
        preload_path: str = "",
        fix: bool = False,
        use_deepfilter: bool = True,
    ):
        """Initialize Lrac Generator with DeepFilterNet.

        Args:
            sample_rate: Audio sampling rate (24000)
            encoder_params: Encoder configuration
            decoder_params: Decoder configuration
            quantizer_params: Quantizer configuration
            preload: Whether to preload weights
            preload_path: Path to preload weights from
            fix: Whether to freeze generator parameters
            use_deepfilter: Whether to use DeepFilterNet enhancement
        """
        super().__init__()

        self.use_deepfilter = use_deepfilter
        self.sample_rate = sample_rate

        if self.use_deepfilter:
            self.init_deepfilter()

        # Initialize codec components with default params if needed
        if not encoder_params:
            encoder_params = self.get_default_init_params()["encoder_params"]
        if not decoder_params:
            decoder_params = self.get_default_init_params()["decoder_params"]
        if not quantizer_params:
            quantizer_params = self.get_default_init_params()["quantizer_params"]

        self.encoder = GenericSEANetEncoder(**encoder_params)
        self.decoder = GenericSEANetDecoder(**decoder_params)

        # Extract and remove target_bandwidth from quantizer_params
        self.target_bandwidths = quantizer_params.pop("target_bandwidth", [1, 6])

        self.quantizer = ResidualVectorQuantizer(
            dimension=encoder_params['output_dimension'],
            **quantizer_params
        )

        self.frame_rate = math.ceil(sample_rate / np.prod(encoder_params['strides']))

        # Loss functions
        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

        if preload:
            self.load_pretrained(preload_path)

        if fix:
            for param in self.parameters():
                param.requires_grad = False
            logging.info("All generator parameters have been frozen.")

    def init_deepfilter(self):
        """Initialize DeepFilterNet model for 24kHz processing."""
        # Model directory
        model_base_dir = os.path.expanduser("~/.cache/DeepFilterNet/DeepFilterNet2")

        if not os.path.isdir(model_base_dir):
            raise NotADirectoryError(f"DeepFilterNet model directory not found at {model_base_dir}")

        # Load config
        config.load(
            os.path.join(model_base_dir, "config.ini"),
            config_must_exist=True,
            allow_defaults=True,
            allow_reload=True,
        )

        # Create model parameters
        p = ModelParams()

        # Create DF states for 48k (original) and 24k (half)
        self.df_state = DF(
            sr=p.sr,  # 48000
            fft_size=p.fft_size,  # 960
            hop_size=p.hop_size,  # 480
            nb_bands=p.nb_erb,  # 32
            min_nb_erb_freqs=p.min_nb_freqs,  # 2
        )

        self.df_state_24k = DF(
            sr=24000,
            fft_size=480,
            hop_size=240,
            nb_bands=p.nb_erb,
            min_nb_erb_freqs=p.min_nb_freqs,
        )

        # Load model
        checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
        self.df_model, epoch = load_model_cp(checkpoint_dir, self.df_state, epoch="best", mask_only=False)

        if epoch is None or epoch == 0:
            raise RuntimeError("Could not find DeepFilterNet checkpoint")

        logging.info(f"Loaded DeepFilterNet checkpoint from epoch {epoch}")

        # Setup model
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        self.df_model = self.df_model.to(device)
        self.df_model.eval()

        # Freeze DeepFilterNet parameters
        for param in self.df_model.parameters():
            param.requires_grad = False

        # Store parameters
        self.nb_df = p.nb_df  # 96
        self.nb_erb = p.nb_erb  # 32
        self.n_fft_24k = 480
        self.hop_size_24k = 240

        logging.info(f"DeepFilterNet initialized on device {device}")

    def df_features_24k(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract DeepFilterNet features for 24kHz audio.

        Args:
            audio: Audio tensor [B, C, T]

        Returns:
            Tuple of (spec, erb_feat, spec_feat) tensors
        """
        B, C, T = audio.shape
        device = audio.device

        # Process each item in batch
        specs = []
        erb_feats = []
        spec_feats = []

        for b in range(B):
            # Get single audio [C, T]
            audio_single = audio[b].cpu().numpy()

            # STFT using DF state (returns complex spectrogram)
            spec_24k = self.df_state_24k.analysis(audio_single)  # [C, T_frames, F=241]

            # Pad to 48k frequency dimension
            spec_24k_torch = torch.from_numpy(spec_24k).to(device)
            spec_48k = F.pad(spec_24k_torch, (0, 240), "constant", 0)  # [C, T_frames, F=481]

            # Convert back to numpy for libdf functions
            spec_np = spec_48k.cpu().numpy()

            # Get normalization alpha
            a = get_norm_alpha(False)

            # Compute ERB features
            erb_fb = self.df_state.erb_widths()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                erb_output = erb_norm(erb(spec_np, erb_fb), a)  # [C, T_frames, nb_erb]

            # Compute spec features
            spec_feat_output = unit_norm(spec_np[..., :self.nb_df], a)  # [C, T_frames, nb_df]

            # Convert to tensors and add to lists
            erb_feats.append(torch.from_numpy(erb_output))
            spec_feats.append(torch.from_numpy(spec_feat_output))
            specs.append(spec_48k)

        # Stack batch dimension
        spec_batch = torch.stack(specs, dim=0).to(device)  # [B, C, T_frames, F]
        erb_feat_batch = torch.stack(erb_feats, dim=0).to(device)  # [B, C, T_frames, nb_erb]
        spec_feat_batch = torch.stack(spec_feats, dim=0).to(device)  # [B, C, T_frames, nb_df]

        # Convert to real representation and add dimension for model
        spec_real = as_real(spec_batch).unsqueeze(1)  # [B, 1, C, T_frames, F, 2]
        erb_feat_real = erb_feat_batch.unsqueeze(1)  # [B, 1, C, T_frames, nb_erb]
        spec_feat_real = as_real(spec_feat_batch).unsqueeze(1)  # [B, 1, C, T_frames, nb_df, 2]

        # Take only first channel if multi-channel
        if C > 1:
            spec_real = spec_real[:, :, 0:1]
            erb_feat_real = erb_feat_real[:, :, 0:1]
            spec_feat_real = spec_feat_real[:, :, 0:1]

        # Squeeze extra dimension
        spec_real = spec_real.squeeze(1)  # [B, C=1, T_frames, F, 2]
        erb_feat_real = erb_feat_real.squeeze(1)  # [B, C=1, T_frames, nb_erb]
        spec_feat_real = spec_feat_real.squeeze(1)  # [B, C=1, T_frames, nb_df, 2]

        return spec_real, erb_feat_real, spec_feat_real

    @torch.no_grad()
    def enhance_24k_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply DeepFilterNet enhancement to batch of 24kHz audio.

        Args:
            audio: Audio tensor [B, C, T]

        Returns:
            Enhanced audio [B, C, T]
        """
        if not self.use_deepfilter:
            return audio

        B, C, T_orig = audio.shape
        device = audio.device

        # Reset hidden states
        if hasattr(self.df_model, "reset_h0"):
            self.df_model.reset_h0(batch_size=B, device=device)

        # Pad audio
        audio_padded = F.pad(audio, (0, self.n_fft_24k))

        # Extract features
        spec, erb_feat, spec_feat = self.df_features_24k(audio_padded)

        # Run enhancement model
        enhanced_spec = self.df_model(spec, erb_feat, spec_feat)[0]  # [B, 1, T_frames, F, 2]

        # Convert to complex
        enhanced_complex = as_complex(enhanced_spec.squeeze(1))  # [B, T_frames, F]

        # Extract 24k bins
        enhanced_24k = enhanced_complex[..., :241]  # [B, T_frames, F=241]

        # Synthesis for each item in batch
        enhanced_audio = []
        for b in range(B):
            # Get single enhanced spectrum [C=1, T_frames, F]
            enhanced_single = enhanced_24k[b].unsqueeze(0).cpu().numpy()

            # iSTFT using DF state
            audio_enhanced = self.df_state_24k.synthesis(enhanced_single)  # [C=1, T]
            enhanced_audio.append(torch.from_numpy(audio_enhanced))

        # Stack batch
        enhanced_batch = torch.stack(enhanced_audio, dim=0).to(device)  # [B, C, T]

        # Remove padding
        d = self.n_fft_24k - self.hop_size_24k
        enhanced_batch = enhanced_batch[:, :, d:T_orig + d]

        return enhanced_batch

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """Forward pass with optional DeepFilterNet enhancement.

        Args:
            x: Input audio [B, C, T] at 24kHz
            use_dual_decoder: Whether to use dual decoder

        Returns:
            Tuple of (resyn_audio, commit_loss, quantization_loss, resyn_audio_real)
        """
        # Apply enhancement if enabled
        if self.use_deepfilter:
            x = self.enhance_24k_batch(x)

        # Encode
        encoder_out = self.encoder(x)

        # Select bandwidth
        max_idx = len(self.target_bandwidths) - 1
        bw = self.target_bandwidths[random.randint(0, max_idx)]

        # Quantize
        quantized, _, _, commit_loss = self.quantizer(encoder_out, self.frame_rate, bw)

        # Compute quantization loss
        quantization_loss = self.l1_quantization_loss(
            encoder_out, quantized.detach()
        ) + self.l2_quantization_loss(encoder_out, quantized.detach())

        # Decode
        resyn_audio = self.decoder(quantized)

        # Optional dual decoder
        if use_dual_decoder:
            resyn_audio_real = self.decoder(encoder_out)
        else:
            resyn_audio_real = None

        return resyn_audio, commit_loss, quantization_loss, resyn_audio_real

    def encode(self, x: torch.Tensor, target_bw: Optional[float] = None):
        """Encode audio to codes."""
        if self.use_deepfilter:
            x = self.enhance_24k_batch(x)

        encoder_out = self.encoder(x)
        bw = target_bw if target_bw is not None else self.target_bandwidths[-1]
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor):
        """Decode codes to audio."""
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)
        return resyn_audio

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained generator weights."""
        logging.info(f"Loading generator weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract generator state dict
        full_state_dict = checkpoint.get('state_dict', checkpoint)
        prefix = 'codec.generator.'

        generator_state_dict = {
            k.replace(prefix, ''): v
            for k, v in full_state_dict.items()
            if k.startswith(prefix)
        }

        if not generator_state_dict:
            raise KeyError(f"No generator weights found with prefix '{prefix}'")

        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(generator_state_dict, strict=False)

        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")

        logging.info("Generator weights loaded successfully")

    @staticmethod
    def get_default_init_params():
        """Get default initialization parameters."""
        return {
            "encoder_params": GenericSEANetEncoder.get_default_init_params(),
            "decoder_params": GenericSEANetDecoder.get_default_init_params(),
            "quantizer_params": {
                "codebook_dim": 128,
                "n_q": 6,
                "bins": 1024,
                "decay": 0.99,
                "kmeans_init": True,
                "kmeans_iters": 50,
                "threshold_ema_dead_code": 2,
                "target_bandwidth": [1, 6]
            }
        }