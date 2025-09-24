# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Lrac with DeepFilterNet Enhancement Module."""
import functools
import math
import random
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.shared.encoder.generic_seanet import GenericSEANetEncoder
from espnet2.gan_codec.shared.decoder.generic_seanet import GenericSEANetDecoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.loss.semantic_loss import HubertLoss
from espnet2.gan_codec.shared.loss.loss_balancer import Balancer
from espnet2.gan_codec.shared.loss.arecho_loss import ArechoLoss
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_codec.shared.discriminator.msstft_discriminator import (
    MultiScaleSTFTDiscriminator,
)
from espnet2.gan_codec.shared.discriminator.msmpmb_discriminator import (
    MultiScaleMultiPeriodMultiBandDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable

# DeepFilterNet imports
from df.checkpoint import load_model as load_model_cp
from df.config import config
from df.model import ModelParams
from df.utils import as_complex, as_real, get_norm_alpha
from libdf import DF, erb, erb_norm, unit_norm


class LracEnhancedGenerator(nn.Module):
    """Lrac Generator with DeepFilterNet Enhancement."""

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
        if self.use_deepfilter:
            self.init_deepfilter_24k()

        # Initialize codec components
        encoder_params = encoder_params or self.get_default_init_params()["encoder_params"]
        decoder_params = decoder_params or self.get_default_init_params()["decoder_params"]
        quantizer_params = quantizer_params or self.get_default_init_params()["quantizer_params"]

        self.encoder = GenericSEANetEncoder(**encoder_params)
        self.decoder = GenericSEANetDecoder(**decoder_params)
        self.target_bandwidths = quantizer_params.pop("target_bandwidth", None)
        self.quantizer = ResidualVectorQuantizer(
            dimension=encoder_params['output_dimension'],
            **quantizer_params
        )
        self.sample_rate = sample_rate
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

    def init_deepfilter_24k(self):
        """Initialize DeepFilterNet model with hardcoded 24k configuration."""
        # Hardcoded configuration for 24k processing
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

        # Create DF states - using half size for 24k (original is 48k)
        self.df_state_48k = DF(
            sr=p.sr,  # 48000
            fft_size=p.fft_size,  # 960
            hop_size=p.hop_size,  # 480
            nb_bands=p.nb_erb,  # 32
            min_nb_erb_freqs=p.min_nb_freqs,  # 2
        )

        self.df_state_24k = DF(
            sr=24000,  # Force 24k
            fft_size=480,  # Half of 960
            hop_size=240,  # Half of 480
            nb_bands=p.nb_erb,  # 32
            min_nb_erb_freqs=p.min_nb_freqs,  # 2
        )

        # Load model checkpoint
        checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
        self.df_model, epoch = load_model_cp(checkpoint_dir, self.df_state_48k, epoch="best", mask_only=False)

        if epoch is None or epoch == 0:
            raise RuntimeError("Could not find a DeepFilterNet checkpoint")

        logging.info(f"Loaded DeepFilterNet checkpoint from epoch {epoch}")

        # Move model to current device and set to eval mode
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        self.df_model = self.df_model.to(device)
        self.df_model.eval()

        # Freeze DeepFilterNet model parameters
        for param in self.df_model.parameters():
            param.requires_grad = False

        # Store configuration parameters
        self.nb_df = p.nb_df  # 96
        self.nb_erb = p.nb_erb  # 32
        self.n_fft_24k = 480
        self.hop_size_24k = 240

        # Pre-compute ERB filter bank
        self.erb_fb = torch.from_numpy(self.df_state_48k.erb_widths()).to(device)
        self.norm_alpha = get_norm_alpha(False)

        logging.info(f"DeepFilterNet initialized on device {device}")

    def torch_stft_24k_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """Perform batched STFT using torch.

        Args:
            audio: Batch tensor [B, C, T] at 24kHz

        Returns:
            Complex STFT tensor [B, C, T_frames, F_bins]
        """
        device = audio.device
        B, C, T = audio.shape

        # Create window - use square root hann for perfect reconstruction
        if not hasattr(self, 'stft_window_24k'):
            # Using sqrt(hann) window for COLA constraint with 50% overlap
            window = torch.hann_window(self.n_fft_24k)
            # For 50% overlap (hop = n_fft/2), sqrt(hann) gives perfect reconstruction
            self.stft_window_24k = torch.sqrt(window).to(device)

        # Reshape for batch STFT
        audio_flat = audio.reshape(B * C, T)  # [B*C, T]

        # Perform batched STFT
        stft_result = torch.stft(
            audio_flat,
            n_fft=self.n_fft_24k,
            hop_length=self.hop_size_24k,
            win_length=self.n_fft_24k,
            window=self.stft_window_24k,
            return_complex=True,
            center=True,  # Use center=True for proper padding
            normalized=False,
            onesided=True
        )  # [B*C, F, T_frames]

        # Reshape and permute
        _, F, T_frames = stft_result.shape
        stft_result = stft_result.reshape(B, C, F, T_frames)
        stft_result = stft_result.permute(0, 1, 3, 2)  # [B, C, T_frames, F]

        return stft_result

    def torch_istft_24k_batch(self, spec: torch.Tensor) -> torch.Tensor:
        """Perform batched inverse STFT using torch.

        Args:
            spec: Complex STFT tensor [B, C, T_frames, F] at 24kHz

        Returns:
            Audio tensor [B, C, T]
        """
        device = spec.device
        B, C, T_frames, F = spec.shape

        # Create window - use square root hann for perfect reconstruction
        if not hasattr(self, 'stft_window_24k'):
            window = torch.hann_window(self.n_fft_24k)
            self.stft_window_24k = torch.sqrt(window).to(device)

        # Permute and reshape for batch iSTFT
        spec = spec.permute(0, 1, 3, 2)  # [B, C, F, T_frames]
        spec_flat = spec.reshape(B * C, F, T_frames)  # [B*C, F, T_frames]

        # Perform batched iSTFT
        audio_flat = torch.istft(
            spec_flat,
            n_fft=self.n_fft_24k,
            hop_length=self.hop_size_24k,
            win_length=self.n_fft_24k,
            window=self.stft_window_24k,
            center=True,  # Match STFT
            normalized=False,
            onesided=True
        )  # [B*C, T]

        # Reshape back
        T = audio_flat.shape[-1]
        audio = audio_flat.reshape(B, C, T)

        return audio

    def compute_df_features_batch(self, spec_24k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DeepFilterNet features for batch.

        Args:
            spec_24k: Complex spectrogram [B, C, T, F=241] at 24kHz

        Returns:
            Tuple of (spec, erb_feat, spec_feat) for 48kHz model
        """
        device = spec_24k.device
        B, C, T_frames, _ = spec_24k.shape

        # Pad to 48k frequency dimension
        spec_48k = F.pad(spec_24k, (0, 240), "constant", 0)  # [B, C, T, F=481]

        # Convert to numpy for libdf processing
        spec_np = spec_48k.detach().cpu().numpy()

        # Process ERB and spec features
        erb_feats = []
        spec_feats = []

        for b in range(B):
            # Process each item in batch
            spec_b = spec_np[b]  # [C, T, F] - this is the expected format for libdf

            # Compute ERB features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                erb_output = erb_norm(erb(spec_b, self.erb_fb.cpu().numpy()), self.norm_alpha)
                erb_feats.append(erb_output)

            # Compute spec features
            spec_norm = unit_norm(spec_b[..., :self.nb_df], self.norm_alpha)
            spec_feats.append(spec_norm)

        # Stack and convert to tensors
        erb_feat = torch.from_numpy(np.stack(erb_feats, axis=0)).to(device)  # [B, C, T, nb_erb]
        spec_feat_complex = torch.from_numpy(np.stack(spec_feats, axis=0)).to(device)  # [B, C, T, nb_df]
        spec_feat = as_real(spec_feat_complex)  # [B, C, T, nb_df, 2]

        # Convert full spec to real
        spec_real = as_real(spec_48k)  # [B, C, T, F, 2]

        return spec_real, erb_feat, spec_feat

    @torch.no_grad()
    def enhance_batch_24k(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply DeepFilterNet enhancement to batch of 24k audio.

        Args:
            audio: Input audio [B, C, T] at 24kHz

        Returns:
            Enhanced audio [B, C, T]
        """
        if not self.use_deepfilter:
            return audio

        B = audio.shape[0]
        device = audio.device
        orig_len = audio.shape[-1]

        # Reset hidden states
        if hasattr(self.df_model, "reset_h0"):
            self.df_model.reset_h0(batch_size=B, device=device)

        # Pad for STFT processing
        audio_padded = F.pad(audio, (0, self.n_fft_24k))

        # Compute STFT
        spec_24k = self.torch_stft_24k_batch(audio_padded)  # [B, C, T_frames, F=241]

        # Compute features
        spec, erb_feat, spec_feat = self.compute_df_features_batch(spec_24k)

        # Reshape for model input
        # Model expects: spec [B, 1, T, F, 2], erb [B, 1, T, nb_erb], spec_feat [B, 1, T, nb_df, 2]
        if spec.dim() == 5 and spec.shape[1] == 1:
            pass  # Already correct shape
        elif spec.dim() == 5:
            spec = spec[:, 0:1]  # Take first channel only
            erb_feat = erb_feat[:, 0:1]
            spec_feat = spec_feat[:, 0:1]

        # Run enhancement model
        enhanced_spec = self.df_model(spec, erb_feat, spec_feat)[0]  # [B, 1, T, F, 2]

        # Convert to complex and extract 24k bins
        enhanced_complex = as_complex(enhanced_spec.squeeze(1))  # [B, T, F]
        enhanced_24k = enhanced_complex[..., :241]  # [B, T, F=241]

        # Add channel dimension for iSTFT
        enhanced_24k = enhanced_24k.unsqueeze(1)  # [B, C=1, T, F]

        # Perform iSTFT
        enhanced_audio = self.torch_istft_24k_batch(enhanced_24k)

        # Remove padding
        d = self.n_fft_24k - self.hop_size_24k
        enhanced_audio = enhanced_audio[:, :, d:orig_len + d]

        return enhanced_audio

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """Forward pass with optional DeepFilterNet enhancement.

        Args:
            x: Input audio [B, C, T] at 24kHz
            use_dual_decoder: Whether to use dual decoder

        Returns:
            Tuple of (resyn_audio, commit_loss, quantization_loss, resyn_audio_real)
        """
        # Apply DeepFilterNet enhancement if enabled
        if self.use_deepfilter:
            x_enhanced = self.enhance_batch_24k(x)
        else:
            x_enhanced = x

        # Encode
        encoder_out = self.encoder(x_enhanced)

        # Randomly select bandwidth for training
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
        """Encode audio to codes with optional enhancement.

        Args:
            x: Input audio [B, C, T]
            target_bw: Target bandwidth

        Returns:
            Codes tensor
        """
        # Apply enhancement if enabled
        if self.use_deepfilter:
            x = self.enhance_batch_24k(x)

        encoder_out = self.encoder(x)
        bw = target_bw if target_bw is not None else self.target_bandwidths[-1]
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor):
        """Decode codes to audio.

        Args:
            codes: Neural codes

        Returns:
            Decoded audio
        """
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
                "quantizer_target_bandwidth": [1, 6]
            }
        }