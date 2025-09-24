# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Lrac with DeepFilterNet Enhancement - Full GPU Batch Processing."""
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
from libdf import DF


class LracGeneratorWithDFGPU(nn.Module):
    """Lrac Generator with DeepFilterNet Enhancement - Full GPU Processing."""

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

        # We only need DF state for getting ERB filter banks
        self.df_state = DF(
            sr=p.sr,  # 48000
            fft_size=p.fft_size,  # 960
            hop_size=p.hop_size,  # 480
            nb_bands=p.nb_erb,  # 32
            min_nb_erb_freqs=p.min_nb_freqs,  # 2
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

        # Store parameters for 24k processing
        self.nb_df = p.nb_df  # 96
        self.nb_erb = p.nb_erb  # 32
        self.n_fft_24k = 480
        self.hop_size_24k = 240
        self.n_fft_48k = p.fft_size  # 960
        self.hop_size_48k = p.hop_size  # 480

        # Pre-compute ERB filter bank as tensor for GPU processing
        # The erb_widths() returns the ERB filterbank weights
        erb_fb_np = self.df_state.erb_widths()

        # Check dimensions - it might be 1D or 2D
        if erb_fb_np.ndim == 1:
            # 1D array - this is the ERB widths, we need to construct the filterbank
            # For now, let's create a simple ERB filterbank matrix
            n_freqs = 481  # 48k STFT frequency bins
            erb_fb = np.zeros((self.nb_erb, n_freqs))

            # Create triangular filters for each ERB band
            freqs = np.linspace(0, 24000, n_freqs)  # Frequency axis up to Nyquist
            erb_centers = np.linspace(0, 24000, self.nb_erb + 2)[1:-1]  # ERB center frequencies

            for i in range(self.nb_erb):
                center = erb_centers[i]
                # Simple triangular filter centered at ERB frequency
                if i > 0:
                    low = erb_centers[i-1]
                else:
                    low = 0
                if i < self.nb_erb - 1:
                    high = erb_centers[i+1]
                else:
                    high = 24000

                # Create triangular filter
                erb_fb[i] = np.maximum(0, 1 - np.abs(freqs - center) / (high - low))

            self.erb_fb = torch.from_numpy(erb_fb).float().to(device)
        elif erb_fb_np.ndim == 2:
            # 2D array - use as is
            if erb_fb_np.shape[0] == self.nb_erb:
                self.erb_fb = torch.from_numpy(erb_fb_np).float().to(device)
            else:
                self.erb_fb = torch.from_numpy(erb_fb_np.T).float().to(device)
        else:
            raise ValueError(f"Unexpected ERB filter bank dimensions: {erb_fb_np.shape}")

        logging.info(f"ERB filter bank shape: {self.erb_fb.shape}")
        self.norm_alpha = get_norm_alpha(False)

        # Create windows for STFT/iSTFT
        self.register_buffer('window_24k', torch.hann_window(self.n_fft_24k))

        logging.info(f"DeepFilterNet initialized on device {device}")

    def stft_24k(self, audio: torch.Tensor) -> torch.Tensor:
        """Perform STFT on 24kHz audio using PyTorch.

        Args:
            audio: [B, C, T] audio tensor

        Returns:
            Complex STFT [B, C, T_frames, F_bins]
        """
        B, C, T = audio.shape
        device = audio.device

        # Flatten batch and channel for STFT
        audio_flat = audio.view(B * C, T)

        # Perform STFT
        stft = torch.stft(
            audio_flat,
            n_fft=self.n_fft_24k,
            hop_length=self.hop_size_24k,
            win_length=self.n_fft_24k,
            window=self.window_24k,
            center=False,
            return_complex=True,
            onesided=False  # Full spectrum
        )  # [B*C, F, T_frames]

        # Reshape and permute
        _, F, T_frames = stft.shape
        stft = stft.view(B, C, F, T_frames).permute(0, 1, 3, 2)  # [B, C, T_frames, F]

        # Only keep positive frequencies (241 bins for 24kHz with n_fft=480)
        stft = stft[..., :241]

        return stft

    def istft_24k(self, spec: torch.Tensor) -> torch.Tensor:
        """Perform inverse STFT for 24kHz using PyTorch.

        Args:
            spec: Complex STFT [B, C, T_frames, F=241]

        Returns:
            Audio tensor [B, C, T]
        """
        B, C, T_frames, F = spec.shape
        device = spec.device

        # Create full spectrum by mirroring (for real-valued output)
        # DC and Nyquist should not be duplicated
        spec_conj = torch.conj(spec[..., 1:-1].flip(-1))  # Mirror and conjugate
        spec_full = torch.cat([spec, spec_conj], dim=-1)  # [B, C, T_frames, F=480]

        # Permute and flatten
        spec_flat = spec_full.permute(0, 1, 3, 2).reshape(B * C, self.n_fft_24k, T_frames)

        # Perform iSTFT
        audio_flat = torch.istft(
            spec_flat,
            n_fft=self.n_fft_24k,
            hop_length=self.hop_size_24k,
            win_length=self.n_fft_24k,
            window=self.window_24k,
            center=False,
            return_complex=False,
            onesided=False
        )

        # Reshape
        T = audio_flat.shape[-1]
        audio = audio_flat.view(B, C, T)

        return audio

    def compute_erb_features_gpu(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute ERB features using GPU operations.

        Args:
            spec: Complex spectrogram [B, C, T_frames, F]

        Returns:
            ERB features [B, C, T_frames, nb_erb]
        """
        # Compute power spectrum
        power = torch.abs(spec) ** 2  # [B, C, T_frames, F]

        # Apply ERB filterbank using matrix multiplication
        # power: [B, C, T_frames, F=481], erb_fb: [nb_erb=32, F=481]
        # We need to transpose erb_fb to [F=481, nb_erb=32] for matmul
        # Result: [B, C, T_frames, nb_erb]
        erb_power = torch.matmul(power, self.erb_fb.transpose(0, 1))

        # Apply ERB normalization (similar to erb_norm but on GPU)
        erb_feat = torch.sqrt(erb_power + 1e-10)

        # Normalize with alpha
        erb_feat = torch.pow(erb_feat, self.norm_alpha)

        return erb_feat

    def compute_unit_norm_features_gpu(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute unit norm features using GPU operations.

        Args:
            spec: Complex spectrogram [B, C, T_frames, F]

        Returns:
            Unit norm features [B, C, T_frames, F]
        """
        # Apply unit normalization (similar to unit_norm but on GPU)
        magnitude = torch.abs(spec)

        # Normalize magnitude with alpha
        magnitude_norm = torch.pow(magnitude + 1e-10, self.norm_alpha)

        # Preserve phase
        phase = torch.angle(spec)
        spec_norm = magnitude_norm * torch.exp(1j * phase)

        return spec_norm

    @torch.no_grad()
    def enhance_24k_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply DeepFilterNet enhancement to batch of 24kHz audio - fully on GPU.

        Args:
            audio: Audio tensor [B, C, T] at 24kHz

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

        # STFT using PyTorch (GPU)
        spec_24k = self.stft_24k(audio_padded)  # [B, C, T_frames, F=241]

        # Pad to 48k frequency dimension
        spec_48k = F.pad(spec_24k, (0, 240), "constant", 0)  # [B, C, T_frames, F=481]

        # Compute ERB features on GPU
        erb_feat = self.compute_erb_features_gpu(spec_48k)  # [B, C, T_frames, nb_erb]

        # Compute unit norm features on GPU
        spec_feat = self.compute_unit_norm_features_gpu(spec_48k[..., :self.nb_df])  # [B, C, T_frames, nb_df]

        # Convert to real representation for model input
        spec_real = as_real(spec_48k)  # [B, C, T_frames, F, 2]
        spec_feat_real = as_real(spec_feat)  # [B, C, T_frames, nb_df, 2]

        # Model expects [B, 1, T_frames, F, 2] for spec
        if C > 1:
            # Process only first channel
            spec_real = spec_real[:, 0:1]
            erb_feat = erb_feat[:, 0:1]
            spec_feat_real = spec_feat_real[:, 0:1]

        # Run enhancement model (fully on GPU)
        enhanced_spec = self.df_model(spec_real, erb_feat, spec_feat_real)[0]  # [B, 1, T_frames, F, 2]

        # Convert back to complex
        enhanced_complex = as_complex(enhanced_spec.squeeze(1))  # [B, T_frames, F]

        # Extract 24k bins
        enhanced_24k = enhanced_complex[..., :241]  # [B, T_frames, F=241]

        # Add channel dimension back
        enhanced_24k = enhanced_24k.unsqueeze(1)  # [B, C=1, T_frames, F]

        # iSTFT using PyTorch (GPU)
        enhanced_audio = self.istft_24k(enhanced_24k)  # [B, C, T]

        # Remove padding
        d = self.n_fft_24k - self.hop_size_24k
        enhanced_audio = enhanced_audio[:, :, d:T_orig + d]

        return enhanced_audio

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """Forward pass with optional DeepFilterNet enhancement.

        Args:
            x: Input audio [B, C, T] at 24kHz
            use_dual_decoder: Whether to use dual decoder

        Returns:
            Tuple of (resyn_audio, commit_loss, quantization_loss, resyn_audio_real)
        """
        # Apply enhancement if enabled (fully on GPU)
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