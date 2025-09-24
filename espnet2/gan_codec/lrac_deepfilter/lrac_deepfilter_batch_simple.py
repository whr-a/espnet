# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Lrac with DeepFilterNet Enhancement - Simplified Batch Processing."""
import functools
import math
import random
import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple

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


class LracGeneratorBatchDF(nn.Module):
    """Lrac Generator with DeepFilterNet - Simplified Batch Processing."""

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
        """Initialize Lrac Generator with DeepFilterNet."""
        super().__init__()

        self.use_deepfilter = use_deepfilter
        self.sample_rate = sample_rate

        if self.use_deepfilter:
            self.init_deepfilter()

        # Initialize codec components
        if not encoder_params:
            encoder_params = self.get_default_init_params()["encoder_params"]
        if not decoder_params:
            decoder_params = self.get_default_init_params()["decoder_params"]
        if not quantizer_params:
            quantizer_params = self.get_default_init_params()["quantizer_params"]

        self.encoder = GenericSEANetEncoder(**encoder_params)
        self.decoder = GenericSEANetDecoder(**decoder_params)

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
            logging.info(f"Loading pretrained encoder weights from {encoder_path}")
            checkpoint = torch.load(preload_path, map_location="cpu")
            # Adjust this prefix based on how your checkpoint is saved
            prefix = 'codec.generator.encoder.' 
            encoder_state_dict = {
                k.replace(prefix, ''): v
                for k, v in checkpoint.get('state_dict', checkpoint).items()
                if k.startswith(prefix)
            }
            if not encoder_state_dict:
                raise KeyError(f"No encoder weights found with prefix '{prefix}' in {encoder_path}")
            self.encoder.load_state_dict(encoder_state_dict)

            # Load pretrained VQ if a path is provided
            logging.info(f"Loading pretrained VQ weights from {preload_path}")
            checkpoint = torch.load(preload_path, map_location="cpu")
            # Adjust this prefix based on how your checkpoint is saved
            prefix = 'codec.generator.quantizer.'
            vq_state_dict = {
                k.replace(prefix, ''): v
                for k, v in checkpoint.get('state_dict', checkpoint).items()
                if k.startswith(prefix)
            }
            if not vq_state_dict:
                raise KeyError(f"No VQ weights found with prefix '{prefix}' in {preload_path}")
            self.quantizer.load_state_dict(vq_state_dict)

        if fix:
            for param in self.parameters():
                param.requires_grad = False
            logging.info("All generator parameters have been frozen.")

    def init_deepfilter(self):
        """Initialize DeepFilterNet model."""
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

        p = ModelParams()

        # Create DF states
        self.df_state = DF(
            sr=p.sr,  # 48000
            fft_size=p.fft_size,  # 960
            hop_size=p.hop_size,  # 480
            nb_bands=p.nb_erb,
            min_nb_erb_freqs=p.min_nb_freqs,
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

        # Freeze parameters
        for param in self.df_model.parameters():
            param.requires_grad = False

        # Store parameters
        self.nb_df = p.nb_df
        self.nb_erb = p.nb_erb
        self.n_fft_24k = 480
        self.hop_size_24k = 240

        # Get ERB filterbank weights
        self.erb_fb_weights = self.df_state.erb_widths()
        self.norm_alpha = get_norm_alpha(False)

        logging.info(f"DeepFilterNet initialized on device {device}")

    def process_batch_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Process batch of audio through STFT using DF state.

        Args:
            audio: [B, C, T] audio tensor

        Returns:
            Complex STFT tensor [B, C, T_frames, F]
        """
        B, C, T = audio.shape
        device = audio.device

        specs = []
        for b in range(B):
            # Process each item
            audio_np = audio[b].cpu().numpy()  # [C, T]
            spec = self.df_state_24k.analysis(audio_np)  # [C, T_frames, F=241]
            specs.append(torch.from_numpy(spec))

        # Stack batch
        spec_batch = torch.stack(specs, dim=0).to(device)  # [B, C, T_frames, F]

        return spec_batch

    def process_batch_istft(self, spec: torch.Tensor) -> torch.Tensor:
        """Process batch of spectrogram through iSTFT using DF state.

        Args:
            spec: Complex spectrogram [B, C, T_frames, F]

        Returns:
            Audio tensor [B, C, T]
        """
        B, C, T_frames, F = spec.shape
        device = spec.device

        audios = []
        for b in range(B):
            # Process each item
            spec_np = spec[b].cpu().numpy()  # [C, T_frames, F]
            audio = self.df_state_24k.synthesis(spec_np)  # [C, T]
            audios.append(torch.from_numpy(audio))

        # Stack batch
        audio_batch = torch.stack(audios, dim=0).to(device)  # [B, C, T]

        return audio_batch

    @torch.no_grad()
    def enhance_24k_batch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply DeepFilterNet enhancement to batch.

        Uses vectorized operations where possible, but falls back to
        processing ERB/unit norm features in a loop for compatibility.
        """
        import time
        if not self.use_deepfilter:
            return audio

        B, C, T_orig = audio.shape
        device = audio.device

        # Reset hidden states
        if hasattr(self.df_model, "reset_h0"):
            self.df_model.reset_h0(batch_size=B, device=device)

        # Pad audio
        audio_padded = F.pad(audio, (0, self.n_fft_24k))
        t1 = time.time()
        # STFT
        spec_24k = self.process_batch_stft(audio_padded)  # [B, C, T_frames, F=241]
        # t2 = time.time() - t1
        # print(t2)
        # Pad to 48k
        spec_48k = F.pad(spec_24k, (0, 240), "constant", 0)  # [B, C, T_frames, F=481]

        # Process features - vectorized for batch
        specs = []
        erb_feats = []
        spec_feats = []

        spec_np = spec_48k.cpu().numpy()

        # Process all items in parallel using numpy operations
        for b in range(B):
            spec_b = spec_np[b]  # [C, T_frames, F]

            # ERB features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                erb_output = erb_norm(erb(spec_b, self.erb_fb_weights), self.norm_alpha)

            # Spec features
            spec_feat = unit_norm(spec_b[..., :self.nb_df], self.norm_alpha)

            erb_feats.append(erb_output)
            spec_feats.append(spec_feat)
            specs.append(spec_b)
        # t3 = time.time() - t1 - t2
        # print(t3)
        # Convert to tensors
        spec_real = as_real(torch.from_numpy(np.stack(specs, axis=0)).to(device))  # [B, C, T, F, 2]
        erb_feat = torch.from_numpy(np.stack(erb_feats, axis=0)).to(device)  # [B, C, T, nb_erb]
        spec_feat = as_real(torch.from_numpy(np.stack(spec_feats, axis=0)).to(device))  # [B, C, T, nb_df, 2]

        # Model input shape adjustment
        if C > 1:
            spec_real = spec_real[:, 0:1]
            erb_feat = erb_feat[:, 0:1]
            spec_feat = spec_feat[:, 0:1]

        # Run model (GPU)
        enhanced_spec = self.df_model(spec_real, erb_feat, spec_feat)[0]  # [B, 1, T, F, 2]
        # Convert to complex
        enhanced_complex = as_complex(enhanced_spec.squeeze(1))  # [B, T, F]

        # Extract 24k bins
        enhanced_24k = enhanced_complex[..., :241]  # [B, T, F=241]
        enhanced_24k = enhanced_24k.unsqueeze(1)  # [B, C=1, T, F]
        # t4 = time.time() - t1 - t2 - t3
        # print(t4)
        # iSTFT
        enhanced_audio = self.process_batch_istft(enhanced_24k)  # [B, C, T]

        # Remove padding
        d = self.n_fft_24k - self.hop_size_24k
        enhanced_audio = enhanced_audio[:, :, d:T_orig + d]
        # t5 = time.time() - t1 - t2 - t3 - t4
        # print(t5)
        return enhanced_audio

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """Forward pass with optional DeepFilterNet enhancement."""
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

    # def load_pretrained(self, checkpoint_path: str):
    #     """Load pretrained generator weights."""
    #     logging.info(f"Loading generator weights from {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path, map_location="cpu")

    #     full_state_dict = checkpoint.get('state_dict', checkpoint)
    #     prefix = 'codec.generator.'

    #     generator_state_dict = {
    #         k.replace(prefix, ''): v
    #         for k, v in full_state_dict.items()
    #         if k.startswith(prefix)
    #     }

    #     if not generator_state_dict:
    #         raise KeyError(f"No generator weights found with prefix '{prefix}'")

    #     missing_keys, unexpected_keys = self.load_state_dict(generator_state_dict, strict=False)

    #     if missing_keys:
    #         logging.warning(f"Missing keys: {missing_keys}")
    #     if unexpected_keys:
    #         logging.warning(f"Unexpected keys: {unexpected_keys}")

    #     logging.info("Generator weights loaded successfully")

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