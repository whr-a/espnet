# Copyright 2024 Yihan Wu (Modified for paired data training)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""LRACodec Modules for Paired Clean-Noisy Data Training."""
import functools
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_codec.lracodec.lracodec import (
    LRACodec,
    LRACodecGenerator,
    LRACodecDiscriminator,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class LRACodecPaired(LRACodec):
    """LRACodec model for paired clean-noisy data training.
    
    This extends the base LRACodec to support training with paired data where:
    - Noisy audio is used as input to the encoder
    - Clean audio is used as the reconstruction target
    """

    def forward(
        self,
        audio: torch.Tensor,
        audio_noisy: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform forward with optional paired data.

        Args:
            audio (Tensor): Clean audio waveform tensor (B, T_wav).
            audio_noisy (Optional[Tensor]): Noisy audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if audio_noisy is not None:
            # Use paired data training
            if forward_generator:
                return self._forward_generator_paired(
                    audio_clean=audio,
                    audio_noisy=audio_noisy,
                    **kwargs,
                )
            else:
                return self._forward_discriminator_paired(
                    audio_clean=audio,
                    audio_noisy=audio_noisy,
                    **kwargs,
                )
        else:
            # Fall back to standard self-reconstruction
            if forward_generator:
                return self._forward_generator(
                    audio=audio,
                    **kwargs,
                )
            else:
                return self._forward_discrminator(
                    audio=audio,
                    **kwargs,
                )

    def _forward_generator_paired(
        self,
        audio_clean: torch.Tensor,
        audio_noisy: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward with paired clean-noisy data.

        Args:
            audio_clean (Tensor): Clean audio waveform tensor (B, T_wav).
            audio_noisy (Tensor): Noisy audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = audio_clean.size(0)

        # Add channel dimension
        audio_clean = audio_clean.unsqueeze(1)
        audio_noisy = audio_noisy.unsqueeze(1)

        # calculate generator outputs using noisy input
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            # Use the paired forward method of the generator
            if isinstance(self.generator, LRACodecGeneratorPaired):
                audio_hat, codec_commit_loss, quantization_loss, audio_hat_real = (
                    self.generator.forward_paired(
                        x_noisy=audio_noisy,
                        x_clean=audio_clean,
                        use_dual_decoder=self.use_dual_decoder
                    )
                )
            else:
                # Fallback for compatibility
                audio_hat, codec_commit_loss, quantization_loss, audio_hat_real = (
                    self.generator(audio_noisy, use_dual_decoder=self.use_dual_decoder)
                )
        else:
            audio_hat, codec_commit_loss, quantization_loss, audio_hat_real = (
                self._cache
            )

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                audio_hat_real,
            )

        # calculate discriminator outputs
        p_hat = self.discriminator(audio_hat)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(audio_clean)  # Use clean audio as real

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        codec_commit_loss = codec_commit_loss * self.lambda_commit
        codec_quantization_loss = quantization_loss * self.lambda_quantization
        # Reconstruction loss: compare output with clean target
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio_clean, audio_hat) * self.lambda_reconstruct
        )
        codec_loss = codec_commit_loss + codec_quantization_loss
        loss = adv_loss + codec_loss + reconstruct_loss
        stats = dict(
            adv_loss=adv_loss.item(),
            codec_loss=codec_loss.item(),
            codec_commit_loss=codec_commit_loss.item(),
            codec_quantization_loss=codec_quantization_loss.item(),
            reconstruct_loss=reconstruct_loss.item(),
        )
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            mel_loss = self.mel_loss(audio_hat, audio_clean)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + mel_loss
            stats.update(mel_loss=mel_loss.item())
            if self.use_dual_decoder and audio_hat_real is not None:
                mel_loss_real = self.mel_loss(audio_hat_real, audio_clean)
                mel_loss_real = self.lambda_mel * mel_loss_real
                loss = loss + mel_loss_real
                stats.update(mel_loss_real=mel_loss_real.item())

        stats.update(loss=loss.item())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def _forward_discriminator_paired(
        self,
        audio_clean: torch.Tensor,
        audio_noisy: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform discriminator forward with paired clean-noisy data.

        Args:
            audio_clean (Tensor): Clean audio waveform tensor (B, T_wav).
            audio_noisy (Tensor): Noisy audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = audio_clean.size(0)
        audio_clean = audio_clean.unsqueeze(1)
        audio_noisy = audio_noisy.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            if isinstance(self.generator, LRACodecGeneratorPaired):
                audio_hat, codec_commit_loss, codec_quantization_loss, audio_hat_real = (
                    self.generator.forward_paired(
                        x_noisy=audio_noisy,
                        x_clean=audio_clean,
                        use_dual_decoder=self.use_dual_decoder,
                    )
                )
            else:
                # Fallback for compatibility
                audio_hat, codec_commit_loss, codec_quantization_loss, audio_hat_real = (
                    self.generator(audio_noisy, use_dual_decoder=self.use_dual_decoder)
                )
        else:
            audio_hat, codec_commit_loss, codec_quantization_loss, audio_hat_real = (
                self._cache
            )

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                codec_quantization_loss,
                audio_hat_real,
            )

        # calculate discriminator outputs
        p_hat = self.discriminator(audio_hat.detach())
        p = self.discriminator(audio_clean)  # Use clean audio as real

        # calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            real_loss=real_loss.item(),
            fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }


class LRACodecGeneratorPaired(LRACodecGenerator):
    """LRACodec generator module with paired data support."""

    def forward_paired(
        self, 
        x_noisy: torch.Tensor, 
        x_clean: torch.Tensor,
        use_dual_decoder: bool = False
    ):
        """Forward propagation with paired clean-noisy data.

        Args:
            x_noisy (torch.Tensor): Noisy input tensor of shape (B, 1, T).
            x_clean (torch.Tensor): Clean target tensor of shape (B, 1, T).
            use_dual_decoder (bool): Whether to use dual decoder for encoder out.
            
        Returns:
            torch.Tensor: resynthesized audio.
            torch.Tensor: commitment loss.
            torch.Tensor: quantization loss
            torch.Tensor: resynthesized audio from encoder (if dual decoder).
        """
        # Extract acoustic features from noisy input
        encoder_out = self.encoder(x_noisy)
        
        # Use AutoGroupResidualVectorQuantize
        quantized, codes, latents, commit_loss, quantization_loss = self.quantizer(encoder_out)
        
        # Get semantic codes from clean audio for better semantic extraction
        # This helps the model learn to extract clean semantics even from noisy input
        semantic_codes = self.get_semantic_codes(x_clean)  # [B, T_semantic]
        
        # Dequantize semantic codes for decoding
        semantic_features_dequant = self.semantic_dequantizer(semantic_codes)  # [B, T_semantic, semantic_dim]
        semantic_features_dequant = semantic_features_dequant.transpose(1, 2)  # [B, semantic_dim, T_semantic]
        
        # Upsample semantic features to match acoustic feature length
        if semantic_features_dequant.shape[-1] != quantized.shape[-1]:
            semantic_features_dequant = F.interpolate(
                semantic_features_dequant, 
                size=quantized.shape[-1], 
                mode='linear', 
                align_corners=False
            )
        
        # Concatenate semantic and acoustic features
        combined_features = torch.cat(
            [semantic_features_dequant, quantized], 
            dim=1
        )  # [B, semantic_dim + hidden_dim, T]
        
        # Project to decoder input dimension
        combined_features = combined_features.transpose(1, 2)  # [B, T, semantic_dim + hidden_dim]
        decoder_input = self.decoder_proj(combined_features).transpose(1, 2)  # [B, hidden_dim, T]

        resyn_audio = self.decoder(decoder_input)

        if use_dual_decoder:
            # For dual decoder, also combine semantic with original encoder output
            encoder_combined = torch.cat(
                [semantic_features_dequant, encoder_out], 
                dim=1
            )
            encoder_combined = encoder_combined.transpose(1, 2)
            encoder_proj = self.decoder_proj(encoder_combined).transpose(1, 2)
            resyn_audio_real = self.decoder(encoder_proj)
        else:
            resyn_audio_real = None
            
        return resyn_audio, commit_loss, quantization_loss, resyn_audio_real