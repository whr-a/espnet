# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Lrac Modules."""
import functools
import math
import random
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
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

from df.checkpoint import load_model as load_model_cp
from df.config import config
from df.io import load_audio, load_audio_24k, resample, save_audio
from df.logger import init_logger
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, as_real, download_file, get_cache_dir, get_norm_alpha
from df.version import version
from libdf import DF, erb, erb_norm, unit_norm
import warnings
import os

PRETRAINED_MODELS = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")
DEFAULT_MODEL = "DeepFilterNet2"

class Lrac_rewrite(AbsGANCodec):
    """Lrac model."""

    @typechecked
    def __init__(
        self,
        apply_enhancement: bool = False,
        sampling_rate: int = 24000,
        preload: bool = False,
        preload_path: str = "",
        fix_gen: bool = False,
        encoder_params: Dict[str, Any] = None,
        decoder_params: Dict[str, Any] = None,
        quantizer_params: Dict[str, Any] = None,
        discriminator_params: Dict[str, Any] = None,
        # loss related
        generator_adv_loss_params: Dict[str, Any] = None,
        discriminator_adv_loss_params: Dict[str, Any] = None,
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Dict[str, Any] = None,
        use_mel_loss: bool = True,
        mel_loss_params: Dict[str, Any] = None,
        use_semantic_loss: bool = False,
        use_arecho_loss: bool = False,
        arecho_loss_params: Dict[str, Any] = {},
        semantic_loss_params: Dict[str, Any] = None,
        use_dual_decoder: bool = True,
        lambda_quantization: float = 1.0,
        lambda_commit: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_mel: float = 45.0,
        lambda_feat_match: float = 2.0,
        lambda_semantic: float = 0.0,
        lambda_arecho: float = 0.0,
        cache_generator_outputs: bool = False,
        use_loss_balancer: bool = False,
        balance_ema_decay: float = 0.99,
    ):
        """Intialize Lrac model.

        Args:
             TODO(jiatong)
        """
        super().__init__()

        # Whether the codec applies speech enhancement such as
        # denoising and dereverb or not
        self.apply_enhancement = apply_enhancement
        # define modules

        self.generator = Lrac_rewriteGenerator(
            sample_rate=sampling_rate,
            preload=preload,
            preload_path=preload_path,
            fix=fix_gen,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            quantizer_params=quantizer_params,
        )
        self.discriminator = Lrac_rewriteDiscriminator(**discriminator_params)
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.generator_reconstruct_loss = torch.nn.L1Loss(reduction="mean")
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(
                **feat_match_loss_params,
            )
        self.use_mel_loss = use_mel_loss
        mel_loss_params.update(fs=sampling_rate)
        if self.use_mel_loss:
            self.mel_loss = MultiScaleMelSpectrogramLoss(
                **mel_loss_params,
            )
        self.use_dual_decoder = use_dual_decoder
        if self.use_dual_decoder:
            assert self.use_mel_loss, "only use dual decoder with Mel loss"
        
        self.use_semantic_loss = use_semantic_loss
        if self.use_semantic_loss:
            semantic_loss_params = semantic_loss_params or {
                "sample_rate": 24000,
                "model_name": "WAVLM_LARGE",
                "feature_ids": None
            }
            self.semantic_loss = HubertLoss(
                **semantic_loss_params)
        self.use_arecho_loss = use_arecho_loss
        if self.use_arecho_loss:
            arecho_loss_params = arecho_loss_params or {
                "target_metrics": ['scoreq_ref', 'nomad', 'utmos', 'scoreq_nr', 'sheet_ssqa', 'audiobox_aesthetics_CE', 'audiobox_aesthetics_PQ', 'audiobox_aesthetics_CU'],
                "loss_type": "mae",
                "model_tag": None,
                "arecho_train_config": "/work/nvme/bbjs/bsu5/universa/espnet/egs2/universa_unite/uni_versa1/exp/universa_universa_ar_overall_scale_token_wavlm_decode_lrac/config.yaml",
                "model_file": "/work/nvme/bbjs/shi3/evaluation/espnet/egs2/universa_unite/uni_versa1/exp/universa_universa_ar_overall_scale_token_wavlm/latest.pth",
                "dtype": "float32",
                "seed": 777,
                "always_fix_seed": False,
                "beam_size": 1,
                "skip_meta_label_score": False,
                "save_token_seq": False,
                "use_fixed_order": False,
                "fixed_metric_name_order": "",
                "device": "cuda",
            }
            self.arecho_loss = ArechoLoss(
                **arecho_loss_params)

        # coefficients
        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_commit = lambda_commit
        self.lambda_adv = lambda_adv
        if self.use_feat_match_loss:
            self.lambda_feat_match = lambda_feat_match
        if self.use_mel_loss:
            self.lambda_mel = lambda_mel
        if self.use_semantic_loss:
            self.lambda_semantic = lambda_semantic
        if self.use_arecho_loss:
            self.lambda_arecho = lambda_arecho
        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate
        self.num_streams = quantizer_params["n_q"]
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, encoder_params["strides"]
        )
        self.code_size_per_stream = [
            quantizer_params["bins"]
        ] * self.num_streams

        # loss balancer
        if use_loss_balancer:
            self.loss_balancer = Balancer(
                ema_decay=balance_ema_decay,
                per_batch_item=True,
            )
        else:
            self.loss_balancer = None

    def meta_info(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "num_streams": self.num_streams,
            "frame_shift": self.frame_shift,
            "code_size_per_stream": self.code_size_per_stream,
        }

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
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

    def _forward_generator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = audio.size(0)

        # TODO(jiatong): double check the multi-channel input
        audio = audio.unsqueeze(1)
        if self.apply_enhancement:
            ref_audio = kwargs['speech_ref1'].unsqueeze(1)
            assert audio.shape == ref_audio.shape, \
                'Length mismatch between input and reference audio'
        else:
            ref_audio = audio

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat, codec_commit_loss, quantization_loss, audio_hat_real = (
                self.generator(audio, use_dual_decoder=self.use_dual_decoder)
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
            p = self.discriminator(ref_audio)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        codec_commit_loss = codec_commit_loss * self.lambda_commit
        codec_quantization_loss = quantization_loss * self.lambda_quantization
        reconstruct_loss = (
            self.generator_reconstruct_loss(ref_audio, audio_hat) * self.lambda_reconstruct
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
            mel_loss = self.mel_loss(audio_hat, ref_audio)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + mel_loss
            stats.update(mel_loss=mel_loss.item())
            if self.use_dual_decoder:
                mel_loss_real = self.mel_loss(audio_hat_real, ref_audio)
                mel_loss_real = self.lambda_mel * mel_loss_real
                loss = loss + mel_loss_real
                stats.update(mel_loss_real=mel_loss_real.item())
        if self.use_semantic_loss:
            semantic_loss = self.semantic_loss(audio_hat, ref_audio)
            semantic_loss = self.lambda_semantic * semantic_loss
            loss = loss + semantic_loss
            stats.update(semantic_loss=semantic_loss.item())
        if self.use_arecho_loss:
            arecho_loss = self.arecho_loss(audio_hat, ref_audio)
            arecho_loss = self.lambda_arecho * arecho_loss
            loss = loss + arecho_loss
            # logging.info("-"*100)
            # logging.info(arecho_loss.grad_fn)
            # logging.info("-"*100)
            stats.update(arecho_loss=arecho_loss.item())

        stats.update(loss=loss.item())

        if self.loss_balancer is not None and self.training:
            # any loss built on audio_hat is processed by balancer
            balanced_losses = {
                "reconstruct": reconstruct_loss,
                "adv": adv_loss,
            }
            if self.use_feat_match_loss:
                balanced_losses.update(feat_match=feat_match_loss)
            if self.use_mel_loss:
                balanced_losses.update(mel=mel_loss)
            if self.use_semantic_loss:
                balanced_losses.update(semantic=semantic_loss)
            if self.use_arecho_loss:
                balanced_losses.update(arecho=arecho_loss)
            balanced_loss, norm_stats = self.loss_balancer(balanced_losses, audio_hat)
            stats.update(norm_stats)

            loss = sum(balanced_loss.values()) + codec_loss
            if self.use_mel_loss and self.use_dual_decoder:
                loss = loss + mel_loss_real
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

    def _forward_discrminator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """

        # setup
        batch_size = audio.size(0)
        audio = audio.unsqueeze(1)

        if self.apply_enhancement:
            ref_audio = kwargs['speech_ref1'].unsqueeze(1)
            assert audio.shape == ref_audio.shape, \
                'Length mismatch between input and reference audio'
        else:
            ref_audio = audio

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat, codec_commit_loss, codec_quantization_loss, audio_hat_real = (
                self.generator(
                    audio,
                    use_dual_decoder=self.use_dual_decoder,
                )
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
        p = self.discriminator(ref_audio)

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

    def inference(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * codec (Tensor): Generated neural codec (T_code, N_stream).

        """
        codec = self.generator.encode(x, **kwargs)
        wav = self.generator.decode(codec)

        return {"wav": wav, "codec": codec}

    def encode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Tensor: Generated codes (T_code, N_stream).

        """
        target_bw = kwargs.get('target_bw', None)
        return self.generator.encode(x, target_bw=target_bw)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input codes (T_code, N_stream).

        Returns:
            Tensor: Generated waveform (T_wav,).

        """
        return self.generator.decode(x)


class Lrac_rewriteGenerator(nn.Module):
    """SoundStream generator module."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        encoder_params: Dict[str, Any] = None,
        decoder_params: Dict[str, Any] = None,
        quantizer_params: Dict[str, Any] = None,
        preload: bool = False,
        preload_path: str = "",
        fix: bool = False,
    ):
        """Initialize SoundStream Generator.

        Args:
            TODO(jiatong)
        """
        super().__init__()

        # Initialize DeepFilterNet enhancement module
        self.init_deepfilter_24k()

        encoder_params = encoder_params or self.get_default_init_params()["encoder_params"]
        decoder_params = decoder_params or self.get_default_init_params()["decoder_params"]
        quantizer_params = quantizer_params or self.get_default_init_params()["quantizer_params"]

        # Initialize encoder
        self.encoder = GenericSEANetEncoder(**encoder_params)
        self.decoder = GenericSEANetDecoder(**decoder_params)
        self.target_bandwidths = quantizer_params.pop("target_bandwidth", None)
        self.quantizer = ResidualVectorQuantizer(
            dimension=encoder_params['output_dimension'],
            **quantizer_params
        )
        self.sample_rate = sample_rate
        self.frame_rate = math.ceil(sample_rate / np.prod(encoder_params['strides']))

        # quantization loss
        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

        if preload:
            logging.info(f"Attempting to preload generator weights from {preload_path}")
            try:
                checkpoint = torch.load(preload_path, map_location="cpu")
                
                full_state_dict = checkpoint.get('state_dict', checkpoint)

                prefix = 'codec.generator.'
                
                generator_state_dict = {
                    k.replace(prefix, ''): v 
                    for k, v in full_state_dict.items() 
                    if k.startswith(prefix)
                }

                if not generator_state_dict:
                    raise KeyError(
                        f"Could not find any keys with the prefix '{prefix}' in the checkpoint at '{preload_path}'. "
                        "Please verify the checkpoint structure."
                    )

                missing_keys, unexpected_keys = self.load_state_dict(generator_state_dict, strict=True)

                if unexpected_keys:
                    logging.warning(f"Unexpected keys in checkpoint not loaded: {unexpected_keys}")
                if missing_keys:
                    logging.warning(f"Missing keys in model not initialized: {missing_keys}")
                
                logging.info(f"Successfully preloaded generator from {preload_path}")

            except FileNotFoundError:
                logging.error(f"Preload checkpoint file not found: {preload_path}")
                raise
            except Exception as e:
                logging.error(f"An error occurred while preloading the model: {e}")
                raise
        if fix:
            for param in self.parameters():
                param.requires_grad = False
            logging.info("All generator parameters have been frozen. They will not be updated during training.")

    def init_deepfilter_24k(self):
        """Initialize DeepFilterNet model with hardcoded 24k configuration."""
        import numpy as np

        # Hardcoded configuration for 24k processing
        model_base_dir = os.path.expanduser("~/.cache/DeepFilterNet/DeepFilterNet2")
        post_filter: bool = False,
        log_level: str = "INFO",
        log_file: Optional[str] = "enhance.log",
        config_allow_defaults: bool = True,
        epoch: Union[str, int, None] = "best",
        default_model: str = DEFAULT_MODEL,
        mask_only: bool = False,
    ) -> Tuple[nn.Module, DF, str, int]:
        """Initializes and loads config, model and deep filtering state.

        Args:
            model_base_dir (str): Path to the model directory containing checkpoint and config. If None,
                load the pretrained DeepFilterNet2 model.
            post_filter (bool): Enable post filter for some minor, extra noise reduction.
            log_level (str): Control amount of logging. Defaults to `INFO`.
            log_file (str): Optional log file name. None disables it. Defaults to `enhance.log`.
            config_allow_defaults (bool): Whether to allow initializing new config values with defaults.
            epoch (str): Checkpoint epoch to load. Options are `best`, `latest`, `<int>`, and `none`.
                `none` disables checkpoint loading. Defaults to `best`.

        Returns:
            model (nn.Modules): Intialized model, moved to GPU if available.
            df_state (DF): Deep filtering state for stft/istft/erb
            suffix (str): Suffix based on the model name. This can be used for saving the enhanced
                audio.
            epoch (int): Epoch number of the loaded checkpoint.
        """
        try:
            from icecream import ic, install

            ic.configureOutput(includeContext=True)
            install()
        except ImportError:
            pass
        use_default_model = model_base_dir is None or model_base_dir in PRETRAINED_MODELS
        model_base_dir = get_model_basedir(model_base_dir or default_model)

        if not os.path.isdir(model_base_dir):
            raise NotADirectoryError("Base directory not found at {}".format(model_base_dir))
        log_file = os.path.join(model_base_dir, log_file) if log_file is not None else None
        init_logger(file=log_file, level=log_level, model=model_base_dir)
        if use_default_model:
            logger.info(f"Using {default_model} model at {model_base_dir}")
        config.load(
            os.path.join(model_base_dir, "config.ini"),
            config_must_exist=True,
            allow_defaults=config_allow_defaults,
            allow_reload=True,
        )
        if post_filter:
            config.set("mask_pf", True, bool, ModelParams().section)
            try:
                beta = config.get("pf_beta", float, ModelParams().section)
                beta = f"(beta: {beta})"
            except KeyError:
                beta = ""
            logger.info(f"Running with post-filter {beta}")
        p = ModelParams()
        df_state = DF(
            sr=p.sr,
            fft_size=p.fft_size,
            hop_size=p.hop_size,
            nb_bands=p.nb_erb,
            min_nb_erb_freqs=p.min_nb_freqs,
        )
        df_state_24k = DF(
            sr=p.sr // 2,
            fft_size=p.fft_size // 2,
            hop_size=p.hop_size // 2,
            nb_bands=p.nb_erb,
            min_nb_erb_freqs=p.min_nb_freqs,
        )
        checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
        load_cp = epoch is not None and not (isinstance(epoch, str) and epoch.lower() == "none")
        if not load_cp:
            checkpoint_dir = None
        mask_only = mask_only or config(
            "mask_only", cast=bool, section="train", default=False, save=False
        )
        model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=epoch, mask_only=mask_only)
        if (epoch is None or epoch == 0) and load_cp:
            logger.error("Could not find a checkpoint")
            exit(1)
        logger.debug(f"Loaded checkpoint from epoch {epoch}")
        model = model.to(get_device())
        # Set suffix to model name
        suffix = os.path.basename(os.path.abspath(model_base_dir))
        if post_filter:
            suffix += "_pf"
        logger.info("Running on device {}".format(get_device()))
        logger.info("Model loaded")
        return model, df_state, df_state_24k, suffix, epoch
    @staticmethod
    def init_deepfilter():
        self.model, _, self.df_state, suffix, epoch = init_df_24k(
            args.model_base_dir,
            post_filter=args.pf,
            log_level=args.log_level,
            config_allow_defaults=True,
            epoch=args.epoch,
            mask_only=args.no_df_stage,
        )
        self.model.eval()
        self.nb_df = 96
        self.n_fft = 480
        self.hop_size = 240
    def df_features_24k(audio: Tensor, df: DF, nb_df: int, device=None) -> Tuple[Tensor, Tensor, Tensor]:
        device = audio.device
        spec_24k = df.analysis(audio)
        assert spec_24k.shape[-1] == 241, "wrong 24k stft"
        spec = F.pad(spec_24k, (0, 240), "constant", 0)
        a = get_norm_alpha(False)
        erb_fb = df.erb_widths()
        if not isinstance(erb_fb, torch.Tensor):
            erb_fb = torch.from_numpy(erb_fb).to(device)
        elif erb_fb.device != device:
            erb_fb = erb_fb.to(device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            erb_output = erb_norm(erb(spec, erb_fb), a)
            erb_feat = torch.as_tensor(erb_output, device=device).unsqueeze(1)

        unit_norm_output = unit_norm(spec[..., :nb_df], a)
        spec_feat = as_real(torch.as_tensor(unit_norm_output, device=device)).unsqueeze(1)

        spec = as_real(torch.as_tensor(spec, device=device)).unsqueeze(1)
        return spec, erb_feat, spec_feat
    @torch.no_grad()
    def enhance_24k(
        model: nn.Module, df_state: DF, audio: Tensor, pad=True, atten_lim_db: Optional[float] = None
    ):
        """Enhance a single audio given a preloaded model and DF state.

        Args:
            model (nn.Module): A DeepFilterNet model.
            df_state (DF): DF state for STFT/ISTFT and feature calculation.
            audio (Tensor): Time domain audio of shape [C, T]. Sampling rate needs to match to `model` and `df_state`.
            pad (bool): Pad the audio to compensate for delay due to STFT/ISTFT.
            atten_lim_db (float): An optional noise attenuation limit in dB. E.g. an attenuation limit of
                12 dB only suppresses 12 dB and keeps the remaining noise in the resulting audio.

        Returns:
            enhanced audio (Tensor): If `pad` was `False` of shape [C, T'] where T'<T slightly delayed due to STFT.
                If `pad` was `True` it has the same shape as the input.
        """
        bs = audio.shape[0]
        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=bs, device=audio.device())
        orig_len = audio.shape[-1]
        if pad:
            audio = F.pad(audio, (0, self.n_fft))

        # print(df_state.fft_size(), df_state.hop_size())
        spec, erb_feat, spec_feat = df_features_24k(audio, df_state, nb_df, device=audio.device())
        # print(spec.shape)
        enhanced = model(spec.clone(), erb_feat, spec_feat)
        enhanced = as_complex(enhanced.squeeze(1))
        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / 20)
            enhanced = as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)
        enhanced = enhanced[..., :241]
        # print(enhanced.shape)
        # print(df_state.fft_size(), df_state.hop_size())
        audio = torch.as_tensor(df_state.synthesis(enhanced.contiguous().cpu().numpy()))
        if pad:
            # The frame size is equal to p.hop_size. Given a new frame, the STFT loop requires e.g.
            # ceil((n_fft-hop)/hop). I.e. for 50% overlap, then hop=n_fft//2
            # requires 1 additional frame lookahead; 75% requires 3 additional frames lookahead.
            # Thus, the STFT/ISTFT loop introduces an algorithmic delay of n_fft - hop.
            assert n_fft % hop == 0  # This is only tested for 50% and 75% overlap
            d = n_fft - hop
            audio = audio[:, d : orig_len + d]
        # print(audio.shape)
        return audio    

    @staticmethod
    def get_default_init_params():
        init_params= {
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
        return init_params

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """Soundstream forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
            use_dual_decoder (bool): Whether to use dual decoder for encoder out
        Returns:
            torch.Tensor: resynthesized audio.
            torch.Tensor: commitment loss.
            torch.Tensor: quantization loss
            torch.Tensor: resynthesized audio from encoder.
        """
        encoder_out = self.encoder(x)
        max_idx = len(self.target_bandwidths) - 1

        # randomly pick up one bandwidth
        bw = self.target_bandwidths[random.randint(0, max_idx)]

        # Forward quantizer
        quantized, _, _, commit_loss = self.quantizer(encoder_out, self.frame_rate, bw)

        quantization_loss = self.l1_quantization_loss(
            encoder_out, quantized.detach()
        ) + self.l2_quantization_loss(encoder_out, quantized.detach())

        resyn_audio = self.decoder(quantized)

        if use_dual_decoder:
            resyn_audio_real = self.decoder(encoder_out)
        else:
            resyn_audio_real = None
        return resyn_audio, commit_loss, quantization_loss, resyn_audio_real

    def encode(
        self,
        x: torch.Tensor,
        target_bw: Optional[float] = None,
    ):
        """Soundstream codec encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: neural codecs in shape ().
        """
        encoder_out = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor):
        """Soundstream codec decoding.

        Args:
            codecs (torch.Tensor): neural codecs in shape ().
        Returns:
            torch.Tensor: resynthesized audio.
        """
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)
        return resyn_audio


class Lrac_rewriteDiscriminator(torch.nn.Module):
    """Lrac Discriminator with only Multi-Scale STFT discriminator module"""

    def __init__(
        self,
        choose: str="msstft",
        preload: bool = False,
        preload_path: str = "",
        fix: bool = False,
        msstft_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "filters": 32,
            "norm": "weight_norm",
            "n_fft": [1024, 2048, 512, 256, 128],
            "hop_lengths": [256, 512, 128, 64, 32],
            "win_lengths": [1024, 2048, 512, 256, 128],
            "activation": "LeakyReLU",
            # "activation_params": {"negative_slope: 0.3"},
            "activation_params": {"negative_slope": 0.3}, # Bug fix. the above commented code is fixed!!
        },
        msmpmb_discriminator_params: Dict[str, Any] = {
            "rates": [],
            "fft_sizes": [2048, 1024, 512],
            "sample_rate": 24000,
            "periods": [2, 3, 5, 7, 11],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "band_discriminator_params": {
                "hop_factor": 0.25,
                "sample_rate": 24000,
                "bands": [
                    (0.0, 0.1),
                    (0.1, 0.25),
                    (0.25, 0.5),
                    (0.5, 0.75),
                    (0.75, 1.0),
                ],
                "channel": 32,
            },
        },
    ):
        """Initialize Encodec Discriminator module.

        Args: msstft_discriminator_params (Dict[str, Any]) with following arguments:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            filters (int): Number of filters in convolutions.
            norm (str): normalization choice of Convolutional layers
            n_ffts (Sequence[int]): Size of FFT for each scale.
            hop_lengths (Sequence[int]): Length of hop between STFT windows for
                each scale.
            win_lengths (Sequence[int]): Window size for each scale.
            activation (str): activation function choice of convolutional layer
            activation_params (Dict[str, Any]): parameters for activation function)
        """

        super().__init__()
        self.choose = choose
        if choose == "msstft":
            self.msstft = MultiScaleSTFTDiscriminator(**msstft_discriminator_params)
        elif choose == "msmpmb":
            self.msmpmb = MultiScaleMultiPeriodMultiBandDiscriminator(**msmpmb_discriminator_params)
        if preload:
            logging.info(f"Attempting to preload discriminator weights from {preload_path}")
            try:
                checkpoint = torch.load(preload_path, map_location="cpu")
                
                full_state_dict = checkpoint.get('state_dict', checkpoint)
                if choose == "msstft":
                    prefix = 'codec.discriminator.msstft.'
                elif choose == "msmpmb":
                    prefix = 'codec.discriminator.msmpmb_discriminator.'
                
                discriminator_state_dict = {
                    k.replace(prefix, ''): v 
                    for k, v in full_state_dict.items() 
                    if k.startswith(prefix)
                }

                if not discriminator_state_dict:
                    raise KeyError(
                        f"Could not find any keys with the prefix '{prefix}' in the checkpoint at '{preload_path}'. "
                        "Please verify the checkpoint structure."
                    )

                if choose == "msstft":
                    missing_keys, unexpected_keys = self.msstft.load_state_dict(discriminator_state_dict, strict=True)
                elif choose == "msmpmb":
                    missing_keys, unexpected_keys = self.msmpmb.load_state_dict(discriminator_state_dict, strict=True)
                

                if unexpected_keys:
                    logging.warning(f"Unexpected keys in checkpoint not loaded: {unexpected_keys}")
                if missing_keys:
                    logging.warning(f"Missing keys in model not initialized: {missing_keys}")
                
                logging.info(f"Successfully preloaded discriminator from {preload_path}")

            except FileNotFoundError:
                logging.error(f"Preload checkpoint file not found: {preload_path}")
                raise
            except Exception as e:
                logging.error(f"An error occurred while preloading the model: {e}")
                raise
        if fix:
            for param in self.parameters():
                param.requires_grad = False
            logging.info("All generator parameters have been frozen. They will not be updated during training.")

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs,
                which consists of each layer output tensors. Only one
                discriminator here, but still make it as List of List for
                consistency.
        """
        if self.choose == "msstft":
            out = self.msstft(x)
        elif self.choose == "msmpmb":
            out = self.msmpmb(x)
        return out
