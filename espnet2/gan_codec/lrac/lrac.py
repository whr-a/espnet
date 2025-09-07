# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

from typing import Any, Dict
import math
import numpy as np

from espnet2.gan_codec.soundstream.soundstream import SoundStreamGenerator
from espnet2.gan_codec.encodec.encodec import Encodec
from espnet2.gan_codec.shared.encoder.generic_seanet import GenericSEANetEncoder
from espnet2.gan_codec.shared.decoder.generic_seanet import GenericSEANetDecoder

class LRACGenerator(SoundStreamGenerator):
    def __init__(self, 
                 sample_rate: int = 24000,
                 encoder_params: Dict[str, Any] = None,
                 decoder_params: Dict[str, Any] = None,
                 quantizer_params: Dict[str, Any] = None):
        
        encoder_params = encoder_params or self.get_default_init_params()["encoder_params"]
        decoder_params = decoder_params or self.get_default_init_params()["decoder_params"]
        quantizer_params = quantizer_params or self.get_default_init_params()["quantizer_params"]

        super().__init__(
            sample_rate=sample_rate,
            hidden_dim=encoder_params['output_dimension'],
            **quantizer_params)
        
        self.frame_rate = math.ceil(sample_rate / np.prod(encoder_params["strides"]))

        # Overwrite encoder and decoder
        self.encoder = GenericSEANetEncoder(**encoder_params)
        self.decoder = GenericSEANetDecoder(**decoder_params)
    
    @staticmethod
    def get_default_init_params():
        init_params= {
            "encoder_params": GenericSEANetEncoder.get_default_init_params(),
            "decoder_params": GenericSEANetDecoder.get_default_init_params(),
            "quantizer_params": {
                "quantizer_codebook_dim": 128,
                "quantizer_n_q": 6,
                "quantizer_bins": 1024,
                "quantizer_decay": 0.99,
                "quantizer_kmeans_init": True,
                "quantizer_kmeans_iters": 50,
                "quantizer_threshold_ema_dead_code": 2,
                "quantizer_target_bandwidth": [1, 6]
            }
        }
        return init_params

    

class LRACConvBaseline(Encodec):
    """
    The convolutional codec model for LRAC challenge track1:
    https://crowdsourcing.cisco.com/lrac-challenge/2025/#track-1--transparency-codecs

    This codec is very similar to Encodec with more control on hyperparameters. 
    It also supports semantic loss which is a deep feature matching loss
    computed on the intermediate layer representation space of a pretrained
    self-supervised speech model such as WAVLM_LARGE.

    """
    def __init__(self,
                 apply_enhancement: bool = False,
                 sampling_rate: int = 24000,
                 encoder_params: Dict[str, Any] = None,
                 decoder_params: Dict[str, Any] = None,
                 quantizer_params: Dict[str, Any] = None,
                 discriminator_params: Dict[str, Any] = None,
                 generator_adv_loss_params: Dict[str, Any] = None,
                 discriminator_adv_loss_params: Dict[str, Any] = None,
                 use_feat_match_loss = True,
                 feat_match_loss_params: Dict[str, Any] = None,
                 use_mel_loss = True,
                 mel_loss_params: Dict[str, Any] = None,
                 use_semantic_loss: bool = True,
                 semantic_loss_params: Dict[str, Any] = None,
                 use_dual_decoder = True, 
                 lambda_quantization: float = 1, 
                 lambda_reconstruct: float = 1, 
                 lambda_commit: float = 0.25, 
                 lambda_adv: float = 1, 
                 lambda_feat_match: float = 1, 
                 lambda_mel: float = 1,
                 lambda_semantic: float = 0.1,
                 cache_generator_outputs: bool = True, 
                 use_loss_balancer: bool = False, 
                 balance_ema_decay: float = 0.99):
        
        encoder_params = encoder_params or self.get_default_init_params()['encoder_params']
        decoder_params = decoder_params or self.get_default_init_params()['decoder_params']
        quantizer_params = quantizer_params or self.get_default_init_params()['quantizer_params']
        discriminator_params = discriminator_params or self.get_default_init_params()['discriminator_params']
        generator_adv_loss_params = generator_adv_loss_params or self.get_default_init_params()['generator_adv_loss_params']
        discriminator_adv_loss_params = discriminator_adv_loss_params or self.get_default_init_params['discriminator_adv_loss_params']
        feat_match_loss_params = feat_match_loss_params or self.get_default_init_params()['feat_match_loss_params']
        mel_loss_params = mel_loss_params or self.get_default_init_params()['mel_loss_params']
        semantic_loss_params = semantic_loss_params or self.get_default_init_params()['semantic_loss_params']

        super().__init__(
            apply_enhancement=apply_enhancement,
            sampling_rate=sampling_rate, 
            discriminator_params=discriminator_params,
            generator_adv_loss_params=generator_adv_loss_params, 
            discriminator_adv_loss_params=discriminator_adv_loss_params, 
            use_feat_match_loss=use_feat_match_loss, 
            feat_match_loss_params=feat_match_loss_params, 
            use_mel_loss=use_mel_loss, 
            mel_loss_params=mel_loss_params,
            use_semantic_loss=use_semantic_loss,
            semantic_loss_params=semantic_loss_params,
            use_dual_decoder=use_dual_decoder, 
            lambda_quantization=lambda_quantization, 
            lambda_reconstruct=lambda_reconstruct, 
            lambda_commit=lambda_commit, 
            lambda_adv=lambda_adv, 
            lambda_feat_match=lambda_feat_match, 
            lambda_mel=lambda_mel,
            lambda_semantic=lambda_semantic,
            cache_generator_outputs=cache_generator_outputs, 
            use_loss_balancer=use_loss_balancer, 
            balance_ema_decay=balance_ema_decay)

        # Overwrite the generator
        self.generator = LRACGenerator(
            sample_rate=sampling_rate, 
            encoder_params=encoder_params, decoder_params=decoder_params,
            quantizer_params=quantizer_params
            )
    
    @staticmethod
    def get_default_init_params():
        init_params = {
            "encoder_params": LRACGenerator.get_default_init_params()['encoder_params'],
            "decoder_params": LRACGenerator.get_default_init_params()['decoder_params'],
            "quantizer_params": LRACGenerator.get_default_init_params()['quantizer_params'],
            "discriminator_params": {
                "msstft_discriminator_params": {
                    "filters": 16,
                    "in_channels": 1,
                    "out_channels": 1,
                    "sep_channels": False,
                    "norm": "weight_norm",
                    "n_ffts": [128, 256, 512, 1024, 2048],
                    "hop_lengths": [32, 64, 128, 256, 512],
                    "win_lengths": [128, 256, 512, 1024, 2048],
                    "activation": "LeakyReLU",
                    "activation_params": {"negative_slope": 0.3}
                }
            },
            "generator_adv_loss_params": {
                "average_by_discriminators": True,
                "loss_type": "mse",
            },
            "discriminator_adv_loss_params": {
                "average_by_discriminators": True,
                "loss_type": "mse",
            },
            "use_feat_match_loss": True,
            "feat_match_loss_params": {
                "average_by_discriminators": True,
                "average_by_layers": False,
                "include_final_outputs": True,
            },
            "use_mel_loss": True,
            "mel_loss_params": {
                "fs": 24000,
                "range_start": 6,
                "range_end": 11,
                "window": "hann",
                "n_mels": [10, 20, 40, 80, 160, 320],
                "fmin": 0,
                "fmax": None,
                "log_base": None,
            },
            "use_semantic_loss": True,
            "semantic_loss_params": {
                "sample_rate": 24000,
                "model_name": "WAVLM_LARGE",
                "feature_ids": None 
            },
            "use_dual_decoder": True,
            "lambda_quantization": 1.0,
            "lambda_reconstruct": 1.0,
            "lambda_commit": 1.0,
            "lambda_adv": 1.0,
            "lambda_feat_match": 2.0,
            "lambda_mel": 45.0,
            "lambda_semantic": 0.1,
            "cache_generator_outputs": False,
            "use_loss_balancer": False,
            "balance_ema_decay": 0.99
        }
        return init_params
