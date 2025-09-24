#!/usr/bin/env python3

"""Inference script for ESPnet Universa model."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch.nn as nn

import numpy as np
import torch
import torchaudio.transforms as T
from typeguard import typechecked

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.spk.pooling.mean_pooling import MeanPooling
from espnet2.spk.projector.xvector_projector import XvectorProjector
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.universa.abs_universa import AbsUniversa
from espnet2.universa.base.loss import masked_l1_loss, masked_mse_loss
from espnet2.universa.metric_tokenizer.metric_tokenizer import MetricTokenizer
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.tasks.universa import UniversaTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from typing import List
import pdb

class UniversaInference:
    """Inference class for ESPnet Universa model."""

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        device: str = "cuda",
    ):
        """Initialize UniversaInference class."""

        # setup model
        model, train_args = UniversaTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, "float32"))#.eval()
        self.device = device
        self.dtype = "float32"
        self.train_args = train_args
        self.model = model
        self.universa = model.universa
        self.frontend = model.frontend
        self.preprocess_fn = UniversaTask.build_preprocess_fn(train_args, False)
        self.metric_tokenizer = self.preprocess_fn.metric_tokenizer

        metric_list = list(self.model.universa.metric2id.keys())

        logging.info(f"Frontend: {model.frontend}")
        logging.info(f"Universa: {model.universa}")

    @typechecked
    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        audio_lengths: Union[np.ndarray, torch.Tensor] = None,
        ref_audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_audio_lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_text: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        ref_text_lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        "Run universa."

        # check the input type
        if self.model.use_ref_audio and ref_audio is None:
            logging.warning("Universa model pretrained with ref_audio is used.")
        if self.model.use_ref_text and ref_text is None:
            logging.warning("Universa model pretrained with ref_text is used.")
        if not self.model.use_ref_audio and ref_audio is not None:
            logging.warning("Universa model not pretrained with ref_audio is used.")
        if not self.model.use_ref_text and ref_text is not None:
            logging.warning("Universa model not pretrained with ref_text is used.")

        # prepare batch
        batch = dict(audio=audio, audio_lengths=audio_lengths)
        if ref_audio is not None:
            batch.update(ref_audio=ref_audio, ref_audio_lengths=ref_audio_lengths)
        if ref_text is not None:
            if isinstance(ref_text, str):
                ref_text = self.preprocess_fn("<dummy>", dict(ref_text=ref_text))[
                    "ref_text"
                ]
                ref_text = np.expand_dims(ref_text, axis=0)
                ref_text_lengths = torch.tensor([len(ref_text)])
            batch.update(ref_text=ref_text, ref_text_lengths=ref_text_lengths)
        batch = to_device(batch, device=self.device)

        # inference

        with torch.no_grad():
            output_dict = self.model.inference(**batch, **kwargs)

        output_dict.pop("use_tokenizer_metrics", None)
        output_dict.pop("sequential_metrics", None)
        return output_dict

    @property
    def use_ref_audio(self):
        return self.model.use_ref_audio

    @property
    def use_ref_text(self):
        return self.model.use_ref_text

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build UniversaInference from pretrained model."""
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))
        return UniversaInference(**kwargs)

class UniversaLoss(nn.Module):
    def __init__(self,
                 target_metrics: List[str] = [],
                 loss_type: str = "score_direct",
                 model_tag: Optional[str] = None,
                 universa_train_config: Optional[str] = None,
                 model_file: Optional[str] = None,
                 device: str = "cuda",
    ):
        super().__init__()
        self.universa_inference = UniversaInference.from_pretrained(
            model_tag=model_tag,
            train_config=universa_train_config,
            model_file=model_file,
            device=device,
        )
        print(self.universa_inference)
        self.target_metrics = target_metrics
        self.device = device
        self.loss_type = loss_type
        # Initialize resampler for downsampling from 24kHz to 16kHz
        self.resampler = T.Resample(orig_freq=24000, new_freq=16000)
    def forward(self, audio, ref_audio):
        batch_size, _, audio_length = audio.shape
        
        # Downsample audio from 24kHz to 16kHz
        audio_downsampled = self.resampler(audio)
        # ref_audio_downsampled = self.resampler(ref_audio)
        audio_length_downsampled = audio_downsampled.shape[2]
        audio_len = torch.tensor([audio_length_downsampled]*batch_size)
        empty_audio = torch.zeros((batch_size, 1, 8000))
        empty_audio_length = torch.tensor([8000]*batch_size)
        loss = 0
        results = self.universa_inference(audio_downsampled.squeeze(1), audio_len, empty_audio.squeeze(1), empty_audio_length)
        for target_metric in self.target_metrics:
            if self.loss_type == "score_direct":
                if target_metric == "scoreq_ref":
                    loss += results[target_metric]
                else:
                    loss += results[target_metric] * (-1)
        loss = sum(loss / len(self.target_metrics) / batch_size)

        return loss