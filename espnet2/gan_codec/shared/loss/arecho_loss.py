#!/usr/bin/env python3

"""Inference script for ESPnet Universa model."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch.nn as nn

import numpy as np
import torch
from typeguard import typechecked

from espnet2.tasks.universa import UniversaTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from typing import List

class UniversaInference:
    """Inference class for ESPnet Universa model."""

    @typechecked
    def __init__(
        self,
        train_config: Union[Path, str, None] = None,
        model_file: Union[Path, str, None] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
        beam_size: int = 1,
        skip_meta_label_score: bool = False,
        save_token_seq: bool = False,
        use_fixed_order: bool = False,
        fixed_metric_name_order: str = "",
    ):
        """Initialize UniversaInference class."""

        # setup model
        model, train_args = UniversaTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.universa = model.universa
        self.frontend = model.frontend
        self.preprocess_fn = UniversaTask.build_preprocess_fn(train_args, False)
        self.metric_tokenizer = self.preprocess_fn.metric_tokenizer
        self.seed = seed
        self.always_fix_seed = always_fix_seed

        self.beam_size = beam_size
        self.skip_meta_label_score = skip_meta_label_score
        self.save_token_seq = save_token_seq
        # TODO(jiatong): to set fixed order cases
        self.use_fixed_order = use_fixed_order
        self.fixed_metric_name_order = fixed_metric_name_order

        if self.model.universa.sequential_metrics:
            metric_list = list(self.model.universa.metric2id.keys())
            self.model.universa.set_inference(
                beam_size=beam_size,
                metric_list=metric_list,
                skip_meta_label_score=skip_meta_label_score,
                save_token_seq=save_token_seq,
            )

        logging.info(f"Frontend: {model.frontend}")
        logging.info(f"Universa: {model.universa}")

    @torch.no_grad()
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
        if self.always_fix_seed:
            set_all_random_seed(self.seed)

        output_dict = self.model.inference(**batch, **kwargs)

        output_dict.pop("use_tokenizer_metrics")
        output_dict.pop("sequential_metrics")
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

class ArechoLoss(nn.Module):
    def __init__(self,
                 target_metrics: List[str] = [],
                 loss_type: str = "mae",
                 model_tag: Optional[str] = None,
                 train_config: Optional[str] = None,
                 model_file: Optional[str] = None,
                 dtype: str = "float32",
                 seed: int = 777,
                 always_fix_seed: bool = False,
                 beam_size: int = 1,
                 skip_meta_label_score: bool = False,
                 save_token_seq: bool = False,
                 use_fixed_order: bool = False,
                 fixed_metric_name_order: str = "",
                 device: str = "cpu",
    ):
        super().__init__()
        self.universa_inference = UniversaInference.from_pretrained(
            model_tag=model_tag,
            train_config=train_config,
            model_file=model_file,
            dtype=dtype,
            seed=seed,
            always_fix_seed=always_fix_seed,
            beam_size=beam_size,
            skip_meta_label_score=skip_meta_label_score,
            save_token_seq=save_token_seq,
            use_fixed_order=use_fixed_order,
            fixed_metric_name_order=fixed_metric_name_order,
            device=device,
        )
        self.target_metrics = target_metrics
        self.loss_type = loss_type
    def forward(self, audio, ref_audio):
        audio_len = audio.shape[1]
        ref_audio_len = ref_audio.shape[1]
        results = self.universa_inference(audio, audio_len)
        ref_results = self.universa_inference(ref_audio, ref_audio_len)
        loss = 0
        for target_metric in self.target_metrics:
            if self.loss_type == "mae":
                loss += nn.functional.l1_loss(results[target_metric], ref_results[target_metric])
            elif self.loss_type == "mse":
                loss += nn.functional.mse_loss(results[target_metric], ref_results[target_metric])
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
        loss = loss / len(self.target_metrics)
        return loss