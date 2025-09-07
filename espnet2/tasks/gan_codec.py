# Copyright 2024 Jiatong Shi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based neural codec task."""

import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec  # noqa
from espnet2.gan_codec.lrac.lrac import LRACConvBaseline
from espnet2.gan_codec.dac.dac import DAC
from espnet2.gan_codec.encodec.encodec import Encodec
from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
from espnet2.gan_codec.funcodec.funcodec import FunCodec
from espnet2.gan_codec.soundstream.soundstream import SoundStream
from espnet2.tasks.abs_task import AbsTask, optim_classes
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.gan_trainer import GANTrainer
from espnet2.train.preprocessor import CommonPreprocessor, EnhPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none  # noqa

codec_choices = ClassChoices(
    "codec",
    classes=dict(
        soundstream=SoundStream,
        encodec=Encodec,
        dac=DAC,
        funcodec=FunCodec,
        lrac=LRACConvBaseline
    ),
    default="soundstream",
)


class GANCodecTask(AbsTask):
    """GAN-based neural codec task."""

    # GAN requires two optimizers
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [
        # --codec and --codec_conf
        codec_choices,
    ]

    # Use GANTrainer instead of Trainer
    trainer = GANTrainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")  # noqa

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetGANCodecModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--apply_enhancement",
            type=str2bool,
            default=False,
            help="Apply enhancement to input data or not",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )

        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--validation_rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=0.5,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--validation_noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=0.8,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="-5_30",
            help="The range of signal-to-noise ratio (SNR) level in decibel.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        
        group.add_argument(
            "--speech_volume_normalize",
            type=str_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value or range. "
            "e.g. --speech_volume_normalize 1.0 scales it to 1.0.\n"
            "--speech_volume_normalize 0.5_1.0 scales it to a random number in "
            "the range [0.5, 1.0)",
        )

        group.add_argument(
            "--use_reverberant_ref",
            type=str2bool,
            default=False,
            help="Whether to use reverberant speech references "
            "instead of anechoic ones",
        )
        group.add_argument(
            "--num_spk",
            type=int,
            default=1,
            help="Number of speakers in the input signal.",
        )
        group.add_argument(
            "--num_noise_type",
            type=int,
            default=1,
            help="Number of noise types.",
        )
        group.add_argument(
            "--sample_rate",
            type=int,
            default=24000,
            help="Sampling rate of the data (in Hz).",
        )
        group.add_argument(
            "--force_single_channel",
            type=str2bool,
            default=True,
            help="Whether to force all data to be single-channel.",
        )
        group.add_argument(
            "--channel_reordering",
            type=str2bool,
            default=False,
            help="Whether to randomly reorder the channels of the "
            "multi-channel signals.",
        )
        group.add_argument(
            "--categories",
            nargs="+",
            default=[],
            type=str,
            help="The set of all possible categories in the dataset. Used to add the "
            "category information to each sample",
        )
        group.add_argument(
            "--speech_segment",
            type=int_or_none,
            default=None,
            help="Truncate the audios to the specified length (in samples) if not None",
        )
        group.add_argument(
            "--avoid_allzero_segment",
            type=str2bool,
            default=True,
            help="Only used when --speech_segment is specified. If True, make sure "
            "all truncated segments are not all-zero",
        )
        group.add_argument(
            "--flexible_numspk",
            type=str2bool,
            default=False,
            help="Whether to load variable numbers of speakers in each sample. "
            "In this case, only the first-speaker files such as 'spk1.scp' and "
            "'dereverb1.scp' are used, which are expected to have multiple columns. "
            "Other numbered files such as 'spk2.scp' and 'dereverb2.scp' are ignored.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if args.use_preprocessor:
            if getattr(args, 'apply_enhancement', False):
                kwargs = dict(
                    rir_scp=getattr(args, "rir_scp", None),
                    rir_apply_prob=getattr(args, "rir_apply_prob", 0.5),
                    noise_scp=getattr(args, "noise_scp", None),
                    noise_apply_prob=getattr(args, "noise_apply_prob", 0.8),
                    noise_db_range=getattr(args, "noise_db_range", "-5_30"),
                    short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                    speech_volume_normalize=getattr(
                        args, "speech_volume_normalize", None
                    ),
                    use_reverberant_ref=getattr(args, "use_reverberant_ref", False),
                    num_spk=getattr(args, "num_spk", 1),
                    num_noise_type=getattr(args, "num_noise_type", 1),
                    sample_rate=getattr(args, "sample_rate", 24000),
                    force_single_channel=getattr(args, "force_single_channel", True),
                    channel_reordering=getattr(args, "channel_reordering", False),
                    categories=getattr(args, "categories", None),
                    speech_segment=getattr(args, "speech_segment", None),
                    avoid_allzero_segment=getattr(args, "avoid_allzero_segment", True),
                    flexible_numspk=getattr(args, "flexible_numspk", False),
                )
                
                if not train:
                    # We are in validation mode
                    # Check if we need to also augment the validation data
                    # if so go back to train mode.
                    # In inference mode during test time, make sure to remove the 
                    # validation_rir_scp and validation_noise_scp keys 
                    # before creating the preprocessor
                    validation_rir_scp = getattr(args, "validation_rir_scp", None)
                    validation_noise_scp = getattr(args, "validation_noise_scp", None)
                    if validation_rir_scp or validation_noise_scp:
                        # Validation reverb and/or noise provided
                        # We are asked to augment the data also during validation
                        kwargs['rir_scp'] = getattr(args, "validation_rir_scp", kwargs['rir_scp'])
                        kwargs['noise_scp'] = getattr(args, "validation_noise_scp", kwargs['noise_scp'])
                        train=True

                retval = EnhPreprocessor(
                    train=train,
                    speech_name="audio",
                    **kwargs
                )
            else:
                # additional check for chunk iterator, to use short utterance in training
                if args.iterator_type == "chunk":
                    min_sample_size = args.chunk_length
                else:
                    min_sample_size = -1

                retval = CommonPreprocessor(
                    train=train,
                    token_type=None,  # disable the text process
                    speech_name="audio",
                    min_sample_size=min_sample_size,
                    audio_pad_value=0.0,
                    force_single_channel=True,  # NOTE(jiatong): single channel only now
                )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("audio",)
        else:
            # Inference mode
            retval = ("audio",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("speech_ref1",)
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetGANCodecModel:

        # 1. Codec
        codec_class = codec_choices.get_class(args.codec)
        codec = codec_class(**args.codec_conf)

        # 2. Build model
        model = ESPnetGANCodecModel(
            codec=codec,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: ESPnetGANCodecModel,
    ) -> List[torch.optim.Optimizer]:
        # check
        assert hasattr(model.codec, "generator")
        assert hasattr(model.codec, "discriminator")

        # define generator optimizer
        optim_g_class = optim_classes.get(args.optim)
        if optim_g_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_g = fairscale.optim.oss.OSS(
                params=model.codec.generator.parameters(),
                optim=optim_g_class,
                **args.optim_conf,
            )
        else:
            optim_g = optim_g_class(
                model.codec.generator.parameters(),
                **args.optim_conf,
            )
        optimizers = [optim_g]

        # define discriminator optimizer
        optim_d_class = optim_classes.get(args.optim2)
        if optim_d_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        if args.sharded_ddp:
            try:
                import fairscale
            except ImportError:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim_d = fairscale.optim.oss.OSS(
                params=model.codec.discriminator.parameters(),
                optim=optim_d_class,
                **args.optim2_conf,
            )
        else:
            optim_d = optim_d_class(
                model.codec.discriminator.parameters(),
                **args.optim2_conf,
            )
        optimizers += [optim_d]

        return optimizers
