"""Small native helpers; architecture implementations stay in their own folders."""

import importlib
from pathlib import Path

import numpy as np
import torch


def resolve_callable(name):
    module, symbol = name.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def build_policy(config, device):
    return resolve_callable(config.adapter_factory)(config, device)


def object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def get_device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_device():
    return torch.cuda


def get_nccl_backend():
    return "nccl"


def load_audio(source, sample_rate, window_samples=None):
    import soundfile as sf

    if "audio_array" in source:
        waveform = np.asarray(source["audio_array"], dtype=np.float32)
        rate = int(source.get("sample_rate", sample_rate))
    else:
        waveform, rate = sf.read(Path(source["audio"]), dtype="float32")
    if rate != sample_rate:
        raise ValueError("Manifest audio must be prepared at the model sample rate")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Expected nonempty finite audio")
    if window_samples is not None:
        if len(waveform) > window_samples:
            raise ValueError("Segment audio offline to fit the model's context window")
        waveform = np.pad(waveform, (0, window_samples - len(waveform)))
    return waveform


def owsm_symbol_id(token_list, value, *, language=False):
    from espnet2.s2t.rl.policy import owsm_symbol_id as implementation

    return implementation(token_list, value, language=language)
