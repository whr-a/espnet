"""Small native helpers; architecture implementations stay in their own folders."""

import importlib

import numpy as np
import torch


def resolve_callable(name):
    module, symbol = name.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def build_policy(config, device):
    return resolve_callable(config.adapter_factory)(config, device)


def object_array(values):
    # np.array(list_of_equal_length_lists) creates an unwanted extra batch axis.
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def get_device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_device():
    return torch.cuda


def get_nccl_backend():
    return "nccl"
