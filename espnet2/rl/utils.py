"""Small native helpers; architecture implementations stay in their own folders."""

import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch


def resolve_callable(name):
    module, symbol = name.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def sha256(path):
    """Hex SHA256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_export_files(directory, recorded):
    """Require a complete hash list of regular files contained in this export."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir() or not isinstance(recorded, dict) or not recorded:
        raise ValueError("Existing inference export has no recorded files")
    paths = {}
    for relative, expected in recorded.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"Invalid recorded inference export path/hash: {relative}")
        path = root / relative
        if any(
            parent.is_symlink()
            for parent in (path, *path.parents)
            if parent != root and parent.is_relative_to(root)
        ):
            raise ValueError(f"Inference export contains a symlink: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(
                f"Inference export file is outside its directory: {relative}"
            )
        paths[relative] = resolved
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"Inference export contains a symlink: {path.relative_to(root)}"
            )
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if set(paths) != actual:
        raise ValueError(
            "Recorded inference export files differ from its complete file set"
        )
    for relative, path in paths.items():
        if sha256(path) != recorded[relative]:
            raise ValueError(f"Existing inference export file changed: {relative}")


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
