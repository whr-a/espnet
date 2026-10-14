# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional vLLM OWSM exporter preserving the native vocabulary and tensors."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from vllm.transformers_utils.configs.owsm import OWSMConfig


def export_owsm(config_path, checkpoint_path, output_dir, model_version=None):
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    native = yaml.safe_load(config_path.read_text())
    if native.get("decoder", "transformer") != "transformer":
        raise ValueError(
            "OWSM-CTC and other decoders require a separate rollout adapter"
        )
    if native.get("decoder_conf", {}).get("input_layer", "embed") != "embed":
        raise ValueError("OWSM vLLM requires an embedding-input Transformer decoder")
    if native.get("decoder_conf", {}).get("use_output_layer", True) is False:
        raise ValueError("OWSM vLLM requires decoder.output_layer")
    tokens = native.get("token_list")
    if isinstance(tokens, str):
        token_path = Path(tokens)
        if not token_path.is_file():
            token_path = config_path.parent / tokens
        tokens = token_path.read_text().splitlines()
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("config must contain the exact checkpoint token_list")
    native["token_list"] = tokens
    bpe = Path(native.get("bpemodel") or "bpe.model")
    if not bpe.is_file():
        bpe = config_path.parent / bpe
    if not bpe.is_file():
        raise FileNotFoundError(
            "Set bpemodel in the export config to the downloaded checkpoint bpe.model"
        )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    state = {
        k.removeprefix("module."): v.detach().cpu().contiguous()
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }
    if not any(k.startswith("decoder.") for k in state):
        raise ValueError("Checkpoint has no autoregressive decoder weights")
    expected = (len(tokens), native.get("encoder_conf", {}).get("output_size", 1024))
    if tuple(state["decoder.output_layer.weight"].shape) != expected:
        raise ValueError(
            "Checkpoint decoder dimensions disagree with config/token_list"
        )
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    # These files are for native training/provenance. The inference model uses
    # the frozen JSON config and checkpoint buffers rather than external paths.
    shutil.copyfile(bpe, destination / "bpe.model")
    native["bpemodel"] = "bpe.model"
    native["specaug"] = None
    native["init"] = None
    config = OWSMConfig(
        espnet_config=native,
        vocab_size=len(tokens),
        model_version=model_version,
        architectures=["OWSMForConditionalGeneration"],
    )
    if config.bos_token_id is None or config.eos_token_id is None:
        raise ValueError(
            "Checkpoint token_list lacks configured OWSM start/end symbols"
        )
    config.save_pretrained(destination)
    (destination / "tokens.txt").write_text("\n".join(tokens) + "\n")
    (destination / "espnet_config.yaml").write_text(
        yaml.safe_dump(native, sort_keys=False)
    )
    save_file(state, destination / "model.safetensors", metadata={"format": "pt"})
    (destination / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": config.bos_token_id,
                "eos_token_id": config.eos_token_id,
                "pad_token_id": config.pad_token_id,
                "decoder_start_token_id": config.bos_token_id,
            },
            indent=2,
        )
        + "\n"
    )
    (destination / "export_manifest.json").write_text(
        json.dumps(
            {
                "model_version": model_version,
                "source_config": str(config_path.resolve()),
                "source_checkpoint": str(checkpoint_path.resolve()),
                "source_config_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                "source_checkpoint_sha256": hashlib.file_digest(
                    checkpoint_path.open("rb"), "sha256"
                ).hexdigest(),
                "vocab_size": len(tokens),
                "weights": len(state),
                "tensor_parallel_size": 1,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version")
    args = parser.parse_args()
    print(export_owsm(args.config, args.checkpoint, args.output, args.model_version))


if __name__ == "__main__":
    main()
