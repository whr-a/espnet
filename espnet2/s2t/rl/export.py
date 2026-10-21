"""Export exact native OWSM weights for the optional SGLang rollout model."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file


def export_owsm(config, checkpoint, output):
    """Miles exporter hook: no vocabulary or trained tensor is transformed."""
    if getattr(config, "rollout_backend", "sglang") == "vllm":
        from espnet2.s2t.rl.vllm_export import export_owsm as export_vllm

        return export_vllm(
            config.train_config, checkpoint, output, model_version=config.model_version
        )
    source_config = Path(config.train_config)
    native = yaml.safe_load(source_config.read_text())
    if (
        native.get("model", "espnet") != "espnet"
        or native.get("decoder", "transformer") != "transformer"
    ):
        raise ValueError("This RL adapter supports autoregressive OWSM, not OWSM-CTC")
    target = Path(output)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite export: {target}")
    target.mkdir(parents=True, exist_ok=True)
    tokens = native["token_list"]
    if isinstance(tokens, str):
        token_path = Path(tokens)
        if not token_path.is_file():
            token_path = source_config.parent / tokens
        tokens = token_path.read_text().splitlines()
    native["token_list"] = tokens
    for key, filename in (("bpemodel", "bpe.model"),):
        original = Path(native[key])
        if not original.is_file():
            original = source_config.parent / original
        shutil.copyfile(original, target / filename)
        native[key] = filename
    if native.get("normalize") == "global_mvn":
        original = Path(native["normalize_conf"]["stats_file"])
        if not original.is_file():
            original = source_config.parent / original
        shutil.copyfile(original, target / "feats_stats.npz")
        native["normalize_conf"]["stats_file"] = "feats_stats.npz"
    native["specaug"] = None
    native["init"] = None
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for wrapper in ("module", "state_dict", "model"):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
    state = {
        key.removeprefix("module."): tensor.detach().cpu().contiguous()
        for key, tensor in state.items()
    }
    hidden = int(native["encoder_conf"]["output_size"])
    decoder = native.get("decoder_conf", {})
    if tuple(state["decoder.output_layer.weight"].shape) != (len(tokens), hidden):
        raise ValueError("Native output layer and vocabulary dimensions disagree")
    # LlamaConfig is a scheduler metadata carrier only. EntryClass below builds
    # ESPnet S2TTask, not a Llama model; native weights are loaded strictly.
    metadata = {
        "model_type": "llama",
        "architectures": ["ESPnetOWSMForCausalLM"],
        "vocab_size": len(tokens),
        "hidden_size": hidden,
        "intermediate_size": int(decoder.get("linear_units", 2048)),
        "num_hidden_layers": int(decoder.get("num_blocks", 6)),
        "num_attention_heads": int(decoder.get("attention_heads", 4)),
        "num_key_value_heads": int(decoder.get("attention_heads", 4)),
        "max_position_embeddings": 2048,
        "bos_token_id": tokens.index(
            native.get("model_conf", {}).get("sym_sos", "<sos>")
        ),
        "eos_token_id": tokens.index(
            native.get("model_conf", {}).get("sym_eos", "<eos>")
        ),
        "pad_token_id": tokens.index("<blank>"),
        "tie_word_embeddings": False,
        "torch_dtype": config.dtype,
        "espnet_config": native,
        "espnet_model_version": config.model_version,
        "espnet_rollout_contract": "rid-sources/native-full-prefix/v2",
        "native_checkpoint_sha256": hashlib.file_digest(
            Path(checkpoint).open("rb"), "sha256"
        ).hexdigest(),
    }
    (target / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (target / "espnet_config.yaml").write_text(yaml.safe_dump(native, sort_keys=False))
    (target / "tokens.txt").write_text("\n".join(tokens) + "\n")
    save_file(state, target / "model.safetensors", metadata={"format": "pt"})
    (target / "export_manifest.json").write_text(
        json.dumps(
            {
                "source_config": str(source_config.resolve()),
                "source_config_sha256": hashlib.sha256(
                    source_config.read_bytes()
                ).hexdigest(),
                "source_checkpoint": str(Path(checkpoint).resolve()),
                "source_checkpoint_sha256": metadata["native_checkpoint_sha256"],
                "native_weights": len(state),
                "vocab_size": len(tokens),
                "backend": "sglang",
                "contract": metadata["espnet_rollout_contract"],
            },
            indent=2,
        )
        + "\n"
    )
    return target
