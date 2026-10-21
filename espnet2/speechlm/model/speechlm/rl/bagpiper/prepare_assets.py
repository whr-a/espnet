# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Build Bagpiper RL metadata from pinned public configs and tokenizers on CPU.

No model weights are downloaded or loaded. The native checkpoint supplies the
text, audio encoder and codec tensors; these files describe their geometry and
the native vocabulary. Original native preprocessing remains the prompt API.
"""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml

SOURCES = {
    "text": (
        "Qwen/Qwen3-8B-Base",
        "b86f722ec0768e8849ddeb2d8ce3a518bf4fd504",
        (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ),
    ),
    "audio": (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "26291f793822fb6be9555850f06dfe95f2d7e695",
        ("config.json", "preprocessor_config.json"),
    ),
    "codec": (
        "hf-audio/xcodec-hubert-general",
        "240735a1ade315cb9b22604fe898e7df03d67378",
        ("config.json",),
    ),
}
SPECIALS = [
    f"<|{name}|>"
    for name in (
        "pad",
        "bos",
        "eos",
        "eot",
        "system",
        "user",
        "assistant",
        "text",
        "audio",
        "image",
        "video",
        "toolcall",
    )
]
SPECIALS += [f"<|unused_{index}|>" for index in range(12, 256)]


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata_source(name, local=None):
    if local is not None:
        return Path(local).resolve(strict=True)
    from huggingface_hub import snapshot_download

    repo, revision, files = SOURCES[name]
    return Path(snapshot_download(repo, revision=revision, allow_patterns=list(files)))


def shifted_tokenizer(text, config):
    """Merge config-only added IDs before shifting, preserving BPE behavior."""
    result = copy.deepcopy(text)
    require(
        result["model"]["type"] == "BPE", "Expected the released Qwen BPE tokenizer"
    )
    vocab = dict(result["model"]["vocab"])
    added = {
        item["id"]: {key: value for key, value in item.items() if key != "id"}
        for item in result.get("added_tokens", [])
    }
    for key, item in config.get("added_tokens_decoder", {}).items():
        index = int(key)
        require(
            index not in added or added[index]["content"] == item["content"],
            "Conflicting added-token identity",
        )
        added[index] = dict(item)
    inverse = {value: token for token, value in vocab.items()}
    for index, item in added.items():
        token = item["content"]
        require(
            (token not in vocab or vocab[token] == index)
            and (index not in inverse or inverse[index] == token),
            "Conflicting tokenizer IDs",
        )
        vocab[token], inverse[index] = index, token
    require(
        len(vocab) == 151669 and set(vocab.values()) == set(range(151669)),
        "Expected 151669 actual Qwen tokens; reserve its configured 151936 text slots",
    )
    for index, token in enumerate(
        ("<tool_response>", "</tool_response>", "<think>", "</think>"), 151665
    ):
        require(
            vocab.get(token) == index, "Missing or shifted config-only Qwen tail token"
        )
    native = {token: index for index, token in enumerate(SPECIALS)}
    require(not (set(native) & set(vocab)), "Native and text token symbols overlap")
    native.update({token: index + 256 for token, index in vocab.items()})
    reserved = {f"<unused_{index}>": index + 256 for index in range(151669, 151936)}
    codec = {
        f"<codec_layer{stream}_{code}>": 152192 + stream * 1025 + code + 1
        for stream in range(8)
        for code in range(1024)
    }
    codec.update(
        {f"<codec_layer{stream}_pad>": 152192 + stream * 1025 for stream in range(8)}
    )
    native.update(reserved)
    native.update(codec)
    require(
        len(native) == 160392 and set(native.values()) == set(range(160392)),
        "Expanded vocabulary is not contiguous",
    )
    defaults = {
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    entries = [
        {"id": index, "content": token, **defaults}
        for token, index in {
            **{token: index for index, token in enumerate(SPECIALS)},
            **reserved,
            **codec,
        }.items()
    ]
    entries += [
        {"id": index + 256, **defaults, **item} for index, item in added.items()
    ]
    result["model"]["vocab"] = native
    result["added_tokens"] = sorted(entries, key=lambda item: item["id"])
    result["padding"], result["truncation"] = None, None
    decoder = {
        str(item["id"]): {key: value for key, value in item.items() if key != "id"}
        for item in result["added_tokens"]
    }
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 32768,
        # Preserve the effective native Qwen frontend under generic fast loading.
        "fix_mistral_regex": False,
        "added_tokens_decoder": decoder,
    }
    return result, tokenizer_config


def prepare(train_config, output, text, audio, codec):
    """Create new local metadata and a path-resolved copy of the released YAML."""
    train_config, output = Path(train_config), Path(output).resolve()
    recipe = yaml.safe_load(train_config.read_text())
    modalities = recipe.get("multimodal_io", {})
    require(
        recipe.get("job_type") == "speechlm"
        and list(modalities) == ["text", "discrete_audio", "continuous_audio"],
        "Expected the released Bagpiper native modality order: "
        "text, discrete_audio, continuous_audio",
    )
    require(
        recipe["model"]["model_choice"] == "parallel"
        and recipe["model"]["model_hf_tag"] == SOURCES["text"][0]
        and modalities["text"]["tokenizer_name"] == SOURCES["text"][0],
        "Expected the released Qwen3-8B-Base backbone/tokenizer",
    )
    discrete = modalities["discrete_audio"]
    require(
        discrete.get("codec_choice") == "Xcodec"
        and discrete.get("codec_hf_model_tag") == SOURCES["codec"][0]
        and discrete.get("ssl_choice") is None
        and discrete.get("delay_interleave") is True
        and discrete.get("codec_max_token_per_frame", 8) == 8
        and discrete.get("stream_weights") == [0.125] * 8,
        "Expected the released eight-stream Xcodec layout",
    )
    require(
        modalities["continuous_audio"]["encoder_hf_model_tag"] == SOURCES["audio"][0],
        "Unexpected audio encoder",
    )
    text_config = read(text / "config.json")
    for key, value in {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000,
    }.items():
        require(text_config.get(key) == value, f"Unexpected Qwen text geometry: {key}")
    audio_config = read(audio / "config.json")["thinker_config"]["audio_config"]
    for key, value in {
        "num_mel_bins": 128,
        "encoder_layers": 32,
        "encoder_attention_heads": 20,
        "d_model": 1280,
        "output_dim": 2048,
        "n_window": 50,
        "n_window_infer": 800,
    }.items():
        require(
            audio_config.get(key) == value, f"Unexpected Qwen audio geometry: {key}"
        )
    processor = read(audio / "preprocessor_config.json")
    require(
        all(
            processor.get(key) == value
            for key, value in {
                "sampling_rate": 16000,
                "feature_size": 128,
                "hop_length": 160,
                "n_fft": 400,
            }.items()
        ),
        "Unexpected audio frontend geometry",
    )
    codec_config = read(codec / "config.json")
    require(
        codec_config.get("model_type") == "xcodec"
        and codec_config.get("codebook_size") == 1024
        and codec_config.get("sample_rate") == 16000
        and codec_config.get("acoustic_model_config", {}).get("hop_length") == 320,
        "Unexpected Xcodec geometry",
    )
    from transformers import AutoTokenizer, __version__

    # Match HuggingFaceTextIO's actual tokenizer, including config-only tokens
    # and the installed Qwen tokenizer's frontend defaults. No model is built.
    native = AutoTokenizer.from_pretrained(text, use_fast=True, local_files_only=True)
    tokenizer, tokenizer_config = shifted_tokenizer(
        json.loads(native.backend_tokenizer.to_str()),
        read(text / "tokenizer_config.json"),
    )
    metadata = {
        "architectures": ["BagpiperForConditionalGeneration"],
        "model_type": "bagpiper",
        "text_config": {**text_config, "vocab_size": 160392},
        "audio_config": audio_config,
        "num_stream": 8,
        "adaptor_input_dim": 2048,
        "vocab_size": 160392,
        "hidden_size": 4096,
        "text_token_offset": 256,
        "text_token_end": 152192,
        "codec_base_offset": 152192,
        "codec_layer_size": 1025,
        "xcodec_hf_model_tag": SOURCES["codec"][0],
        "xcodec_sample_rate": 16000,
        **{
            f"{name}_token_id": index
            for index, name in enumerate(
                (
                    "pad",
                    "bos",
                    "eos",
                    "eot",
                    "system",
                    "user",
                    "assistant",
                    "text",
                    "audio",
                    "image",
                    "video",
                    "toolcall",
                )
            )
        },
        "audio_placeholder_token_id": 8,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, source in (("text", text), ("audio", audio), ("codec", codec)):
        folder = output / name
        folder.mkdir()
        for filename in SOURCES[name][2]:
            if (source / filename).is_file():
                shutil.copyfile(source / filename, folder / filename)
    assets = output / "reference_assets"
    assets.mkdir()

    def write(name, value):
        (assets / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )

    write("config.json", metadata)
    write("tokenizer.json", tokenizer)
    write("tokenizer_config.json", tokenizer_config)
    write(
        "special_tokens_map.json",
        {
            name: tokenizer_config[name]
            for name in ("bos_token", "eos_token", "pad_token")
        },
    )
    write(
        "generation_config.json",
        {
            name: metadata[name]
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
    )
    modalities["text"]["tokenizer_name"] = str(output / "text")
    recipe["model"]["model_hf_tag"] = str(output / "text")
    (output / "native.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    options = {
        "reference_assets": str(assets),
        "audio_config_path": str(output / "audio"),
        "audio_processor_path": str(output / "audio"),
        "codec_config_path": str(output / "codec"),
    }
    receipt = {
        "schema": "espnet-bagpiper-metadata-v1",
        "model_weights_downloaded_or_loaded": False,
        "transformers_version": __version__,
        "train_config": str(output / "native.yaml"),
        "adapter_options": options,
        "upstream_sources": {
            name: {"repo": repo, "revision": revision, "files": list(files)}
            for name, (repo, revision, files) in SOURCES.items()
        },
        "input_directories": {
            "text": str(text),
            "audio": str(audio),
            "codec": str(codec),
        },
        "local_input_note": (
            "Explicit local directories are caller-supplied metadata; upstream "
            "revision identity is not independently attested for local overrides."
        ),
        "sha256": {
            path.relative_to(output)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "asset_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name in SOURCES:
        parser.add_argument(
            f"--{name}-assets",
            type=Path,
            help="Use local metadata instead of the pinned public download",
        )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("The output directory must not exist")
    paths = {
        name: metadata_source(name, getattr(args, f"{name}_assets")) for name in SOURCES
    }
    receipt = prepare(args.train_config, args.output_dir, **paths)
    print(
        json.dumps(
            {
                "train_config": receipt["train_config"],
                "adapter_options": receipt["adapter_options"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
