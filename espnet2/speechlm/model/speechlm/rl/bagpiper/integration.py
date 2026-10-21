# Copyright 2025 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Concrete checkpoint conversion and requests for supported SpeechLM families."""

import copy
import functools
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path

from espnet2.rl.data import prepared_sha256


def prepared_native_metadata(source):
    """Only native conditioning crosses from the inference worker to learner.

    Feature arrays stay numpy arrays, hashed by their bytes; the rollout
    process writes each to one binary sidecar.
    """
    fields = (
        "native_model_inputs",
        "prompt_codec_mask",
        "sampling_extra_args",
        "mode",
        "native_preparation",
    )
    metadata = {name: source[name] for name in fields if name in source}
    return {
        "source_id": source["id"],
        "data": metadata,
        "sha256": prepared_sha256(source["id"], metadata),
    }


def _speechlm_waveform(source, target_rate):
    """Follow native SpeechLM first-channel and torchaudio resampling rules."""
    import numpy as np
    import torch
    import torchaudio

    if "audio_array" in source:
        audio = np.asarray(source["audio_array"], dtype=np.float32)
        rate = int(source.get("sample_rate", target_rate))
        if audio.ndim == 2 and source.get("channel_first", False):
            audio = audio.T
    else:
        import soundfile

        audio, rate = soundfile.read(source["audio"], dtype="float32", always_2d=True)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if audio.ndim != 1 or not audio.size or rate <= 0 or not np.isfinite(audio).all():
        raise ValueError(
            "SpeechLM audio requires finite nonempty waveform and positive sample rate"
        )
    original_hash = hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest()
    if rate != target_rate:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio.copy()), rate, target_rate
        ).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32), {
        "original_sample_rate": rate,
        "sample_rate": target_rate,
        "channel_rule": "first",
        "resampler": "torchaudio.functional.resample",
        "original_first_channel_sha256": original_hash,
    }


@functools.lru_cache(maxsize=2)
def _bagpiper_frontend(train_config, processor_path, revision):
    """The recipe's mel frontend and native text handler, built once.

    Loading them costs about 0.6 s, against milliseconds for one prompt.
    """
    import yaml
    from transformers import AutoFeatureExtractor

    from espnet2.speechlm.model.speechlm.multimodal_io.text import HuggingFaceTextIO

    recipe = yaml.safe_load(Path(train_config).read_text())
    io = recipe["multimodal_io"]["continuous_audio"]
    tag = io.get("encoder_hf_model_tag", "Qwen/Qwen2.5-Omni-7B")
    processor_source = processor_path or tag
    if revision is None and not Path(processor_source).is_dir():
        raise ValueError(
            "Freeze audio_processor_revision or a local processor path before waveform RL"
        )
    frontend = AutoFeatureExtractor.from_pretrained(processor_source, revision=revision)
    processor_files = {}
    if Path(processor_source).is_dir():
        processor_files = {
            str(path.relative_to(processor_source)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(processor_source).rglob("*.json"))
            if path.is_file()
        }
    text_handler = HuggingFaceTextIO(**recipe["multimodal_io"]["text"])
    return recipe, tag, processor_source, frontend, processor_files, text_handler


@functools.lru_cache(maxsize=2)
def _tokenizer(hf_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_dir, use_fast=True)


def _bagpiper_audio_source(config, metadata, source):
    """Use the recipe mel frontend for both native and SGLang audio towers.

    Only the feature extractor is constructed here. The learner retains raw
    mel features and executes its trainable audio encoder/adaptor itself.
    """
    import numpy as np
    import torch

    from espnet2.speechlm.model.speechlm.multimodal_io.audio import ContinuousAudioIO

    options = config.adapter_options
    revision = options.get("audio_processor_revision")
    recipe, tag, processor_source, frontend, processor_files, text_handler = (
        _bagpiper_frontend(
            config.train_config, options.get("audio_processor_path"), revision
        )
    )
    waveform, waveform_contract = _speechlm_waveform(
        source, int(frontend.sampling_rate)
    )
    # Stable ContinuousAudioIO only needs processor geometry during preprocess.
    # Bypass its full multimodal pretrained constructor; no encoder is loaded.
    handler = ContinuousAudioIO.__new__(ContinuousAudioIO)
    torch.nn.Module.__init__(handler)
    handler.processor = frontend
    handler.sample_rate = frontend.sampling_rate
    handler.hop_length = frontend.hop_length
    handler.encoder_hf_model_tag = tag
    _, (encoded_length, features), _ = handler.preprocess(
        (waveform[None], int(frontend.sampling_rate))
    )
    if not encoded_length or not len(features):
        raise ValueError("SpeechLM waveform has no native encoder frames")
    count = int(metadata["num_stream"])
    from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMPreprocessor

    # Keep the native recipe tokenizer, vocabulary offset and message markers.
    # The exported tokenizer is used only for decoding generated text.
    handler.modality, handler.is_discrete = "audio", False
    preprocessor = SpeechLMPreprocessor.__new__(SpeechLMPreprocessor)
    preprocessor.is_train = False
    preprocessor.multimodal_io = {"continuous_audio": handler, "text": text_handler}
    preprocessor.audio_input = "continuous_audio"
    preprocessor.audio_output = "discrete_audio"
    preprocessor.loss_region = "assistant"
    preprocessor.audio_cfg = 0.0
    preprocessor.num_stream = count
    preprocessor.pad_id = 0
    specials = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|eot|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|text|>",
        "<|audio|>",
        "<|image|>",
        "<|video|>",
        "<|toolcall|>",
    ]
    preprocessor.vocab = specials + [f"<|unused_{i}|>" for i in range(12, 256)]
    preprocessor.vocab_intervals = {
        "special_token": [(0, 256)],
        "text": [(metadata["text_token_offset"], metadata["text_token_end"])],
    }
    question = source.get("prompt", source.get("question", ""))
    dialogue = []
    native_qa = source.get("qa_template") == "bagpiper"
    if native_qa:
        from .qa_contract import BAGPIPER_QA_SYSTEM_PROMPT, native_qa_prompt

        if (
            source.get("task") != "qa"
            or source.get("system_prompt") != BAGPIPER_QA_SYSTEM_PROMPT
            or question != native_qa_prompt(source["question"], source["choices"])
        ):
            raise ValueError("Native Bagpiper QA requires its matching user prompt")
        dialogue.append(["system", "text", BAGPIPER_QA_SYSTEM_PROMPT])
    dialogue.append(["user", "audio", (waveform[None], int(frontend.sampling_rate))])
    if question:
        dialogue.append(["user", "text", question])
    processed = preprocessor.preprocessing(
        ("dialogue", source["id"], "rl"), {"dialogue": dialogue}
    )
    prefix = np.concatenate(
        (processed["sequence"], preprocessor.special_token("<|assistant|>"))
    )
    source["prompt_token_ids"] = prefix.tolist()
    primary = prefix[:, 0].tolist()
    _, audio_start, native_encoder_length, native_features = processed["conti_feats"][0]
    if native_encoder_length != encoded_length or not np.array_equal(
        native_features, features
    ):
        raise ValueError(
            "Native chat preprocessing changed the prepared audio features"
        )
    source["native_model_inputs"] = {
        "continuous_audio_feats": np.ascontiguousarray(features[None], np.float32),
        "continuous_audio_lengths": [len(features)],
        "continuous_audio_indices": [[0, audio_start, int(encoded_length)]],
    }
    source["prompt_codec_mask"] = [False] * len(primary)
    source["native_preparation"] = {
        "family": "bagpiper",
        "frontend": tag,
        "processor_source": processor_source,
        "revision": revision,
        "sample_rate": int(frontend.sampling_rate),
        "processor_files_sha256": dict(processor_files),
        "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
        "features_sha256": hashlib.sha256(
            np.ascontiguousarray(features).tobytes()
        ).hexdigest(),
        "feature_length": len(features),
        "encoder_length": int(encoded_length),
        "chat_template": (
            "native SpeechLMPreprocessor/dialogue/system,user-audio,user-text/bagpiper-qa-v2"
            if native_qa
            else "native SpeechLMPreprocessor/dialogue/user-audio,user-text/v1"
        ),
        "text_tokenizer_source": recipe["multimodal_io"]["text"]["tokenizer_name"],
        "native_prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
        "conditioning": "native recipe mel; trainable encoder/adaptor; SGLang precomputed feature input",
        **waveform_contract,
    }
    return source


def exporter(config, native_checkpoint_path, hf_output_dir):
    """Export complete native tensors with the checkpoint-bound token assets."""
    import torch
    from safetensors.torch import save_file

    directory = Path(hf_output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assets = Path(config.adapter_options["reference_assets"])
    for item in assets.iterdir():
        if item.is_file() and item.suffix in (".json", ".txt", ".jinja"):
            import shutil

            shutil.copy2(item, directory / item.name)
    metadata = json.loads((assets / "config.json").read_text())
    _native_export_geometry(config, metadata)
    flat = dict(metadata["text_config"])
    flat.update(
        model_type="qwen3",
        architectures=["ESPnetBagpiperForCausalLM"],
        espnet_bagpiper=metadata,
        eos_token_id=metadata["eos_token_id"],
        bos_token_id=metadata["bos_token_id"],
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    # Explicit training profiles only: absent/false options preserve the
    # previous export and baseline evaluation configuration verbatim.
    if config.family == "bagpiper":
        for option in ("match_native_bf16", "stateless_sampling"):
            if config.adapter_options.get(option, False):
                flat[f"espnet_bagpiper_{option}"] = True
    (directory / "config.json").write_text(json.dumps(flat, indent=2) + "\n")
    state = torch.load(native_checkpoint_path, map_location="cpu", weights_only=True)
    state = state.get("module", state)
    excluded = {"vocab_weight"}
    weights = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if key not in excluded
    }
    names, shard, count, size = {}, {}, 1, 0
    for key, value in weights.items():
        if shard and size + value.numel() * value.element_size() > 5 * 1024**3:
            filename = f"model-{count:05d}.safetensors"
            save_file(shard, str(directory / filename))
            count += 1
            shard = {}
            size = 0
        shard[key] = value
        size += value.numel() * value.element_size()
        names[key] = f"model-{count:05d}.safetensors"
    if shard:
        save_file(shard, str(directory / f"model-{count:05d}.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": names}, indent=2) + "\n"
    )
    (directory / "rl_export.json").write_text(
        json.dumps(
            {
                "source": str(native_checkpoint_path),
                "tensor_count": len(weights),
                "excluded": list(excluded),
                "mapping": "verbatim native; loader explicitly omits frozen codec",
                "recipe": config.train_config,
            },
            indent=2,
        )
        + "\n"
    )


def _native_export_geometry(config, metadata):
    """Copy effective native backbone settings without replacing speech tokens.

    Reference exports determine the speech vocabulary and checkpoint key mapping,
    but their default norm, RoPE, context or audio-window settings can differ from
    the actual recipe. Resolve the same config sources used by the native factory.
    """
    import yaml
    from transformers import AutoConfig

    recipe = yaml.safe_load(Path(config.train_config).read_text())
    model = recipe.get("model", {})
    if config.family == "bagpiper":
        tag = model["model_hf_tag"]
    else:
        tag = recipe.get("transformer_conf", {}).get("hf_model_tag") or recipe.get(
            "corelm_conf", {}
        ).get("hf_model_tag")
    if not tag:
        raise ValueError(
            "SpeechLM export requires the same backbone config as its native recipe"
        )
    native = AutoConfig.from_pretrained(tag).to_dict()
    fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "rms_norm_eps",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "attention_bias",
        "attention_dropout",
        "sliding_window",
        "use_sliding_window",
        "max_window_layers",
        "layer_types",
    )
    values = {key: native[key] for key in fields if key in native}
    rope = native.get("rope_parameters") or {}
    if "rope_theta" not in values and "rope_theta" in rope:
        values["rope_theta"] = rope["rope_theta"]
    if config.family != "bagpiper" and (
        native.get("rope_scaling")
        or rope.get("rope_type", "default") != "default"
        or native.get("hidden_act", "silu") != "silu"
        or native.get("attention_bias", False)
    ):
        raise ValueError(
            "The OpusLM SGLang backbone does not support this native activation/attention/RoPE config"
        )
    target = metadata["text_config"] if config.family == "bagpiper" else metadata
    target.update(values)
    target["tie_word_embeddings"] = (
        model.get("model_conf", {}).get("tie_word_embeddings", False)
        if config.family == "bagpiper"
        else recipe.get("corelm_conf", {}).get("share_emb", False)
    )
    provenance = {"text_config_source": tag, "text_geometry": values}
    if config.family == "bagpiper":
        metadata["hidden_size"] = target["hidden_size"]
        audio = recipe.get("multimodal_io", {}).get("continuous_audio")
        if audio:
            tag = config.adapter_options.get(
                "audio_config_path", audio["encoder_hf_model_tag"]
            )
            audio_geometry = AutoConfig.from_pretrained(
                tag, revision=config.adapter_options.get("audio_config_revision")
            ).thinker_config.audio_config.to_dict()
            audio_values = {
                key: value
                for key, value in audio_geometry.items()
                if not key.startswith("_")
                and key
                not in (
                    "model_type",
                    "architectures",
                    "transformers_version",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "vocab_size",
                )
            }
            metadata["audio_config"].update(audio_values)
            metadata["adaptor_input_dim"] = metadata["audio_config"]["output_dim"]
            provenance.update(audio_config_source=tag, audio_geometry=audio_values)
    return provenance


def request_factory(config, hf_dir, sources):
    """Create exact native multimodal prompts and preserve all codebooks."""
    metadata = json.loads((Path(hf_dir) / "config.json").read_text())["espnet_bagpiper"]
    count = int(metadata["num_stream"])
    requests, prefixes = [], []
    for source in sources:
        if (
            source.get("task") != "tts_asr_wer"
            and ("audio" in source or "audio_array" in source)
            and not source.get("prompt_token_ids")
        ):
            _bagpiper_audio_source(config, metadata, source)
        prefix = source.get("prompt_token_ids")
        if prefix is None:
            primary = [
                1,
                5,
                7,
                *_tokenizer(str(hf_dir)).encode(
                    source.get("prompt", source.get("reference", "")),
                    add_special_tokens=False,
                ),
                2,
                6,
            ]
            prefix = [[value, *([0] * (count - 1))] for value in primary]
        source["prompt_token_ids"] = prefix
        source["mode"] = (
            "text_audio" if source.get("task") == "tts_asr_wer" else "text_only"
        )
        # The temperature is the rollout's sampling_contract.
        source["sampling_extra_args"] = {
            "mode": source["mode"],
            "phase": "text",
            "cfg": 1.0,
            "audio_topk": 0,
        }
        requests.append({"prompt_token_ids": [frame[0] for frame in prefix]})
        prefixes.append(prefix)
    return requests, prefixes
