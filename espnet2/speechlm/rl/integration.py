# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""ESPnet checkpoint export and actual SGLang text rollouts for OpusLM."""

import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml

from espnet2.rl.engine import engine_kwargs


def prepared_native_metadata(source):
    fields = (
        "native_model_inputs",
        "prompt_codec_mask",
        "sampling_extra_args",
        "mode",
        "native_preparation",
    )
    data = {key: source[key] for key in fields if key in source}
    value = {"source_id": source["id"], "data": data}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**value, "sha256": hashlib.sha256(encoded).hexdigest()}


def exporter(config, native_checkpoint_path, hf_output_dir):
    """Export every native model parameter with a strict reversible key map.

    Tokenizer / speech geometry assets must be from the same released model.
    They can be built by the optional setup flow or supplied via reference_assets.
    Native backbone geometry always takes precedence over historical exports.
    """
    from safetensors.torch import save_file
    from transformers import AutoConfig

    output = Path(hf_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assets = Path(config.adapter_options["reference_assets"])
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        if (assets / name).is_file():
            shutil.copy2(assets / name, output / name)
    metadata = json.loads((output / "config.json").read_text())
    recipe = yaml.safe_load(Path(config.train_config).read_text())
    tag = (
        recipe.get("transformer_conf", {}).get("hf_model_tag")
        or recipe["corelm_conf"]["hf_model_tag"]
    )
    geometry = AutoConfig.from_pretrained(tag).to_dict()
    vocab_size = metadata["vocab_size"]
    metadata.update(geometry)
    metadata["vocab_size"] = vocab_size
    metadata["model_type"] = "olmo2" if config.family == "opuslm" else "llama"
    metadata["architectures"] = [
        (
            "OpusLMForConditionalGeneration"
            if config.family == "opuslm"
            else "OpusLMDialogueForConditionalGeneration"
        )
    ]
    metadata["tie_word_embeddings"] = recipe.get("corelm_conf", {}).get(
        "share_emb", False
    )
    # Backbone BOS/EOS IDs belong to its unshifted tokenizer; the native speech
    # vocabulary has its own IDs and must retain them after geometry copying.
    speech = json.loads((assets / "config.json").read_text())
    for key in (
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "text_bpe_start_end_token_id",
    ):
        metadata[key] = speech[key]
    metadata["rl_inference_semantics"] = "native_pad_cleared"
    metadata["rl_native_recipe"] = str(Path(config.train_config).resolve())
    state = torch.load(
        native_checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    state = state.get("module", state)
    mapped, key_map, dropped = {}, {}, []
    for key, value in state.items():
        if key.startswith("criterion."):
            dropped.append(key)
            continue
        if key == "corelm.emb.weight":
            target = "model.embed_tokens.weight"
        elif key == "corelm.lm_head.weight":
            target = "lm_head.weight"
        elif key == "corelm.head_emb.weight":
            target = "head_emb.weight"
        elif key.startswith("corelm.decoders.model."):
            target = "model." + key.removeprefix("corelm.decoders.model.")
        else:
            raise ValueError(f"Unmapped native model state: {key}")
        if target in mapped:
            raise ValueError(f"Duplicate exported state: {target}")
        mapped[target], key_map[key] = value.contiguous(), target
    shards, current, size = [], {}, 0
    for key, value in mapped.items():
        nbytes = value.numel() * value.element_size()
        if current and size + nbytes > 5 * 1024**3:
            shards.append(current)
            current, size = {}, 0
        current[key] = value
        size += nbytes
    if current:
        shards.append(current)
    index = {}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(output / name))
        index.update({key: name for key in shard})
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        v.numel() * v.element_size() for v in mapped.values()
                    )
                },
                "weight_map": index,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "rl_export_mapping.json").write_text(
        json.dumps(
            {
                "key_map": key_map,
                "dropped_nonpolicy": dropped,
                "native_geometry": geometry,
            },
            indent=2,
        )
        + "\n"
    )


def _embedding_weight(model_path):
    from safetensors import safe_open

    path = Path(model_path)
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(
        path / index["model.embed_tokens.weight"], framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor("model.embed_tokens.weight")


def request_factory(config, model_path, sources):
    """Build exact native summed embeddings; never discard secondary streams."""
    from transformers import AutoTokenizer

    metadata = json.loads((Path(model_path) / "config.json").read_text())
    embedding = _embedding_weight(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nq = int(metadata.get("nq", 9))
    requests, prefixes = [], []
    for source in sources:
        prefix = source.get("prompt_token_ids")
        if prefix is None:
            if "audio" in source:
                raise ValueError(
                    "Waveform sources must be prepared by the native codec/SSL stage"
                )
            content = [
                i + metadata["text_token_start"]
                for i in tokenizer.encode(
                    source.get("prompt", source.get("text", "")),
                    add_special_tokens=False,
                )
            ]
            if config.family == "opuslm_dialogue":
                primary = [5, 88, 9, 35, *content, *([0] * (nq - 1)), 10, 35]
            else:
                primary = [5, 64, *content]
            prefix = [[i] + [0] * (nq - 1) for i in primary]
        ids = torch.tensor(prefix, dtype=torch.long)
        if ids.ndim != 2 or ids.shape[1] != nq or not len(ids):
            raise ValueError("Native Opus prompt must be nonempty [P,9]")
        if ((ids < 0) | (ids >= len(embedding))).any():
            raise ValueError("Native prompt token outside released vocabulary")
        embeds = (
            embedding[ids]
            .masked_fill((ids == metadata["pad_token_id"])[..., None], 0)
            .sum(1)
        )
        requests.append({"input_embeds": embeds.float().tolist()})
        prefixes.append(ids.tolist())
    return requests, prefixes


def sglang_generate(config, model_path, sources, step):
    """Use the actual pinned SGLang engine and retain every sampled probability."""
    import sglang as sgl
    from transformers import AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = json.loads((Path(model_path) / "config.json").read_text())
    if metadata.get("rl_inference_semantics") != "native_pad_cleared":
        raise ValueError(
            "Opus RL requires explicitly frozen native SGLang embedding semantics"
        )
    requests, prefixes = request_factory(config, model_path, sources)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = engine_kwargs(
        config,
        step,
        defaults=dict(
            model_path=model_path,
            dtype=config.dtype,
            skip_tokenizer_init=True,
            attention_backend="triton",
            mem_fraction_static=0.8,
        ),
        frozen=("model_path", "dtype", "tp_size"),
    )
    engine = sgl.Engine(**kwargs)
    generations = []
    try:
        for source, request, prefix in zip(sources, requests, prefixes, strict=True):
            for sample in range(config.group_size):
                result = engine.generate(
                    **request,
                    sampling_params={
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_new_tokens": config.max_tokens,
                        "stop_token_ids": [metadata["eos_token_id"]],
                        "repetition_penalty": 1.0,
                        "sampling_seed": config.seed + step * 10000 + sample,
                    },
                    return_logprob=True,
                    logprob_start_len=-1,
                )
                primary = result["output_ids"]
                probabilities = result["meta_info"]["output_token_logprobs"]
                if len(primary) != len(probabilities) or not primary:
                    raise RuntimeError(
                        "SGLang returned incomplete behavior probabilities"
                    )
                if any(
                    int(pair[1]) != int(token)
                    for pair, token in zip(probabilities, primary, strict=True)
                ):
                    raise RuntimeError(
                        "SGLang output IDs and sampled logprob IDs differ"
                    )
                nq = int(metadata.get("nq", 9))
                active = [True] + [False] * (nq - 1)
                spec = {
                    "temperature": 1.0,
                    "top_k": 0,
                    "allowed_ranges": [
                        [metadata["text_token_start"], metadata["text_token_end"]]
                    ],
                    "allowed_token_ids": [
                        metadata["eos_token_id"],
                        metadata["text_bpe_start_end_token_id"],
                    ],
                }
                source = {
                    **source,
                    "mode": "audio_text" if "audio" in source else "text_text",
                }
                text_ids = [
                    i - metadata["text_token_start"]
                    for i in primary
                    if metadata["text_token_start"] <= i < metadata["text_token_end"]
                ]
                generations.append(
                    {
                        "source_id": source["id"],
                        "token_ids": [[i] + [0] * (nq - 1) for i in primary],
                        "log_probs": [float(pair[0]) for pair in probabilities],
                        "stream_log_probs": [
                            [float(pair[0])] + [0.0] * (nq - 1)
                            for pair in probabilities
                        ],
                        "action_mask": [True] * len(primary),
                        "stream_mask": [active] * len(primary),
                        "sampling_contracts": [
                            [spec] + [{"temperature": 1.0, "top_k": 0}] * (nq - 1)
                            for _ in primary
                        ],
                        "codec_frame_mask": [False] * len(primary),
                        "prompt_token_ids": prefix,
                        "text": tokenizer.decode(text_ids, skip_special_tokens=True),
                        "finish_reason": result["meta_info"].get("finish_reason"),
                        "prepared_native_source": prepared_native_metadata(source),
                        "sglang_raw_response": result,
                    }
                )
    finally:
        engine.shutdown()
    return generations


def asr_wer_reward(source, generation):
    """WER on sampled text; accept native audio-to-text conditioning modes."""
    import unicodedata

    from espnet2.rl.rewards import edit_distance

    if source.get("mode", "audio_text") not in (
        "audio_text",
        "text_text",
        "text_dialogue",
    ):
        raise ValueError("Opus ASR WER requires text output")
    reference = unicodedata.normalize("NFKC", source["reference"]).casefold().split()
    hypothesis = unicodedata.normalize("NFKC", generation["text"]).casefold().split()
    return -edit_distance(reference, hypothesis) / max(1, len(reference))
