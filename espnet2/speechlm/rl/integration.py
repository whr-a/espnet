# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""ESPnet checkpoint export and actual SGLang text rollouts for OpusLM."""

import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml

from espnet2.rl.engine import SGLangRolloutEngine
from espnet2.rl.rewards import word_error_rate

# The released text tokenizer, copied into every export with config.json.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)


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
    for name in ("config.json", *TOKENIZER_FILES):
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


def _metadata(model_path):
    metadata = json.loads((Path(model_path) / "config.json").read_text())
    if metadata.get("rl_inference_semantics") != "native_pad_cleared":
        raise ValueError(
            "Opus RL requires explicitly frozen native SGLang embedding semantics"
        )
    return metadata


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


def _native_prompt(config, source, metadata, tokenizer):
    """The source's nine-stream prompt; never discard secondary streams."""
    if source.get("prompt_token_ids") is not None:
        return source["prompt_token_ids"]
    if "audio" in source:
        raise ValueError(
            "Waveform sources must be prepared by the native codec/SSL stage"
        )
    nq = int(metadata.get("nq", 9))
    content = [
        i + metadata["text_token_start"]
        for i in tokenizer.encode(
            source.get("prompt", source.get("text", "")), add_special_tokens=False
        )
    ]
    if config.family == "opuslm_dialogue":
        primary = [5, 88, 9, 35, *content, *([0] * (nq - 1)), 10, 35]
    else:
        primary = [5, 64, *content]
    return [[i] + [0] * (nq - 1) for i in primary]


def _sampling_seed(seed, rollout_id, source_id, sample):
    digest = hashlib.sha256(f"{seed}:{rollout_id}:{source_id}:{sample}".encode())
    return int.from_bytes(digest.digest()[:8], "big") >> 1


class OpusLMEngine(SGLangRolloutEngine):
    """OpusLM's released backbone inside SGLang: nine-stream prompts, text out.

    A prompt reaches the engine as its exact native embedding, the pad-cleared
    sum of all nine streams, computed here from the export's own embedding
    table; the engine embeds every sampled stream-zero token itself. One
    ``generate`` call submits every candidate of the batch, each with its own
    sampling parameters, and request IDs name rollout, source and sample.
    """

    # input_embeds requests reach the scheduler with placeholder token IDs, so
    # a radix cache would share KV between different prompts. The engine never
    # tokenizes, and CUDA graphs are not validated for the text-support mask.
    frozen = ("skip_tokenizer_init", "disable_cuda_graph", "disable_radix_cache")

    @property
    def defaults(self):
        # The engine computes in the native policy's own precision.
        return dict(
            dtype=self.config.dtype,
            skip_tokenizer_init=True,
            attention_backend="triton",
        )

    def __init__(self, config, model_path, *, directory, replica=0):
        self.metadata = _metadata(model_path)
        super().__init__(config, model_path, directory=directory, replica=replica)
        # A retracted input_embeds request would restart without its output.
        self.check_kv_pool()
        self.tokenizer_files = self.tokenizer = self.embedding = None
        self._load_prompt_assets(model_path)

    def update(self, model_path):
        if _metadata(model_path) != self.metadata:
            raise ValueError("An update cannot change the exported OpusLM vocabulary")
        super().update(model_path)
        self._load_prompt_assets(model_path)

    def _load_prompt_assets(self, model_path):
        """Read the export's embedding table, which training changes, and its
        tokenizer when that differs from the one already loaded."""
        from transformers import AutoTokenizer

        path = Path(model_path)
        files = {
            name: (path / name).read_bytes()
            for name in TOKENIZER_FILES
            if (path / name).is_file()
        }
        if files != self.tokenizer_files:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.tokenizer_files = files
        self.embedding = _embedding_weight(path)

    def generate(self, sources, rollout_id):
        metadata, group = self.metadata, self.config.group_size
        nq = int(metadata.get("nq", 9))
        prompts, embeds = [], []
        for source in sources:
            ids = torch.tensor(
                _native_prompt(self.config, source, metadata, self.tokenizer),
                dtype=torch.long,
            )
            if ids.ndim != 2 or ids.shape[1] != nq or not len(ids):
                raise ValueError("Native Opus prompt must be nonempty [P,9]")
            if ((ids < 0) | (ids >= len(self.embedding))).any():
                raise ValueError("Native prompt token outside released vocabulary")
            # Native ARDelay embedding: zero every pad entry, then sum streams.
            summed = (
                self.embedding[ids]
                .masked_fill((ids == metadata["pad_token_id"])[..., None], 0)
                .sum(1)
            )
            prompts.append(ids.tolist())
            embeds.append(summed.float().tolist())
        self.check_request(max(map(len, prompts)), self.config.max_tokens)
        rids, inputs, sampling = [], [], []
        for index, source in enumerate(sources):
            for sample in range(group):
                rids.append(f"{rollout_id}:{index}:{sample}")
                # One list per prompt; its candidates share it.
                inputs.append(embeds[index])
                sampling.append(
                    dict(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        max_new_tokens=self.config.max_tokens,
                        stop_token_ids=[metadata["eos_token_id"]],
                        repetition_penalty=1.0,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        # Read only by deterministic inference, where equal
                        # seeds would make a group's candidates identical.
                        sampling_seed=_sampling_seed(
                            self.config.seed, rollout_id, source["id"], sample
                        ),
                    )
                )
        outputs = self.engine.generate(
            input_embeds=inputs, sampling_params=sampling, rid=rids, return_logprob=True
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        if len(outputs) != len(rids) or any(
            output["meta_info"]["id"] != rid
            for output, rid in zip(outputs, rids, strict=True)
        ):
            raise RuntimeError("SGLang returned incomplete OpusLM prompt groups")
        start, end = metadata["text_token_start"], metadata["text_token_end"]
        active = [True] + [False] * (nq - 1)
        spec = {
            "temperature": 1.0,
            "top_k": 0,
            "allowed_ranges": [[start, end]],
            "allowed_token_ids": [
                metadata["eos_token_id"],
                metadata["text_bpe_start_end_token_id"],
            ],
        }
        contract = [spec] + [{"temperature": 1.0, "top_k": 0}] * (nq - 1)
        prepared = [
            prepared_native_metadata(
                {**source, "mode": "audio_text" if "audio" in source else "text_text"}
            )
            for source in sources
        ]
        generations = []
        for index, (output, rid) in enumerate(zip(outputs, rids, strict=True)):
            source_index = index // group
            reason = output["meta_info"].get("finish_reason")
            if isinstance(reason, dict) and reason.get("type") == "abort":
                raise RuntimeError(f"SGLang aborted {rid}: {reason.get('message')}")
            primary = list(output["output_ids"])
            probabilities = output["meta_info"].get("output_token_logprobs")
            if (
                not primary
                or probabilities is None
                or len(probabilities) != len(primary)
            ):
                raise RuntimeError("SGLang returned incomplete behavior probabilities")
            if any(
                int(pair[1]) != int(token)
                for pair, token in zip(probabilities, primary, strict=True)
            ):
                raise RuntimeError("SGLang output IDs and sampled logprob IDs differ")
            text_ids = [i - start for i in primary if start <= i < end]
            generations.append(
                {
                    "source_id": sources[source_index]["id"],
                    "token_ids": [[i] + [0] * (nq - 1) for i in primary],
                    "log_probs": [float(pair[0]) for pair in probabilities],
                    "stream_log_probs": [
                        [float(pair[0])] + [0.0] * (nq - 1) for pair in probabilities
                    ],
                    "action_mask": [True] * len(primary),
                    "stream_mask": [active] * len(primary),
                    "sampling_contracts": [contract] * len(primary),
                    "codec_frame_mask": [False] * len(primary),
                    "prompt_token_ids": prompts[source_index],
                    "text": self.tokenizer.decode(text_ids, skip_special_tokens=True),
                    "finish_reason": reason,
                    "prepared_native_source": prepared[source_index],
                    "sglang_raw_response": output,
                }
            )
        return generations


def asr_wer_reward(source, generation):
    """Negative WER of the sampled text, normalized as OpusLM evaluation is."""
    if source.get("mode", "audio_text") not in (
        "audio_text",
        "text_text",
        "text_dialogue",
    ):
        raise ValueError("Opus ASR WER requires text output")
    return -word_error_rate(source["reference"], generation["text"])
