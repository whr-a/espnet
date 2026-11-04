# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""ESPnet checkpoint export and actual SGLang text rollouts for OpusLM."""

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch
import yaml

from espnet2.rl.engine import SGLangRolloutEngine, engine_kwargs
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


def tts_prompt(text_ids, metadata, task_token=None):
    """Native plain-TTS prompt: sos, task, <text>...</text>, <audio-open>.

    Mirrors prepare.py:asr_prompt in the other direction. The block the model
    then generates is the delayed nine-stream codec; the audio-open marker is
    the last prompt frame, so the first sampled frame is already stream 0 of
    the audio.
    """
    nq = int(metadata.get("nq", 9))
    task = int(
        metadata["codec_ssl_plain_tts_task_token_id"]
        if task_token is None
        else task_token
    )
    primary = [
        int(metadata["eos_token_id"]),
        task,
        int(metadata["text_bpe_start_end_token_id"]),
        *[int(i) + int(metadata["text_token_start"]) for i in text_ids],
        int(metadata["text_bpe_start_end_token_id"]),
        int(metadata["codec_ssl_start_end_token_id"]),
    ]
    return [[i] + [0] * (nq - 1) for i in primary]


def multistream_exporter(config, native_checkpoint_path, hf_output_dir):
    """The same export, named as the nine-stream codec architecture.

    Only ``architectures`` differs from ``exporter``: the weights, the key map
    and the frozen native semantics are identical, so an export can be scored
    by either interface.
    """
    exporter(config, native_checkpoint_path, hf_output_dir)
    path = Path(hf_output_dir) / "config.json"
    metadata = json.loads(path.read_text())
    metadata["architectures"] = [
        (
            "OpusLMMultiStreamForConditionalGeneration"
            if config.family == "opuslm"
            else "OpusLMDialogueMultiStreamForConditionalGeneration"
        )
    ]
    path.write_text(json.dumps(metadata, indent=2) + "\n")


class OpusLMMultiStreamEngine(SGLangRolloutEngine):
    """OpusLM sampling real nine-stream codec inside SGLang.

    The engine submits the primary stream as ``input_ids`` and writes the
    batch's nine-stream prompts to ``directory/sources.pt``, which the model
    reads to rebuild the native pad-cleared stream sum. The model samples
    streams 1..nq-1 itself and reports every frame's activity, actions and log
    probabilities through ``customized_info["espnet_opuslm"]``; SGLang samples
    stream 0 from the support the model leaves open.

    ``max_running_requests`` requests of ``context_length`` tokens must fit the
    KV pool (checked at startup): a retracted request would be recomputed from
    primary ids alone, without its secondary streams.
    """

    defaults = dict(
        skip_tokenizer_init=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        context_length=2048,
        max_running_requests=8,
        attention_backend="triton",
    )
    frozen = (
        "skip_tokenizer_init",
        "disable_cuda_graph",
        "disable_radix_cache",
        "chunked_prefill_size",
        "disable_overlap_schedule",
    )

    def __init__(self, config, model_path, *, directory, replica=0):
        # Read when SGLang's scheduler imports its sampler. The model divides
        # the logits by the temperature itself and SGLang samples them at 1, so
        # the reported primary log probability is the log_softmax of exactly
        # the logits the learner scores, not the log of their softmax.
        os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "1"
        if config.dtype == "float32":
            # Triton's TF32 default would put float32 attention at the gate.
            os.environ.update(TRITON_F32_DEFAULT="ieee", NVIDIA_TF32_OVERRIDE="0")
        self.metadata = _metadata(model_path)
        super().__init__(config, model_path, directory=directory, replica=replica)
        self.check_kv_pool()
        self.model_path = Path(model_path)
        self.tokenizer_files = self.tokenizer = None
        self._load_prompt_assets(model_path)
        self.batches = 0

    @property
    def sources_path(self):
        return self.directory / "sources.pt"

    def settings(self, model_path):
        settings = engine_kwargs(
            self.config,
            self.replica,
            defaults=dict(self.defaults, model_path=str(model_path),
                          dtype=self.config.dtype),
            frozen=self.frozen,
        )
        if self.config.dtype == "float32" and settings["attention_backend"] not in (
            "triton",
            "torch_native",
        ):
            raise ValueError("float32 OpusLM needs triton or torch_native attention")
        overrides = json.loads(settings.get("json_model_override_args", "{}"))
        overrides["espnet_rollout_sources"] = str(self.sources_path.resolve())
        settings["json_model_override_args"] = json.dumps(overrides)
        return settings

    def update(self, model_path):
        if _metadata(model_path) != self.metadata:
            raise ValueError("An update cannot change the exported OpusLM vocabulary")
        super().update(model_path)
        self.model_path = Path(model_path)
        self._load_prompt_assets(model_path)

    def _load_prompt_assets(self, model_path):
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

    def _prompt(self, source):
        """The source's nine-stream prompt, or a plain-TTS one from its text."""
        if source.get("prompt_token_ids") is not None:
            return [list(map(int, frame)) for frame in source["prompt_token_ids"]]
        if "audio" in source and source.get("mode") != "text_audio":
            raise ValueError(
                "Waveform sources must be prepared by the native codec/SSL stage"
            )
        text = source.get("prompt", source.get("text", source.get("reference", "")))
        return tts_prompt(
            self.tokenizer.encode(text, add_special_tokens=False), self.metadata
        )

    def generate(self, sources, rollout_id):
        metadata, group = self.metadata, self.config.group_size
        nq = int(metadata["nq"])
        temperature = float(self.config.temperature)
        prompts = [self._prompt(source) for source in sources]
        for prompt in prompts:
            if not prompt or any(len(frame) != nq for frame in prompt):
                raise ValueError("Native Opus prompt must be nonempty [P,9]")
        self.check_request(max(map(len, prompts)), self.config.max_tokens)
        batch, self.batches = self.batches, self.batches + 1
        prepared = [
            dict(id=source["id"], prompt=torch.tensor(prompt, dtype=torch.long))
            for source, prompt in zip(sources, prompts, strict=True)
        ]
        staging = self.sources_path.with_suffix(".tmp")
        torch.save(
            dict(
                batch=batch,
                rollout_id=rollout_id,
                temperature=temperature,
                sources=prepared,
            ),
            staging,
        )
        staging.replace(self.sources_path)
        rids, parameters = [], []
        for index in range(len(sources)):
            for sample in range(group):
                rids.append(f"{batch}:{index}:{sample}")
                parameters.append(
                    dict(
                        # Model-side temperature scaling is deliberate; see
                        # __init__ on SGLANG_RETURN_ORIGINAL_LOGPROB.
                        temperature=1.0 if temperature else 0.0,
                        top_p=1.0,
                        top_k=-1,
                        min_p=0.0,
                        max_new_tokens=self.config.max_tokens,
                        stop_token_ids=[int(metadata["eos_token_id"])],
                        repetition_penalty=1.0,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                    )
                )
        outputs = self.engine.generate(
            input_ids=[
                [frame[0] for frame in prompts[index]]
                for index in range(len(sources))
                for _ in range(group)
            ],
            sampling_params=parameters,
            rid=rids,
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=0,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        if [output["meta_info"]["id"] for output in outputs] != rids:
            raise RuntimeError("SGLang returned incomplete OpusLM prompt groups")
        generations = []
        for index, source in enumerate(sources):
            native = prepared_native_metadata(
                {**source, "mode": source.get("mode", "text_audio")}
            )
            for output in outputs[index * group : (index + 1) * group]:
                generations.append(
                    self._record(source, prompts[index], output, native, temperature)
                )
        return generations

    def _record(self, source, prompt, output, native, temperature):
        """One generation in the native rollout schema, plus its supports."""
        from espnet2.speechlm.rl.multistream import (
            codec_supports,
            deinterleave_codec,
        )

        metadata = self.metadata
        nq = int(metadata["nq"])
        info = output["meta_info"]
        tokens = list(output["output_ids"])
        logprobs = info.get("output_token_logprobs")
        trace = info.get("espnet_opuslm")
        if info.get("num_retractions"):
            raise RuntimeError("SGLang retracted an OpusLM request")
        if (
            not tokens
            or logprobs is None
            or not len(tokens) == len(logprobs) == len(trace or ())
        ):
            raise RuntimeError("SGLang returned an incomplete OpusLM action trace")
        supports = codec_supports(metadata, temperature=temperature, top_k=0)
        eos = int(metadata["eos_token_id"])
        frames, scores, stream_scores, masks, contracts = [], [], [], [], []
        for token, (logprob, sampled, _), row in zip(
            tokens, logprobs, trace, strict=True
        ):
            active = [bool(flag) for flag in row[:nq]]
            ids = [int(value) for value in row[nq : nq + nq - 1]]
            codec_scores = [float(value) for value in row[nq + nq - 1 :]]
            if sampled != token or not math.isfinite(logprob):
                raise RuntimeError("SGLang primary action and log probability differ")
            if len(ids) != nq - 1 or len(codec_scores) != nq - 1:
                raise RuntimeError("SGLang OpusLM trace lost a secondary stream")
            # Stream 0 is only an action while the block is open; during the
            # delay flush and at EOS its support is a single forced id.
            primary_score = logprob if active[0] else 0.0
            frame = [int(token)] + [
                ids[stream - 1] if active[stream] else 0 for stream in range(1, nq)
            ]
            per_stream = [primary_score] + [
                codec_scores[stream - 1] if active[stream] else 0.0
                for stream in range(1, nq)
            ]
            frames.append(frame)
            scores.append(sum(per_stream))
            stream_scores.append(per_stream)
            masks.append(active)
            contracts.append([dict(spec) for spec in supports])
            if token == eos:
                break
        if not any(any(mask) for mask in masks):
            raise RuntimeError("An OpusLM rollout sampled no action at all")
        # A frame whose every stream is forced carries no action; the learner
        # requires action_mask == stream_mask.any(-1), so drop such a tail.
        while masks and not any(masks[-1]):
            for sequence in (frames, scores, stream_scores, masks, contracts):
                sequence.pop()
        # Report now whether the reward side could decode this block, so a
        # rollout that sampled undecodable codec is visible in the record
        # rather than only when a TTS reward later fails on it.
        _, status = deinterleave_codec(frames, metadata)
        return {
            "source_id": source["id"],
            "token_ids": frames,
            "log_probs": scores,
            "stream_log_probs": stream_scores,
            "stream_mask": masks,
            "action_mask": [any(mask) for mask in masks],
            "codec_frame_mask": [bool(mask[1]) for mask in masks],
            "sampling_contracts": contracts,
            "prompt_token_ids": prompt,
            "text": "",
            "finish_reason": info.get("finish_reason"),
            "prepared_native_source": native,
            "rollout_engine": "sglang",
            "codec_status": status,
        }
