# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Bagpiper rollouts: one batched SGLang generate per rollout batch."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from espnet2.rl.engine import SGLangRolloutEngine, engine_kwargs

from .integration import prepared_native_metadata, request_factory


class BagpiperEngine(SGLangRolloutEngine):
    """Bagpiper's native multi-stream policy inside SGLang's scheduler.

    Every request carries the real primary prompt and the ID
    ``batch:source:sample``; the model reads the batch's prepared sources from
    ``directory/sources.pt``, written before every batch, and traces every
    frame's support and secondary actions. ``max_running_requests`` requests
    of ``context_length`` tokens must fit the KV pool (checked at startup):
    SGLang would otherwise retract requests and recompute them without their
    secondary streams.
    """

    defaults = dict(
        skip_tokenizer_init=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        # A QA prompt of up to 510 frames with 1,536 new tokens. 32 such
        # requests (65.5k KV tokens) fit an FP32 engine alone on an 80 GB
        # GPU (about 120k); recipes size both to their own pool.
        context_length=2048,
        max_running_requests=32,
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
        # the logits by the temperature itself and SGLang samples them at 1,
        # so this is the primary's log_softmax of exactly those logits, as the
        # learner computes it, rather than the log of their softmax.
        os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "1"
        if config.dtype == "float32":
            # Triton's TF32 default puts float32 attention at the drift gate.
            os.environ.update(TRITON_F32_DEFAULT="ieee", NVIDIA_TF32_OVERRIDE="0")
        super().__init__(config, model_path, directory=directory, replica=replica)
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)
        self.geometry = json.loads((self.model_path / "config.json").read_text())[
            "espnet_bagpiper"
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.check_kv_pool()
        self.batches = 0

    @property
    def sources_path(self):
        return self.directory / "sources.pt"

    def settings(self, model_path):
        attention = "triton" if self.config.dtype == "float32" else "fa3"
        settings = engine_kwargs(
            self.config,
            self.replica,
            defaults=dict(
                self.defaults,
                model_path=str(model_path),
                dtype=self.config.dtype,
                attention_backend=attention,
            ),
            frozen=self.frozen,
        )
        if self.config.dtype == "float32" and settings["attention_backend"] not in (
            "triton",
            "torch_native",
        ):
            raise ValueError("float32 Bagpiper needs triton or torch_native attention")
        overrides = json.loads(settings.get("json_model_override_args", "{}"))
        overrides["espnet_rollout_sources"] = str(self.sources_path.resolve())
        settings["json_model_override_args"] = json.dumps(overrides)
        if self.config.adapter_options.get("stateless_sampling", False):
            if settings.get("enable_deterministic_inference") is False:
                raise ValueError(
                    "Stateless Bagpiper sampling requires deterministic inference"
                )
            # SGLang constructs request seed tensors only in this mode.
            settings["enable_deterministic_inference"] = True
        return settings

    def update(self, model_path):
        super().update(model_path)
        self.model_path = Path(model_path)

    def generate(self, sources, rollout_id, sampling=None):
        """Return ``group_size`` generations per source, in source order.

        Standard RL samples every stream from its full support at the recipe's
        temperature; WEST QA uses its explicit top-k for the primary stream.
        ``sampling`` overrides the following for evaluation only:
        ``temperature`` (0 is greedy in every stream, at most 1),
        ``max_new_tokens`` and ``samples`` per source.
        """
        sampling = {
            "temperature": self.config.temperature,
            "max_new_tokens": self.config.max_tokens,
            "samples": self.config.group_size,
            **(sampling or {}),
        }
        if set(sampling) != {"temperature", "max_new_tokens", "samples"} or not (
            0 <= sampling["temperature"] <= 1
        ):
            raise ValueError(
                "Bagpiper sampling overrides temperature (0 to 1), max_new_tokens, "
                "samples"
            )
        _, prefixes = request_factory(self.config, self.model_path, sources)
        group = sampling["samples"]
        west = getattr(self.config, "training_objective", "miles_ppo") == "west_grpo"
        top_k = self.config.top_k if west else -1
        stateless = self.config.adapter_options.get("stateless_sampling", False)
        if (west or stateless) and any(
            source.get("mode") != "text_only" for source in sources
        ):
            raise ValueError(
                "WEST and stateless Bagpiper sampling require text_only QA"
            )
        self.check_request(max(map(len, prefixes)), sampling["max_new_tokens"])
        batch, self.batches = self.batches, self.batches + 1
        prepared = []
        for source, prefix in zip(sources, prefixes, strict=True):
            native, audio = source.get("native_model_inputs", {}), None
            if native.get("continuous_audio_feats") is not None:
                ((row, start, length),) = native["continuous_audio_indices"]
                if row != 0:
                    raise ValueError("A Bagpiper prompt holds exactly one utterance")
                audio = dict(
                    feats=torch.as_tensor(
                        native["continuous_audio_feats"], dtype=torch.float32
                    ),
                    lengths=torch.as_tensor(
                        native["continuous_audio_lengths"], dtype=torch.long
                    ),
                    start=start,
                    length=length,
                )
            prepared.append(
                dict(
                    id=source["id"],
                    prompt=torch.tensor(prefix),
                    text_audio=source["mode"] == "text_audio",
                    samples=group,
                    audio=audio,
                )
            )
        staging = self.sources_path.with_suffix(".tmp")
        record = dict(
            batch=batch,
            rollout_id=rollout_id,
            temperature=sampling["temperature"],
            sources=prepared,
        )
        if stateless:
            record["stateless_sampling"] = True
        torch.save(record, staging)
        staging.replace(self.sources_path)
        rids = [f"{batch}:{i}:{s}" for i in range(len(sources)) for s in range(group)]
        parameters = []
        for source in sources:
            for candidate in range(group):
                params = dict(
                    # Model-side temperature scaling is deliberate: the
                    # original logprob is the tempered protocol distribution
                    # before SGLang's top-k. It audits inference integrity only.
                    temperature=1.0 if sampling["temperature"] else 0.0,
                    top_p=1.0,
                    top_k=top_k,
                    min_p=0.0,
                    max_new_tokens=sampling["max_new_tokens"],
                    stop_token_ids=[self.geometry["eos_token_id"]],
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                )
                if stateless:
                    material = json.dumps(
                        [self.config.seed, int(rollout_id), source["id"], candidate],
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                    params["sampling_seed"] = int.from_bytes(
                        hashlib.sha256(material).digest()[:4], "big"
                    )
                parameters.append(params)
        outputs = self.engine.generate(
            input_ids=[
                [frame[0] for frame in p] for p in prefixes for _ in range(group)
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
            raise RuntimeError("SGLang returned incomplete Bagpiper prompt groups")
        generations = []
        for index, (source, prefix) in enumerate(zip(sources, prefixes, strict=True)):
            # Identical for the whole group: hash the native inputs once.
            native = prepared_native_metadata(source)
            for candidate, output in enumerate(
                outputs[index * group : (index + 1) * group]
            ):
                generations.append(
                    self._record(
                        source,
                        prefix,
                        output,
                        native,
                        sampling,
                        sampling_seed=parameters[index * group + candidate].get(
                            "sampling_seed"
                        ),
                    )
                )
        return generations

    def _record(self, source, prefix, output, native, sampling, *, sampling_seed=None):
        """One generation in the native rollout schema, plus its supports."""
        info, tokens = output["meta_info"], list(output["output_ids"])
        logprobs, trace = info.get("output_token_logprobs"), info.get("espnet_bagpiper")
        if info.get("num_retractions"):
            raise RuntimeError("SGLang retracted a Bagpiper request")
        if (
            not tokens
            or logprobs is None
            or not len(tokens) == len(logprobs) == len(trace or ())
        ):
            raise RuntimeError("SGLang returned an incomplete Bagpiper action trace")
        backend_finish = info.get("finish_reason") or {}
        finish = backend_finish.get("type")
        eos = self.geometry["eos_token_id"]
        if (
            finish not in ("stop", "length")
            or (finish == "stop" and tokens[-1] != eos)
            or eos in tokens[:-1]
            or len(tokens) > sampling["max_new_tokens"]
            or (finish == "length" and len(tokens) != sampling["max_new_tokens"])
        ):
            raise RuntimeError(
                "SGLang ended an invalid Bagpiper request: "
                + json.dumps(
                    {
                        "id": info.get("id"),
                        "finish_reason": backend_finish,
                        "frames": len(tokens),
                        "max_new_tokens": sampling["max_new_tokens"],
                        "eos_positions": [
                            i for i, token in enumerate(tokens) if token == eos
                        ],
                        "tail_tokens": tokens[-8:],
                    }
                )
            )
        # SGLang checks its length cap before EOS. EOS sampled as the last
        # allowed action is complete, and remains in the recorded loss mask.
        if finish == "length" and tokens[-1] == eos:
            finish = "stop"
        secondary = self.geometry["num_stream"] - 1
        frames, scores, stream_scores, masks, supports = [], [], [], [], []
        for token, (logprob, sampled, _), (support, active, *codec) in zip(
            tokens, logprobs, trace, strict=True
        ):
            if (
                sampled != token
                or not math.isfinite(logprob)
                or not (active or logprob == 0.0)
            ):
                raise RuntimeError("SGLang primary action and log probability differ")
            ids, codec_scores = (
                (codec[:secondary], codec[secondary:])
                if codec
                else ([0] * secondary, [0.0] * secondary)
            )
            mask = [bool(active)] + [bool(codec)] * secondary
            per_stream = [logprob if active else 0.0, *codec_scores]
            frames.append([token, *ids])
            scores.append(sum(per_stream))
            stream_scores.append(per_stream)
            masks.append(mask)
            supports.append(support)
        text = self.tokenizer.decode(
            [
                frame[0]
                for frame in frames
                if self.geometry["text_token_offset"]
                <= frame[0]
                < self.geometry["text_token_end"]
            ],
            skip_special_tokens=True,
        )
        west = getattr(self.config, "training_objective", "miles_ppo") == "west_grpo"
        contract = {
            "temperature": sampling["temperature"],
            "top_k": self.config.top_k if west and self.config.top_k > 0 else 0,
        }
        if west:
            contract["log_prob_domain"] = "tempered_protocol_before_top_k"
        result = {
            "source_id": source["id"],
            "token_ids": frames,
            "log_probs": scores,
            "stream_log_probs": stream_scores,
            "stream_mask": masks,
            "action_mask": [any(mask) for mask in masks],
            # WEST keeps pre-top-k integrity scores separate from its raw loss.
            "sampling_contract": contract,
            "codec_frame_mask": [mask[1] for mask in masks],
            "support": supports,
            "prompt_token_ids": prefix,
            "text": text,
            "finish_reason": finish,
            "prepared_native_source": native,
            "rollout_engine": "sglang",
        }
        if sampling_seed is not None:
            result["sampling_seed"] = sampling_seed
        if finish != backend_finish["type"]:
            result["backend_finish_reason"] = backend_finish
        return result
