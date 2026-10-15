# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Actual SGLang generation with lossless Bagpiper multi-stream traces."""

import copy
import json
import os
from pathlib import Path

from espnet2.rl.engine import engine_kwargs

from .events import freeze_native_inputs, request_record
from .integration import prepared_native_metadata, request_factory


def generate(config, model_path, sources, step):
    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "1"
    model_path = Path(model_path)
    _, prefixes = request_factory(config, model_path, sources)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    metadata = json.loads((model_path / "config.json").read_text())
    payload_path = Path(metadata["espnet_rl_payload_path"])
    # Export directory is atomically moved after freezing. Resolve its current
    # path explicitly rather than leaving the staging path in server config.
    payload_path = model_path.parent / "payload.json"
    model_args = config.sglang_engine_kwargs.get("json_model_override_args", "{}")
    model_args = (
        json.loads(model_args) if isinstance(model_args, str) else dict(model_args)
    )
    model_args["espnet_rl_payload_path"] = str(payload_path)
    kwargs = engine_kwargs(
        config,
        step,
        defaults=dict(
            model_path=str(model_path),
            dtype=config.dtype,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
            max_running_requests=1,
            mem_fraction_static=0.65,
        ),
        frozen=(
            "chunked_prefill_size",
            "disable_cuda_graph",
            "disable_radix_cache",
            "disable_overlap_schedule",
            # The native event capture assumes one request in flight.
            "max_running_requests",
        ),
    )
    # The payload path is resolved per run and is never a recipe setting.
    kwargs["json_model_override_args"] = json.dumps(model_args)
    engine = sgl.Engine(**kwargs)
    generations = []
    events = model_path.parent / "sglang_events.jsonl"
    try:
        for source, prefix in zip(sources, prefixes, strict=True):
            native_reference = freeze_native_inputs(
                model_path.parent, source.get("native_model_inputs", {})
            )
            for sample in range(config.group_size):
                sequence = copy.deepcopy(prefix)
                phase = "text"
                frames = []
                scores = []
                stream_scores = []
                masks = []
                contracts = []
                codec = []
                ended = False
                for position in range(config.max_tokens):
                    payload = {
                        "sequence": sequence,
                        "native_model_inputs": source.get("native_model_inputs", {}),
                        "mode": source["mode"],
                        "phase": phase,
                        "source_id": source["id"],
                        "sample": sample,
                        "step": step,
                        "position": position,
                        "cache_reset": position == 0,
                        "cache_release": position == config.max_tokens - 1,
                    }
                    payload_path.write_text(json.dumps(payload) + "\n")
                    result = engine.generate(
                        input_ids=[frame[0] for frame in sequence],
                        sampling_params={
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_new_tokens": 1,
                            "ignore_eos": True,
                        },
                        return_logprob=True,
                        logprob_start_len=-1,
                    )
                    with events.open("a") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "schema": "bagpiper-inference-event-v2",
                                    "request": request_record(
                                        payload, native_reference
                                    ),
                                    "response": result,
                                }
                            )
                            + "\n"
                        )
                    info = result["meta_info"]
                    output = info["output_token_logprobs"]
                    if len(output) != 1:
                        raise RuntimeError("SGLang must return exactly one real action")
                    lp, primary, *_ = output[0]
                    trace = info["espnet_token_trace"][0]
                    frame = [int(primary), *trace["secondary_ids"]]
                    mask = [trace["primary_active"], *trace["secondary_mask"]]
                    per_stream = [
                        float(lp) if mask[0] else 0.0,
                        *trace["secondary_log_probs"],
                    ]
                    frames.append(frame)
                    sequence.append(frame)
                    scores.append(sum(per_stream))
                    stream_scores.append(per_stream)
                    masks.append(mask)
                    contracts.append(trace["sampling_contracts"])
                    codec.append(any(mask[1:]))
                    if source["mode"] == "text_audio":
                        if phase == "text" and primary == 3:
                            phase = "transition"
                        elif phase == "transition":
                            phase = "audio"
                        elif phase == "audio" and primary == 3:
                            phase = "audio_stop"
                    if primary == 2:
                        ended = True
                        break
                text = tokenizer.decode(
                    [
                        frame[0]
                        for frame in frames
                        if 256
                        <= frame[0]
                        < metadata["espnet_bagpiper"]["text_token_end"]
                    ],
                    skip_special_tokens=True,
                )
                generations.append(
                    {
                        "source_id": source["id"],
                        "token_ids": frames,
                        "log_probs": scores,
                        "stream_log_probs": stream_scores,
                        "stream_mask": masks,
                        "action_mask": [any(row) for row in masks],
                        "sampling_contracts": contracts,
                        "codec_frame_mask": codec,
                        "prompt_token_ids": prefix,
                        "text": text,
                        "finish_reason": "stop" if ended else "length",
                        "prepared_native_source": prepared_native_metadata(source),
                        "rollout_engine": "sglang",
                    }
                )
    finally:
        engine.shutdown()
    return generations
