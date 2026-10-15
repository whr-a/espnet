"""Explicit optional OWSM vLLM Engine backend, independent from SGLang."""

import json
from pathlib import Path

from espnet2.s2t.rl.policy import load_audio
from espnet2.s2t.rl.sglang_rollout import _prefix


def requests(config, model_path, sources):
    """Prepare vLLM encoder/decoder requests; this function does not generate."""
    from vllm.inputs import ExplicitEncoderDecoderPrompt

    metadata = json.loads((Path(model_path) / "config.json").read_text())
    native = metadata["espnet_config"]
    preprocessing = native.get("preprocessor_conf", {})
    sample_rate = int(preprocessing.get("fs", 16000))
    window = (
        int(sample_rate * preprocessing["speech_length"])
        if "speech_length" in preprocessing
        else None
    )
    result, prefixes = [], []
    for source in sources:
        prefix = _prefix(config, source, metadata)
        waveform = load_audio(source, sample_rate, window)
        result.append(
            ExplicitEncoderDecoderPrompt(
                encoder_prompt={
                    "prompt_token_ids": [0],
                    "multi_modal_data": {"audio": (waveform, sample_rate)},
                },
                decoder_prompt={"prompt_token_ids": prefix},
            )
        )
        prefixes.append(prefix)
    return result, prefixes


def generate(config, model_path, sources, step):
    """Call actual vLLM LLM.generate and retain raw sampled log probabilities."""
    from vllm import LLM, SamplingParams

    request_rows, prefixes = requests(config, model_path, sources)
    kwargs = dict(config.vllm_engine_kwargs)
    frozen = {
        "model",
        "seed",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "dtype",
        "logprobs_mode",
    }
    if frozen & kwargs.keys():
        raise ValueError("vllm_engine_kwargs cannot override frozen policy settings")
    if kwargs.get("speculative_config"):
        raise ValueError("OWSM native RL does not support speculative decoding")
    kwargs["tokenizer_mode"] = "owsm"
    engine = LLM(
        model=str(model_path),
        seed=config.seed + step,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        gpu_memory_utilization=config.rollout_mem_fraction or 0.8,
        dtype=config.dtype,
        logprobs_mode="raw_logprobs",
        **kwargs,
    )
    params = SamplingParams(
        n=config.group_size,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=config.max_tokens,
        logprobs=0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        detokenize=True,
    )
    outputs = engine.generate(request_rows, params, use_tqdm=False)
    if len(outputs) != len(sources):
        raise RuntimeError("vLLM returned incomplete OWSM prompt groups")
    result = []
    for source, output, prefix in zip(sources, outputs, prefixes, strict=True):
        if len(output.outputs) != config.group_size:
            raise RuntimeError("vLLM returned an incomplete GRPO group")
        for completion in output.outputs:
            tokens = list(completion.token_ids)
            if completion.logprobs is None or len(completion.logprobs) != len(tokens):
                raise RuntimeError("vLLM omitted sampled action log probabilities")
            result.append(
                dict(
                    source_id=source["id"],
                    prompt_token_ids=prefix,
                    token_ids=tokens,
                    log_probs=[
                        row[token].logprob
                        for row, token in zip(completion.logprobs, tokens, strict=True)
                    ],
                    text=completion.text,
                    finish_reason=completion.finish_reason,
                    rollout_backend="vllm",
                    vllm_request_id=output.request_id,
                    vllm_completion_index=completion.index,
                )
            )
    return result
