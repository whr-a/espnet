"""Actual SGLang Engine rollout hook for native OWSM policies."""

import hashlib
import json
import math
from pathlib import Path

from espnet2.s2t.rl.policy import owsm_symbol_id


def _prefix(config, source, metadata):
    tokens = metadata["espnet_config"]["token_list"]
    ids = {token: index for index, token in enumerate(tokens)}
    if "prompt_token_ids" in source:
        return list(source["prompt_token_ids"])
    model = metadata["espnet_config"].get("model_conf", {})
    prefix = [
        ids[model.get("sym_sos", "<sos>")],
        owsm_symbol_id(tokens, source.get("language", config.language), language=True),
        owsm_symbol_id(tokens, source.get("task", config.task)),
    ]
    if not source.get("predict_timestamps", config.predict_timestamps):
        prefix.append(ids["<notimestamps>"])
    if source.get("previous_token_ids"):
        prefix = [
            ids[model.get("sym_sop", "<sop>")],
            *source["previous_token_ids"],
            *prefix,
        ]
    return prefix


def _text(token_ids, model_path):
    import sentencepiece as spm

    tokens = (Path(model_path) / "tokens.txt").read_text().splitlines()
    pieces = [
        tokens[token]
        for token in token_ids
        if not (tokens[token].startswith("<") and tokens[token].endswith(">"))
    ]
    processor = spm.SentencePieceProcessor(
        model_file=str(Path(model_path) / "bpe.model")
    )
    return processor.decode(pieces)


def generate(config, model_path, sources, step):
    from sglang import Engine

    metadata = json.loads((Path(model_path) / "config.json").read_text())
    request_path = Path(model_path).parent / "requests.json"
    if json.loads(request_path.read_text()) != sources:
        raise ValueError("SGLang sources differ from frozen rollout requests")
    defaults = dict(
        model_path=str(model_path),
        dtype="float32",
        tp_size=1,
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        attention_backend="torch_native",
        sampling_backend="pytorch",
        mem_fraction_static=0.25,
        context_length=2048,
        max_running_requests=16,
        random_seed=config.seed + step,
        log_level="info",
    )
    overrides = dict(config.sglang_engine_kwargs)
    frozen = {
        "model_path",
        "dtype",
        "tp_size",
        "skip_tokenizer_init",
        "disable_cuda_graph",
        "disable_radix_cache",
        "chunked_prefill_size",
    }
    if any(key in overrides and overrides[key] != defaults[key] for key in frozen):
        raise ValueError(
            "sglang_engine_kwargs cannot change the native OWSM execution contract"
        )
    defaults.update(overrides)
    engine = Engine(**defaults)
    prefixes = [_prefix(config, source, metadata) for source in sources]
    inputs, sampling = [], []
    for source_index, prefix in enumerate(prefixes):
        if source_index >= metadata["vocab_size"]:
            raise ValueError("One rollout batch exceeds source-index carrier capacity")
        for group_index in range(config.group_size):
            inputs.append([source_index, *prefix])
            sampling.append(
                dict(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_new_tokens=config.max_tokens,
                    stop_token_ids=[metadata["eos_token_id"]],
                    repetition_penalty=1.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                )
            )
    try:
        outputs = engine.generate(
            input_ids=inputs,
            sampling_params=sampling,
            return_logprob=True,
            logprob_start_len=[len(row) - 1 for row in inputs],
            top_logprobs_num=0,
        )
    finally:
        engine.shutdown()
    if isinstance(outputs, dict):
        outputs = [outputs]
    if len(outputs) != len(inputs):
        raise RuntimeError("SGLang returned incomplete OWSM prompt groups")
    generations = []
    source_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    for index, output in enumerate(outputs):
        source_index = index // config.group_size
        source = sources[source_index]
        token_ids = list(output["output_ids"])
        logprobs = output["meta_info"].get("output_token_logprobs")
        if logprobs is None or len(logprobs) != len(token_ids):
            raise RuntimeError("SGLang omitted actual sampled action logprobs")
        if any(
            int(item[1]) != token
            for item, token in zip(logprobs, token_ids, strict=True)
        ):
            raise RuntimeError("SGLang output IDs and behavior logprob IDs disagree")
        probabilities = [float(item[0]) for item in logprobs]
        if not token_ids or not all(math.isfinite(value) for value in probabilities):
            raise RuntimeError("SGLang generated empty or invalid behavior traces")
        generations.append(
            dict(
                source_id=source["id"],
                prompt_token_ids=prefixes[source_index],
                token_ids=token_ids,
                log_probs=probabilities,
                text=_text(token_ids, model_path),
                finish_reason=output["meta_info"].get("finish_reason"),
                rollout_backend="sglang",
                source_sha256=source_sha256,
                sglang_raw_response=output,
            )
        )
    return generations
