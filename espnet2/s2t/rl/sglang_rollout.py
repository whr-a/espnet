"""Actual SGLang Engine rollout for native OWSM policies."""

import hashlib
import json
import math
import os
from pathlib import Path

from espnet2.rl.engine import SGLangRolloutEngine
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


def _detokenizer(model_path):
    """Load the exported token list and BPE model once per engine."""
    import sentencepiece as spm

    tokens = (Path(model_path) / "tokens.txt").read_text().splitlines()
    processor = spm.SentencePieceProcessor(
        model_file=str(Path(model_path) / "bpe.model")
    )

    def text(token_ids):
        pieces = [
            tokens[token]
            for token in token_ids
            if not (tokens[token].startswith("<") and tokens[token].endswith(">"))
        ]
        return processor.decode(pieces)

    return text


class OWSMEngine(SGLangRolloutEngine):
    """OWSM's native encoder-decoder inside SGLang's scheduler and sampler.

    Requests carry the real OWSM prefix. Each request ID names its rollout,
    source and sample; the model reads the rollout's sources from
    ``directory/sources.json``, written before every batch.
    """

    defaults = dict(
        dtype="float32",
        skip_tokenizer_init=True,
        attention_backend="torch_native",
        sampling_backend="pytorch",
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        mem_fraction_static=0.25,
        context_length=2048,
        max_running_requests=16,
    )
    frozen = (
        "skip_tokenizer_init",
        "disable_cuda_graph",
        "disable_radix_cache",
        "chunked_prefill_size",
    )

    def __init__(self, config, model_path, *, directory, replica=0):
        # Read when SGLang's scheduler imports its sampler: each sampled
        # token's log probability at the sampling temperature,
        # log(softmax(logits / T)), not the one before it.
        os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "0"
        super().__init__(config, model_path, directory=directory, replica=replica)
        # The vocabulary and prompt symbols are the architecture's; every
        # later export of this run shares them.
        self.metadata = json.loads((Path(model_path) / "config.json").read_text())
        self.text = _detokenizer(model_path)

    @property
    def sources_path(self):
        return self.directory / "sources.json"

    def settings(self, model_path):
        settings = super().settings(model_path)
        overrides = json.loads(settings.get("json_model_override_args", "{}"))
        overrides["espnet_rollout_sources"] = str(self.sources_path.resolve())
        settings["json_model_override_args"] = json.dumps(overrides)
        return settings

    def generate(self, sources, rollout_id):
        prefixes = [_prefix(self.config, source, self.metadata) for source in sources]
        encoded = json.dumps({"rollout_id": rollout_id, "sources": sources}).encode()
        staging = self.sources_path.with_suffix(".tmp")
        staging.write_bytes(encoded)
        staging.replace(self.sources_path)
        source_sha256 = hashlib.sha256(encoded).hexdigest()
        inputs, rids, sampling = [], [], []
        for source_index, prefix in enumerate(prefixes):
            for sample in range(self.config.group_size):
                inputs.append(prefix)
                rids.append(f"{rollout_id}:{source_index}:{sample}")
                sampling.append(
                    dict(
                        temperature=self.config.temperature,
                        top_p=1.0,
                        top_k=-1,
                        max_new_tokens=self.config.max_tokens,
                        stop_token_ids=[self.metadata["eos_token_id"]],
                        repetition_penalty=1.0,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                    )
                )
        outputs = self.engine.generate(
            input_ids=inputs,
            sampling_params=sampling,
            rid=rids,
            return_logprob=True,
            logprob_start_len=[len(row) - 1 for row in inputs],
            top_logprobs_num=0,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        if len(outputs) != len(inputs) or any(
            output["meta_info"]["id"] != rid
            for output, rid in zip(outputs, rids, strict=True)
        ):
            raise RuntimeError("SGLang returned incomplete OWSM prompt groups")
        generations = []
        for index, output in enumerate(outputs):
            source_index = index // self.config.group_size
            source = sources[source_index]
            token_ids = list(output["output_ids"])
            logprobs = output["meta_info"].get("output_token_logprobs")
            if logprobs is None or len(logprobs) != len(token_ids):
                raise RuntimeError("SGLang omitted actual sampled action logprobs")
            if any(
                int(item[1]) != token
                for item, token in zip(logprobs, token_ids, strict=True)
            ):
                raise RuntimeError(
                    "SGLang output IDs and behavior logprob IDs disagree"
                )
            probabilities = [float(item[0]) for item in logprobs]
            if not token_ids or not all(
                math.isfinite(value) for value in probabilities
            ):
                raise RuntimeError("SGLang generated empty or invalid behavior traces")
            generations.append(
                dict(
                    source_id=source["id"],
                    prompt_token_ids=prefixes[source_index],
                    token_ids=token_ids,
                    log_probs=probabilities,
                    text=self.text(token_ids),
                    finish_reason=output["meta_info"].get("finish_reason"),
                    rollout_backend="sglang",
                    source_sha256=source_sha256,
                    sglang_raw_response=output,
                )
            )
        return generations
