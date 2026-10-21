"""Native OWSM decoder executed by SGLang's actual model runner and sampler.

This version prioritizes exact native probabilities over KV throughput: each
step recomputes every running request's full prefix with the native decoder,
batched into padded calls exactly as the native scorer pads its candidates.
It supports TP=1, full prefill and disabled radix cache/CUDA graph. Input IDs
are the real OWSM prefix. Each request ID is ``rollout:source:sample``; the
rollout's sources are read from the file named by ``espnet_rollout_sources``,
which the engine rewrites before every batch.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from espnet2.s2t.rl.policy import OWSMPolicy, load_audio

# Upper bound on the full-vocabulary FP32 logits one decoder call materializes.
LOGITS_BUDGET_BYTES = 2 * 1024**3


class ESPnetOWSMForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Native OWSM RL does not support weight quantization")
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        from espnet2.tasks.s2t import S2TTask

        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Native OWSM SGLang requires TP=1")
        self.config = config
        self.directory = Path(config._name_or_path).resolve()
        if config.espnet_rollout_contract != "rid-sources/native-full-prefix/v2":
            raise ValueError("Unknown native OWSM rollout contract")
        self.target_device = torch.empty(0).device
        native = dict(config.espnet_config)
        native["bpemodel"] = str(self.directory / native["bpemodel"])
        if native.get("normalize") == "global_mvn":
            native["normalize_conf"] = dict(native["normalize_conf"])
            native["normalize_conf"]["stats_file"] = str(
                self.directory / native["normalize_conf"]["stats_file"]
            )
        args = argparse.Namespace(**native)
        # Native positional caches and frontend are initialized in FP32 on CPU,
        # just as they are in the train actor, then moved by model_runner.
        with torch.device("cpu"):
            saved = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.policy = OWSMPolicy(
                    S2TTask.build_model(args), args, train_encoder=True
                )
            finally:
                torch.set_default_dtype(saved)
        self.policy.eval()
        self.source_path = Path(config.espnet_rollout_sources)
        self.rollout_id = self.sources = self.source_sha256 = None
        self.memory = {}
        self.request_prefixes = {}

    def load_weights(self, weights):
        source = {key: tensor for key, tensor in weights}
        target = self.policy.model.state_dict()
        normalized = {}
        for key, tensor in source.items():
            name = key
            if name not in target:
                name = name.replace(".embed.out.0.", ".embed.out.").replace(
                    ".embed.out.1.", ".embed.pos_enc."
                )
            if name in normalized:
                raise ValueError(f"Duplicate native checkpoint key: {name}")
            normalized[name] = tensor
        self.policy.model.load_state_dict(normalized, strict=True)
        # FP32/IEEE is the accepted initial contract. Do not silently recast
        # frontend mel statistics or sinusoidal caches to reduced precision.
        if (
            str(getattr(self.config, "torch_dtype", "float32")).replace("torch.", "")
            != "float32"
        ):
            raise ValueError("The OWSM initial SGLang RL contract requires float32")
        self.policy.to(device=self.target_device, dtype=torch.float32)
        self.policy.eval()
        # Every update is a complete policy and the encoder is trained, so
        # nothing computed with the previous weights may survive it.
        self.memory.clear()
        self.request_prefixes.clear()
        return set(normalized)

    def _load_sources(self, rids):
        rollouts = {int(rid.split(":", 1)[0]) for rid in rids}
        if len(rollouts) != 1:
            raise ValueError("One SGLang batch mixes OWSM rollouts")
        (rollout_id,) = rollouts
        if rollout_id == self.rollout_id:
            return
        encoded = self.source_path.read_bytes()
        record = json.loads(encoded)
        if record["rollout_id"] != rollout_id:
            raise ValueError("OWSM rollout sources belong to a different rollout")
        self.rollout_id, self.sources = rollout_id, record["sources"]
        self.source_sha256 = hashlib.sha256(encoded).hexdigest()
        self.memory.clear()
        self.request_prefixes.clear()

    def _memory(self, source_index, device):
        if source_index not in self.memory:
            if not 0 <= source_index < len(self.sources):
                raise ValueError("OWSM request names a source outside its rollout")
            audio = load_audio(
                self.sources[source_index],
                self.policy.sample_rate,
                self.policy.window_samples,
            )
            speech = torch.as_tensor(
                audio, dtype=torch.float32, device=device
            ).unsqueeze(0)
            length = torch.tensor([speech.size(1)], device=device)
            memory, lengths = self.policy._encode(speech, length)
            self.memory[source_index] = (memory, lengths)
        return self.memory[source_index]

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        device = next(self.policy.parameters()).device
        pool_ids = forward_batch.req_pool_indices.tolist()
        if forward_batch.forward_mode.is_decode():
            chunks = list(input_ids.split(1))
        elif forward_batch.forward_mode.is_extend():
            lengths = forward_batch.extend_seq_lens_cpu
            prefixes = forward_batch.extend_prefix_lens_cpu
            if any(prefixes):
                raise ValueError(
                    "OWSM native full-prefix contract requires disabled radix/prefix cache"
                )
            chunks = list(input_ids.split(lengths))
        elif forward_batch.forward_mode.is_idle():
            return LogitsProcessorOutput(
                next_token_logits=torch.empty(0, self.config.vocab_size, device=device)
            )
        else:
            raise ValueError("Unsupported SGLang forward mode for native OWSM")
        self._load_sources(forward_batch.rids)
        rows = []
        for pool_id, rid, chunk in zip(
            pool_ids, forward_batch.rids, chunks, strict=True
        ):
            if forward_batch.forward_mode.is_decode():
                if pool_id not in self.request_prefixes:
                    raise RuntimeError(
                        "SGLang decode request is missing native prefill"
                    )
                source_index, prefix = self.request_prefixes[pool_id]
                prefix = [*prefix, *chunk.tolist()]
            else:
                source_index, prefix = int(rid.split(":")[1]), chunk.tolist()
                if not prefix:
                    raise ValueError("OWSM native decoder prefix is empty")
            self.request_prefixes[pool_id] = (source_index, prefix)
            rows.append((source_index, prefix))
        audit = (
            not forward_batch.forward_mode.is_decode() and forward_batch.return_logprob
        )
        if audit and any(len(prefix) < 2 for _, prefix in rows):
            raise ValueError(
                "OWSM prefix probability audit requires at least two native prompt tokens"
            )
        # Every running request goes through the native decoder together,
        # padded exactly as the native scorer pads its candidates. The decoder
        # still recomputes each full prefix; batching only removes the
        # per-request launches. Rows are grouped so the full-vocabulary logits
        # of one call stay within a fixed memory budget.
        longest = max(len(prefix) for _, prefix in rows)
        per_row = longest * self.config.vocab_size * 4
        width = max(1, LOGITS_BUDGET_BYTES // per_row)
        outputs, prefix_logprobs = [], []
        for start in range(0, len(rows), width):
            group = rows[start : start + width]
            memories = [self._memory(source, device) for source, _ in group]
            memory = pad_sequence([m[0] for m, _ in memories], batch_first=True)
            memory_lengths = torch.cat([lengths for _, lengths in memories])
            tokens = pad_sequence(
                [torch.tensor(p, dtype=torch.long, device=device) for _, p in group],
                batch_first=True,
                padding_value=self.policy.model.eos,
            )
            token_lengths = torch.tensor([len(p) for _, p in group], device=device)
            logits, _ = self.policy.model.decoder(
                memory, memory_lengths, tokens, token_lengths
            )
            index = torch.arange(len(group), device=device)
            last = token_lengths - 1
            outputs.append(logits[index, last].float())
            if audit:
                # logprob_start_len is deliberately the final prompt position,
                # so SGLang returns the real final native prompt token probability.
                previous = torch.log_softmax(logits[index, last - 1].float(), -1)
                prefix_logprobs.append(previous[index, tokens[index, last]])
        result = LogitsProcessorOutput(next_token_logits=torch.cat(outputs))
        if audit:
            result.input_token_logprobs = torch.cat(prefix_logprobs)
        result.customized_info = {
            "owsm_source_sha256": [self.source_sha256] * len(pool_ids)
        }
        return result
