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
"""Configuration for the native torch ESPnet/Miles backend."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import get_device_name


@dataclass
class ESPnetRLConfig:
    family: str = "owsm"
    execution_mode: str = "sync"
    rollout_backend: str = "sglang"
    sglang_source: str = ""
    sglang_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    sglang_generate_factory: str | None = None
    miles_source: str = ""
    miles_commit: str = "a22fd824191c64ba7c5d6c531f125406e4ce9cc6"
    eval_manifest: str | None = None
    train_config: str = ""
    checkpoint: str = ""
    manifest: str = ""
    output_dir: str = "outputs/espnet-grpo"
    model_version: str | None = "v4"
    device: str = field(default_factory=get_device_name)
    reference_device: str | None = None
    optimizer_device: str | None = None
    distributed_strategy: str = "ddp"
    # Cover the full collocate runtime budget, including rank-zero rollout waits.
    distributed_timeout_minutes: float = 72 * 60
    allreduce_chunk_bytes: int = 64 * 1024 * 1024
    dtype: str = "float32"
    seed: int = 42
    steps: int = 10
    warmup_steps: int = 0
    # Prompts drawn per optimizer update and candidates sampled per prompt.
    prompts_per_step: int = 2
    group_size: int = 4
    update_weights_interval: int = 1
    max_weight_staleness: int = 1
    tis_clip_low: float = 0.5
    tis_clip: float = 2.0
    micro_batch_size: int = 1
    ppo_epochs: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    kl_type: str = "k3"
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    loss_agg_mode: str = "token-mean"
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    score_tolerance: float = 0.03
    language: str = "<eng>"
    task: str = "<asr>"
    predict_timestamps: bool = False
    train_encoder: bool = True
    rollout_python: str = "python"
    # Physical GPU ids per role. One training rank per entry in train_gpus;
    # collocate by naming the same device in both lists.
    train_gpus: str = "0"
    rollout_gpus: str = "0"
    rollout_mem_fraction: float | None = None
    vllm_source: str = ""
    vllm_tensor_parallel_size: int = 1
    vllm_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    # Additional families supply real ESPnet scorer/exporter/request factory.
    adapter_factory: str | None = None
    exporter: str | None = None
    vllm_generate_factory: str | None = None
    reward_function: str | None = None
    adapter_options: dict[str, Any] = field(default_factory=dict)

    @property
    def global_candidate_batch_size(self):
        return self.prompts_per_step * self.group_size

    @staticmethod
    def _devices(spec):
        return [item for item in str(spec).split(",") if item != ""]

    @property
    def train_device_ids(self):
        """Physical GPU ids of the training ranks, in rank order."""
        return self._devices(self.train_gpus)

    @property
    def rollout_device_ids(self):
        return self._devices(self.rollout_gpus)

    @property
    def train_ranks(self):
        """One training rank per declared training GPU."""
        return len(self.train_device_ids)

    @property
    def collocated(self):
        """Whether both roles declare a common device id.

        On one node that means the rollout engine shares a training GPU. Across
        nodes the ids belong to different hosts and nothing is shared.
        """
        return bool(set(self.train_device_ids) & set(self.rollout_device_ids))

    @property
    def rollout_replicas(self):
        """One SGLang replica per rollout GPU; vLLM uses one tensor-parallel engine."""
        if self.rollout_backend == "vllm":
            return 1
        return len(self.rollout_device_ids)

    def validate(self):
        if self.execution_mode not in ("sync", "disagg_async"):
            raise ValueError("execution_mode must be sync or disagg_async")
        if (
            type(self.warmup_steps) is not int
            or not 0 <= self.warmup_steps < self.steps
        ):
            raise ValueError("warmup_steps must be within the total update budget")
        if self.execution_mode == "disagg_async":
            if self.rollout_backend != "sglang":
                raise ValueError("Disaggregated async training requires SGLang")
            if self.update_weights_interval != 1 or self.max_weight_staleness != 1:
                raise ValueError(
                    "The one-step async pipeline requires weight interval and staleness 1"
                )
            if not 0 < self.tis_clip_low <= 1 <= self.tis_clip:
                raise ValueError("Invalid IcePop importance-ratio bounds")
        for name in ("train_gpus", "rollout_gpus"):
            devices = self._devices(getattr(self, name))
            if not devices or len(set(devices)) != len(devices):
                raise ValueError(f"{name} must list distinct physical GPU ids")
        if self.rollout_backend == "vllm" and (
            len(self.rollout_device_ids) != self.vllm_tensor_parallel_size
        ):
            raise ValueError("vLLM rollout_gpus must match vllm_tensor_parallel_size")
        if self.rollout_mem_fraction is not None and not (
            0 < self.rollout_mem_fraction < 1
        ):
            raise ValueError("rollout_mem_fraction must lie in (0, 1)")
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        for name in (
            "steps",
            "prompts_per_step",
            "micro_batch_size",
            "ppo_epochs",
            "max_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.temperature != 1.0 or self.top_p != 1.0 or self.top_k != -1:
            raise ValueError(
                "The first native backend requires temperature=1, top_p=1, top_k=-1 so engine raw "
                "logprobs match the native actor distribution. Truncated sampling requires a separate "
                "explicit behavior-policy implementation."
            )
        if self.rollout_backend not in ("sglang", "vllm"):
            raise ValueError("rollout_backend must be sglang or vllm")
        if not self.adapter_factory or not self.exporter:
            raise ValueError(
                "Recipe must supply its ESPnet adapter_factory and exporter"
            )
        if self.rollout_backend == "sglang" and not self.sglang_generate_factory:
            raise ValueError("SGLang requires an architecture generation factory")
        if self.rollout_backend == "vllm" and not self.vllm_generate_factory:
            raise ValueError("vLLM requires an architecture generation factory")
        if self.kl_type not in ("k1", "k2", "k3", "low_var_kl"):
            raise ValueError("Unsupported Miles KL estimator")
        if self.loss_agg_mode != "token-mean":
            raise ValueError(
                "Initial native trainer requires global token-mean reduction"
            )
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.optimizer_device not in (None, "cpu"):
            raise ValueError("optimizer_device must be null or cpu")
        if self.distributed_strategy not in ("ddp", "native_allreduce"):
            raise ValueError("distributed_strategy must be ddp or native_allreduce")
        if type(self.distributed_timeout_minutes) not in (
            int,
            float,
        ) or not 0 < self.distributed_timeout_minutes < float("inf"):
            raise ValueError("distributed_timeout_minutes must be positive and finite")
        if self.allreduce_chunk_bytes < 1:
            raise ValueError("allreduce_chunk_bytes must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.kl_coef < 0:
            raise ValueError("Invalid optimizer or KL settings")
        if self.score_tolerance < 0:
            raise ValueError("score_tolerance must be nonnegative")

    @classmethod
    def load(cls, path: str):
        if Path(path).suffix.lower() == ".json":
            with Path(path).open() as stream:
                config = cls(**json.load(stream))
            config.validate()
            return config
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: str):
        with Path(path).open() as stream:
            config = cls(**yaml.safe_load(stream))
        config.validate()
        return config

    def to_dict(self):
        return asdict(self)
