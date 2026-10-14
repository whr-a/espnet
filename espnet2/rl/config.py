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

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import get_device_name


@dataclass
class ESPnetRLConfig:
    family: str = "owsm"
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
    dtype: str = "float32"
    seed: int = 42
    steps: int = 10
    prompts_per_step: int = 2
    group_size: int = 4
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
    rollout_gpus: str = "6,7"
    vllm_source: str = ""
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.8
    vllm_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    # Additional families supply real ESPnet scorer/exporter/request factory.
    adapter_factory: str | None = None
    exporter: str | None = None
    request_factory: str | None = None
    reward_function: str | None = None
    adapter_options: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        for name in ("steps", "prompts_per_step", "micro_batch_size", "ppo_epochs", "max_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.temperature != 1.0 or self.top_p != 1.0 or self.top_k != -1:
            raise ValueError(
                "The first native backend requires temperature=1, top_p=1, top_k=-1 so vLLM raw "
                "logprobs match the native actor distribution. Truncated sampling requires a separate "
                "explicit behavior-policy implementation."
            )
        if self.rollout_backend not in ("sglang", "vllm"):
            raise ValueError("rollout_backend must be sglang or vllm")
        if not self.adapter_factory or not self.exporter:
            raise ValueError("Recipe must supply its ESPnet adapter_factory and exporter")
        if self.rollout_backend == "sglang" and not self.sglang_generate_factory:
            raise ValueError("SGLang requires an architecture generation factory")
        if self.kl_type not in ("k1", "k2", "k3", "low_var_kl"):
            raise ValueError("Unsupported Miles KL estimator")
        if self.loss_agg_mode != "token-mean":
            raise ValueError("Initial native trainer requires global token-mean reduction")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.optimizer_device not in (None, "cpu"):
            raise ValueError("optimizer_device must be null or cpu")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.kl_coef < 0:
            raise ValueError("Invalid optimizer or KL settings")
        if self.score_tolerance < 0:
            raise ValueError("score_tolerance must be nonnegative")

    @classmethod
    def from_yaml(cls, path: str):
        with Path(path).open() as stream:
            config = cls(**yaml.safe_load(stream))
        config.validate()
        return config

    def to_dict(self):
        return asdict(self)
