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

# Families whose rollout engine samples, and whose learner scores, every action
# at the recipe's temperature (README, "Sampling temperature").
TEMPERATURE_FAMILIES = ("owsm", "bagpiper")


@dataclass
class ESPnetRLConfig:
    family: str = "owsm"
    execution_mode: str = "sync"
    rollout_backend: str = "sglang"
    # Miles selects CUDA IPC/NCCL; miles_shm uses collocated CPU shared memory.
    weight_sync_backend: str = "disk"
    profile_phases: bool = False
    sglang_source: str = ""
    sglang_engine_kwargs: dict[str, Any] = field(default_factory=dict)
    # module:Class of the architecture's espnet2.rl.engine.RolloutEngine.
    sglang_engine: str | None = None
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
    # Score the frozen reference on its GPU, then keep its weights on CPU
    # between passes so actor backward can reuse that GPU memory.
    reference_cpu_offload: bool = False
    optimizer_device: str | None = None
    distributed_strategy: str = "ddp"
    # Cover the full collocate runtime budget, including rank-zero rollout waits.
    distributed_timeout_minutes: float = 72 * 60
    allreduce_chunk_bytes: int = 64 * 1024 * 1024
    dtype: str = "float32"
    seed: int = 42
    steps: int = 10
    warmup_steps: int = 0
    # Write a full checkpoint every save_interval updates and after the last.
    save_interval: int = 1
    # Rollout exports kept on disk: once every engine has loaded a policy, all
    # but the newest keep_exports are deleted.
    keep_exports: int = 2
    # Prompts drawn per optimizer update and candidates sampled per prompt.
    prompts_per_step: int = 2
    group_size: int = 4
    update_weights_interval: int = 1
    max_weight_staleness: int = 1
    tis_clip_low: float = 0.5
    tis_clip: float = 2.0
    micro_batch_size: int = 1
    # With a token budget, micro-batches take candidates in order of length
    # and hold at most micro_batch_tokens right-padded prompt+response tokens.
    micro_batch_tokens: int | None = None
    ppo_epochs: int = 1
    # The default retains the pinned Miles PPO/IcePop objective. WEST uses
    # on-policy GRPO and gives every completed answer equal loss weight.
    training_objective: str = "miles_ppo"
    learning_rate: float = 1e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    shuffle_train_data: bool = False
    data_seed: int = 42
    resume_checkpoint: str = ""
    periodic_eval_interval: int = 0
    periodic_eval_function: str = ""
    periodic_eval_options: dict[str, Any] = field(default_factory=dict)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    kl_type: str = "k3"
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    loss_agg_mode: str = "token-mean"
    max_tokens: int = 256
    # Actions are sampled from softmax(logits / temperature) over their whole
    # support; every log probability is that distribution's.
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    # Drift gate over the sampled actions' native/rollout logprob differences:
    # The maximum is optional only for synchronous WEST training with a
    # positive mean bound. All finite statistics remain recorded and checked.
    score_tolerance: float | None = 0.03
    score_mean_tolerance: float | None = None
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
        if type(self.profile_phases) is not bool:
            raise ValueError("profile_phases must be a boolean")
        if self.weight_sync_backend not in ("disk", "miles", "miles_shm"):
            raise ValueError("weight_sync_backend must be disk, miles or miles_shm")
        if self.weight_sync_backend != "disk" and (
            self.family != "bagpiper"
            or self.rollout_backend != "sglang"
            or self.train_ranks != 1
            or self.rollout_replicas != 1
        ):
            raise ValueError(
                "Miles weight transfer requires Bagpiper SGLang with one training "
                "rank and one rollout replica"
            )
        if self.weight_sync_backend == "miles_shm" and (
            self.execution_mode != "sync" or not self.collocated
        ):
            raise ValueError("miles_shm requires collocated synchronous training")
        if self.execution_mode not in ("sync", "disagg_sync", "disagg_async"):
            raise ValueError("execution_mode must be sync, disagg_sync or disagg_async")
        if (
            self.execution_mode == "disagg_sync"
            and self.training_objective != "west_grpo"
        ):
            raise ValueError("disagg_sync requires the versioned WEST GRPO objective")
        if type(self.reference_cpu_offload) is not bool:
            raise ValueError("reference_cpu_offload must be a boolean")
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
        # IcePop corrects the policy-gradient term in both modes.
        if not 0 < self.tis_clip_low <= 1 <= self.tis_clip:
            raise ValueError("Invalid IcePop importance-ratio bounds")
        for name in ("train_gpus", "rollout_gpus"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a comma-separated string")
            devices = self._devices(value)
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
            "save_interval",
            "prompts_per_step",
            "micro_batch_size",
            "ppo_epochs",
            "max_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.micro_batch_tokens is not None and self.micro_batch_tokens < 1:
            raise ValueError("micro_batch_tokens must be positive")
        if (
            type(self.temperature) not in (int, float)
            or not 0 < self.temperature <= 1
            or self.top_p != 1.0
            or self.top_k
            not in ((-1, 50) if self.training_objective == "west_grpo" else (-1,))
        ):
            raise ValueError(
                "RL samples every action from its whole support at a temperature in "
                "(0, 1]: top_p=1, top_k=-1. The versioned WEST objective also "
                "supports top_k=50 with separately audited sampling probabilities."
            )
        if self.rollout_backend not in ("sglang", "vllm"):
            raise ValueError("rollout_backend must be sglang or vllm")
        if self.temperature != 1 and (
            self.family not in TEMPERATURE_FAMILIES or self.rollout_backend != "sglang"
        ):
            # vLLM's frozen raw_logprobs are taken before its temperature.
            raise ValueError(
                f"{self.family} on {self.rollout_backend} samples at temperature 1 "
                "only: its engine or learner does not apply another"
            )
        if not self.adapter_factory or not self.exporter:
            raise ValueError(
                "Recipe must supply its ESPnet adapter_factory and exporter"
            )
        if self.rollout_backend == "sglang" and not self.sglang_engine:
            raise ValueError("SGLang requires an architecture rollout engine")
        if self.rollout_backend == "vllm" and not self.vllm_generate_factory:
            raise ValueError("vLLM requires an architecture generation factory")
        if self.kl_type not in ("k1", "k2", "k3", "low_var_kl"):
            raise ValueError("Unsupported Miles KL estimator")
        if self.training_objective not in ("miles_ppo", "west_grpo"):
            raise ValueError("training_objective must be miles_ppo or west_grpo")
        if self.training_objective == "west_grpo":
            if (
                self.execution_mode not in ("sync", "disagg_sync")
                or self.ppo_epochs != 1
            ):
                raise ValueError("WEST GRPO requires one synchronous on-policy update")
            if self.update_weights_interval != 1:
                raise ValueError(
                    "WEST GRPO publishes the current policy before each rollout"
                )
            if self.loss_agg_mode != "seq-mean-token-mean":
                raise ValueError("WEST GRPO requires per-answer token-mean reduction")
            if self.kl_type != "k3" or not self.normalize_advantage:
                raise ValueError("WEST GRPO requires k3 KL and normalized advantages")
            if self.entropy_coef != 0:
                raise ValueError("WEST GRPO does not add an entropy loss")
        elif self.loss_agg_mode != "token-mean":
            raise ValueError(
                "Initial native trainer requires global token-mean reduction"
            )
        if self.lr_scheduler not in ("constant", "linear"):
            raise ValueError("lr_scheduler must be constant or linear")
        if type(self.lr_warmup_steps) is not int or not (
            0 <= self.lr_warmup_steps < self.steps
        ):
            raise ValueError("lr_warmup_steps must be within the total update budget")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("Adam beta values must lie in [0, 1)")
        if not 0 < self.adam_epsilon < float("inf"):
            raise ValueError("adam_epsilon must be positive and finite")
        if type(self.shuffle_train_data) is not bool or type(self.data_seed) is not int:
            raise ValueError(
                "shuffle_train_data must be bool and data_seed must be int"
            )
        if (
            type(self.periodic_eval_interval) is not int
            or self.periodic_eval_interval < 0
        ):
            raise ValueError("periodic_eval_interval must be a nonnegative integer")
        if bool(self.periodic_eval_interval) != bool(self.periodic_eval_function):
            raise ValueError("Periodic evaluation requires both interval and function")
        if (
            self.periodic_eval_interval
            and self.save_interval > self.periodic_eval_interval
        ):
            raise ValueError(
                "Periodic evaluation requires a checkpoint at each boundary"
            )
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.optimizer_device not in (None, "cpu"):
            raise ValueError("optimizer_device must be null or cpu")
        if self.dtype != "float32" and self.optimizer_device != "cpu":
            # Only the CPU optimizer keeps FP32 master weights.
            raise ValueError("A low-precision actor requires optimizer_device: cpu")
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
        if self.score_tolerance is None:
            if self.training_objective != "west_grpo" or self.execution_mode not in (
                "sync",
                "disagg_sync",
            ):
                raise ValueError(
                    "score_tolerance must be finite and nonnegative outside "
                    "synchronous WEST training"
                )
            mean = self.score_mean_tolerance
            if type(mean) not in (int, float) or not 0 < mean < float("inf"):
                raise ValueError(
                    "A mean-only WEST guard requires score_mean_tolerance "
                    "to be finite and positive"
                )
        for name in ("score_tolerance", "score_mean_tolerance"):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.keep_exports) is not int or self.keep_exports < 1:
            raise ValueError("keep_exports must be a positive integer")

    @classmethod
    def load(cls, path: str):
        if Path(path).suffix.lower() == ".json":
            with Path(path).open() as stream:
                config = cls.from_dict(json.load(stream))
            config.validate()
            return config
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: str):
        with Path(path).open() as stream:
            config = cls.from_dict(yaml.safe_load(stream))
        config.validate()
        return config

    @classmethod
    def from_dict(cls, values):
        if "sglang_generate_factory" in values:
            raise ValueError(
                "sglang_generate_factory was replaced by sglang_engine, the "
                "module:Class of the family's espnet2.rl.engine.RolloutEngine"
            )
        return cls(**values)

    def to_dict(self):
        return asdict(self)
