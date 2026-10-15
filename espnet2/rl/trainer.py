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
"""Synchronous GRPO with native ESPnet torch/DDP and pinned Miles algorithms."""

import copy
import json
import os
import random
import traceback
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from .data import collate_rollouts, read_manifest
from .miles_library import (
    agg_loss,
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
    kl_penalty,
    verify_miles,
)
from .parallel import average_native_gradients, synchronize_native_model
from .parallel_optimizer import CPUOffloadedAdamW
from .rewards import outcome_rewards, word_error_reward
from .rollout import CheckpointRollout, save_checkpoint
from .types import validate_token_scores
from .utils import build_policy, get_device_name, get_nccl_backend, get_torch_device


def clone_reference(policy, device=None):
    """Copy native policy tensors directly to the reference scoring device.

    Seeding deepcopy's memo preserves shared weights and ordinary model
    structure without first allocating a complete second actor on its GPU.
    Tensor attributes such as ESPnet's unregistered positional tables are
    included along with parameters, buffers and tensor containers.
    """
    if device is None:
        return copy.deepcopy(policy)
    memo, visited = {}, set()

    def seed(value):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=device, copy=True)
            memo[identity] = (
                torch.nn.Parameter(tensor, requires_grad=value.requires_grad)
                if isinstance(value, torch.nn.Parameter)
                else tensor
            )
        elif isinstance(value, torch.nn.Module):
            seed(vars(value))
        elif isinstance(value, dict):
            for item in value.values():
                seed(item)
        elif isinstance(value, list | tuple | set):
            for item in value:
                seed(item)

    seed(policy)
    return copy.deepcopy(policy, memo).to(device)


class NativeGRPOTrainer:
    def __init__(self, policy, config, reference=None, rollout=None):
        config.validate()
        self.config = config
        self.rank = distributed.get_rank() if distributed.is_initialized() else 0
        self.world_size = (
            distributed.get_world_size() if distributed.is_initialized() else 1
        )
        self.policy = policy.eval()
        self.device = next(policy.parameters()).device
        if self.world_size > 1:
            # Synchronize buffers once as well. DDP per-forward buffer sync is
            # disabled because checkpoint-owned native buffers can reside on CPU.
            synchronize_native_model(policy, chunk_bytes=config.allreduce_chunk_bytes)
        if self.world_size > 1 and config.distributed_strategy == "ddp":
            device_ids = [self.device.index] if self.device.type != "cpu" else None
            self.actor = DistributedDataParallel(
                policy,
                device_ids=device_ids,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                # Eval-mode native buffers are fixed checkpoint tensors. Do not
                # broadcast them at every microbatch or mix CPU/GPU buffers.
                broadcast_buffers=False,
                init_sync=False,
            )
        else:
            self.actor = policy
        # Replica synchronization must precede reference cloning: all ranks reference
        # the same initial actor, including when loading consumed local RNG.
        self.reference = (
            reference
            if reference is not None
            else clone_reference(policy, config.reference_device)
        ).eval()
        if config.reference_device is not None:
            self.reference = self.reference.to(config.reference_device)
        self.reference.requires_grad_(False)
        optimizer_class = (
            CPUOffloadedAdamW
            if config.optimizer_device == "cpu" and self.world_size > 1
            else torch.optim.AdamW
        )
        self.optimizer = optimizer_class(
            [p for p in policy.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **(
                {"chunk_bytes": config.allreduce_chunk_bytes}
                if optimizer_class is CPUOffloadedAdamW
                else {}
            ),
        )
        self.rollout = rollout
        from types import SimpleNamespace

        self.actor_config = SimpleNamespace(
            clip_ratio_low=config.clip_ratio,
            clip_ratio_high=config.clip_ratio,
            tis_clip_low=config.tis_clip_low,
            tis_clip=config.tis_clip,
            global_batch_info={},
        )

    def _score_chunks(self, policy, batch):
        results = []
        scoring_device = next(policy.parameters()).device
        for start in range(0, len(batch), self.config.micro_batch_size):
            subset = batch.select_idxs(
                np.arange(start, min(len(batch), start + self.config.micro_batch_size))
            )
            subset.to(scoring_device)
            scores = policy(subset)
            validate_token_scores(scores, subset.batch["response_mask"])
            results.append(scores.log_probs.detach().cpu())
        return torch.cat(results)

    def prepare_batch(self, batch):
        with torch.no_grad():
            old = self._score_chunks(self.policy, batch)
            reference = self._score_chunks(self.reference, batch)
        mask = batch.batch["response_mask"]
        if old.shape != mask.shape or reference.shape != mask.shape:
            raise ValueError(
                "The ESPnet scorer must return exactly one probability per action"
            )
        drift = ((old - batch.batch["rollout_log_probs"]).abs() * mask).max().item()
        if not np.isfinite(drift) or drift > self.config.score_tolerance:
            raise RuntimeError(
                f"Rollout/native logprob disagreement {drift:.6g} > {self.config.score_tolerance}; "
                "check checkpoint version, prefix, audio frontend and sampling distribution"
            )
        scores, token_rewards = outcome_rewards(batch, self.config.reward_function)
        advantages, returns = compute_grpo_outcome_advantage(
            token_rewards,
            mask,
            batch.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=self.config.normalize_advantage,
        )
        batch.batch.update(
            {
                "old_log_probs": old,
                "ref_log_probs": reference,
                "advantages": advantages,
                "returns": returns,
                "token_level_rewards": token_rewards,
            }
        )
        return {
            "reward/mean": scores.mean().item(),
            "reward/min": scores.min().item(),
            "reward/max": scores.max().item(),
            "rollout/logprob_max_error": drift,
        }

    def prepare_async_batch(
        self,
        batch,
        behavior_policy=None,
        *,
        behavior_version,
        learner_version,
        behavior_checkpoint_sha256,
    ):
        """Audit the behavior actor independently of the current learner.

        Native/engine numerical agreement uses the exact generating checkpoint.
        The expected one-update policy lag is handled by Miles IcePop in the
        policy-gradient loss, rather than weakening the numerical audit.
        """
        if self.config.execution_mode != "disagg_async":
            raise ValueError("Async batch preparation requires disagg_async mode")
        if (
            type(behavior_version) is not int
            or type(learner_version) is not int
            or not 0
            <= learner_version - behavior_version
            <= self.config.max_weight_staleness
            or behavior_version < 0
        ):
            raise ValueError("Invalid or excessive behavior policy staleness")
        if len(behavior_checkpoint_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in behavior_checkpoint_sha256
        ):
            raise ValueError(
                "The async batch requires an exact behavior checkpoint SHA256"
            )
        if len(batch) != self.config.global_candidate_batch_size:
            raise ValueError(
                "Async batch does not contain the full global candidate count"
            )
        identities = batch.non_tensor_batch["uid"].tolist()
        size = self.config.group_size
        groups = [identities[i : i + size] for i in range(0, len(batch), size)]
        if (
            any(len(set(group)) != 1 for group in groups)
            or len({group[0] for group in groups}) != self.config.prompts_per_step
        ):
            raise ValueError("Async batches must contain complete unique prompt groups")
        for generation in batch.non_tensor_batch["generation"]:
            if (
                generation.get("behavior_policy_version") != behavior_version
                or generation.get("behavior_checkpoint_sha256")
                != behavior_checkpoint_sha256
            ):
                raise ValueError(
                    "Mixed or unrecorded behavior checkpoint in async batch"
                )
        if behavior_version == learner_version:
            behavior_policy = self.policy
        elif behavior_policy is None:
            raise ValueError(
                "Stale rollout requires the independently loaded behavior policy"
            )
        elif behavior_policy is self.policy:
            raise ValueError("Stale rollout cannot be audited with the current actor")
        with torch.no_grad():
            old = self._score_chunks(self.policy, batch)
            behavior = (
                old
                if behavior_policy is self.policy
                else self._score_chunks(behavior_policy, batch)
            )
            reference = self._score_chunks(self.reference, batch)
        mask = batch.batch["response_mask"]
        for scores in (old, behavior, reference):
            if scores.shape != mask.shape:
                raise ValueError(
                    "Native async scores must preserve every sampled action"
                )
        numerical_error = (
            ((behavior - batch.batch["rollout_log_probs"]).abs() * mask).max().item()
        )
        if (
            not np.isfinite(numerical_error)
            or numerical_error > self.config.score_tolerance
        ):
            raise RuntimeError(
                f"Behavior native/SGLang logprob disagreement {numerical_error:.6g} "
                f"> {self.config.score_tolerance}; the generating actor must match"
            )
        scores, token_rewards = outcome_rewards(batch, self.config.reward_function)
        advantages, returns = compute_grpo_outcome_advantage(
            token_rewards,
            mask,
            batch.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=self.config.normalize_advantage,
        )
        batch.batch.update(
            old_log_probs=old,
            ref_log_probs=reference,
            behavior_native_log_probs=behavior,
            advantages=advantages,
            returns=returns,
            token_level_rewards=token_rewards,
        )
        return {
            "reward/mean": scores.mean().item(),
            "reward/min": scores.min().item(),
            "reward/max": scores.max().item(),
            "rollout/logprob_max_error": numerical_error,
            "rollout/current_behavior_logprob_max_difference": (
                (old - behavior).abs() * mask
            )
            .max()
            .item(),
            "rollout/behavior_policy_version": behavior_version,
            "rollout/learner_policy_version": learner_version,
            "rollout/weight_staleness": learner_version - behavior_version,
            "rollout/global_prompt_batch_size": len(groups),
            "rollout/global_candidate_batch_size": len(batch),
        }

    def update_actor(self, batch):
        # Each group baseline is computed globally before DDP sharding.
        if len(batch) % self.world_size:
            raise ValueError("Global rollout count must be divisible by DDP world_size")
        indices = np.arange(len(batch)).reshape(self.world_size, -1)[self.rank]
        local = batch.select_idxs(indices)
        denominator = int(batch.batch["response_mask"].sum())
        self.actor_config.global_batch_info.update(
            {
                "dp_size": self.world_size,
                "batch_num_tokens": denominator,
                "global_batch_size": len(batch),
            }
        )
        if denominator == 0:
            raise ValueError("No valid sampled actions")
        metrics = {"actor/loss": 0.0, "actor/kl_loss": 0.0, "actor/entropy": 0.0}
        for _ in range(self.config.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(local), self.config.micro_batch_size):
                part = local.select_idxs(
                    np.arange(
                        start, min(len(local), start + self.config.micro_batch_size)
                    )
                )
                part.to(self.device)
                final_microbatch = start + self.config.micro_batch_size >= len(local)
                synchronization = (
                    self.actor.no_sync()
                    if isinstance(self.actor, DistributedDataParallel)
                    and not final_microbatch
                    else nullcontext()
                )
                # no_sync must include forward and backward. One reduction of
                # accumulated gradients replaces one full-model reduction per
                # candidate without changing the global token denominator.
                with synchronization:
                    current = self.actor(part)
                    mask = part.batch["response_mask"]
                    validate_token_scores(current, mask)
                    loss, ppo_metrics = compute_policy_loss_vanilla(
                        part.batch["old_log_probs"],
                        current.log_probs,
                        part.batch["advantages"],
                        mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                        config=self.actor_config,
                        rollout_log_probs=(
                            part.batch["rollout_log_probs"]
                            if self.config.execution_mode == "disagg_async"
                            else None
                        ),
                    )
                    kl = agg_loss(
                        kl_penalty(
                            current.log_probs,
                            part.batch["ref_log_probs"],
                            self.config.kl_type,
                        ),
                        mask,
                        self.config.loss_agg_mode,
                        **self.actor_config.global_batch_info,
                    )
                    entropy = agg_loss(
                        current.entropy,
                        mask,
                        self.config.loss_agg_mode,
                        **self.actor_config.global_batch_info,
                    )
                    total = (
                        loss
                        + self.config.kl_coef * kl
                        - self.config.entropy_coef * entropy
                    )
                    if not torch.isfinite(total):
                        raise FloatingPointError("Nonfinite policy loss")
                    total.backward()
                metrics["actor/loss"] += total.detach().item() / self.config.ppo_epochs
                metrics["actor/kl_loss"] += kl.detach().item() / self.config.ppo_epochs
                metrics["actor/entropy"] += (
                    entropy.detach().item() / self.config.ppo_epochs
                )
                token_weight = (
                    mask.sum().item()
                    * self.world_size
                    / denominator
                    / self.config.ppo_epochs
                )
                for name, value in ppo_metrics.items():
                    metrics[name] = metrics.get(name, 0.0) + value * token_weight
            if (
                self.world_size > 1
                and self.config.distributed_strategy == "native_allreduce"
            ):
                average_native_gradients(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    chunk_bytes=self.config.allreduce_chunk_bytes,
                )
            norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            if not torch.isfinite(norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Nonfinite gradient norm; optimizer was not stepped"
                )
            self._optimizer_step()
            metrics["actor/grad_norm"] = norm.item()
        if self.world_size > 1:
            keys = sorted(metrics)
            values = torch.tensor([metrics[key] for key in keys], device=self.device)
            distributed.all_reduce(values)
            values /= self.world_size
            metrics = dict(zip(keys, values.cpu().tolist(), strict=True))
        return metrics

    def _optimizer_step(self):
        """Keep Adam state on CPU while preserving the native parameter objects.

        Offload follows backward and gradient clipping. Multi-rank actors use
        owner CPU shadows, keeping their original devices and storage. The
        legacy single-rank branch temporarily moves optimizer-owned tensors;
        it restores devices before scoring, exporting or the next update.
        """
        if self.config.optimizer_device is None or isinstance(
            self.optimizer, CPUOffloadedAdamW
        ):
            self.optimizer.step()
            return
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        devices = [parameter.device for parameter in parameters]
        try:
            for parameter in parameters:
                gradient = parameter.grad
                parameter.data = parameter.detach().to("cpu")
                if gradient is not None:
                    parameter.grad = gradient.detach().to("cpu")
            self.optimizer.step()
        finally:
            for parameter, device in zip(parameters, devices, strict=True):
                parameter.grad = None
                parameter.data = parameter.detach().to(device)

    def train(self, sources):
        if self.config.execution_mode != "sync":
            raise ValueError("Use the Miles async driver for disagg_async training")
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.rollout is None and self.rank == 0:
            self.rollout = CheckpointRollout(self.config)
        if len({source["id"] for source in sources}) < self.config.prompts_per_step:
            raise ValueError("The prompt batch requires distinct source records")
        for step in range(self.config.steps):
            step_start = time.monotonic()
            if self.rank == 0:
                try:
                    chosen = [
                        sources[
                            (step * self.config.prompts_per_step + i) % len(sources)
                        ]
                        for i in range(self.config.prompts_per_step)
                    ]
                    self.rollout.sync_weights(self.policy, step)
                    generations = self.rollout.generate(chosen, step)
                    batch = collate_rollouts(
                        chosen, generations, self.config.group_size
                    )
                    metrics = self.prepare_batch(batch)
                    batch.save_to_disk(output / f"rollout-{step:06d}.pkl")
                    payload = [batch, metrics, None]
                except Exception as error:
                    (output / f"rollout-{step:06d}.error.txt").write_text(
                        traceback.format_exc()
                    )
                    payload = [None, None, f"{type(error).__name__}: {error}"]
            else:
                payload = [None, None, None]
            if self.world_size > 1:
                distributed.broadcast_object_list(payload, src=0)
            batch, metrics, error = payload
            if error:
                raise RuntimeError(
                    f"Rollout/score stage failed at step {step}: {error}"
                )
            update_start = time.monotonic()
            metrics.update(self.update_actor(batch))
            if self.device.type != "cpu":
                get_torch_device().synchronize()
            update_end = time.monotonic()
            status = [None]
            if self.rank == 0:
                try:
                    metrics["step"] = step + 1
                    elapsed, wait = update_end - update_start, update_start - step_start
                    metrics.update(
                        {
                            "perf/train_time": elapsed,
                            "perf/train_wait_time": wait,
                            "perf/step_time": elapsed + wait,
                            "perf/wait_time_ratio": wait / (elapsed + wait),
                            "perf/warmup": step < self.config.warmup_steps,
                            "runtime/train_world_size": self.world_size,
                            "runtime/rollout_replicas": self.config.rollout_num_replicas,
                        }
                    )
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(metrics) + "\n")
                    print(json.dumps(metrics), flush=True)
                    save_checkpoint(
                        output / f"step-{step + 1:06d}.pt",
                        self.policy,
                        self.optimizer,
                        step + 1,
                        self.config,
                    )
                except Exception as error:
                    (output / f"checkpoint-{step + 1:06d}.error.txt").write_text(
                        traceback.format_exc()
                    )
                    status[0] = f"{type(error).__name__}: {error}"
            if self.world_size > 1:
                distributed.broadcast_object_list(status, src=0)
            if status[0]:
                raise RuntimeError(
                    f"Checkpoint stage failed at step {step + 1}: {status[0]}"
                )


def run(config):
    if config.dtype == "float32":
        # Actor and rollout must use the same declared full FP32 precision.
        # Rollout sets Triton's independent tl.dot option before child imports.
        backend = getattr(torch.backends, get_device_name(), None)
        if backend is not None and hasattr(backend, "matmul"):
            backend.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config.validate()
    miles_evidence = verify_miles(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        device_type = torch.device(config.device).type
        distributed.init_process_group(
            get_nccl_backend() if device_type != "cpu" else "gloo"
        )
        if device_type != "cpu":
            get_torch_device().set_device(int(os.environ["LOCAL_RANK"]))
            config.device = f"{device_type}:{os.environ['LOCAL_RANK']}"
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    output = Path(config.output_dir).resolve()
    config.output_dir = str(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        frozen = output / "frozen_config.json"
        if frozen.exists():
            raise FileExistsError(
                f"Use a fresh output_dir; refusing to change frozen config {frozen}"
            )
        frozen.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (output / "miles_library.json").write_text(
            json.dumps(miles_evidence, indent=2) + "\n"
        )
    if distributed.is_initialized():
        distributed.barrier()
    sources = read_manifest(config.manifest)
    # Reward implementations read the same effective configuration as the
    # actor and rollout, including command-line overrides. This path is host
    # metadata and does not become part of the model's prompt.
    for source in sources:
        source["reward_config"] = str(output / "frozen_config.json")
    if config.reward_function is None:
        # Reject missing supervision or an audio-only reward configuration
        # before loading models or occupying inference GPUs.
        for source in sources:
            word_error_reward(source, {"text": ""})
    policy = build_policy(config, config.device)
    trainer = NativeGRPOTrainer(policy, config)
    try:
        trainer.train(sources)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
