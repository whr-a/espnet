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
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
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
from .process_identity import host_identity
from .rewards import outcome_rewards, word_error_reward
from .rollout import CheckpointRollout, save_checkpoint
from .rollout_worker import engine_class
from .scoring import score_native_batch
from .timing import learner_timing
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
        if self.world_size > 1:
            self._object_collective_group()
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
            CPUOffloadedAdamW if config.optimizer_device == "cpu" else torch.optim.AdamW
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

    def _micro_batches(self, batch):
        """Candidate indices of each micro-batch of ``batch``.

        At most micro_batch_size candidates each. With micro_batch_tokens,
        candidates are taken in order of prompt+response length, and a
        micro-batch also keeps its count times its longest length (its
        right-padded size) within that budget.
        """
        size, budget = self.config.micro_batch_size, self.config.micro_batch_tokens
        if budget is None:
            return [
                np.arange(start, min(len(batch), start + size))
                for start in range(0, len(batch), size)
            ]
        lengths = [
            len(prompt) + length
            for prompt, length in zip(
                batch.non_tensor_batch["prompt_token_ids"],
                batch.batch["response_lengths"].tolist(),
                strict=True,
            )
        ]
        parts, part = [], []
        for index in sorted(range(len(batch)), key=lengths.__getitem__):
            if part and (
                len(part) == size or (len(part) + 1) * lengths[index] > budget
            ):
                parts.append(np.array(part))
                part = []
            part.append(index)
        if part:
            parts.append(np.array(part))
        return parts

    def _score_chunks(self, policy, batch):
        if isinstance(policy, AbstractContextManager):
            # A weight snapshot lends its values to a module for this pass.
            with policy as module:
                return self._score_chunks(module, batch)
        results = torch.zeros(batch.batch["response_mask"].shape)
        scoring_device = next(policy.parameters()).device
        for indices in self._micro_batches(batch):
            subset = batch.select_idxs(indices)
            subset.to(scoring_device)
            scores = policy(subset)
            validate_token_scores(scores, subset.batch["response_mask"])
            results[torch.as_tensor(indices)] = scores.log_probs.detach().cpu()
        return results

    def _drift_gate(self, native, batch, source, hint):
        """Check native against rollout logprobs of the sampled actions.

        The maximum catches a gross error in any one action; the optional mean
        catches a small systematic offset that a low-precision engine's
        per-action noise would hide under any usable maximum.
        """
        mask = batch.batch["response_mask"].bool()
        errors = (native - batch.batch["rollout_log_probs"]).abs()[mask].double()
        statistics = {
            "max": errors.max().item(),
            "mean": errors.mean().item(),
            "p99": np.quantile(errors.numpy(), 0.99).item(),
        }
        for name, tolerance in (
            ("max", self.config.score_tolerance),
            ("mean", self.config.score_mean_tolerance),
        ):
            value = statistics[name]
            if not np.isfinite(value) or (tolerance is not None and value > tolerance):
                raise RuntimeError(
                    f"{source} {name} logprob disagreement {value:.6g} > "
                    f"{tolerance}; {hint}"
                )
        return {
            f"rollout/logprob_{name}_error": value for name, value in statistics.items()
        }

    def prepare_batch(self, batch, native_scores=None):
        if native_scores is None:
            with torch.no_grad():
                old = self._score_chunks(self.policy, batch)
                reference = self._score_chunks(self.reference, batch)
        else:
            if not isinstance(native_scores, (tuple, list)) or len(native_scores) != 2:
                raise ValueError("Native scores require current and reference tensors")
            old, reference = native_scores
        mask = batch.batch["response_mask"]
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != mask.shape
            or value.requires_grad
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
            for value in (old, reference)
        ):
            raise ValueError(
                "The ESPnet scorer must return finite detached CPU probabilities per action"
            )
        drift = self._drift_gate(
            old,
            batch,
            "Rollout/native",
            "check checkpoint version, prefix, audio frontend and sampling distribution",
        )
        _, token_rewards, metrics = outcome_rewards(batch, self.config.reward_function)
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
        return {**metrics, **drift}

    def prepare_async_batch(
        self,
        batch,
        behavior_policy=None,
        *,
        behavior_version,
        learner_version,
        behavior_checkpoint_sha256,
        native_scores=None,
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
            raise ValueError("Stale rollout requires the policy that generated it")
        elif behavior_policy is self.policy:
            raise ValueError("Stale rollout cannot be audited with the current actor")
        if native_scores is None:
            with torch.no_grad():
                old = self._score_chunks(self.policy, batch)
                behavior = (
                    old
                    if behavior_policy is self.policy
                    else self._score_chunks(behavior_policy, batch)
                )
                reference = self._score_chunks(self.reference, batch)
        else:
            if not isinstance(native_scores, (tuple, list)) or len(native_scores) != 3:
                raise ValueError(
                    "Native async scores require current, behavior, reference"
                )
            old, behavior, reference = native_scores
        mask = batch.batch["response_mask"]
        for scores in (old, behavior, reference):
            if (
                not isinstance(scores, torch.Tensor)
                or scores.shape != mask.shape
                or not torch.isfinite(scores).all()
                or scores.requires_grad
                or scores.device.type != "cpu"
            ):
                raise ValueError(
                    "Native async scores must be finite, detached CPU tensors preserving sampled actions"
                )
        if behavior_version == learner_version and not torch.equal(old, behavior):
            raise ValueError(
                "Current behavior native scores differ from learner scores"
            )
        drift = self._drift_gate(
            behavior, batch, "Behavior native/SGLang", "the generating actor must match"
        )
        _, token_rewards, metrics = outcome_rewards(batch, self.config.reward_function)
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
            **metrics,
            **drift,
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
            parts = self._micro_batches(local)
            for number, indices in enumerate(parts):
                part = local.select_idxs(indices)
                part.to(self.device)
                final_microbatch = number == len(parts) - 1
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
                        # IcePop corrects whatever separates the engine that
                        # sampled the actions from the learner: the async
                        # policy lag and a low-precision engine's numerics.
                        rollout_log_probs=part.batch["rollout_log_probs"],
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
            self.optimizer.step()
            # Release the gradients here rather than at the next update: on a
            # collocated device they would otherwise hold a full copy of the
            # model through rollout, which is exactly the memory the engine
            # subprocess needs.
            self.optimizer.zero_grad(set_to_none=True)
            metrics["actor/grad_norm"] = norm.item()
        if self.world_size > 1:
            keys = sorted(metrics)
            values = torch.tensor([metrics[key] for key in keys], device=self.device)
            distributed.all_reduce(values)
            values /= self.world_size
            metrics = dict(zip(keys, values.cpu().tolist(), strict=True))
        return metrics

    def train(self, sources):
        if self.config.execution_mode != "sync":
            raise ValueError("Use the Miles async driver for disagg_async training")
        # Every rank must create the group before rank zero starts generation.
        # Lazy creation after a long rollout leaves other ranks in new_group.
        group = self._object_collective_group() if self.world_size > 1 else None
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.rollout is None and self.rank == 0:
            self.rollout = CheckpointRollout(self.config)
        if len({source["id"] for source in sources}) < self.config.prompts_per_step:
            raise ValueError("The prompt batch requires distinct source records")
        last_end = None
        for step in range(self.config.steps):
            if self.rank == 0:
                try:
                    chosen = [
                        sources[
                            (step * self.config.prompts_per_step + i) % len(sources)
                        ]
                        for i in range(self.config.prompts_per_step)
                    ]
                    if self.config.collocated and self.device.type != "cpu":
                        # A collocated rollout engine is a separate process on
                        # this same device. Return cached allocator blocks so
                        # it can reserve its own static fraction.
                        get_torch_device().empty_cache()
                    self.rollout.sync_weights(self.policy, step)
                    generations = self.rollout.generate(chosen, step)
                    batch = collate_rollouts(
                        chosen, generations, self.config.group_size
                    )
                    payload = [batch, None]
                except Exception as error:
                    (output / f"rollout-{step:06d}.error.txt").write_text(
                        traceback.format_exc()
                    )
                    payload = [None, f"{type(error).__name__}: {error}"]
            else:
                payload = [None, None]
            train_start, train_start_wall = time.monotonic(), time.time()
            if self.world_size > 1:
                distributed.broadcast_object_list(payload, src=0, group=group)
            batch, error = payload
            if error:
                raise RuntimeError(
                    f"Rollout/score stage failed at step {step}: {error}"
                )
            try:
                old, _, reference = score_native_batch(self, batch)
            except Exception:
                (output / f"rollout-{step:06d}.rank-{self.rank}.error.txt").write_text(
                    traceback.format_exc()
                )
                raise
            payload = [None, None, None]
            if self.rank == 0:
                try:
                    metrics = self.prepare_batch(batch, native_scores=(old, reference))
                    batch.save_to_disk(output / f"rollout-{step:06d}.pkl")
                    payload = [batch, metrics, None]
                except Exception as error:
                    (output / f"rollout-{step:06d}.error.txt").write_text(
                        traceback.format_exc()
                    )
                    payload[2] = f"{type(error).__name__}: {error}"
            if self.world_size > 1:
                distributed.broadcast_object_list(payload, src=0, group=group)
            batch, metrics, error = payload
            if error:
                raise RuntimeError(
                    f"Reward/prepare stage failed at step {step}: {error}"
                )
            metrics.update(self.update_actor(batch))
            if self.device.type != "cpu":
                get_torch_device().synchronize()
            update_end = time.monotonic()
            update_end_wall = time.time()
            status = [None]
            if self.rank == 0:
                try:
                    metrics["step"] = step + 1
                    metrics.update(
                        learner_timing(
                            train_start,
                            update_end,
                            last_end,
                            warmup=step < self.config.warmup_steps,
                            start_wall=train_start_wall,
                            end_wall=update_end_wall,
                        )
                    )
                    metrics.update(
                        {
                            "runtime/train_world_size": self.world_size,
                            "runtime/rollout_replicas": self.config.rollout_replicas,
                        }
                    )
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(metrics) + "\n")
                    print(json.dumps(metrics), flush=True)
                    # The same cadence the Miles async loop applies.
                    if (
                        (step + 1) % self.config.save_interval == 0
                        or step + 1 == self.config.steps
                    ):
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
            last_end = update_end
            if self.world_size > 1:
                distributed.broadcast_object_list(status, src=0, group=group)
            if status[0]:
                raise RuntimeError(
                    f"Checkpoint stage failed at step {step + 1}: {status[0]}"
                )

    def _object_collective_group(self):
        """Use one explicit-timeout Gloo group for native rollout/score objects.

        Call collectively before any rank-zero rollout or reward preparation.
        Even a Gloo default group may have a shorter caller-supplied timeout.
        """
        if not hasattr(self, "_rollout_object_group"):
            self._rollout_object_group = distributed.new_group(
                backend="gloo",
                timeout=timedelta(minutes=self.config.distributed_timeout_minutes),
            )
        return self._rollout_object_group


def run(config):
    if config.dtype == "float32":
        # Actor and rollout must use the same declared full FP32 precision.
        # Rollout sets Triton's independent tl.dot option before child imports.
        backend = getattr(torch.backends, get_device_name(), None)
        if backend is not None and hasattr(backend, "matmul"):
            backend.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config.validate()
    engine_class(config)
    miles_evidence = verify_miles(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        device_type = torch.device(config.device).type
        distributed.init_process_group(
            get_nccl_backend() if device_type != "cpu" else "gloo",
            timeout=timedelta(minutes=config.distributed_timeout_minutes),
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
    # Record where every training rank actually landed, then prove the ranks
    # hold one distinct GPU each on a single host. The expected count comes
    # from the recipe's train_gpus, so one rank is as valid as eight.
    if torch.device(config.device).type != "cpu":
        world_size = distributed.get_world_size() if distributed.is_initialized() else 1
        # A single rank is not launched by torchrun and owns local device 0.
        local_rank = int(os.environ.get("LOCAL_RANK", "0" if world_size == 1 else "-1"))
        visible = [
            item
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item
        ]
        census = {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "config_device": config.device,
            "cuda_visible_devices": visible,
            "visible_gpu_count": get_torch_device().device_count(),
            "current_device": get_torch_device().current_device(),
            "physical_gpu_id": (
                visible[local_rank] if local_rank in range(len(visible)) else None
            ),
            "host_identity": host_identity(),
            "pid": os.getpid(),
        }
        Path(output, f"train-device-rank-{rank}.json").write_text(
            json.dumps(census, indent=2) + "\n"
        )
        expected = config.train_ranks
        if world_size != expected:
            raise ValueError(
                f"The recipe declares {expected} training ranks; this rank sees "
                f"world_size={world_size}"
            )
        if expected > 1:
            # Several ranks are launched by this toolkit, which pins the
            # devices, so each rank must see exactly them and own its own.
            if census["visible_gpu_count"] != expected or (
                visible and visible != config.train_device_ids
            ):
                raise ValueError(
                    f"Training ranks must occupy the recipe's train_gpus "
                    f"{config.train_device_ids}; this rank sees {visible} and "
                    f"{census['visible_gpu_count']} devices"
                )
            if census["current_device"] != local_rank:
                raise ValueError("Each training rank must own its local device")
        if distributed.is_initialized():
            all_census = [None] * world_size
            distributed.all_gather_object(all_census, census)
            if len({item["physical_gpu_id"] for item in all_census}) != expected or any(
                item["host_identity"] != census["host_identity"] for item in all_census
            ):
                raise ValueError(
                    f"Training ranks do not occupy {expected} distinct GPUs on one host"
                )
        else:
            all_census = [census]
        if rank == 0:
            Path(output, "train_census.json").write_text(
                json.dumps(all_census, indent=2) + "\n"
            )
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
        if trainer.rollout is not None:
            # The rollout processes hold their engines until training ends.
            trainer.rollout.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()
