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
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from espnet2.torch_utils.set_all_random_seed import set_all_random_seed

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
from .timing import PhaseProfiler, learner_timing, profile_phase
from .training_objectives import (
    build_lr_scheduler,
    select_training_sources,
    west_group_advantages,
    west_policy_loss,
)
from .training_state import (
    capture_rng_state,
    load_native_model_state,
    restore_rng_state,
    validate_resume_checkpoint,
)
from .types import drift_gate, support_counts, validate_token_scores
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
        self.reference_scoring_device = config.reference_device or self.device
        if (
            config.reference_cpu_offload
            and torch.device(self.reference_scoring_device).type == "cpu"
        ):
            raise ValueError("Reference CPU offload requires a GPU scoring device")
        reference_storage = (
            "cpu" if config.reference_cpu_offload else config.reference_device
        )
        self.reference = (
            reference
            if reference is not None
            else clone_reference(policy, reference_storage)
        ).eval()
        if reference_storage is not None:
            self.reference = self.reference.to(reference_storage)
        self.reference.requires_grad_(False)
        optimizer_class = (
            CPUOffloadedAdamW if config.optimizer_device == "cpu" else torch.optim.AdamW
        )
        self.optimizer = optimizer_class(
            [p for p in policy.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            **(
                {"chunk_bytes": config.allreduce_chunk_bytes}
                if optimizer_class is CPUOffloadedAdamW
                else {}
            ),
        )
        self.scheduler = build_lr_scheduler(self.optimizer, config)
        self.profiler = PhaseProfiler(
            self.device, getattr(config, "profile_phases", False)
        )
        self.optimizer.profiler = self.profiler
        if self.profiler.enabled:
            path = Path(config.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            budget = {
                "policy_trainable_bytes": sum(
                    p.numel() * p.element_size()
                    for p in policy.parameters()
                    if p.requires_grad
                ),
                "policy_frozen_bytes": sum(
                    p.numel() * p.element_size()
                    for p in policy.parameters()
                    if not p.requires_grad
                ),
                "reference_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in self.reference.parameters()
                ),
                "reference_scoring_device": str(self.reference_scoring_device),
                "cpu_optimizer_master_bytes": sum(
                    p.numel() * p.element_size()
                    for p in getattr(self.optimizer, "_shadow", [])
                ),
            }
            (path / f"memory-rank-{self.rank}.json").write_text(
                json.dumps(budget, indent=2) + "\n"
            )
        self.completed_updates = 0
        self.rollout = rollout
        # Set by every distributed scoring pass (espnet2.rl.scoring).
        self.support_metrics = {}
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

    def _score_chunks(self, policy, batch, *, score_raw=False):
        """Native log probabilities of the batch's actions.

        Also leaves in ``support_pass`` this pass's frames that a policy
        scoring under recorded supports (Bagpiper) would have put under another
        support, and all frames it scored: ``[0, 0]`` for other families.
        """
        if isinstance(policy, AbstractContextManager):
            # A weight snapshot lends its values to a module for this pass.
            with policy as module:
                return self._score_chunks(module, batch, score_raw=score_raw)
        if policy is getattr(self, "reference", None) and getattr(
            self.config, "reference_cpu_offload", False
        ):
            try:
                with profile_phase(self, "reference_h2d", memory=False):
                    policy.to(self.reference_scoring_device)
                return self._score_chunks_on_device(policy, batch, score_raw=score_raw)
            finally:
                # Detached scores already live on CPU. Offload even if transfer
                # or scoring failed; the next pass alone brings weights back.
                with profile_phase(self, "reference_d2h", memory=False):
                    policy.to("cpu")
        return self._score_chunks_on_device(policy, batch, score_raw=score_raw)

    def _score_chunks_on_device(self, policy, batch, *, score_raw=False):
        before = support_counts(policy)
        results = torch.zeros(batch.batch["response_mask"].shape)
        scoring_device = next(policy.parameters()).device
        for indices in self._micro_batches(batch):
            subset = batch.select_idxs(indices)
            subset.to(scoring_device)
            scores = policy(subset, score_raw=True) if score_raw else policy(subset)
            validate_token_scores(scores, subset.batch["response_mask"])
            results[torch.as_tensor(indices)] = scores.log_probs.detach().cpu()
        self.support_pass = [
            now - then for now, then in zip(support_counts(policy), before)
        ]
        return results

    def _drift_gate(self, native, batch, source, hint):
        statistics = drift_gate(
            native,
            batch.batch["rollout_log_probs"],
            batch.batch["response_mask"],
            self.config,
            source,
            hint,
        )
        return {
            f"rollout/logprob_{name}_error": value for name, value in statistics.items()
        }

    def prepare_batch(self, batch, native_scores=None):
        if native_scores is None:
            with torch.no_grad():
                old = self._score_chunks(self.policy, batch)
                reference = self._score_chunks(
                    self.reference,
                    batch,
                    score_raw=self.config.training_objective == "west_grpo",
                )
        else:
            if not isinstance(native_scores, (tuple, list)) or len(native_scores) != 2:
                raise ValueError("Native scores require current and reference tensors")
            old, reference = native_scores
        return self._prepare_scored_batch(batch, old, reference)

    def _prepare_scored_batch(
        self, batch, old, reference, behavior=None, *, current_behavior=False
    ):
        """Validate detached scores and build one globally normalized reward batch."""
        mask = batch.batch["response_mask"]
        scores = (old, reference) if behavior is None else (old, behavior, reference)
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != mask.shape
            or value.requires_grad
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
            for value in scores
        ):
            raise ValueError(
                "Native scores must be finite, detached CPU tensors preserving sampled actions"
            )
        if current_behavior and not torch.equal(old, behavior):
            raise ValueError(
                "Current behavior native scores differ from learner scores"
            )
        drift = self._drift_gate(
            old if behavior is None else behavior,
            batch,
            "Rollout/native" if behavior is None else "Behavior native/SGLang",
            "check checkpoint version, prefix, audio frontend and sampling distribution",
        )
        _, token_rewards, metrics = outcome_rewards(batch, self.config.reward_function)
        metrics.update(self.support_metrics)
        advantages, returns = self._advantages(token_rewards, mask, batch)
        batch.batch.update(
            old_log_probs=old,
            ref_log_probs=reference,
            advantages=advantages,
            returns=returns,
            token_level_rewards=token_rewards,
        )
        if behavior is not None:
            batch.batch["behavior_native_log_probs"] = behavior
        return {**metrics, **drift}

    def _advantages(self, token_rewards, mask, batch):
        if self.config.training_objective == "west_grpo":
            return west_group_advantages(
                token_rewards, mask, batch.non_tensor_batch["uid"]
            )
        return compute_grpo_outcome_advantage(
            token_rewards,
            mask,
            batch.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=self.config.normalize_advantage,
        )

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
        self.validate_behavior_batch(
            batch, behavior_version, behavior_checkpoint_sha256
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
        metrics = self._prepare_scored_batch(
            batch,
            old,
            reference,
            behavior,
            current_behavior=behavior_version == learner_version,
        )
        mask = batch.batch["response_mask"]
        return {
            **metrics,
            **self.behavior_metrics(batch, behavior_version, learner_version),
            "rollout/current_behavior_logprob_max_difference": (
                (old - behavior).abs() * mask
            )
            .max()
            .item(),
        }

    def validate_behavior_batch(self, batch, version, digest):
        """A versioned rollout must contain complete, unique, uniformly bound groups."""
        if len(batch) != self.config.global_candidate_batch_size:
            raise ValueError("Rollout does not contain the full global candidate count")
        identities = batch.non_tensor_batch["uid"].tolist()
        size = self.config.group_size
        groups = [identities[i : i + size] for i in range(0, len(batch), size)]
        if (
            any(len(set(group)) != 1 for group in groups)
            or len({group[0] for group in groups}) != self.config.prompts_per_step
        ):
            raise ValueError("Rollouts require complete unique prompt groups")
        for generation in batch.non_tensor_batch["generation"]:
            if (
                generation.get("behavior_policy_version") != version
                or generation.get("behavior_checkpoint_sha256") != digest
            ):
                raise ValueError(
                    "Mixed or unrecorded behavior checkpoint in rollout batch"
                )

    def behavior_metrics(self, batch, behavior_version, learner_version):
        return {
            "rollout/behavior_policy_version": behavior_version,
            "rollout/learner_policy_version": learner_version,
            "rollout/weight_staleness": learner_version - behavior_version,
            "rollout/global_prompt_batch_size": len(batch) // self.config.group_size,
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
            # Includes actor forward/loss, backward, gradient reduction and clipping.
            self.profiler.start("backward")
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
                    current = (
                        self.actor(part, score_raw=True)
                        if self.config.training_objective == "west_grpo"
                        else self.actor(part)
                    )
                    mask = part.batch["response_mask"]
                    validate_token_scores(current, mask)
                    if self.config.training_objective == "west_grpo":
                        loss = west_policy_loss(
                            current.log_probs,
                            part.batch["advantages"],
                            mask,
                            **self.actor_config.global_batch_info,
                        )
                        ppo_metrics = {}
                    else:
                        loss, ppo_metrics = compute_policy_loss_vanilla(
                            part.batch["old_log_probs"],
                            current.log_probs,
                            part.batch["advantages"],
                            mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                            config=self.actor_config,
                            # Correct both async lag and engine numerics.
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
            learning_rate = self.optimizer.param_groups[0]["lr"]
            self.profiler.stop("backward")
            with profile_phase(self, "optimizer"):
                self.optimizer.step()
            self.scheduler.step()
            # Release the gradients here rather than at the next update: on a
            # collocated device they would otherwise hold a full copy of the
            # model through rollout, which is exactly the memory the engine
            # subprocess needs.
            self.optimizer.zero_grad(set_to_none=True)
            metrics["actor/grad_norm"] = norm.item()
            metrics["actor/learning_rate"] = learning_rate
            metrics["actor/next_learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.completed_updates += 1
        if self.world_size > 1:
            keys = sorted(metrics)
            values = torch.tensor([metrics[key] for key in keys], device=self.device)
            distributed.all_reduce(values)
            values /= self.world_size
            metrics = dict(zip(keys, values.cpu().tolist(), strict=True))
        return metrics

    def save_training_checkpoint(self, path, step, *, extra_state=None):
        """Collectively save optimizer, scheduler, RNG and absolute data cursor."""
        local = {"rank": self.rank, "rng": None, "error": None}
        try:
            if step != self.completed_updates:
                raise ValueError(
                    "Checkpoint step differs from completed optimizer updates"
                )
            local["rng"] = capture_rng_state(self.device)
        except Exception as error:
            local["error"] = f"{type(error).__name__}: {error}"
        reports = [None] * self.world_size
        if self.world_size > 1:
            distributed.all_gather_object(
                reports, local, group=self._object_collective_group()
            )
        else:
            reports[0] = local
        failures = [
            f"rank {value['rank']}: {value['error']}"
            for value in reports
            if value["error"] is not None
        ]
        if failures:
            raise RuntimeError(
                "Collective checkpoint capture failed: " + "; ".join(failures)
            )
        states = [value["rng"] for value in reports]
        trainer_state = dict(extra_state or {})
        trainer_state.update(
            version=1,
            completed_updates=step,
            next_update=step,
            consumed_prompts=step * self.config.prompts_per_step,
            world_size=self.world_size,
            rank_rng_states=states,
            data_seed=self.config.data_seed,
            data_order=(
                "torch_randperm_seed_plus_epoch_v1"
                if self.config.shuffle_train_data
                else "cyclic_v1"
            ),
        )
        if hasattr(self, "source_count"):
            trainer_state["source_count"] = self.source_count
        status = [None]
        if self.rank == 0:
            try:
                save_checkpoint(
                    path,
                    self.policy,
                    self.optimizer,
                    step,
                    self.config,
                    scheduler=self.scheduler,
                    trainer_state=trainer_state,
                )
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        if self.world_size > 1:
            distributed.broadcast_object_list(
                status, src=0, group=self._object_collective_group()
            )
        if status[0]:
            raise RuntimeError(f"Collective checkpoint save failed: {status[0]}")

    def load_training_checkpoint(self, path):
        """Restore after cloning the original fixed SFT reference, on every rank.

        Rollout workers restart from the restored policy. Their sampling seed
        is derived from the absolute update index by the rollout engine; their
        previous process RNG is not presented as a bitwise continuation.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = validate_resume_checkpoint(
            checkpoint,
            self.config,
            self.world_size,
            getattr(self, "source_count", None),
        )
        load_native_model_state(self.policy, checkpoint["model"])
        if (
            not isinstance(self.optimizer, CPUOffloadedAdamW)
            or self.rank == self.optimizer.owner_rank
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        # All ranks must advance the same LR, including CPU-optimizer nonowners
        # that intentionally do not allocate the full Adam state.
        for group, saved in zip(
            self.optimizer.param_groups,
            checkpoint["optimizer"]["param_groups"],
            strict=True,
        ):
            for name in ("lr", "initial_lr", "betas", "eps", "weight_decay"):
                if name in saved:
                    group[name] = saved[name]
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.completed_updates = checkpoint["step"]
        restore_rng_state(state["rank_rng_states"][self.rank], self.device)
        return state

    def train_batch(
        self,
        batch,
        rollout_id,
        *,
        behavior=None,
        prepare=None,
        snapshot=None,
        on_prepare_error=None,
        scorer=None,
    ):
        """Run the same audited native update for local and Ray transports.

        All ranks score before rank zero prepares rewards; the prepared result or
        error reaches every rank before any optimizer or snapshot can change.
        Transport-specific identity checks and clean-state preservation stay with
        the caller, while scoring, persistence and update ordering live here.
        """
        self.profiler.reset()
        try:
            scores = (scorer or score_native_batch)(self, batch, behavior)
        except Exception:
            self.record_rollout_error(rollout_id)
            raise
        payload = [None]
        if self.rank == 0:
            try:
                with profile_phase(self, "reward_advantage"):
                    metrics = (
                        prepare(scores)
                        if prepare is not None
                        else self.prepare_batch(
                            batch, native_scores=(scores[0], scores[2])
                        )
                    )
                with profile_phase(self, "batch_save"):
                    batch.save_to_disk(
                        Path(self.config.output_dir, f"rollout-{rollout_id:06d}.pkl")
                    )
                payload[0] = {"batch": batch, "metrics": metrics, "error": None}
            except Exception as error:
                self.record_rollout_error(rollout_id, scores=scores, batch=batch)
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        del scores
        if self.world_size > 1:
            distributed.broadcast_object_list(
                payload, src=0, group=self._object_collective_group()
            )
        prepared = payload[0]
        if prepared["error"]:
            if on_prepare_error is not None:
                on_prepare_error(prepared["error"])
            raise RuntimeError(prepared["error"])
        if snapshot is not None:
            with profile_phase(self, "behavior_snapshot"):
                snapshot.take(self.completed_updates)
        metrics = prepared["metrics"]
        try:
            metrics.update(self.update_actor(prepared["batch"]))
        except Exception:
            self.record_rollout_error(rollout_id)
            raise
        if self.profiler.enabled:
            profiles = [None] * self.world_size
            local_profile = self.profiler.result()
            if self.world_size > 1:
                distributed.all_gather_object(
                    profiles, local_profile, group=self._object_collective_group()
                )
            else:
                profiles[0] = local_profile
            # Report phase walls/residency maxima, including owner-only CPU work.
            metrics.update(
                {
                    key: max(part.get(key, 0) for part in profiles)
                    for key in set().union(*profiles)
                }
            )
            metrics["perf/profile_rank_max"] = True
        return metrics

    def record_rollout_error(self, rollout_id, *, scores=None, batch=None):
        """Preserve the original failure even if its logging path is unwritable."""
        try:
            suffix = "" if self.rank == 0 else f".rank-{self.rank}"
            Path(
                self.config.output_dir, f"rollout-{rollout_id:06d}{suffix}.error.txt"
            ).write_text(traceback.format_exc())
            if self.profiler.enabled:
                base = Path(
                    self.config.output_dir, f"rollout-{rollout_id:06d}{suffix}.failed"
                )
                base.with_suffix(".failed.json").write_text(
                    json.dumps(self.profiler.result(), indent=2) + "\n"
                )
                if scores is not None:
                    torch.save(
                        {
                            "native_scores": scores,
                            "rollout_log_probs": batch.batch["rollout_log_probs"],
                            "response_mask": batch.batch["response_mask"],
                        },
                        base.with_suffix(".failed.pt"),
                    )
        except Exception:
            pass

    def record_metrics(
        self, metrics, start, start_wall, last_end, *, extra=None, echo=False
    ):
        """Finish the native update timing before checkpoint or publication work."""
        if self.device.type != "cpu":
            get_torch_device().synchronize()
        end, end_wall = time.monotonic(), time.time()
        status = [None]
        if self.rank == 0:
            try:
                metrics.update(
                    learner_timing(
                        start,
                        end,
                        last_end,
                        warmup=self.completed_updates <= self.config.warmup_steps,
                        start_wall=start_wall,
                        end_wall=end_wall,
                    )
                )
                metrics.update(
                    step=self.completed_updates,
                    **{
                        "runtime/train_world_size": self.world_size,
                        **(extra or {}),
                    },
                )
                with Path(self.config.output_dir, "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                if echo:
                    print(json.dumps(metrics), flush=True)
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        if self.world_size > 1:
            distributed.broadcast_object_list(
                status, src=0, group=self._object_collective_group()
            )
        if status[0]:
            raise RuntimeError(f"Metrics publication failed: {status[0]}")
        return end

    def train(self, sources):
        if self.config.execution_mode != "sync":
            raise ValueError("Use the Miles async driver for disagg_async training")
        # Every rank must create the group before rank zero starts generation.
        # Lazy creation after a long rollout leaves other ranks in new_group.
        group = self._object_collective_group() if self.world_size > 1 else None
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.source_count = len(sources)
        if self.rollout is None and self.rank == 0:
            self.rollout = CheckpointRollout(self.config)
        if len({source["id"] for source in sources}) < self.config.prompts_per_step:
            raise ValueError("The prompt batch requires distinct source records")
        last_end = None
        for step in range(getattr(self, "completed_updates", 0), self.config.steps):
            if self.rank == 0:
                try:
                    chosen = select_training_sources(sources, step, self.config)
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
            metrics = self.train_batch(batch, step)
            last_end = self.record_metrics(
                metrics,
                train_start,
                train_start_wall,
                last_end,
                extra={"runtime/rollout_replicas": self.config.rollout_replicas},
                echo=True,
            )
            if (
                step + 1
            ) % self.config.save_interval == 0 or step + 1 == self.config.steps:
                self.save_training_checkpoint(
                    output / f"step-{step + 1:06d}.pt", step + 1
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
    config.validate()
    sources = read_manifest(config.manifest, config=config)
    if config.dtype == "float32":
        # Actor and rollout must use the same declared full FP32 precision.
        # Rollout sets Triton's independent tl.dot option before child imports.
        backend = getattr(torch.backends, get_device_name(), None)
        if backend is not None and hasattr(backend, "matmul"):
            backend.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    engine_class(config)
    miles_evidence = verify_miles(config)
    set_all_random_seed(config.seed)
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
    trainer.source_count = len(sources)
    if config.resume_checkpoint:
        trainer.load_training_checkpoint(config.resume_checkpoint)
    try:
        trainer.train(sources)
    finally:
        if trainer.rollout is not None:
            # The rollout processes hold their engines until training ends.
            trainer.rollout.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()
