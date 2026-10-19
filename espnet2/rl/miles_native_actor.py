"""Native ESPnet training worker implementing Miles' TrainRayActor contract."""

import copy
import json
import os
import time
import traceback
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train_actor import TrainRayActor
from miles.utils import object_store

from .config import ESPnetRLConfig
from .failure_checkpoint import preserve_preupdate_failure
from .rollout import CheckpointRollout
from .scoring import score_native_batch
from .timing import learner_timing
from .trainer import NativeGRPOTrainer
from .utils import build_policy


class BehaviorSnapshot:
    """One policy version's trainable weights, held on the CPU in their dtype.

    Scoring with the snapshot (a context manager, see the trainer's
    ``_score_chunks``) swaps these values into the learner's own module and
    the current ones back out afterwards, so a stale batch is audited by the
    native scorer with exactly the weights that generated it, without a
    second model. Frozen parameters never change and are not copied.
    """

    def __init__(self, policy):
        self.policy = policy
        self.version = None
        self.parameters = [p for p in policy.parameters() if p.requires_grad]
        # Views of one buffer per dtype: the pinned allocator rounds every
        # allocation up to a power of two, per parameter a third more host
        # memory than the weights (Bagpiper bf16: 20.3 instead of 16 GiB).
        sizes = {}
        for p in self.parameters:
            sizes[p.dtype] = sizes.get(p.dtype, 0) + p.numel()
        pinned = any(p.is_cuda for p in self.parameters)
        buffers = {
            dtype: torch.empty(size, dtype=dtype, pin_memory=pinned)
            for dtype, size in sizes.items()
        }
        self.values, used = [], dict.fromkeys(buffers, 0)
        for p in self.parameters:
            start, used[p.dtype] = used[p.dtype], used[p.dtype] + p.numel()
            self.values.append(buffers[p.dtype][start : used[p.dtype]].view(p.shape))

    @torch.no_grad()
    def take(self, version):
        for parameter, value in zip(self.parameters, self.values, strict=True):
            value.copy_(parameter)
        self.version = version

    @torch.no_grad()
    def _swap(self):
        for parameter, value in zip(self.parameters, self.values, strict=True):
            current = parameter.detach().clone()
            parameter.copy_(value)
            value.copy_(current)

    def __enter__(self):
        self._swap()
        return self.policy

    def __exit__(self, *failure):
        self._swap()


class ESPnetTrainRayActor(TrainRayActor):
    def init(
        self,
        args,
        role,
        *,
        with_ref=False,
        with_opd_teacher=False,
        recv_ckpt_src_rank=None,
        indep_dp_info=None,
        indep_dp_store_addr=None,
    ):
        if role != "actor" or with_opd_teacher or recv_ckpt_src_rank is not None:
            raise ValueError("The native short-run actor supports fresh GRPO only")
        config = ESPnetRLConfig(**args.espnet_config)
        config.validate()
        args = copy.copy(args)
        args.distributed_timeout_minutes = config.distributed_timeout_minutes
        self._init_common(args, role, with_ref=with_ref)
        assigned = ray.get_gpu_ids()
        visible = [
            value
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value
        ]
        context = ray.get_runtime_context()
        census = {
            "rank": self._rank,
            "world_size": self._world_size,
            "ray_node_id": context.get_node_id(),
            "ray_gpu_ids": assigned,
            "cuda_visible_devices": visible,
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "visible_gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "pid": os.getpid(),
        }
        self.device_census = census
        Path(config.output_dir, f"train-device-rank-{self._rank}.json").write_text(
            json.dumps(census, indent=2) + "\n"
        )
        if (
            len(assigned) != 1
            or len(visible) != 1
            or census["local_rank"] != 0
            or census["visible_gpu_count"] != 1
            or census["current_device"] != 0
        ):
            raise ValueError(
                "Native Ray actor requires one assigned visible GPU at local rank zero"
            )
        config.device = f"cuda:{os.environ['LOCAL_RANK']}"
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.trainer = NativeGRPOTrainer(build_policy(config, config.device), config)
        self.trainer.source_count = args.espnet_source_count
        self.config = config
        if config.resume_checkpoint:
            self.trainer.load_training_checkpoint(config.resume_checkpoint)
        self.version = self.trainer.completed_updates
        self.snapshot = (
            BehaviorSnapshot(self.trainer.policy)
            if config.execution_mode == "disagg_async"
            else None
        )
        self.exporter = CheckpointRollout(config) if self._rank == 0 else None
        # Rank zero's record of each exported version's native sha256.
        self.exported = {}
        self.last_end = None
        self.last_metrics = None
        return self.version

    @property
    def train_parallel_config(self):
        return {"dp_size": self._world_size}

    def _get_parallel_config(self):
        return self.train_parallel_config

    def get_device_census(self):
        return self.device_census

    def get_training_version(self):
        return self.version

    def sleep(self):
        raise ValueError("The disaggregated learner stays resident")

    def wake_up(self):
        return None

    def train(
        self,
        rollout_id,
        rollout_data_ref,
        witness_info=None,
        attempt=0,
        external_data=None,
    ):
        if attempt or external_data is not None or witness_info is not None:
            raise ValueError("Short-run recovery, critic, and witness are disabled")
        if rollout_id != self.version:
            raise ValueError("Unexpected or repeated optimizer update")
        start = time.monotonic()
        start_wall = time.time()
        payload = [None]
        if self._rank == 0:
            try:
                with object_store.get_instance().get(rollout_data_ref) as data:
                    batch = data["batch"]
                    version = data["behavior_policy_version"]
                    if (
                        type(version) is not int
                        or version < 0
                        or not 0
                        <= self.version - version
                        <= self.config.max_weight_staleness
                    ):
                        raise ValueError("Invalid behavior checkpoint staleness")
                    if (
                        self.config.execution_mode != "disagg_async"
                        and version != self.version
                    ):
                        raise ValueError("Synchronous training rejects a stale policy")
                    digest = data["behavior_checkpoint_sha256"]
                    if digest != self.exported.get(version):
                        raise ValueError(
                            "Behavior checkpoint differs from this learner's export"
                        )
                    if version != self.version and (
                        self.snapshot is None or self.snapshot.version != version
                    ):
                        raise ValueError("No snapshot holds the behavior policy")
                    payload[0] = {
                        "batch": batch,
                        "behavior_version": version,
                        "behavior_checkpoint_sha256": digest,
                        "error": None,
                    }
            except Exception as error:
                self._record_rollout_error(rollout_id)
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        self._broadcast_payload(payload)
        data = payload[0]
        if data["error"]:
            raise RuntimeError(data["error"])
        batch = data["batch"]
        version = data["behavior_version"]
        # Each actor scores only its native shard. A stale batch is audited by
        # the learner's own module holding the weights that generated it, which
        # every rank snapshotted just before that version's update.
        # Rank-zero-only reward decoding avoids TTS races.
        behavior = self.snapshot if version != self.version else None
        try:
            native_scores = score_native_batch(self.trainer, batch, behavior)
        except Exception:
            self._record_rollout_error(rollout_id)
            raise
        payload = [None]
        if self._rank == 0:
            try:
                if self.config.execution_mode != "disagg_async":
                    metrics = self._prepare_synchronous_batch(
                        batch, data, native_scores
                    )
                else:
                    metrics = self.trainer.prepare_async_batch(
                        batch,
                        behavior,
                        behavior_version=version,
                        learner_version=self.version,
                        behavior_checkpoint_sha256=data["behavior_checkpoint_sha256"],
                        native_scores=native_scores,
                    )
                batch.save_to_disk(
                    Path(self.config.output_dir) / f"rollout-{rollout_id:06d}.pkl"
                )
                payload[0] = {"batch": batch, "metrics": metrics, "error": None}
            except Exception as error:
                self._record_rollout_error(rollout_id)
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        del native_scores
        self._broadcast_payload(payload)
        prepared = payload[0]
        if prepared["error"]:
            # The error has reached every rank. No optimizer update has begun.
            failure = RuntimeError(prepared["error"])
            try:
                preservation = preserve_preupdate_failure(
                    self, rollout_id, prepared["error"]
                )
                failure.add_note(
                    "Pre-update checkpoint preservation: "
                    + json.dumps(preservation, allow_nan=False)
                )
            except Exception as checkpoint_error:
                # Collective/CUDA failure cannot be repaired here. Retain the
                # original integrity failure as the primary exception.
                failure.add_note(
                    "Checkpoint preservation failed: "
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                )
            raise failure from None
        batch, metrics = prepared["batch"], prepared["metrics"]
        # Nothing changes these weights before the optimizer step below. They
        # were exported as this version, so they generate the next batch and
        # this copy audits it.
        if self.snapshot is not None:
            self.snapshot.take(self.version)
        metrics.update(self.trainer.update_actor(batch))
        if self.trainer.device.type == "cuda":
            torch.cuda.synchronize(self.trainer.device)
        end = time.monotonic()
        end_wall = time.time()
        self.version += 1
        if self._rank == 0:
            metrics.update(
                learner_timing(
                    start,
                    end,
                    self.last_end,
                    warmup=self.version <= self.config.warmup_steps,
                    start_wall=start_wall,
                    end_wall=end_wall,
                )
            )
            metrics.update(
                {
                    "step": self.version,
                    "runtime/train_world_size": self._world_size,
                }
            )
            with Path(self.config.output_dir, "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            self.last_metrics = metrics
        self.last_end = end
        return TrainStepOutput(TrainStepOutcome.NORMAL)

    def _prepare_synchronous_batch(self, batch, data, native_scores):
        """Validate the on-policy global groups before WEST reward normalization."""
        if (
            data["behavior_version"] != self.version
            or len(batch) != self.config.global_candidate_batch_size
        ):
            raise ValueError("Synchronous training requires the current complete batch")
        identities = batch.non_tensor_batch["uid"].tolist()
        size = self.config.group_size
        groups = [identities[i : i + size] for i in range(0, len(batch), size)]
        if (
            any(len(set(group)) != 1 for group in groups)
            or len({group[0] for group in groups}) != self.config.prompts_per_step
        ):
            raise ValueError("Synchronous training requires complete unique groups")
        for generation in batch.non_tensor_batch["generation"]:
            if (
                generation.get("behavior_policy_version") != self.version
                or generation.get("behavior_checkpoint_sha256")
                != data["behavior_checkpoint_sha256"]
            ):
                raise ValueError("Synchronous batch mixes or omits behavior identities")
        if not torch.equal(native_scores[0], native_scores[1]):
            raise ValueError("Synchronous native behavior and current scores differ")
        metrics = self.trainer.prepare_batch(
            batch, native_scores=(native_scores[0], native_scores[2])
        )
        return {
            **metrics,
            "rollout/current_behavior_logprob_max_difference": 0.0,
            "rollout/behavior_policy_version": self.version,
            "rollout/learner_policy_version": self.version,
            "rollout/weight_staleness": 0,
            "rollout/global_prompt_batch_size": len(groups),
            "rollout/global_candidate_batch_size": len(batch),
        }

    def _broadcast_payload(self, payload):
        if self._world_size > 1:
            dist.broadcast_object_list(
                payload, src=0, group=self.trainer._object_collective_group()
            )

    def _record_rollout_error(self, rollout_id):
        """Preserve failures without masking them with a logging-path exception."""
        try:
            suffix = "" if self._rank == 0 else f".rank-{self._rank}"
            Path(
                self.config.output_dir, f"rollout-{rollout_id:06d}{suffix}.error.txt"
            ).write_text(traceback.format_exc())
        except OSError:
            pass

    def save_model(self, rollout_id, force_sync=False):
        if rollout_id + 1 != self.version:
            raise ValueError("Checkpoint update identity mismatch")
        self.trainer.save_training_checkpoint(
            Path(self.config.output_dir, f"step-{self.version:06d}.pt"),
            self.version,
            extra_state={
                "source_count": self.trainer.source_count,
                "next_update": self.version,
                "consumed_prompts": self.version * self.config.prompts_per_step,
                "data_seed": self.config.data_seed,
                "data_order": (
                    "torch_randperm_seed_plus_epoch_v1"
                    if self.config.shuffle_train_data
                    else "cyclic_v1"
                ),
            },
        )

    def export_async_snapshot(self):
        if self._rank != 0:
            return None
        # Export only: the rollout actors load it onto the rollout GPUs.
        metadata = self.exporter.export(self.trainer.policy, self.version)
        self.exported[self.version] = metadata["native_sha256"]
        return {**metadata, "path": str(self.exporter.version.resolve())}

    def update_weights(self, info):
        raise ValueError("This ESPnet profile uses immutable checkpoint publication")

    def dispose(self):
        if dist.is_initialized():
            dist.destroy_process_group()
