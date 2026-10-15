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
from .rollout import CheckpointRollout, save_checkpoint, sha256
from .scoring import score_native_batch
from .timing import learner_timing
from .trainer import NativeGRPOTrainer
from .utils import build_policy


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
        self.config = config
        self.version = 0
        self.exporter = CheckpointRollout(config) if self._rank == 0 else None
        self.last_end = None
        self.last_metrics = None
        return 0

    @property
    def train_parallel_config(self):
        return {"dp_size": self._world_size}

    def _get_parallel_config(self):
        return self.train_parallel_config

    def get_device_census(self):
        return self.device_census

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
                    checkpoint = Path(data["behavior_checkpoint"])
                    actual_hash = sha256(checkpoint)
                    if actual_hash != data["behavior_checkpoint_sha256"]:
                        raise ValueError(
                            "Behavior checkpoint file differs from rollout identity"
                        )
                    if (
                        type(version) is not int
                        or version < 0
                        or not 0
                        <= self.version - version
                        <= self.config.max_weight_staleness
                    ):
                        raise ValueError("Invalid behavior checkpoint staleness")
                    payload[0] = {
                        "batch": batch,
                        "behavior_version": version,
                        "behavior_checkpoint": str(checkpoint),
                        "behavior_checkpoint_sha256": actual_hash,
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
        # Each actor scores only its native shard. A stale behavior actor is a
        # frozen policy copy like the reference, so it is loaded independently
        # onto the reference's device: CPU for recipes that keep the reference
        # there, the training GPU for small models that keep it on the GPU.
        # Rank-zero-only reward decoding avoids TTS races.
        behavior = None
        load_error = None
        try:
            if version != self.version:
                config = copy.deepcopy(self.config)
                config.checkpoint = data["behavior_checkpoint"]
                device = self.config.reference_device or self.config.device
                behavior = build_policy(config, device).eval().requires_grad_(False)
        except Exception as error:
            load_error = f"rank {self._rank}: {type(error).__name__}: {error}"
            self._record_rollout_error(rollout_id)
        if self._world_size > 1:
            failures = [None] * self._world_size
            dist.all_gather_object(
                failures, load_error, group=self.trainer._object_collective_group()
            )
        else:
            failures = [load_error]
        failures = [failure for failure in failures if failure]
        if failures:
            raise RuntimeError("Behavior loading failed: " + "; ".join(failures))
        try:
            native_scores = score_native_batch(self.trainer, batch, behavior)
        except Exception:
            self._record_rollout_error(rollout_id)
            raise
        payload = [None]
        if self._rank == 0:
            try:
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
        del behavior, native_scores
        self._broadcast_payload(payload)
        prepared = payload[0]
        if prepared["error"]:
            raise RuntimeError(prepared["error"])
        batch, metrics = prepared["batch"], prepared["metrics"]
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
        if self._rank == 0:
            save_checkpoint(
                Path(self.config.output_dir, f"step-{self.version:06d}.pt"),
                self.trainer.policy,
                self.trainer.optimizer,
                self.version,
                self.config,
            )

    def export_async_snapshot(self):
        if self._rank != 0:
            return None
        metadata = self.exporter.sync_weights(self.trainer.policy, self.version)
        return {**metadata, "path": str(self.exporter.version.resolve())}

    def update_weights(self, info):
        raise ValueError("This ESPnet profile uses immutable checkpoint publication")

    def dispose(self):
        if dist.is_initialized():
            dist.destroy_process_group()
