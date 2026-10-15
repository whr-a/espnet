"""Native ESPnet training worker implementing Miles' TrainRayActor contract."""

import copy
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train_actor import TrainRayActor
from miles.utils import object_store

from .config import ESPnetRLConfig
from .rollout import CheckpointRollout, save_checkpoint, sha256
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
        self._init_common(args, role, with_ref=with_ref)
        config = ESPnetRLConfig(**args.espnet_config)
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
        payload = [None, None, None]
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
                    behavior = None
                    if version != self.version:
                        config = copy.deepcopy(self.config)
                        config.checkpoint = str(checkpoint)
                        behavior = (
                            build_policy(config, "cpu").eval().requires_grad_(False)
                        )
                    metrics = self.trainer.prepare_async_batch(
                        batch,
                        behavior,
                        behavior_version=version,
                        learner_version=self.version,
                        behavior_checkpoint_sha256=actual_hash,
                    )
                    del behavior
                batch.save_to_disk(
                    Path(self.config.output_dir) / f"rollout-{rollout_id:06d}.pkl"
                )
                payload = [batch, metrics, None]
            except Exception as error:
                Path(
                    self.config.output_dir, f"rollout-{rollout_id:06d}.error.txt"
                ).write_text(traceback.format_exc())
                payload[2] = f"{type(error).__name__}: {error}"
        if self._world_size > 1:
            from miles.utils.distributed_utils import get_gloo_group

            dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
        batch, metrics, error = payload
        if error:
            raise RuntimeError(error)
        metrics.update(self.trainer.update_actor(batch))
        torch.cuda.synchronize()
        end = time.monotonic()
        self.version += 1
        if self._rank == 0:
            wait = 0.0 if self.last_end is None else start - self.last_end
            metrics.update(
                {
                    "step": self.version,
                    "perf/train_time": end - start,
                    "perf/train_wait_time": wait,
                    "perf/step_time": end - start + wait,
                    "perf/wait_time_ratio": wait / (end - start + wait),
                    "perf/warmup": self.version <= self.config.warmup_steps,
                    "perf/train_start_wall": time.time() - (end - start),
                    "perf/train_end_wall": time.time(),
                    "runtime/train_world_size": self._world_size,
                }
            )
            with Path(self.config.output_dir, "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            self.last_metrics = metrics
        self.last_end = end
        return TrainStepOutput(TrainStepOutcome.NORMAL)

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
