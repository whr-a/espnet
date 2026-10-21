"""Native ESPnet training worker implementing Miles' TrainRayActor contract."""

import copy
import json
import os
import time
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train_actor import TrainRayActor
from miles.utils import object_store

from espnet2.torch_utils.set_all_random_seed import set_all_random_seed

from .config import ESPnetRLConfig
from .failure_checkpoint import preserve_preupdate_failure
from .rollout import CheckpointRollout
from .scoring import score_native_batch
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
        set_all_random_seed(config.seed)
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
        self.exported_hash_kinds = {}
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

    def preserve_rollout_failure(self, rollout_id, original_error):
        """Collectively preserve idle learners before failed generation disposal.

        The synchronous driver dispatches this RPC to every rank and awaits
        every reply. No train RPC for this rollout has started.
        """
        return preserve_preupdate_failure(
            self,
            rollout_id,
            original_error,
            failure_stage="rollout_generation_before_update",
        )

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
                    kind = getattr(self, "exported_hash_kinds", {}).get(
                        version, "native-file/v1"
                    )
                    if data.get("behavior_hash_kind", "native-file/v1") != kind or any(
                        generation.get("behavior_hash_kind", "native-file/v1") != kind
                        for generation in batch.non_tensor_batch["generation"]
                    ):
                        raise ValueError(
                            "Behavior digest protocol differs from publication"
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

        def prepare(scores):
            if self.config.execution_mode != "disagg_async":
                return self._prepare_synchronous_batch(batch, data, scores)
            return self.trainer.prepare_async_batch(
                batch,
                behavior,
                behavior_version=version,
                learner_version=self.version,
                behavior_checkpoint_sha256=data["behavior_checkpoint_sha256"],
                native_scores=scores,
            )

        def preserve(error):
            failure = RuntimeError(error)
            try:
                result = preserve_preupdate_failure(self, rollout_id, error)
                failure.add_note(
                    "Pre-update checkpoint preservation: "
                    + json.dumps(result, allow_nan=False)
                )
            except Exception as checkpoint_error:
                failure.add_note(
                    "Checkpoint preservation failed: "
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                )
            raise failure from None

        metrics = self.trainer.train_batch(
            batch,
            rollout_id,
            behavior=behavior,
            prepare=prepare,
            snapshot=self.snapshot,
            on_prepare_error=preserve,
            scorer=score_native_batch,
        )
        self.version = self.trainer.completed_updates
        self.last_end = self.trainer.record_metrics(
            metrics, start, start_wall, self.last_end
        )
        if self._rank == 0:
            self.last_metrics = metrics
        return TrainStepOutput(TrainStepOutcome.NORMAL)

    def _prepare_synchronous_batch(self, batch, data, native_scores):
        """Validate the on-policy global groups before WEST reward normalization."""
        if (
            data["behavior_version"] != self.version
            or len(batch) != self.config.global_candidate_batch_size
        ):
            raise ValueError("Synchronous training requires the current complete batch")
        self.trainer.validate_behavior_batch(
            batch, self.version, data["behavior_checkpoint_sha256"]
        )
        if not torch.equal(native_scores[0], native_scores[1]):
            raise ValueError("Synchronous native behavior and current scores differ")
        metrics = self.trainer.prepare_batch(
            batch, native_scores=(native_scores[0], native_scores[2])
        )
        return {
            **metrics,
            "rollout/current_behavior_logprob_max_difference": 0.0,
            **self.trainer.behavior_metrics(batch, self.version, self.version),
        }

    def _broadcast_payload(self, payload):
        if self._world_size > 1:
            dist.broadcast_object_list(
                payload, src=0, group=self.trainer._object_collective_group()
            )

    def _record_rollout_error(self, rollout_id):
        self.trainer.record_rollout_error(rollout_id)

    def save_model(self, rollout_id, force_sync=False):
        if rollout_id + 1 != self.version:
            raise ValueError("Checkpoint update identity mismatch")
        self.trainer.save_training_checkpoint(
            Path(self.config.output_dir, f"step-{self.version:06d}.pt"),
            self.version,
        )

    def export_async_snapshot(self):
        if self._rank != 0:
            return None
        # A periodic evaluator can replace the rollout process; recreate its
        # NCCL group at the following in-memory publication.
        self.close_weight_publisher()
        self.exporter.publisher = None
        # Export only: the rollout actors load it onto the rollout GPUs.
        metadata = self.exporter.export(self.trainer.policy, self.version)
        self.exported[self.version] = metadata["native_sha256"]
        self.exported_hash_kinds[self.version] = "native-file/v1"
        return {**metadata, "path": str(self.exporter.version.resolve())}

    def publish_async_weights(self, actors):
        if self._rank != 0:
            return None
        from .weight_transport import EngineClient, MilesWeightPublisher

        if self.exporter.publisher is None:
            self.exporter.publisher = MilesWeightPublisher(
                [EngineClient(actor, remote=True) for actor in actors],
                collocated=False,
                sglang_source=self.config.sglang_source,
            )
        metadata = self.exporter.publisher.publish(self.trainer.policy, self.version)
        self.exported[self.version] = metadata["native_sha256"]
        self.exported_hash_kinds[self.version] = metadata["hash_kind"]
        return {**metadata, "path": str(self.exporter.version.resolve())}

    def close_weight_publisher(self):
        if self._rank == 0 and self.exporter.publisher is not None:
            self.exporter.publisher.close()

    def update_weights(self, info):
        raise ValueError(
            "Native publication uses export_async_snapshot or publish_async_weights"
        )

    def dispose(self):
        try:
            self.close_weight_publisher()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
