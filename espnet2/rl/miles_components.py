"""ESPnet Ray deployment, rollout isolation, and weight acknowledgements."""

import asyncio
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from .data import collate_rollouts
from .miles_native_actor import ESPnetTrainRayActor
from .rollout import sha256


def replica_generation(
    config, policy_path, sources, rollout_id, replica, expected_hash, policy_version
):
    """One GPU and one payload directory per engine, with complete prompt groups."""
    from .config import ESPnetRLConfig

    config = ESPnetRLConfig(**config)
    from .rollout_replicas import run_replica

    result = run_replica(
        config,
        policy_path,
        sources,
        rollout_id,
        replica,
        os.environ["CUDA_VISIBLE_DEVICES"],
        expected_hash,
        policy_version=policy_version,
    )
    result.update(
        node=ray.get_runtime_context().get_node_id(), gpu_ids=ray.get_gpu_ids()
    )
    return result


def cleanup_rollout_processes(output_dir):
    """Run on the rollout node and clean only sessions recorded by this run."""
    from concurrent.futures import ThreadPoolExecutor

    from .rollout_replicas import terminate_owned_process

    directories = [
        path.parent
        for path in Path(output_dir, "rollout_batches").glob(
            "batch-*/replica-*/process.json"
        )
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(terminate_owned_process, directories))


def acknowledge_policy(policy):
    """The rollout node verifies shared files before publication."""
    path = Path(policy["path"])
    if sha256(path / "native.pt") != policy["native_sha256"]:
        raise ValueError("Rollout node received a different native checkpoint")
    for relative, expected in policy["hf_files"].items():
        if sha256(path / "hf" / relative) != expected:
            raise ValueError("Rollout export differs from its frozen file manifest")
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "version": policy["step"],
        "native_sha256": policy["native_sha256"],
        "export_file_count": len(policy["hf_files"]),
    }


class NativeTrainerGroup:
    def __init__(self, workers):
        self.workers = workers
        self.disposed = False

    async def call(self, name, *args, **kwargs):
        return await asyncio.gather(
            *[getattr(worker, name).remote(*args, **kwargs) for worker in self.workers]
        )

    async def train(self, rollout_id, data):
        return await self.call("train", rollout_id, data.data_ref)

    async def save_model(self, rollout_id, force_sync=False):
        return await self.call("save_model", rollout_id, force_sync=force_sync)

    async def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        try:
            await asyncio.wait_for(self.call("dispose"), timeout=30)
        finally:
            for worker in self.workers:
                ray.kill(worker, no_restart=True)


class NativeRolloutExecutor:
    def __init__(self, components):
        self.components = components
        self.policy = None
        self.in_flight = False
        self._active_get_task = None
        self._tasks = []
        self.disposed = False

    async def get(self, rollout_id):
        from miles.utils import object_store
        from miles.utils.data import RolloutDataPack

        if self.policy is None or self.in_flight or self.disposed:
            raise ValueError("Missing policy or overlapping rollout batches")
        self.in_flight = True
        self._active_get_task = asyncio.current_task()
        try:
            config = self.components.config
            start = rollout_id * config.prompts_per_step
            all_sources = self.components.sources
            sources = [
                copy.deepcopy(all_sources[(start + i) % len(all_sources)])
                for i in range(config.prompts_per_step)
            ]
            if len({source["id"] for source in sources}) != config.prompts_per_step:
                raise ValueError(
                    f"A rollout requires {config.prompts_per_step} distinct prompts"
                )
            policy = dict(self.policy)
            tasks = self._tasks = []
            replicas = self.components.rollout_replicas
            for replica in range(replicas):
                subset = sources[replica::replicas]
                task = (
                    ray.remote(num_gpus=1, num_cpus=2, max_retries=0)(
                        replica_generation
                    )
                    .options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=self.components.rollout_node, soft=False
                        )
                    )
                    .remote(
                        config.to_dict(),
                        policy["path"],
                        subset,
                        rollout_id,
                        replica,
                        policy["native_sha256"],
                        policy["step"],
                    )
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            self._tasks = []
            if (
                any(
                    result["node"] != self.components.rollout_node
                    or len(result["gpu_ids"]) != 1
                    or result["gpu"] != str(result["gpu_ids"][0])
                    for result in results
                )
                or len({tuple(result["gpu_ids"]) for result in results}) != replicas
            ):
                raise ValueError(
                    f"Rollout replicas did not occupy {replicas} distinct GPUs "
                    "on the requested node"
                )
            if self.components.train_node == self.components.rollout_node:
                # Ray device ids are per node, so this comparison only means
                # anything when both roles sit on the same one.
                occupied = self.components.train_gpu_ids
                held = {str(gpu) for item in results for gpu in item["gpu_ids"]}
                if occupied & held:
                    raise ValueError("A rollout replica landed on a training GPU")
            from .rollout_replicas import merge_generations

            for result in results:
                for generation in result["generations"]:
                    generation.update(
                        behavior_policy_version=policy["step"],
                        behavior_checkpoint_sha256=policy["native_sha256"],
                    )
            generations = merge_generations(sources, results, config.group_size)
            batch = collate_rollouts(sources, generations, config.group_size)
            if len(batch) != config.global_candidate_batch_size:
                raise ValueError(
                    f"Incomplete {config.global_candidate_batch_size}-candidate rollout"
                )
            payload = {
                "batch": batch,
                "behavior_policy_version": policy["step"],
                "behavior_checkpoint": str(Path(policy["path"], "native.pt")),
                "behavior_checkpoint_sha256": policy["native_sha256"],
            }
            with Path(config.output_dir, "rollout_timing.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "rollout_id": rollout_id,
                            "behavior_version": policy["step"],
                            "replicas": [
                                {
                                    key: value
                                    for key, value in result.items()
                                    if key != "generations"
                                }
                                for result in results
                            ],
                        },
                        default=str,
                    )
                    + "\n"
                )
            return RolloutDataPack(
                data_ref=object_store.get_instance().put(payload),
                sample_indices=list(range(len(batch))),
            )
        finally:
            if self._tasks:
                for task in self._tasks:
                    ray.cancel(task, force=False, recursive=True)
                await self._cleanup_processes()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._tasks, return_exceptions=True), timeout=10
                    )
                except asyncio.TimeoutError:
                    for task in self._tasks:
                        ray.cancel(task, force=True, recursive=True)
                self._tasks = []
            self.in_flight = False
            self._active_get_task = None

    async def save(self, rollout_id):
        return None

    async def _cleanup_processes(self):
        await ray.remote(num_cpus=0.2, max_retries=0)(
            cleanup_rollout_processes
        ).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=self.components.rollout_node, soft=False
            )
        ).remote(
            self.components.config.output_dir
        )

    async def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        task = self._active_get_task
        if task is not None:
            task.cancel()
        for ref in self._tasks:
            ray.cancel(ref, force=False, recursive=True)
        # Cleanup remains on the exact rollout node and checks PID/start time.
        await self._cleanup_processes()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)


class NativeInferenceController:
    async def prepare_rollout(self, rollout_id):
        return None

    async def dispose(self):
        return None


class NativeComponents:
    def __init__(self, config, sources, train_node, rollout_node):
        config.validate()
        self.config, self.sources = config, sources
        self.train_node, self.rollout_node = train_node, rollout_node
        # Geometry comes from the recipe: one training rank and one rollout
        # replica per declared GPU. Both roles may share a node as long as
        # they hold disjoint devices.
        self.train_ranks = config.train_ranks
        self.rollout_replicas = config.rollout_replicas
        self.train_gpu_ids = set()
        if train_node == rollout_node and config.collocated:
            raise ValueError(
                "A shared node requires disjoint train_gpus and rollout_gpus"
            )
        self.rollout = NativeRolloutExecutor(self)
        self.inference = NativeInferenceController()
        self.trainer = None
        self._previous_object_store = None
        self._owned_object_store = None

    def initialize(self, args, *, disposer):
        from miles.utils import object_store

        self._previous_object_store = object_store._INSTANCE
        if self._previous_object_store is not None:
            raise ValueError("Start this orchestration with a fresh Miles object store")
        self._owned_object_store = object_store.init_instance(
            SimpleNamespace(object_store_backend="ray", worker_comm_backend="ray")
        )
        disposer.add(self.dispose)

    async def dispose(self):
        from miles.utils import object_store

        try:
            await self.rollout.dispose()
        finally:
            try:
                if self.trainer is not None:
                    await self.trainer.dispose()
            finally:
                if object_store._INSTANCE is self._owned_object_store:
                    object_store._INSTANCE = self._previous_object_store

    async def create_rollout_components(self, args):
        return self.inference, self.rollout, None

    async def create_training_models(self, args, rollout_executor):
        actor_args = SimpleNamespace(
            espnet_config=self.config.to_dict(),
            trainer_model_id=None,
            distributed_backend="nccl",
            distributed_timeout_minutes=self.config.distributed_timeout_minutes,
            debug_deterministic_collective=False,
            object_store_backend="ray",
            worker_comm_backend="ray",
            env_report_interval_seconds=0,
            save_debug_event_data=None,
        )
        workers = [
            ray.remote(num_gpus=1, num_cpus=2, max_restarts=0)(ESPnetTrainRayActor)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=self.train_node, soft=False
                )
            )
            .remote(
                args=actor_args,
                world_size=self.train_ranks,
                rank=rank,
                role="actor",
                cell_index=0,
            )
            for rank in range(self.train_ranks)
        ]
        group = NativeTrainerGroup(workers)
        self.trainer = group
        host, port = await workers[0].propose_master_addr_and_port.remote()
        await group.call(
            "configure_master_addr_and_port", master_addr=host, master_port=port
        )
        await group.call("init", actor_args, "actor", with_ref=True)
        census = await group.call("get_device_census")
        if (
            any(item["ray_node_id"] != self.train_node for item in census)
            or sorted(item["rank"] for item in census) != list(range(self.train_ranks))
            or len({tuple(item["ray_gpu_ids"]) for item in census}) != self.train_ranks
        ):
            raise ValueError(
                f"Training workers do not occupy {self.train_ranks} distinct GPUs "
                "on the requested node"
            )
        # Remember the occupied devices so a shared node can prove that the
        # rollout replicas never land on a training GPU.
        self.train_gpu_ids = {
            str(gpu) for item in census for gpu in item["ray_gpu_ids"]
        }
        Path(self.config.output_dir, "train_census.json").write_text(
            json.dumps(census, indent=2) + "\n"
        )
        self.trainer = group
        return group, None

    async def update_weights(
        self,
        args,
        actor_model,
        rollout_executor,
        inference_controller,
        *,
        rollout_id=None,
    ):
        if self.rollout.in_flight:
            raise ValueError("Weight publication attempted during generation")
        results = await actor_model.call("export_async_snapshot")
        policy = results[0]
        if any(value is not None for value in results[1:]):
            raise ValueError("Only rank 0 can publish an actor")
        checkpoint = Path(policy["path"], "native.pt")
        if sha256(checkpoint) != policy["native_sha256"]:
            raise ValueError("Published actor checksum mismatch")
        acknowledgment = (
            await ray.remote(num_cpus=1)(acknowledge_policy)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=self.rollout_node, soft=False
                )
            )
            .remote(policy)
        )
        if acknowledgment["node_id"] != self.rollout_node:
            raise ValueError("Weight acknowledgment did not run on the rollout node")
        if (
            acknowledgment["version"] != policy["step"]
            or acknowledgment["native_sha256"] != policy["native_sha256"]
            or acknowledgment["export_file_count"] != len(policy["hf_files"])
        ):
            raise ValueError("Weight acknowledgment differs from the published version")
        self.rollout.policy = policy
        with Path(self.config.output_dir, "weight_ack.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "version": policy["step"],
                        "sha256": policy["native_sha256"],
                        "rollout_id": rollout_id,
                        "safe_point": True,
                        "rollout_ack": acknowledgment,
                        "ack_time": time.time(),
                    }
                )
                + "\n"
            )
