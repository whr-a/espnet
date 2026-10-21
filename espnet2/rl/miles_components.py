"""ESPnet Ray deployment, persistent rollout actors and weight publication."""

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
from .rollout import (
    RolloutProcess,
    merge_generations,
    prune_exports,
    record_publication,
)
from .utils import resolve_callable


class RolloutActor:
    """One rollout GPU: a rollout process that holds its engine for the run."""

    def __init__(self, config, replica):
        from .config import ESPnetRLConfig

        config = ESPnetRLConfig(**config)
        self.config = config
        self.replica = replica
        self.generation = 0
        self.identity = {
            "replica": replica,
            "node": ray.get_runtime_context().get_node_id(),
            "gpu_ids": ray.get_gpu_ids(),
            "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        }
        directory = Path(config.output_dir, "rollout_batches", f"replica-{replica}")
        if config.resume_checkpoint and directory.exists():
            directory = self._next_process_directory()
        self.process = RolloutProcess(
            config,
            os.environ["CUDA_VISIBLE_DEVICES"],
            directory,
            replica,
        )

    def _next_process_directory(self):
        """A recovery or evaluator restoration never overwrites prior batches."""
        while True:
            self.generation += 1
            directory = Path(
                self.config.output_dir,
                "rollout_batches",
                f"replica-{self.replica}-generation-{self.generation:06d}",
            )
            if not directory.exists():
                return directory

    def load(self, model, version):
        self.process.load(model, version)
        return {**self.identity, "version": version}

    def weight_update(self, method, args, kwargs):
        result = self.process.weight_update(method, args, kwargs)
        return {**result, **self.identity, "version": self.process.version}

    def generate(self, sources, rollout_id):
        start = time.time()
        generations = self.process.generate(sources, rollout_id)
        return {
            **self.identity,
            "version": self.process.version,
            "rollout_id": rollout_id,
            "start": start,
            "end": time.time(),
            "generations": generations,
        }

    def close(self):
        self.process.close()

    def evaluate(self, policy):
        """Temporarily reuse this actor's reserved GPU for a family evaluator.

        Only replica zero runs this method. Learner actors and the other rollout
        actor remain resident and idle; no additional Ray/GPU reservation occurs.
        The evaluator owns when to release and restore this actor's process so a
        verified completed evaluation can be reused without restarting inference.
        """
        if self.replica != 0 or self.process.version != policy["step"]:
            raise ValueError(
                "Periodic evaluation requires replica0 at the published policy"
            )
        function = resolve_callable(self.config.periodic_eval_function)

        def release():
            process = self.process
            process.close()
            return {
                "pid": process.process.pid,
                "exit_code": process.process.returncode,
                "closed": process.connection.closed,
            }

        def restore():
            directory = self._next_process_directory()
            self.process = RolloutProcess(
                self.config, self.identity["gpu"], directory, self.replica
            )
            self.process.load(str(Path(policy["path"], "hf")), policy["step"])
            return {
                **self.identity,
                "version": self.process.version,
                "pid": self.process.process.pid,
                "directory": str(directory),
            }

        return function(
            self.config,
            policy,
            self.identity,
            release_engine=release,
            restore_engine=restore,
        )


class NativeTrainerGroup:
    def __init__(self, workers):
        self.workers = workers
        self.disposed = False

    async def call(self, name, *args, **kwargs):
        from miles.utils.async_utils import gather_and_raise_first

        # Every rank must finish its checkpoint/exception protocol before an
        # exception reaches with_disposer and Miles destroys the actor group.
        return await gather_and_raise_first(
            [getattr(worker, name).remote(*args, **kwargs) for worker in self.workers]
        )

    async def train(self, rollout_id, data):
        return await self.call("train", rollout_id, data.data_ref)

    async def save_model(self, rollout_id, force_sync=False):
        return await self.call("save_model", rollout_id, force_sync=force_sync)

    async def preserve_rollout_failure(self, rollout_id, original_error):
        return await self.call("preserve_rollout_failure", rollout_id, original_error)

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
        self.actors = []
        self.policy = None
        self.in_flight = False
        self._active_get_task = None
        self.disposed = False

    def check_placement(self, replies, version):
        """Every reply names its own GPU on the rollout node and the policy."""
        replicas = self.components.rollout_replicas
        if (
            len(replies) != replicas
            or any(
                reply["node"] != self.components.rollout_node
                or len(reply["gpu_ids"]) != 1
                or reply["gpu"] != str(reply["gpu_ids"][0])
                for reply in replies
            )
            or len({tuple(reply["gpu_ids"]) for reply in replies}) != replicas
        ):
            raise ValueError(
                f"Rollout replicas did not occupy {replicas} distinct GPUs "
                "on the requested node"
            )
        if self.components.train_node == self.components.rollout_node:
            # Ray device ids are per node, so this comparison only means
            # anything when both roles sit on the same one.
            occupied = self.components.train_gpu_ids
            held = {str(gpu) for reply in replies for gpu in reply["gpu_ids"]}
            if occupied & held:
                raise ValueError("A rollout replica landed on a training GPU")
        if any(reply["version"] != version for reply in replies):
            raise ValueError("A rollout replica holds a different policy version")

    async def get(self, rollout_id):
        from miles.utils import object_store
        from miles.utils.data import RolloutDataPack

        if self.policy is None or self.in_flight or self.disposed:
            raise ValueError("Missing policy or overlapping rollout batches")
        self.in_flight = True
        self._active_get_task = asyncio.current_task()
        try:
            config = self.components.config
            from .training_objectives import select_training_sources

            sources = [
                copy.deepcopy(source)
                for source in select_training_sources(
                    self.components.sources, rollout_id, config
                )
            ]
            if len({source["id"] for source in sources}) != config.prompts_per_step:
                raise ValueError(
                    f"A rollout requires {config.prompts_per_step} distinct prompts"
                )
            policy = dict(self.policy)
            replicas = len(self.actors)
            results = await asyncio.gather(
                *[
                    actor.generate.remote(sources[replica::replicas], rollout_id)
                    for replica, actor in enumerate(self.actors)
                ]
            )
            self.check_placement(results, policy["step"])
            for result in results:
                for generation in result["generations"]:
                    generation.update(
                        behavior_policy_version=policy["step"],
                        behavior_checkpoint_sha256=policy["native_sha256"],
                        behavior_hash_kind=policy.get("hash_kind", "native-file/v1"),
                    )
            generations = merge_generations(
                sources,
                [result["generations"] for result in results],
                config.group_size,
            )
            batch = collate_rollouts(sources, generations, config.group_size)
            if len(batch) != config.global_candidate_batch_size:
                raise ValueError(
                    f"Incomplete {config.global_candidate_batch_size}-candidate rollout"
                )
            payload = {
                "batch": batch,
                "behavior_policy_version": policy["step"],
                "behavior_checkpoint_sha256": policy["native_sha256"],
                "behavior_hash_kind": policy.get("hash_kind", "native-file/v1"),
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
            self.in_flight = False
            self._active_get_task = None

    async def save(self, rollout_id):
        return None

    async def dispose(self):
        if self.disposed:
            return
        self.disposed = True
        task = self._active_get_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            # An actor answers in order, so a close waits for its generation.
            await asyncio.wait_for(
                asyncio.gather(
                    *[actor.close.remote() for actor in self.actors],
                    return_exceptions=True,
                ),
                timeout=180,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            for actor in self.actors:
                ray.kill(actor, no_restart=True)


class NativeInferenceController:
    async def prepare_rollout(self, rollout_id):
        return None

    async def dispose(self):
        return None


class NativeComponents:
    def __init__(self, config, sources, train_node, rollout_node):
        config.validate()
        if config.weight_sync_backend == "miles_shm":
            raise ValueError("miles_shm requires the local collocated launcher")
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
        # One actor per rollout GPU for the whole run; each starts its rollout
        # process now and builds the engine from the first published policy.
        self.rollout.actors = [
            ray.remote(num_gpus=1, num_cpus=2, max_restarts=0)(RolloutActor)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=self.rollout_node, soft=False
                )
            )
            .remote(self.config.to_dict(), replica)
            for replica in range(self.rollout_replicas)
        ]
        return self.inference, self.rollout, None

    async def create_training_models(self, args, rollout_executor):
        actor_args = SimpleNamespace(
            espnet_config=self.config.to_dict(),
            espnet_source_count=len(self.sources),
            trainer_model_id=None,
            distributed_backend="nccl",
            distributed_timeout_minutes=self.config.distributed_timeout_minutes,
            debug_deterministic_collective=False,
            object_store_backend="ray",
            worker_comm_backend="ray",
            env_report="",
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
        versions = await group.call("get_training_version")
        if len(set(versions)) != 1:
            raise ValueError("Training ranks restored different optimizer steps")
        self.start_rollout_id = versions[0]
        if type(self.start_rollout_id) is not int or not (
            0 <= self.start_rollout_id <= args.num_rollout
        ):
            raise ValueError("Restored optimizer cursor is outside the update budget")
        args.start_rollout_id = self.start_rollout_id
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
        interval = self.config.periodic_eval_interval
        evaluation = (
            interval and rollout_id is not None and (rollout_id + 1) % interval == 0
        )
        fast = (
            self.config.weight_sync_backend == "miles"
            and self.rollout.policy is not None
            and not evaluation
        )
        results = (
            await actor_model.call("publish_async_weights", self.rollout.actors)
            if fast
            else await actor_model.call("export_async_snapshot")
        )
        policy = results[0]
        if any(value is not None for value in results[1:]):
            raise ValueError("Only rank 0 can publish an actor")
        # Each actor loads the export in place; its reply is the acknowledgment.
        load_start = time.monotonic()
        acknowledgments = (
            policy.pop("acknowledgments")
            if fast
            else await asyncio.gather(
                *[
                    actor.load.remote(str(Path(policy["path"], "hf")), policy["step"])
                    for actor in self.rollout.actors
                ]
            )
        )
        self.rollout.check_placement(acknowledgments, policy["step"])
        if self.config.profile_phases and not fast:
            policy["perf/weight_load_ack_time"] = time.monotonic() - load_start
        record_publication(self.config, policy)
        self.rollout.policy = policy
        with Path(self.config.output_dir, "weight_ack.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "version": policy["step"],
                        "sha256": policy["native_sha256"],
                        "hash_kind": policy.get("hash_kind", "native-file/v1"),
                        **{
                            name: policy[name]
                            for name in (
                                "hash_time",
                                "publish_time",
                                "weight_sync_backend",
                            )
                            if name in policy
                        },
                        "rollout_id": rollout_id,
                        "safe_point": True,
                        "rollout_ack": acknowledgments,
                        "ack_time": time.time(),
                    }
                )
                + "\n"
            )
        prune_exports(Path(policy["path"]).parent, self.config.keep_exports)
        if interval and policy["step"] and policy["step"] % interval == 0:
            # The synchronous driver awaits this safe point before requesting
            # any next rollout. The same four actors retain their reservations.
            await self.rollout.actors[0].evaluate.remote(policy)
