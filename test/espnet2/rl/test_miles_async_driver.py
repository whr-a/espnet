"""Run the exact upstream loop against instrumented component contracts."""

import asyncio

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.miles_async_driver import load_async_loop, orchestration_args


def test_pinned_loop_overlaps_train_and_rollout_and_publishes_at_safe_points():
    from miles.utils import object_store
    from miles.utils.async_utils import with_disposer
    from miles.utils.data import RolloutDataPack

    events = []
    versions = []

    class Store:
        def remove(self, ref):
            events.append(("remove", ref))

    previous_store = object_store._INSTANCE
    object_store._INSTANCE = Store()

    class Components:
        def __init__(self):
            self.in_flight = False
            self.version = 0
            self.started = asyncio.Event()
            self.finish = asyncio.Event()

        def initialize(self, args, *, disposer):
            return None

        async def create_rollout_components(self, args):
            return self, self, None

        async def create_training_models(self, args, rollout_executor):
            return self, None

        async def prepare_rollout(self, rollout_id):
            return None

        async def get(self, rollout_id):
            assert not self.in_flight
            self.in_flight = True
            versions.append(self.version)
            events.append(("rollout_start", rollout_id))
            if rollout_id:
                self.started.set()
                await self.finish.wait()
            self.in_flight = False
            events.append(("rollout_end", rollout_id))
            return RolloutDataPack()

        async def train(self, rollout_id, data):
            events.append(("train_start", rollout_id))
            if rollout_id < 4:
                await self.started.wait()
                assert self.in_flight
                self.started.clear()
                self.finish.set()
                await asyncio.sleep(0)
                self.finish.clear()
            self.version += 1
            events.append(("train_end", rollout_id))

        async def save_model(self, rollout_id, force_sync=False):
            events.append(("save", rollout_id))

        async def save(self, rollout_id):
            return None

        async def update_weights(
            self,
            args,
            actor_model,
            rollout_executor,
            inference_controller,
            *,
            rollout_id=None,
        ):
            assert not self.in_flight
            events.append(("publish", self.version))

        async def dispose(self):
            return None

        async def drain(self):
            return None

    async def execute():
        config = ESPnetRLConfig(
            execution_mode="disagg_async",
            steps=5,
            warmup_steps=1,
            train_batch_size=32,
            n_samples_per_prompt=8,
            rollout_num_replicas=8,
            adapter_factory="unused:factory",
            exporter="unused:exporter",
            sglang_generate_factory="unused:generate",
        )
        components = Components()
        driver, evidence = load_async_loop(config, components)
        assert driver.__code__.co_filename.endswith("train_async.py")
        assert evidence["runtime"]["source_modified"] is False
        assert evidence["runtime"]["same_code_object"] is True
        await with_disposer(driver, orchestration_args(config))

    try:
        asyncio.run(execute())
    finally:
        object_store._INSTANCE = previous_store
    assert versions == [0, 0, 1, 2, 3]
    assert [event for event in events if event[0] == "train_end"] == [
        ("train_end", i) for i in range(5)
    ]
    assert [event for event in events if event[0] == "publish"] == [
        ("publish", i) for i in range(6)
    ]
    for update in range(4):
        assert events.index(("rollout_start", update + 1)) < events.index(
            ("train_end", update)
        )
        assert events.index(("rollout_end", update + 1)) < events.index(
            ("publish", update + 1)
        )
