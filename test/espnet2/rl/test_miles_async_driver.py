"""Run the exact upstream loop against instrumented component contracts."""

import asyncio
from types import SimpleNamespace

import pytest

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
            prompts_per_step=32,
            group_size=8,
            # One eight-GPU node per role, declared as eight device ids each.
            train_gpus="0,1,2,3,4,5,6,7",
            rollout_gpus="0,1,2,3,4,5,6,7",
            adapter_factory="unused:factory",
            exporter="unused:exporter",
            sglang_engine="unused:Engine",
        )
        components = Components()
        driver, evidence = load_async_loop(config, components)
        assert driver.__code__.co_filename.endswith("train_async.py")
        assert evidence["runtime"]["source_modified"] is False
        assert evidence["runtime"]["same_code_object"] is True
        arguments = orchestration_args(config)
        # The pinned loop owns the save cadence; it reads the recipe's interval.
        assert arguments.save_interval == config.save_interval == 1
        await with_disposer(driver, arguments)

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


@pytest.mark.parametrize("restored", [2, 6])
def test_pinned_async_loop_consumes_restored_native_cursor(
    tmp_path, monkeypatch, restored
):
    from miles.utils import object_store
    from miles.utils.async_utils import with_disposer
    from miles.utils.data import RolloutDataPack

    from espnet2.rl import miles_components
    from espnet2.rl.miles_components import NativeComponents

    events = []
    node_id = "1" * 56

    class Worker:
        version = restored

        def __getattr__(self, name):
            async def call(*args, **kwargs):
                if name == "get_training_version":
                    return self.version
                if name == "get_device_census":
                    return {"rank": 0, "ray_node_id": node_id, "ray_gpu_ids": [0]}
                if name == "propose_master_addr_and_port":
                    return "host", 12345
                if name == "train":
                    assert args[0] == self.version
                    events.append(("train", args[0]))
                    self.version += 1
                elif name == "save_model":
                    assert args[0] + 1 == self.version
                    events.append(("save", self.version))

            return SimpleNamespace(remote=call)

    worker = Worker()
    factory = SimpleNamespace(
        options=lambda **kwargs: factory, remote=lambda **kwargs: worker
    )
    monkeypatch.setattr(
        miles_components.ray, "remote", lambda **kwargs: lambda cls: factory
    )
    monkeypatch.setattr(miles_components.ray, "kill", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        object_store, "_INSTANCE", SimpleNamespace(remove=lambda ref: None)
    )

    class Components(NativeComponents):
        def initialize(self, args, *, disposer):
            pass

        async def create_rollout_components(self, args):
            return self, self, None

        async def prepare_rollout(self, rollout_id):
            pass

        async def get(self, rollout_id):
            events.append(("rollout", rollout_id))
            return RolloutDataPack()

        async def save(self, rollout_id):
            pass

        async def update_weights(self, args, *roles, rollout_id=None):
            events.append(("publish", worker.version))

        async def dispose(self):
            pass

    config = ESPnetRLConfig(
        execution_mode="disagg_async",
        train_gpus="0",
        rollout_gpus="1",
        steps=5,
        warmup_steps=1,
        output_dir=str(tmp_path),
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )
    components = Components(config, [], node_id, node_id)
    driver, evidence = load_async_loop(config, components)
    assert evidence["runtime"]["same_code_object"] is True

    async def execute():
        await with_disposer(driver, orchestration_args(config))

    if restored > config.steps:
        with pytest.raises(ValueError, match="outside the update budget"):
            asyncio.run(execute())
        assert not events
    else:
        asyncio.run(execute())
        assert [value for event, value in events if event == "rollout"] == [2, 3, 4]
        assert [value for event, value in events if event == "train"] == [2, 3, 4]
        assert [value for event, value in events if event == "save"] == [3, 4, 5]
        assert [value for event, value in events if event == "publish"] == [2, 3, 4, 5]
