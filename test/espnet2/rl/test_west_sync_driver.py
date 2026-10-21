"""Ordered rollout/update/save/evaluation and resume cursor contracts."""

import asyncio
import json

from espnet2.bin.rl_launch import launch_command, parser
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.miles_async_driver import load_async_loop, orchestration_args


def config():
    return ESPnetRLConfig(
        family="bagpiper",
        training_objective="west_grpo",
        execution_mode="sync",
        loss_agg_mode="seq-mean-token-mean",
        steps=5,
        save_interval=3,
        train_gpus="0,1",
        rollout_gpus="2,3",
        top_k=50,
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )


def test_sync_ray_launch_routes_to_native_actor_driver(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config().to_dict()))
    args = parser().parse_args(
        [
            "--config",
            str(path),
            "--ray-address",
            "ray",
            "--train-node",
            "n",
            "--rollout-node",
            "n",
        ]
    )
    command, _ = launch_command(args, python="python")
    assert command[:3] == ["python", "-m", "espnet2.bin.rl_train"]
    assert command[command.index("--worker") + 1] == "ray"


def test_sync_driver_has_no_stale_overlap_and_saves_before_evaluation_on_resume():
    from miles.utils import object_store
    from miles.utils.async_utils import with_disposer
    from miles.utils.data import RolloutDataPack

    events = []

    class Store:
        def remove(self, ref):
            events.append(("remove", ref))

    class Components:
        start_rollout_id = 2
        version = 2
        busy = False
        saved = {2}

        def initialize(self, args, *, disposer):
            pass

        async def create_rollout_components(self, args):
            return self, self, None

        async def create_training_models(self, args, rollout):
            return self, None

        async def prepare_rollout(self, rollout_id):
            assert not self.busy

        async def get(self, rollout_id):
            assert not self.busy
            assert rollout_id == self.version
            self.busy = True
            events.append(("generate", rollout_id))
            await asyncio.sleep(0)
            self.busy = False
            return RolloutDataPack()

        async def train(self, rollout_id, data):
            assert not self.busy
            assert rollout_id == self.version
            self.busy = True
            events.append(("train", rollout_id))
            await asyncio.sleep(0)
            self.version += 1
            self.busy = False

        async def save_model(self, rollout_id, force_sync=False):
            assert not self.busy
            assert rollout_id + 1 == self.version
            events.append(("save", self.version))
            self.saved.add(self.version)

        async def save(self, rollout_id):
            pass

        async def update_weights(
            self, args, actor, rollout, inference, *, rollout_id=None
        ):
            assert not self.busy
            events.append(("publish", self.version))
            if self.version % 2 == 0:
                assert self.version in self.saved
                self.busy = True
                events.append(("eval_start", self.version))
                await asyncio.sleep(0)
                events.append(("eval_end", self.version))
                self.busy = False

        async def dispose(self):
            pass

    async def execute():
        cfg = config()
        components = Components()
        driver, evidence = load_async_loop(cfg, components)
        assert evidence["runtime"]["same_code_object"] is False
        assert evidence["runtime"]["source_modified"] is False
        args = orchestration_args(cfg)
        args.native_eval_interval = 2
        assert args.use_tis is False
        await with_disposer(driver, args)
        assert args.start_rollout_id == 2

    previous = object_store._INSTANCE
    object_store._INSTANCE = Store()
    try:
        asyncio.run(execute())
    finally:
        object_store._INSTANCE = previous
    assert [event for event in events if event[0] == "generate"] == [
        ("generate", 2),
        ("generate", 3),
        ("generate", 4),
    ]
    assert [event for event in events if event[0] == "save"] == [
        ("save", 3),
        ("save", 4),
        ("save", 5),
    ]
    assert (
        events.index(("save", 4))
        < events.index(("publish", 4))
        < events.index(("eval_end", 4))
        < events.index(("generate", 4))
    )
