"""Persistent rollout processes with a CPU stand-in for an architecture engine.

Each test starts real worker interpreters over the real socket protocol. The
engine below records what it was asked to do instead of running a model.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.engine import SGLangRolloutEngine
from espnet2.rl.rollout import CheckpointRollout, RolloutProcess, read_json

ENGINE = """
import json, os, time
import numpy as np
from espnet2.rl.engine import RolloutEngine


class RecordingEngine(RolloutEngine):
    def __init__(self, config, model_path, *, directory, replica=0):
        super().__init__(config, model_path, directory=directory, replica=replica)
        self.model = str(model_path)
        self.record("build", self.model)

    def record(self, *event):
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\\n")

    def update(self, model_path):
        self.model = str(model_path)
        self.record("update", self.model)

    def generate(self, sources, rollout_id):
        self.record("generate", rollout_id)
        if sources[0].get("fail") == "raise":
            raise ValueError("deliberate engine failure")
        if sources[0].get("fail") == "exit":
            os._exit(3)
        if sources[0].get("fail") == "hang":
            time.sleep(600)
        # One array per source, shared by all its generations.
        arrays = {
            source["id"]: {"features": np.full(source["array"], 0.5, np.float32)}
            for source in sources
            if "array" in source
        }
        rows = [
            dict(
                source_id=source["id"],
                prompt_token_ids=[1],
                token_ids=[sample + 1],
                log_probs=[-1.0],
                model=self.model,
                replica=self.replica,
                gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                rank=os.environ.get("RANK"),
                **arrays.get(source["id"], {}),
            )
            for source in sources
            for sample in range(self.config.group_size)
        ]
        return rows[:-1] if sources[-1].get("fail") == "short" else rows

    def release(self):
        self.record("release")

    def close(self):
        self.record("close")
"""


@pytest.fixture
def config(tmp_path, monkeypatch):
    (tmp_path / "recording_engine.py").write_text(ENGINE)
    # The worker inherits this path after its own source paths.
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])
    )
    # A training rank's identity never reaches a rollout process.
    monkeypatch.setenv("RANK", "5")
    return ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="recording_engine:RecordingEngine",
        rollout_python=sys.executable,
        output_dir=str(tmp_path / "run"),
        prompts_per_step=4,
        group_size=2,
    )


def events(directory):
    lines = (Path(directory) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_one_engine_serves_every_load_and_generation(config, tmp_path):
    directory = tmp_path / "run" / "replica-0"
    process = RolloutProcess(config, "6", directory, 0)
    sources = [{"id": "a"}, {"id": "b"}]
    try:
        process.load(tmp_path / "policy-000000" / "hf", 0)
        first = process.generate(sources, 0)
        process.load(tmp_path / "policy-000001" / "hf", 1)
        second = process.generate(sources, 1)
    finally:
        process.close()
    # Built once from the first export and updated in place by the second;
    # training shares this GPU, so every batch gives its memory back.
    assert [event[0] for event in events(directory)] == [
        "build",
        "generate",
        "release",
        "update",
        "generate",
        "release",
        "close",
    ]
    assert events(directory)[3][1] == str(tmp_path / "policy-000001" / "hf")
    assert [row["source_id"] for row in first] == ["a", "a", "b", "b"]
    assert {row["policy_version"] for row in first} == {0}
    assert {row["policy_version"] for row in second} == {1}
    assert second[0]["model"].endswith("policy-000001/hf")
    assert {row["gpu"] for row in first} == {"6"} and first[0]["rank"] is None
    # Requests and answers of every rollout stay for audit.
    assert json.loads((directory / "requests-000001.json").read_text()) == sources
    recorded = json.loads((directory / "responses-000001.json").read_text())
    assert recorded["rollout_id"] == 1 and recorded["policy_version"] == 1
    assert process.process.returncode == 0
    process.close()  # Closing twice is harmless.


def test_arrays_travel_once_each_as_binary_sidecars(config, tmp_path):
    directory = tmp_path / "replica-0"
    process = RolloutProcess(config, "0", directory, 0)
    try:
        process.load(tmp_path / "hf", 0)
        sources = [{"id": "a", "array": 3}, {"id": "b"}, {"id": "c", "array": 2}]
        generations = process.generate(sources, 4)
    finally:
        process.close()
    arrays = [row.get("features") for row in generations]
    # Each array crossed once, and the generations of a source share it.
    assert arrays[0] is arrays[1] and arrays[4] is arrays[5] and arrays[2] is None
    np.testing.assert_array_equal(arrays[0], np.full(3, 0.5, np.float32))
    assert arrays[4].dtype == np.float32 and arrays[4].shape == (2,)
    assert sorted(path.name for path in directory.glob("*.npy")) == [
        "responses-000004.0.npy",
        "responses-000004.1.npy",
    ]
    np.save(directory / "responses-000004.1.npy", np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="differs from its record"):
        read_json(directory / "responses-000004.json")


def test_engine_errors_reach_the_client_and_the_worker_keeps_serving(config, tmp_path):
    process = RolloutProcess(config, "0", tmp_path / "replica-0", 0)
    try:
        with pytest.raises(RuntimeError, match="Load a policy before generation"):
            process.generate([{"id": "a"}], 0)
        process.load(tmp_path / "hf", 3)
        with pytest.raises(RuntimeError, match="deliberate engine failure"):
            process.generate([{"id": "a", "fail": "raise"}], 1)
        assert len(process.generate([{"id": "a"}], 2)) == 2
        # That batch released the engine to the training rank on its GPU.
        with pytest.raises(RuntimeError, match="Load a policy before generation"):
            process.generate([{"id": "a"}], 3)
        process.load(tmp_path / "hf", 4)
        assert len(process.generate([{"id": "a"}], 4)) == 2
    finally:
        process.close()
    assert process.process.returncode == 0


@pytest.mark.parametrize("memory_saver", [True, False])
def test_release_hands_back_weights_only_with_the_memory_saver(
    memory_saver, tmp_path, monkeypatch
):
    calls = []

    class Engine:
        """Stands in for sglang.Engine and records what it is asked."""

        def __init__(self, **settings):
            calls.append(("build", settings["enable_memory_saver"]))
            self.loop = asyncio.new_event_loop()
            self.tokenizer_manager = SimpleNamespace(flush_cache=self.flush_cache)

        async def flush_cache(self, timeout_s):
            calls.append(("flush", timeout_s))
            return SimpleNamespace(success=True, message="")

        def update_weights_from_disk(self, model_path):
            calls.append(("load", Path(model_path).name))
            return True, "", 0

        def release_memory_occupation(self):
            calls.append(("release",))

        def resume_memory_occupation(self):
            calls.append(("resume",))

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        sglang_engine_kwargs={"enable_memory_saver": memory_saver},
    )
    engine = SGLangRolloutEngine(config, tmp_path / "p0", directory=tmp_path)
    try:
        engine.release()
        engine.update(tmp_path / "p1")
        engine.update(tmp_path / "p2")
        engine.release()
    finally:
        engine.engine.loop.close()
    if memory_saver:
        # The weights and KV pool come back before the next policy fills them,
        # and only when they were given away.
        assert calls == [
            ("build", True),
            ("flush", 60),
            ("release",),
            ("resume",),
            ("load", "p1"),
            ("load", "p2"),
            ("flush", 60),
            ("release",),
        ]
    else:
        assert calls == [
            ("build", False),
            ("flush", 60),
            ("load", "p1"),
            ("load", "p2"),
            ("flush", 60),
        ]


def test_a_dead_worker_is_reported_with_its_log(config, tmp_path):
    process = RolloutProcess(config, "0", tmp_path / "replica-0", 0)
    try:
        process.load(tmp_path / "hf", 0)
        with pytest.raises(RuntimeError, match="exited") as failure:
            process.generate([{"id": "a", "fail": "exit"}], 0)
        # The log tail names what the worker did last.
        assert "rollout engine built for policy 0" in str(failure.value)
        with pytest.raises(RuntimeError, match="exited"):
            process.load(tmp_path / "hf", 1)
    finally:
        process.close()
    assert process.process.returncode == 3


CLIENT = """
import json, sys, threading
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.rollout import RolloutProcess

process = RolloutProcess(ESPnetRLConfig(**json.loads(sys.argv[1])), "0", sys.argv[2], 0)
process.load(sys.argv[2], 0)
batch = [{"id": "a", "fail": "hang"}]
threading.Thread(target=process.generate, args=(batch, 0)).start()
print(process.process.pid, flush=True)
"""


def alive(pid):
    try:
        return "zombie" not in Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False


def test_a_worker_dies_with_its_client(config, tmp_path):
    """A client killed mid-batch takes its worker's whole session along."""
    directory = tmp_path / "replica-0"
    client = subprocess.Popen(
        [sys.executable, "-c", CLIENT, json.dumps(config.to_dict()), str(directory)],
        stdout=subprocess.PIPE,
        text=True,
    )
    worker = int(client.stdout.readline())
    try:
        deadline = time.monotonic() + 60
        while "generate" not in {event[0] for event in events(directory)}:
            assert time.monotonic() < deadline
            time.sleep(0.1)
        client.kill()
        client.wait()
        deadline = time.monotonic() + 20
        while alive(worker):
            assert time.monotonic() < deadline, "the worker outlived its client"
            time.sleep(0.2)
    finally:
        client.kill()
        if alive(worker):
            os.killpg(worker, signal.SIGKILL)


def test_checkpoint_rollout_shards_sources_and_merges_complete_groups(config):
    config.train_gpus, config.rollout_gpus = "7", "0,1,2"
    rollout = CheckpointRollout(config)
    export = rollout.directory / "policy-000000"
    sources = [{"id": f"p{index}"} for index in range(7)]
    with pytest.raises(RuntimeError, match="synchronized"):
        rollout.generate(sources, 0)
    try:
        rollout.load(export, 0)
        generations = rollout.generate(sources, 0)
        with pytest.raises(ValueError, match="own prompt shard"):
            rollout.generate(sources[:2], 1)
        with pytest.raises(ValueError, match="complete prompt group"):
            rollout.generate([*sources[:6], {"id": "p6", "fail": "short"}], 2)
    finally:
        rollout.close()
    assert rollout.processes == []
    # Every source keeps its complete group in the original order, and each
    # replica sampled only its own strided shard on its own GPU.
    assert [row["source_id"] for row in generations] == [
        source["id"] for source in sources for _ in range(2)
    ]
    assert {row["source_id"]: (row["replica"], row["gpu"]) for row in generations} == {
        f"p{index}": (index % 3, str(index % 3)) for index in range(7)
    }
    for replica in range(3):
        directory = rollout.directory / f"replica-{replica}"
        requests = json.loads((directory / "requests-000000.json").read_text())
        assert requests == sources[replica::3]
        assert events(directory)[0] == ["build", str((export / "hf").resolve())]
        # No training rank shares these GPUs, so nothing is released.
        assert "release" not in {event[0] for event in events(directory)}


@pytest.mark.parametrize("keep", [0, 1.5, None])
def test_keep_exports_is_a_positive_integer(config, keep):
    assert config.keep_exports == 2
    config.keep_exports = 1
    config.validate()
    config.keep_exports = keep
    with pytest.raises(ValueError, match="keep_exports"):
        config.validate()


def test_sync_keeps_the_newest_exports_once_every_engine_loaded_one(config):
    config.train_gpus, config.rollout_gpus = "7", "0,1"
    rollout = CheckpointRollout(config)

    def export(policy, step):
        directory = rollout.directory / f"policy-{step:06d}"
        (directory / "hf").mkdir(parents=True)
        rollout.version = directory
        return {"step": step}

    rollout.export = export
    kept = []
    try:
        for step in range(4):
            rollout.sync_weights(None, step)
            kept.append(
                sorted(path.name for path in rollout.directory.glob("policy-*"))
            )
    finally:
        rollout.close()
    assert kept == [
        ["policy-000000"],
        ["policy-000000", "policy-000001"],
        ["policy-000001", "policy-000002"],
        ["policy-000002", "policy-000003"],
    ]
    for replica in range(2):
        loads = [
            Path(event[1]).parent.name
            for event in events(rollout.directory / f"replica-{replica}")
            if event[0] in ("build", "update")
        ]
        assert loads == [f"policy-{step:06d}" for step in range(4)]


class Actor:
    """A rollout actor handle whose calls answer like RolloutActor."""

    def __init__(self, replica, node="rollout", gpu=None):
        self.identity = {
            "replica": replica,
            "node": node,
            "gpu_ids": [replica if gpu is None else gpu],
            "gpu": str(replica if gpu is None else gpu),
        }
        self.calls, self.version = [], None
        for name in ("load", "generate", "close"):
            setattr(self, name, SimpleNamespace(remote=self._remote(name)))

    def _remote(self, name):
        async def call(*args):
            self.calls.append((name, *args))
            if name == "load":
                self.version = args[1]
                return {**self.identity, "version": self.version}
            if name == "generate":
                sources, rollout_id = args
                generations = [
                    dict(
                        source_id=source["id"],
                        prompt_token_ids=[1],
                        token_ids=[sample + 1],
                        log_probs=[-1.0],
                        policy_version=self.version,
                    )
                    for source in sources
                    for sample in range(2)
                ]
                return {
                    **self.identity,
                    "version": self.version,
                    "rollout_id": rollout_id,
                    "start": 0.0,
                    "end": 1.0,
                    "generations": generations,
                }

        return call


def test_async_rollout_publishes_to_every_actor_and_stamps_its_batch(
    tmp_path, monkeypatch
):
    pytest.importorskip("miles")
    from miles.utils import object_store

    from espnet2.rl.miles_components import NativeComponents

    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        execution_mode="disagg_async",
        output_dir=str(tmp_path),
        steps=2,
        prompts_per_step=4,
        group_size=2,
        train_gpus="0",
        rollout_gpus="0,1",
    )
    stored = []
    monkeypatch.setattr(
        object_store,
        "_INSTANCE",
        SimpleNamespace(put=lambda value: stored.append(value)),
    )
    sources = [{"id": f"p{index}"} for index in range(4)]
    components = NativeComponents(config, sources, "train", "rollout")
    actors = components.rollout.actors = [Actor(0), Actor(1)]
    policy = {
        "step": 0,
        "path": str(tmp_path / "policy-000000"),
        "native_sha256": "f" * 64,
        "hf_files": {},
    }

    class Trainer:
        async def call(self, name):
            assert name == "export_async_snapshot"
            return [policy]

    async def run():
        await components.update_weights(None, Trainer(), None, None, rollout_id=None)
        await components.rollout.get(0)
        # A placement that does not match the recipe is refused.
        actors[1].identity["node"] = "elsewhere"
        with pytest.raises(ValueError, match="distinct GPUs"):
            await components.rollout.get(1)
        actors[1].identity["node"] = "rollout"
        await components.rollout.dispose()

    killed = []
    monkeypatch.setattr(
        "espnet2.rl.miles_components.ray.kill",
        lambda actor, no_restart: killed.append(actor),
    )
    asyncio.run(run())
    for replica, actor in enumerate(actors):
        assert actor.calls[0] == ("load", str(tmp_path / "policy-000000" / "hf"), 0)
        assert actor.calls[1] == ("generate", sources[replica::2], 0)
        assert actor.calls[-1] == ("close",)
    assert killed == actors
    (acknowledgment,) = [
        json.loads(line)
        for line in (tmp_path / "weight_ack.jsonl").read_text().splitlines()
    ]
    assert [reply["replica"] for reply in acknowledgment["rollout_ack"]] == [0, 1]
    assert {reply["version"] for reply in acknowledgment["rollout_ack"]} == {0}
    (payload,) = stored
    # The behavior audit swaps in the learner's own snapshot: no file path.
    assert set(payload) == {
        "batch",
        "behavior_policy_version",
        "behavior_checkpoint_sha256",
        "behavior_hash_kind",
    }
    generations = payload["batch"].non_tensor_batch["generation"]
    assert [row["source_id"] for row in generations] == [
        source["id"] for source in sources for _ in range(2)
    ]
    assert {row["behavior_policy_version"] for row in generations} == {0}
    assert {row["behavior_checkpoint_sha256"] for row in generations} == {"f" * 64}
    timing = json.loads((tmp_path / "rollout_timing.jsonl").read_text())
    assert [replica["gpu"] for replica in timing["replicas"]] == ["0", "1"]


def test_async_publication_keeps_the_newest_exports_after_each_load(
    tmp_path, monkeypatch
):
    pytest.importorskip("miles")
    from espnet2.rl.miles_components import NativeComponents

    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        execution_mode="disagg_async",
        output_dir=str(tmp_path),
        steps=4,
        prompts_per_step=2,
        group_size=2,
        train_gpus="0",
        rollout_gpus="0,1",
        keep_exports=1,
    )
    components = NativeComponents(config, [], "train", "rollout")
    actors = components.rollout.actors = [Actor(0), Actor(1)]
    exports = tmp_path / "rollout"

    class Trainer:
        step = -1

        async def call(self, name):
            self.step += 1
            directory = exports / f"policy-{self.step:06d}"
            (directory / "hf").mkdir(parents=True)
            return [
                {"step": self.step, "path": str(directory), "native_sha256": "f" * 64}
            ]

    async def run():
        trainer = Trainer()
        kept = []
        for _ in range(3):
            await components.update_weights(None, trainer, None, None)
            kept.append(sorted(path.name for path in exports.iterdir()))
        return kept

    assert asyncio.run(run()) == [
        ["policy-000000"],
        ["policy-000001"],
        ["policy-000002"],
    ]
    for actor in actors:
        assert [call[2] for call in actor.calls] == [0, 1, 2]
