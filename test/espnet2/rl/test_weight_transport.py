"""CPU contracts for identity, original Miles transport calls and safe lifetimes."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.engine import SGLangRolloutEngine
from espnet2.rl.rollout import CheckpointRollout
from espnet2.rl.rollout_worker import RolloutServer
from espnet2.rl.weight_transport import (
    CODEC,
    HASH_KIND,
    MARKER,
    EngineClient,
    MilesWeightPublisher,
    tensor_digest,
    verify_weights,
)


def sealed(weights, step=3):
    identity = dict(
        step=step, native_sha256=tensor_digest(weights), hash_kind=HASH_KIND
    )
    marker = torch.tensor(list(json.dumps(identity).encode()), dtype=torch.uint8)
    return [*weights, (MARKER, marker)], identity


def test_digest_binds_names_shapes_dtypes_and_frozen_values(tmp_path):
    tensors = [("actor", torch.arange(6.0).reshape(2, 3)), ("frozen", torch.ones(2))]
    digest = tensor_digest(tensors)
    assert tensor_digest(tensors[::-1]) == digest
    changes = [
        [("renamed", tensors[0][1]), tensors[1]],
        [("actor", tensors[0][1].reshape(3, 2)), tensors[1]],
        [("actor", tensors[0][1].double()), tensors[1]],
        [tensors[0], ("frozen", torch.zeros(2))],
    ]
    assert all(tensor_digest(changed) != digest for changed in changes)
    with pytest.raises(ValueError, match="Repeated"):
        tensor_digest([*tensors, tensors[0]])
    payload, identity = sealed(tensors)
    receipt = tmp_path / "verified.json"
    assert verify_weights(payload, receipt) == tensors
    assert json.loads(receipt.read_text()) == identity
    receipt.unlink()
    payload[1][1].zero_()
    with pytest.raises(ValueError, match="SHA256"):
        verify_weights(payload, receipt)
    assert not receipt.exists()
    assert verify_weights(tensors, receipt) == tensors  # Legacy disk bootstrap.
    assert not receipt.exists()


@pytest.mark.parametrize("change", ["duplicate", "protocol", "version", "digest"])
def test_receiver_rejects_invalid_identity(tmp_path, change):
    weights = [("weight", torch.ones(2))]
    payload, identity = sealed(weights)
    if change == "duplicate":
        payload.append(payload[-1])
    else:
        identity[
            {"protocol": "hash_kind", "version": "step", "digest": "native_sha256"}[
                change
            ]
        ] = {"protocol": "unknown", "version": -1, "digest": "0" * 64}[change]
        payload[-1] = (
            MARKER,
            torch.tensor(list(json.dumps(identity).encode()), dtype=torch.uint8),
        )
    with pytest.raises(ValueError):
        verify_weights(payload, tmp_path / "receipt")


def test_original_miles_nccl_helper_transfers_one_complete_policy(
    tmp_path, monkeypatch
):
    from miles.backends.training_utils.weight_update.protocols import broadcast

    # The network/GPU collective is replaced; Miles' real metadata dispatch,
    # async client, per-tensor broadcast loop and result checks all execute.
    sent, ready, events = [], threading.Event(), []
    receipt = tmp_path / "receipt.json"

    class Receiver:
        def weight_update(self, method, args, kwargs):
            events.append(method)
            if method == "begin":
                self.expected = args[0]
            elif method == "update_weights_from_distributed":
                assert ready.wait(10)
                assert kwargs["names"] == ["weight", "audio_frozen", MARKER]
                verify_weights(list(zip(kwargs["names"], sent)), receipt)
            elif method == "finish":
                assert args[0] == self.expected == json.loads(receipt.read_text())
                return {**args[0], "version": args[0]["step"]}
            return {"success": True}

    def transfer(value, src, group, async_op):
        assert src == 0 and async_op and group is connection
        sent.append(value.clone())
        if len(sent) == 3:
            ready.set()
        return SimpleNamespace(wait=lambda: None)

    connection = object()
    monkeypatch.setattr(
        broadcast,
        "connect_rollout_engines_from_distributed",
        lambda *a, **kw: connection,
    )
    monkeypatch.setattr(
        broadcast,
        "disconnect_rollout_engines_from_distributed",
        lambda *a: events.append("disconnect"),
    )
    monkeypatch.setattr(broadcast.dist, "broadcast", transfer)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    state = {
        "weight": torch.arange(3.0),
        "audio_frozen": torch.tensor([2, 4], dtype=torch.bfloat16),
        CODEC + "weight": torch.tensor([999.0]),
        "vocab_weight": torch.ones(1),
    }
    publisher = MilesWeightPublisher([EngineClient(Receiver())], collocated=False)
    result = publisher.publish(SimpleNamespace(checkpoint_state_dict=lambda: state), 7)
    assert result["native_sha256"] == tensor_digest(list(state.items())[:2])
    assert result["hash_kind"] == HASH_KIND
    assert result["weight_sync_backend"] == "miles_nccl"
    assert result["acknowledgments"][0]["version"] == 7
    assert publisher.failed_storage is None
    publisher.close()
    assert events == [
        "begin",
        "update_weights_from_distributed",
        "finish",
        "disconnect",
    ]


def test_receiver_commit_requires_independent_digest_receipt(tmp_path):
    engine = object.__new__(SGLangRolloutEngine)
    engine.directory, engine.released = tmp_path, True
    engine.memory_saver = True
    resumed = []
    engine.engine = SimpleNamespace(
        resume_memory_occupation=lambda tags: resumed.append(tags),
        flush_cache=lambda: SimpleNamespace(success=True),
    )
    payload, identity = sealed([("frozen_audio", torch.arange(4.0))])
    server = RolloutServer(SimpleNamespace(), tmp_path, 0)
    server.engine, server.version = engine, 2
    server.weight_update("begin", (identity,), {})
    assert server.version is None and not engine.released and resumed == [["weights"]]
    with pytest.raises(ValueError, match="already in progress"):
        server.weight_update("begin", (identity,), {})
    with pytest.raises(FileNotFoundError):
        server.weight_update("finish", (identity,), {})
    verify_weights(payload, tmp_path / "sources.weights.json")
    with pytest.raises(ValueError, match="acknowledgment"):
        server.weight_update("finish", ({**identity, "step": 4},), {})
    assert server.version is None
    assert server.weight_update("finish", (identity,), {}) == identity
    assert server.version == 3
    assert resumed == [["weights"], ["kv_cache", "cuda_graph"]]


@pytest.mark.parametrize("released", [False, True])
def test_publication_keeps_full_receive_buffer_separate_from_kv(tmp_path, released):
    payload, identity = sealed([("weight", torch.ones(4))])
    kv = {"kv_cache", "cuda_graph"}
    offloaded = {"weights", *kv} if released else set()
    calls = []

    class TaggedEngine:
        def release_memory_occupation(self, tags):
            assert not (set(tags) & offloaded)
            offloaded.update(tags)
            calls.append(("release", tags))

        def resume_memory_occupation(self, tags):
            assert set(tags) <= offloaded
            offloaded.difference_update(tags)
            calls.append(("resume", tags))

        def update_weights_from_distributed(self, **kwargs):
            # The real SG receiver allocates the entire incoming policy here.
            assert offloaded == kv and "weights" not in offloaded
            verify_weights(payload, tmp_path / "sources.weights.json")
            calls.append(("receive",))
            return True, "loaded"

        def flush_cache(self):
            assert offloaded == kv
            calls.append(("trim",))
            return SimpleNamespace(success=True)

    engine = object.__new__(SGLangRolloutEngine)
    engine.directory, engine.released, engine.memory_saver = tmp_path, released, True
    engine.engine = TaggedEngine()
    server = RolloutServer(SimpleNamespace(), tmp_path, 0)
    server.engine, server.version = engine, 2
    server.weight_update("begin", (identity,), {})
    assert server.version is None and engine.publication_kv_released
    server.weight_update("update_weights_from_distributed", (), {})
    assert offloaded == kv and server.version is None
    server.weight_update("finish", (identity,), {})
    assert not offloaded and not engine.publication_kv_released
    assert server.version == 3
    first = (
        ("resume", ["weights"]) if released else ("release", ["kv_cache", "cuda_graph"])
    )
    assert calls == [
        first,
        ("receive",),
        ("trim",),
        ("resume", ["kv_cache", "cuda_graph"]),
    ]


@pytest.mark.parametrize("phase", ["begin", "trim", "finish"])
def test_memory_tag_failure_never_advertises_a_usable_policy(tmp_path, phase):
    payload, identity = sealed([("weight", torch.ones(4))])
    engine = object.__new__(SGLangRolloutEngine)
    engine.directory, engine.released, engine.memory_saver = tmp_path, False, True

    def fail(tags):
        raise RuntimeError("synthetic memory tag failure")

    engine.engine = SimpleNamespace(release_memory_occupation=fail)
    server = RolloutServer(SimpleNamespace(), tmp_path, 0)
    server.engine, server.version = engine, 2
    if phase != "begin":
        engine.engine.release_memory_occupation = lambda tags: None
        server.weight_update("begin", (identity,), {})
        verify_weights(payload, tmp_path / "sources.weights.json")
        engine.engine.resume_memory_occupation = fail
        engine.engine.flush_cache = lambda: SimpleNamespace(
            success=phase != "trim", message="synthetic memory tag failure"
        )
    with pytest.raises(RuntimeError, match="memory tag failure"):
        server.weight_update("begin" if phase == "begin" else "finish", (identity,), {})
    assert server.version is None
    with pytest.raises(RuntimeError, match="Load a policy"):
        server.generate("unused", "unused", 3)
    if phase != "begin":
        assert engine.expected_weights == identity and engine.publication_kv_released


@pytest.mark.parametrize("fail", [False, True])
def test_original_miles_ipc_serializer_and_storage_lifetime(
    tmp_path, monkeypatch, fail
):
    from miles.backends.training_utils.weight_update.protocols import (
        broadcast,
        cuda_ipc,
    )

    events, receipt = [], tmp_path / "receipt.json"

    class Receiver:
        def weight_update(self, method, args, kwargs):
            events.append(method)
            if method == "update_weights_from_tensor":
                assert kwargs["load_format"] == "flattened_bucket"
                assert len(kwargs["serialized_named_tensors"]) == 1
                payload = cuda_ipc.MultiprocessingSerializer.deserialize(
                    kwargs["serialized_named_tensors"][0]
                )
                bucket = cuda_ipc.FlattenedTensorBucket(**payload)
                tensors = bucket.reconstruct_tensors()
                assert [name for name, _ in tensors] == [
                    "actor",
                    "audio_frozen",
                    MARKER,
                ]
                verify_weights(tensors, receipt)
                if fail:
                    raise RuntimeError("synthetic receiver failure")
            elif method == "finish":
                assert json.loads(receipt.read_text()) == args[0]
                return args[0]
            elif method == "abort":
                assert publisher.failed_storage is not None
            return {"success": True}

    def gather(value, object_gather_list, dst, group):
        assert dst == 0 and group is publisher.ipc_group
        object_gather_list[0] = value

    monkeypatch.setattr(cuda_ipc.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(cuda_ipc.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(cuda_ipc.dist, "gather_object", gather)
    monkeypatch.setattr(
        cuda_ipc.dist, "destroy_process_group", lambda *a: events.append("destroy")
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: events.append("collect"))
    publisher = MilesWeightPublisher([EngineClient(Receiver())], collocated=True)
    publisher.ipc_group = object()  # Only the original metadata gather is faked.
    state = {
        "actor": torch.arange(4.0),
        "audio_frozen": torch.ones(4, dtype=torch.bfloat16),
    }
    policy = SimpleNamespace(checkpoint_state_dict=lambda: state)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic receiver failure"):
            publisher.publish(policy, 3)
        assert publisher.failed_storage is not None and "collect" not in events
        publisher.close()
        assert publisher.failed_storage is None
        assert events == [
            "begin",
            "update_weights_from_tensor",
            "abort",
            "collect",
            "destroy",
        ]
    else:
        result = publisher.publish(policy, 3)
        assert result["weight_sync_backend"] == "miles_ipc"
        assert publisher.failed_storage is None
        publisher.close()
        assert events == [
            "begin",
            "update_weights_from_tensor",
            "finish",
            "collect",
            "destroy",
        ]


def test_shared_memory_policy_reaches_an_independent_interpreter(tmp_path):
    """Real Miles helper, real Gloo gather and real named shared-memory receive."""
    previous = torch.multiprocessing.get_sharing_strategy()
    receipt = tmp_path / "sources.weights.json"
    receiver = """
import json, sys, torch
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from espnet2.rl.weight_transport import verify_weights
payload = MultiprocessingSerializer.deserialize(sys.stdin.read())
tensors = verify_weights(FlattenedTensorBucket(**payload).reconstruct_tensors(), sys.argv[1])
assert [name for name, _ in tensors] == ['actor', 'audio_frozen']
assert torch.equal(tensors[0][1], torch.arange(4.))
assert torch.equal(tensors[1][1], torch.ones(4, dtype=torch.bfloat16))
print(json.dumps({'verified': True}))
"""

    class Receiver:
        def weight_update(self, method, args, kwargs):
            if method == "update_weights_from_tensor":
                assert len(kwargs["serialized_named_tensors"]) == 1
                completed = subprocess.run(
                    [sys.executable, "-c", receiver, str(receipt)],
                    input=kwargs["serialized_named_tensors"][0],
                    text=True,
                    capture_output=True,
                    timeout=90,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                )
                assert completed.returncode == 0, completed.stderr
                assert json.loads(completed.stdout.splitlines()[-1])["verified"]
            elif method == "finish":
                assert json.loads(receipt.read_text()) == args[0]
                return args[0]
            return {"success": True}

    policy = SimpleNamespace(
        checkpoint_state_dict=lambda: {
            "actor": torch.arange(4.0),
            "audio_frozen": torch.ones(4, dtype=torch.bfloat16),
        }
    )
    publisher = MilesWeightPublisher(
        [EngineClient(Receiver())], collocated=True, shared_memory=True
    )
    try:
        result = publisher.publish(policy, 5)
        assert result["weight_sync_backend"] == "miles_shm"
        assert result["hash_kind"] == HASH_KIND and result["step"] == 5
        assert torch.multiprocessing.get_sharing_strategy() == previous
    finally:
        publisher.close()
    assert torch.multiprocessing.get_sharing_strategy() == previous
    assert not torch.distributed.is_initialized()


def test_fast_sync_reuses_bootstrap_and_does_not_export_per_update(
    tmp_path, monkeypatch
):
    import espnet2.rl.weight_transport as transport

    config = ESPnetRLConfig(
        family="bagpiper", weight_sync_backend="miles", output_dir=str(tmp_path)
    )
    rollout = CheckpointRollout(config)
    calls = []
    rollout.processes = [object()]
    rollout.export = lambda *a: pytest.fail("Fast publication must not export")

    class Publisher:
        def __init__(self, clients, *, collocated, sglang_source, shared_memory):
            assert len(clients) == 1 and collocated

        def publish(self, policy, step):
            calls.append(step)
            return {"step": step, "hash_kind": HASH_KIND}

    monkeypatch.setattr(transport, "MilesWeightPublisher", Publisher)
    rollout.sync_weights(object(), 1)
    publisher = rollout.publisher
    rollout.sync_weights(object(), 2)
    assert rollout.publisher is publisher and calls == [1, 2]
    assert (
        json.loads((rollout.directory / "weight-sync-000002.json").read_text())["step"]
        == 2
    )


def test_disk_publication_phase_clock_includes_save_export_hash_and_ack(
    tmp_path, monkeypatch
):
    import espnet2.rl.rollout as module

    config = ESPnetRLConfig(output_dir=str(tmp_path), profile_phases=True)
    rollout = CheckpointRollout(config)
    clock = iter([0.0, 1.0, 3.0, 6.0, 7.0, 11.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    def export(args, **kwargs):
        directory = Path(args[args.index("--output") + 1])
        directory.mkdir()
        (directory / "weights.bin").write_bytes(b"immutable export")

    monkeypatch.setattr(module.subprocess, "run", export)
    rollout.load = lambda export, step: None
    policy = SimpleNamespace(checkpoint_state_dict=lambda: {"weight": torch.ones(2)})
    result = rollout.sync_weights(policy, 0)
    assert result["hash_kind"] == "native-file/v1"
    assert result["perf/native_snapshot_time"] == 1
    assert result["perf/hf_export_time"] == 2
    assert result["perf/weight_hash_time"] == 3
    assert result["perf/weight_load_ack_time"] == 4
    assert json.loads((tmp_path / "weight_publication.jsonl").read_text()) == result


def test_ray_publication_uses_memory_then_current_disk_export_for_periodic_eval(
    tmp_path,
):
    from espnet2.rl.miles_components import NativeComponents

    config = ESPnetRLConfig(
        family="bagpiper",
        weight_sync_backend="miles",
        execution_mode="disagg_async",
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        train_gpus="0",
        rollout_gpus="1",
        output_dir=str(tmp_path),
        periodic_eval_interval=2,
        periodic_eval_function="unused:evaluate",
    )
    components = NativeComponents(config, [], "host", "host")
    components.train_gpu_ids = {"0"}
    calls = []
    acknowledgment = dict(replica=0, node="host", gpu_ids=[1], gpu="1", version=1)
    bootstrap = tmp_path / "rollout" / "policy-000000"
    bootstrap.mkdir(parents=True)
    components.rollout.policy = {"step": 0}

    async def load(path, version):
        calls.append(("load", path, version))
        return {**acknowledgment, "version": version}

    async def evaluate(policy):
        calls.append(("evaluate", policy["step"]))

    actor = SimpleNamespace(
        load=SimpleNamespace(remote=load), evaluate=SimpleNamespace(remote=evaluate)
    )
    components.rollout.actors = [actor]

    class Trainer:
        async def call(self, method, *args):
            calls.append((method,))
            if method == "publish_async_weights":
                assert args == ([actor],)
                return [
                    dict(
                        step=1,
                        path=str(bootstrap),
                        native_sha256="a" * 64,
                        hash_kind=HASH_KIND,
                        acknowledgments=[acknowledgment],
                    )
                ]
            assert method == "export_async_snapshot"
            return [
                dict(
                    step=2,
                    path=str(bootstrap.parent / "policy-000002"),
                    native_sha256="b" * 64,
                )
            ]

    async def run():
        await components.update_weights(None, Trainer(), None, None, rollout_id=0)
        assert components.rollout.policy["hash_kind"] == HASH_KIND
        await components.update_weights(None, Trainer(), None, None, rollout_id=1)

    asyncio.run(run())
    assert calls == [
        ("publish_async_weights",),
        ("export_async_snapshot",),
        ("load", str(bootstrap.parent / "policy-000002" / "hf"), 2),
        ("evaluate", 2),
    ]
    acknowledgments = [
        json.loads(line)
        for line in (tmp_path / "weight_ack.jsonl").read_text().splitlines()
    ]
    assert [item["hash_kind"] for item in acknowledgments] == [
        HASH_KIND,
        "native-file/v1",
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"family": "owsm"},
        {"rollout_backend": "vllm"},
        {"train_gpus": "0,1"},
        {"rollout_gpus": "0,1"},
    ],
)
def test_fast_mode_rejects_unsupported_topologies(options):
    config = ESPnetRLConfig(family="bagpiper", weight_sync_backend="miles")
    for name, value in options.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match="Miles weight transfer requires"):
        config.validate()


@pytest.mark.parametrize(
    "options", [{"execution_mode": "disagg_async"}, {"rollout_gpus": "1"}]
)
def test_shared_memory_rejects_noncollocated_recipes(options):
    config = ESPnetRLConfig(
        family="bagpiper", weight_sync_backend="miles_shm", **options
    )
    with pytest.raises(ValueError, match="collocated synchronous"):
        config.validate()
