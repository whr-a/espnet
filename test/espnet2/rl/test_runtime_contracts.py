"""CPU contract checks; these do not establish H100 training or placement."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from espnet2.bin.rl_train_async import resolve_nodes
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import verify_data_admission
from espnet2.rl.engine import engine_kwargs
from espnet2.rl.process_identity import host_identity
from espnet2.rl.rollout import sha256
from espnet2.rl.rollout_replicas import (
    _process_start_ticks,
    run_owned_subprocess,
    run_replica,
    terminate_owned_process,
)


def configuration(**kwargs):
    defaults = dict(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_generate_factory="unused:generate",
        prompts_per_step=32,
        group_size=8,
        steps=5,
        warmup_steps=1,
        train_gpus="0,1,2,3",
        rollout_gpus="4,5,6,7",
    )
    return ESPnetRLConfig(**{**defaults, **kwargs})


def admission_files(tmp_path):
    rows = []
    for index in range(40):
        audio = tmp_path / f"{index}.wav"
        audio.write_bytes(f"unique audio {index}".encode())
        rows.append(
            {"id": str(index), "audio": str(audio), "audio_sha256": sha256(audio)}
        )
    train, heldout = tmp_path / "train32.jsonl", tmp_path / "heldout.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows[:32]))
    heldout.write_text("".join(json.dumps(row) + "\n" for row in rows[32:]))
    receipt = {
        "training_ready": True,
        "manifest_sha256": {"train": sha256(train), "heldout": sha256(heldout)},
        "counts": {"train": 32, "heldout": 8},
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(receipt))
    return configuration(manifest=str(train), eval_manifest=str(heldout)), path


@pytest.mark.parametrize(
    "changed", ["audio", "train_manifest", "heldout_manifest", "counts", "not_ready"]
)
def test_admission_detects_live_file_or_receipt_changes(tmp_path, changed):
    config, path = admission_files(tmp_path)
    assert verify_data_admission(config, path)["counts"]["train"] == 32
    if changed == "audio":
        (tmp_path / "0.wav").write_bytes(b"different audio")
    elif changed == "train_manifest":
        Path(config.manifest).write_text("changed manifest")
    elif changed == "heldout_manifest":
        Path(config.eval_manifest).write_text("changed manifest")
    else:
        receipt = json.loads(path.read_text())
        if changed == "counts":
            receipt["counts"]["train"] = 31
        else:
            receipt["training_ready"] = False
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_data_admission(config, path)


@pytest.mark.parametrize(
    "train, rollout, ranks, replicas, collocated",
    [
        ("0", "0", 1, 1, True),
        ("0", "1", 1, 1, False),
        ("0,1,2,3", "4,5,6,7", 4, 4, False),
        ("0,1,2,3,4,5,6,7", "0,1,2,3,4,5,6,7", 8, 8, True),
    ],
)
def test_geometry_comes_from_the_declared_devices(
    train, rollout, ranks, replicas, collocated
):
    config = configuration(train_gpus=train, rollout_gpus=rollout)
    config.validate()
    assert config.train_ranks == ranks
    assert config.rollout_replicas == replicas
    assert config.collocated is collocated


@pytest.mark.parametrize("name", ["train_gpus", "rollout_gpus"])
def test_repeated_or_empty_device_lists_are_rejected(name):
    for value in ("0,0", ""):
        config = configuration(**{name: value})
        with pytest.raises(ValueError, match="distinct physical GPU ids"):
            config.validate()


def test_engine_kwargs_sizes_the_engine_and_keeps_frozen_settings():
    """rollout_mem_fraction is the one way a recipe sizes its engine."""
    sized = configuration(rollout_mem_fraction=0.2)
    sized.validate()
    settings = engine_kwargs(sized, 3, defaults={"mem_fraction_static": 0.65})
    assert settings["mem_fraction_static"] == 0.2
    assert settings["random_seed"] == sized.seed + 3
    assert settings["tp_size"] == 1
    assert settings["disable_cuda_graph"] is True
    assert settings["disable_radix_cache"] is True
    # Left unset, the family's own default survives untouched.
    assert (
        engine_kwargs(configuration(), 0, defaults={"mem_fraction_static": 0.65})[
            "mem_fraction_static"
        ]
        == 0.65
    )
    # Model identity, dtype, one device per replica and a step-advancing seed
    # are frozen for every family, not per family.
    repeated = configuration(sglang_engine_kwargs={"dtype": "float32", "tp_size": 1})
    assert (
        engine_kwargs(repeated, 0, defaults={"dtype": "float32"})["dtype"] == "float32"
    )
    changed = configuration(sglang_engine_kwargs={"dtype": "bfloat16"})
    with pytest.raises(ValueError, match="native execution contract"):
        engine_kwargs(changed, 0, defaults={"dtype": "float32"})
    # A constant seed would make every group's candidates identical, which
    # leaves GRPO without a signal.
    pinned = configuration(sglang_engine_kwargs={"random_seed": 7})
    with pytest.raises(ValueError, match="native execution contract"):
        engine_kwargs(pinned, 3)
    # A setting a family does not freeze stays a recipe decision.
    loud = configuration(sglang_engine_kwargs={"log_level": "debug"})
    assert engine_kwargs(loud, 0)["log_level"] == "debug"
    # The family's own frozen settings compose with the shared ones.
    unsafe = configuration(sglang_engine_kwargs={"max_running_requests": 4})
    with pytest.raises(ValueError, match="native execution contract"):
        engine_kwargs(
            unsafe,
            0,
            defaults={"max_running_requests": 1},
            frozen=("max_running_requests",),
        )
    # A memory fraction is still a fraction, whichever way it arrives.
    for kwargs in ({"mem_fraction_static": 1.4}, {"mem_fraction_static": 0}):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            engine_kwargs(configuration(sglang_engine_kwargs=kwargs), 0)
    # The two ways to size the engine may not disagree.
    conflict = configuration(
        rollout_mem_fraction=0.2, sglang_engine_kwargs={"mem_fraction_static": 0.5}
    )
    with pytest.raises(ValueError, match="disagree"):
        engine_kwargs(conflict, 0)
    agreed = configuration(
        rollout_mem_fraction=0.2, sglang_engine_kwargs={"mem_fraction_static": 0.2}
    )
    assert engine_kwargs(agreed, 0)["mem_fraction_static"] == 0.2


def test_node_resolution_allows_one_node_that_holds_both_roles():
    nodes = [
        {
            "NodeID": str(i),
            "NodeManagerAddress": f"10.0.0.{i}",
            "Alive": True,
            "Resources": {"GPU": 8},
        }
        for i in range(2)
    ]
    assert resolve_nodes(
        nodes, "0", "10.0.0.1", train_ranks=8, rollout_replicas=8
    ) == tuple(nodes)
    # One node may host both roles when it exposes enough GPUs for each.
    first, second = resolve_nodes(
        nodes, "0", "10.0.0.0", train_ranks=1, rollout_replicas=1
    )
    assert first["NodeID"] == second["NodeID"] == "0"
    with pytest.raises(ValueError, match="both roles"):
        resolve_nodes(nodes, "0", "10.0.0.0", train_ranks=4, rollout_replicas=8)
    with pytest.raises(ValueError, match="fewer than 8"):
        resolve_nodes(
            [nodes[0], {**nodes[1], "Resources": {"GPU": 6}}],
            "0",
            "1",
            train_ranks=8,
            rollout_replicas=8,
        )


def test_replica_keeps_rollout_id_distinct_from_behavior_version(tmp_path, monkeypatch):
    policy = tmp_path / "policy-000000"
    (policy / "hf").mkdir(parents=True)
    (policy / "native.pt").write_bytes(b"native actor zero")
    (policy / "hf/config.json").write_text("{}")
    observed = {}

    def child(command, *, environment, directory, stream):
        observed.update(command=command, gpu=environment["CUDA_VISIBLE_DEVICES"])
        output = command[command.index("--output") + 1]
        Path(output).write_text(
            json.dumps(
                {
                    "policy_step": 0,
                    "rollout_id": 1,
                    "native_sha256": sha256(policy / "native.pt"),
                    "generations": [],
                }
            )
        )

    monkeypatch.setattr("espnet2.rl.rollout_replicas.run_owned_subprocess", child)
    config = configuration(output_dir=str(tmp_path / "run"))
    result = run_replica(
        config,
        policy,
        [{"id": "p"}],
        1,
        0,
        "7",
        sha256(policy / "native.pt"),
        policy_version=0,
    )
    assert observed["command"][observed["command"].index("--step") + 1] == "1"
    assert observed["command"][observed["command"].index("--policy-version") + 1] == "0"
    assert observed["gpu"] == "7"
    assert result["behavior_version"] == 0 and result["rollout_id"] == 1


def test_owned_child_session_cleanup_leaves_unrelated_process_alive(tmp_path):
    errors = []
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )

    def run():
        try:
            with (tmp_path / "child.log").open("w") as stream:
                run_owned_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    environment=dict(os.environ),
                    directory=tmp_path,
                    stream=stream,
                )
        except subprocess.CalledProcessError as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "process.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        receipt = json.loads((tmp_path / "process.json").read_text())
        assert receipt["pid"] == receipt["pgid"]
        assert receipt["host_identity"] == host_identity()
        assert receipt["start_ticks"] == _process_start_ticks(receipt["pid"])
        assert terminate_owned_process(tmp_path)
        thread.join(timeout=10)
        assert not thread.is_alive() and len(errors) == 1
        assert unrelated.poll() is None
        assert not terminate_owned_process(tmp_path)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)
        if thread.is_alive():
            terminate_owned_process(tmp_path)
            thread.join(timeout=10)


def test_stale_process_receipt_does_not_signal_reused_pid(tmp_path):
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        (tmp_path / "process.json").write_text(
            json.dumps(
                {
                    "pid": other.pid,
                    "pgid": other.pid,
                    "start_ticks": "stale",
                    "host_identity": host_identity(),
                }
            )
        )
        assert not terminate_owned_process(tmp_path)
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_shared_filesystem_receipt_does_not_signal_other_host_pid(tmp_path):
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        receipt = {
            "pid": other.pid,
            "pgid": other.pid,
            "start_ticks": _process_start_ticks(other.pid),
            "host_identity": {**host_identity(), "hostname": "different-host"},
        }
        (tmp_path / "process.json").write_text(json.dumps(receipt))
        assert not terminate_owned_process(tmp_path)
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_ray_object_store_roundtrip_uses_actual_reference_and_cpu_batch():
    import ray
    import torch
    from miles.utils import object_store
    from miles.utils.data import RolloutDataPack, remove_rollout_data_refs

    previous = object_store._INSTANCE
    ray.init(
        address="local",
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=80 * 1024 * 1024,
    )
    try:
        object_store._INSTANCE = None
        store = object_store.init_instance(
            SimpleNamespace(object_store_backend="ray", worker_comm_backend="ray")
        )
        ref = store.put({"actions": torch.tensor([1, 2]), "behavior_version": 0})
        pack = RolloutDataPack(data_ref=ref, sample_indices=[0, 1])
        with store.get(pack.data_ref) as payload:
            assert payload["behavior_version"] == 0
            torch.testing.assert_close(payload["actions"], torch.tensor([1, 2]))
        remove_rollout_data_refs(SimpleNamespace(), pack)
        # Ray wire mode uses normal reference lifetime, not force-free.
        assert store._frees_objects is False
    finally:
        object_store._INSTANCE = previous
        ray.shutdown()
