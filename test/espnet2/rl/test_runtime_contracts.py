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
from espnet2.rl.rollout import sha256
from espnet2.rl.process_identity import host_identity
from espnet2.rl.rollout_replicas import (
    _process_start_ticks,
    run_owned_subprocess,
    run_replica,
    terminate_owned_process,
)
from espnet2.rl.short_run import (
    validate_collocate_geometry,
    verify_data_admission,
)


def configuration(**kwargs):
    return ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_generate_factory="unused:generate",
        train_batch_size=32,
        n_samples_per_prompt=8,
        steps=5,
        warmup_steps=1,
        rollout_num_replicas=4,
        rollout_gpus="4,5,6,7",
        **kwargs,
    )


def admission_files(tmp_path):
    protected = tmp_path / "registered.json"
    protected.write_text("registered protected metadata")
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
        "schema": "espnet-short-run-data-v1",
        "training_ready": True,
        "scoped_component_admission": "VERIFIED_REGISTERED_PROTECTION",
        "exact_key_gate": "PASSED",
        "verified_protection_scope": {
            "specialist_records": 1389,
            "registered_audio_count": 1201,
            "registered_audio_decoded": 1201,
        },
        "protection_bindings": {str(protected): sha256(protected)},
        "manifest_sha256": {"train32": sha256(train), "heldout": sha256(heldout)},
        "counts": {"train": 32, "heldout": 8},
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(receipt))
    return (
        configuration(manifest=str(train), eval_manifest=str(heldout)),
        path,
        protected,
    )


@pytest.mark.parametrize(
    "changed",
    [
        "audio",
        "train_manifest",
        "heldout_manifest",
        "protected",
        "bindings",
        "specialist_scope",
    ],
)
def test_admission_detects_live_file_or_protection_changes(tmp_path, changed):
    config, path, protected = admission_files(tmp_path)
    assert verify_data_admission(config, path)["counts"]["train"] == 32
    if changed == "audio":
        (tmp_path / "0.wav").write_bytes(b"different audio")
    elif changed == "train_manifest":
        Path(config.manifest).write_text("changed manifest")
    elif changed == "heldout_manifest":
        Path(config.eval_manifest).write_text("changed manifest")
    elif changed == "protected":
        protected.write_text("changed protection")
    else:
        receipt = json.loads(path.read_text())
        if changed == "bindings":
            receipt["protection_bindings"] = {}
        else:
            receipt["verified_protection_scope"]["specialist_records"] = 1374
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_data_admission(config, path)


def test_collocate_geometry_checks_actual_torchrun_environment(monkeypatch):
    config = configuration()
    validate_collocate_geometry(config, running=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    validate_collocate_geometry(config, running=True)
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="four torchrun"):
        validate_collocate_geometry(config, running=True)
    config.rollout_gpus = "0,1,2,3"
    with pytest.raises(ValueError, match="physical GPUs"):
        validate_collocate_geometry(config, running=False)


def test_exact_two_node_resolution_rejects_same_or_fragmented_node():
    nodes = [
        {
            "NodeID": str(i),
            "NodeManagerAddress": f"10.0.0.{i}",
            "Alive": True,
            "Resources": {"GPU": 8},
        }
        for i in range(2)
    ]
    assert resolve_nodes(nodes, "0", "10.0.0.1") == tuple(nodes)
    with pytest.raises(ValueError, match="distinct"):
        resolve_nodes(nodes, "0", "10.0.0.0")
    nodes[1]["Resources"]["GPU"] = 6
    with pytest.raises(ValueError, match="8 GPUs"):
        resolve_nodes(nodes, "0", "1")


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
