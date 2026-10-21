# Copyright 2025 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Persistent inference processes and immutable checkpoint synchronization."""

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, **options):
    """Write JSON with every distinct numpy array in a binary sidecar.

    The array becomes ``{"__ndarray__": "<stem>.<n>.npy", "sha256": ...}``,
    written once however often it occurs (a prompt's features are shared by
    all its candidates).
    """
    path, stored = Path(path), {}

    def sidecar(item):
        if not isinstance(item, np.ndarray):
            raise TypeError(
                f"Object of type {type(item).__name__} is not JSON serializable"
            )
        if id(item) not in stored:
            name = f"{path.stem}.{len(stored)}.npy"
            np.save(path.with_name(name), item, allow_pickle=False)
            stored[id(item)] = {
                "__ndarray__": name,
                "sha256": sha256(path.with_name(name)),
            }
        return stored[id(item)]

    path.write_text(json.dumps(value, default=sidecar, **options) + "\n")


def read_json(path):
    """Read write_json output; the references to one sidecar share its array."""
    path, loaded = Path(path), {}

    def array(item):
        name = item.get("__ndarray__")
        if name is None or set(item) != {"__ndarray__", "sha256"}:
            return item
        if name not in loaded:
            sidecar = path.with_name(name)
            if Path(name).name != name or sha256(sidecar) != item["sha256"]:
                raise ValueError(f"Array sidecar {sidecar} differs from its record")
            loaded[name] = np.load(sidecar, allow_pickle=False)
        return loaded[name]

    return json.loads(path.read_text(), object_hook=array)


def subprocess_environment(config, gpus):
    """An inference interpreter's environment: its devices and source paths."""
    environment = dict(os.environ)
    # Memory saver owns the rollout allocator; learner tuning stays local.
    if config.rollout_backend == "sglang" and config.sglang_engine_kwargs.get(
        "enable_memory_saver"
    ):
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            environment.pop(name, None)
    paths = [str(Path(__file__).resolve().parents[2])]
    if config.rollout_backend == "sglang" and config.sglang_source:
        paths.insert(0, str(Path(config.sglang_source).resolve() / "python"))
    elif config.vllm_source:
        paths.insert(0, str(Path(config.vllm_source).resolve()))
    if config.miles_source:
        paths.append(str(Path(config.miles_source).resolve()))
    environment["PYTHONPATH"] = os.pathsep.join(
        paths + [environment.get("PYTHONPATH", "")]
    )
    environment["CUDA_VISIBLE_DEVICES"] = gpus
    if config.rollout_backend == "vllm":
        environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if config.dtype == "float32":
        environment["TRITON_F32_DEFAULT"] = "ieee"
        environment["NVIDIA_TF32_OVERRIDE"] = "0"
        environment["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    # A subprocess is an independent GPU consumer, never a torchrun rank.
    training_identity = {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    }
    for name in list(environment):
        if name in training_identity or name.startswith("TORCHELASTIC_"):
            # torchrun injects these after the outer launcher is cleaned.
            # USE_AGENT_STORE=True makes even an explicit TP1 tcp:// rank0
            # a TCPStore client, leaving its own rendezvous without a server.
            environment.pop(name)
    return environment


class RolloutProcess:
    """One rollout worker interpreter that holds its engine for a whole run.

    The worker runs in its own session on ``gpu`` and logs to
    ``directory/rollout.log``. Requests and generations of every rollout stay
    in ``directory`` for audit.
    """

    def __init__(self, config, gpu, directory, replica):
        self.config, self.replica = config, replica
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.version = None
        frozen = self.directory / "config.json"
        frozen.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        self.log = self.directory / "rollout.log"
        # A socket pair rather than a socket path: run directories exceed the
        # 107-byte AF_UNIX limit.
        ours, theirs = socket.socketpair()
        try:
            with self.log.open("w") as stream:
                self.process = subprocess.Popen(
                    [
                        config.rollout_python,
                        "-m",
                        "espnet2.rl.rollout_worker",
                        "--config",
                        str(frozen.resolve()),
                        "--fd",
                        str(theirs.fileno()),
                        "--replica",
                        str(replica),
                        "--directory",
                        str(self.directory.resolve()),
                    ],
                    env=subprocess_environment(config, str(gpu)),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    pass_fds=(theirs.fileno(),),
                    start_new_session=True,
                )
        except BaseException:
            ours.close()
            raise
        finally:
            theirs.close()
        self.connection = Connection(ours.detach())

    def _call(self, **request):
        try:
            self.connection.send(request)
            reply = self.connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"Rollout replica {self.replica} exited "
                f"(status {self.process.poll()}) during {request['op']}; "
                f"last lines of {self.log}:\n{self._tail()}"
            ) from error
        if "error" in reply:
            raise RuntimeError(
                f"Rollout replica {self.replica} failed {request['op']}:\n"
                + reply["error"]
            )
        return reply

    def _tail(self, lines=40):
        try:
            text = self.log.read_text(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def load(self, model, version):
        """Build the engine from the first export; load later ones in place."""
        self._call(op="load", model=str(Path(model).resolve()), version=version)
        self.version = version

    def weight_update(self, method, args, kwargs):
        if method == "abort":
            self.close()
            return {}
        result = self._call(op="weight_update", method=method, args=args, kwargs=kwargs)
        if method == "begin":
            self.version = None
        elif method == "finish":
            self.version = result["step"]
        return result

    def generate(self, sources, rollout_id):
        requests = self.directory / f"requests-{rollout_id:06d}.json"
        responses = self.directory / f"responses-{rollout_id:06d}.json"
        write_json(requests, sources)
        self._call(
            op="generate",
            requests=str(requests),
            output=str(responses),
            rollout_id=rollout_id,
        )
        result = read_json(responses)
        if result["rollout_id"] != rollout_id or any(
            generation["policy_version"] != self.version
            for generation in result["generations"]
        ):
            raise RuntimeError(
                f"Rollout replica {self.replica} answered for another rollout "
                "or policy version"
            )
        return result["generations"]

    def close(self, timeout=120):
        """Close politely, then stop whatever is left of the worker's session."""
        if self.connection.closed:
            return
        if self.process.poll() is None:
            try:
                self.connection.send({"op": "close"})
                if self.connection.poll(timeout):
                    self.connection.recv()
                self.process.wait(timeout=timeout)
            except (EOFError, OSError, subprocess.TimeoutExpired):
                pass
        try:
            # SGLang's scheduler children share the worker's session.
            os.killpg(self.process.pid, signal.SIGKILL)
        except OSError:
            pass  # Nothing is left of it.
        self.process.wait()
        self.connection.close()


def prune_exports(directory, keep):
    """Delete all but the newest ``keep`` policy exports in ``directory``.

    Call it once every engine holds the newest export: an engine reads only
    the export it holds, and the async behavior audit reads none.
    """
    exports = sorted(
        Path(directory).glob("policy-*"),
        key=lambda path: int(path.name.removeprefix("policy-")),
    )
    for export in exports[:-keep]:
        shutil.rmtree(export)


def record_publication(config, metadata):
    """Opt-in publication phases, outside the learner's update clock."""
    if config.profile_phases:
        with Path(config.output_dir, "weight_publication.jsonl").open("a") as stream:
            stream.write(json.dumps(metadata, allow_nan=False) + "\n")


def merge_generations(sources, shards, group_size):
    """Restore the source order of complete prompt groups from every shard."""
    grouped = {}
    expected = {source["id"] for source in sources}
    for generations in shards:
        for generation in generations:
            identity = generation["source_id"]
            if identity not in expected:
                raise ValueError("Rollout replica returned an unknown source")
            grouped.setdefault(identity, []).append(generation)
    if set(grouped) != expected or any(
        len(group) != group_size for group in grouped.values()
    ):
        raise ValueError("Rollout replicas must return every complete prompt group")
    return [generation for source in sources for generation in grouped[source["id"]]]


class CheckpointRollout:
    """Rank zero's rollout: one persistent engine process per rollout GPU.

    Disk publication exports each version; the optional Miles path reuses one
    bootstrap export and transfers complete policies in memory. Keeping engines
    in their own interpreters preserves their torch/transformers environments.
    """

    def __init__(self, config):
        self.config = config
        self.directory = Path(config.output_dir) / "rollout"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.version = None
        self.processes = []
        self.publisher = None

    def export(self, policy, step):
        start = time.monotonic()
        directory = self.directory / f"policy-{step:06d}"
        staging = self.directory / f".policy-{step:06d}.tmp"
        if directory.exists() or staging.exists():
            raise FileExistsError(
                f"Refusing to overwrite a frozen policy export: {directory}"
            )
        staging.mkdir()
        native = staging / "native.pt"
        torch.save(
            {
                key: value.detach().cpu().contiguous()
                for key, value in policy.checkpoint_state_dict().items()
            },
            native,
        )
        snapshot_end = time.monotonic()
        args = [
            self.config.rollout_python,
            "-m",
            "espnet2.rl.export_worker",
            "--config",
            str(Path(self.config.output_dir) / "frozen_config.json"),
            "--checkpoint",
            str(native.resolve()),
            "--output",
            str((staging / "hf").resolve()),
        ]
        with (staging / "export.log").open("w") as stream:
            subprocess.run(
                args,
                env=self._environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        export_end = time.monotonic()
        metadata = {
            "step": step,
            "native_sha256": sha256(native),
            "hash_kind": "native-file/v1",
            "hf_files": {},
        }
        for path in sorted((staging / "hf").rglob("*")):
            if path.is_file():
                metadata["hf_files"][str(path.relative_to(staging / "hf"))] = sha256(
                    path
                )
        if not metadata["hf_files"]:
            raise RuntimeError("The ESPnet exporter produced no model files")
        if self.config.profile_phases:
            metadata.update(
                {
                    "perf/native_snapshot_time": snapshot_end - start,
                    "perf/hf_export_time": export_end - snapshot_end,
                    "perf/weight_hash_time": time.monotonic() - export_end,
                }
            )
        (staging / "sync.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(directory)
        self.version = directory
        return metadata

    def sync_weights(self, policy, step):
        if self.config.weight_sync_backend != "disk" and self.processes:
            from .weight_transport import EngineClient, MilesWeightPublisher

            if self.publisher is None:
                self.publisher = MilesWeightPublisher(
                    [EngineClient(process) for process in self.processes],
                    collocated=self.config.collocated,
                    sglang_source=self.config.sglang_source,
                    shared_memory=self.config.weight_sync_backend == "miles_shm",
                )
            metadata = self.publisher.publish(policy, step)
            path = self.directory / f"weight-sync-{step:06d}.json"
            path.write_text(json.dumps(metadata, indent=2) + "\n")
            record_publication(self.config, metadata)
            return metadata
        metadata = self.export(policy, step)
        start = time.monotonic()
        self.load(self.version, step)
        if self.config.profile_phases:
            metadata["perf/weight_load_ack_time"] = time.monotonic() - start
        record_publication(self.config, metadata)
        prune_exports(self.directory, self.config.keep_exports)
        return metadata

    def load(self, export, step):
        """Load one export into every process, starting them the first time."""
        if not self.processes:
            # vLLM is one tensor-parallel engine over every rollout GPU.
            gpus = (
                [self.config.rollout_gpus]
                if self.config.rollout_backend == "vllm"
                else self.config.rollout_device_ids
            )
            self.processes = [
                RolloutProcess(
                    self.config, gpu, self.directory / f"replica-{replica}", replica
                )
                for replica, gpu in enumerate(gpus)
            ]
        self._each(lambda process: process.load(Path(export) / "hf", step))

    def generate(self, sources, step):
        if not self.processes:
            raise RuntimeError("Weights must be synchronized before rollout")
        count = len(self.processes)
        if len(sources) < count:
            raise ValueError("Each rollout replica requires its own prompt shard")
        shards = self._each(
            lambda process: process.generate(sources[process.replica :: count], step)
        )
        return merge_generations(sources, shards, self.config.group_size)

    def _each(self, call):
        with ThreadPoolExecutor(max_workers=len(self.processes)) as executor:
            return list(executor.map(call, self.processes))

    def close(self):
        try:
            if self.publisher is not None:
                self.publisher.close()
        finally:
            if self.processes:
                self._each(RolloutProcess.close)
            self.processes = []
            self.publisher = None

    def _environment(self):
        return subprocess_environment(self.config, self.config.rollout_gpus)


def save_checkpoint(
    path, policy, optimizer, step, config, *, scheduler=None, trainer_state=None
):
    """Atomic native snapshot, optionally with collective trainer resume state.

    ``NativeGRPOTrainer.save_training_checkpoint`` supplies all-rank RNG and
    the data cursor. Legacy callers retain their CPU-only RNG snapshot.
    """
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    snapshot = {
        "model": policy.checkpoint_state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if scheduler is not None:
        snapshot["scheduler"] = scheduler.state_dict()
    if trainer_state is not None:
        snapshot["trainer_state"] = trainer_state
    torch.save(snapshot, temporary)
    temporary.replace(path)
    # Native .pth permits direct ESPnet inference and exporter reuse.
    native = path.with_name(path.stem + ".native.pth")
    torch.save(snapshot["model"], native)
    if config.train_config:
        shutil.copyfile(config.train_config, path.parent / "espnet_train_config.yaml")
