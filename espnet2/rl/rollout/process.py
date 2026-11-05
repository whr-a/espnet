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
"""One rollout worker process: its client, its environment and its messages.

Requests and generations cross the process boundary as JSON files whose numpy
arrays live in binary sidecars (``write_json``/``read_json``).
"""

import json
import os
import signal
import socket
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

from ..utils import sha256


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
    paths = [str(Path(__file__).resolve().parents[3])]
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
                        "espnet2.rl.rollout.worker",
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
