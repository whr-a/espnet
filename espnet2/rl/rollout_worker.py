"""Serve the architecture's actual inference engine for a whole run.

The client is ``espnet2.rl.rollout.RolloutProcess``; it talks over one end of
a socket pair whose other end is this process's ``--fd``. ``load`` builds the
engine from the first export and loads every later one in place,
``generate`` answers one batch through JSON files in ``--directory`` (numpy
arrays in binary sidecars, espnet2.rl.rollout.write_json) and
``close`` shuts the engine down. Each request is answered with
``{"ok": True, ...}`` or ``{"error": traceback}``.
"""

import argparse
import json
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

from .config import ESPnetRLConfig
from .engine import RolloutEngine
from .rollout import read_json, write_json
from .utils import resolve_callable


class FunctionEngine(RolloutEngine):
    """A per-call generation function: the optional OWSM vLLM backend."""

    def __init__(self, config, model_path, *, directory, replica=0):
        super().__init__(config, model_path, directory=directory, replica=replica)
        self.function = resolve_callable(config.vllm_generate_factory)
        self.model_path = str(model_path)

    def update(self, model_path):
        self.model_path = str(model_path)

    def generate(self, sources, rollout_id):
        # The function reads nothing but the export directory it is given.
        return self.function(self.config, self.model_path, sources, rollout_id)


class RolloutServer:
    """One engine and the integer policy version it currently holds."""

    def __init__(self, config, directory, replica):
        self.config, self.directory, self.replica = config, directory, replica
        self.engine = self.version = None

    def load(self, model, version):
        start = time.monotonic()
        if self.engine is None:
            build = (
                resolve_callable(self.config.sglang_engine)
                if self.config.rollout_backend == "sglang"
                else FunctionEngine
            )
            self.engine = build(
                self.config, model, directory=self.directory, replica=self.replica
            )
            action = "built"
        else:
            self.engine.update(model)
            action = "updated"
        self.version = version
        print(
            f"rollout engine {action} for policy {version} from {model} "
            f"in {time.monotonic() - start:.1f} s",
            flush=True,
        )
        return {"version": version}

    def generate(self, requests, output, rollout_id):
        if self.version is None:
            raise RuntimeError("Load a policy before generation")
        start = time.monotonic()
        sources = read_json(requests)
        generations = self.engine.generate(sources, rollout_id)
        version = self.version
        if self.config.execution_mode == "sync" and self.config.collocated:
            # The training rank on this GPU needs the memory until the next
            # load, and a released engine may keep no weights until then.
            self.engine.release()
            self.version = None
        for generation in generations:
            generation["policy_version"] = version
        result = {
            "rollout_id": rollout_id,
            "policy_version": version,
            "rollout_backend": self.config.rollout_backend,
            "generations": generations,
        }
        write_json(output, result, allow_nan=False)
        print(
            f"rollout {rollout_id}: {len(generations)} generations with policy "
            f"{version} in {time.monotonic() - start:.1f} s",
            flush=True,
        )
        return {"version": version}

    def close(self):
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()


def serve(connection, server):
    """Answer requests until ``close`` or until the client goes away."""
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            operation = request.pop("op")
            try:
                if operation not in ("load", "generate", "close"):
                    raise ValueError(f"Unknown rollout operation {operation!r}")
                reply = {"ok": True, **(getattr(server, operation)(**request) or {})}
            except Exception:
                reply = {"error": traceback.format_exc()}
            try:
                connection.send(reply)
            except OSError:
                return  # The client died while this request ran.
            if operation == "close":
                return
    finally:
        server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    config = ESPnetRLConfig(**json.loads(Path(args.config).read_text()))
    config.validate()
    import torch

    if config.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with Connection(args.fd) as connection:
        serve(connection, RolloutServer(config, Path(args.directory), args.replica))


if __name__ == "__main__":
    main()
