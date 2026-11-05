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
"""Rank zero's local rollout: one persistent engine process per rollout GPU.

Disk publication exports each policy version; the optional Miles path reuses
one bootstrap export and transfers complete policies in memory.
"""

import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ..utils import sha256
from .process import RolloutProcess, subprocess_environment


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
            "espnet2.rl.rollout.export_worker",
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
            from .weights import EngineClient, MilesWeightPublisher

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
