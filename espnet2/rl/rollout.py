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
"""Inference subprocess isolation and immutable checkpoint synchronization."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointRollout:
    """Each sampling subprocess loads the exact newly exported actor version.

    Keeping inference in its own interpreter avoids forcing native ESPnet to use
    the inference fork's torch/transformers ABI. No serving group is touched.
    """

    def __init__(self, config):
        self.config = config
        self.directory = Path(config.output_dir) / "rollout"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.version = None

    def sync_weights(self, policy, step):
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
        environment = self._environment()
        with (staging / "export.log").open("w") as stream:
            subprocess.run(
                args,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        metadata = {"step": step, "native_sha256": sha256(native), "hf_files": {}}
        for path in sorted((staging / "hf").rglob("*")):
            if path.is_file():
                metadata["hf_files"][str(path.relative_to(staging / "hf"))] = sha256(
                    path
                )
        if not metadata["hf_files"]:
            raise RuntimeError("The ESPnet exporter produced no model files")
        (staging / "sync.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(directory)
        self.version = directory
        return metadata

    def _environment(self):
        environment = dict(os.environ)
        paths = [str(Path(__file__).resolve().parents[2])]
        if self.config.rollout_backend == "sglang" and self.config.sglang_source:
            paths.insert(0, str(Path(self.config.sglang_source).resolve() / "python"))
        elif self.config.vllm_source:
            paths.insert(0, str(Path(self.config.vllm_source).resolve()))
        if self.config.miles_source:
            paths.append(str(Path(self.config.miles_source).resolve()))
        environment["PYTHONPATH"] = os.pathsep.join(
            paths + [environment.get("PYTHONPATH", "")]
        )
        environment["CUDA_VISIBLE_DEVICES"] = self.config.rollout_gpus
        if self.config.rollout_backend == "vllm":
            environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.config.dtype == "float32":
            environment["TRITON_F32_DEFAULT"] = "ieee"
            environment["NVIDIA_TF32_OVERRIDE"] = "0"
            environment["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        # A subprocess is an independent GPU consumer, never a torchrun rank.
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            environment.pop(name, None)
        return environment

    def generate(self, sources, step):
        if self.version is None:
            raise RuntimeError("Weights must be synchronized before rollout")
        requests = self.version / "requests.json"
        responses = self.version / "responses.json"
        requests.write_text(json.dumps(sources) + "\n")
        args = [
            self.config.rollout_python,
            "-m",
            "espnet2.rl.rollout_worker",
            "--config",
            str(Path(self.config.output_dir) / "frozen_config.json"),
            "--model",
            str((self.version / "hf").resolve()),
            "--requests",
            str(requests.resolve()),
            "--output",
            str(responses.resolve()),
            "--step",
            str(step),
        ]
        with (self.version / "generation.log").open("w") as stream:
            subprocess.run(
                args,
                env=self._environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result = json.loads(responses.read_text())
        if result["policy_step"] != step or result["native_sha256"] != sha256(
            self.version / "native.pt"
        ):
            raise RuntimeError(
                "Rollout consumed a stale or different policy checkpoint"
            )
        return result["generations"]


def save_checkpoint(path, policy, optimizer, step, config):
    """Atomic native ESPnet checkpoint, optimizer and CPU torch RNG snapshot.

    Runs use immutable directories. Resume is not implemented; this snapshot
    alone is not a complete distributed or CUDA random-state recovery recipe.
    """
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": policy.checkpoint_state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)
    # Native .pth permits direct ESPnet inference and exporter reuse.
    native = path.with_name(path.stem + ".native.pth")
    torch.save(policy.checkpoint_state_dict(), native)
    if config.train_config:
        shutil.copyfile(config.train_config, path.parent / "espnet_train_config.yaml")
