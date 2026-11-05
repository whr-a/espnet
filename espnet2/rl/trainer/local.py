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
"""Single-host entry: process group, frozen run directory, device census.

``run(config)`` is what ``espnet2.bin.rl_train`` executes for a local
layout, in every torchrun rank when the recipe declares several training GPUs.
"""

import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed

from espnet2.rl.rewards.asr import word_error_reward
from espnet2.rl.rollout.engine import engine_class
from espnet2.rl.tasks.manifest import read_manifest
from espnet2.rl.utils import (
    build_policy,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed

from .algorithms import verify_miles
from .learner import NativeGRPOTrainer


def host_identity():
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "network_namespace": os.readlink("/proc/self/ns/net"),
    }


def run(config):
    config.validate()
    sources = read_manifest(config.manifest, config=config)
    if config.dtype == "float32":
        # Actor and rollout must use the same declared full FP32 precision.
        # Rollout sets Triton's independent tl.dot option before child imports.
        backend = getattr(torch.backends, get_device_name(), None)
        if backend is not None and hasattr(backend, "matmul"):
            backend.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    engine_class(config)
    miles_evidence = verify_miles(config)
    set_all_random_seed(config.seed)
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        device_type = torch.device(config.device).type
        distributed.init_process_group(
            get_nccl_backend() if device_type != "cpu" else "gloo",
            timeout=timedelta(minutes=config.distributed_timeout_minutes),
        )
        if device_type != "cpu":
            get_torch_device().set_device(int(os.environ["LOCAL_RANK"]))
            config.device = f"{device_type}:{os.environ['LOCAL_RANK']}"
    else:
        device = torch.device(config.device)
        if device.type == "cuda" and device.index is not None:
            # One rank keeps every GPU visible, so its current device would stay
            # 0; make the configured card current, as each multi-rank worker does.
            get_torch_device().set_device(device)
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    output = Path(config.output_dir).resolve()
    config.output_dir = str(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        frozen = output / "frozen_config.json"
        if frozen.exists():
            raise FileExistsError(
                f"Use a fresh output_dir; refusing to change frozen config {frozen}"
            )
        frozen.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (output / "miles_library.json").write_text(
            json.dumps(miles_evidence, indent=2) + "\n"
        )
    if distributed.is_initialized():
        distributed.barrier()
    # Record where every training rank actually landed, then prove the ranks
    # hold one distinct GPU each on a single host. The expected count comes
    # from the recipe's train_gpus, so one rank is as valid as eight.
    if torch.device(config.device).type != "cpu":
        world_size = distributed.get_world_size() if distributed.is_initialized() else 1
        # A single rank is not launched by torchrun, so its local rank is 0.
        local_rank = int(os.environ.get("LOCAL_RANK", "0" if world_size == 1 else "-1"))
        visible = [
            item
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item
        ]
        census = {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "config_device": config.device,
            "cuda_visible_devices": visible,
            "visible_gpu_count": get_torch_device().device_count(),
            "current_device": get_torch_device().current_device(),
            "physical_gpu_id": (
                visible[local_rank] if local_rank in range(len(visible)) else None
            ),
            "host_identity": host_identity(),
            "pid": os.getpid(),
        }
        Path(output, f"train-device-rank-{rank}.json").write_text(
            json.dumps(census, indent=2) + "\n"
        )
        expected = config.train_ranks
        if world_size != expected:
            raise ValueError(
                f"The recipe declares {expected} training ranks; this rank sees "
                f"world_size={world_size}"
            )
        if expected > 1:
            # Several ranks are launched by this toolkit, which pins the
            # devices, so each rank must see exactly them and own its own.
            if census["visible_gpu_count"] != expected or (
                visible and visible != config.train_device_ids
            ):
                raise ValueError(
                    f"Training ranks must occupy the recipe's train_gpus "
                    f"{config.train_device_ids}; this rank sees {visible} and "
                    f"{census['visible_gpu_count']} devices"
                )
            if census["current_device"] != local_rank:
                raise ValueError("Each training rank must own its local device")
        if distributed.is_initialized():
            all_census = [None] * world_size
            distributed.all_gather_object(all_census, census)
            if len({item["physical_gpu_id"] for item in all_census}) != expected or any(
                item["host_identity"] != census["host_identity"] for item in all_census
            ):
                raise ValueError(
                    f"Training ranks do not occupy {expected} distinct GPUs on one host"
                )
        else:
            all_census = [census]
        if rank == 0:
            Path(output, "train_census.json").write_text(
                json.dumps(all_census, indent=2) + "\n"
            )
    # Reward implementations read the same effective configuration as the
    # actor and rollout, including command-line overrides. This path is host
    # metadata and does not become part of the model's prompt.
    for source in sources:
        source["reward_config"] = str(output / "frozen_config.json")
    if config.reward_function is None:
        # Reject missing supervision or an audio-only reward configuration
        # before loading models or occupying inference GPUs.
        for source in sources:
            word_error_reward(source, {"text": ""})
    policy = build_policy(config, config.device)
    trainer = NativeGRPOTrainer(policy, config)
    trainer.source_count = len(sources)
    if config.resume_checkpoint:
        trainer.load_training_checkpoint(config.resume_checkpoint)
    try:
        trainer.train(sources)
    finally:
        if trainer.rollout is not None:
            # The rollout processes hold their engines until training ends.
            trainer.rollout.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()
