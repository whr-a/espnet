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
"""Auditable multimodal rollout conversion for native ESPnet training."""

import copy
import hashlib
import json
from pathlib import Path

import torch

from .batch import RolloutBatch

from .utils import object_array


def read_manifest(path):
    records = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"Manifest line {line_number} requires a unique id")
            if "audio" in record:
                audio = Path(record["audio"])
                record["audio"] = str(audio if audio.is_absolute() else Path(path).parent / audio)
            records.append(record)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("The manifest must be nonempty and contain unique ids")
    return records


def collate_rollouts(sources, generations, group_size):
    """Generations carry exactly the actions sampled by the inference worker."""
    if len(generations) != len(sources) * group_size:
        raise ValueError("Incomplete rollout groups")
    repeated_sources = []
    raw_generations = []
    prefixes, actions, action_masks, log_probs, ids, texts, stream_masks = [], [], [], [], [], [], []
    for index, generation in enumerate(generations):
        source = sources[index // group_size]
        if generation["source_id"] != source["id"]:
            raise ValueError("Rollout ordering/source identity differs from manifest")
        action = torch.as_tensor(generation["token_ids"], dtype=torch.long)
        probability = torch.as_tensor(generation["log_probs"], dtype=torch.float32)
        if action.ndim not in (1, 2) or not len(action):
            raise ValueError("Empty or malformed rollout actions")
        if probability.shape != action.shape[:1] or not torch.isfinite(probability).all():
            raise ValueError("Every sampled action needs a finite behavior log probability")
        action_mask = torch.as_tensor(generation.get("action_mask", [1] * len(action)), dtype=torch.float32)
        if action_mask.shape != probability.shape or not action_mask.sum():
            raise ValueError("Every rollout requires at least one active action")
        actions.append(action)
        action_masks.append(action_mask)
        log_probs.append(probability)
        sample_source = copy.deepcopy(source)
        prepared = generation.get("prepared_native_source")
        if prepared is not None:
            if prepared.get("source_id") != source["id"]:
                raise ValueError("Prepared native source identity differs from the original manifest")
            payload = prepared["data"]
            allowed = {"native_model_inputs", "prompt_codec_mask", "sampling_extra_args", "mode", "native_preparation"}
            if set(payload) - allowed:
                raise ValueError("Inference preparation attempted to replace source identity or supervision")
            encoded = json.dumps(
                {"source_id": prepared["source_id"], "data": payload},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            if hashlib.sha256(encoded).hexdigest() != prepared["sha256"]:
                raise ValueError("Prepared native source hash differs from rollout evidence")
            sample_source.update(copy.deepcopy(payload))
        if "sampling_contracts" in generation:
            sample_source["sampling_contracts"] = generation["sampling_contracts"]
        if "codec_frame_mask" in generation:
            sample_source["codec_frame_mask"] = generation["codec_frame_mask"]
        repeated_sources.append(sample_source)
        raw_generations.append(copy.deepcopy(generation))
        if action.ndim == 2:
            stream_mask = torch.as_tensor(generation.get("stream_mask"), dtype=torch.float32)
            if stream_mask.shape != action.shape:
                raise ValueError("Multistream rollout requires the true per-stream sampling mask")
            stream_masks.append(stream_mask)
        prefixes.append(generation["prompt_token_ids"])
        ids.append(source["id"])
        texts.append(generation.get("text", ""))
    rank = actions[0].ndim
    if any(action.ndim != rank or action.shape[1:] != actions[0].shape[1:] for action in actions):
        raise ValueError("One batch cannot mix scalar tokens and different codebook counts")
    width = max(map(len, actions))
    response = torch.zeros((len(actions), width, *actions[0].shape[1:]), dtype=torch.long)
    mask, behavior = torch.zeros((len(actions), width)), torch.zeros((len(actions), width))
    for row, (action, active, probability) in enumerate(zip(actions, action_masks, log_probs, strict=True)):
        response[row, : len(action)] = action
        mask[row, : len(action)] = active
        behavior[row, : len(action)] = probability
    tensors = {
        "responses": response,
        "response_mask": mask,
        "response_lengths": torch.tensor(list(map(len, actions))),
        "rollout_log_probs": behavior,
    }
    if rank == 2:
        stream_mask = torch.zeros_like(response, dtype=torch.float32)
        for row, active in enumerate(stream_masks):
            stream_mask[row, : len(active)] = active
        tensors["response_stream_mask"] = stream_mask
    return RolloutBatch.from_dict(
        tensors=tensors,
        non_tensors={
            "source": object_array(repeated_sources),
            "prompt_token_ids": object_array(prefixes),
            "uid": object_array(ids),
            "text": object_array(texts),
            "generation": object_array(raw_generations),
        },
    )
