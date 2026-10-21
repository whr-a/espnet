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

import numpy as np
import torch

from .batch import RolloutBatch
from .types import validate_binary_mask
from .utils import object_array, resolve_callable


def read_manifest(path, *, config=None):
    """Read sources and optionally enforce the configured reward's contract.

    A reward may expose ``validate_sources(records)`` for CPU checks before
    models or distributed workers start. Legacy readers do not resolve hooks.
    """
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
                record["audio"] = str(
                    audio if audio.is_absolute() else Path(path).parent / audio
                )
            records.append(record)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("The manifest must be nonempty and contain unique ids")
    if config is not None and config.reward_function:
        callback = resolve_callable(config.reward_function)
        validator = getattr(callback, "validate_sources", None)
        if validator is not None:
            validator(records)
    return records


def verify_data_admission(config, path):
    """Bind the live manifests and their media to a prepared admission receipt.

    The receipt records the SHA256 of every manifest it admitted and the
    record counts it saw. Training re-derives both from disk, re-hashes each
    referenced audio file, and requires the training and heldout media to stay
    disjoint. Recipe-specific provenance stays in the receipt itself.
    """
    from .rollout import sha256

    admission = json.loads(Path(path).read_text())
    if admission.get("training_ready") is not True:
        raise ValueError("Data has not passed its preparation admission")
    recorded = admission.get("manifest_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Data admission must record its manifest SHA256 values")
    if not config.manifest or not config.eval_manifest:
        raise ValueError("Data admission covers a training and a heldout manifest")
    manifests = {"train": config.manifest, "heldout": config.eval_manifest}
    splits = {}
    for role, manifest in manifests.items():
        # Earlier 32-prompt receipts called the training manifest "train32".
        # Keep that format readable while binding each hash to its own role.
        expected = recorded.get(role)
        if role == "train" and expected is None:
            expected = recorded.get("train32")
        if sha256(manifest) != expected:
            raise ValueError(f"The {role} manifest differs from its admission")
        splits[role] = read_manifest(manifest)
    if admission.get("counts") != {role: len(rows) for role, rows in splits.items()}:
        raise ValueError("Manifest record counts differ from the admission")
    media = {}
    for role, rows in splits.items():
        hashes = set()
        for row in rows:
            if (
                not row.get("audio")
                or not row.get("audio_sha256")
                or sha256(row["audio"]) != row["audio_sha256"]
            ):
                raise ValueError("Admitted audio changed or lacks a frozen SHA256")
            if row["audio_sha256"] in hashes:
                raise ValueError("Independent prompts cannot duplicate admitted audio")
            hashes.add(row["audio_sha256"])
        media[role] = hashes
    if media.get("train", set()) & media.get("heldout", set()):
        raise ValueError("Training and heldout audio are not independent")
    return admission


def prepared_sha256(source_id, data, digests=None):
    """SHA256 of a source's prepared native inputs.

    Canonical JSON in which a numpy array enters by its dtype, shape and the
    SHA256 of its bytes rather than one number at a time. ``digests`` keeps
    the entries of arrays already seen, by identity, across calls.
    """
    digests = {} if digests is None else digests

    def array(value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Prepared native inputs cannot hold {type(value)}")
        if id(value) not in digests:
            contiguous = np.ascontiguousarray(value)
            digests[id(value)] = value, {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous).hexdigest(),
            }
        return digests[id(value)][1]

    encoded = json.dumps(
        {"source_id": source_id, "data": data},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=array,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _shared_arrays(value, memo):
    """A deepcopy memo under which copies share the numpy arrays of ``value``.

    Native feature arrays are hashed by their bytes and never written, so the
    candidates of one prompt can hold the same one.
    """
    if isinstance(value, np.ndarray):
        memo[id(value)] = value
    elif isinstance(value, dict):
        for item in value.values():
            _shared_arrays(item, memo)
    elif isinstance(value, list | tuple):
        for item in value:
            _shared_arrays(item, memo)
    return memo


def collate_rollouts(sources, generations, group_size):
    """Generations carry exactly the actions sampled by the inference worker."""
    if len(generations) != len(sources) * group_size:
        raise ValueError("Incomplete rollout groups")
    repeated_sources = []
    raw_generations = []
    digests = {}
    prefixes, actions, action_masks, log_probs, ids, texts, stream_masks = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, generation in enumerate(generations):
        source = sources[index // group_size]
        if generation["source_id"] != source["id"]:
            raise ValueError("Rollout ordering/source identity differs from manifest")
        action = torch.as_tensor(generation["token_ids"], dtype=torch.long)
        probability = torch.as_tensor(generation["log_probs"], dtype=torch.float32)
        if action.ndim not in (1, 2) or not len(action):
            raise ValueError("Empty or malformed rollout actions")
        if (
            probability.shape != action.shape[:1]
            or not torch.isfinite(probability).all()
        ):
            raise ValueError(
                "Every sampled action needs a finite behavior log probability"
            )
        action_mask = torch.as_tensor(
            generation.get("action_mask", [1] * len(action)), dtype=torch.float32
        )
        validate_binary_mask(action_mask, "Rollout action_mask")
        if action_mask.shape != probability.shape or not action_mask.sum():
            raise ValueError("Every rollout requires at least one active action")
        actions.append(action)
        action_masks.append(action_mask)
        log_probs.append(probability)
        sample_source = copy.deepcopy(source)
        prepared = generation.get("prepared_native_source")
        if prepared is not None:
            if prepared.get("source_id") != source["id"]:
                raise ValueError(
                    "Prepared native source identity differs from the original manifest"
                )
            payload = prepared["data"]
            allowed = {
                "native_model_inputs",
                "prompt_codec_mask",
                "sampling_extra_args",
                "mode",
                "native_preparation",
            }
            if set(payload) - allowed:
                raise ValueError(
                    "Inference preparation attempted to replace source identity or supervision"
                )
            digest = prepared_sha256(prepared["source_id"], payload, digests)
            if digest != prepared["sha256"]:
                raise ValueError(
                    "Prepared native source hash differs from rollout evidence"
                )
            sample_source.update(copy.deepcopy(payload, _shared_arrays(payload, {})))
        # Records of how the engine sampled, for the native scorer.
        for name in (
            "sampling_contract",
            "sampling_contracts",
            "codec_frame_mask",
            "support",
        ):
            if name in generation:
                sample_source[name] = generation[name]
        repeated_sources.append(sample_source)
        # Every rollout decodes fresh records that are only read from here on;
        # a deep copy would cost seconds per batch of frame lists.
        raw_generations.append(generation)
        if action.ndim == 2:
            if generation.get("stream_mask") is None:
                raise ValueError("Multistream rollout requires a sampling mask")
            stream_mask = torch.as_tensor(
                generation.get("stream_mask"), dtype=torch.float32
            )
            if stream_mask.shape != action.shape:
                raise ValueError(
                    "Multistream rollout requires the true per-stream sampling mask"
                )
            validate_binary_mask(stream_mask, "Rollout stream_mask")
            if not torch.equal(action_mask.bool(), stream_mask.bool().any(-1)):
                raise ValueError("Rollout action_mask differs from sampled streams")
            stream_masks.append(stream_mask)
        prefixes.append(generation["prompt_token_ids"])
        ids.append(source["id"])
        texts.append(generation.get("text", ""))
    rank = actions[0].ndim
    if any(
        action.ndim != rank or action.shape[1:] != actions[0].shape[1:]
        for action in actions
    ):
        raise ValueError(
            "One batch cannot mix scalar tokens and different codebook counts"
        )
    width = max(map(len, actions))
    response = torch.zeros(
        (len(actions), width, *actions[0].shape[1:]), dtype=torch.long
    )
    mask, behavior = torch.zeros((len(actions), width)), torch.zeros(
        (len(actions), width)
    )
    for row, (action, active, probability) in enumerate(
        zip(actions, action_masks, log_probs, strict=True)
    ):
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
