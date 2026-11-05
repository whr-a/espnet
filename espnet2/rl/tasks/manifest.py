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
"""The training manifest: reading it, binding it to its admission receipt and
choosing each update's prompts.

``read_manifest`` runs the configured reward's ``validate_sources`` before
any model, GPU or Ray actor starts.
"""

import json
from pathlib import Path

import torch

from espnet2.rl.rewards import get_reward
from espnet2.rl.utils import sha256


def read_manifest(path, *, config=None):
    """Read sources and optionally enforce the configured reward's contract.

    A reward may expose ``validate_sources(records)`` for CPU checks before
    models or distributed workers start. Without a configured reward
    function nothing is resolved.
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
        callback = get_reward(config.reward_function)
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


def select_training_sources(sources, update_index, config):
    """Read a deterministic absolute prompt cursor, including epoch crossings."""
    if len({source["id"] for source in sources}) < config.prompts_per_step:
        raise ValueError("The prompt batch requires distinct source records")
    start = update_index * config.prompts_per_step
    permutations = {}
    result = []
    for absolute_index in range(start, start + config.prompts_per_step):
        epoch, offset = divmod(absolute_index, len(sources))
        if epoch not in permutations:
            order = list(range(len(sources)))
            if config.shuffle_train_data:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(config.data_seed + epoch)
                order = torch.randperm(len(sources), generator=generator).tolist()
            permutations[epoch] = order
        result.append(sources[permutations[epoch][offset]])
    if len({source["id"] for source in result}) != config.prompts_per_step:
        raise ValueError("An epoch boundary produced duplicate prompts in one batch")
    return result
