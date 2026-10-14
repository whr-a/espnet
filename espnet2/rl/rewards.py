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
"""Local WER reward and an explicit custom reward hook."""

import math
import unicodedata

import torch

from .utils import resolve_callable


def edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        updated = [i]
        for j, actual in enumerate(hypothesis, 1):
            updated.append(
                min(updated[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual))
            )
        row = updated
    return row[-1]


def word_error_reward(source, generation):
    mode = source.get("mode", "")
    if mode and mode not in ("text_only", "text_text", "text_dialogue"):
        raise ValueError(
            "Audio/dialogue rollout modes require an explicit task reward_function; WER expects text output"
        )
    if "reference" not in source:
        raise ValueError("WER reward requires a reference in every manifest record")
    reference = unicodedata.normalize("NFKC", source["reference"]).casefold().split()
    hypothesis = unicodedata.normalize("NFKC", generation["text"]).casefold().split()
    return -edit_distance(reference, hypothesis) / max(1, len(reference))


def outcome_rewards(batch, function=None):
    callback = resolve_callable(function) if function else word_error_reward
    values = []
    for source, text, tokens, length, raw in zip(
        batch.non_tensor_batch["source"],
        batch.non_tensor_batch["text"],
        batch.batch["responses"],
        batch.batch["response_lengths"],
        batch.non_tensor_batch["generation"],
        strict=True,
    ):
        value = float(
            callback(
                source,
                {**raw, "text": text, "token_ids": tokens[: int(length)].tolist()},
            )
        )
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite reward for {source['id']}")
        values.append(value)
    scalar = torch.tensor(values, dtype=torch.float32)
    token_rewards = torch.zeros_like(batch.batch["response_mask"])
    # An action mask may contain forced codec-delay frames: reward is assigned
    # to the last active action, rather than to a masked padding frame.
    positions = torch.arange(token_rewards.size(1)).expand_as(token_rewards)
    last_active = (
        positions.masked_fill(batch.batch["response_mask"] == 0, -1).max(-1).values
    )
    token_rewards[torch.arange(len(batch)), last_active] = scalar
    return scalar, token_rewards
