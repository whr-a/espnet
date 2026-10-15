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
import re
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


def words(text):
    """Words the WER compares: NFKC, casefolded, punctuation removed.

    LibriSpeech-style references are unpunctuated while models such as OWSM
    write punctuated text, so raw tokens would count punctuation as word
    errors and a WER reward would mostly teach the model to drop it.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w\s]", " ", normalized).split()


def word_error_rate(reference, hypothesis):
    expected = words(reference)
    return edit_distance(expected, words(hypothesis)) / max(1, len(expected))


def word_error_reward(source, generation):
    mode = source.get("mode", "")
    if mode and mode not in ("text_only", "text_text", "text_dialogue"):
        raise ValueError(
            "Audio/dialogue rollout modes require an explicit task reward_function; WER expects text output"
        )
    if "reference" not in source:
        raise ValueError("WER reward requires a reference in every manifest record")
    return -word_error_rate(source["reference"], generation["text"])


def outcome_rewards(batch, function=None):
    """Score every candidate; return the rewards, their token-level placement
    and the metrics to log.

    A reward function scores one candidate, ``function(source, generation)``,
    or, when it sets ``function.batched = True``, the whole batch at once,
    ``function(sources, generations)``, returning one value per candidate. A
    value is the reward, or a dict holding it as ``"reward"`` next to the
    numbers it was made of (accuracy, format, WER, ...); every candidate must
    report the same ones, and their batch means are logged as ``reward/<name>``.
    """
    callback = resolve_callable(function) if function else word_error_reward
    sources = list(batch.non_tensor_batch["source"])
    generations = [
        {**raw, "text": text, "token_ids": tokens[: int(length)].tolist()}
        for text, tokens, length, raw in zip(
            batch.non_tensor_batch["text"],
            batch.batch["responses"],
            batch.batch["response_lengths"],
            batch.non_tensor_batch["generation"],
            strict=True,
        )
    ]
    if getattr(callback, "batched", False):
        results = list(callback(sources, generations))
    else:
        results = [
            callback(source, generation)
            for source, generation in zip(sources, generations, strict=True)
        ]
    if len(results) != len(sources):
        raise ValueError("The reward function must score every candidate")
    rows = [
        value if isinstance(value, dict) else {"reward": value} for value in results
    ]
    columns = {name: [] for name in rows[0]}
    for source, row in zip(sources, rows, strict=True):
        if row.keys() != columns.keys():
            raise ValueError("Every candidate must report the same reward components")
        for name, value in row.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite reward {name} for {source['id']}")
            columns[name].append(value)
    scalar = torch.tensor(columns.pop("reward"), dtype=torch.float32)
    metrics = {
        "reward/mean": scalar.mean().item(),
        "reward/min": scalar.min().item(),
        "reward/max": scalar.max().item(),
        **{
            f"reward/{name}": sum(values) / len(values)
            for name, values in columns.items()
        },
    }
    token_rewards = torch.zeros_like(batch.batch["response_mask"])
    # An action mask may contain forced codec-delay frames: reward is assigned
    # to the last active action, rather than to a masked padding frame.
    positions = torch.arange(token_rewards.size(1)).expand_as(token_rewards)
    last_active = (
        positions.masked_fill(batch.batch["response_mask"] == 0, -1).max(-1).values
    )
    token_rewards[torch.arange(len(batch)), last_active] = scalar
    return scalar, token_rewards, metrics
