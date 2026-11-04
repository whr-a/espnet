# Copyright 2025 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Rewards: a registry of named rewards and the scorer the trainer calls.

A reward scores one candidate, ``reward(source, generation)``, or, when it
sets ``reward.batched = True``, a whole batch, ``reward(sources,
generations)``, returning one value per candidate. A value is the reward, or
a dict holding it as ``"reward"`` next to the numbers it was made of
(accuracy, format, WER, ...); every candidate reports the same ones, and
their batch means are logged as ``reward/<name>``. A reward may also expose
``reward.validate_sources(records)``, which
:func:`espnet2.rl.data.read_manifest` calls before any model, GPU or Ray
actor starts.

A recipe names its reward in ``reward_function`` either by a name in
:data:`REWARDS` or as ``module:callable``. A new reward is therefore one
function in any module plus, for a short name, one line in :data:`REWARDS`;
the registry stores import paths, so naming a model family's reward here
imports nothing until a recipe asks for it.
"""

import math

import torch

from espnet2.rl.utils import resolve_callable

#: Reward names a recipe's ``reward_function`` or a mixed manifest's
#: ``reward_task`` may use instead of a ``module:callable`` path.
REWARDS = {
    # Model-agnostic rewards on generated text. "wer" is -WER, the default;
    # "asr_wer" is -min(WER, 1) with the WER logged, as mixed manifests use it.
    "wer": "espnet2.rl.rewards.asr:word_error_reward",
    "asr_wer": "espnet2.rl.rewards.asr:asr_wer_reward",
    "longform_wer": "espnet2.rl.rewards.asr:longform_wer_reward",
    "speaker_id": "espnet2.rl.rewards.speaker:speaker_id_reward",
    "speaker_verification": "espnet2.rl.rewards.speaker:speaker_verification_reward",
    "multispeaker_cpwer": "espnet2.rl.rewards.cpwer:multispeaker_cpwer_reward",
    "multispeaker_cpwer_shaped": (
        "espnet2.rl.rewards.cpwer:multispeaker_cpwer_shaped_reward"
    ),
    "mixed": "espnet2.rl.rewards.mixed:mixed_task_reward",
    # Rewards that need a model family's own decoding or parsing.
    "codec_block": "espnet2.speechlm.rl.rewards:codec_block_reward",
    "bagpiper_qa_v1": (
        "espnet2.speechlm.model.speechlm.rl.bagpiper.rewards:qa_v1_reward"
    ),
    "bagpiper_qa_think": (
        "espnet2.speechlm.model.speechlm.rl.bagpiper.rewards:qa_think_reward"
    ),
    "bagpiper_qa_native": (
        "espnet2.speechlm.model.speechlm.rl.bagpiper.rewards:qa_bagpiper_v2_reward"
    ),
    "bagpiper_tts_asr_wer": (
        "espnet2.speechlm.model.speechlm.rl.bagpiper.rewards:tts_asr_wer"
    ),
}

#: The reward of a recipe that names none.
DEFAULT_REWARD = "wer"

#: Rollout modes whose sampled actions are text. Audio-output modes need a
#: reward that decodes codec actions, such as ``bagpiper_tts_asr_wer``.
TEXT_MODES = ("", "audio_text", "text_only", "text_text", "text_dialogue")


def get_reward(name=None):
    """The reward a recipe names: a :data:`REWARDS` name or ``module:callable``.

    No name selects :data:`DEFAULT_REWARD`.
    """
    name = name or DEFAULT_REWARD
    if name in REWARDS:
        return resolve_callable(REWARDS[name])
    if ":" not in str(name):
        raise ValueError(
            f"Unknown reward {name!r}: use one of {', '.join(sorted(REWARDS))} "
            "or an explicit module:callable"
        )
    return resolve_callable(name)


def score_candidates(reward, sources, generations):
    """One dict per candidate holding ``"reward"``, honoring a batched reward."""
    if getattr(reward, "batched", False):
        results = list(reward(sources, generations))
    else:
        results = [
            reward(source, generation)
            for source, generation in zip(sources, generations, strict=True)
        ]
    if len(results) != len(sources):
        raise ValueError("The reward function must score every candidate")
    return [
        value if isinstance(value, dict) else {"reward": value} for value in results
    ]


def outcome_rewards(batch, function=None):
    """Score every candidate; return the rewards, their token-level placement
    and the metrics to log. ``function`` is the recipe's ``reward_function``.
    """
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
    rows = score_candidates(get_reward(function), sources, generations)
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


def require_text_mode(source, task):
    mode = source.get("mode", "")
    if mode not in TEXT_MODES:
        raise ValueError(
            f"{task} scores generated text; rollout mode {mode!r} emits other actions"
        )


def require_fields(records, task, fields):
    """Reject a manifest that cannot supervise ``task`` before training starts."""
    for record in records:
        missing = [name for name in fields if record.get(name) in (None, "", [])]
        if missing:
            raise ValueError(
                f"{task} requires {', '.join(fields)} in every manifest record; "
                f"record {record.get('id')!r} is missing {', '.join(missing)}. "
                f"Regenerate the manifest with the matching espnet2.rl.task_data "
                f"preparer."
            )
        declared = record.get("reward_task")
        if declared is not None and declared != task:
            raise ValueError(
                f"Record {record.get('id')!r} declares reward_task {declared!r} but "
                f"the configured reward is {task}. Point reward_function at the "
                f"declared task, or regenerate the manifest."
            )
