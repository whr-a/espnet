# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""One RL run over several tasks, each record scored by its own reward."""

from . import get_reward, score_candidates


def mixed_task_reward(sources, generations):
    """One RL run over several tasks: each record is scored by its own reward.

    Every record declares ``reward_task``; its candidates are scored by that
    task's reward and keep its value as ``reward``. Mixing scales is safe
    because GRPO normalizes within a prompt group: the trainer passes the
    per-prompt ``uid`` to ``compute_grpo_outcome_advantage``, which groups
    candidates by that id before Miles normalizes. Two tasks therefore never
    share a baseline.

    ``outcome_rewards`` requires one fixed set of logged components, so this
    reward reports, for every task present in the manifest, ``share_<task>``
    (1 on that task's candidates) and ``sum_<task>`` (its reward there, 0
    elsewhere). The batch mean reward of one task is
    ``reward/sum_<task> / reward/share_<task>``; a task's own components stay
    in its single-task runs.
    """
    tasks = sorted({str(source["reward_task"]) for source in sources})
    rows = [None] * len(sources)
    for task in tasks:
        indices = [
            i for i, source in enumerate(sources) if str(source["reward_task"]) == task
        ]
        scored = score_candidates(
            get_reward(task),
            [sources[i] for i in indices],
            [generations[i] for i in indices],
        )
        for index, row in zip(indices, scored, strict=True):
            value = float(row["reward"])
            rows[index] = {
                "reward": value,
                **{f"share_{name}": float(name == task) for name in tasks},
                **{f"sum_{name}": value if name == task else 0.0 for name in tasks},
            }
    return rows


mixed_task_reward.batched = True


def _validate_mixed(records):
    missing = [r.get("id") for r in records if not r.get("reward_task")]
    if missing:
        raise ValueError(
            f"mixed_task_reward needs reward_task in every record; missing for "
            f"{', '.join(map(repr, missing[:5]))}"
        )
    grouped = {}
    for record in records:
        grouped.setdefault(str(record["reward_task"]), []).append(record)
    for task, subset in grouped.items():
        validator = getattr(get_reward(task), "validate_sources", None)
        if validator is not None:
            validator(subset)


mixed_task_reward.validate_sources = _validate_mixed
