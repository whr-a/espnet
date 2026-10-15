"""Common learner-window timing for synchronous and asynchronous RL."""

import math


def learner_timing(start, end, previous_end, *, warmup, start_wall, end_wall):
    """Include input transfer, native scoring, reward and the optimizer update.

    Wait starts at the previous learner window's end and includes everything
    outside the next learner call, such as save/export/rollout and scheduling.
    Initial model loading and first rollout are outside the first wait metric;
    report full orchestration wall time separately. The first update is warmup.
    """
    train = end - start
    wait = 0.0 if previous_end is None else start - previous_end
    if (
        not all(math.isfinite(value) for value in (train, wait, start_wall, end_wall))
        or train <= 0
        or wait < 0
    ):
        raise ValueError("Learner timing requires finite ordered monotonic boundaries")
    return {
        "perf/train_time": train,
        "perf/train_wait_time": wait,
        "perf/step_time": train + wait,
        "perf/wait_time_ratio": wait / (train + wait),
        "perf/warmup": warmup,
        "perf/train_start_wall": start_wall,
        "perf/train_end_wall": end_wall,
        "perf/timing_scope": "input_transfer_native_scoring_reward_optimizer",
        "perf/initial_wait_excluded": previous_end is None,
    }
