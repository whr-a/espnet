"""Common learner-window timing for synchronous and asynchronous RL."""

import math
import time
from contextlib import contextmanager

import torch


class PhaseProfiler:
    """Opt-in synchronized phase walls and allocator samples on one learner GPU.

    Memory samples are phase-end residency; peaks cover the whole update.
    Nested transfer/optimizer timings are subsets, not additional update time.
    """

    def __init__(self, device, enabled=False):
        self.device, self.enabled = torch.device(device), enabled
        self.metrics, self.starts = {}, {}

    def synchronize(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self):
        self.metrics.clear()
        self.starts.clear()
        self.synchronize()
        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def start(self, name):
        if self.enabled:
            self.synchronize()
            self.starts[name] = time.monotonic()

    def stop(self, name, *, memory=True):
        if not self.enabled:
            return
        self.synchronize()
        key = f"perf/{name}_time"
        self.metrics[key] = (
            self.metrics.get(key, 0.0) + time.monotonic() - self.starts.pop(name)
        )
        if memory and self.device.type == "cuda":
            for kind in ("allocated", "reserved"):
                self.metrics[f"memory/{name}_{kind}_bytes"] = getattr(
                    torch.cuda, f"memory_{kind}"
                )(self.device)

    def result(self):
        if self.enabled and self.device.type == "cuda":
            for kind in ("allocated", "reserved"):
                self.metrics[f"memory/peak_{kind}_bytes"] = getattr(
                    torch.cuda, f"max_memory_{kind}"
                )(self.device)
        return dict(self.metrics)


@contextmanager
def profile_phase(owner, name, *, memory=True):
    profiler = getattr(owner, "profiler", None)
    if profiler is not None:
        profiler.start(name)
    try:
        yield
    finally:
        if profiler is not None:
            profiler.stop(name, memory=memory)


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
