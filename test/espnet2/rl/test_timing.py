"""Clock oracles for equal learner windows and save/export wait attribution."""

import json
from types import SimpleNamespace

import pytest
import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.timing import learner_timing
from espnet2.rl.trainer import NativeGRPOTrainer


def test_common_window_excludes_initialization_and_includes_outside_call_wait():
    warmup = learner_timing(100, 110, None, warmup=True, start_wall=1000, end_wall=1010)
    measured = learner_timing(
        140, 170, 110, warmup=False, start_wall=1040, end_wall=1070
    )
    assert warmup["perf/train_wait_time"] == 0
    assert warmup["perf/initial_wait_excluded"] is True
    assert measured["perf/train_time"] == 30
    assert measured["perf/train_wait_time"] == 30
    assert measured["perf/step_time"] == 60
    assert measured["perf/wait_time_ratio"] == 0.5
    assert (
        measured["perf/timing_scope"]
        == "input_transfer_native_scoring_reward_optimizer"
    )
    with pytest.raises(ValueError, match="ordered"):
        learner_timing(10, 9, 8, warmup=False, start_wall=1, end_wall=2)


def test_sync_clock_includes_score_reward_update_and_next_wait_includes_save(
    tmp_path, monkeypatch
):
    import espnet2.rl.trainer as module

    class Clock:
        value = 0.0

        def monotonic(self):
            return self.value

        def time(self):
            return 1000 + self.value

        def advance(self, duration):
            self.value += duration

    clock = Clock()
    monkeypatch.setattr(module, "time", clock)

    class Batch:
        def save_to_disk(self, path):
            clock.advance(2)

    class Rollout:
        def sync_weights(self, policy, step):
            clock.advance(5)

        def generate(self, sources, step):
            clock.advance(10)
            return []

    class Trainer:
        config = ESPnetRLConfig(
            steps=2, warmup_steps=1, prompts_per_step=1, output_dir=str(tmp_path)
        )
        rank, world_size = 0, 1
        device = torch.device("cpu")
        policy, optimizer = None, None
        rollout = Rollout()

        def prepare_batch(self, batch, native_scores):
            clock.advance(4)
            return {}

        def update_actor(self, batch):
            clock.advance(6)
            return {}

    def collate(*args):
        clock.advance(1)
        return Batch()

    def score(*args):
        clock.advance(3)
        return None, None, None

    monkeypatch.setattr(module, "collate_rollouts", collate)
    monkeypatch.setattr(module, "score_native_batch", score)
    monkeypatch.setattr(module, "save_checkpoint", lambda *args: clock.advance(7))
    NativeGRPOTrainer.train(Trainer(), [{"id": "p"}])
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    # 3score + 4reward + 2evidence I/O + 6optimizer = 15 learner seconds.
    assert [row["perf/train_time"] for row in rows] == [15, 15]
    assert [row["perf/train_wait_time"] for row in rows] == [0, 23]
    # Next wait includes prior7save + current5export+10rollout+1collate.
    assert rows[1]["perf/step_time"] == 38
    assert rows[1]["perf/wait_time_ratio"] == pytest.approx(23 / 38)
    assert rows[0]["perf/warmup"] and not rows[1]["perf/warmup"]
    assert rows[1]["perf/train_end_wall"] - rows[1]["perf/train_start_wall"] == 15


def test_full_checkpoints_follow_the_miles_save_cadence(tmp_path, monkeypatch):
    """Every save_interval updates and after the last one, as Miles saves."""
    import espnet2.rl.trainer as module
    from miles.utils.misc import should_run_periodic_action

    class Batch:
        def save_to_disk(self, path):
            pass

    class Rollout:
        def sync_weights(self, policy, step):
            pass

        def generate(self, sources, step):
            return []

    class Trainer:
        config = ESPnetRLConfig(
            steps=7, save_interval=3, prompts_per_step=1, output_dir=str(tmp_path)
        )
        rank, world_size = 0, 1
        device = torch.device("cpu")
        policy, optimizer = None, None
        rollout = Rollout()

        def prepare_batch(self, batch, native_scores):
            return {}

        def update_actor(self, batch):
            return {}

    saved = []
    monkeypatch.setattr(module, "collate_rollouts", lambda *args: Batch())
    monkeypatch.setattr(module, "score_native_batch", lambda *args: (None,) * 3)
    monkeypatch.setattr(
        module,
        "save_checkpoint",
        lambda path, policy, opt, step, config: saved.append(step),
    )
    NativeGRPOTrainer.train(Trainer(), [{"id": "p"}])
    assert saved == [3, 6, 7]
    assert saved == [
        step + 1 for step in range(7) if should_run_periodic_action(step, 3, None, 7)
    ]
    with pytest.raises(ValueError, match="save_interval"):
        ESPnetRLConfig(
            adapter_factory="a:b",
            exporter="a:b",
            sglang_engine="a:b",
            save_interval=0,
        ).validate()
