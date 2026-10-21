"""Clock oracles for equal learner windows and save/export wait attribution."""

import json

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


class Clock:
    value = 0.0

    def monotonic(self):
        return self.value

    def time(self):
        return 1000 + self.value

    def advance(self, duration):
        self.value += duration


class Batch:
    def __init__(self, clock=None):
        self.clock = clock

    def save_to_disk(self, path):
        if self.clock:
            self.clock.advance(2)


class Rollout:
    def __init__(self, clock=None):
        self.clock = clock

    def sync_weights(self, policy, step):
        if self.clock:
            self.clock.advance(5)

    def generate(self, sources, step):
        if self.clock:
            self.clock.advance(10)
        return []


class ClockTrainer(NativeGRPOTrainer):
    """Real shared orchestration with controlled reward/optimizer phase durations."""

    def __init__(self, root, *, steps, save_interval=1, warmup_steps=0, clock=None):
        config = ESPnetRLConfig(
            steps=steps,
            save_interval=save_interval,
            warmup_steps=warmup_steps,
            prompts_per_step=1,
            output_dir=str(root),
            device="cpu",
            adapter_factory="unused:factory",
            exporter="unused:exporter",
            sglang_engine="unused:Engine",
        )
        super().__init__(torch.nn.Linear(1, 1), config, rollout=Rollout(clock))
        self.clock = clock

    def prepare_batch(self, batch, native_scores):
        if self.clock:
            self.clock.advance(4)
        return {}

    def update_actor(self, batch):
        if self.clock:
            self.clock.advance(6)
        self.optimizer.step()
        self.scheduler.step()
        self.completed_updates += 1
        return {}


def test_sync_clock_includes_score_reward_update_and_next_wait_includes_save(
    tmp_path, monkeypatch
):
    import espnet2.rl.trainer as module

    clock = Clock()
    monkeypatch.setattr(module, "time", clock)

    def collate(*args):
        clock.advance(1)
        return Batch(clock)

    def score(*args):
        clock.advance(3)
        return None, None, None

    saved = []

    def save(path, policy, optimizer, step, config, *, scheduler, trainer_state):
        assert (
            trainer_state["completed_updates"]
            == scheduler.state_dict()["last_epoch"]
            == step
        )
        assert len(trainer_state["rank_rng_states"]) == 1
        saved.append(step)
        clock.advance(7)

    monkeypatch.setattr(module, "collate_rollouts", collate)
    monkeypatch.setattr(module, "score_native_batch", score)
    monkeypatch.setattr(module, "save_checkpoint", save)
    trainer = ClockTrainer(tmp_path, steps=2, warmup_steps=1, clock=clock)
    trainer.train([{"id": "p"}])
    assert saved == [1, 2]
    assert trainer.completed_updates == 2
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
    from miles.utils.misc import should_run_periodic_action

    import espnet2.rl.trainer as module

    saved = []

    def save(path, policy, optimizer, step, config, *, scheduler, trainer_state):
        assert scheduler.state_dict()["last_epoch"] == step
        assert (
            trainer_state["completed_updates"] == trainer_state["next_update"] == step
        )
        assert trainer_state["consumed_prompts"] == step
        assert trainer_state["source_count"] == 1
        assert len(trainer_state["rank_rng_states"]) == 1
        saved.append(step)

    monkeypatch.setattr(module, "collate_rollouts", lambda *args: Batch())
    monkeypatch.setattr(module, "score_native_batch", lambda *args: (None,) * 3)
    monkeypatch.setattr(module, "save_checkpoint", save)
    trainer = ClockTrainer(tmp_path, steps=7, save_interval=3)
    trainer.train([{"id": "p"}])
    assert saved == [3, 6, 7]
    assert (
        trainer.completed_updates == trainer.scheduler.state_dict()["last_epoch"] == 7
    )
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
