"""Profiling accounts for nested work without changing a real CPU RL update."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.timing import PhaseProfiler, profile_phase
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


def test_phase_accounting_memory_samples_and_disabled_cuda(monkeypatch):
    import espnet2.rl.timing as module

    clock, calls = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: calls.append("reset")
    )
    for name, value in {
        "memory_allocated": 120, "memory_reserved": 256,
        "max_memory_allocated": 192, "max_memory_reserved": 512,
    }.items():
        monkeypatch.setattr(torch.cuda, name, lambda device, value=value: value)
    owner = SimpleNamespace(profiler=PhaseProfiler("cuda:0", enabled=False))
    owner.profiler.reset()
    with profile_phase(owner, "reference_score"):
        pass
    assert owner.profiler.result() == {} and calls == []

    owner.profiler.enabled = True
    owner.profiler.reset()
    with profile_phase(owner, "reference_score"):
        with profile_phase(owner, "reference_h2d", memory=False):
            clock[0] += 2
        clock[0] += 3
        with profile_phase(owner, "reference_d2h", memory=False):
            clock[0] += 1
    result = owner.profiler.result()
    assert result["perf/reference_score_time"] == 6
    assert result["perf/reference_h2d_time"] == 2
    assert result["perf/reference_d2h_time"] == 1
    assert result["memory/reference_score_allocated_bytes"] == 120
    assert result["memory/reference_score_reserved_bytes"] == 256
    assert result["memory/peak_allocated_bytes"] == 192
    assert result["memory/peak_reserved_bytes"] == 512
    assert calls.count("reset") == 1  # Nested phases must not erase update peaks.
    assert calls.count("sync") == 7
    owner.profiler.reset()
    assert owner.profiler.metrics == {}


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.fixed = torch.nn.Parameter(torch.tensor([7.0, 8.0]), requires_grad=False)

    def forward(self, batch):
        mask = batch.batch["response_mask"]
        logits = self.weight.expand(*mask.shape, -1)
        logp = logits.log_softmax(-1)
        selected = logp.gather(-1, batch.batch["responses"].unsqueeze(-1)).squeeze(-1)
        return TokenScores(selected * mask, -(logp.exp() * logp).sum(-1), mask)


def test_profiled_real_updates_preserve_optimizer_rng_and_byte_budget(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU profiling must not initialize or synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    trainers, outputs, rng = [], [], []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        directory.mkdir()
        config = ESPnetRLConfig(
            adapter_factory="unused:factory", exporter="unused:exporter",
            sglang_engine="unused:Engine", output_dir=str(directory),
            optimizer_device="cpu", reference_device="cpu", device="cpu",
            prompts_per_step=1, group_size=2, micro_batch_size=1,
            learning_rate=0.01, score_tolerance=5, profile_phases=enabled,
        )
        torch.manual_seed(42)
        trainer = NativeGRPOTrainer(Policy(), config)
        source = {"id": "p", "reference": "yes"}
        for step in range(2):
            batch = collate_rollouts([source], [
                {"source_id": "p", "prompt_token_ids": [0], "token_ids": [token],
                 "log_probs": [-1.0], "action_mask": [1], "text": answer}
                for token, answer in [(0, "yes"), (1, "no")]
            ], 2)
            metrics = trainer.train_batch(batch, step)
        trainers.append(trainer)
        outputs.append(metrics)
        rng.append(torch.get_rng_state())
    assert not any(key.startswith("perf/") for key in outputs[0])
    for name in ("current_score", "reference_score", "backward", "optimizer",
                 "optimizer_d2h", "optimizer_cpu", "optimizer_h2d"):
        assert outputs[1][f"perf/{name}_time"] >= 0
    # Current behavior reuses scores, rather than reporting duplicate compute.
    assert "perf/behavior_score_time" not in outputs[1]
    assert not any(key.startswith("memory/") for key in outputs[1])
    assert outputs[1]["perf/profile_rank_max"] is True
    assert torch.equal(rng[0], rng[1])
    for left, right in zip(
        trainers[0].policy.parameters(), trainers[1].policy.parameters()
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        trainers[0].optimizer.state.values(), trainers[1].optimizer.state.values()
    ):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(left[key], right[key])
    assert trainers[0].scheduler.state_dict() == trainers[1].scheduler.state_dict()
    budget = json.loads((tmp_path / "True/memory-rank-0.json").read_text())
    assert budget["policy_trainable_bytes"] == 12
    assert budget["cpu_optimizer_master_bytes"] == 12
    assert budget["policy_frozen_bytes"] == 8
    assert budget["reference_parameter_bytes"] == 20
    assert not (tmp_path / "False/memory-rank-0.json").exists()


def _profile_worker(rank, root):
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60),
        init_method=f"file://{root}/gloo",
    )
    try:
        config = ESPnetRLConfig(
            adapter_factory="unused:factory", exporter="unused:exporter",
            sglang_engine="unused:Engine", output_dir=root,
            optimizer_device="cpu", reference_device="cpu", device="cpu",
            prompts_per_step=1, group_size=2, micro_batch_size=1,
            distributed_strategy="native_allreduce", allreduce_chunk_bytes=8,
            score_tolerance=5, profile_phases=True,
        )
        trainer = NativeGRPOTrainer(Policy(), config)
        batch = collate_rollouts([{"id": "p", "reference": "yes"}], [
            {"source_id": "p", "prompt_token_ids": [0], "token_ids": [token],
             "log_probs": [-1.0], "action_mask": [1], "text": answer}
            for token, answer in [(0, "yes"), (1, "no")]
        ], 2)
        metrics = trainer.train_batch(batch, 0)
        Path(root, f"profile-rank-{rank}.json").write_text(json.dumps(metrics))
    finally:
        distributed.destroy_process_group()


def test_two_rank_profile_includes_owner_work_and_reports_equal_global_metrics(
    tmp_path,
):
    multiprocessing.spawn(_profile_worker, args=(str(tmp_path),), nprocs=2, join=True)
    profiles = [json.loads((tmp_path / f"profile-rank-{rank}.json").read_text())
                for rank in range(2)]
    assert profiles[0] == profiles[1]
    assert profiles[0]["perf/optimizer_cpu_time"] > 0
    assert profiles[0]["perf/optimizer_time"] >= profiles[0]["perf/optimizer_cpu_time"]
    budgets = [json.loads((tmp_path / f"memory-rank-{rank}.json").read_text())
               for rank in range(2)]
    assert [item["cpu_optimizer_master_bytes"] for item in budgets] == [12, 0]
