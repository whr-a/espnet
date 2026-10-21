"""Independent WEST loss/gradient and restart oracles, entirely on CPU."""

import copy
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn import functional as F

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.parallel_optimizer import CPUOffloadedAdamW
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.training_objectives import (
    build_lr_scheduler,
    select_training_sources,
    west_group_advantages,
    west_policy_loss,
)
from espnet2.rl.types import TokenScores


def config(root, micro_batch_size=1, strategy="native_allreduce", dtype="float32"):
    return ESPnetRLConfig(
        family="bagpiper",
        training_objective="west_grpo",
        loss_agg_mode="seq-mean-token-mean",
        lr_scheduler="linear",
        lr_warmup_steps=0,
        steps=1000,
        optimizer_device="cpu",
        reference_device="cpu",
        device="cpu",
        dtype=dtype,
        distributed_strategy=strategy,
        allreduce_chunk_bytes=16,
        prompts_per_step=4,
        group_size=2,
        micro_batch_size=micro_batch_size,
        learning_rate=1e-3,
        max_grad_norm=0.7,
        weight_decay=0.01,
        kl_coef=0.04,
        shuffle_train_data=True,
        output_dir=str(root),
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )


class TinyPolicy(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.1, -0.2, 0.04], dtype=dtype))
        self.frozen = nn.Parameter(
            torch.tensor([0.7], dtype=dtype), requires_grad=False
        )

    def checkpoint_state_dict(self):
        return self.state_dict()

    def forward(self, batch, *, score_raw=False):
        coefficients = torch.tensor(
            [row["coefficient"] for row in batch.non_tensor_batch["source"]]
        )
        logits = coefficients[:, None] * self.weight.float()
        if not score_raw:
            logits = logits / 0.7
        logp = F.log_softmax(logits, -1)[:, None].expand(
            -1, batch.batch["responses"].shape[1], -1
        )
        selected = logp.gather(-1, batch.batch["responses"].unsqueeze(-1)).squeeze(-1)
        mask = batch.batch["response_mask"]
        return TokenScores(selected * mask, -(logp.exp() * logp).sum(-1) * mask, mask)


def make_batch():
    sources = [{"id": f"p{i}", "coefficient": 0.3 + i / 4} for i in range(4)]
    generations = []
    for i, source in enumerate(sources):
        for candidate in range(2):
            length = (i + candidate) % 4 + 1
            generations.append(
                dict(
                    source_id=source["id"],
                    prompt_token_ids=[1],
                    token_ids=[(i + candidate + j) % 3 for j in range(length)],
                    log_probs=[-9.0] * length,
                )
            )
    batch = collate_rollouts(sources, generations, 2)
    mask = batch.batch["response_mask"]
    outcomes = torch.tensor([1.1, 0.1, 0.1, 0.0, 1.1, 1.1, 0.0, 1.1])
    grouped = outcomes.view(-1, 2)
    scalar_advantage = (
        (grouped - grouped.mean(1, keepdim=True))
        / (grouped.std(1, keepdim=True) + 1e-4)
    ).flatten()
    batch.batch["advantages"] = scalar_advantage[:, None] * mask
    with torch.no_grad():
        batch.batch["old_log_probs"] = TinyPolicy()(batch).log_probs
        reference = TinyPolicy()
        reference.weight.add_(torch.tensor([0.04, -0.02, 0.01]))
        batch.batch["ref_log_probs"] = reference(batch, score_raw=True).log_probs
    return batch


def oracle_update(policy, batch, optimizer, step, cfg):
    """Literal WEST formula, without importing ESPnet objective/reduction helpers."""
    optimizer.zero_grad(set_to_none=True)
    current = policy(batch, score_raw=True).log_probs
    reference = batch.batch["ref_log_probs"]
    advantages = batch.batch["advantages"]
    mask = batch.batch["response_mask"]
    delta = reference - current
    per_token_kl = torch.exp(delta) - delta - 1
    values = -(
        torch.exp(current - current.detach()) * advantages - cfg.kl_coef * per_token_kl
    )
    loss = ((values * mask).sum(1) / mask.sum(1)).mean()
    kl = ((per_token_kl * mask).sum(1) / mask.sum(1)).mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    for group in optimizer.param_groups:
        group["lr"] = cfg.learning_rate * (1 - step / cfg.steps)
    optimizer.step()
    return loss.item(), kl.item(), norm.item()


def test_west_advantages_match_grouped_formula_including_constant_decimal_groups():
    mask = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.float32
    )
    outcomes = torch.tensor([0.1, 1.1, 0.1, 0.1])
    rewards = torch.zeros_like(mask)
    rewards[:, 0] = outcomes
    actual, returns = west_group_advantages(rewards, mask, ["a", "a", "b", "b"])
    grouped = outcomes.view(-1, 2)
    expected = (
        (grouped - grouped.mean(1, keepdim=True))
        / (grouped.std(1, keepdim=True) + 1e-4)
    ).flatten()[:, None] * mask
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(returns, expected, rtol=0, atol=0)


def test_answer_reduction_weights_short_and_long_answers_equally():
    current = torch.tensor([[-0.3, 0, 0], [-0.7, -0.5, -0.4]], requires_grad=True)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float32)
    advantages = torch.tensor([[1, 0, 0], [-1, -1, -1]], dtype=torch.float32)
    loss = west_policy_loss(current, advantages, mask, global_batch_size=2, dp_size=1)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(
        current.grad, torch.tensor([[-0.5, 0, 0], [1 / 6, 1 / 6, 1 / 6]])
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("parts", [[list(range(4))], [[0], [1, 2], [3]]])
def test_west_miles_primitive_preserves_exact_values_and_gradients(dtype, parts):
    current = torch.tensor(
        [
            [-0.1, -20.0, -0.4],
            [-100.0, -0.2, -0.7],
            [-1.3, -2.4, -3.1],
            [-2.0, -0.6, -0.9],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    original = current.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=dtype)
    # Include both advantage signs, zero, and a constant-group rounding residue.
    advantages = torch.tensor(
        [[0.53994524] * 3, [-0.123456789] * 3, [0.0] * 3, [1e-7] * 3],
        dtype=dtype,
    )
    values = -(torch.exp(original - original.detach()) * advantages)
    expected = sum(
        ((values[indices] * mask[indices]).sum(-1) / mask[indices].sum(-1)).sum() / 4
        for indices in parts
    )
    expected.backward()
    actual = sum(
        west_policy_loss(
            current[indices],
            advantages[indices],
            mask[indices],
            global_batch_size=4,
            dp_size=1,
        )
        for indices in parts
    )
    actual.backward()
    # FP32 is the production scoring dtype; non-FP32 callers retain the prior
    # path because Miles itself rounds its log-ratio and upstream gradient.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(current.grad, original.grad, rtol=0, atol=0)


@pytest.mark.parametrize("micro_batch_size", [1, 3, 8])
def test_real_trainer_matches_literal_west_formula_and_raw_gradient(
    tmp_path, micro_batch_size
):
    torch.set_num_threads(1)
    cfg = config(tmp_path, micro_batch_size)
    trainer = NativeGRPOTrainer(TinyPolicy(), cfg)
    oracle = TinyPolicy()
    optimizer = torch.optim.AdamW(
        oracle.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        foreach=False,
    )
    batch = make_batch()
    for step in range(3):
        expected = oracle_update(oracle, batch, optimizer, step, cfg)
        actual = trainer.update_actor(copy.deepcopy(batch))
        for name, value in trainer.policy.state_dict().items():
            torch.testing.assert_close(
                value, oracle.state_dict()[name], rtol=1e-6, atol=1e-7
            )
        assert actual["actor/loss"] == pytest.approx(expected[0], abs=2e-7)
        assert actual["actor/kl_loss"] == pytest.approx(expected[1], abs=2e-7)
        assert actual["actor/grad_norm"] == pytest.approx(expected[2], abs=2e-7)
        assert actual["actor/learning_rate"] == pytest.approx(
            cfg.learning_rate * (1 - step / 1000)
        )
        assert actual["actor/next_learning_rate"] == pytest.approx(
            cfg.learning_rate * (1 - (step + 1) / 1000)
        )


def test_bfloat16_checkpoint_restores_master_weights_lr_rng_and_cursor(tmp_path):
    torch.set_num_threads(1)
    cfg = config(tmp_path, dtype="bfloat16")
    original = NativeGRPOTrainer(TinyPolicy(torch.bfloat16), cfg)
    original.source_count = 8
    batch = make_batch()
    for _ in range(3):
        original.update_actor(copy.deepcopy(batch))
    path = tmp_path / "step-000003.pt"
    original.save_training_checkpoint(path, 3)
    expected_random = torch.rand(5)
    new_cfg = copy.deepcopy(cfg)
    new_cfg.output_dir = str(tmp_path / "resumed")
    new_cfg.resume_checkpoint = str(path)
    restored = NativeGRPOTrainer(TinyPolicy(torch.bfloat16), new_cfg)
    restored.source_count = 8
    frozen_reference = copy.deepcopy(restored.reference.state_dict())
    state = restored.load_training_checkpoint(path)
    torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
    assert restored.completed_updates == state["next_update"] == 3
    assert state["consumed_prompts"] == 12
    assert restored.scheduler.last_epoch == 3
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(
        cfg.learning_rate * 0.997
    )
    for first, second in zip(
        original.optimizer._shadow, restored.optimizer._shadow, strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for _ in range(2):
        original.update_actor(copy.deepcopy(batch))
        restored.update_actor(copy.deepcopy(batch))
    for first, second in zip(
        original.optimizer._shadow, restored.optimizer._shadow, strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, value in frozen_reference.items():
        torch.testing.assert_close(
            restored.reference.state_dict()[name], value, rtol=0, atol=0
        )
    invalid = copy.deepcopy(new_cfg)
    invalid.steps = 1001
    with pytest.raises(ValueError, match="frozen configuration: steps"):
        NativeGRPOTrainer(TinyPolicy(torch.bfloat16), invalid).load_training_checkpoint(
            path
        )


def test_cpu_master_weights_retain_updates_below_bfloat16_resolution():
    parameter = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    original = CPUOffloadedAdamW([parameter], lr=1e-4)
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        original.step()
    assert parameter.item() == 1
    assert original._shadow[0].item() < 0.9999
    state = copy.deepcopy(original.state_dict())
    restored = CPUOffloadedAdamW([nn.Parameter(parameter.detach().clone())], lr=0.9)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored._shadow[0], original._shadow[0], rtol=0, atol=0)
    del state["espnet_master_weights"]
    with pytest.raises(ValueError, match="FP32 masters"):
        restored.load_state_dict(state)


def test_linear_schedule_uses_the_entire_1000_update_horizon_without_lr_warmup(
    tmp_path,
):
    cfg = config(tmp_path)
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6)
    scheduler = build_lr_scheduler(optimizer, cfg)
    learning_rates = []
    for _ in range(1000):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert learning_rates[0] == 1e-6
    assert learning_rates[99] == pytest.approx(9.01e-7)
    assert learning_rates[999] == pytest.approx(1e-9)
    assert optimizer.param_groups[0]["lr"] == 0
    assert scheduler.last_epoch == 1000


def test_sampler_matches_torch_seedable_sampler_and_absolute_resume_cursor(tmp_path):
    cfg = config(tmp_path)
    sources = [{"id": str(i)} for i in range(8)]
    expected = []
    for epoch in range(3):
        generator = torch.Generator().manual_seed(42 + epoch)
        expected.extend(str(i) for i in torch.randperm(8, generator=generator).tolist())
    actual = [
        row["id"]
        for step in range(6)
        for row in select_training_sources(sources, step, cfg)
    ]
    assert actual == expected
    assert [row["id"] for row in select_training_sources(sources, 3, cfg)] == expected[
        12:16
    ]


def _distributed_worker(rank, root, strategy):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        trainer = NativeGRPOTrainer(
            TinyPolicy(), config(root, micro_batch_size=3, strategy=strategy)
        )
        trainer.source_count = 8
        metric = trainer.update_actor(make_batch())
        trainer.save_training_checkpoint(Path(root) / "parallel.pt", 1)
        if rank == 0:
            (Path(root) / "metrics.json").write_text(json.dumps(metric))
        restored_config = config(root, micro_batch_size=3, strategy=strategy)
        restored_config.output_dir = str(Path(root) / "resumed")
        restored_config.resume_checkpoint = str(Path(root) / "parallel.pt")
        restored = NativeGRPOTrainer(TinyPolicy(), restored_config)
        restored.source_count = 8
        restored.load_training_checkpoint(Path(root) / "parallel.pt")
        assert restored.completed_updates == 1
        assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(
            restored_config.learning_rate * 0.999
        )
        restored.update_actor(make_batch())
        restored.save_training_checkpoint(Path(root) / "resumed.pt", 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["ddp", "native_allreduce"])
def test_multirank_microbatch_weighting_matches_independent_west_oracle(
    tmp_path, strategy
):
    torch.set_num_threads(1)
    multiprocessing.start_processes(
        _distributed_worker,
        args=(str(tmp_path), strategy),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    checkpoint = torch.load(tmp_path / "parallel.pt", weights_only=False)
    metric = json.loads((tmp_path / "metrics.json").read_text())
    cfg = config(tmp_path)
    oracle = TinyPolicy()
    optimizer = torch.optim.AdamW(
        oracle.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        foreach=False,
    )
    expected = oracle_update(oracle, make_batch(), optimizer, 0, cfg)
    for name, value in checkpoint["model"].items():
        torch.testing.assert_close(
            value, oracle.state_dict()[name], rtol=1e-6, atol=1e-7
        )
    for name, value in zip(
        ("actor/loss", "actor/kl_loss", "actor/grad_norm"), expected, strict=True
    ):
        assert metric[name] == pytest.approx(value, abs=2e-7)
    assert len(checkpoint["trainer_state"]["rank_rng_states"]) == 2
    assert checkpoint["scheduler"]["last_epoch"] == 1
    oracle_update(oracle, make_batch(), optimizer, 1, cfg)
    resumed = torch.load(tmp_path / "resumed.pt", weights_only=False)
    for name, value in resumed["model"].items():
        torch.testing.assert_close(
            value, oracle.state_dict()[name], rtol=1e-6, atol=1e-7
        )
    assert resumed["scheduler"]["last_epoch"] == 2
    assert resumed["optimizer"]["param_groups"][0]["lr"] == pytest.approx(
        cfg.learning_rate * 0.998
    )


def shared_step_reward(source, generation):
    return float(generation["token_ids"][0] % 2)


@pytest.mark.parametrize("objective", ["miles_ppo", "west_grpo"])
def test_shared_step_matches_explicit_scoring_reward_and_optimizer(tmp_path, objective):
    """Transport consolidation preserves real parameters, moments and fixed reference."""
    from espnet2.rl.scoring import score_native_batch

    cfg = config(tmp_path)
    cfg.training_objective = objective
    cfg.loss_agg_mode = (
        "token-mean" if objective == "miles_ppo" else "seq-mean-token-mean"
    )
    cfg.reward_function = f"{__name__}:shared_step_reward"
    combined = NativeGRPOTrainer(TinyPolicy(), cfg)
    explicit = NativeGRPOTrainer(TinyPolicy(), copy.deepcopy(cfg))
    batch = make_batch()
    batch.batch["rollout_log_probs"] = batch.batch["old_log_probs"].clone()
    manual = copy.deepcopy(batch)
    fixed_reference = copy.deepcopy(combined.reference.state_dict())
    old, _, reference = score_native_batch(explicit, manual)
    expected = explicit.prepare_batch(manual, native_scores=(old, reference))
    expected.update(explicit.update_actor(manual))
    actual = combined.train_batch(batch, 0)
    assert actual == expected
    for name, value in combined.policy.state_dict().items():
        torch.testing.assert_close(
            value, explicit.policy.state_dict()[name], rtol=0, atol=0
        )
    for name, value in combined.reference.state_dict().items():
        torch.testing.assert_close(value, fixed_reference[name], rtol=0, atol=0)
    observed = combined.optimizer.state_dict()
    wanted = explicit.optimizer.state_dict()
    assert observed["param_groups"] == wanted["param_groups"]
    for key, state in observed["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, wanted["state"][key][field], rtol=0, atol=0
            )
    assert combined.scheduler.state_dict() == explicit.scheduler.state_dict()
    assert combined.completed_updates == explicit.completed_updates == 1
    assert (tmp_path / "rollout-000000.pkl").is_file()
