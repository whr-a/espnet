"""CPU lifecycle oracles for a reference that scores on GPU and idles on CPU.

Only the test module's device transfers are simulated. The trainer's scoring,
detached results, GRPO/KL backward, optimizer and frozen weights are exercised.
"""

import copy
from contextlib import contextmanager

import pytest
import torch

from espnet2.rl import trainer as trainer_module
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.scoring import score_native_batch
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


class Policy(torch.nn.Module):
    def __init__(self, value=-0.2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.register_buffer("fixed", torch.tensor([1.0, 2.0]))

    def forward(self, batch):
        mask = batch.batch["response_mask"]
        return TokenScores(self.weight * mask, torch.zeros_like(mask), mask)


class Reference(Policy):
    def __init__(self):
        super().__init__(-0.3)
        self.location, self.moves, self.calls = "cpu", [], []
        self.failure = None

    def to(self, device):
        # Simulate only transport; every real tensor stays on CPU in this oracle.
        self.location = str(device)
        self.moves.append(self.location)
        if self.failure == "transfer" and self.location != "cpu":
            raise RuntimeError("reference transfer failed")
        return super().to("cpu")

    def forward(self, batch):
        assert self.location == "cuda:7", "Reference must not score while offloaded"
        assert not torch.is_grad_enabled()
        self.calls.append(self.location)
        if self.failure == "forward":
            raise RuntimeError("reference forward failed")
        return super().forward(batch)


def configuration(**changes):
    return ESPnetRLConfig(
        device="cpu",
        reference_device="cuda:7",
        optimizer_device="cpu",
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        learning_rate=1e-3,
        **changes,
    )


def batch():
    sources = [{"id": "a", "reference": "hello"}, {"id": "b", "reference": "hello"}]
    generations = [
        {
            "source_id": source["id"],
            "prompt_token_ids": [1],
            "token_ids": [2, 0],
            "log_probs": [-0.2, 0.0],
            "action_mask": [1, 0],
            "text": "hello",
        }
        for source in sources
        for _ in range(4)
    ]
    return collate_rollouts(sources, generations, 4)


def test_reference_scores_on_configured_device_and_offloads_before_backward():
    reference = Reference()
    initial = copy.deepcopy(reference.state_dict())
    actor = Policy()
    trainer = NativeGRPOTrainer(
        actor, configuration(reference_cpu_offload=True), reference=reference
    )
    assert reference.location == "cpu"

    def backward(gradient):
        assert reference.location == "cpu"
        return gradient

    actor.weight.register_hook(backward)
    first = batch()
    trainer.prepare_batch(first)
    expected = first.batch["response_mask"] * initial["weight"]
    torch.testing.assert_close(first.batch["ref_log_probs"], expected, rtol=0, atol=0)
    assert first.batch["ref_log_probs"].device.type == "cpu"
    assert not first.batch["ref_log_probs"].requires_grad
    trainer.update_actor(first)
    assert actor.weight.item() != pytest.approx(-0.2, abs=1e-5)
    second = batch()
    trainer.prepare_batch(second)
    torch.testing.assert_close(
        second.batch["ref_log_probs"], first.batch["ref_log_probs"], rtol=0, atol=0
    )
    assert reference.location == "cpu"
    assert reference.moves == ["cpu", "cuda:7", "cpu", "cuda:7", "cpu"]
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())


@pytest.mark.parametrize("failure", ["forward", "transfer"])
def test_reference_returns_to_cpu_after_scoring_or_transfer_failure(failure):
    reference = Reference()
    trainer = NativeGRPOTrainer(
        Policy(), configuration(reference_cpu_offload=True), reference=reference
    )
    reference.failure = failure
    with torch.no_grad(), pytest.raises(RuntimeError, match=f"reference {failure} failed"):
        trainer._score_chunks(reference, batch())
    assert reference.location == "cpu"
    assert reference.moves[-2:] == ["cuda:7", "cpu"]
    reference.failure = None
    with torch.no_grad():
        result = trainer._score_chunks(reference, batch())
    assert not result.requires_grad and result.device.type == "cpu"
    assert reference.location == "cpu"


def test_default_reference_stays_on_its_configured_device():
    config, reference = configuration(), Reference()
    assert config.reference_cpu_offload is False
    trainer = NativeGRPOTrainer(Policy(), config, reference=reference)
    assert reference.location == "cuda:7"
    moves = list(reference.moves)
    trainer.prepare_batch(batch())
    assert reference.location == "cuda:7"
    assert reference.moves == moves


def test_reference_clone_starts_on_cpu_and_behavior_snapshot_is_restored(monkeypatch):
    reference, cloned_on = Reference(), []

    def clone(policy, device):
        cloned_on.append(device)
        return reference

    monkeypatch.setattr(trainer_module, "clone_reference", clone)
    actor = Policy()
    trainer = NativeGRPOTrainer(actor, configuration(reference_cpu_offload=True))
    assert cloned_on == ["cpu"]

    @contextmanager
    def behavior():
        current = actor.weight.detach().clone()
        assert reference.location == "cpu"
        with torch.no_grad():
            actor.weight.fill_(-0.25)
        try:
            yield actor
        finally:
            with torch.no_grad():
                actor.weight.copy_(current)

    current, stale, frozen = score_native_batch(trainer, batch(), behavior())
    mask = batch().batch["response_mask"]
    for scores, value in ((current, -0.2), (stale, -0.25), (frozen, -0.3)):
        torch.testing.assert_close(scores, mask * value, rtol=0, atol=0)
    assert actor.weight.item() == pytest.approx(-0.2)
    assert reference.location == "cpu"


def test_reference_offload_rejects_cpu_scoring_and_nonboolean_configuration():
    config = configuration(reference_cpu_offload=True)
    config.reference_device = "cpu"
    with pytest.raises(ValueError, match="GPU scoring device"):
        NativeGRPOTrainer(Policy(), config)
    config.reference_cpu_offload = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        config.validate()
