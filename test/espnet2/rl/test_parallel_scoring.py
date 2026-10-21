"""Eight-rank CPU oracle and collective error propagation for native scoring."""

import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn

from espnet2.rl.data import collate_rollouts
from espnet2.rl.scoring import score_native_batch
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


class NativeScoringOracle(nn.Module):
    def __init__(self, offset=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(offset))
        self.calls = 0
        self.sample_indices = []
        self.failure = None

    def forward(self, batch):
        assert not torch.is_grad_enabled()
        self.calls += 1
        self.sample_indices.extend(
            generation["candidate_index"]
            for generation in batch.non_tensor_batch["generation"]
        )
        if self.failure == "exception":
            raise ValueError("deliberate native scorer exception")
        mask = batch.batch["response_mask"]
        indices = torch.tensor(self.sample_indices[-len(batch) :], dtype=torch.float32)
        value = (
            -indices[:, None] / 100 - batch.batch["responses"] / 10 + self.weight
        ) * mask
        if self.failure == "nonfinite":
            value[0, 0] = float("nan")
        native_mask = torch.ones_like(mask) if self.failure == "mask" else mask
        return TokenScores(value, torch.zeros_like(mask), native_mask)


class SupportOracle(NativeScoringOracle):
    """Counts supports like the Bagpiper policy: ``flips`` frames of candidate
    0 get another support from its own logits than their rollout recorded."""

    support_mismatch_rate = 2e-3

    def __init__(self, offset=0.0, flips=0):
        super().__init__(offset)
        self.flips, self.support_mismatches, self.support_frames = flips, 0, 0

    def forward(self, batch):
        scores = super().forward(batch)
        self.support_frames += int(batch.batch["response_lengths"].sum())
        self.support_mismatches += self.flips * self.sample_indices[
            -len(batch) :
        ].count(0)
        return scores


def _global_batch():
    sources = [
        {"id": f"prompt-{i:02d}", "feature": torch.tensor([i])} for i in range(32)
    ]
    generations = []
    for prompt, source in enumerate(sources):
        for candidate in range(8):
            generations.append(
                {
                    "source_id": source["id"],
                    "candidate_index": prompt * 8 + candidate,
                    "prompt_token_ids": [prompt],
                    "token_ids": [candidate, candidate + 1, candidate + 2],
                    "log_probs": [-0.1, 0.0, -0.2],
                    "action_mask": [1, 0, 1],
                }
            )
    return collate_rollouts(sources, generations, 8)


def _trainer(rank=0, world_size=1):
    trainer = NativeGRPOTrainer.__new__(NativeGRPOTrainer)
    trainer.rank = rank
    trainer.world_size = world_size
    trainer.policy = NativeScoringOracle(0.02)
    trainer.reference = NativeScoringOracle(-0.04).requires_grad_(False)
    trainer.config = SimpleNamespace(
        micro_batch_size=5, micro_batch_tokens=None, distributed_timeout_minutes=1.5
    )
    return trainer


def _eight_rank_scoring_worker(rank, root):
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        world_size=8,
        rank=rank,
        timeout=timedelta(seconds=90),
    )
    try:
        trainer = _trainer(rank, 8)
        behavior = NativeScoringOracle(-0.08).requires_grad_(False)
        batch = _global_batch()
        preserved = copy.deepcopy(batch.batch)
        old, generated, reference = score_native_batch(trainer, batch, behavior)
        indices = torch.arange(256, dtype=torch.float32)[:, None]
        mask = batch.batch["response_mask"]
        baseline = -indices / 100 - batch.batch["responses"] / 10
        for result, offset in ((old, 0.02), (generated, -0.08), (reference, -0.04)):
            torch.testing.assert_close(
                result, (baseline + offset) * mask, rtol=0, atol=0
            )
            assert result.device.type == "cpu" and not result.requires_grad
        expected_indices = list(range(rank * 32, (rank + 1) * 32))
        for policy in (trainer.policy, trainer.reference, behavior):
            assert policy.sample_indices == expected_indices
            assert policy.calls == 7  # six microbatch5 + final2.
            assert policy.weight.grad is None
        for key, tensor in batch.batch.items():
            torch.testing.assert_close(tensor, preserved[key], rtol=0, atol=0)
        before = trainer.policy.calls
        current_old, current_behavior, _ = score_native_batch(trainer, batch)
        assert trainer.policy.calls - before == 7
        torch.testing.assert_close(current_old, current_behavior, rtol=0, atol=0)

        for failure in (
            "exception",
            "nonfinite",
            "mask",
            "identity",
            "inputs",
            "behavior_mode",
        ):
            bad_batch = _global_batch()
            if rank == 3:
                if failure == "identity":
                    bad_batch.non_tensor_batch["generation"][0]["source_id"] = "wrong"
                elif failure == "inputs":
                    bad_batch.batch["responses"][0, 0] += 1
                else:
                    trainer.policy.failure = failure
            try:
                score_native_batch(
                    trainer,
                    bad_batch,
                    None if failure == "behavior_mode" and rank == 3 else behavior,
                )
            except RuntimeError as error:
                if failure == "inputs":
                    assert "input identities differ" in str(error)
                elif failure == "behavior_mode":
                    assert "behavior identity differs" in str(error)
                else:
                    assert "rank 3" in str(error)
            else:
                raise AssertionError(f"Missing all-rank propagation for {failure}")
            trainer.policy.failure = None
        (Path(root) / f"rank-{rank}.passed").write_text("passed\n")
    finally:
        distributed.destroy_process_group()


class TestParallelScoring(unittest.TestCase):
    def test_eight_rank_scores_masks_identity_and_error_propagation(self):
        with tempfile.TemporaryDirectory() as root:
            multiprocessing.start_processes(
                _eight_rank_scoring_worker,
                args=(root,),
                nprocs=8,
                start_method="spawn",
                join=True,
            )
            self.assertEqual(len(list(Path(root).glob("rank-*.passed"))), 8)

    def test_serial_reuses_current_behavior_and_reports_invalid_mask(self):
        torch.set_num_threads(1)
        trainer = _trainer()
        batch = _global_batch()
        old, behavior, reference = score_native_batch(trainer, batch)
        self.assertEqual(trainer.policy.calls, 52)
        torch.testing.assert_close(old, behavior, rtol=0, atol=0)
        self.assertFalse(torch.equal(old, reference))
        batch.batch["response_mask"][0, 0] = 0.5
        with self.assertRaisesRegex(RuntimeError, "finite binary"):
            score_native_batch(trainer, batch)

    def test_support_mismatches_are_gated_per_pass_but_not_for_the_reference(self):
        # One flipped frame of 768 passes the 2e-3 rate, although each forward
        # of five candidates (15 frames) alone would exceed it.
        trainer = _trainer()
        trainer.policy = SupportOracle(0.02, flips=1)
        trainer.reference = SupportOracle(-0.04, flips=9).requires_grad_(False)
        behavior = SupportOracle(-0.08, flips=1).requires_grad_(False)
        score_native_batch(trainer, _global_batch(), behavior)
        self.assertEqual(
            trainer.support_metrics,
            {
                "rollout/support_mismatch_rate": 1 / 768,
                "rollout/behavior_support_mismatch_rate": 1 / 768,
                "rollout/reference_support_mismatch_rate": 9 / 768,
            },
        )
        trainer.policy.flips = 2
        with self.assertRaisesRegex(RuntimeError, "current policy gives 2 of 768"):
            score_native_batch(trainer, _global_batch())


if __name__ == "__main__":
    unittest.main(verbosity=2)
