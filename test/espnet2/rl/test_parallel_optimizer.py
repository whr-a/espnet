"""CPU multi-process oracles for native DDP with one CPU Adam owner.

Run with ``python test/espnet2/rl/test_parallel_optimizer.py`` when the optional
runtime intentionally omits pytest. Tests use Gloo and never initialize CUDA.
"""

import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn import functional as F

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.parallel_optimizer import CPUOffloadedAdamW
from espnet2.rl.rollout import save_checkpoint
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


class TinyNativePolicy(nn.Module):
    def __init__(self, rank_offset=0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.10, -0.04, 0.02]) + rank_offset)
        self.bias = nn.Parameter(torch.tensor([0.04, 0.01, -0.03]) + rank_offset)
        self.optional = nn.Parameter(torch.tensor([0.02, -0.02, 0.01]) + rank_offset)
        self.never_used = nn.Parameter(torch.tensor([0.05]) + rank_offset)
        self.frozen = nn.Parameter(
            torch.tensor([0.25]) + rank_offset, requires_grad=False
        )
        self.register_buffer("fixed_buffer", torch.tensor([0.5]) + rank_offset)
        self.backward_calls = 0
        self.weight.register_hook(self._count_backward)

    def _count_backward(self, gradient):
        self.backward_calls += 1
        return gradient

    def checkpoint_state_dict(self):
        return self.state_dict()

    def forward(self, batch):
        rows = []
        for source in batch.non_tensor_batch["source"]:
            logits = self.weight * source["coefficient"] + self.bias
            # Only rank 0's first two groups use this branch. Its final
            # synchronized microbatch must include accumulated earlier usage.
            if source["special"]:
                logits = logits + self.optional * source["coefficient"]
            rows.append(logits)
        logits = torch.stack(rows).unsqueeze(1).expand(-1, 3, -1)
        logp = F.log_softmax(logits, dim=-1)
        selected = logp.gather(-1, batch.batch["responses"].unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        mask = batch.batch["response_mask"]
        return TokenScores(selected * mask, entropy * mask, mask)


def _config(output, micro_batch_size=4, strategy="ddp"):
    return ESPnetRLConfig(
        optimizer_device="cpu",
        reference_device="cpu",
        device="cpu",
        distributed_strategy=strategy,
        allreduce_chunk_bytes=16,
        output_dir=str(output),
        prompts_per_step=32,
        group_size=8,
        micro_batch_size=micro_batch_size,
        learning_rate=0.002,
        weight_decay=0.01,
        kl_coef=0.04,
        max_grad_norm=0.7,
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )


def _batch():
    sources = [
        {"id": f"prompt-{i:02d}", "coefficient": 0.1 + i / 20, "special": i < 2}
        for i in range(32)
    ]
    generations = []
    for i, source in enumerate(sources):
        for candidate in range(8):
            mask = [1, int((i + candidate) % 3 != 0), 1]
            generations.append(
                {
                    "source_id": source["id"],
                    "prompt_token_ids": [0],
                    "token_ids": [(i + candidate) % 3, candidate % 3, i % 3],
                    "log_probs": [0.0, 0.0, 0.0],
                    "action_mask": mask,
                }
            )
    batch = collate_rollouts(sources, generations, 8)
    mask = batch.batch["response_mask"]
    batch.batch["advantages"] = (
        torch.tensor([[(i % 8 - 3.5) / 3.5] * 3 for i in range(256)]) * mask
    )
    with torch.no_grad():
        old = TinyNativePolicy()(batch).log_probs
    batch.batch["old_log_probs"] = old
    batch.batch["ref_log_probs"] = old.clone()
    return batch


def _eight_rank_worker(rank, world_size, root, strategy):
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        policy = TinyNativePolicy(rank_offset=rank)
        trainer = NativeGRPOTrainer(policy, _config(root, strategy=strategy))
        assert isinstance(trainer.optimizer, CPUOffloadedAdamW)
        parameter_ids = [id(parameter) for parameter in policy.parameters()]
        storage_ids = [parameter.data_ptr() for parameter in policy.parameters()]
        reference = copy.deepcopy(trainer.reference.state_dict())
        initial = copy.deepcopy(policy.state_dict())
        for name, tensor in reference.items():
            torch.testing.assert_close(tensor, initial[name], rtol=0, atol=0)
        batch = _batch()
        metrics = []
        for _ in range(2):
            metrics.append(trainer.update_actor(batch))
            assert parameter_ids == [id(p) for p in policy.parameters()]
            assert storage_ids == [p.data_ptr() for p in policy.parameters()]
            for name, tensor in trainer.reference.state_dict().items():
                torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)
            for parameter in policy.parameters():
                gathered = [torch.empty_like(parameter) for _ in range(world_size)]
                distributed.all_gather(gathered, parameter.detach())
                for other in gathered:
                    torch.testing.assert_close(
                        parameter.detach(), other, rtol=0, atol=0
                    )
        assert policy.backward_calls == 16  # 32 local candidates / microbatch4 * 2.
        torch.testing.assert_close(policy.frozen, initial["frozen"], rtol=0, atol=0)
        torch.testing.assert_close(
            policy.fixed_buffer, initial["fixed_buffer"], rtol=0, atol=0
        )
        if rank == 0:
            state = trainer.optimizer.state_dict()
            assert (
                len(state["state"]) == 3
            )  # weight, bias, optional; never_used absent.
            for entry in state["state"].values():
                assert entry["step"].item() == 2
                for tensor in entry.values():
                    assert tensor.device.type == "cpu"
                    assert torch.isfinite(tensor).all()
            save_checkpoint(
                Path(root) / "parallel.pt",
                policy,
                trainer.optimizer,
                2,
                _config(root, strategy=strategy),
            )
            (Path(root) / "parallel_metrics.json").write_text(json.dumps(metrics))
        else:
            assert not trainer.optimizer.state
            assert not trainer.optimizer._shadow
            try:
                trainer.optimizer.state_dict()
            except RuntimeError as error:
                assert "complete state_dict" in str(error)
            else:
                raise AssertionError("Non-owner must not save incomplete Adam state")
        (Path(root) / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "backward_calls": policy.backward_calls,
                    "local_candidates": 32,
                }
            )
        )
    finally:
        distributed.destroy_process_group()


def _owner_error_worker(rank, world_size, root):
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous-error",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        parameter = nn.Parameter(torch.tensor([0.5]))
        optimizer = CPUOffloadedAdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([float("nan") if rank == 0 else 1.0])
        try:
            optimizer.step()
        except RuntimeError as error:
            assert "Nonfinite CPU Adam gradient" in str(error)
        else:
            raise AssertionError("Owner failure must reach every optimizer rank")
        torch.testing.assert_close(parameter, torch.tensor([0.5]), rtol=0, atol=0)
        if rank == 0:
            assert not optimizer.state_dict()["state"]
    finally:
        distributed.destroy_process_group()


class TestCPUOffloadDDP(unittest.TestCase):
    def test_owner_failure_is_reported_to_every_rank_without_stepping(self):
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.start_processes(
                _owner_error_worker,
                args=(2, directory),
                nprocs=2,
                start_method="spawn",
                join=True,
            )

    def test_eight_ranks_match_serial_global_masked_grpo_and_checkpoint(self):
        self._verify_strategy("ddp")

    def test_eight_native_allreduce_ranks_match_global_grpo_without_ddp_buckets(self):
        self._verify_strategy("native_allreduce")

    def _verify_strategy(self, strategy):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.start_processes(
                _eight_rank_worker,
                args=(8, directory, strategy),
                nprocs=8,
                start_method="spawn",
                join=True,
            )
            serial = NativeGRPOTrainer(TinyNativePolicy(), _config(directory))
            serial_batch = _batch()
            serial_metrics = [serial.update_actor(serial_batch) for _ in range(2)]
            checkpoint = torch.load(Path(directory) / "parallel.pt", weights_only=False)
            native = torch.load(
                Path(directory) / "parallel.native.pth", weights_only=True
            )
            for name, tensor in serial.policy.state_dict().items():
                torch.testing.assert_close(
                    checkpoint["model"][name], tensor, rtol=2e-6, atol=1e-7
                )
                torch.testing.assert_close(
                    native[name], checkpoint["model"][name], rtol=0, atol=0
                )
            serial_state = serial.optimizer.state_dict()
            for key, entry in serial_state["state"].items():
                for name, tensor in entry.items():
                    torch.testing.assert_close(
                        checkpoint["optimizer"]["state"][key][name],
                        tensor,
                        rtol=2e-5,
                        atol=1e-7,
                    )
            parallel_metrics = json.loads(
                (Path(directory) / "parallel_metrics.json").read_text()
            )
            for expected, actual in zip(serial_metrics, parallel_metrics, strict=True):
                for name in ("actor/loss", "actor/kl_loss", "actor/grad_norm"):
                    self.assertAlmostEqual(expected[name], actual[name], delta=1e-6)
            self.assertEqual(len(list(Path(directory).glob("rank-*.json"))), 8)

    def test_stable_storage_standard_adam_state_and_restore(self):
        torch.set_num_threads(1)
        original = nn.Parameter(torch.tensor([0.2, -0.3]))
        oracle = nn.Parameter(original.detach().clone())
        offloaded = CPUOffloadedAdamW([original], lr=0.03, weight_decay=0.05)
        standard = torch.optim.AdamW(
            [oracle], lr=0.03, weight_decay=0.05, foreach=False
        )
        identity, storage = id(original), original.data_ptr()
        for gradient in ([0.1, -0.2], [-0.2, 0.15]):
            original.grad = torch.tensor(gradient)
            oracle.grad = torch.tensor(gradient)
            offloaded.step()
            standard.step()
            self.assertEqual((id(original), original.data_ptr()), (identity, storage))
            torch.testing.assert_close(original, oracle, rtol=0, atol=0)
        restored = CPUOffloadedAdamW([original], lr=0.9)
        restored.load_state_dict(offloaded.state_dict())
        original.grad = torch.tensor([0.15, 0.4])
        oracle.grad = original.grad.clone()
        restored.step()
        standard.step()
        torch.testing.assert_close(original, oracle, rtol=0, atol=0)
        self.assertEqual(restored.param_groups[0]["lr"], 0.03)

    def test_bfloat16_actor_accumulates_updates_in_fp32_master(self):
        torch.set_num_threads(1)
        native = nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.bfloat16))
        master = nn.Parameter(native.detach().float())
        offloaded = CPUOffloadedAdamW([native], lr=1e-4)
        reference = torch.optim.AdamW(
            [master], lr=1e-4, weight_decay=0.0, foreach=False
        )
        for _ in range(50):
            native.grad = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
            master.grad = native.grad.float()
            offloaded.step()
            reference.step()
        # Each step moves the weights by about 1e-4, below bfloat16's 2**-7
        # spacing at 1.0, yet 50 of them add up to a visible change.
        self.assertEqual(native.dtype, torch.bfloat16)
        torch.testing.assert_close(offloaded._shadow[0], master, rtol=0, atol=0)
        torch.testing.assert_close(native, master.to(torch.bfloat16), rtol=0, atol=0)
        self.assertNotEqual(native[0].item(), 1.0)
        self.assertEqual(
            offloaded.state_dict()["state"][0]["exp_avg"].dtype, torch.float32
        )

    def test_low_precision_actor_requires_the_cpu_optimizer(self):
        config = ESPnetRLConfig(
            adapter_factory="a:b",
            exporter="c:d",
            sglang_engine="e:f",
            dtype="bfloat16",
        )
        with self.assertRaisesRegex(ValueError, "optimizer_device: cpu"):
            config.validate()
        config.optimizer_device = "cpu"
        config.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
