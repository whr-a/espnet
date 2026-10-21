"""CPU Gloo oracles for native views, bounded collectives and real Adam updates."""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn

from espnet2.rl.parallel import (
    _chunks,
    _validate_tensor,
    average_native_gradients,
    synchronize_native_model,
)
from espnet2.rl.parallel_optimizer import CPUOffloadedAdamW


class StridedPolicy(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.transposed = nn.Parameter(torch.arange(12.0).reshape(4, 3).t())
        self.contiguous = nn.Parameter(torch.arange(12.0).reshape(3, 4))
        self.rank_local = nn.Parameter(torch.arange(12.0).reshape(3, 4))
        self.unused = nn.Parameter(torch.ones(2))
        # Exact OWSM frontend mel matrix layout observed by strict CPU loading.
        self.register_buffer("melmat", torch.arange(257 * 128.0).reshape(128, 257).t())
        backing = torch.full((20, 20), -100.0)
        self.register_buffer("offset_slice", backing[2:8:2, 3:11:2])
        for tensor in list(self.parameters()) + list(self.buffers()):
            with torch.no_grad():
                tensor.add_(rank * 1000)


def identity(tensor):
    return (id(tensor), tensor.data_ptr(), tensor.storage_offset(), tensor.stride())


def gradient(step, rank, shape):
    # Build a noncontiguous gradient with a separately derived logical oracle.
    values = torch.arange(1.0, 1.0 + shape[0] * shape[1]).reshape(shape[1], shape[0])
    return values.t() * ((step + 1) * (rank + 1))


def strided_worker(rank, world_size, root):
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        model = StridedPolicy(rank)
        tensors = list(model.parameters()) + list(model.buffers())
        before = [identity(tensor) for tensor in tensors]
        synchronize_native_model(model, chunk_bytes=64)
        expected_model = StridedPolicy(0)
        for value, expected in zip(
            model.state_dict().values(),
            expected_model.state_dict().values(),
            strict=True,
        ):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert before == [identity(tensor) for tensor in tensors]
        # Strided slice sync must not write the rest of the backing allocation.
        backing = model.offset_slice.as_strided((20, 20), (20, 1), storage_offset=0)
        assert backing[0, 0].item() == -100.0

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.01, weight_decay=0.02, foreach=False
        )
        oracle = torch.optim.AdamW(
            expected_model.parameters(), lr=0.01, weight_decay=0.02, foreach=False
        )
        for step in range(2):
            for name in ("transposed", "contiguous"):
                parameter = getattr(model, name)
                parameter.grad = gradient(step, rank, parameter.shape)
                expected = getattr(expected_model, name)
                expected.grad = (
                    gradient(step, 0, parameter.shape)
                    + gradient(step, 1, parameter.shape)
                ) / world_size
            model.rank_local.grad = (
                gradient(step, 0, model.rank_local.shape) if rank == 0 else None
            )
            expected_model.rank_local.grad = (
                gradient(step, 0, model.rank_local.shape) / world_size
            )
            saved_grad = identity(model.contiguous.grad)
            average_native_gradients(model.parameters(), chunk_bytes=20)
            assert saved_grad == identity(model.contiguous.grad)
            assert model.unused.grad is None
            for name in ("transposed", "contiguous", "rank_local"):
                torch.testing.assert_close(
                    getattr(model, name).grad,
                    getattr(expected_model, name).grad,
                    rtol=0,
                    atol=0,
                )
            optimizer.step()
            oracle.step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, expected_model.state_dict()[name], rtol=0, atol=0
                )
            assert before == [identity(tensor) for tensor in tensors]
            optimizer.zero_grad(set_to_none=True)
            oracle.zero_grad(set_to_none=True)
        assert optimizer.state[model.transposed]["step"].item() == 2
        assert (
            torch.count_nonzero(optimizer.state[model.transposed]["exp_avg_sq"]) == 12
        )
        assert model.unused not in optimizer.state

        # Production CPU Adam owner also accepts a noncontiguous gradient on a
        # contiguous native parameter without changing native storage.
        native = nn.Parameter(torch.arange(12.0).reshape(3, 4))
        reference = nn.Parameter(native.detach().clone())
        native_identity = identity(native)
        cpu_adam = CPUOffloadedAdamW([native], lr=0.01, chunk_bytes=16)
        serial_adam = torch.optim.AdamW(
            [reference], lr=0.01, weight_decay=0.0, foreach=False
        )
        for step in range(2):
            native.grad = gradient(step, rank, native.shape)
            average_native_gradients([native], chunk_bytes=20)
            reference.grad = (
                gradient(step, 0, native.shape) + gradient(step, 1, native.shape)
            ) / world_size
            cpu_adam.step()
            serial_adam.step()
            torch.testing.assert_close(native, reference, rtol=0, atol=0)
            assert native_identity == identity(native)
        if rank == 0:
            assert cpu_adam.state_dict()["state"][0]["step"].item() == 2
        assert not torch.cuda.is_initialized()
        Path(root, f"rank-{rank}.passed").write_text(
            "synchronization/gradient/Adam/identity passed\n"
        )
    finally:
        distributed.destroy_process_group()


class NativeStridedCollectives(unittest.TestCase):
    def test_bounded_views_cover_logical_values_and_preserve_storage(self):
        for tensor in (
            torch.arange(257 * 128).reshape(128, 257).t(),
            torch.arange(400).reshape(20, 20)[2:8:2, 3:11:2],
            torch.arange(60).reshape(3, 4, 5).permute(2, 0, 1),
            torch.tensor(7),
        ):
            with self.subTest(shape=tuple(tensor.shape)):
                _validate_tensor(tensor)
                chunks = list(_chunks(tensor, 64))
                self.assertEqual(sum(chunk.numel() for chunk in chunks), tensor.numel())
                self.assertTrue(
                    all(chunk.numel() * tensor.element_size() <= 64 for chunk in chunks)
                )
                self.assertTrue(
                    all(
                        chunk.untyped_storage().data_ptr()
                        == tensor.untyped_storage().data_ptr()
                        for chunk in chunks
                    )
                )
                expected = tensor.clone()
                before = identity(tensor)
                for chunk in chunks:
                    chunk.copy_(chunk.contiguous() + 3)
                torch.testing.assert_close(tensor, expected + 3, rtol=0, atol=0)
                self.assertEqual(identity(tensor), before)

    def test_overlapping_sparse_and_invalid_chunks_rejected(self):
        for tensor in (
            torch.ones(1).expand(4),
            torch.arange(4.0).as_strided((2, 2), (1, 1)),
            torch.eye(3).to_sparse(),
        ):
            with self.assertRaises(ValueError):
                _validate_tensor(tensor)
        with self.assertRaises(ValueError):
            list(_chunks(torch.ones(3), 0))

    def test_two_rank_sync_gradient_and_real_adam_oracles(self):
        with tempfile.TemporaryDirectory() as root:
            multiprocessing.spawn(strided_worker, args=(2, root), nprocs=2, join=True)
            self.assertEqual(len(list(Path(root).glob("rank-*.passed"))), 2)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
