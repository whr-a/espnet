"""Complete native Bagpiper policies over Miles' original IPC/NCCL helpers.

The small marker binds the version to a canonical digest of every inference
tensor. The receiver checks it before the family's strict, complete loader.
Neither CUDA storage nor an unfinished update is allowed past the acknowledgment.
"""

import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

HASH_KIND = "native-rollout-tensors/v1"
MARKER = "__espnet_policy_identity__"
CODEC = "multimodal_io_dict.discrete_audio.codec_model."


def tensor_digest(weights):
    """Hash names, exact dtypes/shapes and bytes, independent of iteration order."""
    digest, names = hashlib.sha256(), set()
    for name, value in sorted(weights, key=lambda item: item[0]):
        if name in names:
            raise ValueError(f"Repeated native rollout tensor {name}")
        names.add(name)
        header = json.dumps([name, str(value.dtype), list(value.shape)])
        digest.update(header.encode() + b"\0")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def verify_weights(weights, receipt):
    """Verify a transferred policy before loading; ordinary disk exports pass."""
    weights = list(weights)
    markers = [value for name, value in weights if name == MARKER]
    if not markers:
        return weights
    if len(markers) != 1:
        raise ValueError("A transferred policy requires exactly one identity marker")
    identity = json.loads(bytes(markers[0].cpu().tolist()))
    tensors = [(name, value) for name, value in weights if name != MARKER]
    if (
        set(identity) != {"step", "native_sha256", "hash_kind"}
        or identity["hash_kind"] != HASH_KIND
        or type(identity["step"]) is not int
        or identity["step"] < 0
        or tensor_digest(tensors) != identity["native_sha256"]
    ):
        raise ValueError("Native rollout tensor identity or SHA256 mismatch")
    Path(receipt).write_text(json.dumps(identity) + "\n")
    return tensors


class EngineClient:
    """Miles' asynchronous API over the existing socket or Ray actor RPC."""

    def __init__(self, target, *, remote=False):
        self.target, self.remote = target, remote

    def __getattr__(self, method):
        async def call(*args, **kwargs):
            if self.remote:
                return await self.target.weight_update.remote(method, args, kwargs)
            return await asyncio.to_thread(
                self.target.weight_update, method, args, kwargs
            )

        return call


class MilesWeightPublisher:
    """A single native learner and one TP1 engine, with persistent NCCL state."""

    def __init__(self, clients, *, collocated, sglang_source="", shared_memory=False):
        if len(clients) != 1:
            raise ValueError("Miles native weight transfer requires one engine")
        if shared_memory and not collocated:
            raise ValueError("CPU shared-memory transfer requires a collocated engine")
        if sglang_source:
            source = Path(sglang_source).resolve() / "python"
            imported = sys.modules.get("sglang")
            if imported is not None and not Path(
                imported.__file__
            ).resolve().is_relative_to(source):
                raise ValueError(
                    "Learner and rollout import different SGLang transport sources"
                )
            sys.path.insert(0, str(source))
        self.clients, self.collocated = clients, collocated
        self.shared_memory = shared_memory
        self.group = self.ipc_group = None
        self.owns_world = False
        self.rendezvous = None
        self.failed_storage = None
        self.name = "espnet-native-weights"

    def _wait(self, calls):
        from miles.utils.async_utils import submit, wait_futures

        return wait_futures([submit(call) for call in calls])

    def publish(self, policy, step):
        from miles.backends.training_utils.weight_update.protocols import broadcast
        from miles.backends.training_utils.weight_update.session import (
            check_weight_sync_results,
        )
        from miles.utils.async_utils import wait_futures

        start = time.monotonic()
        # Exactly the verbatim export mapping; the frozen codec serves the TTS
        # reward on the learner and is explicitly skipped by the rollout loader.
        weights = [
            (name, value.detach().contiguous())
            for name, value in policy.checkpoint_state_dict().items()
            if name != "vocab_weight" and not name.startswith(CODEC)
        ]
        if not weights or (
            not self.shared_memory and any(not value.is_cuda for _, value in weights)
        ):
            raise ValueError("Miles weight transfer requires complete CUDA tensors")
        if self.shared_memory:
            weights = [(name, value.cpu()) for name, value in weights]
        stage_end = time.monotonic()
        identity = {
            "step": step,
            "native_sha256": tensor_digest(weights),
            "hash_kind": HASH_KIND,
        }
        hash_end = time.monotonic()
        marker = torch.tensor(
            list(json.dumps(identity).encode()),
            dtype=torch.uint8,
            device=weights[0][1].device,
        )
        weights.append((MARKER, marker))
        if self.shared_memory and shutil.disk_usage("/dev/shm").free < sum(
            value.nbytes for _, value in weights
        ):
            raise RuntimeError("Insufficient /dev/shm for the complete native policy")
        self._wait([client.begin(identity) for client in self.clients])
        begin_end = time.monotonic()
        retained, completed = None, False
        try:
            if self.collocated:
                from miles.backends.training_utils.weight_update.protocols import (
                    cuda_ipc,
                )

                if not cuda_ipc.FlattenedTensorBucket.supports_multi_dtypes:
                    raise RuntimeError(
                        "Complete native IPC requires mixed-dtype buckets"
                    )

                offset = 0
                for name, value in weights:
                    if offset % value.element_size():
                        raise ValueError(f"Unaligned IPC tensor {name}")
                    offset += value.nbytes
                if self.ipc_group is None:
                    if not dist.is_initialized():
                        self.rendezvous = tempfile.TemporaryDirectory()
                        dist.init_process_group(
                            "gloo",
                            init_method=f"file://{self.rendezvous.name}/store",
                            rank=0,
                            world_size=1,
                        )
                        self.owns_world = True
                    if dist.get_world_size() != 1:
                        raise ValueError("Native IPC requires a single training rank")
                    self.ipc_group = dist.new_group(backend="gloo")
                strategy = torch.multiprocessing.get_sharing_strategy()
                try:
                    if self.shared_memory:
                        torch.multiprocessing.set_sharing_strategy("file_system")
                    futures, retained = cuda_ipc._send_to_colocated_engine(
                        weights,
                        ipc_engine=self.clients[0],
                        ipc_gather_src=0,
                        ipc_gather_group=self.ipc_group,
                    )
                finally:
                    if self.shared_memory:
                        torch.multiprocessing.set_sharing_strategy(strategy)
            else:
                if self.group is None:
                    self.group = broadcast.connect_rollout_engines_from_distributed(
                        SimpleNamespace(),
                        self.name,
                        self.clients,
                        engine_gpu_counts=[1],
                    )
                futures = broadcast.update_weights_from_distributed(
                    self.name, self.group, self.clients, weights
                )
            check_weight_sync_results(wait_futures(futures), is_lora=False)
            transfer_end = time.monotonic()
            acknowledgments = self._wait(
                [client.finish(identity) for client in self.clients]
            )
            completed = True
            return {
                **identity,
                "weight_sync_backend": (
                    "miles_shm"
                    if self.shared_memory
                    else ("miles_ipc" if self.collocated else "miles_nccl")
                ),
                "hash_time": hash_end - stage_end,
                "publish_time": time.monotonic() - hash_end,
                "perf/weight_stage_time": stage_end - start,
                "perf/weight_hash_time": hash_end - stage_end,
                "perf/weight_resume_time": begin_end - hash_end,
                "perf/weight_transfer_verify_load_time": transfer_end - begin_end,
                "perf/weight_ack_time": time.monotonic() - transfer_end,
                "tensor_bytes": sum(value.nbytes for _, value in weights),
                "acknowledgments": acknowledgments,
            }
        finally:
            # A failed connection cannot prove that the receiver has finished
            # using its handles. Retain storage until that engine is disposed.
            if not completed:
                self.failed_storage = (retained, weights, marker)
            del retained, weights, marker
            if self.collocated and completed and not self.shared_memory:
                torch.cuda.ipc_collect()
            if not self.shared_memory:
                torch.cuda.empty_cache()

    def close(self):
        from miles.backends.training_utils.weight_update.protocols import broadcast

        if self.failed_storage is not None and self.collocated:
            self._wait([client.abort() for client in self.clients])
            self.failed_storage = None
            if not self.shared_memory:
                torch.cuda.ipc_collect()
        if self.group is not None:
            broadcast.disconnect_rollout_engines_from_distributed(
                SimpleNamespace(), self.name, self.group, self.clients
            )
            self.group = None
        if self.ipc_group is not None:
            dist.destroy_process_group(self.ipc_group)
            self.ipc_group = None
        if self.owns_world:
            dist.destroy_process_group()
            self.owns_world = False
        if self.rendezvous is not None:
            self.rendezvous.cleanup()
            self.rendezvous = None
