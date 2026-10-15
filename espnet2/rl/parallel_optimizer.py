# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""CPU AdamW with stable native parameter storage and replica synchronization.

Data parallel actors keep their original parameters throughout optimization.
Only rank zero owns CPU shadow parameters and Adam moments. Its AdamW update
is copied into the original storage and broadcast after all ranks have finished
backward. This is data parallel training, not parameter or gradient sharding.
"""

import torch
import torch.distributed as distributed


class CPUOffloadedAdamW(torch.optim.Optimizer):
    """Keep one auditable CPU Adam state without moving DDP parameters.

    ``param_groups`` and ``state_dict`` use native parameter ordering and standard
    AdamW checkpoint keys. Every rank must call ``step`` together after gradient
    averaging. Only the owner may save or restore the complete optimizer.
    Frozen parameters and model buffers are never passed to this optimizer.
    """

    def __init__(
        self,
        parameters,
        *,
        lr,
        weight_decay=0.0,
        owner_rank=0,
        chunk_bytes=64 * 1024 * 1024,
    ):
        parameters = list(parameters)
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("CPU AdamW requires distinct native parameters")
        if any(not p.requires_grad for p in parameters):
            raise ValueError("CPU AdamW accepts only trainable parameters")
        if any(not p.is_contiguous() for p in parameters):
            raise ValueError("CPU AdamW requires contiguous native parameter storage")
        super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay})
        self.rank = distributed.get_rank() if distributed.is_initialized() else 0
        self.world_size = (
            distributed.get_world_size() if distributed.is_initialized() else 1
        )
        if not 0 <= owner_rank < self.world_size:
            raise ValueError("Optimizer owner_rank is outside the process group")
        self.owner_rank = owner_rank
        if chunk_bytes < 1:
            raise ValueError("CPU AdamW broadcast chunk_bytes must be positive")
        self.chunk_bytes = chunk_bytes
        self._native = parameters
        self._shadow = []
        self._cpu_optimizer = None
        if self.rank == owner_rank:
            self._shadow = [
                torch.nn.Parameter(p.detach().to("cpu", copy=True)) for p in parameters
            ]
            # foreach allocates tensor-list intermediates at model scale.
            self._cpu_optimizer = torch.optim.AdamW(
                self._shadow, lr=lr, weight_decay=weight_decay, foreach=False
            )

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if self._cpu_optimizer is not None:
            self._cpu_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Native CPU AdamW does not support optimizer closures")
        error = [None]
        if self.rank == self.owner_rank:
            try:
                for group, cpu_group in zip(
                    self.param_groups, self._cpu_optimizer.param_groups, strict=True
                ):
                    cpu_group["lr"] = group["lr"]
                    cpu_group["weight_decay"] = group["weight_decay"]
                for native, shadow in zip(self._native, self._shadow, strict=True):
                    shadow.grad = (
                        None
                        if native.grad is None
                        else native.grad.detach().to("cpu", copy=True)
                    )
                    if (
                        shadow.grad is not None
                        and not torch.isfinite(shadow.grad).all()
                    ):
                        raise FloatingPointError("Nonfinite CPU Adam gradient")
                self._cpu_optimizer.step()
                for native, shadow in zip(self._native, self._shadow, strict=True):
                    # copy_ preserves native Parameter identity, device and storage.
                    native.copy_(shadow)
                    if shadow in self._cpu_optimizer.state:
                        self.state[native] = self._cpu_optimizer.state[shadow]
            except Exception as failure:
                error[0] = f"{type(failure).__name__}: {failure}"
            finally:
                self._cpu_optimizer.zero_grad(set_to_none=True)
        if self.world_size > 1:
            # Tell all ranks about an owner-side error before tensor broadcasts.
            # NCCL object collectives use the already selected rank-local device.
            distributed.broadcast_object_list(error, src=self.owner_rank)
        if error[0] is not None:
            raise RuntimeError(f"CPU AdamW owner step failed: {error[0]}")
        if self.world_size > 1:
            for parameter in self._native:
                width = max(1, self.chunk_bytes // parameter.element_size())
                flattened = parameter.detach().view(-1)
                for start in range(0, flattened.numel(), width):
                    distributed.broadcast(
                        flattened[start : start + width], src=self.owner_rank
                    )

    def state_dict(self):
        if self.rank != self.owner_rank:
            raise RuntimeError("Only the CPU AdamW owner has a complete state_dict")
        return self._cpu_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        if self.rank != self.owner_rank:
            raise RuntimeError("Only the CPU AdamW owner may restore optimizer state")
        self._cpu_optimizer.load_state_dict(state_dict)
        # Restore CPU placement even when loading a GPU-produced Adam checkpoint.
        for native, shadow in zip(self._native, self._shadow, strict=True):
            if shadow not in self._cpu_optimizer.state:
                continue
            state = self._cpu_optimizer.state[shadow]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu")
            self.state[native] = state
        for group, cpu_group in zip(
            self.param_groups, self._cpu_optimizer.param_groups, strict=True
        ):
            group["lr"] = cpu_group["lr"]
            group["weight_decay"] = cpu_group["weight_decay"]
