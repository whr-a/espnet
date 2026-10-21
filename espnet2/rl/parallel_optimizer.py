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

from .timing import profile_phase


class CPUOffloadedAdamW(torch.optim.Optimizer):
    """Keep one auditable CPU Adam state without moving DDP parameters.

    ``param_groups`` and ``state_dict`` use native parameter ordering and standard
    AdamW checkpoint keys. Every rank must call ``step`` together after gradient
    averaging. Only the owner may save or restore the complete optimizer.
    Frozen parameters and model buffers are never passed to this optimizer.
    The CPU copy is always FP32, so a bfloat16 actor keeps FP32 master weights:
    updates far below a low-precision parameter's resolution accumulate here
    instead of rounding away on every step.
    """

    def __init__(
        self,
        parameters,
        *,
        lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
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
        super().__init__(
            parameters,
            {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps},
        )
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
                torch.nn.Parameter(p.detach().to("cpu", torch.float32, copy=True))
                for p in parameters
            ]
            # foreach allocates tensor-list intermediates at model scale.
            self._cpu_optimizer = torch.optim.AdamW(
                self._shadow,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                foreach=False,
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
                    for key in ("lr", "weight_decay", "betas", "eps"):
                        cpu_group[key] = group[key]
                with profile_phase(self, "optimizer_d2h", memory=False):
                    for native, shadow in zip(self._native, self._shadow, strict=True):
                        shadow.grad = (
                            None
                            if native.grad is None
                            else native.grad.detach().to(
                                "cpu", torch.float32, copy=True
                            )
                        )
                        if (
                            shadow.grad is not None
                            and not torch.isfinite(shadow.grad).all()
                        ):
                            raise FloatingPointError("Nonfinite CPU Adam gradient")
                with profile_phase(self, "optimizer_cpu", memory=False):
                    self._cpu_optimizer.step()
                with profile_phase(self, "optimizer_h2d", memory=False):
                    for native, shadow in zip(self._native, self._shadow, strict=True):
                        # Preserve Parameter identity/storage and rounded masters.
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
        # A scheduler advances the public groups after step(), while the CPU
        # Adam group still carries the LR used for that just-finished update.
        # Save the next-update LR, as torch AdamW + HF Trainer would.
        for group, cpu_group in zip(
            self.param_groups, self._cpu_optimizer.param_groups, strict=True
        ):
            for key in ("lr", "weight_decay", "betas", "eps", "initial_lr"):
                if key in group:
                    cpu_group[key] = group[key]
        state = self._cpu_optimizer.state_dict()
        # Adam moments alone cannot resume sub-BF16 updates. The CPU masters
        # are part of this optimizer's state, in native parameter order.
        state["espnet_master_weights_version"] = 1
        # Like standard optimizer state_dict(), these are live tensor views.
        # The synchronous checkpoint save finishes before another update; an
        # extra full FP32 model copy would only raise host peak memory.
        state["espnet_master_weights"] = [p.detach() for p in self._shadow]
        return state

    def load_state_dict(self, state_dict):
        if self.rank != self.owner_rank:
            raise RuntimeError("Only the CPU AdamW owner may restore optimizer state")
        masters = state_dict.get("espnet_master_weights")
        if masters is None and any(p.dtype != torch.float32 for p in self._native):
            raise ValueError(
                "A low-precision resume requires checkpointed FP32 masters"
            )
        if masters is not None:
            if state_dict.get("espnet_master_weights_version") != 1 or len(
                masters
            ) != len(self._shadow):
                raise ValueError("Invalid CPU AdamW master-weight checkpoint")
            for native, shadow, value in zip(
                self._native, self._shadow, masters, strict=True
            ):
                if value.shape != shadow.shape or value.dtype != torch.float32:
                    raise ValueError("CPU AdamW master-weight shape or dtype differs")
                if not torch.isfinite(value).all():
                    raise ValueError("CPU AdamW master weights must be finite")
                if not torch.equal(value.to(dtype=native.dtype), native.detach().cpu()):
                    raise ValueError(
                        "CPU AdamW master weights differ from the native checkpoint"
                    )
                shadow.data.copy_(value)
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
            for key in ("lr", "weight_decay", "betas", "eps", "initial_lr"):
                if key in cpu_group:
                    group[key] = cpu_group[key]
