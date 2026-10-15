# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Native data parallel reduction without a second full-model gradient buffer.

Every rank owns the full native actor and executes backward for its rollout
shard. Gradients are averaged in their existing storage in bounded contiguous
chunks. The initial model is synchronized before frozen reference cloning.
"""

import torch
import torch.distributed as distributed


@torch.no_grad()
def synchronize_native_model(policy, *, chunk_bytes):
    """Verify replica tensor identities and broadcast parameters and buffers."""
    tensors = list(policy.parameters()) + list(policy.buffers())
    signature = [
        (name, tuple(tensor.shape), str(tensor.dtype), tensor.requires_grad)
        for name, tensor in list(policy.named_parameters())
        + list(policy.named_buffers())
    ]
    replicas = [None] * distributed.get_world_size()
    distributed.all_gather_object(replicas, signature)
    if any(replica != signature for replica in replicas):
        raise ValueError("Native data parallel model tensor identities differ by rank")
    collective_device = next(policy.parameters()).device
    for tensor in tensors:
        _validate_tensor(tensor)
        for chunk in _chunks(tensor, chunk_bytes):
            if chunk.device == collective_device:
                distributed.broadcast(chunk, src=0)
            else:
                # NCCL cannot broadcast a CPU buffer. Stage only this bounded
                # chunk and preserve the native buffer's device and storage.
                communication = chunk.to(collective_device)
                distributed.broadcast(communication, src=0)
                chunk.copy_(communication)


def _validate_tensor(tensor):
    if not tensor.is_contiguous() or tensor.layout != torch.strided:
        raise ValueError("Native allreduce requires contiguous dense parameter storage")


def _chunks(tensor, chunk_bytes):
    width = max(1, chunk_bytes // tensor.element_size())
    flattened = tensor.view(-1)
    for start in range(0, flattened.numel(), width):
        yield flattened[start : start + width]


@torch.no_grad()
def average_native_gradients(parameters, *, chunk_bytes):
    """Average full accumulated gradients, including rank-local unused paths.

    An initial small usage collective prevents mismatched tensor collectives when
    different prompts use different heads. Globally unused parameters retain a
    None gradient, preserving Adam's standard unused-parameter semantics.
    """
    parameters = list(parameters)
    if not parameters:
        raise ValueError("Native allreduce requires trainable parameters")
    device = parameters[0].device
    used = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=device,
        dtype=torch.int32,
    )
    distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
    world_size = distributed.get_world_size()
    for parameter, count in zip(parameters, used.tolist(), strict=True):
        if not count:
            continue
        _validate_tensor(parameter)
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradient = parameter.grad
        if gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise ValueError("Native allreduce gradient shape or dtype mismatch")
        _validate_tensor(gradient)
        for chunk in _chunks(gradient, chunk_bytes):
            distributed.all_reduce(chunk, op=distributed.ReduceOp.SUM)
            chunk.div_(world_size)
