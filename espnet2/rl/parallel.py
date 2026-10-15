# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Native data parallel reduction without a second full-model gradient buffer.

Every rank owns the full native actor and executes backward for its rollout
shard. Contiguous tensors communicate in place. Strided native views use bounded
contiguous staging chunks, then copy back without replacing their storage.
The initial model is synchronized before frozen reference cloning.
"""

import torch
import torch.distributed as distributed


@torch.no_grad()
def synchronize_native_model(policy, *, chunk_bytes):
    """Verify replica tensor identities and broadcast parameters and buffers."""
    tensors = list(policy.parameters()) + list(policy.buffers())
    signature = [
        (
            name,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            tensor.requires_grad,
        )
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
            if chunk.is_contiguous() and chunk.device == collective_device:
                distributed.broadcast(chunk, src=0)
            else:
                # NCCL requires device-resident contiguous storage. Stage only
                # this logical view, keeping the original strides and storage.
                communication = torch.empty(
                    chunk.shape, dtype=chunk.dtype, device=collective_device
                )
                communication.copy_(chunk)
                distributed.broadcast(communication, src=0)
                chunk.copy_(communication)


def _validate_tensor(tensor):
    if tensor.layout != torch.strided:
        raise ValueError("Native allreduce requires nonoverlapping strided storage")
    # Transposes and stepped slices are native views, not reasons to replace a
    # Parameter or frontend buffer. Expanded/overlapping views cannot safely
    # receive different collective results through copy_. Prove that dimensions
    # occupy disjoint storage spans, including views with gaps between elements.
    span = 1
    for stride, size in sorted(
        (stride, size)
        for stride, size in zip(tensor.stride(), tensor.shape, strict=True)
        if size > 1
    ):
        if stride < span:
            raise ValueError("Native allreduce requires nonoverlapping strided storage")
        span += (size - 1) * stride


def _chunks(tensor, chunk_bytes, *, logical_order=False):
    if chunk_bytes < 1:
        raise ValueError("Native collective chunk_bytes must be positive")
    width = max(1, chunk_bytes // tensor.element_size())
    if tensor.is_contiguous() and not logical_order:
        flattened = tensor.view(-1)
        for start in range(0, flattened.numel(), width):
            yield flattened[start : start + width]
    else:
        yield from _strided_chunks(tensor, width)


def _strided_chunks(tensor, width, dimension=0):
    """Partition logical indices into bounded views; never reshape-copy a tensor."""
    if tensor.numel() == 0:
        return
    if tensor.numel() <= width:
        yield tensor
        return
    while tensor.shape[dimension] <= 1:
        dimension += 1
    per_index = tensor.numel() // tensor.shape[dimension]
    count = max(1, width // per_index)
    for start in range(0, tensor.shape[dimension], count):
        view = tensor.narrow(
            dimension, start, min(count, tensor.shape[dimension] - start)
        )
        yield from _strided_chunks(view, width, dimension + 1)


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
        # A rank-local unused path may create a contiguous zero gradient while
        # another rank owns a transpose. Logical partitions must still have the
        # same element counts and order on every rank, regardless of grad stride.
        for chunk in _chunks(gradient, chunk_bytes, logical_order=True):
            communication = chunk if chunk.is_contiguous() else chunk.contiguous()
            distributed.all_reduce(communication, op=distributed.ReduceOp.SUM)
            communication.div_(world_size)
            if communication is not chunk:
                chunk.copy_(communication)
