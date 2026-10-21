# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Score a complete rollout batch across native actor replicas.

Every rank supplies the same full batch and scores only its contiguous shard.
Results and local errors are gathered once, preserving original candidate order.
The caller owns behavior-version validation and GRPO normalization on the global
batch. A behavior snapshot lends its weights to the learner's own module for the
behavior pass only (espnet2.rl.miles_native_actor.BehaviorSnapshot), so a policy
must not cache tensors derived from its trainable weights across calls.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed

from .timing import profile_phase
from .types import support_mismatch_rate, validate_binary_mask


def _numbers(value):
    """A list nested to equal lengths around numbers or booleans, as one array."""
    try:
        array = np.asarray(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return array if array.dtype.kind in "biuf" else None


def _identity_digest(batch):
    """Hash inputs including native frontend sidechannels, without repr objects.

    Arrays, tensors and lists of numbers are hashed by dtype, shape and bytes,
    each array once however many candidates share it.
    """
    digest = hashlib.sha256()
    arrays = {}

    def text(value):
        serialized = str(value).encode()
        digest.update(str(len(serialized)).encode() + b":" + serialized)

    def put(value):
        if isinstance(value, torch.Tensor):
            digest.update(b"tensor")
            text((value.dtype, tuple(value.shape)))
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(b"array")
            text((value.dtype, value.shape))
            if value.dtype.hasobject:
                put(value.tolist())
            else:
                if id(value) not in arrays:
                    contiguous = np.ascontiguousarray(value)
                    arrays[id(value)] = value, hashlib.sha256(contiguous).digest()
                digest.update(arrays[id(value)][1])
        elif isinstance(value, np.generic):
            put(value.item())
        elif isinstance(value, Mapping):
            digest.update(b"mapping")
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Native score sidechannel keys must be strings")
            for key in sorted(value):
                put(key)
                put(value[key])
            digest.update(b"endmapping")
        elif isinstance(value, (list, tuple)):
            array = _numbers(value)
            if array is not None:
                put(array)
                return
            digest.update(b"sequence")
            put(len(value))
            for item in value:
                put(item)
            digest.update(b"endsequence")
        elif isinstance(value, Path):
            put(str(value))
        elif value is None or isinstance(value, (bool, int, float, str)):
            text(f"{type(value).__name__}:{value}")
        else:
            raise ValueError(f"Unsupported native scoring sidechannel {type(value)}")

    put(batch.batch)
    put(batch.non_tensor_batch)
    return digest.hexdigest()


def support_gate(policy, replicas):
    """Gate the current and behavior passes' support mismatches over all ranks
    and log their rates. The frozen reference drifts from the rollout's
    supports by design, so its rate is only logged."""
    metrics = {}
    for name in replicas[0]["supports"]:
        counts = [
            sum(part["supports"][name][index] for part in replicas) for index in (0, 1)
        ]
        rate = support_mismatch_rate(policy, counts, name, gate=name != "reference")
        if rate is not None:
            prefix = "" if name == "current" else f"{name}_"
            metrics[f"rollout/{prefix}support_mismatch_rate"] = rate
    return metrics


def score_native_batch(trainer, batch, behavior_policy=None):
    """Return complete CPU ``(current_old, behavior, reference)`` logprobs.

    All ranks must call this function in the same order. ``behavior_policy=None``
    means the caller has verified behavior is the current learner; current scoring
    is reused. A stale behavior is an independent native policy or a context
    manager that lends its weights to a module for the pass. Current, reference and
    behavior each keep their configured GPU/CPU device.
    Exceptions in local validation or scoring are reported to every live rank before
    any result is returned. Process death still relies on the process-group timeout.
    """
    world_size = distributed.get_world_size() if distributed.is_initialized() else 1
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    group = trainer._object_collective_group() if world_size > 1 else None
    payload = {"rank": rank, "error": None}
    try:
        if trainer.rank != rank or trainer.world_size != world_size:
            raise ValueError("Native scorer rank differs from its process group")
        if not len(batch) or len(batch) % world_size:
            raise ValueError("Native score candidate count must divide world_size")
        mask = batch.batch["response_mask"]
        validate_binary_mask(mask, "Global native scoring response_mask")
        if mask.ndim != 2 or not mask.sum(-1).gt(0).all():
            raise ValueError("Every native scoring candidate requires sampled actions")
        for index, (identity, source, generation) in enumerate(
            zip(
                batch.non_tensor_batch["uid"],
                batch.non_tensor_batch["source"],
                batch.non_tensor_batch["generation"],
                strict=True,
            )
        ):
            if identity != source["id"] or identity != generation["source_id"]:
                raise ValueError(f"Native score candidate identity differs at {index}")
        payload["identity"] = _identity_digest(batch)
        payload["behavior_is_current"] = (
            behavior_policy is None or behavior_policy is trainer.policy
        )
        width = len(batch) // world_size
        indices = np.arange(rank * width, (rank + 1) * width)
        local = batch.select_idxs(indices)
        supports = payload["supports"] = {}
        with torch.no_grad():
            with profile_phase(trainer, "current_score"):
                old = trainer._score_chunks(trainer.policy, local)
            supports["current"] = trainer.support_pass
            if behavior_policy is None or behavior_policy is trainer.policy:
                behavior = old
            else:
                with profile_phase(trainer, "behavior_score"):
                    behavior = trainer._score_chunks(behavior_policy, local)
                supports["behavior"] = trainer.support_pass
            if (
                getattr(trainer.config, "training_objective", "miles_ppo")
                == "west_grpo"
            ):
                # WEST's reference KL uses the raw model distribution. Current
                # and behavior passes above audit only rollout integrity; their
                # tempered, pre-top-k probabilities never enter the WEST loss.
                with profile_phase(trainer, "reference_score"):
                    reference = trainer._score_chunks(
                        trainer.reference, local, score_raw=True
                    )
            else:
                with profile_phase(trainer, "reference_score"):
                    reference = trainer._score_chunks(trainer.reference, local)
            supports["reference"] = trainer.support_pass
        for scores in (old, behavior, reference):
            if scores.shape != local.batch["response_mask"].shape:
                raise ValueError("Native scorer output action shape differs from input")
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite gathered native logprobs")
        payload.update(
            indices=indices.tolist(),
            old=old.detach().cpu(),
            behavior=behavior.detach().cpu(),
            reference=reference.detach().cpu(),
        )
    except Exception as error:
        payload = {"rank": rank, "error": f"{type(error).__name__}: {error}"}
    replicas = [None] * world_size
    if world_size > 1:
        # Tensor scoring above performs no collectives, so all local exceptions
        # reach this single gather and cannot strand other ranks in later phases.
        distributed.all_gather_object(replicas, payload, group=group)
    else:
        replicas[0] = payload
    failures = [
        f"rank {part['rank']}: {part['error']}" for part in replicas if part["error"]
    ]
    if failures:
        raise RuntimeError("Distributed native scoring failed: " + "; ".join(failures))
    identities = {part["identity"] for part in replicas}
    if len(identities) != 1:
        raise RuntimeError("Distributed native scoring input identities differ by rank")
    if len({part["behavior_is_current"] for part in replicas}) != 1:
        raise RuntimeError(
            "Distributed native scoring behavior identity differs by rank"
        )
    expected = list(range(len(batch)))
    actual = [index for part in replicas for index in part["indices"]]
    if actual != expected:
        raise RuntimeError("Gathered native score candidates changed order or coverage")
    trainer.support_metrics = support_gate(trainer.policy, replicas)
    return tuple(
        torch.cat([part[key] for part in replicas], dim=0)
        for key in ("old", "behavior", "reference")
    )
