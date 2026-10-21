"""Call the pinned Miles library for GRPO normalization, returns, PPO and KL.

Only tensor layout and native reduction are adapted here. Miles is installed
as an optional library; no source is copied into ESPnet or patched at runtime.
"""

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

PIN = "a22fd824191c64ba7c5d6c531f125406e4ce9cc6"


def _algorithms():
    from miles.backends.training_utils.loss.hub import math_utils

    return math_utils


def verify_miles(config, *, runtime=False):
    """Verify the imported algorithm pin and, optionally, the async runtime.

    Runtime binding additionally requires the original driver and an unchanged
    tracked worktree, including files outside the algorithm boundary.
    """
    # Editable installs can expose ``miles`` as a namespace package. Verify
    # the imported algorithm module itself, which must have a real source file.
    root = Path(_algorithms().__file__).resolve().parents[5]
    requested = config.miles_commit
    if requested != PIN:
        raise ValueError("The recipe must use the reviewed Miles pin")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != requested:
        raise ValueError(
            f"Miles revision {revision} differs from recipe pin {requested}"
        )
    paths = [
        "miles/backends/training_utils/loss/hub/math_utils.py",
        "miles/ray/rollout/train_data_conversion.py",
        "miles/backends/training_utils/loss/hub/corrections.py",
    ]
    fast_weights = getattr(config, "weight_sync_backend", "disk") in (
        "miles",
        "miles_shm",
    )
    if fast_weights:
        paths.extend(
            "miles/backends/training_utils/weight_update/" + name
            for name in (
                "protocols/cuda_ipc.py",
                "protocols/broadcast.py",
                "session.py",
            )
        )
    if runtime:
        paths.append("train_async.py")
    result = {"commit": revision, "source": str(root), "algorithms": {}}
    for relative in paths:
        original = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        contents = (root / relative).read_bytes()
        if original != contents:
            if relative == "train_async.py":
                raise ValueError("Pinned Miles async driver has local changes")
            raise ValueError(f"Pinned Miles algorithm has local changes: {relative}")
        digest = hashlib.sha256(contents).hexdigest()
        if relative == "train_async.py":
            result["async_driver_sha256"] = digest
        else:
            result["algorithms"][relative] = digest
    if runtime or fast_weights:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ValueError(
                "The Miles runtime requires an unchanged tracked source tree"
            )
    return result


def compute_grpo_outcome_advantage(
    rewards, mask, identities, norm_adv_by_std_in_grpo=True
):
    from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
    from miles.utils.types import Sample

    groups = {}
    samples = []
    for index, identity in enumerate(identities):
        group = groups.setdefault(identity, len(groups))
        samples.append(Sample(group_index=group, index=index, rollout_id=index))
    args = SimpleNamespace(
        advantage_estimator="grpo", grpo_std_normalization=norm_adv_by_std_in_grpo
    )
    normalized = _normalize_rewards_by_rollout(
        args, samples, rewards.sum(-1).tolist(), None
    )
    scalar = torch.tensor(normalized, dtype=rewards.dtype, device=rewards.device)
    returns = torch.stack(_algorithms().get_grpo_returns(scalar, list(mask))) * mask
    return returns, returns.clone()


def agg_loss(
    values, mask, mode="token-mean", dp_size=1, batch_num_tokens=None, **kwargs
):
    if mode == "seq-mean-token-mean":
        lengths = mask.sum(-1)
        if (lengths <= 0).any():
            raise ValueError("Each WEST answer requires valid sampled actions")
        denominator = kwargs.get("global_batch_size", len(mask))
        if denominator <= 0:
            raise ValueError("No valid rollout answers")
        return ((values * mask).sum(-1) / lengths).sum() * dp_size / denominator
    if mode != "token-mean":
        raise ValueError("Unsupported native loss reduction")
    denominator = batch_num_tokens if batch_num_tokens is not None else mask.sum()
    if denominator <= 0:
        raise ValueError("No valid rollout actions")
    return (values * mask).sum() * dp_size / denominator


def compute_policy_loss_vanilla(
    old, current, advantages, mask, loss_agg_mode, config, *, rollout_log_probs=None
):
    if rollout_log_probs is not None and rollout_log_probs.shape != mask.shape:
        raise ValueError(
            "Behavior probabilities must preserve the sampled-action shape"
        )
    if not mask.bool().any():
        raise ValueError("No valid rollout actions")
    losses, clipped = _algorithms().compute_policy_loss(
        old - current,
        advantages,
        config.clip_ratio_low,
        config.clip_ratio_high,
    )
    metrics = {
        "actor/clip_fraction": ((clipped * mask).sum() / mask.sum()).detach().item()
    }
    if rollout_log_probs is not None:
        from miles.backends.training_utils.loss.hub.corrections import icepop_function

        # IcePop acts on the policy-gradient term only; KL keeps its own mask.
        lengths = [int(row.sum()) for row in mask.bool()]
        train_rows = [
            values[row.bool()].detach() for values, row in zip(old, mask, strict=True)
        ]
        behavior_rows = [
            values[row.bool()].detach()
            for values, row in zip(rollout_log_probs, mask, strict=True)
        ]
        valid = losses[mask.bool()]
        corrected, _, correction = icepop_function(
            config,
            pg_loss=valid,
            train_log_probs=train_rows,
            rollout_log_probs=behavior_rows,
            loss_masks=[torch.ones(length, device=valid.device) for length in lengths],
        )
        losses = torch.zeros_like(losses).masked_scatter(mask.bool(), corrected)
        for name, values in correction.items():
            metrics[f"actor/{name}"] = values.mean().detach().item()
    loss = agg_loss(losses, mask, loss_agg_mode, **config.global_batch_info)
    return loss, metrics


def kl_penalty(current, reference, kind):
    return _algorithms().compute_approx_kl(current, reference, kind)
