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


def verify_miles(config):
    from miles import __file__ as module_file

    root = Path(module_file).resolve().parent.parent
    requested = config.miles_commit
    if requested != PIN:
        raise ValueError("The recipe must use the reviewed Miles pin")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != requested:
        raise ValueError(f"Miles revision {revision} differs from recipe pin {requested}")
    paths = [
        root / "miles/backends/training_utils/loss/hub/math_utils.py",
        root / "miles/ray/rollout/train_data_conversion.py",
    ]
    result = {"commit": revision, "source": str(root), "algorithms": {}}
    for path in paths:
        relative = str(path.relative_to(root))
        original = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        contents = path.read_bytes()
        if original != contents:
            raise ValueError(f"Pinned Miles algorithm has local changes: {relative}")
        result["algorithms"][relative] = hashlib.sha256(contents).hexdigest()
    _algorithms()
    return result


def compute_grpo_outcome_advantage(rewards, mask, identities, norm_adv_by_std_in_grpo=True):
    from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
    from miles.utils.types import Sample

    groups = {}
    samples = []
    for index, identity in enumerate(identities):
        group = groups.setdefault(identity, len(groups))
        samples.append(Sample(group_index=group, index=index, rollout_id=index))
    args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=norm_adv_by_std_in_grpo)
    normalized = _normalize_rewards_by_rollout(args, samples, rewards.sum(-1).tolist(), None)
    scalar = torch.tensor(normalized, dtype=rewards.dtype, device=rewards.device)
    returns = torch.stack(_algorithms().get_grpo_returns(scalar, list(mask))) * mask
    return returns, returns.clone()


def agg_loss(values, mask, mode="token-mean", dp_size=1, batch_num_tokens=None, **kwargs):
    if mode != "token-mean":
        raise ValueError("Only global token-mean reduction is supported")
    denominator = batch_num_tokens if batch_num_tokens is not None else mask.sum()
    if denominator <= 0:
        raise ValueError("No valid rollout actions")
    return (values * mask).sum() * dp_size / denominator


def compute_policy_loss_vanilla(old, current, advantages, mask, loss_agg_mode, config):
    losses, clipped = _algorithms().compute_policy_loss(
        old - current, advantages, config.clip_ratio_low, config.clip_ratio_high,
    )
    loss = agg_loss(losses, mask, loss_agg_mode, **config.global_batch_info)
    return loss, {"actor/clip_fraction": ((clipped * mask).sum() / mask.sum()).detach().item()}


def kl_penalty(current, reference, kind):
    return _algorithms().compute_approx_kl(current, reference, kind)

