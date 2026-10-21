"""Explicit WEST GRPO compatibility and reproducible training cursors.

The WEST formulas follow wenet-e2e/west b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb.
They intentionally remain separate from Miles' versioned PPO/IcePop default.
"""

import torch

from .miles_library import _algorithms, agg_loss


def west_group_advantages(rewards, mask, identities):
    """Normalize each prompt's scalar outcome with sample std plus 1e-4.

    No all-equal special case is added: this preserves WEST's FP32 arithmetic,
    including a possible tiny centering residual on constant decimal rewards.
    Miles still expands the resulting outcome to the valid sampled actions.
    """
    groups = {}
    for row, identity in enumerate(identities):
        groups.setdefault(identity, []).append(row)
    scalar = rewards.sum(-1)
    normalized = torch.empty_like(scalar)
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("WEST GRPO requires at least two answers per prompt")
        values = scalar[indices]
        normalized[indices] = (values - values.mean()) / (values.std() + 1e-4)
    returns = torch.stack(_algorithms().get_grpo_returns(normalized, list(mask))) * mask
    return returns, returns.clone()


def west_policy_loss(current, advantages, mask, **global_batch_info):
    """WEST's unit-valued on-policy ratio with a nonzero policy gradient."""
    # The ratio is exactly one, strictly inside both clip bounds. Miles' PPO
    # primitive therefore supplies WEST's original value and policy gradient.
    if current.dtype == torch.float32:
        values, _ = _algorithms().compute_policy_loss(
            current.detach() - current, advantages, 0.2, 0.2
        )
    else:
        # Native scores are FP32. Preserve the helper's other dtypes rather
        # than inherit the Miles primitive's unconditional FP32 conversion.
        values = -(torch.exp(current - current.detach()) * advantages)
    return agg_loss(values, mask, "seq-mean-token-mean", **global_batch_info)


def build_lr_scheduler(optimizer, config):
    """Match HF's constant or linear schedule, indexed by optimizer updates."""

    def factor(completed_updates):
        if completed_updates < config.lr_warmup_steps:
            return completed_updates / max(1, config.lr_warmup_steps)
        if config.lr_scheduler == "constant":
            return 1.0
        return max(
            0.0,
            (config.steps - completed_updates)
            / max(1, config.steps - config.lr_warmup_steps),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def select_training_sources(sources, update_index, config):
    """Read a deterministic absolute prompt cursor, including epoch crossings."""
    if len({source["id"] for source in sources}) < config.prompts_per_step:
        raise ValueError("The prompt batch requires distinct source records")
    start = update_index * config.prompts_per_step
    permutations = {}
    result = []
    for absolute_index in range(start, start + config.prompts_per_step):
        epoch, offset = divmod(absolute_index, len(sources))
        if epoch not in permutations:
            order = list(range(len(sources)))
            if config.shuffle_train_data:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(config.data_seed + epoch)
                order = torch.randperm(len(sources), generator=generator).tolist()
            permutations[epoch] = order
        result.append(sources[permutations[epoch][offset]])
    if len({source["id"] for source in result}) != config.prompts_per_step:
        raise ValueError("An epoch boundary produced duplicate prompts in one batch")
    return result
