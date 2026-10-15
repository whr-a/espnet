# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Shared SGLang engine settings for the ESPnet RL families.

Every family runs its rollout engine in its own process with one device, so
the settings that keep inference numerically comparable to the native actor
are the same everywhere. A family adds only what its own model needs, and a
recipe overrides anything that is not part of that native contract.
"""

# Deterministic, cache-free execution so engine log probabilities stay
# comparable with the native actor's own forward pass. One device per replica
# means tensor parallelism is always one.
SHARED_DEFAULTS = {
    "tp_size": 1,
    "disable_cuda_graph": True,
    "disable_radix_cache": True,
    "log_level": "info",
}
# Every family freezes these: the seed has to advance with the policy step or
# a group's candidates stop differing, and GRPO has no signal without that.
SHARED_FROZEN = ("model_path", "dtype", "tp_size", "random_seed")


def engine_kwargs(config, step, *, defaults=None, frozen=()):
    """Resolve the engine settings for one rollout.

    ``defaults`` carries the family's own settings and ``frozen`` names the
    ones that define its native execution contract: a recipe may repeat such
    a value but never change it. ``rollout_mem_fraction`` is the uniform way
    to size a collocated engine, so it may not disagree with an explicit
    ``mem_fraction_static`` in ``sglang_engine_kwargs``.
    """
    resolved = dict(SHARED_DEFAULTS)
    resolved.update(defaults or {})
    resolved["random_seed"] = config.seed + step
    frozen = tuple(SHARED_FROZEN) + tuple(frozen)
    overrides = dict(config.sglang_engine_kwargs)
    if config.rollout_mem_fraction is not None:
        requested = overrides.get("mem_fraction_static", config.rollout_mem_fraction)
        if requested != config.rollout_mem_fraction:
            raise ValueError(
                "rollout_mem_fraction and sglang_engine_kwargs.mem_fraction_static "
                "disagree"
            )
        # Injected before the contract check so a family that freezes its
        # memory fraction still rejects an attempt to resize the engine.
        overrides["mem_fraction_static"] = config.rollout_mem_fraction
    conflicts = [
        name
        for name in frozen
        if name in overrides and overrides[name] != resolved.get(name)
    ]
    if conflicts:
        raise ValueError(
            "sglang_engine_kwargs cannot change the native execution contract: "
            + ", ".join(sorted(conflicts))
        )
    resolved.update(overrides)
    fraction = resolved.get("mem_fraction_static")
    if fraction is not None and not 0 < fraction < 1:
        raise ValueError("mem_fraction_static must lie in (0, 1)")
    return resolved
