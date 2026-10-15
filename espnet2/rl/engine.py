# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Rollout engines and their shared SGLang settings for the ESPnet RL families.

Every family runs its rollout engine in its own process with one device, so
the settings that keep inference numerically comparable to the native actor
are the same everywhere. A family adds only what its own model needs, and a
recipe overrides anything that is not part of that native contract.
"""

from pathlib import Path

# Deterministic, cache-free execution so engine log probabilities stay
# comparable with the native actor's own forward pass. One device per replica
# means tensor parallelism is always one.
SHARED_DEFAULTS = {
    "tp_size": 1,
    "disable_cuda_graph": True,
    "disable_radix_cache": True,
    "log_level": "info",
}
# Every family freezes these. The seed is fixed per replica for the engine's
# whole life; its RNG then advances from batch to batch, so the candidates of
# a group and of successive updates differ, and replicas never repeat each
# other.
SHARED_FROZEN = ("model_path", "dtype", "tp_size", "random_seed")


def engine_kwargs(config, replica, *, defaults=None, frozen=()):
    """Resolve the engine settings for one rollout replica.

    ``defaults`` carries the family's own settings and ``frozen`` names the
    ones that define its native execution contract: a recipe may repeat such
    a value but never change it. ``rollout_mem_fraction`` is the uniform way
    to size a collocated engine, so it may not disagree with an explicit
    ``mem_fraction_static`` in ``sglang_engine_kwargs``.
    """
    resolved = dict(SHARED_DEFAULTS)
    resolved.update(defaults or {})
    resolved["random_seed"] = config.seed + replica
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


class RolloutEngine:
    """An architecture's inference engine, alive for a whole run.

    The rollout process builds it from the first exported policy, loads every
    later export in place with ``update`` and asks it for one batch per
    ``generate``. ``directory`` is this engine's private scratch directory.
    """

    def __init__(self, config, model_path, *, directory, replica=0):
        self.config, self.directory, self.replica = config, Path(directory), replica

    def update(self, model_path):
        raise NotImplementedError

    def generate(self, sources, rollout_id):
        """Return ``config.group_size`` generations per source, grouped by
        source in the order given, in the generation schema used today."""
        raise NotImplementedError

    def release(self):
        """Give the last batch's working memory back to a GPU it shares with
        a training rank, keeping what the next batch needs from the start."""

    def close(self):
        pass


class SGLangRolloutEngine(RolloutEngine):
    # The family's own settings and the names that define its native
    # execution contract (see engine_kwargs).
    defaults = {}
    frozen = ()

    def __init__(self, config, model_path, *, directory, replica=0):
        super().__init__(config, model_path, directory=directory, replica=replica)
        from sglang import Engine

        self.engine = Engine(**self.settings(model_path))

    def settings(self, model_path):
        return engine_kwargs(
            self.config,
            self.replica,
            defaults=dict(self.defaults, model_path=str(model_path)),
            frozen=self.frozen,
        )

    def update(self, model_path):
        # Every export holds the complete policy; SGLang streams it into the
        # model's own load_weights and flushes the idle engine's caches.
        success, message, _ = self.engine.update_weights_from_disk(str(model_path))
        if not success:
            raise RuntimeError(f"SGLang could not load {model_path}: {message}")

    def release(self):
        # Waits until the scheduler is idle, then empties its allocator cache;
        # the weights and the static KV pool stay allocated.
        result = self.engine.loop.run_until_complete(
            self.engine.tokenizer_manager.flush_cache(timeout_s=60)
        )
        if not result.success:
            raise RuntimeError(f"SGLang kept its batch memory: {result.message}")

    def close(self):
        self.engine.shutdown()
