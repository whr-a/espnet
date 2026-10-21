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


def fit_context(config, defaults, width):
    """Size ``config``'s engine for requests of up to ``width`` tokens on its
    recipe's KV budget, as an evaluation with longer prompts or responses
    needs: a longer context_length runs proportionally fewer requests, so
    max_running_requests x context_length, which the recipe fitted to its KV
    pool, does not grow.
    """
    settings = {**defaults, **config.sglang_engine_kwargs}
    context = settings.get("context_length")
    running = settings.get("max_running_requests")
    if context is None or running is None or width <= context:
        return
    config.sglang_engine_kwargs = {
        **config.sglang_engine_kwargs,
        "context_length": width,
        "max_running_requests": max(1, running * context // width),
    }


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
        """Give memory back to a GPU this engine shares with a training rank
        until the next ``update``, which takes back whatever it needs."""

    def close(self):
        pass


class SGLangRolloutEngine(RolloutEngine):
    # The family's own settings and the names that define its native
    # execution contract (see engine_kwargs).
    defaults = {}
    frozen = ()
    # A memory-saver engine releases its weights and KV pool as well; it then
    # holds no policy until update() resumes them and loads the next one.
    memory_saver = released = False
    publication_kv_released = False

    def __init__(self, config, model_path, *, directory, replica=0):
        super().__init__(config, model_path, directory=directory, replica=replica)
        from sglang import Engine

        settings = self.settings(model_path)
        self.memory_saver = bool(settings.get("enable_memory_saver"))
        self.context_length = settings.get("context_length")
        self.engine = Engine(**settings)

    def settings(self, model_path):
        return engine_kwargs(
            self.config,
            self.replica,
            defaults=dict(self.defaults, model_path=str(model_path)),
            frozen=self.frozen,
        )

    def check_kv_pool(self):
        """Fail unless SGLang can never retract a running request.

        SGLang retracts running requests when its KV pool runs out and
        recomputes them from token ids alone, without the input embeddings
        or secondary streams a family gave them. Each request stays within
        context_length tokens (check_request), so the pool must hold
        max_running_requests such requests at once.
        """
        if self.context_length is None:
            raise ValueError("sglang_engine_kwargs must bound context_length")
        info = self.engine.get_server_info()
        pool = info["max_total_num_tokens"]
        running = info["internal_states"][0]["effective_max_running_requests_per_dp"]
        page = info.get("page_size") or 1
        width = -(-self.context_length // page) * page
        print(
            f"SGLang KV pool: {pool} tokens for at most {running} running requests "
            f"of up to {self.context_length} tokens",
            flush=True,
        )
        if running * width > pool:
            raise ValueError(
                f"{running} running requests of context_length {self.context_length} "
                f"need {running * width} KV tokens but the pool holds {pool}: lower "
                f"max_running_requests to {pool // width} or context_length, so "
                "that SGLang never retracts a request"
            )

    def check_request(self, prompt_tokens, max_new_tokens):
        """Fail unless the longest request fits context_length. SGLang would
        reject a longer one, or silently shorten an embedded prompt's budget."""
        # SGLang keeps prompt + max_new_tokens below context_length - 1.
        if prompt_tokens + max_new_tokens > self.context_length - 2:
            raise ValueError(
                f"A prompt of {prompt_tokens} tokens and {max_new_tokens} new "
                f"tokens exceed context_length {self.context_length} - 2"
            )

    def update(self, model_path):
        if self.released:
            # The resumed pages hold no weights until the load below.
            self.engine.resume_memory_occupation()
            self.released = False
        # Every export holds the complete policy; SGLang streams it into the
        # model's own load_weights and flushes the idle engine's caches.
        success, message, _ = self.engine.update_weights_from_disk(str(model_path))
        if not success:
            raise RuntimeError(f"SGLang could not load {model_path}: {message}")

    def weight_update(self, method, args, kwargs):
        """Expose Miles' existing SGLang control calls to the socket client."""
        from .weight_transport import HASH_KIND

        receipt = self.directory / "sources.weights.json"
        if method == "begin":
            (identity,) = args
            if getattr(self, "expected_weights", None) is not None:
                raise ValueError("A native weight update is already in progress")
            if identity["hash_kind"] != HASH_KIND:
                raise ValueError("Unknown native weight digest protocol")
            if self.memory_saver:
                # Miles onloads weights before publication and KV/graphs only
                # afterwards. A full receive buffer must fit beside the model.
                if self.released:
                    self.engine.resume_memory_occupation(tags=["weights"])
                else:
                    self.engine.release_memory_occupation(
                        tags=["kv_cache", "cuda_graph"]
                    )
                self.publication_kv_released = True
                self.released = False
            receipt.unlink(missing_ok=True)
            self.expected_weights = dict(identity)
            return {}
        if method == "finish":
            import json

            (identity,) = args
            if (
                identity != self.expected_weights
                or json.loads(receipt.read_text()) != identity
            ):
                raise ValueError("Rollout acknowledgment differs from verified tensors")
            if self.publication_kv_released:
                # SG's per-update flush keeps allocator blocks by default.
                # Return the completed full receive buffer before remapping KV.
                result = self.engine.flush_cache()
                if not result.success:
                    raise RuntimeError(
                        f"SGLang kept its receive buffer: {result.message}"
                    )
                self.engine.resume_memory_occupation(tags=["kv_cache", "cuda_graph"])
                self.publication_kv_released = False
            self.expected_weights = None
            return dict(identity)
        if method not in (
            "init_weights_update_group",
            "destroy_weights_update_group",
            "update_weights_from_tensor",
            "update_weights_from_distributed",
        ):
            raise ValueError(f"Unknown Miles weight operation {method}")
        kwargs = dict(kwargs)
        if kwargs.pop("selector", "all") != "all":
            raise ValueError("Native weight transfer requires the complete policy")
        if method == "update_weights_from_tensor":
            from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

            kwargs["flush_cache"] = True
            request = UpdateWeightsFromTensorReqInput(**kwargs)
            result = self.engine.loop.run_until_complete(
                self.engine.tokenizer_manager.update_weights_from_tensor(request, None)
            )
        else:
            if "dtypes" in kwargs:
                kwargs["dtypes"] = [
                    str(t).removeprefix("torch.") for t in kwargs["dtypes"]
                ]
                kwargs["shapes"] = [list(shape) for shape in kwargs["shapes"]]
            result = getattr(self.engine, method)(*args, **kwargs)
        success, message = result
        if not success:
            raise RuntimeError(f"SGLang weight update failed: {message}")
        return {"success": success, "message": message}

    def release(self):
        # Waits until the scheduler is idle, then empties its allocator cache;
        # without the memory saver the weights and static KV pool stay.
        result = self.engine.loop.run_until_complete(
            self.engine.tokenizer_manager.flush_cache(timeout_s=60)
        )
        if not result.success:
            raise RuntimeError(f"SGLang kept its batch memory: {result.message}")
        if self.memory_saver:
            # The weights and KV pool go too: the next update resumes them
            # and loads a complete policy, so nothing needs a CPU backup.
            self.engine.release_memory_occupation()
            self.released = True

    def close(self):
        self.engine.shutdown()
