"""Bind native ESPnet components to the unchanged pinned Miles async loop.

Miles currently exposes orchestration factories as module globals. A private
function namespace supplies ESPnet implementations without changing the module,
its function, or any Miles source. The exact loop code and its safe-point weight
update ordering remain Miles' implementation. This boundary is pinned and audited.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import FunctionType, SimpleNamespace

from ..miles_library import verify_miles


def load_async_loop(config, components):
    evidence = verify_miles(config, runtime=True)
    root = Path(evidence["source"])
    path = root / "train_async.py"
    if config.execution_mode in ("sync", "disagg_sync"):
        if config.training_objective != "west_grpo":
            raise ValueError("The synchronous Ray adapter requires WEST GRPO")

        async def driver(args, *, disposer):
            return await run_synchronous_loop(args, components, disposer=disposer)

        evidence["runtime"] = {
            "entrypoint": "espnet2.rl.miles.driver:run_synchronous_loop",
            "source_modified": False,
            "original_function_modified": False,
            "same_code_object": False,
            "orchestration": "ESPnet synchronous adapter; pinned Miles library utilities",
        }
        return driver, evidence
    spec = importlib.util.spec_from_file_location("espnet_pinned_miles_async", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factories = {
        "init_orchestration_script": components.initialize,
        "create_rollout_components": components.create_rollout_components,
        "create_training_models": components.create_training_models,
        "update_weights": components.update_weights,
    }
    namespace = {**module.train.__globals__, **factories}
    driver = FunctionType(
        module.train.__code__,
        namespace,
        module.train.__name__,
        module.train.__defaults__,
        module.train.__closure__,
    )
    driver.__kwdefaults__ = module.train.__kwdefaults__
    evidence["runtime"] = {
        "entrypoint": "train_async.py:train",
        "sha256": evidence["async_driver_sha256"],
        "bindings": sorted(factories),
        "source_modified": False,
        "original_function_modified": False,
        "same_code_object": driver.__code__ is module.train.__code__,
    }
    return driver, evidence


async def run_synchronous_loop(args, components, *, disposer):
    """One current-policy rollout and one update, then save and publish.

    WEST's unit on-policy ratio cannot consume a pipelined stale rollout. This
    small adapter reuses native Miles actor/data/cadence contracts while keeping
    all generation, optimizer, checkpoint and evaluation safe points ordered.
    """
    from miles.utils.data import remove_rollout_data_refs
    from miles.utils.misc import should_run_periodic_action

    components.initialize(args, disposer=disposer)
    inference, rollout, per_epoch = await components.create_rollout_components(args)
    disposer.add(inference, rollout)
    actor, critic = await components.create_training_models(args, rollout)
    if critic is not None:
        raise ValueError("The synchronous native GRPO profile has no critic")
    disposer.add(actor)
    start = getattr(components, "start_rollout_id", args.start_rollout_id)
    if not 0 <= start <= args.num_rollout:
        raise ValueError("Restored optimizer cursor is outside the update budget")
    args.start_rollout_id = start
    await components.update_weights(args, actor, rollout, inference)
    for rollout_id in range(start, args.num_rollout):
        try:
            await inference.prepare_rollout(rollout_id)
            data = await rollout.get(rollout_id)
        except Exception as error:
            # No train RPC has started: every learner still holds the last
            # completed optimizer state. Await all ranks' preservation before
            # with_disposer can destroy them, even if one save RPC fails.
            try:
                preservation = await actor.preserve_rollout_failure(
                    rollout_id, f"{type(error).__name__}: {error}"
                )
                error.add_note(
                    "Pre-update checkpoint preservation: "
                    + json.dumps(preservation, allow_nan=False)
                )
            except Exception as checkpoint_error:
                error.add_note(
                    "Checkpoint preservation failed: "
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                )
            raise
        await actor.train(rollout_id, data)
        remove_rollout_data_refs(args, data)
        evaluation_due = bool(
            args.native_eval_interval
            and (rollout_id + 1) % args.native_eval_interval == 0
        )
        if evaluation_due or should_run_periodic_action(
            rollout_id, args.save_interval, per_epoch, args.num_rollout
        ):
            await actor.save_model(
                rollout_id,
                force_sync=evaluation_due or rollout_id + 1 == args.num_rollout,
            )
            await rollout.save(rollout_id)
        # update_weights includes the family's awaited periodic evaluation.
        # No following get() exists until it has completely restored inference.
        await components.update_weights(
            args, actor, rollout, inference, rollout_id=rollout_id
        )


def orchestration_args(config):
    """Expose the explicit native actor geometry to the selected driver."""
    config.validate()
    if (
        config.execution_mode != "disagg_async"
        and config.training_objective != "west_grpo"
    ):
        raise ValueError("Synchronous Ray orchestration requires WEST GRPO")
    return SimpleNamespace(
        colocate=False,
        use_critic=False,
        use_rollout_logprobs=False,
        use_tis=config.execution_mode == "disagg_async",
        keep_old_actor=config.execution_mode == "disagg_async",
        fully_async=False,
        num_rollout=config.steps,
        start_rollout_id=0,
        update_weights_interval=config.update_weights_interval,
        offload_train=False,
        num_critic_only_steps=0,
        check_weight_update_equal=False,
        eval_interval=None,
        skip_eval_before_train=True,
        hf_checkpoint=None,
        save_trigger_sentinel=None,
        save_interval=config.save_interval,
        api_server_port=None,
        mini_ft_controller_enable=False,
        debug_exit_after_rollout=None,
        eval_uses_snapshots=False,
        native_eval_interval=config.periodic_eval_interval,
    )


def resolve_nodes(nodes, train, rollout, *, train_ranks, rollout_replicas):
    """Resolve exact live Ray nodes with enough GPUs for both roles."""

    def resolve(selector, needed):
        matches = [
            node
            for node in nodes
            if node["Alive"]
            and selector in (node["NodeID"], node["NodeManagerAddress"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one live Ray node for {selector}")
        node = matches[0]
        if node["Resources"].get("GPU", 0) < needed:
            raise ValueError(f"Node {selector} exposes fewer than {needed} GPUs")
        return node

    first, second = resolve(train, train_ranks), resolve(rollout, rollout_replicas)
    if first["NodeID"] == second["NodeID"] and first["Resources"].get("GPU", 0) < (
        train_ranks + rollout_replicas
    ):
        raise ValueError("A shared node must expose both roles' GPUs at once")
    return first, second


def run_ray(config, args, sources, admission):
    import ray
    from miles.utils.async_utils import with_disposer

    from .components import NativeComponents

    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output)
    with (output / "frozen_config.json").open("x") as stream:
        stream.write(json.dumps(config.to_dict(), indent=2) + "\n")
    if admission is not None:
        (output / "data_admission.json").write_text(
            json.dumps(admission, indent=2) + "\n"
        )
    for source in sources:
        source["reward_config"] = str(output / "frozen_config.json")
    # Match Miles' trainer default and SGLang's pre-initialization override.
    # Both ranks must use the same NCCL allocation mode for weight transfer.
    nccl_cumem = (
        "1"
        if config.sglang_engine_kwargs.get("enable_symm_mem", False)
        else os.environ.get("NCCL_CUMEM_ENABLE", "0")
    )
    ray.init(
        address=args.ray_address,
        runtime_env={"env_vars": {"NCCL_CUMEM_ENABLE": nccl_cumem}},
    )
    try:
        first, second = resolve_nodes(
            ray.nodes(),
            args.train_node,
            args.rollout_node,
            train_ranks=config.train_ranks,
            rollout_replicas=config.rollout_replicas,
        )
        (output / "placement.json").write_text(
            json.dumps({"train": first, "rollout": second}, indent=2) + "\n"
        )
        components = NativeComponents(
            config, sources, first["NodeID"], second["NodeID"]
        )
        driver, evidence = load_async_loop(config, components)
        (output / "miles_library.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
        asyncio.run(with_disposer(driver, orchestration_args(config)))
    finally:
        ray.shutdown()
