"""Run the pinned Miles async loop on two existing 8-GPU Ray nodes."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import read_manifest, verify_data_admission


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    for name in (
        "manifest",
        "eval-manifest",
        "output-dir",
        "checkpoint",
        "train-config",
    ):
        result.add_argument(f"--{name}")
    result.add_argument("--ray-address", required=True)
    result.add_argument("--train-node", required=True, help="Exact Ray node ID or IP")
    result.add_argument(
        "--rollout-node", required=True, help="Exact second Ray node ID or IP"
    )
    result.add_argument(
        "--data-admission",
        help="Prepared data admission JSON binding manifest and media SHA256 values",
    )
    result.add_argument("--validate-only", action="store_true")
    return result


def resolve_nodes(nodes, train, rollout, *, train_ranks, rollout_replicas):
    """Resolve the Ray nodes for both roles; one node may host both."""

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

    first = resolve(train, train_ranks)
    second = resolve(rollout, rollout_replicas)
    if first["NodeID"] == second["NodeID"] and first["Resources"].get("GPU", 0) < (
        train_ranks + rollout_replicas
    ):
        raise ValueError("A shared node must expose both roles' GPUs at once")
    return first, second


def main():
    args = parser().parse_args()
    config = ESPnetRLConfig.load(args.config)
    for field in (
        "manifest",
        "eval_manifest",
        "output_dir",
        "checkpoint",
        "train_config",
    ):
        if (value := getattr(args, field)) is not None:
            setattr(config, field, value)
    config.validate()
    if config.execution_mode != "disagg_async":
        raise ValueError("This launcher requires execution_mode=disagg_async")
    if config.miles_source:
        sys.path.insert(0, config.miles_source)
    from espnet2.rl.miles_library import verify_miles

    evidence = verify_miles(config)
    admission = (
        verify_data_admission(config, args.data_admission)
        if args.data_admission
        else None
    )
    sources = read_manifest(config.manifest)
    if len({source["id"] for source in sources}) < config.prompts_per_step:
        raise ValueError(
            f"Training requires at least {config.prompts_per_step} distinct prompts"
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "source": evidence,
                    "data": admission,
                    "prompt_count": len(sources),
                    "requested_nodes": [args.train_node, args.rollout_node],
                }
            )
        )
        return
    import ray
    from miles.utils.async_utils import with_disposer

    from espnet2.rl.miles_async_driver import load_async_loop, orchestration_args
    from espnet2.rl.miles_components import NativeComponents

    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_config.json"
    if frozen.exists():
        raise FileExistsError("Use a fresh output_dir for this run")
    config.output_dir = str(output)
    frozen.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    if admission is not None:
        (output / "data_admission.json").write_text(
            json.dumps(admission, indent=2) + "\n"
        )
    for source in sources:
        source["reward_config"] = str(frozen)
    ray.init(address=args.ray_address)
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


if __name__ == "__main__":
    main()
