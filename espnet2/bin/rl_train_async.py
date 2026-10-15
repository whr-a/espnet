"""Run the pinned Miles async loop on two existing 8-GPU Ray nodes."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import read_manifest
from espnet2.rl.short_run import validate_short_run, verify_data_admission


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
        required=True,
        help="Frozen data admission JSON with manifest hashes and full component checks",
    )
    result.add_argument("--validate-only", action="store_true")
    return result


def resolve_nodes(nodes, train, rollout):
    def resolve(selector):
        matches = [
            node
            for node in nodes
            if node["Alive"]
            and selector in (node["NodeID"], node["NodeManagerAddress"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one live Ray node for {selector}")
        node = matches[0]
        if node["Resources"].get("GPU", 0) != 8:
            raise ValueError("Each requested node must expose exactly 8 GPUs")
        return node

    first, second = resolve(train), resolve(rollout)
    if first["NodeID"] == second["NodeID"]:
        raise ValueError("The async profile requires two distinct nodes")
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
    validate_short_run(config)
    if config.miles_source:
        sys.path.insert(0, config.miles_source)
    from espnet2.rl.miles_library import verify_miles

    evidence = verify_miles(config)
    admission = verify_data_admission(config, args.data_admission)
    sources = read_manifest(config.manifest)
    if len({source["id"] for source in sources}) < 32:
        raise ValueError("Training requires at least 32 independent prompts")
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
    (output / "data_admission.json").write_text(json.dumps(admission, indent=2) + "\n")
    for source in sources:
        source["reward_config"] = str(frozen)
    ray.init(address=args.ray_address)
    try:
        first, second = resolve_nodes(ray.nodes(), args.train_node, args.rollout_node)
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
