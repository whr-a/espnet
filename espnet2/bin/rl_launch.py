"""Launch recipe RL with its configured synchronous or Miles async geometry."""

import argparse
import os
import subprocess
import sys

from espnet2.rl.config import ESPnetRLConfig


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    for name in (
        "manifest",
        "eval-manifest",
        "output-dir",
        "checkpoint",
        "train-config",
        "data-admission",
        "ray-address",
        "train-node",
        "rollout-node",
    ):
        result.add_argument(f"--{name}")
    result.add_argument("--validate-only", action="store_true")
    return result


def launch_command(args, *, python=sys.executable):
    """Resolve the real CLI; validation and execution use the same config."""
    config = ESPnetRLConfig.load(args.config)
    for field in (
        "manifest",
        "eval_manifest",
        "output_dir",
        "checkpoint",
        "train_config",
    ):
        if getattr(args, field) is not None:
            setattr(config, field, getattr(args, field))
    config.validate()
    command = [python, "-m"]
    environment = dict(os.environ)
    if config.execution_mode == "disagg_async":
        for name in ("ray_address", "train_node", "rollout_node", "data_admission"):
            if not getattr(args, name):
                raise ValueError(f"disagg_async requires --{name.replace('_', '-')}")
        command.append("espnet2.bin.rl_train_async")
    else:
        if any(
            getattr(args, name)
            for name in ("ray_address", "train_node", "rollout_node")
        ):
            raise ValueError("Ray node selectors require a disagg_async config")
        if config.prompts_per_step == 32 and config.group_size == 8:
            from espnet2.rl.short_run import validate_collocate_geometry

            validate_collocate_geometry(config, running=False)
            if not args.data_admission:
                raise ValueError("The 32×8 collocate recipe requires --data-admission")
            if not args.validate_only:
                if int(environment.get("WORLD_SIZE", "1")) != 1:
                    raise ValueError(
                        "Invoke rl_launch once outside an existing distributed launcher"
                    )
                for name in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "MILES_SUPERVISOR_SUBPROCESS_INDEX",
                    "MILES_CELL_INDEX",
                    "MILES_POD_INDEX",
                    "MILES_BASE_GPU_ID",
                    "MILES_K8S_NAMESPACE",
                    "MILES_K8S_RELEASE",
                ):
                    environment.pop(name, None)
                environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                command.extend(
                    [
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node=4",
                        "-m",
                    ]
                )
        command.append("espnet2.bin.rl_train")
    command.extend(["--config", args.config])
    for name in (
        "manifest",
        "eval_manifest",
        "output_dir",
        "checkpoint",
        "train_config",
        "data_admission",
        "ray_address",
        "train_node",
        "rollout_node",
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), value])
    if args.validate_only:
        command.append("--validate-only")
    return command, environment


def main():
    args = parser().parse_args()
    command, environment = launch_command(args)
    effective = ESPnetRLConfig.load(args.config)
    for field in (
        "manifest",
        "eval_manifest",
        "output_dir",
        "checkpoint",
        "train_config",
    ):
        if getattr(args, field) is not None:
            setattr(effective, field, getattr(args, field))
    if args.data_admission:
        from espnet2.rl.short_run import verify_data_admission

        verify_data_admission(effective, args.data_admission)
    subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
