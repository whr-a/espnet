"""One recipe interface for native ESPnet training with Miles algorithms."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import ESPnetRLConfig

OVERRIDES = ("manifest", "eval_manifest", "output_dir", "checkpoint", "train_config")
RAY_SELECTORS = ("ray_address", "train_node", "rollout_node")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    for name in (*OVERRIDES, "data_admission", *RAY_SELECTORS):
        result.add_argument("--" + name.replace("_", "-"))
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--worker", choices=("local", "ray"), help=argparse.SUPPRESS)
    return result


def effective_config(args):
    config = ESPnetRLConfig.load(args.config)
    for name in OVERRIDES:
        if (value := getattr(args, name)) is not None:
            setattr(config, name, value)
    config.validate()
    return config


def execution_backend(config, args):
    selectors = any(getattr(args, name) for name in RAY_SELECTORS)
    use_ray = config.execution_mode != "sync" or (
        config.training_objective == "west_grpo" and selectors
    )
    if use_ray:
        for name in RAY_SELECTORS:
            if not getattr(args, name):
                raise ValueError(f"Ray training requires --{name.replace('_', '-')}")
    elif selectors:
        raise ValueError("Ray node selectors require a disaggregated configuration")
    return "ray" if use_ray else "local"


def launch_command(args, *, python=sys.executable, config=None):
    """Keep platform environment cleanup and device selection in one place."""
    config = effective_config(args) if config is None else config
    backend = execution_backend(config, args)
    command, environment = [python, "-m"], dict(os.environ)
    if backend == "local" and not args.validate_only:
        # A platform pod's distributed group is not the recipe's torchrun group.
        for name in (
            "RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT "
            "MILES_SUPERVISOR_SUBPROCESS_INDEX MILES_CELL_INDEX MILES_POD_INDEX "
            "MILES_BASE_GPU_ID MILES_K8S_NAMESPACE MILES_K8S_RELEASE"
        ).split():
            environment.pop(name, None)
        if config.train_ranks > 1:
            environment["CUDA_VISIBLE_DEVICES"] = config.train_gpus
            command += [
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={config.train_ranks}",
                "-m",
            ]
    command += ["espnet2.bin.rl_train", "--worker", backend, "--config", args.config]
    for name in (*OVERRIDES, "data_admission", *RAY_SELECTORS):
        if (value := getattr(args, name)) is not None:
            command += ["--" + name.replace("_", "-"), value]
    if args.validate_only:
        command.append("--validate-only")
    return command, environment


def validate_inputs(config, args):
    from .tasks.manifest import read_manifest, verify_data_admission
    from .rollout_worker import engine_class

    sources = read_manifest(config.manifest, config=config)
    engine_class(config)
    if config.miles_source:
        sys.path.insert(0, config.miles_source)
    from .miles_library import verify_miles

    evidence = verify_miles(config)
    admission = (
        verify_data_admission(config, args.data_admission)
        if args.data_admission
        else None
    )
    if len({source["id"] for source in sources}) < config.prompts_per_step:
        raise ValueError(
            f"Training requires at least {config.prompts_per_step} distinct prompts"
        )
    return sources, evidence, admission


def main(argv=None, *, worker=None):
    args = parser().parse_args(argv)
    config = effective_config(args)
    backend = execution_backend(config, args)
    worker = worker or args.worker
    if worker is None and os.environ.get("TORCHELASTIC_RUN_ID"):
        # Preserve the old direct torchrun entry without mistaking a platform
        # pod's WORLD_SIZE alone for an already launched training worker.
        if backend != "local":
            raise ValueError("Launch Ray training once, without torchrun")
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        world_size = int(os.environ.get("WORLD_SIZE", "0"))
        rank = int(os.environ.get("RANK", "-1"))
        if not (
            local_size == config.train_ranks
            and world_size == local_size
            and 0 <= local_rank < local_size
            and 0 <= rank < world_size
        ):
            raise ValueError("Torchrun worker geometry differs from train_gpus")
        worker = "local"
    if worker is not None and worker != backend:
        raise ValueError("Worker backend differs from the effective configuration")
    if worker is None and not args.validate_only:
        from .tasks.manifest import read_manifest

        # Reject incompatible reward/prompt contracts before spawning any ranks.
        read_manifest(config.manifest, config=config)
        command, environment = launch_command(args, config=config)
        subprocess.run(command, env=environment, check=True)
        return
    sources, evidence, admission = validate_inputs(config, args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "source": evidence,
                    "data": admission,
                    "prompt_count": len(sources),
                    "backend": backend,
                }
            )
        )
    elif backend == "ray":
        from .miles.driver import run_ray

        run_ray(config, args, sources, admission)
    else:
        from .trainer import run

        if admission is not None and int(os.environ.get("RANK", "0")) == 0:
            output = Path(config.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "data_admission.json").open("x") as stream:
                stream.write(json.dumps(admission, indent=2) + "\n")
        run(config)


if __name__ == "__main__":
    main()
