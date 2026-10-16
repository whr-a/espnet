"""Optional native ESPnet RL stage powered by the pinned Miles library."""

import argparse
import json
import os
import sys
from pathlib import Path

from espnet2.rl.config import ESPnetRLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--manifest", help="Prepared training JSONL; overrides the YAML value"
    )
    parser.add_argument("--eval-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-config")
    parser.add_argument("--data-admission")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = ESPnetRLConfig.load(args.config)
    for field in (
        "manifest",
        "eval_manifest",
        "output_dir",
        "checkpoint",
        "train_config",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    config.validate()
    if config.execution_mode != "sync":
        raise ValueError("Use rl_train_async for disagg_async execution")
    from espnet2.rl.data import read_manifest

    sources = read_manifest(config.manifest, config=config)
    admission = None
    if args.data_admission:
        from espnet2.rl.data import verify_data_admission

        admission = verify_data_admission(config, args.data_admission)
    if config.miles_source:
        sys.path.insert(0, config.miles_source)
    from espnet2.rl.miles_library import verify_miles

    verify_miles(config)
    if args.validate_only:
        print(f"Validated pinned Miles, configuration and {len(sources)} records")
    else:
        from espnet2.rl.trainer import run

        if admission is not None and int(os.environ.get("RANK", "0")) == 0:
            output = Path(config.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            path = output / "data_admission.json"
            with path.open("x") as stream:
                stream.write(json.dumps(admission, indent=2) + "\n")
        run(config)


if __name__ == "__main__":
    main()
