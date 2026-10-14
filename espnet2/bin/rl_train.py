"""Optional native ESPnet RL stage powered by the pinned Miles library."""

import argparse
import sys

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
    if config.miles_source:
        sys.path.insert(0, config.miles_source)
    from espnet2.rl.miles_library import verify_miles

    verify_miles(config)
    if args.validate_only:
        from espnet2.rl.data import read_manifest

        sources = read_manifest(config.manifest)
        print(f"Validated pinned Miles, configuration and {len(sources)} records")
    else:
        from espnet2.rl.trainer import run

        run(config)


if __name__ == "__main__":
    main()
