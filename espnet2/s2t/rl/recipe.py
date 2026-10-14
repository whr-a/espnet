"""Build the OWSM Miles stage configuration from a Kaldi data directory."""

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--rollout-gpus", default="2")
    parser.add_argument("--rollout-python", default="python")
    parser.add_argument("--sglang-source", default="")
    parser.add_argument("--miles-source", default="")
    parser.add_argument("--language", default="<eng>")
    args = parser.parse_args()
    from espnet2.rl.prepare import prepare_kaldi_manifest

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / "train.jsonl"
    prepare_kaldi_manifest(args.wav_scp, args.text, manifest, task="asr_wer")
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(
        train_config=str(Path(args.train_config).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()),
        manifest=str(manifest),
        output_dir=str(work / "train"),
        steps=args.steps,
        device=args.device,
        reference_device=args.reference_device,
        rollout_gpus=args.rollout_gpus,
        rollout_python=args.rollout_python,
        sglang_source=args.sglang_source,
        miles_source=args.miles_source,
        language=args.language,
    )
    path = work / "rl_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)


if __name__ == "__main__":
    main()
