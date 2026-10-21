"""Build the OWSM Miles stage configuration from a Kaldi data directory."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp")
    parser.add_argument("--text")
    parser.add_argument("--manifest")
    parser.add_argument("--eval-manifest")
    parser.add_argument("--data-admission")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--reference-device")
    parser.add_argument("--train-gpus")
    parser.add_argument("--rollout-gpus")
    parser.add_argument("--rollout-python")
    parser.add_argument("--sglang-source")
    parser.add_argument("--miles-source")
    parser.add_argument("--language")
    args = parser.parse_args()
    from espnet2.rl.prepare import prepare_kaldi_manifest

    work = Path(args.work_dir).resolve()
    config = yaml.safe_load(Path(args.config).read_text())
    work.mkdir(parents=True, exist_ok=True)
    prepared = args.manifest or (
        config.get("manifest") if args.data_admission else None
    )
    if prepared:
        manifest = Path(prepared).resolve()
    else:
        if not args.wav_scp or not args.text:
            parser.error("Provide --manifest or both --wav-scp and --text")
        manifest = work / "train.jsonl"
        prepare_kaldi_manifest(args.wav_scp, args.text, manifest, task="asr_wer")
    config.update(
        train_config=str(Path(args.train_config).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()),
        manifest=str(manifest),
        output_dir=str(work / "train"),
    )
    # Single-GPU defaults: the reference policy stays on CPU and the rollout
    # engine shares the one training device. A recipe yaml or these options
    # override any of it.
    defaults = {
        "steps": 100,
        "device": "cuda:0",
        "reference_device": "cpu",
        "train_gpus": "0",
        "rollout_gpus": "0",
        "rollout_python": "python",
        "sglang_source": "",
        "miles_source": "",
        "language": "<eng>",
    }
    for field, default in defaults.items():
        value = getattr(args, field)
        if value is not None:
            config[field] = value
        elif field not in config:
            config[field] = default
    if args.eval_manifest:
        config["eval_manifest"] = str(Path(args.eval_manifest).resolve())
    from espnet2.rl.config import ESPnetRLConfig

    native = ESPnetRLConfig(**config)
    native.validate()
    if args.data_admission:
        from espnet2.rl.data import verify_data_admission

        verify_data_admission(native, args.data_admission)
    path = work / "rl_config.yaml"
    with path.open("x") as stream:
        stream.write(yaml.safe_dump(config, sort_keys=False))
    print(path)


if __name__ == "__main__":
    main()
