"""Run the architecture's actual inference engine in an isolated interpreter."""

import argparse
import json
from pathlib import Path

from .config import ESPnetRLConfig
from .rollout import sha256
from .utils import resolve_callable


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "model", "requests", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    config = ESPnetRLConfig(**json.loads(Path(args.config).read_text()))
    config.validate()
    import torch

    if config.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sources = json.loads(Path(args.requests).read_text())
    if config.rollout_backend == "sglang":
        callback = resolve_callable(config.sglang_generate_factory)
    else:
        callback = resolve_callable(config.vllm_generate_factory)
    generations = callback(config, args.model, sources, args.step)
    result = {
        "policy_step": args.step,
        "native_sha256": sha256(Path(args.model).parent / "native.pt"),
        "rollout_backend": config.rollout_backend,
        "generations": generations,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

