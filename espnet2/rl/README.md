# Optional Miles post-training stage

Existing `egs2` recipes opt into `python -m espnet2.bin.rl_train --config CONFIG`.
The new 32-prompt × 8-candidate, five-update dual-mode templates and data preparation
are described in [SHORT_RUN_RECIPES.md](SHORT_RUN_RECIPES.md).
ESPnet owns the native model, waveform preparation, architecture adapter,
optimizer and checkpoints. Miles is an unmodified installed library pinned to
official v0.1.2 (`a22fd824191c64ba7c5d6c531f125406e4ce9cc6`). Its GRPO
reward normalization/returns, PPO clipping and KL functions are called directly.
The training entry refuses a different pin or changed algorithm files.

Install optional Miles support with `cd tools; make miles_rl.done`. The installer
requires an existing compatible Torch environment and keeps the pinned checkout
as an editable library, so the source and revision remain reviewable. It is not
an ESPnet-specific Miles fork. PyPI distribution metadata intentionally does not
contain VCS dependency URLs; optional source libraries follow `tools/installers`.

SGLang runs in a separate interpreter configured with `rollout_python` and
`sglang_source`. The user's fork is https://github.com/whr-a/sglang. Each recipe
declares its architecture's generation factory, native scorer and exact exporter.
`make sglang_rl.done` accepts `SGLANG_SOURCE` and `SGLANG_REVISION` for a reviewed
checkout; use a dedicated environment matching its reviewed dependency profile
in [ENVIRONMENT.md](ENVIRONMENT.md).
An inference environment may be different from the native training environment.

`prepare_kaldi_manifest` accepts `wav.scp` and `text`, joins by utterance ID,
materializes recipe pipe commands, converts mono/sample rate and records hashes.
QA recipes additionally supply questions and labels through their architecture
preparer. TTS rewards consume the actual complete generation record, including
codec actions and waveform metadata; forced stream padding remains masked.

Each training update exports immutable native and inference checkpoints, starts
a fresh real SGLang process, saves actual actions/behavior logprobs, validates
native action probabilities, then performs a native AdamW update. Exports,
requests, raw responses, rewards, masks and optimizer state are retained. Use a
new output directory for every run. This initial implementation does not resume
partially completed runs or silently switch to another inference backend.

The initial correctness implementation is full precision, TP=1, synchronous,
with full prompt prefill. Model-specific full-prefix implementations deliberately
trade throughput for exact native semantics; they are not optimized KV-cache
implementations. Two successful updates establish an engineering loop, not a
claim of general training quality. Held-out evaluation must retain its own
manifest, raw responses and explicit denominator.

To install the inference fork in its dedicated environment, run the installer
with that environment's Python (the Makefile otherwise uses active ESPnet Python):

```bash
python3 -m venv /path/to/sglang-env
SGLANG_SOURCE=/path/to/sglang SGLANG_REVISION=EXACT_REVIEWED_COMMIT \
  PYTHON=/path/to/sglang-env/bin/python bash tools/installers/install_sglang_rl.sh
```

Create this environment without `--system-site-packages`. An existing system
SGLang installation can shadow the editable fork even after pip reports success;
the installer rejects that case by checking the actual import path. Install the
checkout's dependencies in the dedicated environment before freezing a run.

The editable installation checks the actual imported module and frozen API/model
files. Record the SGLang source pin alongside each recipe config; model registration
bridges import their implementations from the corresponding `espnet2` RL directory.
`--manifest`, `--checkpoint`, `--train-config` and `--output-dir` on `rl_train`
can select prepared inputs without rewriting the template YAML. Relative template
paths are resolved from the recipe working directory.
