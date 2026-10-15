# Optional Miles post-training stage

Existing `egs2` recipes opt into `python -m espnet2.bin.rl_launch --config CONFIG`.
The new 32-prompt × 8-candidate, five-update dual-mode templates and data preparation
are described in [SHORT_RUN_RECIPES.md](SHORT_RUN_RECIPES.md).
ESPnet owns the native model, waveform preparation, architecture adapter,
optimizer and checkpoints. Miles is an unmodified source library pinned to
official v0.1.2 (`a22fd824191c64ba7c5d6c531f125406e4ce9cc6`). Its GRPO
reward normalization/returns, PPO clipping and KL functions are called directly.
The training entry refuses a different pin or changed algorithm files.

Synchronous profiles use ESPnet's native loop. `disagg_async` binds ESPnet's
native training actors, rollout executor and checkpoint publication to the
unchanged `train_async.py:train` loop at that same Miles pin. The binding retains
the original function code and replaces its four orchestration factories; it
does not use Miles' optional Megatron or other model backends.

For the 32×8 profiles, use the independent hash-locked source environment in
[`ENVIRONMENT.md`](ENVIRONMENT.md): Miles
`a22fd824191c64ba7c5d6c531f125406e4ce9cc6` and SGLang
`283d68c8bff309a42476c95f7c71dad895d09da6`. Its installer checks full runtime
imports and all 233 dependency artifacts, with Miles loaded from its unchanged
checkout rather than installed with every optional backend dependency.

For the earlier synchronous library profile, install optional Miles support
with `cd tools; make miles_rl.done`. The installer
requires an existing compatible Torch environment and keeps the pinned checkout
as an editable library, so the source and revision remain reviewable. It is not
an ESPnet-specific Miles fork. PyPI distribution metadata intentionally does not
contain VCS dependency URLs; optional source libraries follow `tools/installers`.

SGLang runs in a separate interpreter configured with `rollout_python` and
`sglang_source`. The user's fork is https://github.com/whr-a/sglang. Each recipe
declares its architecture's rollout engine (`sglang_engine: module:Class`, an
`espnet2.rl.engine.RolloutEngine`), native scorer and exact exporter.
`make sglang_rl.done` accepts `SGLANG_SOURCE` and `SGLANG_REVISION` for a reviewed
checkout; use a dedicated environment matching its reviewed dependency profile
in [ENVIRONMENT.md](ENVIRONMENT.md).
An inference environment may be different from the native training environment.

`prepare_kaldi_manifest` accepts `wav.scp` and `text`, joins by utterance ID,
materializes recipe pipe commands, converts mono/sample rate and records hashes.
QA recipes additionally supply questions and labels through their architecture
preparer. TTS rewards consume the actual complete generation record, including
codec actions and waveform metadata; forced stream padding remains masked.

Each rollout GPU runs one inference process for the whole run, which builds
the real SGLang engine from the first export. Each training update exports
immutable native and inference checkpoints, loads that export into every engine
in place, saves actual actions/behavior logprobs, validates native action
probabilities, then performs a native AdamW update. Exports, requests, raw
responses, rewards, masks and optimizer state are retained. Use a new output
directory for every run. This initial implementation does not resume partially
completed runs or silently switch to another inference backend.

Multi-rank trainers create their ESPnet Gloo object group collectively before
rank zero starts rollout. Rollout payloads, native scoring results, behavior-load
errors and reward preparation share that group. `distributed_timeout_minutes`
defaults to a finite 4320 minutes (72 hours), covering the collocate runtime
budget and the 48-hour disaggregated async budget. The default training group
uses the same timeout. The runtime supervisor enforces each frozen run's
own deadline and health checks. Set this field before freezing a new run when a
different runtime budget is required.

The correctness profiles use full precision, independent TP=1 rollout replicas
and full prompt prefill. Collocate uses four native training ranks and four
rollout GPUs on one node; async uses eight training ranks and eight rollout GPUs
on distinct nodes. The selected YAML fixes the layout and global batch.
Model-specific full-prefix implementations deliberately
trade throughput for exact native semantics. Opus uses its native decoder
cache. This delivery tree also enables a request-local audio feature cache and
HF DynamicCache for Bagpiper, with bounded state and a full-prefix path when
disabled or beyond the frame bound. That cache has CPU equivalence checks;
its independent GPU comparison and native probability audit remain pending.
The frozen source004 production runs use the earlier uncached Bagpiper path.
Neither layout alone establishes optimal throughput. Short runs establish an
engineering loop; held-out evaluation retains its own manifest, raw responses
and explicit denominator.

For an earlier reviewed inference-only environment, run the optional installer
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
