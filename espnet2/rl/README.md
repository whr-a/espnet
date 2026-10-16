# Optional Miles post-training stage

Existing `egs2` recipes add an optional RL stage through
`python -m espnet2.bin.rl_launch --config CONFIG`. Supervised stages keep their
existing behavior. See the [recipe guide](SHORT_RUN_RECIPES.md),
[architecture diagram](ARCHITECTURE.md) and [optional environment](ENVIRONMENT.md).

ESPnet owns data preparation, native speech models, rollout adapters, AdamW
and checkpoints. It calls unchanged Miles v0.1.2 at
`a22fd824191c64ba7c5d6c531f125406e4ce9cc6` for GRPO normalization/returns,
PPO clipping, KL and IcePop. Startup verifies the source identity.

Synchronous training uses ESPnet's native loop. `disagg_async` also reuses that
pin's original `train_async.py:train` function code, binding four orchestration
factories to ESPnet Ray actors. It overlaps one rollout batch with training,
with weight staleness at most one. It does not use Miles' Megatron backend or
its unrestricted fully asynchronous mode.

## Resource profiles

Public recipes focus on one H100 for collocated training and inference, or two
H100s for disaggregated async. A two-GPU Ray node can host both roles; two
one-GPU nodes can host them separately. Supplied profiles use at most four
GPUs. Large models also need substantial CPU RAM for the reference policy,
FP32 master weights and Adam moments. Plain OpusLM starts with a two-GPU
profile. Bagpiper's new one/two-GPU full loop still needs GPU validation.

`train_gpus` declares one training rank per entry; `rollout_gpus` declares one
SGLang replica per entry. Synchronous one-GPU profiles share device 0 between
phases. Async roles on the same host require disjoint devices; Ray records
and checks their actual assignment. Optional data parallelism is described in
[PARALLEL.md](PARALLEL.md).

## Runtime and model boundary

Use the source-pinned environment in [ENVIRONMENT.md](ENVIRONMENT.md). Miles
is exposed from its unchanged checkout without unused model backends. SGLang
may run in a separate interpreter selected by `rollout_python`; its registration
bridges import model code from ESPnet. Each family supplies a native policy,
exact exporter and `sglang_engine: module:Class` implementing `RolloutEngine`.

One rollout subprocess per GPU stays alive for the run. The learner publishes
immutable exports, which the engine loads between generation batches. After
successful publication, all but the newest `keep_exports` (default two) are
removed. Actual actions, behavior logprobs, masks and rewards are recorded;
native scoring checks the generating policy before the update. A stale async
batch is checked by temporarily loading its generating weights from a CPU
snapshot. Bagpiper additionally records per-frame multi-stream sampling support.

OWSM retains native encoder/decoder semantics under the SGLang scheduler and
sampler. Bagpiper uses SGLang Qwen3 layers with paged KV; OpusLM uses Olmo2 or
Llama layers with native multi-stream prompt embeddings and text output. Their
KV pools must fit active requests without retraction. OWSM retains an explicit
vLLM option. Model-specific code stays in the existing architecture directories;
Miles is not patched.

## Inputs and outputs

ASR/TTS preparation joins `wav.scp` and `text` by utterance ID, materializes
recipe pipe commands, normalizes waveform format and records hashes. QA also
requires questions, choices and reference answers. Public preparers support
LibriSpeech, LibriTTS and AVQA. User manifests work without private registries.
`--data-admission` optionally checks a prepared hash-bound data/protection
receipt; frozen project runs retain their original receipts and protections.

Resolve source/interpreter paths in the YAML, then select native model assets
and prepared data with `--checkpoint`, `--train-config`, `--manifest`,
`--eval-manifest` and a fresh `--output-dir`. Logs retain rollout records,
loss/timing metrics, native checkpoints and standard Adam state. Held-out
evaluation has a separate manifest and explicit denominator. Interrupted-run
optimizer recovery is not implemented.

Short-run success establishes an engineering path. It does not establish
quality improvement, inference-backend equivalence or an optimal GPU layout.
