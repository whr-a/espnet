# Optional Miles post-training stage

Existing `egs2` recipes add an optional RL stage through
`python -m espnet2.bin.rl_launch --config CONFIG`. Supervised stages keep their
existing behavior. See the [recipe guide](SHORT_RUN_RECIPES.md),
[architecture diagram](ARCHITECTURE.md) and [optional environment](ENVIRONMENT.md).

This is the local, unpublished `codex/bagpiper-west-h100-fp32-20261016` feature
branch. Import the reviewed contributor bundle and verify its revision/hash as
described in the [source setup](ENVIRONMENT.md#4-obtain-the-reviewed-rl-sources);
an ordinary upstream clone does not yet provide this feature.

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

## Sampling temperature

`temperature` T, 0 < T ≤ 1 (default 1), sets the distribution every action is
sampled from, always over its whole support (`top_p: 1`, `top_k: -1`): the
tempered policy π_T(a) = softmax(z / T)[a], with z the raw logits over the
action's support. RL optimizes π_T; T = 1 is the native policy. The families
in `TEMPERATURE_FAMILIES` (`config.py`) implement the contract below;
validation keeps every other family, and the vLLM backend, whose frozen
`raw_logprobs` precede its temperature, at T = 1.

- **Engine.** Samples every action from π_T and returns log π_T(a) of each
  sampled action (a multi-stream frame: the sum over its sampled streams).
  SGLang returns the log probability after its temperature unless
  `SGLANG_RETURN_ORIGINAL_LOGPROB=1`, which returns log softmax(z), the one
  before it. The scheduler reads the variable when it imports its sampler, so
  every engine sets it before starting SGLang instead of inheriting it. OWSM
  sets it to 0 and lets SGLang sample at T. Bagpiper's model divides every
  stream's logits by T itself; SGLang samples the primary from them at its own
  temperature 1 and, with the variable at 1, returns log_softmax of exactly
  those logits; the model samples and scores the secondary streams the same
  way. Every Bagpiper generation records
  `sampling_contract = {"temperature": T, "top_k": 0}`.
- **Learner.** Every pass computes log_softmax(z / T)[a] of the recorded
  actions: the current (old) scores, the async behavior snapshot's, the
  reference's and the update's forward. The FP32 logits are divided as
  `espnet2.rl.types.tempered` divides them: not at all at T = 1, otherwise by
  a tensor, which matches SGLang's division bit for bit. Gradients flow
  through z / T. Bagpiper scores under the recorded supports and rejects a
  record whose contract names another temperature.
- **KL.** `kl_type` is estimated on the sampled actions from log π_T −
  log π_ref,T, the reference being the frozen initial policy at the same
  temperature: the KL between the two tempered policies.
- **Entropy.** `actor/entropy` and `entropy_coef` use the entropy of π_T over
  the support (a multi-stream frame: summed over its sampled streams).
- **Drift gate.** Compares the learner's and the engine's log π_T of the same
  actions. A logit error ε becomes about ε / T in log π_T; the tolerances are
  unchanged.
- **IcePop.** The importance ratio exp(log π_T(old) − log π_T(rollout)) has
  tempered log probabilities on both sides, so it corrects exactly what
  separates the tempered engine from the tempered learner, and the async lag.
- At T = 1 nothing is divided and every engine returns the log probabilities
  it returned before, so existing recipes' numerics are unchanged.

A greedy decode (temperature 0) samples no π_T: the Bagpiper evaluator verifies
native probabilities only for candidates sampled at the recipe's temperature.
In WEST revision
[`b9a629d68`](https://github.com/wenet-e2e/west/blob/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/west/trainer/grpo_trainer.py#L149),
`_get_per_token_logps` passes raw model logits to
`selective_log_softmax(logits, input_ids)` without a temperature argument,
while `examples/grpo/run.sh` samples at 0.7. That implementation scores the
untempered policy; here every term scores the sampling policy π_T. Matching
WEST's prompt, rule-based reward and sampling settings therefore does not
establish identical objectives or numerical reproduction of its training run.
