# Bagpiper QA RL results

## Accepted WEST default four-GPU FP32 run (2026-10-16)

The frozen AVQA run completed **100 optimizer updates** and the full paired
**1,000-question MMAU-Mini evaluation**, with pipeline `Succeeded` / exit 0.
Official MMAU accuracy increased from **289/1000 (28.9%)** to
**310/1000 (31.0%)**, a **+2.1 percentage-point** descriptive change. This is
one matched greedy evaluation, without a statistical significance claim.

### Training and numerical checks

The public profile is `conf/rl_west_4gpu_async.yaml`: four H100 80 GB GPUs on
one node, with two data-parallel native learner replicas and two independent
SGLang TP=1 rollout replicas. Every update uses **32 prompts × 8 candidates**:
3,200 training prompt instances and 25,600 sampled candidates in total.
Native SDPA and SGLang Triton use FP32; training uses temperature 0.7, a
1024-token cap, learning rate 1e-6 and KL coefficient 0.04. The objective is the
literal WEST default **exact-string accuracy + strict whole-response format**.
CPU AdamW, reference CPU offload between GPU scoring passes, activation CPU
offload and a 2048-token scoring micro-batch budget limit GPU memory pressure.

All 100 updates have finite metrics and nonzero gradients (pre-clipping norms
0.162415–0.307451). Maximum native/SGLang behavior action-log-probability error
is **0.001337171**, below the unchanged **0.03** gate. All emitted support
mismatch rates are zero. Weight staleness is 0–1, and **both rollout replicas
acknowledged every published version 0–100**. Placement receipts identify two
learner GPUs and two disjoint rollout GPUs on the same Ray node.

Training exact-answer rewards are nonzero for 13,519/25,600 candidates
(52.8086%). **Strict format reward is zero for all 25,600 candidates.** A
single answer block inside a caption/reasoning response does not satisfy the
whole-response format regex. Training-window rewards are diagnostics over
different questions and samples, separate from the paired held-out result.

Full training-phase wall time was **45,846.964 s (12.735 h)**, including
initialization, initial rollout and final publication/shutdown. Excluding the
first timing-warmup update, measured steps averaged 452.051 s: 261.633 s inside
the learner window and 190.419 s outside it. Outside-window time includes
save/export, publication, rollout and scheduling; it is not pure inference
wait. Final 1,000-question evaluation took **646.497 s**.

### Matched MMAU-Mini evaluation

Both phases used the same ordered 1,000 questions (904 distinct audio
identities), WEST prompting and answer extraction, official MMAU matching,
SGLang greedy decoding, 2048 new tokens and a 30-second audio limit. The
benchmark checked matching prompt/prepared-audio identities and non-weight
export assets; each export matched 1,381 native tensors. The final checkpoint
was the actual step-100 policy. The saved baseline was reused after paired
input/export checks; the after phase generated from that final policy. There
were **87 corrected and 66 regressed** questions.

| Official MMAU accuracy | Before | After | Change |
| --- | ---: | ---: | ---: |
| All questions | 289/1000 (28.90%) | 310/1000 (31.00%) | +2.10 pp |
| Music | 97/334 (29.04%) | 93/334 (27.84%) | -1.20 pp |
| Sound | 90/333 (27.03%) | 113/333 (33.93%) | +6.91 pp |
| Speech | 102/333 (30.63%) | 104/333 (31.23%) | +0.60 pp |

The following are separate diagnostics on those same greedy evaluation
outputs, not additional primary metrics or the sampled AVQA training rewards:

| Diagnostic | Before | After |
| --- | ---: | ---: |
| V1 answer extraction + official matcher | 340/1000 | 372/1000 |
| WEST exact-string training-answer check | 52/1000 | 97/1000 |
| Strict whole-response format | 0/1000 | 0/1000 |
| Single answer block anywhere | 143/1000 | 216/1000 |
| Truncated responses | 25/1000 | 27/1000 |

### Provenance and limits

Initialization was a **prepared native checkpoint reconstructed from public
`anonymous-release/vLLM_alm` revision
`83e02343f5647599c444b8ec91084b54576507f6`**, plus the deterministic native
`vocab_weight` buffer. Its recorded initial SHA256 is
`790bb1d4fdaafd3cce3d44eb40b9e77a989d322c6e4dc17b0172c62b3d2610fa`.
It is not verified as tensor-equivalent to the official `espnet/bagpiper-sft`
native release, and its originating training stage and absence of earlier RL
are not established. The README's
[optional reconstruction instructions](README.md#reconstruct-the-recorded-public-hf-initialization-optional)
document the public source and metadata normalization; they do not prove
exact replay or official-SFT equivalence.

Training used frozen ESPnet
`bd432d61b41a319e5ce1e001cd4b93df86b5ad79`; evaluation used frozen ESPnet
`a93eb740112572d8808fd39cd1e233cf76d028e6`. Both used unchanged Miles
`a22fd824191c64ba7c5d6c531f125406e4ce9cc6` and SGLang
`283d68c8bff309a42476c95f7c71dad895d09da6`. The public release draft contains
later CPU-reviewed integration and portability changes; this frozen run is
not a GPU acceptance of every change in that draft. Retained experiment
receipts include `run_006/pipeline_state.json`, `result/TRAINING_100.json` and
`result/paired_quality.json`, with timing, source, checkpoint, input and
publication bindings.

This accepts the **four-GPU FP32** path. Earlier BF16 probes failed the
unchanged numerical gate, including an HF-rounding/FP32-head alignment
attempt; those failures remain failures. One-GPU collocation and two-GPU
Bagpiper disaggregation remain unvalidated. Historical SFT may include MMAU
Mini-derived material, so this is not an SFT-clean blind test. Historical
73.1% used native SFT prompting, vLLM and 4096 tokens, so it is not the baseline
for this WEST-prompt, SGLang, 2048-token comparison.

## Historical evidence (2026-10-14 snapshot)

The following preserves the earlier snapshot. Its unfinished audits were
recorded as running on that date; these are historical observations, not
current process states or the status of the accepted four-GPU run above.

The independently frozen `bagpiper-qa-stage6-006` completed four optimizer
updates and all 32 training candidates at 20:03:27 UTC, with `Succeeded` / exit
zero. Its saved AVQA reward-component audit records exact-answer accuracy
**1/32**, answer format **6/32**, and think format **27/32**. The complete
initial-reference audit and strict reuse audit of all 32 before responses have
succeeded. At that snapshot, the full native/Adam/export/previous-actor
checkpoint audit and fresh MMAU after evaluation were recorded as `Running`;
that historical run had not yet passed final engineering acceptance.
Training reward components are separate from official held-out MMAU accuracy
and do not establish QA quality improvement. Earlier trajectories below remain
preserved as historical results.

The default-format integration run `bagpiper-qa-stage6-004` exited zero after
two updates, but it **did not pass the QA learning-signal acceptance gate**.
All sixteen rewards, advantages and gradients were zero. Complete audits
confirmed that neither update changed any native model tensor. The 402 Adam
states advanced to steps one and two while every first moment remained zero.

The run used a prepared native checkpoint reconstructed from public
`anonymous-release/vLLM_alm` revision `83e02343f5647599c444b8ec91084b54576507f6`,
not a verified official SFT checkpoint. Its original checkpoint SHA256 was
`790bb1d4fdaafd3cce3d44eb40b9e77a989d322c6e4dc17b0172c62b3d2610fa`.
The actual entrypoint was the existing `run.sh --stage 6`, with four AVQA
utterances, two prompts per update and four candidates. All eight originally
prepared AVQA rows remain in the frozen manifest; this two-update run consumed
the first four in order. SGLang was the actual sampler, using a full native
prefix and FP32, with a 1536-token cap, TP=1, and 32-row score recomputation.

| Audit | Update 1 | Update 2 |
| --- | ---: | ---: |
| Sampled candidates | 8 | 8 |
| Maximum native/SGLang action logprob error | 0.00014353 | 0.00017548 |
| Changed native tensors | 0 | 0 |
| Gradient norm | 0 | 0 |
| Nonzero Adam first-moment entries | 0 | 0 |

Every model snapshot and exported tensor matched its native version exactly;
the next rollout used the exact preceding actor version. An independently
reloaded initial reference scored all sixteen candidates with zero error
relative to the stored reference probabilities, and remained tensor-wise
unchanged. These are integrity checks, not successful QA learning.

The input adapter was compared against the real `SpeechLMPreprocessor` on all
eight AVQA and eight MMAU samples. Every native token, audio feature and feature
insertion index matched. This fixed the missing audio end-of-turn/user markers
and zero continuous-audio placeholders from the earlier run. It did not change
the frozen default-format reward. The checkpoint still generated caption and
reasoning prose, and often an incorrect choice or an answer format failing the
strict whole-output match.

Independent held-out MMAU evaluation completed all **8 prompts × 4 candidates =
32 responses** with the official token-set matcher. Before: **0/32**. After:
**0/32**. Since both actor updates left every model tensor unchanged, the after
check explicitly reused the full before responses and independently scored all
32 final-native action traces; it did not claim new after generation. Maximum
action logprob error was 0.00014877. File hashes differ because the native
checkpoint was serialized again; the tensor-wise equality is the reuse basis.

Evidence directory:
`/mnt/project/haoran/terminal_docs/espnet_miles_rl_20261014/runs/`:

- `bagpiper-qa-stage6-004/{freeze.json,train/metrics.jsonl,checkpoint_audit.json,reference_audit.json}`
- `bagpiper-qa-mmau-before-004/eval/{requests.json,responses.json,metrics.json}`
- `bagpiper-qa-mmau-after-reuse-004/metrics.json`
- `bagpiper_qa_native_template_audit_001/audit.json`

Earlier results remain separate: `bagpiper-qa-sg-002` exited zero with zero
learning signal, and `bagpiper-qa-stage6-003` failed during preparation before
training because it attempted to replace a frozen manifest. Neither is a
successful RL acceptance result.

`bagpiper-qa-stage6-005` was an independently frozen WEST think protocol,
planning four updates over all eight AVQA prompts in order. It **failed on the
first backward with CUDA OOM, before any optimizer update**. The complete first
eight candidates remain saved. Their exact-answer accuracy was zero; the two
groups had rewards `[0, 0.1, 0.1, 0]` and `[0.1, 0.2, 0.1, 0.1]` from answer/think
format terms: one answer block and six think blocks across the eight candidates.
There are 8942 recorded frames; one terminal EOS per candidate is deterministic.
All 8934 sampled actions had finite, nonzero group advantages.
Initial native/reference probabilities matched exactly. There is no final
QA005 checkpoint or after evaluation. Its independent think-template MMAU
before evaluation completed all 32 responses with `Succeeded` and official
MMAU accuracy **0/32**. QA006 reuses those original before responses only after
a separate strict saved-response audit; there was no new before generation.

The generic memory repair was developed outside that failed trajectory. It
skips unused Bagpiper stream projections while retaining current support,
sampling-contract and zero-gradient optimizer semantics. Optional lossless CPU
storage of saved activations preserves parameter/buffer storage on the GPU.
On the longest saved answer, 1431 frames plus 179 prompt tokens, the original
and production scorers produced exactly equal log probabilities, entropies and
all 402 parameter gradients. A separate actual PPO/KL backward over all eight
saved candidates accumulated finite gradients with every native parameter
unchanged and zero optimizer steps. Peak allocated memory was
77,508,865,536 bytes. These checks verify the repair, not successful training.
Skipping unused projections alone did not reduce the measured single-candidate
peak. The preflight imported the `e81c25b0e` implementation; its source-file hash
was read at completion after later offline boundary/test edits. The loaded-source
note preserves that distinction. The final generic/all-forced boundary changes
do not trigger on this active Bagpiper batch. Nine repository tests passed in
Torch 2.11, including the CUDA parameter-storage-view guard.

`bagpiper-qa-stage6-006` started on 2026-10-14 at 14:32:26 UTC from a fresh
frozen ESPnet `fb9d9117e68ffc10a2d4491d57b7591969fe98f8`, published SGLang
`4c3f163fe54634f20ef846e1ec0c507b27ae2430`, and unchanged Miles pin. It retains
four updates, all eight AVQA prompts, group size four, FP32, a 1536-token cap,
32-row scoring and lossless activation CPU storage. Training finished with
`Succeeded` / exit zero on 2026-10-14 at 20:03:27 UTC: eight prompts consumed
once in order, 32 candidates, four optimizer updates, and all four native/full
checkpoints saved. Gradient norms were `[0.592595, 0.375921, 0.591220, 0.605122]`.
The independent saved training-component audit found exact-answer accuracy
1/32, one answer block in 6/32, and one think block in 27/32, under the frozen
WEST think reward `0.8*accuracy + 0.1*answer-format + 0.1*think-format`. These
counts describe all saved training candidates, not official MMAU scoring.

`bagpiper-qa-reference-audit-006` succeeded on all 32 candidates: independently
reloaded initial-reference logprob error was zero and reference tensors remained
unchanged. `bagpiper-qa-before-saved-audit-006` also succeeded on all 32 original
QA005 before responses, with 1381 native/export tensors exactly equal and maximum
native action logprob error 0.000156641; `new_before_generation` is false.
The before official MMAU result at this snapshot was 0/32. The fresh after
evaluation was recorded as `Running`, targeting all 32 candidates from the
actual step-four checkpoint with the same effective engine seed 41. The
complete four-update checkpoint audit was also recorded as `Running`:
parameter/Adam/export/previous-actor integrity and the final engineering gate
had not yet passed acceptance at that observation. Training exit zero and
nonzero gradient norms do not replace these independent checks.

Additional evidence under the same task directory:

- `evidence/qa005_terminal_saved_batch_root_audit.json`
- `evidence/bagpiper_memory_fix_independent_acceptance_001.json`
- `evidence/bagpiper_memory_production_gradient_comparison_001.json`
- `runs/bagpiper-qa-full-batch-preflight-001/result.json`
- `runs/bagpiper-qa-full-batch-preflight-001-loaded_source_note.json`
- `evidence/bagpiper_memory_repo_tests_001.json`
- `runs/bagpiper-qa-stage6-006/{freeze.json,state.json}`
- `evidence/qa006_saved_train_reward_components_001.json`
- `runs/bagpiper-qa-reference-audit-006/`
- `runs/bagpiper-qa-before-saved-audit-006/`
- `runs/bagpiper-qa-checkpoint-audit-006/` (audit running at the snapshot)
- `runs/bagpiper-qa-mmau-after-006/` (evaluation running at the snapshot)
- `evidence/qa006_launcher_postrun_independent_review.json`
- `evidence/qa006_registered_protection_audit.json`

The AVQA training and local MMAU held-out audio hashes are disjoint; available
registered protection metadata also has no match. Complete connected-component
mappings for the general benchmark registry are unavailable locally, so this
does not establish complete general benchmark contamination admission. Accuracy
and format components remain separate. Format learning alone does not establish
QA quality improvement.
