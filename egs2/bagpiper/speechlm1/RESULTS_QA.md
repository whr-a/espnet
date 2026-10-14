# Bagpiper QA engineering evidence (2026-10-14)

The independently frozen `bagpiper-qa-stage6-006` completed four optimizer
updates and all 32 training candidates at 20:03:27 UTC, with `Succeeded` / exit
zero. Its saved AVQA reward-component audit records exact-answer accuracy
**1/32**, answer format **6/32**, and think format **27/32**. The complete
initial-reference audit and strict reuse audit of all 32 before responses have
succeeded. The full native/Adam/export/previous-actor checkpoint audit and fresh
MMAU after evaluation remain `Running`; final engineering acceptance is pending.
Training reward components are separate from official held-out MMAU accuracy
and do not establish QA quality improvement. Earlier trajectories below remain
preserved as historical results.

The default-format integration run `bagpiper-qa-stage6-004` exited zero after
two updates, but it **did not pass the QA learning-signal acceptance gate**.
All sixteen rewards, advantages and gradients were zero. Complete audits
confirmed that neither update changed any native model tensor. The 402 Adam
states advanced to steps one and two while every first moment remained zero.

The run used the prior strictly inverse-mapped general SFT native checkpoint,
not a newly downloaded official checkpoint. Its original checkpoint SHA256 was
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
The before official MMAU result remains 0/32. The fresh after evaluation is
still `Running` and targets all 32 candidates from the actual step-four
checkpoint with the same effective engine seed 41. The complete four-update
checkpoint audit is also `Running`; parameter/Adam/export/previous-actor integrity
and the final engineering gate have not yet passed acceptance. Training exit
zero and nonzero gradient norms do not replace these independent checks.

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
- `runs/bagpiper-qa-checkpoint-audit-006/` (audit still running)
- `runs/bagpiper-qa-mmau-after-006/` (fresh evaluation still running)
- `evidence/qa006_launcher_postrun_independent_review.json`
- `evidence/qa006_registered_protection_audit.json`

The AVQA training and local MMAU held-out audio hashes are disjoint; available
registered protection metadata also has no match. Complete connected-component
mappings for the general benchmark registry are unavailable locally, so this
does not establish complete general benchmark contamination admission. Accuracy
and format components remain separate. Format learning alone does not establish
QA quality improvement.
