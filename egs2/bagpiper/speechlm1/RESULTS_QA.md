# Bagpiper QA engineering evidence (2026-10-14)

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

`bagpiper-qa-stage6-005` is a new independently frozen WEST think protocol, with
four updates covering all eight AVQA prompts in order. It has its own matching
think-template MMAU before/after evaluation. Its results must be reported
separately, with accuracy and format components distinguished. A format reward
increase alone does not establish QA quality improvement.
