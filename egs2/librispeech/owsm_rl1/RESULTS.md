# OWSM Miles RL acceptance (2026-10-14)

The existing OWSM v4 stage 18 completed two actual GRPO updates using the
pinned Miles library and the SGLang execution bridge. All evidence is under
`/mnt/project/haoran/terminal_docs/espnet_miles_rl_20261014/runs/owsm_stage18_002/`.

- Native checkpoint: `espnet/owsm_v4_base_102M`, original ESPnet weights.
- Input: LibriSpeech `train.clean.100`, four raw utterances from speaker 374,
  passed as `wav.scp` and `text`; the stage generated its own audited manifest.
- Fixed Miles: `a22fd824191c64ba7c5d6c531f125406e4ce9cc6`.
- SGLang fork base: `4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63`, with
  `sglang.srt.models.espnet_owsm` importing the ESPnet implementation.
- Runtime: Python 3.12, torch 2.11.0+cu130, Transformers 5.12.1;
  SGLang FlashInfer 0.6.15.post1 used its official JIT dependency path.
- SSH73 H100 allocation: actor GPU 0, reference GPU 1, rollout GPU 2.
- Two updates, two prompts per update, four completions per prompt: 16 actual
  SGLang completions. FP32, no radix cache, full prefill, no CUDA graph.

| Check | Update 1 | Update 2 |
| --- | ---: | ---: |
| Mean sampled WER | 30.9659% | 27.5000% |
| Native versus SGLang maximum token logprob error | 0.00006831 | 0.00014472 |
| Gradient norm | 2.5704 | 2.0004 |
| Changed encoder tensors | 238 | 238 |
| Changed decoder tensors | 161 | 161 |
| Adam state tensors | 399 | 399 |
| Adam step | 1 | 2 |

Both updates passed the 0.03 logprob tolerance. Audits compared every exported
weight against its native snapshot, every rollout snapshot against the exact
previous actor version, and every native checkpoint against its model wrapper.
Both optimizer states contain nonzero first moments. Checkpoints are
`recipe/train/step-000001{.pt,.native.pth}` and
`recipe/train/step-000002{.pt,.native.pth}`. Training exited 0.

Heldout evaluation used four disjoint `validation.clean` utterances from
speaker 2277, 67 reference words, with native ESPnet greedy decoding. Before
and after both had 7 word errors: **10.4478% corpus WER**. This small engineering
run establishes end-to-end correctness; it does not demonstrate a heldout
quality improvement. Heldout logs and full hypotheses are under
`runs/owsm_heldout_001/{before.json,after.json}`; evaluation exited 0.

Earlier failed development runs remain intact: `owsm_sglang_smoke_001`
(spawn entrypoint guard), `002` (stale optional FlashInfer cubin), `003`
(missing ESPnet dependency) and `owsm_stage18_001` (non-executable existing
recipe script invoked directly). None was substituted for the successful
run. `owsm_sglang_smoke_004` passed actual generation and a separate native
probability audit with maximum error 0.00003433.

Additional independent checks used the current ESPnet integration source
`c2595666dd0192397a0550a3ce53cacacca85f52`:

- `runs/owsm_vllm_smoke_001` completed actual vLLM sampling, two LibriSpeech
  prompts and two candidates each, using the same v4 base native checkpoint.
  All four sampled probability traces passed independent native scoring;
  maximum error was 0.00015926. vLLM source was pinned to
  `ae2948726949282ecfa8af331615ffbad7e9bacc`. This verifies the optional
  vLLM inference path; the two RL updates above used SGLang.
- `runs/owsm_reference_audit_001` independently reloaded the original native
  reference checkpoint on CPU and recomputed all 16 stored training traces.
  Maximum probability error was 0.00013733. Every reference tensor matched
  the initial checkpoint before and after scoring.
- `runs/owsm_v3_1_sglang_smoke_001` completed actual SGLang sampling for the
  released v3.1 E-Branchformer small checkpoint, two held-out LibriSpeech
  utterances and one candidate each. Independent native probability error
  was at most 0.00008917. This is an older-family compatibility check;
  v3.1 did not receive RL updates in this check.

These runs all exited 0, and their raw responses and complete probability
audits remain in the respective directories. The generic native S2T path
can represent the autoregressive OWSM v1/v2/v3/v4 architectures. This round
has actual SGLang evidence for v4 base and v3.1 small; it does not claim that
every released size has been retrained or re-evaluated. OWSM-CTC remains a
separate non-autoregressive policy and is rejected by this adapter.

Optional library installation was also executed in a fresh SSH73 venv sharing
its existing compatible Torch stack. `optional_install_validation/miles_004`
exited 0 after installing Miles 0.1.2 and checking the actual imported algorithm
module paths, source hashes and fixed Git revision. The installer accepts Git
worktrees and verifies concrete modules even when the top-level package is an
editable namespace. Earlier install/verifier errors are retained separately.

The explicit vLLM export/request path passed
`runs/owsm_vllm_export_validation_001/audit.json`: all 404 exported tensors were
identical to the original native weights and four real encoder-decoder requests
preserved the native prefix. This was a CPU contract check, not a new vLLM
training acceptance. The default accepted training backend remains SGLang.
