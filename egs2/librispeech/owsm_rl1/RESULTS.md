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
