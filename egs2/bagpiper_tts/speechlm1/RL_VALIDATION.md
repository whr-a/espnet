# Bagpiper TTS RL validation

The optional stage in the existing recipe completed an end-to-end engineering
run: prepared LibriTTS input, real SGLang actions, decoded audio, independent
ASR-WER reward, Miles GRPO loss, optimizer updates, checkpoint export and an
independent evaluation of the final native checkpoint. Run 005 completed on
2026-10-14; all submitted TTS processes are terminal.

## Frozen run

- ESPnet training revision: `cc2b3ac27` (current-master integration derivative).
- Miles library: `a22fd824191c64ba7c5d6c531f125406e4ce9cc6`.
- SGLang thin-registration fork: `01d8735a745e0d33efc00ec95149ac1e5a4d52a8`.
- Official native model: `espnet/bagpiper-tts-sft` revision
  `675e2fafccc7dd7205fad6f8fdc4451f9ee6f768`.
- Data: two LibriTTS test-clean training utterances and two disjoint held-out
  utterances, prepared through `wav.scp` and `text`.
- Sampling: group size 2, unit temperature, native recorded support, 1,536-token
  budget. FP32, TP=1; native actor/reference/rollout on separate GPUs;
  CPU AdamW and 32-token score recomputation blocks.

The real recipe entry was `run.sh --stage 4 --wav-scp ... --text ...` with a
frozen RL configuration. Tokenizer, original multimodal configuration order,
audio processor and codec geometry match the released model. All four native
preprocessor prefixes matched the prepared token prefixes exactly.

## Training and checkpoint checks

The two-update run terminated successfully. Update 1 used negative-WER rewards
`[-0.25, -0.25, 0, -0.142857]`, nonzero GRPO advantages (maximum 0.7070998),
and gradient norm 1.2103044. Update 2 had equal rewards within both groups,
zero GRPO advantages and a fixed-KL update (gradient norm 0.00316743).

A separate read-only audit confirmed 400 changed trainable tensors and 400 Adam
states at each update, optimizer step counters 1 and 2, finite nonzero moments,
979 unchanged frozen audio/codec tensors, exact actor synchronization and all
1,381 inference-export tensors equal to their native checkpoints. Independent
recalculation against the unchanged initial reference had zero error for both
batches. Behavior/native log-probability maximum errors were 0.00031662 and
0.00034142, below the frozen 0.03 tolerance.

The original stricter postrun rule requiring nonzero advantages at every update
failed on update 2 and remains recorded as a failure. The successful engineering
audit reports that zero-advantage case explicitly. This was one reward-driven
GRPO update followed by one KL update; no extra optimizer steps were added.

## Final held-out evaluation

The independent evaluation used ESPnet `737e7510b` and final native checkpoint
SHA256 `8b1a1ec6ab695b08e8e96ca65bf79f1b5e1ded8172c1a3ebced215e34a12abc8`.
Evaluation tag `-1` and config seed 42 produce an effective engine seed of 41.
All four candidates stopped normally within the original budget and generated
5,157 real SGLang actions in total.

| Held-out utterance | Candidate 0 WER | Candidate 1 WER |
| --- | ---: | ---: |
| `1089_134686_000003_000000` | 28.5714% | 28.5714% |
| `1089_134686_000002_000001` | 11.1111% | 11.1111% |

The unweighted mean completion WER was **19.8413%**, with a denominator of four
completions and two prompts. Maximum native/SGLang action log-probability error
was **0.00039673**. The audit checked all 19 export-file hashes and equality of
all 1,381 native/export tensors; training and evaluation identities are disjoint.
All actual eight-stream codec actions decoded strictly into four mono 16 kHz WAV
files. Independent Whisper revision `911407f4214e0e1d82085af863093ec0b66f9cd6`
transcripts, audio hashes, reference text and WER are retained for each candidate.
The terminal audit independently recomputed every WER and matched every raw
action event to its final candidate and audio record.

These are small engineering checks. There is no initial-checkpoint TTS held-out
comparison, and this is not a full LibriTTS result or evidence of a quality gain.
Both training and held-out samples came from LibriTTS test-clean. Their local
utterance identities are disjoint, but the run has not established isolation
from the complete registered general benchmark components. Retain this as an
engineering validation record; new training must check the current registered
held-out media and connected-component protection metadata before freezing.
The first SGLang adapter uses synchronous full-prefix FP32 inference; caching,
throughput and tensor-parallel deployment remain future work.

## Retained validation artifacts

The private run bundle is `espnet_miles_rl_20261014/` in the execution workspace.

- `runs/bagpiper-tts-sg-005/`: metrics, optimizer/native checkpoints and
  `checkpoint_audit_engineering.json`.
- `runs/bagpiper-tts-heldout-005b/`: frozen configuration, requests, raw
  `sglang_events.jsonl`, all responses, rollout, export hashes, WAV/ASR audit,
  `metrics.json` and successful completion record.
- `evidence/bagpiper_tts005b_terminal_audit.json`: independent final audit,
  candidate/audio associations, denominators and artifact hashes.
- `BAGPIPER_REPORT.md` and `BAGPIPER_CHECKPOINT.json`: resumable experiment index,
  including earlier failed runs without rewriting their outcomes.

The score-recomputation oracle matched plain forward/backward at the same
32-token block shape, the full saved 1,146-action backward remained finite,
Miles analytical checks passed, JSON scientific-number parsing passed,
CPU AdamW matched the standard optimizer exactly, and all three independent
evaluator contract tests passed. Earlier modality-order, Whisper-argument,
git-safety and JSON-loader failures remain separate zero-update runs.

Bagpiper AVQA/MMAU is a separate recipe and has its own acceptance record;
this TTS result does not imply that the QA training or all-architecture task
has completed.
