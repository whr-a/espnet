# LibriSpeech OWSM Miles post-training

This recipe adds WER GRPO after an existing autoregressive OWSM checkpoint.
The shared training stage imports a pinned Miles library; all OWSM model code,
checkpoint adaptation and rollout hooks live in `espnet2/s2t/rl/`.

Prepare LibriSpeech with the existing `egs2/librispeech/asr1` data stage, then:

```bash
./run.sh \
  --wav-scp ../asr1/data/train_clean_100/wav.scp \
  --text ../asr1/data/train_clean_100/text \
  --train-config /path/to/owsm/config.yaml \
  --checkpoint /path/to/owsm/checkpoint.pth \
  --work-dir exp/owsm_miles_wer \
  --rollout-python /path/to/optional-sglang-env/bin/python \
  --sglang-source /path/to/pinned-sglang-fork \
  --miles-source /path/to/pinned-miles-library \
  --device cuda:0 --reference-device cuda:1 --rollout-gpus 2
```

The same stage is available from the existing OWSM recipe:
`bash egs2/owsm_v4/s2t1/run.sh --stage 18` with the arguments above. `wav.scp` and
`text` must contain the same utterance IDs. Reference text is required to
compute WER; the audio waveform itself cannot supply a supervised reference.
Preparation creates an audited manifest and training freezes its config.

The first SGLang implementation executes the exact native ESPnet encoder and
recomputes the decoder prefix within the SGLang model runner. SGLang schedules
requests, samples tokens and returns actual action log probabilities. The
opaque audio-source carrier ID is removed before native decoding. TP=1,
FP32, disabled radix cache, full prefill and disabled CUDA graphs are required.
This implementation has lower decoding throughput than the earlier vLLM KV
implementation. OWSM-CTC requires a separate non-autoregressive policy.

Checkpoint synchronization, native versus rollout probability checks,
optimizer updates, native checkpoints and all WER outcomes are retained in
`--work-dir`. A short engineering run establishes pipeline correctness;
quality improvement needs a separately budgeted full training/evaluation run.

Two-update engineering acceptance and heldout results are recorded in
[RESULTS.md](RESULTS.md). For independent native greedy heldout evaluation:

```bash
python -m espnet2.s2t.rl.evaluate --config exp/owsm_miles_wer/rl_config.yaml \
  --checkpoint exp/owsm_miles_wer/train/step-000002.native.pth \
  --manifest /path/to/disjoint-heldout.jsonl --output heldout.json
```

An explicit optional vLLM backend is configured by
`egs2/owsm_v4/s2t1/conf/rl_miles_wer_vllm.yaml`. It uses
`espnet2.s2t.rl.vllm_rollout:generate`; request preparation is a separate
function and exports the native OWSM encoder-decoder format. Set `vllm_source`
and `rollout_python` to the previously validated ESPnet vLLM fork and its
inference environment. The default recipe and 2026-10-14 training acceptance
use SGLang. The optional vLLM adapter also passed real engine generation on two
LibriSpeech prompts with two candidates each; its maximum native action-logprob
error was 0.00015927. This was a backend check, with no vLLM optimizer update.
