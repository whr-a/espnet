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
  --device cuda:0 --reference-device cuda:1 --rollout-gpus 2
```

The same stage is available from the existing OWSM recipe:
`egs2/owsm_v4/s2t1/run.sh --stage 18` with the arguments above. `wav.scp` and
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
