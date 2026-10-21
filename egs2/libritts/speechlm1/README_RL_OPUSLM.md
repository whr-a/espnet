# OpusLM native RL

Optional SpeechLM stage 10 uses the unchanged Miles library and persistent
SGLang inference. Native model/scoring/export/preparation live in
`espnet2/speechlm/rl`; model support remains in ESPnet. Use the shared
[RL guide](../../../espnet2/rl/README.md) and
[environment setup](../../../espnet2/rl/ENVIRONMENT.md).

| Architecture | Main configuration | GPUs |
| --- | --- | --- |
| Dialogue | `conf/rl_opuslm_dialogue_librispeech_1gpu.yaml` | One, collocated |
| Dialogue | `conf/rl_opuslm_dialogue_librispeech_2gpu_async.yaml` | Two, learner/rollout |
| Plain | `conf/rl_opuslm_librispeech_2gpu_async.yaml` | Two, learner/rollout |

Profiles retain 32 prompts × 8 candidates and five updates. Shorter historical
names remain symlinks. Plain has no accepted one-GPU profile. These templates
are not a claim that every layout has completed GPU or quality acceptance.

Prepare independent LibriSpeech train/dev splits with its ordinary data recipe,
then encode waveform conditions using the matching released DAC/XEUS/kmeans
assets. Select `opuslm_dialogue` for Dialogue:

```bash
python -m espnet2.speechlm.rl.prepare --family opuslm \
  --wav-scp /path/to/train/wav.scp --text /path/to/train/text \
  --model-assets /path/to/hf_assets --dac-config /path/to/dac.yaml \
  --dac-checkpoint /path/to/dac.pth --xeus-config /path/to/xeus.yaml \
  --xeus-checkpoint /path/to/xeus.pth --kmeans /path/to/km.mdl \
  --device cpu --output data/rl/train.jsonl
```

Record the preparation device: CPU/GPU codec quantization can differ. Retain
native nine-stream conditions, ARDelay padding, released decoder geometry and
asset hashes. The existing entry covers waveform-to-text WER, not full
codec-output TTS RL.

Resolve assets/source paths in the selected YAML, then call
`python -m espnet2.bin.rl_train --config CONFIG` with the train/dev manifests,
native checkpoint/config and fresh output directory. Two-GPU async adds Ray
node selectors. Existing `run_rl_opuslm.sh`/stage 10 options forward to that entry.
Templates keep native FP32 and CPU Adam; preserve each profile's reference
placement and memory settings.

Evaluate every held-out prompt/group independently:

```bash
python -m espnet2.speechlm.rl.evaluate --config /path/to/run/frozen_config.json \
  --checkpoint /path/to/run/step-000005.native.pth --policy-step 5 \
  --manifest /path/to/dev.jsonl --output-dir /path/to/fresh/evaluation \
  --device cpu --rollout-gpus 1
```

Use the same effective paths/assets as training. The evaluator retains raw
responses, native probability checks, completion-mean WER, corpus WER and
failure counts. References supervise WER and remain outside the audio condition.
Historical small-run results are retained in the experiment archive, not used
as evidence of quality improvement or every small-GPU layout.
