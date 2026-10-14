# OpusLM post-training using Miles as a library

This branch extends the existing SpeechLM recipe with optional stage 10. The
released OpusLM / Dialogue model and adapters live under `espnet2/speechlm/rl`;
Miles GRPO, returns, PPO and KL are called through the fixed optional library.
The SGLang fork owns inference models and the input-embedding API extension.

Prepare LibriSpeech `wav.scp` / `text` using its existing data recipe, then:

```bash
python -m espnet2.speechlm.rl.prepare \
  --wav-scp /path/to/librispeech/train/wav.scp --text /path/to/librispeech/train/text \
  --model-assets /path/to/opuslm/assets --family opuslm \
  --dac-config /path/to/dac/config.yaml --dac-checkpoint /path/to/dac/360epoch.pth \
  --xeus-config /path/to/xeus/config.yaml --xeus-checkpoint /path/to/xeus/model.pth \
  --kmeans /path/to/xeus/km_opus_lm.mdl --output data/rl/train.jsonl
./run_rl_opuslm.sh --rl_config conf/rl_opuslm_librispeech.yaml
```

Use the Dialogue config for the 1.7B Llama family. Dataset paths are independent
of the recipe directory, so LibriSpeech is selected by the manifest. Keep the
heldout split separate and set `eval_manifest`. References supervise WER and do
not enter the native audio condition.

`rl/model.py` reuses the exact released HF decoder and original checkpoint names
without depending on an archived ESPnet package. It retains ARDelay padding,
nine-stream embedding sum and head biases. Waveform conditions are generated
with strict original DAC / XEUS weights; every source / artifact / token hash is
recorded. The published DAC alpha was a constant: the current trainable version
is reconstructed as a nonpersistent constant buffer before strict weight loading.

SGLang rollout is real autoregressive generation. The first interface covers
waveform-to-text WER updates; full codec-output TTS RL remains a separate
multi-stream sampler requirement. New native pad-cleared results are distinct
from historical vLLM pad-bias experiments.
