# Native training and SGLang environment profiles

Native ESPnet and rollout use separate Python interpreters. The actual
2026-10-14 H100 runs used Python 3.12.3 with these imported versions:

| Package | Native training / native evaluation | SGLang rollout |
| --- | --- | --- |
| Torch | 2.13.0+cu130 | 2.11.0+cu130 |
| Torchaudio | 2.11.0+cu130 | 2.11.0+cu130 |
| Transformers | 5.17.0 | 5.12.1 |
| NumPy | 2.3.5 | 1.26.4 |
| Safetensors | 0.8.0 | 0.7.0 |
| Ray | 2.55.1 | 2.55.1 |
| Liger kernel | 0.8.0 | 0.8.4 |
| SGLang kernel | not used | 0.4.5 |
| FlashInfer Python | not used by native RL | 0.6.15.post1 |

These are observed working profiles, not a resolver lock or a clean installation
test. OWSM and accepted TTS used the native profile above. Bagpiper QA005 instead
ran both its native learner and SGLang rollout with the rollout interpreter:
Torch 2.11.0+cu130 from `sglang_system_dependencies` and Transformers 5.12.1.
It failed at its first full backward with zero optimizer updates; its longer
audio-conditioned sequences have no successful-gradient acceptance from the
TTS Torch 2.13 run. Keep each run's actual interpreter and package paths with
its frozen configuration rather than assuming every family used one profile. Native Torch 2.13 and Torchaudio 2.11 came from the existing host environment;
this records the actual pair rather than recommending a new unmatched pair.
The rollout environment inherited a declared dependency overlay excluding old
SGLang and editable Miles imports. New Transformers/FlashInfer packages took
precedence over older overlay metadata. Actual module paths and source hashes
were checked in addition to distribution versions.

Miles is the unmodified library at
`a22fd824191c64ba7c5d6c531f125406e4ce9cc6`. The unified SGLang fork is
`https://github.com/whr-a/sglang` at
`4c3f163fe54634f20ef846e1ec0c507b27ae2430`. Its three registration modules
import OWSM, Bagpiper and both OpusLM implementations from this ESPnet checkout.
Architecture code is not duplicated in SGLang.

On the existing shared host, the tested interpreters remain available at
`/mnt/project/haoran/terminal_docs/espnet_rl_20261008/gpu_venv/bin/python` and
`/mnt/project/haoran/terminal_docs/espnet_miles_rl_20261014/sglang_venv/bin/python`.
The native package files resolve into
`/mnt/project/haoran/terminal_docs/espnet_audio_ab_20260910/.venv`;
rollout Torch resolves into `/usr/local/lib/python3.12/dist-packages` through
the safe overlay, while its Transformers/FlashInfer resolve into `sglang_venv`.
Reusing these interpreters with a pinned ESPnet/SGLang checkout is the tested
host reproduction path; the installation recipe below is for another reviewed
compatible environment.

The minimum optional-library flow is to keep the existing compatible native
ESPnet environment, install Miles there, and install SGLang in a separate
interpreter. From this ESPnet checkout:

```bash
PYTHON=/path/to/native-env/bin/python MILES_SOURCE=/path/to/miles \
  bash tools/installers/install_miles_rl.sh

git clone https://github.com/whr-a/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach 4c3f163fe54634f20ef846e1ec0c507b27ae2430
python3 -m venv /path/to/sglang-env
# Supply the reviewed Torch 2.11+cu130 / Torchaudio environment and dependencies
# for this SGLang checkout before freezing the inference environment.
SGLANG_SOURCE=/path/to/sglang \
  SGLANG_REVISION=4c3f163fe54634f20ef846e1ec0c507b27ae2430 \
  PYTHON=/path/to/sglang-env/bin/python bash tools/installers/install_sglang_rl.sh
/path/to/sglang-env/bin/python -m pip install liger-kernel==0.8.4
```

Make this ESPnet checkout importable in both interpreters, through its existing
installation or explicit `PYTHONPATH`. Configure `rollout_python` and
`sglang_source` in the recipe with absolute paths. The rollout subprocess adds
the current ESPnet checkout and the selected SGLang source itself. Install
architecture preparation dependencies in the native environment: Opus audio
preparation additionally loads SoundFile, Torchaudio, Joblib, original DAC/XEUS
weights, and a matching tokenizer/k-means. Bagpiper's package registration
loads the SpeechLM job module and requires Liger even in the inference environment.

Do not use `--system-site-packages` for a new SGLang venv or put an old SGLang
package before its editable checkout. The installer rejects actual import
shadowing. The full editable installer and four-class registry checks passed
with a fresh venv plus the existing safe dependency overlay. No complete
installation without that overlay or successful whole-environment `pip check`
is claimed. The overlay's optional Flash-QLA package reported Tilelang/TVM-FFI
conflicts; the checked RL imports and inference runs did not use it.

The repository's broad `speechlm` extra currently specifies a different
Transformers/Tilelang profile from the validated rollout above. Installing all
training extras into rollout can change its frozen ABI. Keep native and rollout
profiles separate and validate actual model imports and native/action
probabilities when preparing another environment.

Every YAML is a template requiring released checkpoints, original model config,
matching tokenizer/audio assets, dataset paths and GPU assignments. A template
with `/path/to`, `exp/sft` or `REQUIRED_PIN` values is not a downloaded model or a
preconfigured training environment. Data preparation requires `text` references
for WER; QA also requires question/answer metadata. `wav.scp` alone cannot supply
these labels. Save effective absolute paths in `frozen_config.json` for later
evaluation, since relative paths are resolved from the command's working directory.

Bagpiper QA's 1536-token FP32 templates enable
`adapter_options.activation_cpu_offload: true`. This stores saved activation
boundaries on CPU while keeping parameter/buffer storage on the original GPU.
It needs host RAM in addition to the CPU Adam states; provision a large-memory
host and measure the effective native environment before freezing training.
The validation host had 2 TiB RAM. This is an activation storage option, not a
change to the model, sampling budget, GRPO objective or categorical precision.
TTS retains its independently tested default configuration.

The original QA005 Torch 2.11 FP32 run failed on its first backward before any
optimizer update. The independent longest saved-action oracle covered all 1431
actions plus 179 prompt tokens, preserving 32-row score GEMMs. The original
scorer with lossless activation storage and the production scorer produced
identical action probabilities, entropies and all 402 parameter gradients.
An additional preflight applied the actual PPO/KL backward to every one of the
eight saved candidates and accumulated finite gradients without stepping Adam.
Peak allocated GPU memory was 77,508,865,536 bytes. This verifies that saved
batch, and does not predict peak memory for every possible future response.
The original failure and fresh independent training rounds remain separate.
