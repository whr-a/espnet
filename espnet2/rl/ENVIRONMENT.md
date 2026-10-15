# Native training and SGLang environment profiles

## Native Miles source environment for 32×8 profiles

Use `tools/installers/install_miles_async_env.sh` for a new 32-prompt,
eight-candidate native Miles environment. The checked CPython 3.12 / Linux
x86_64 / CUDA 13.0 profile requires glibc 2.39 or later and fixes these sources:

| Source | Exact revision |
| --- | --- |
| Unchanged Miles v0.1.2 | `a22fd824191c64ba7c5d6c531f125406e4ce9cc6` |
| SGLang ESPnet fork | `283d68c8bff309a42476c95f7c71dad895d09da6` |
| SGLang base before the import compatibility backport | `4c3f163fe54634f20ef846e1ec0c507b27ae2430` |
| Upstream encoder reference for that backport | `4076a68579b4a7ae3e9d2ce2f94dec9917237bab` |

The 233-artifact lock and source pins are in
`tools/installers/miles_async/cp312_linux_x86_64_cu130/profile.json`.
Torch/Torchaudio 2.11.0+cu130, Transformers 5.12.1 and NumPy 2.3.5 retain the
validated native/rollout ABI. Miles router 0.3.2, Polars 1.42.1, PyLaTeXEnc 2.10
and xxhash 3.7.1 support the full eager native Miles imports. The SGLang backport
adds the upstream encoder and import helpers while preserving the original
serving dispatch; it does not enable complete V41 serving.

```bash
git clone https://github.com/radixark/miles.git /path/to/miles
git -C /path/to/miles checkout --detach a22fd824191c64ba7c5d6c531f125406e4ce9cc6
git clone https://github.com/whr-a/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach 283d68c8bff309a42476c95f7c71dad895d09da6
bash tools/installers/install_miles_async_env.sh \
  --python python3.12 --venv /path/to/new-native-miles-env \
  --espnet-source /path/to/espnet \
  --sglang-source /path/to/sglang --miles-source /path/to/miles
```

Supply a clean delivery checkout as `--espnet-source`; the installer records its
actual revision. It refuses an existing venv, verifies exact clean Miles/SGLang
sources and lock hashes, installs every artifact with hashes, and checks
`pip check`, actual imports and four architecture registrations. Editable builds
write package metadata into the supplied ESPnet/SGLang checkouts, so use separate
installation copies when preserving frozen source trees.

Set `miles_source: /path/to/miles`, `sglang_source: /path/to/sglang` and
`rollout_python: /path/to/new-native-miles-env/bin/python` in the selected recipe
YAML, and run the learner with that same Python. Miles is exposed through its
unchanged source checkout; its distribution and unused Megatron, ring-flash,
torchft or Kubernetes extras are not installed. This profile supports ESPnet's
native actors and the original Miles async orchestration loop. Synchronous
profiles retain ESPnet's own native loop and use the same mathematical APIs.

Start dedicated Ray head/worker processes from ordinary shells outside torchrun,
without inherited `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`,
`MASTER_ADDR`, `MASTER_PORT`, group/role identity or `TORCHELASTIC_*` variables.
Invoke the async driver with Python, outside torchrun. The public driver connects
to the supplied Ray cluster; it does not clean or restart existing Ray daemons.
Training actors write rank/master identity only in their own processes and are
bound to a different node from rollout tasks. Independent rollout children clear
the full torchrun identity, including `TORCHELASTIC_USE_AGENT_STORE`, before
starting their own inference rendezvous.

The independent environment passed actual full runtime imports, all 233 artifact
hashes and `pip check`. The source validation also passed real CPU native actor,
bounded noncontiguous collective and TCPStore isolation oracles. Installation
and CPU checks do not establish Ray/NCCL, GPU model execution, two-node transport,
five optimizer updates, throughput or held-out quality; each new run needs those
separate checks. Add `--with-tests` for the separately locked pytest dependencies.

The following earlier synchronous profile retains its own 229-artifact lock and
SGLang `4c3f163fe54634f20ef846e1ec0c507b27ae2430` pin. Do not substitute that
pin into the new native Miles installer or infer async acceptance from the
earlier synchronous results.

## Earlier synchronous and inference profiles

Native ESPnet and rollout can use separate Python interpreters. The existing
2026-10-14 H100 interpreters used Python 3.12.3 with these observed profiles:

| Package | Existing native interpreter | Original SGLang interpreter |
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
test. A separate clean environment is validated below; it does not replace
the frozen GPU environments in this table. Accepted Bagpiper TTS training used
the existing native interpreter. OWSM `owsm_stage18_002` used Torch 2.11.0+cu130
and Transformers 5.12.1 in the original SGLang interpreter for both its native
learner and rollout; its independent native
greedy before/after evaluation also used that interpreter. Bagpiper QA005
ran both its native learner and rollout with the same original SGLang profile:
Torch 2.11.0+cu130 from `sglang_system_dependencies` and Transformers 5.12.1.
It failed at its first full backward with zero optimizer updates; its longer
audio-conditioned sequences have no successful-gradient acceptance from the
TTS Torch 2.13 run. The independently frozen QA006 native learner also used
Torch 2.11.0+cu130 and Transformers 5.12.1, completing four optimizer updates;
its complete checkpoint audit and final evaluation remain pending.
Keep each run's actual interpreter and package paths with its frozen
configuration rather than assuming every family used one profile.
Native Torch 2.13 and Torchaudio 2.11 came from the existing host environment;
this records the actual pair rather than recommending a new unmatched pair.
The rollout environment inherited a declared dependency overlay excluding old
SGLang and editable Miles imports. New Transformers/FlashInfer packages took
precedence over older overlay metadata. Actual module paths and source hashes
were checked in addition to distribution versions.

Miles is the unmodified library at
`a22fd824191c64ba7c5d6c531f125406e4ce9cc6`. ESPnet directly calls its GRPO
normalization/returns, PPO clipping and KL mathematical APIs. For these earlier
synchronous runs, ESPnet owns the native training loop, AdamW, checkpoints and
SGLang rollout orchestration. Their installation profile does not establish
the new Miles async runtime. The
SGLang fork is
`https://github.com/whr-a/sglang` at
`4c3f163fe54634f20ef846e1ec0c507b27ae2430`. Its three registration modules
import OWSM, Bagpiper and both OpusLM implementations from this ESPnet checkout.
Architecture code is not duplicated in SGLang.

The historical shared-host interpreters used a separate native environment and
a rollout dependency overlay. Their actual interpreter/package paths remain in
each run's private evidence. These host-specific paths are not portable recipe
defaults; use the clean installation instructions for another environment.

For a new rollout environment, the optional installer provides a hash-locked
CPython 3.12 / Linux x86_64 / CUDA 13.0 profile. It includes all dependencies
from the fixed SGLang core and current ESPnet base, plus Liger and Joblib.
The 229 exact dependency artifacts and SHA256 hashes are in
`tools/installers/sglang_rl/cp312_linux_x86_64_cu130/requirements.txt`.
SGLang and ESPnet are then installed as editables with `--no-deps`, so their
installation does not select different runtime versions.

The fresh local `clean-rl-install-001` environment passed on 2026-10-14 at
22:15:24 UTC: no dependency overlay or system site packages; `pip check`
reported no broken requirements; Python Engine import and all four model
registrations resolved to the ESPnet implementations. NumPy 2.3.5,
Torch/Torchaudio 2.11.0+cu130, Numba 0.65.1, SoundFile, Librosa and Torchaudio
resampling/MelSpectrogram/Kaldi fbank passed CPU numeric checks. The source
pins were ESPnet `296fefe8096ee347569b74f911e0f8ce68a27c9e` and SGLang
`4c3f163fe54634f20ef846e1ec0c507b27ae2430`. This proves the clean CPU
environment profile; no Engine/model instance, GPU kernel, new rollout
probability or RL experiment was executed in that environment.

The same public installer completed an independent clean installation on the
VA2 H100 host in `va2-clean-rl-install-003` at 23:32:52 UTC on 2026-10-14.
Its separate source checkouts were ESPnet
`5212e56c65fc9ca9b5a803b33dd78819bd890b0f` and SGLang
`4c3f163fe54634f20ef846e1ec0c507b27ae2430`. The isolated venv used no dependency
overlay or system site packages, with Torch/Torchaudio 2.11.0+cu130,
Transformers 5.12.1 and NumPy 2.3.5; `pip check` passed.

The subsequent independent `owsm-clean-sglang-001` completed at 23:48:03 UTC.
It generated two LibriSpeech prompts with two candidates each through the real
SGLang Engine: four responses and 78 sampled actions. Independent native scoring
verified every sampled action, with maximum logprob error 0.0000343323 against
the 0.03 tolerance. No optimizer update or held-out quality comparison was
performed. Bagpiper, Plain OpusLM and Dialogue OpusLM passed imports and model
registry identity checks in this clean environment; their GPU model instances
were not executed in this validation. The original installation state,
environment audit, generation summary and native probability audit remain in
the private experiment evidence.

The installer creates a new venv and refuses an existing path. It checks
source dependency metadata against the profile before creating the environment.
Supply non-frozen source checkouts: editable package builds write local metadata
into them. The documented SGLang `SGLANG_BUILD_RUST_EXTS=none` mode is the
default for this Python profile; grpc/multimodal Rust extensions were not
built or checked. Only ANTLR 4.9.3 builds a pure Python dependency wheel:
Hydra requires 4.9.*, whose public releases have no wheel. All other dependency
artifacts remain binary and hash checked. From an ESPnet checkout with matching
base dependencies:

```bash
git clone https://github.com/whr-a/sglang.git /path/to/sglang
git -C /path/to/sglang checkout --detach 4c3f163fe54634f20ef846e1ec0c507b27ae2430
bash tools/installers/install_sglang_rl_env.sh \
  --python python3.12 --venv /path/to/new-sglang-env \
  --espnet-source /path/to/espnet --sglang-source /path/to/sglang
```

Configure the recipe's `rollout_python` with `/path/to/new-sglang-env/bin/python`
and `sglang_source` with `/path/to/sglang`. Keep Miles in the compatible native
training environment:

```bash
PYTHON=/path/to/native-env/bin/python MILES_SOURCE=/path/to/miles \
  bash tools/installers/install_miles_rl.sh
```

The clean rollout profile does not install the Miles distribution or its full
distributed runtime. Its Ray 2.58.0 and Safetensors 0.8.0 satisfy those fixed
Miles dependency bounds; the original native algorithm library is validated
separately. Full `espnet[speechlm]` SFT extras still conflict on
Transformers 5.14.1 versus SGLang 5.12.1 and Tilelang >=0.1.12 versus 0.1.11.
Keep the SFT training profile separate.

For an already reviewed compatible rollout environment, the original optional
library installer remains available. From this ESPnet checkout:

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
shadowing. The earlier full editable installer and four-class registry checks
passed with a fresh venv plus the existing safe dependency overlay. That
historical GPU reproduction path did not pass a whole-environment `pip check`.
The overlay's optional Flash-QLA package reported Tilelang/TVM-FFI conflicts;
the checked RL imports and inference runs did not use it. The separate clean
CPU profile above now supplies installation and `pip check` evidence without
the overlay.

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
