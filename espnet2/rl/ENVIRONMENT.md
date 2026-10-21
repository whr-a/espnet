# Install the native ESPnet/Miles RL environment

This is the single installation guide for collocated and disaggregated native
RL. Training, task preparation and evaluation are in [README.md](README.md).
Use a reviewed integration checkout: this local feature is not yet supplied by
an ordinary upstream ESPnet clone. Record its actual Git revision and the
source/dependency receipts when reproducing a result.

## Prerequisites

| Component | Locked profile |
| --- | --- |
| Host | Linux x86_64, glibc >=2.39 |
| Python | CPython 3.12 with `venv` |
| Torch/Torchaudio | 2.11.0 + CUDA 13.0 |
| Transformers | 5.12.1 |
| Ray | 2.58.0 |
| Source libraries | Unchanged Miles `a22fd824`; SGLang fork `283d68c8` |
| GPU target | H100 80 GB; one collocated or two disaggregated |

Use an existing CUDA 13-compatible NVIDIA driver. GPU/kernel operation must be
checked independently of installation. Host tools include Git, a C/C++ build
toolchain, pkg-config, libsndfile, SoX, ffmpeg and FLAC. The installer uses the
reviewed binary lock, allowing only its specified pure-Python source packages;
optional SFT extensions or first-use GPU kernels may need additional tools.

The complete profile and artifact hashes live in
[`tools/installers/miles_async/cp312_linux_x86_64_cu130`](../../tools/installers/miles_async/cp312_linux_x86_64_cu130).
This historical directory/script name also covers synchronous native RL.
Do not combine it with an old SGLang-only lock or unpinned library installer.

## Install from reviewed sources

Choose a clean integration checkout and a **nonexistent** environment path.
The commands below install software only; they do not start training or submit
resources. Replace the example paths before running them.

```bash
export ESPNET_RL_SOURCE=/absolute/path/to/reviewed-espnet-integration
export ESPNET_MILES_SOURCE=/absolute/path/to/miles
export ESPNET_SGLANG_SOURCE=/absolute/path/to/sglang
export ESPNET_RL_ENV=/absolute/path/to/new-rl-venv

git -C "$ESPNET_RL_SOURCE" rev-parse HEAD
git clone https://github.com/radixark/miles.git "$ESPNET_MILES_SOURCE"
git -C "$ESPNET_MILES_SOURCE" checkout --detach \
  a22fd824191c64ba7c5d6c531f125406e4ce9cc6
git clone https://github.com/whr-a/sglang.git "$ESPNET_SGLANG_SOURCE"
git -C "$ESPNET_SGLANG_SOURCE" checkout --detach \
  283d68c8bff309a42476c95f7c71dad895d09da6

bash "$ESPNET_RL_SOURCE/tools/installers/install_miles_async_env.sh" \
  --python python3.12 --venv "$ESPNET_RL_ENV" \
  --espnet-source "$ESPNET_RL_SOURCE" \
  --sglang-source "$ESPNET_SGLANG_SOURCE" \
  --miles-source "$ESPNET_MILES_SOURCE" --with-tests

. "$ESPNET_RL_ENV/bin/activate"
export PYTHONPATH="$ESPNET_MILES_SOURCE"
python -m pip check
```

The installer rejects dirty sources, wrong pins/platforms and an existing venv.
It hides GPUs during installation, checks hash-bound artifacts, installs ESPnet
and SGLang source editables, and verifies actual source imports/model registry
identities. Miles is imported from its unchanged source checkout; unused
Megatron/ring-flash/Kubernetes distribution extras are not installed. Its
receipt is written to `$ESPNET_RL_ENV.acceptance.json`.

`--with-tests` adds locked test dependencies; it does not execute the test suite.
Review the receipt before separately validating model execution. Keep the same
reviewed sources/interpreter on Ray workers and expose Miles on `PYTHONPATH`
before starting a dedicated Ray cluster. Set `miles_source`, `sglang_source`
and `rollout_python` in the selected configuration. The library does not
install or manage a cluster daemon, remote serving deployment or GPU allocation.

## Keep supervised training separate

Ordinary ESPnet supervised training remains available through its existing
[installation guide](../../doc/installation.md). Its SpeechLM extra requests
Transformers 5.14.1 and TileLang >=0.1.12, whereas this RL profile locks 5.12.1
and 0.1.11. Use separate environments/checkouts and transfer matching native
checkpoints/assets; do not install SFT extras over the locked RL environment.

For example, in a separate ordinary ESPnet checkout:

```bash
cd /absolute/path/to/espnet-sft/tools
./setup_venv.sh "$(command -v python3.12)" /absolute/path/to/sft-venv
. ./activate_python.sh
python -m pip install 'torch==2.11.0' 'torchaudio==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu130
cd ..
python -m pip install -e '.[train,asr,tts]' 'torch==2.11.0' 'torchaudio==2.11.0'
python -m pip check
```

For SpeechLM, use another clean environment and `.[train,speechlm]` instead.
Follow its model/codec/tokenizer requirements and ordinary recipe data stages.
The RL installer does not prepare a checkpoint or download task datasets.

## Resource planning

A one-GPU configuration is a placement target, not proof that a smaller GPU or
any chosen prompt length fits. The reviewed Bagpiper QA profiles passed
five-update engineering checks on one and two H100s, CPU checkpoint audits and
matched bounded greedy evaluations of 32 items per checkpoint. Token-limit
truncations remain in the full denominator; these checks do not establish a
quality gain, full MMAU results or long-run stability. OWSM and Bagpiper TTS GPU
regression on this refactor remains pending.

CPU offload saves GPU memory at the cost of host memory and transfers. For the
reviewed Bagpiper model's 8,268,432,384 trainable elements, full FP32 native
weights occupy about 33.8 GiB and trainable gradients add 30.8 GiB, before
activations and workspaces. This backend's CPU AdamW stores an FP32 shadow plus
two FP32 moments, requiring 92.41 GiB on its CPU optimizer owner.

`reference_device: cuda:0` with `reference_cpu_offload: true` computes reference
scores on the learner GPU and returns its weights to CPU storage before actor
backward. Budget for that reference, asynchronous behavior snapshots, transfer
buffers, file cache and temporary state. Multiple learners retain their own
model/reference; the owner optimizer is not replicated. Keep the original SFT
checkpoint and assets available to rebuild the fixed reference on resume.

The one-GPU `miles_shm` profile needs about 33.2 GiB of free `/dev/shm` for this
model's complete FP32 inference payload, including frozen audio weights and
excluding the reward-only codec and loss-weight buffer. CPU staging and
flattening also need host RAM. The two-GPU profile uses NCCL and a separate
8-GiB Ray object store; it does not stage the full policy in CPU shared memory.
Allow for other shared-memory users in either layout. A universal 128/192-GB
host minimum is not validated.

Plan storage for model/data assets, environments, immutable rollout exports and
full milestones. One historical BF16 Bagpiper native/full-checkpoint pair was
135.60 GB. The accepted two-GPU FP32 step-5 pair occupies 171,975,011,467 bytes
(171.98 GB); it includes both the full optimizer state and native sidecar.
`keep_exports` prunes rollout exports, not all full checkpoints. Measure the
actual workload before promising a resource budget or throughput.

Installation verifies software identity and imports. Acceptance for a new model
or configuration additionally needs actual generation, complete 32 × 8 batches,
finite native gradients, probability/support checks, weight publication,
checkpoint/optimizer evidence and an independent held-out evaluation. Private
cluster policy/admission, experiment protection registries and observer scripts
remain external; the public training framework does not bundle them.
