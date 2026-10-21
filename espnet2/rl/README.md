# Speech reinforcement learning with Miles

ESPnet owns the native speech model, preprocessing, sampled-action scoring,
rewards, optimizer and checkpoints. The unchanged Miles library supplies GRPO
advantages, PPO/KL/importance-weight mathematics and bounded asynchronous
orchestration; SGLang supplies rollout scheduling and model layers. Start with
[environment setup](ENVIRONMENT.md), then use one training command:

```bash
python -m espnet2.bin.rl_train --config /absolute/path/to/recipe.yaml
```

This is a local integration, not an already merged upstream release. The older
`rl_launch` and `rl_train_async` module names are compatibility entries. Existing
supervised recipe stages retain their behavior.

## Architecture and ESPnet2 placement

```text
egs2/<model>/<task>/             recipe stages and model/task configuration
espnet2/bin/rl_train.py          public command
espnet2/rl/cli.py                configuration, validation and execution selection
espnet2/rl/                      native trainer and Miles/rollout adapters
espnet2/s2t/rl/                 OWSM policy, exporter, rollout and WER evaluation
espnet2/speechlm/rl/            OpusLM policy, exporter and ASR evaluation
espnet2/speechlm/model/speechlm/rl/bagpiper/
                               Bagpiper policy, data, QA/TTS reward and evaluation
```

ESPnet2 is the integration point because OWSM and SpeechLM already provide the
native model classes, preprocessors, configuration/checkpoint formats and `egs2`
recipes here. Reusing those components preserves the supervised-to-RL boundary
without porting the models or maintaining a second training model. ESPnet3 is
not a prerequisite for this adapter; a later migration should reuse its public
model and training interfaces rather than create a parallel implementation.

```mermaid
flowchart LR
    C[Recipe and native checkpoint] --> L[ESPnet native learner]
    L -->|publish policy| S[Persistent SGLang rollout]
    S -->|actions, logprobs and masks| R[Task reward and native integrity checks]
    R --> M[Miles advantages and loss mathematics]
    M --> L
    L --> O[Native checkpoint and separate held-out evaluation]
```

The core imports the pinned Miles source, not its Megatron model backend. The
asynchronous adapter binds the original `train_async.py` function to ESPnet
actors; this pin exposes module globals rather than a stable plugin interface.
The adapter and its contract tests must be reviewed when changing that pin.

## Main configurations: one or two H100 GPUs

All main templates use **32 prompts × 8 candidates = 256 responses per update**
and five default updates. Microbatches reduce memory without changing that
batch. Increase `steps` deliberately and provide enough distinct source records.

| Task and recipe directory | One GPU, collocated | Two GPUs, asynchronous | Reward |
| --- | --- | --- | --- |
| OWSM ASR: `egs2/owsm_v4/s2t1` | `conf/rl_miles_wer_1gpu.yaml` | `conf/rl_miles_wer_2gpu_async.yaml` | Negative normalized WER |
| Bagpiper QA: `egs2/bagpiper/speechlm1` | `conf/rl.yaml` | `conf/rl_disagg_async.yaml` | Existing `qa_v1_reward`: accuracy and format, equal weights |
| Bagpiper TTS: `egs2/bagpiper_tts/speechlm1` | `conf/rl.yaml` | `conf/rl_disagg_async.yaml` | Generated audio → frozen ASR → negative WER |

One-GPU `sync` alternates learning and rollout on the same device. Two-GPU
`disagg_async` uses disjoint learner/rollout devices and overlaps one batch,
with at most one version of policy lag. Persistent engines load each published
policy in place. Neither mode creates a cluster allocation or serving group.

Earlier frozen OWSM revisions have complete one-/two-H100 200-update evidence.
This refactor needs a separate GPU regression before inheriting that acceptance.
Bagpiper's one-/two-H100
configurations are templates requiring their own memory, numerical and full-loop
GPU acceptance; four-GPU results do not validate these layouts. No established
held-out quality gain is claimed. Large models also need substantial host RAM
for CPU reference and optimizer state; see [resource planning](ENVIRONMENT.md#resource-planning).

OpusLM remains supported through `egs2/libritts/speechlm1` and its
[task guide](../../egs2/libritts/speechlm1/README_RL_OPUSLM.md). Plain OpusLM's
main template uses two GPUs; no accepted one-GPU profile is claimed. Existing
configuration aliases resolve to the same files, not additional recipes.

## Prepare independent data

Use the selected recipe's ordinary data preparation to obtain real train and
dev splits. ASR joins Kaldi `wav.scp` and `text` by utterance ID, materializes
pipe commands, normalizes audio and records hashes:

```bash
python -m espnet2.rl.prepare --wav-scp data/train/wav.scp \
  --text data/train/text --output data/rl/train.jsonl
python -m espnet2.rl.prepare --wav-scp data/dev/wav.scp \
  --text data/dev/text --output data/rl/dev.jsonl
```

For QA, obtain a local raw train parquet from a fixed revision of
`gijs/avqa-processed`, including audio, questions, choices and reference answers:

```bash
python -m espnet2.rl.qa_data --input /path/to/train.parquet \
  --source-dataset gijs/avqa-processed --source-revision DATASET_COMMIT \
  --source-split train --qa-template default --train-count 3200 \
  --heldout-count 32 --seed 42 --output-dir data/rl/avqa
```

This creates `train.jsonl`, `dev.jsonl`, materialized audio and `admission.json`.
The local dev partition retains its actual source train split; it is not the
official validation/test set. The preparer rejects repeated IDs/audio and can
exclude explicitly provided benchmark manifests with repeatable
`--exclude-manifest`. Its file/decoded-PCM checks do not establish semantic,
component or historical SFT independence. Use a fresh directory and keep the
receipt, audio and manifests together. Private protection registries and
experiment selection/observer scripts are external to the public framework.

For TTS, use a true LibriTTS train split and a separate dev split:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data \
  --dataset libritts --input /path/to/train-split --task tts_asr_wer \
  --max-words 25 --shuffle-seed 42 --output data/rl/tts_train.jsonl
```

Bagpiper's preparer also accepts `--dataset wav_scp --wav-scp ... --text ...`;
QA adds `--questions` and `--choices`. References remain separate from model
inputs. TTS decodes actual eight-stream codec actions, then the pinned ASR
scores the resulting audio. The batched reward is `-min(WER, 1)`; invalid codec
or missing audio scores -1 and is counted. Codec actions, transcripts and
sampled waveform audits remain attached to their candidates.

## Validate and train

Resolve the model, codec/tokenizer/audio assets and the original native training
YAML first. Preserve its modality order, geometry and activation checkpointing.
Set `miles_source`, `sglang_source` and `rollout_python` to the reviewed sources
and interpreter. Low-precision actors require CPU FP32 optimizer master weights.

The same command validates and runs either layout; `--validate-only` performs
configuration/input checks without starting models or checking live Ray placement:

```bash
python -m espnet2.bin.rl_train --config /path/to/one-gpu.yaml \
  --manifest /path/to/train.jsonl --eval-manifest /path/to/dev.jsonl \
  --checkpoint /path/to/native.pth --train-config /path/to/native.yaml \
  --output-dir /path/to/fresh/run --validate-only
```

Remove `--validate-only` to train. Public QA receipts can additionally be bound
with `--data-admission /path/to/admission.json`. Prepare a fresh output directory
for each run; model-only rollout exports do not contain optimizer state.

For two-GPU asynchronous training, activate the reviewed environment on each
host, expose Miles on `PYTHONPATH`, and start a dedicated Ray cluster. On one
host with two available GPUs:

```bash
export PYTHONPATH=/absolute/path/to/miles
ray start --head --num-gpus=2 --port=6379
python -m espnet2.bin.rl_train --config /path/to/two-gpu.yaml \
  --manifest /path/to/train.jsonl --eval-manifest /path/to/dev.jsonl \
  --checkpoint /path/to/native.pth --train-config /path/to/native.yaml \
  --output-dir /path/to/fresh/run --ray-address auto \
  --train-node RAY_NODE_IP --rollout-node RAY_NODE_IP --validate-only
```

For two one-GPU hosts, join both to the dedicated cluster and select their
actual Ray node IDs/IPs separately. Live execution verifies placement. These
commands use existing GPU resources; platform submission and site admission
remain outside the library.

The existing recipe stages call this entry: OWSM stage 18, Bagpiper QA stage 6,
Bagpiper TTS stage 4 and OpusLM stage 10. Recipe `rl.sh` wrappers retain their
preparation options; their training path is the same public module.

## Worked example: Bagpiper QA on one GPU

Use the reviewed environment from [ENVIRONMENT.md](ENVIRONMENT.md), with
`ESPNET_RL_SOURCE`, `ESPNET_RL_ENV`, `ESPNET_MILES_SOURCE` and
`ESPNET_SGLANG_SOURCE` set to its actual paths. Supply a matching native Bagpiper
SFT checkpoint and released SpeechLM YAML, plus a local raw train parquet from
an identified AVQA revision. A native checkpoint is a single model-state file
(raw state or weights under `module`); export a supervised DCP first using
`espnet2.speechlm.bin.export_checkpoint`. Model weights and AVQA are not supplied
by the RL installer. Use fresh output paths throughout this example.

```bash
. "$ESPNET_RL_ENV/bin/activate"
export PYTHONPATH="$ESPNET_MILES_SOURCE"
cd "$ESPNET_RL_SOURCE"
export QA_WORK=/absolute/path/to/new-bagpiper-qa-pilot
export QA_CHECKPOINT=/absolute/path/to/bagpiper-sft/model.pt
export QA_NATIVE_YAML=/absolute/path/to/matching/released-train.yaml
export QA_PARQUET=/absolute/path/to/avqa/train.parquet
export QA_DATASET_REVISION=ACTUAL_DATASET_COMMIT
```

Build checkpoint-compatible tokenizer/codec/audio metadata on CPU. This command
fetches only pinned public metadata, not model weights. Its optional
`--text-assets`, `--audio-assets`, `--codec-assets` use existing local metadata.
The released YAML's modality order, freezing and activation-checkpointing
settings are retained; metadata preparation does not establish model/GPU fit.

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.prepare_assets \
  --train-config "$QA_NATIVE_YAML" --output-dir "$QA_WORK/assets"
python -m espnet2.rl.qa_data --input "$QA_PARQUET" \
  --source-dataset gijs/avqa-processed --source-revision "$QA_DATASET_REVISION" \
  --source-split train --qa-template default --train-count 3200 \
  --heldout-count 32 --seed 42 --output-dir "$QA_WORK/data"
```

The preparer needs enough eligible distinct audio/IDs for both partitions.
It creates `data/train.jsonl`, **`data/dev.jsonl`**, audio and `admission.json`;
use that dev filename rather than the template's `heldout.jsonl`. The default
prompt matches the main template's `qa_v1_reward`; native-answer/WEST-training
profiles have different explicit prompt/reward contracts.

Create a personal YAML outside the source checkout. Merge the asset receipt's
paths into the template's existing adapter options, retaining its memory
settings. The five-step pilot writes its full milestone only at step 5.

```bash
python - <<'PY'
import json, os, sys
from pathlib import Path
import yaml

work = Path(os.environ["QA_WORK"]).resolve()
source = Path(os.environ["ESPNET_RL_SOURCE"]).resolve(strict=True)
receipt = json.loads((work / "assets/asset_manifest.json").read_text())
cfg = yaml.safe_load(
    (source / "egs2/bagpiper/speechlm1/conf/rl.yaml").read_text()
)
cfg["adapter_options"].update(receipt["adapter_options"])
cfg.update(
    train_config=receipt["train_config"],
    checkpoint=str(Path(os.environ["QA_CHECKPOINT"]).resolve(strict=True)),
    manifest=str(work / "data/train.jsonl"),
    eval_manifest=str(work / "data/dev.jsonl"),
    output_dir=str(work / "run"), steps=5, save_interval=5,
    miles_source=str(Path(os.environ["ESPNET_MILES_SOURCE"]).resolve(strict=True)),
    sglang_source=str(Path(os.environ["ESPNET_SGLANG_SOURCE"]).resolve(strict=True)),
    rollout_python=sys.executable,
)
with (work / "pilot.yaml").open("x") as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
PY
python -m espnet2.bin.rl_train --config "$QA_WORK/pilot.yaml" \
  --data-admission "$QA_WORK/data/admission.json" --validate-only
```

Validation checks configuration, sources and admission; it does not load the
checkpoint or validate GPU memory/numerics. Run the following only on an
available authorized H100 with adequate host RAM/disk. No Ray daemon is needed
for this one-GPU synchronous profile. Use a dedicated GPU 0, and wait for each
command to exit before starting the next. Evaluate the original weights first,
train from the same pilot YAML, then evaluate the saved native weights:

```bash
CUDA_VISIBLE_DEVICES=0 python -m espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate \
  --config "$QA_WORK/pilot.yaml" --checkpoint "$QA_CHECKPOINT" --policy-step 0 \
  --manifest "$QA_WORK/data/dev.jsonl" --output "$QA_WORK/before" \
  --qa-protocol west --temperature 0 --samples 1 --max-new-tokens 2048 \
  --audio-seconds 30 --batch-size 32 --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -m espnet2.bin.rl_train --config "$QA_WORK/pilot.yaml" \
  --data-admission "$QA_WORK/data/admission.json"
CUDA_VISIBLE_DEVICES=0 python -m espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate \
  --config "$QA_WORK/run/frozen_config.json" \
  --checkpoint "$QA_WORK/run/step-000005.native.pth" --policy-step 5 \
  --manifest "$QA_WORK/data/dev.jsonl" --output "$QA_WORK/after" \
  --qa-protocol west --temperature 0 --samples 1 --max-new-tokens 2048 \
  --audio-seconds 30 --batch-size 32 --device cuda:0
```

Baseline evaluation uses `pilot.yaml` because the training run does not exist
yet. Never train from an evaluator's frozen config: evaluation sets one sample
per prompt. Greedy evaluation intentionally omits `--verify-native`; training's
sampled-action probability checks are separate. Final evaluation uses
`*.native.pth`, not the combined optimizer `*.pt` checkpoint.

Each evaluation retains `responses.json`, `items.jsonl`, `protocol.json`,
`summary.json` and `completion.json`. Require success and the same ordered 32 dev
IDs/settings in both outputs. Compare **`summary.v1_accuracy`**, matching the
main recipe's answer extractor, and also report `west_accuracy`; this evaluator
still names `west_accuracy` its primary metric. Check `count == 32` and include
truncations/unresolved answers in the reported denominator.

Training retains `run/metrics.jsonl`, rollout evidence, `frozen_config.json`,
`step-000005.pt` (full state) and `step-000005.native.pth`. Five updates and 32
local-dev items are an engineering pilot, not full MMAU or quality-improvement
acceptance. File/PCM exclusions do not prove component/SFT independence. This
one-H100 Bagpiper profile still needs actual memory, numerical and full-loop GPU
validation; CPU tests and successful metadata/config preparation do not supply it.

## Numerical contracts and advanced modes

The ordinary Miles/PPO objective scores the sampled distribution consistently
in learner, behavior and reference passes. For supported OWSM/Bagpiper paths,
`0 < temperature <= 1`, whole-support sampling (`top_p: 1`, `top_k: -1`) and
recorded multi-stream support are checked before optimization. Other families
and the OWSM vLLM option retain their declared temperature limitations.

Bagpiper QA's optional think profiles are under `conf/advanced/`; their named
reward and parser semantics remain distinct from the main `qa_v1_reward`.
The native final-answer reward (`qa_bagpiper_v2_reward`) extracts a terminal
answer without consulting the reference, then applies correctness 1 + format 0.1.
Changing reward/prompt/parser requires an explicit configuration change and a
matched evaluation, not repointing an old recipe alias.

`egs2/bagpiper/speechlm1/conf/advanced/rl_west_4gpu.yaml` is an **advanced
four-H100** synchronous-disaggregated example. It retains its native-answer
reward, raw-temperature-one full-vocabulary detached-current WEST loss,
sequence/token reduction, k3 KL and top-k 50 sampling at temperature 0.7. Its
protocol logprob pass serves integrity checks, not the loss denominator. This
example disables site-specific periodic evaluation; use the ordinary evaluator
below or supply an explicit external `periodic_eval_function` callback. It is
not the accepted one-/two-GPU recipe, and backend distribution equality is not
established by a mean-error guard.

Optional multi-rank learning retains native all-reduce or PyTorch DDP. Global
advantages and loss denominators precede sharding. CPU optimizer offload stores
FP32 master weights and Adam moments on the owner rank, then broadcasts updated
parameters after gradient averaging/clipping. Two-GPU main profiles use one
learner; additional learners are an advanced configuration.

## Evaluate and inspect a run

OWSM evaluates a native checkpoint on a separate manifest:

```bash
python -m espnet2.s2t.rl.evaluate --config /path/to/run/frozen_config.json \
  --checkpoint /path/to/run/step-000005.native.pth \
  --manifest /path/to/dev.jsonl --output /path/to/fresh/wer.json --device cuda:0
```

Bagpiper QA and TTS use the same task evaluator:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate \
  --config /path/to/run/frozen_config.json \
  --checkpoint /path/to/run/step-000005.native.pth --policy-step 5 \
  --manifest /path/to/dev.jsonl --output /path/to/fresh/evaluation
```

Choose the QA protocol explicitly for native-prompt comparisons; historical
fuzzy answer mapping, exact final choice and format reward are different
metrics. Compare the same complete ordered examples, audio/prompt conditioning,
decoding settings and parser. Training reward changes across different sampled
batches do not establish held-out improvement.

Runs retain actual actions, masks/support, behavior/current/reference logprobs,
policy versions, gradients, timing and native/full optimizer checkpoints. A
full checkpoint includes optimizer masters/moments, schedule, rank RNG and
absolute data cursor. Recovery requires compatible full state; checkpoint
structure checks alone do not prove a crash/restart GPU test. Failure
preservation is bounded: publication failures, process death, broken collectives
and partial optimizer updates are not universally recoverable.

`perf/train_time` covers the learner call; `perf/train_wait_time` covers the
interval since its preceding call, including save/export/rollout/scheduling.
Initial model load and first rollout are excluded from the first wait. Exclude
warmup and periodic-evaluation boundaries for steady-state comparisons, report
full wall time/GPU allocation separately, and never add rollout subwindows to
wait a second time. CPU tests validate contracts, not memory fit, throughput or
quality on the target GPU.
