# Native GRPO recipes on one or two GPUs

Main profiles use **32 source prompts × 8 candidates = 256 candidates per
optimizer update**. Microbatch settings control memory without changing that
global batch. The default budget is five updates; increase `steps` in the
selected YAML for a deliberate longer run. Use a fresh output directory. CPU
validation does not establish that model execution fits a GPU.

| Recipe directory | Main configuration | Task |
| --- | --- | --- |
| `egs2/owsm_v4/s2t1` | `rl_miles_wer_1gpu.yaml`, `rl_miles_wer_2gpu_async.yaml` | LibriSpeech WER |
| `egs2/librispeech/owsm_rl1` | `rl_miles_wer.yaml`, `rl_miles_wer_disagg_async.yaml` | LibriSpeech OWSM wrapper |
| `egs2/bagpiper/speechlm1` | `rl.yaml`, `rl_disagg_async.yaml` | AVQA QA; `rl_think*` selects think |
| `egs2/bagpiper_tts/speechlm1` | `rl.yaml`, `rl_disagg_async.yaml` | LibriTTS generated audio → ASR → WER |
| `egs2/libritts/speechlm1` | `rl_opuslm_dialogue_librispeech_1gpu.yaml`, `rl_opuslm_dialogue_librispeech_2gpu_async.yaml` | OpusLM Dialogue ASR |
| `egs2/libritts/speechlm1` | `rl_opuslm_librispeech_2gpu_async.yaml` | Plain OpusLM ASR |

The focus is one-GPU collocation and two-GPU disaggregated async; supplied
layouts use at most four GPUs. OWSM has complete one/two-GPU training evidence.
OpusLM Dialogue has a single-GPU short run. Bagpiper's new small-GPU full loop
and OpusLM's two-GPU full loop still need validation; Plain has no accepted
single-GPU profile. No family has an established RL quality gain in this delivery.

## Prepare public or user data

Prepare independent train and held-out splits. ASR and TTS accept Kaldi
`wav.scp` plus `text`; QA additionally needs questions, choices and reference
answers. Do not repeat a small manifest to simulate 32 independent prompts.
For an engineering subset from official LibriSpeech archives:

```bash
python -m espnet2.rl.corpus_subset --dataset LibriSpeech --split train-clean-100 --minimum-pairs 128 --output-dir data/rl/source_train
python -m espnet2.rl.corpus_subset --dataset LibriSpeech --split dev-clean --minimum-pairs 32 --output-dir data/rl/source_dev
python -m espnet2.rl.prepare --wav-scp data/rl/source_train/wav.scp --text data/rl/source_train/text --output data/rl/train.jsonl
python -m espnet2.rl.prepare --wav-scp data/rl/source_dev/wav.scp --text data/rl/source_dev/text --output data/rl/dev.jsonl
```

`corpus_subset` records source URL, consumed bytes and retained file hashes;
it stops after enough complete pairs and does not claim a full archive checksum.
Existing full-corpus ESPnet preparation stages remain available.

For LibriTTS select `--dataset LibriTTS` and true train/dev splits, then run:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset wav_scp --task tts_asr_wer --wav-scp data/train/wav.scp --text data/train/text --output data/rl/tts_train.jsonl
```

An extracted LibriTTS split can instead use `--dataset libritts --task
 tts_asr_wer --input /path/to/split --output data/rl/tts_train.jsonl`. Prepare
the held-out manifest separately from dev data.

For public AVQA, start with a **local raw train parquet** from a fixed revision
of `gijs/avqa-processed`, including its embedded `audio` and
`instruction`/`question`, `answer` and choices. Record the revision used to
obtain the file; this preparer does not download it or infer its split from its
filename. Run in the optional RL environment; preparation is CPU-only and
requires pandas, PyArrow, NumPy and SoundFile.

```bash
AVQA_TRAIN_PARQUET=/path/to/downloaded/train.parquet
AVQA_REVISION=THE_EXACT_DOWNLOADED_DATASET_COMMIT
python -m espnet2.rl.qa_data \
  --input "$AVQA_TRAIN_PARQUET" \
  --source-dataset gijs/avqa-processed --source-revision "$AVQA_REVISION" \
  --source-split train --qa-template default \
  --train-count 3200 --heldout-count 32 --seed 42 \
  --output-dir data/rl/avqa_default
```

This emits `train.jsonl`, `dev.jsonl`, materialized audio under `prepared/`,
and `admission.json`. The 3,200 training prompts cover **100 updates × 32
independent prompts** without wrapping the manifest; eight candidates per
prompt are generated during RL. For 200 updates select at least 6,400 training
prompts. Use the chosen horizon in the recipe YAML before launching.

Selection orders source IDs by SHA256 of the seed and ID, reserves 32 eligible
records for local dev, and takes the next 3,200 for training. Question IDs,
prepared audio file hashes and decoded PCM identities are unique across both
roles. The output directory must be new. Keep its materialized audio with the
manifests. A different output path produces new path/hash bindings even when
the selected source IDs remain deterministic.

Both roles retain their actual native `train` split in per-record provenance
and `admission.source_split`. **`dev.jsonl` is a local partition of train,
not official validation/test.** The split and source revision are user
attestations bound to the parquet SHA256; available split metadata is checked
for contradictions. Use a separate fresh output and `--qa-template think` for
WEST's optional think prompt.

Optional benchmark exclusion uses an explicitly supplied local manifest. For
example, prepare the complete official MMAU-Mini parquet separately, then add
it to a fresh AVQA preparation:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data \
  --dataset mmau --task qa --input /path/to/official/mmau-mini.parquet \
  --qa-template default --output data/rl/mmau/mmau_mini.jsonl
python -m espnet2.rl.qa_data \
  --input "$AVQA_TRAIN_PARQUET" \
  --source-dataset gijs/avqa-processed --source-revision "$AVQA_REVISION" \
  --source-split train --qa-template default \
  --train-count 3200 --heldout-count 32 --seed 42 \
  --exclude-manifest data/rl/mmau/mmau_mini.jsonl \
  --output-dir data/rl/avqa_excluding_mmau
```

`--exclude-manifest` can be repeated. It excludes matching question IDs,
prepared audio file hashes and decoded PCM, including across different audio
containers with identical decoded samples. It does not establish resampling,
perceptual, semantic, document/group/component or historical SFT independence.
MMAU may contain several questions on the same clip: keep the full benchmark
separate from the unique-audio local admission holdout.

The public receipt has schema `espnet-public-qa-data-v1`. Its
`training_ready: true` means the stated local media/role checks passed;
`full_component_admission` remains `NOT_ASSERTED`. It is not a signed clearance
or a replacement for project-specific protection registries and required
contamination audits. Internal protected-set results cannot automatically
transfer to a new public dataset/checkpoint. The existing runtime
`verify_data_admission` rechecks role-bound manifest hashes, record counts and
audio bytes. Pass all three artifacts together to `espnet2.bin.rl_launch`:
`--manifest data/rl/avqa_default/train.jsonl`,
`--eval-manifest data/rl/avqa_default/dev.jsonl`, and
`--data-admission data/rl/avqa_default/admission.json` (or the corresponding
paths from `avqa_excluding_mmau`). Resolve the model assets as described below
before launcher validation.

The older `espnet2.rl.short_run_data avqa` command instead consumes a
WEST-prepared cache with `extra_info`/`reward_model` and emits **candidate
pools**, not launchable train/dev manifests or admission. Its default
`--pool-count 128` is insufficient for 100 × 32 training. Its separate
`select`/`protect` workflow requires the caller's actual registered protection
inputs; do not fabricate them for public preparation.

QA on user audio accepts:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset wav_scp --task qa --wav-scp data/train/wav.scp --text data/train/text --questions data/train/questions --choices data/train/choices.json --qa-template default --output data/rl/qa_train.jsonl
```

All files share utterance IDs. `text` holds the reference answer, `questions`
holds the question, and `choices.json` maps each ID to answer strings. For
example, `text` may contain `utt001 A dog barking` and choices may be
`{"utt001": ["A dog barking", "A door closing"]}`.

OpusLM's `python -m espnet2.speechlm.rl.prepare --family opuslm` or
`opuslm_dialogue` takes `--wav-scp`, `--text`, `--model-assets`, `--dac-config`,
`--dac-checkpoint`, `--xeus-config`, `--xeus-checkpoint`, `--kmeans`, `--output`
and `--device cpu`. It records codec/SSL assets and nine-stream conditions.
CPU and GPU codec preparation may quantize differently; record which was used.

## Resolve assets and launch

Install the pinned optional environment in [ENVIRONMENT.md](ENVIRONMENT.md).
Set `miles_source`, `sglang_source`, `rollout_python` and model-specific assets
in the YAML. Each run needs a fresh output directory. Public and user data
work without private registries. Optional `--data-admission admission.json`
checks role-bound manifest hashes, record counts and audio hashes. It does not
rebuild external protection indexes. Existing
frozen runs retain their original receipts and agreed protections.

For single-GPU OWSM, from `egs2/owsm_v4/s2t1`:

```bash
bash rl.sh --config conf/rl_miles_wer_1gpu.yaml --manifest data/rl/train.jsonl --eval-manifest data/rl/dev.jsonl --checkpoint /path/to/checkpoint --train-config /path/to/config --work-dir exp/rl_1gpu --validate-only
```

Remove `--validate-only` to train on an authorized GPU. One learner and its
SGLang process share GPU 0; this profile needs no torchrun or resource submission.
For async, start a dedicated Ray head exposing two available GPUs from a clean
shell outside torchrun, then select the two-GPU configuration. Follow the
[activation instructions](ENVIRONMENT.md#5-build-and-activate-the-native-rl-environment)
first, including Miles visibility on every Ray host:

```bash
. /absolute/path/to/rl-env/bin/activate
export PYTHONPATH=/absolute/path/to/miles
ray start --head --num-gpus=2 --port=6379
python -m espnet2.bin.rl_launch --config conf/rl_miles_wer_2gpu_async.yaml --manifest data/rl/train.jsonl --eval-manifest data/rl/dev.jsonl --checkpoint /path/to/checkpoint --train-config /path/to/config --output-dir exp/rl_2gpu --ray-address auto --train-node RAY_NODE_IP --rollout-node RAY_NODE_IP --validate-only
```

Both selectors can name the same Ray node: roles then occupy distinct GPUs.
For two one-GPU hosts, join them to the dedicated Ray cluster and select each
node separately. Use actual Ray node IDs or IPs. Validation checks inputs and
configuration; it does not connect to Ray or check live placement. Training
records and verifies live placement after `--validate-only` is removed. These
commands do not submit, restore or resize platform resources.

| Existing recipe | RL stage / arguments |
| --- | --- |
| OWSM v4 | `run.sh --stage 18`, or `rl.sh --config ...` |
| Bagpiper QA | `run.sh --stage 6`, or `rl.sh --stage 2 --stop-stage 3 --rl-config ...` |
| Bagpiper TTS | `run.sh --stage 4`, or `rl.sh --stage 2 --stop-stage 3 --rl-config ...` |
| OpusLM | `speechlm.sh --stage 10 --stop_stage 10 --rl_config ...` |

Bagpiper stage 1 prepares, 2 validates, and 3 trains; its wrapper accepts
`--manifest`, `--eval-manifest`, `--checkpoint`, `--train-config` and
`--output-dir`. OpusLM uses `--rl_manifest`, `--rl_eval_manifest`,
`--rl_checkpoint`, `--rl_train_config`, `--rl_output_dir`. Async adds
`--ray-address`, `--train-node`, `--rollout-node` (prefixed `--rl_` with
underscores in `speechlm.sh`). Hosts need the same checked runtime profile.

## Interpret the records

Retain actual actions, behavior/current/reference logprobs, masks, policy
versions, gradients, optimizer checkpoints and held-out outputs. Initial model
load and first rollout are outside the first wait metric. Exclude declared
timing warmup and calculate wait ratio as
`sum(train_wait_time) / sum(step_time)`. Report total wall time and allocated
GPU cost separately. Faster components or rising sampled rewards do not
establish better held-out WER or MMAU accuracy.
