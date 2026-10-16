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

AVQA preparation from cached train parquet follows the WEST templates:

```bash
python -m espnet2.rl.short_run_data avqa --input /path/to/train.parquet --qa-template default --output-dir data/rl/avqa
```

Use a distinct output and `--qa-template think` for think. Source/template
hashes are recorded, and deterministic source-ID hashing defines a local
held-out partition of train. Report it as local holdout, not official validation.
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
shell outside torchrun, then select the two-GPU configuration:

```bash
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
