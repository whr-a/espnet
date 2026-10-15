# Native GRPO short-run recipes

Each new `*_collocate.yaml` / `*_disagg_async.yaml` template freezes 32 independent source prompts, eight candidates per prompt and 256 global candidates per update. `steps: 5`, `ppo_epochs: 1`, `warmup_steps: 1` mean five optimizer updates total; update one is warmup for timing. All profiles retain FP32 and a 1536-token cap. Microbatch size changes memory layout, not the global batch.

Select one of these new templates explicitly with `--config` / `--rl-config`;
the earlier small synchronous YAML files retain their declared budgets. The
32×8 profile does not change the ordinary supervised recipe defaults.

| Recipe directory | Template prefix | Task / source |
| --- | --- | --- |
| `egs2/owsm_v4/s2t1/conf` | `rl_miles_wer` | OWSM v4 / LibriSpeech ASR |
| `egs2/librispeech/owsm_rl1/conf` | `rl_miles_wer` | LibriSpeech OWSM wrapper |
| `egs2/bagpiper/speechlm1/conf` | `rl` | AVQA default QA |
| `egs2/bagpiper/speechlm1/conf` | `rl_think` | AVQA think QA |
| `egs2/bagpiper_tts/speechlm1/conf` | `rl` | LibriTTS TTS / ASR reward |
| `egs2/libritts/speechlm1/conf` | `rl_opuslm_librispeech` | OpusLM / LibriSpeech ASR |
| `egs2/libritts/speechlm1/conf` | `rl_opuslm_dialogue_librispeech` | OpusLM Dialogue / LibriSpeech ASR |

Public `collocate` maps to `execution_mode: sync`. Its intended eight-GPU geometry is four native training ranks on GPUs 0–3 and four independent rollout replicas on GPUs 4–7. A TP1 subprocess merely visible to four GPUs still uses one GPU: it does not establish the four-replica layout. Async uses eight native training ranks on one node and eight independent TP1 rollout replicas on a second node. OWSM can use DDP; larger Bagpiper/Opus select chunked native allreduce and CPU reference/optimizer. Verify actual rank/replica placement before timing.

Each SGLang replica retains native model semantics. OpusLM batches a
rollout's candidates in one SGLang call. Bagpiper batches a rollout's candidates on SGLang's native Qwen3
layers and records each frame's support for the learner; the earlier frozen
source004 production runs retain one-request full-prefix Bagpiper generation.
OWSM retains full-prefix generation. These profiles do not establish best throughput merely by selecting
async. Use the independent native Miles source environment described in
[ENVIRONMENT.md](ENVIRONMENT.md), with Miles
`a22fd824191c64ba7c5d6c531f125406e4ce9cc6` and SGLang
`283d68c8bff309a42476c95f7c71dad895d09da6`. Its CPU import/hash/pip-check
acceptance is separate from actual Ray/GPU execution. Earlier synchronous
environments do not establish full-runtime import compatibility.

## Genuine source preparation

Use each corpus's true train split and independent dev split. AVQA's fixed local holdout from cached train must retain its true split and be reported as local holdout, not official validation. Never repeat a small manifest to reach32, or move LibriTTS test-clean into training.

Existing full-corpus recipe stages remain available. A CPU engineering subset from official archives can be prepared with:

```bash
python -m espnet2.rl.corpus_subset --dataset LibriSpeech --split train-clean-100 --minimum-pairs 128 --output-dir data/rl/source_train
python -m espnet2.rl.corpus_subset --dataset LibriSpeech --split dev-clean --minimum-pairs 32 --output-dir data/rl/source_dev
python -m espnet2.rl.prepare --wav-scp data/rl/source_train/wav.scp --text data/rl/source_train/text --output data/rl/train_candidates.jsonl
python -m espnet2.rl.prepare --wav-scp data/rl/source_dev/wav.scp --text data/rl/source_dev/text --output data/rl/dev_candidates.jsonl
```

`corpus_subset` records URL/HTTP identity, actual bytes consumed and every retained member SHA256. It stops after enough complete pairs; the receipt explicitly states no full archive download/checksum verification. For TTS, prepare true LibriTTS train/dev pairs and their candidate manifests:

```bash
python -m espnet2.rl.corpus_subset --dataset LibriTTS --split train-clean-100 --minimum-pairs 128 --output-dir data/rl/tts_source_train
python -m espnet2.rl.corpus_subset --dataset LibriTTS --split dev-clean --minimum-pairs 32 --output-dir data/rl/tts_source_dev
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset wav_scp --task tts_asr_wer --wav-scp data/rl/tts_source_train/wav.scp --text data/rl/tts_source_train/text --output data/rl/tts_train_candidates.jsonl
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset wav_scp --task tts_asr_wer --wav-scp data/rl/tts_source_dev/wav.scp --text data/rl/tts_source_dev/text --output data/rl/tts_dev_candidates.jsonl
```

An existing extracted LibriTTS directory can instead use `--dataset libritts --task tts_asr_wer --input /path/to/split --output data/rl/tts_candidates.jsonl`; it reads `.wav` / `.normalized.txt` pairs. For cached AVQA:

```bash
python -m espnet2.rl.short_run_data avqa --input /path/to/train.parquet --qa-template default --output-dir data/rl/avqa_candidates
```

Use a distinct output and `--qa-template think` for think. Source parquet and exact public template hashes are recorded; fixed source-ID hashing creates independent candidate partitions.

User QA data accepts Kaldi `wav.scp`, `text`, `questions` and a JSON choices
mapping. All use the same utterance IDs. For QA, `text` contains the reference
answer string; `questions` contains the question, and `choices.json` maps each
ID to an array of answer strings. For ASR and TTS, `text` contains the transcript
to recognize or synthesize. Prepare separate train/heldout inputs:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset wav_scp --task qa --wav-scp data/user_train/wav.scp --text data/user_train/text --questions data/user_train/questions --choices data/user_train/choices.json --qa-template default --output data/rl/qa_train_candidates.jsonl
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset wav_scp --task qa --wav-scp data/user_dev/wav.scp --text data/user_dev/text --questions data/user_dev/questions --choices data/user_dev/choices.json --qa-template default --output data/rl/qa_dev_candidates.jsonl
```

For example, `text` may contain `utt001 A dog barking`, `questions` may contain
`utt001 What sound is heard?`, and `choices.json` may contain
`{"utt001": ["A dog barking", "A door closing"]}`. Select `--qa-template think`
in both preparations for the think profile; the selected reward remains in
the training YAML.

For each Opus family, `python -m espnet2.speechlm.rl.prepare --family opuslm` or `opuslm_dialogue` accepts `--wav-scp`, `--text`, `--model-assets`, `--dac-config`, `--dac-checkpoint`, `--xeus-config`, `--xeus-checkpoint`, `--kmeans`, `--output`, `--device cpu`. The preparer records codec/SSL artifacts and nine-stream conditions. CPU/GPU DAC quantization can differ: a new CPU preparation is its own protocol, not identical to historical GPU arrays.

## Registered protection and selection

The 32×8 and async profiles in this tree currently require this project's
complete registered protection files, including Gemini945+human444. These
private files are additional inputs to the public corpus/model assets;
`wav.scp` preparation alone does not satisfy this project admission. Ordinary
supervised stages and earlier synchronous profiles retain their own entry
conditions. A portable user-scope admission is being implemented separately;
these frozen project profiles continue to require the complete registry.

Build the explicit index from current registered metadata. With `--decode-registered-audio`, it verifies complete specialist sample/dialogue metadata and permitted specialist audio. It never follows sealed Test payload references:

```bash
python -m espnet2.rl.short_run_data protect --current /path/to/CURRENT.json --specialist-current /path/to/SPECIALIST_CURRENT.json --boundary-extractor /path/to/boundary_keys.py --test-keys /path/to/test.protected_keys.json --cross-protection /path/to/cross_capability_protection.json --human-closure /path/to/human_component_closure_evidence.json --decode-registered-audio --output data/rl/protection.json
python -m espnet2.rl.short_run_data select --train-candidates data/rl/train_candidates.jsonl --heldout-candidates data/rl/dev_candidates.jsonl --train-split train-clean-100 --heldout-split dev-clean --source-dataset LibriSpeech --source-revision SOURCE_RECEIPT_ID --protection data/rl/protection.json --boundary-extractor /path/to/boundary_keys.py --output-dir data/rl/frozen
```

For AVQA's local partition, both true splits are `train` and add `--heldout-is-train-partition`. Outputs are `train32.jsonl`, `heldout.jsonl`, `admission.json`. The gate checks all registered raw/normalized audio, canonical source/question/transcript keys, declared connected components and transitive candidate closure, and train/heldout independence. All protection files and both manifests are bound by hash. Entire Gemini945+human444 is protected, including text-only records; clean930 is not substituted.

`scoped_component_admission: VERIFIED_REGISTERED_PROTECTION` describes this actual registered scope. Unknown semantic links and the unestablished common blind test remain explicit limits; it does not claim all-capability blind-test admission.

## Resolve and launch

Resolve native checkpoint/config, model assets, exact ESPnet/SGLang/Miles revisions, interpreters, manifests and fresh output directory before freezing. OWSM `rl.sh` and `espnet2.s2t.rl.recipe` preserve the selected template's update budget, including the new five-update profile. Older small synchronous templates keep their declared 100 updates unless explicitly overridden. Bagpiper wrapper stages1/2/3 prepare/validate/train; Opus `speechlm.sh` stage10 accepts `--rl_config`.

In every selected YAML, resolve `miles_source`, `sglang_source` and
`rollout_python` to those checked sources and the native Miles interpreter;
these paths depend on the local installation. The async path retains the
original Miles `train_async.py:train` code and substitutes only the native
ESPnet component factories. Run the async driver directly with Python from
outside torchrun. Start the dedicated Ray head and worker with clean distributed
identity as described in the environment guide; the public CLI connects to the
provided cluster and does not repair a previously started dirty Ray environment.

From the recipe directory and the checked native Miles environment, the async
public entry resolves two already available Ray nodes:

```bash
python -m espnet2.bin.rl_train_async --config conf/rl_miles_wer_disagg_async.yaml --manifest data/rl/frozen/train32.jsonl --eval-manifest data/rl/frozen/heldout.jsonl --data-admission data/rl/frozen/admission.json --checkpoint /path/to/native/checkpoint --train-config /path/to/native/config --output-dir exp/new_async_run --ray-address RAY_ADDRESS --train-node EXACT_TRAIN_NODE_ID_OR_IP --rollout-node EXACT_ROLLOUT_NODE_ID_OR_IP --validate-only
```

Remove `--validate-only` after frozen runtime/environment, node placement, application connectivity, real weight identity and data gates pass. Selectors are actual Ray node IDs/IPs, not SSH entry addresses. This command does not submit/restore/migrate/resize platform resources. For collocate, launch four native ranks while keeping physical GPUs 4–7 for isolated SGLang subprocesses:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 -m espnet2.bin.rl_train --config conf/rl_miles_wer_collocate.yaml --manifest data/rl/frozen/train32.jsonl --eval-manifest data/rl/frozen/heldout.jsonl --data-admission data/rl/frozen/admission.json --checkpoint /path/to/native/checkpoint --train-config /path/to/native/config --output-dir exp/new_collocate_run
```

The config's `rollout_num_replicas: 4` and `rollout_gpus: '4,5,6,7'` are physical inference GPU IDs; training ranks bind their torchrun local IDs. Four-rank/four-replica execution must be verified before claiming eight-GPU usage. Preserve raw rollout, behavior/current/reference probabilities, masks, versions, loss, checkpoints and timing for all five updates.

The existing recipe stages route either config through `espnet2.bin.rl_launch`. A 32×8 synchronous config starts four torchrun ranks automatically; an async config calls the original Miles loop entry and requires `--ray-address`, `--train-node`, `--rollout-node`. Both require `--data-admission` and verify current manifest/media/protection hashes before training. Earlier small synchronous recipes keep their single-process entry.

For OWSM, use `egs2/owsm_v4/s2t1/rl.sh` or `egs2/librispeech/owsm_rl1/run.sh` with `--config`, `--manifest`, `--eval-manifest`, `--data-admission`, `--train-config`, `--checkpoint`, and a fresh `--work-dir`; add the Ray options for async. The template's five-update budget and CPU reference remain intact unless explicitly overridden. `--validate-only` builds and verifies the effective recipe config without training.

For Bagpiper QA/TTS, select the existing optional RL stage (`run.sh --stage 6` / `run.sh --stage 4`) or call `rl.sh --stage 2 --stop-stage 3 --rl-config ... --manifest ... --eval-manifest ... --data-admission ...`; add the same Ray options for async. Stage2 validates; stage3 trains. For Opus, `speechlm.sh --stage 10 --stop_stage 10 --rl_config ... --rl_manifest ... --rl_eval_manifest ... --rl_data_admission ...` routes either mode; async adds `--rl_ray_address`, `--rl_train_node`, `--rl_rollout_node`. `--rl_validate_only true` performs only the actual CLI/data validation.

After preparing and admitting each family's manifests, these commands call
the existing recipe stages. Resolve the model/audio/tokenizer asset paths in
the selected YAML before launching. Each example assumes the current directory
is the corresponding recipe directory from the table above:

```bash
# Bagpiper QA, egs2/bagpiper/speechlm1. For think select rl_think_collocate.yaml.
bash rl.sh --stage 2 --stop-stage 3 --rl-config conf/rl_collocate.yaml --manifest data/rl/frozen/train32.jsonl --eval-manifest data/rl/frozen/heldout.jsonl --data-admission data/rl/frozen/admission.json --checkpoint /path/to/qa/native/checkpoint --train-config /path/to/qa/native/config --output-dir exp/new_qa_collocate

# Bagpiper TTS, egs2/bagpiper_tts/speechlm1.
bash rl.sh --stage 2 --stop-stage 3 --rl-config conf/rl_disagg_async.yaml --manifest data/rl/frozen/train32.jsonl --eval-manifest data/rl/frozen/heldout.jsonl --data-admission data/rl/frozen/admission.json --checkpoint /path/to/tts/native/checkpoint --train-config /path/to/tts/native/config --output-dir exp/new_tts_async --ray-address RAY_ADDRESS --train-node EXACT_TRAIN_NODE_ID_OR_IP --rollout-node EXACT_ROLLOUT_NODE_ID_OR_IP

# OpusLM, egs2/libritts/speechlm1. For Dialogue select rl_opuslm_dialogue_librispeech_collocate.yaml.
bash speechlm.sh --stage 10 --stop_stage 10 --rl_config conf/rl_opuslm_librispeech_collocate.yaml --rl_manifest data/rl/frozen/train32.jsonl --rl_eval_manifest data/rl/frozen/heldout.jsonl --rl_data_admission data/rl/frozen/admission.json --rl_checkpoint /path/to/opus/native/checkpoint --rl_train_config /path/to/opus/native/config --rl_output_dir exp/new_opus_collocate
```

For either QA mode or TTS, switch the YAML suffix to `_disagg_async.yaml` and
provide the three Ray options to test two-node async execution. For Opus add
`--rl_ray_address`, `--rl_train_node` and `--rl_rollout_node` with that suffix.
Use a distinct output directory for each family and execution mode.

Both modes use the same learner timing scope: input transfer, native scoring, task reward, rollout-evidence serialization and optimizer update. `train_wait_time` spans the gap from the previous learner window's end to the next window's start, including checkpoint save, export/publication, rollout and scheduling outside that call. The async behavior audit (the generating weights swapped into the learner from a CPU snapshot) remains inside its learner window. The initial model load and first rollout are excluded from the first wait metric; update1 is timing warmup. Report full orchestration wall time and total allocated GPU cost separately, including initial/final publication and teardown. For updates2–5, calculate cumulative wait ratio as `sum(train_wait_time) / sum(step_time)`, rather than averaging the individual ratios.
