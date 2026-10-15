# Native GRPO short-run recipes

Each new `*_collocate.yaml` / `*_disagg_async.yaml` template freezes 32 independent source prompts, eight candidates per prompt and 256 global candidates per update. `steps: 5`, `ppo_epochs: 1`, `warmup_steps: 1` mean five optimizer updates total; update one is warmup for timing. All profiles retain FP32 and a 1536-token cap. Microbatch size changes memory layout, not the global batch.

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

SGLang currently retains full-prefix and sequential sampling semantics inside each replica. These are correctness profiles; no optimized cache or best-throughput claim follows from the mode. The full Miles async runtime needs a separately reviewed environment. Previously accepted synchronous environments do not establish full-runtime import compatibility.

## Genuine source preparation

Use each corpus's true train split and independent dev split. AVQA's fixed local holdout from cached train must retain its true split and be reported as local holdout, not official validation. Never repeat a small manifest to reach32, or move LibriTTS test-clean into training.

Existing full-corpus recipe stages remain available. A CPU engineering subset from official archives can be prepared with:

```bash
python -m espnet2.rl.corpus_subset --dataset LibriSpeech --split train-clean-100 --minimum-pairs 128 --output-dir data/rl/source_train
python -m espnet2.rl.corpus_subset --dataset LibriSpeech --split dev-clean --minimum-pairs 32 --output-dir data/rl/source_dev
python -m espnet2.rl.prepare --wav-scp data/rl/source_train/wav.scp --text data/rl/source_train/text --output data/rl/train_candidates.jsonl
python -m espnet2.rl.prepare --wav-scp data/rl/source_dev/wav.scp --text data/rl/source_dev/text --output data/rl/dev_candidates.jsonl
```

`corpus_subset` records URL/HTTP identity, actual bytes consumed and every retained member SHA256. It stops after enough complete pairs; the receipt explicitly states no full archive download/checksum verification. For TTS substitute `--dataset LibriTTS`, then use the existing `espnet2.speechlm.model.speechlm.rl.bagpiper.data --dataset libritts --task tts_asr_wer` with `--input` and `--output`. For cached AVQA:

```bash
python -m espnet2.rl.short_run_data avqa --input /path/to/train.parquet --qa-template default --output-dir data/rl/avqa_candidates
```

Use a distinct output and `--qa-template think` for think. Source parquet and exact public template hashes are recorded; fixed source-ID hashing creates independent candidate partitions.

For each Opus family, `python -m espnet2.speechlm.rl.prepare --family opuslm` or `opuslm_dialogue` accepts `--wav-scp`, `--text`, `--model-assets`, `--dac-config`, `--dac-checkpoint`, `--xeus-config`, `--xeus-checkpoint`, `--kmeans`, `--output`, `--device cpu`. The preparer records codec/SSL artifacts and nine-stream conditions. CPU/GPU DAC quantization can differ: a new CPU preparation is its own protocol, not identical to historical GPU arrays.

## Registered protection and selection

Build the explicit index from current registered metadata. With `--decode-registered-audio`, it verifies complete specialist sample/dialogue metadata and permitted specialist audio. It never follows sealed Test payload references:

```bash
python -m espnet2.rl.short_run_data protect --current /path/to/CURRENT.json --specialist-current /path/to/SPECIALIST_CURRENT.json --boundary-extractor /path/to/boundary_keys.py --test-keys /path/to/test.protected_keys.json --cross-protection /path/to/cross_capability_protection.json --human-closure /path/to/human_component_closure_evidence.json --decode-registered-audio --output data/rl/protection.json
python -m espnet2.rl.short_run_data select --train-candidates data/rl/train_candidates.jsonl --heldout-candidates data/rl/dev_candidates.jsonl --train-split train-clean-100 --heldout-split dev-clean --source-dataset LibriSpeech --source-revision SOURCE_RECEIPT_ID --protection data/rl/protection.json --boundary-extractor /path/to/boundary_keys.py --output-dir data/rl/frozen
```

For AVQA's local partition, both true splits are `train` and add `--heldout-is-train-partition`. Outputs are `train32.jsonl`, `heldout.jsonl`, `admission.json`. The gate checks all registered raw/normalized audio, canonical source/question/transcript keys, declared connected components and transitive candidate closure, and train/heldout independence. All protection files and both manifests are bound by hash. Entire Gemini945+human444 is protected, including text-only records; clean930 is not substituted.

`scoped_component_admission: VERIFIED_REGISTERED_PROTECTION` describes this actual registered scope. Unknown semantic links and the unestablished common blind test remain explicit limits; it does not claim all-capability blind-test admission.

## Resolve and launch

Resolve native checkpoint/config, model assets, exact ESPnet/SGLang/Miles revisions, interpreters, manifests and fresh output directory before freezing. Old OWSM `rl.sh` and `espnet2.s2t.rl.recipe` default `--steps` to100: explicitly supply `--steps 5` or use the frozen config directly. Bagpiper wrapper stages1/2/3 prepare/validate/train; Opus `speechlm.sh` stage10 accepts `--rl_config`.

The async public entry resolves two already available Ray nodes:

```bash
python -m espnet2.bin.rl_train_async --config conf/rl_miles_wer_disagg_async.yaml --manifest data/rl/frozen/train32.jsonl --eval-manifest data/rl/frozen/heldout.jsonl --data-admission data/rl/frozen/admission.json --checkpoint /path/to/native/checkpoint --train-config /path/to/native/config --output-dir exp/new_async_run --ray-address RAY_ADDRESS --train-node EXACT_TRAIN_NODE_ID_OR_IP --rollout-node EXACT_ROLLOUT_NODE_ID_OR_IP --validate-only
```

Remove `--validate-only` after frozen runtime/environment, node placement, application connectivity, real weight identity and data gates pass. Selectors are actual Ray node IDs/IPs, not SSH entry addresses. This command does not submit/restore/migrate/resize platform resources. Existing sync entry is `python -m espnet2.bin.rl_train --config CONFIG`; four-rank/four-replica collocate execution must be verified before claiming eight-GPU usage. Preserve raw rollout, behavior/current/reference probabilities, masks, versions, loss, checkpoints and timing for all five updates.
