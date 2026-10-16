# OpusLM post-training using Miles as a library

This branch extends the existing SpeechLM recipe with optional stage 10. The
released OpusLM / Dialogue model and adapters live under `espnet2/speechlm/rl`;
Miles GRPO, returns, PPO and KL are called through the fixed optional library.
The SGLang fork provides the input-embedding API extension and thin model
registrations that import the implementation from ESPnet. The model, native
scorer, data preparation and exporter remain in `espnet2/speechlm/rl`; Miles is
an unmodified library pinned to `a22fd824191c64ba7c5d6c531f125406e4ce9cc6`.
ESPnet owns the native model, scorer, AdamW and checkpoints. Synchronous profiles
use ESPnet's training loop. The new `disagg_async` profiles retain the original
Miles `train_async.py:train` loop with native ESPnet component factories.

The public profiles focus on one and two H100 80 GB GPUs, with a maximum of
four GPUs for custom layouts. All current profiles use 32 prompts × 8
candidates per optimizer update, accumulated over smaller scoring batches.

| Architecture | Configuration | GPUs | Default updates |
| --- | --- | ---: | ---: |
| Dialogue | `conf/rl_opuslm_dialogue_librispeech_1gpu.yaml` | 1, learner and rollout share GPU 0 | 5 |
| Dialogue | `conf/rl_opuslm_dialogue_librispeech_2gpu_async.yaml` | 2, learner on GPU 0 and rollout on GPU 1 | 5 |
| Plain | `conf/rl_opuslm_librispeech_2gpu_async.yaml` | 2, learner on GPU 0 and rollout on GPU 1 | 5 |

The shorter default YAML names and `*_disagg_async.yaml` are relative symlinks
to these profiles; Dialogue's `*_collocate.yaml` points to its one-GPU profile.
Plain has no one-GPU FP32 profile: policy and gradients (55.2 GiB) plus the
engine's weights (27.6 GiB) already exceed 80 GB. Its former eight-GPU
`*_collocate.yaml` has been removed. Historical results remain in their frozen
experiment records.

Each keeps one `OpusLMEngine` per rollout GPU for the whole run and loads each
export in place. The comments in the YAML files derive their memory settings.
These templates do not imply a complete GPU training acceptance or quality gain
for every architecture/layout. Select one with
`speechlm.sh --stage 10 --stop_stage 10 --rl_config ...`; see the
[shared recipe guide](../../../espnet2/rl/SHORT_RUN_RECIPES.md) for the pinned
Miles/SGLang environment and launch options.

For the earlier small synchronous profile, install Miles through
`tools/installers/install_miles_rl.sh` and use
`tools/installers/install_sglang_rl_env.sh` for a new inference environment, with
the hash-locked profile described in
[`espnet2/rl/ENVIRONMENT.md`](../../../espnet2/rl/ENVIRONMENT.md). The original
`install_sglang_rl.sh` remains available for an already reviewed compatible
dependency profile. That earlier SGLang profile uses `https://github.com/whr-a/sglang` at
`4c3f163fe54634f20ef846e1ec0c507b27ae2430`. See
[`espnet2/rl/README.md`](../../../espnet2/rl/README.md) for the optional flow.
Create the inference venv without `--system-site-packages`; its dependencies
and ESPnet's architecture-specific optional dependencies must be installed in
that environment. Do not prepend an old SGLang installation through
`PYTHONPATH`: the installer checks the actual imported source and refuses
shadowing. `rollout_python` may differ from the native ESPnet interpreter.

The public installer passed on the VA2 H100 host in `va2-clean-rl-install-003`.
Independent `owsm-clean-sglang-001` then generated four responses from two prompts,
with 78 sampled actions and maximum native logprob error 0.0000343323. It performed
zero optimizer updates. Bagpiper and both OpusLM architectures passed imports and
registry identity checks in that clean environment; this did not execute their
GPU model instances. See the environment guide for source pins and evidence.

Prepare LibriSpeech `wav.scp` / `text` using its existing data recipe. The
following command creates separate complete train-clean-100 and dev-clean
manifests (omit `--limit` for the full splits):

```bash
for split in train_clean_100 dev_clean; do
  python -m espnet2.speechlm.rl.prepare \
    --wav-scp "/path/to/librispeech/data/${split}/wav.scp" \
    --text "/path/to/librispeech/data/${split}/text" \
    --model-assets /path/to/opuslm/hf_assets --family opuslm --device cuda:0 \
    --dac-config /path/to/dac/config.yaml --dac-checkpoint /path/to/dac/360epoch.pth \
    --xeus-config /path/to/xeus/config.yaml --xeus-checkpoint /path/to/xeus/model.pth \
    --kmeans /path/to/xeus/km_opus_lm.mdl --output "data/rl/${split}.jsonl"
done
```

Set `train_config`, `checkpoint`, `adapter_options.reference_assets`,
`sglang_source` and `rollout_python` in a copy of the appropriate template YAML.
Set `manifest: data/rl/train_clean_100.jsonl` and choose a new `output_dir`.
The templates use native FP32 and CPU AdamW. Dialogue keeps its reference on
the training GPU; Plain scores its reference on CPU. Preserve the documented
memory settings when selecting a profile. Then use the
existing recipe's optional stage 10:

```bash
./run_rl_opuslm.sh --rl_config conf/rl_opuslm_librispeech_2gpu_async.yaml \
  --rl_ray_address auto --rl_train_node RAY_NODE_IP --rl_rollout_node RAY_NODE_IP
# Dialogue uses its released assets, --family opuslm_dialogue during preparation:
./run_rl_opuslm.sh --rl_config conf/rl_opuslm_dialogue_librispeech.yaml
```

Evaluate the final native state independently. Replace the checkpoint step
number with the actual last update. This command generates every heldout
prompt/group with the trained actor, saves raw SGLang responses and verifies
its sampled probabilities against the native model:

```bash
python -m espnet2.speechlm.rl.evaluate \
  --config exp/rl_opuslm_librispeech_2gpu_async/frozen_config.json \
  --checkpoint exp/rl_opuslm_librispeech_2gpu_async/step-000005.native.pth \
  --manifest data/rl/dev_clean.jsonl --policy-step 5 \
  --output-dir exp/rl_opuslm_librispeech_eval --device cpu --rollout-gpus 1
```

The effective `frozen_config.json` preserves training CLI overrides. YAML is
also supported; when using it, include the exact effective training manifest
and architecture assets. Relative paths in either config are resolved from the
command's current directory. Run from the original recipe directory or freeze
absolute paths before moving an evaluation to another working directory.

Use the Dialogue config for the 1.7B Llama family. Dataset paths are independent
of the recipe directory, so LibriSpeech is selected by the manifest. The
evaluator rejects overlapping training/heldout utterance IDs. `eval_manifest`
in the training YAML records the intended heldout set; training itself does not
automatically run evaluation. References supervise WER and do not enter the
native audio condition. The evaluator records source IDs, every group member,
word-error and reference-word counts, completion-mean WER and corpus WER.
Normalization is NFKC, casefold and punctuation removal, the same
`espnet2.rl.rewards.words` the training reward uses. The mean uses
`errors/max(1, reference_words)` for each completion; corpus WER divides
total errors by total reference words across all completions.

`--export-dir /path/to/rollout/policy-000100` can reuse an existing inference
export only after checking its native and every recorded HF file hash. Training
keeps only its newest `keep_exports` exports (default 2), so set it higher for a
run whose intermediate exports you will evaluate this way. This validation
requires the recorded paths to be canonical relative paths, rejects
parent traversal and symlinks, and verifies the complete HF file set.
`--responses`
audits existing real response records instead of generating again; it checks
their checkpoint hash, policy version, completeness and native probabilities.
Each evaluation uses a new immutable output directory.

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

In the historical prototype on a shared 8-H100 host, two optimizer updates
and separate heldout evaluation succeeded for both released architectures. Plain's second update
had equal task rewards within each group and zero GRPO advantage; fixed KL
produced that update. Plain updated all 356 model tensors (7,409,405,952
parameters); Dialogue updated all 220 tensors
(1,867,433,984 parameters). Adam states were finite at steps 1/2, independent
reference logprobs matched the initial checkpoint, and every final HF tensor
equaled its native counterpart. The unweighted sample mean WER across eight
heldout completions was **11.3803%** for Plain and **39.9663%** for Dialogue,
each with four distinct audio prompts and two completions per prompt. Dialogue's
separate corpus WER was **64/134 = 47.7612%**, using reference-word counts.
Native/SGLang heldout logprob errors were at most 2.0504e-05. These small runs
verify the engineering loop and numerical contract, not quality improvement;
they do not establish full-corpus accuracy or full codec-output TTS support.

The successful two-update runs used four LibriSpeech `train.clean.100`
utterances from speaker 374 and four `validation.clean` (dev-clean) utterances
from speaker 2277. They used `opuslm_assets/train_v2.jsonl` and
`opuslm_assets/heldout_v2.jsonl`, built from
`data/librispeech/{train,heldout}/{wav.scp,text}`. The adjacent
`source_audit.json` files record `openslr/librispeech_asr`, config `all`,
rows 0--3 for each split and audio hashes. Training and heldout IDs and
speakers are disjoint in this sample; this does not establish isolation from
every registered evaluation set or the released model's pretraining data.
The complete-split commands above are reusable examples, not commands that
produced these small-run scores.

The private historical experiment archive retains these run identities and pins:

| Architecture | Successful run directory | Frozen ESPnet commit | Frozen SGLang commit |
| --- | --- | --- | --- |
| Plain | `runs/opuslm_modern_sglang_miles_001` | `2e739523ff2cfef77eedba0960d70f03121fdb38` | `df3495c60d51f3f64029527643699f1276d64061` |
| Dialogue | `runs/opuslm_dialogue_integration_sglang_miles_001` | `c2595666dd0192397a0550a3ce53cacacca85f52` | `e1c0658c4bff816f72588699549d854093601682` |

Both `status.json` files report `Succeeded`, exit 0 and two completed updates.
Each directory retains `protocol.json` (source and manifest hashes),
`frozen_config.json`, `metrics.jsonl`, `acceptance.json`, the step-2 native
checkpoint, and raw heldout rollout records. Plain's
`integration_verification.json` additionally checks the saved final state and
existing responses using ESPnet `c2595666dd0192397a0550a3ce53cacacca85f52`,
without rerunning training. These historical pins differ from the reviewed
optional SGLang library pin recommended above. The earlier
`opuslm_dialogue_modern_sglang_miles_001` was superseded before execution and
is not the successful Dialogue run.
