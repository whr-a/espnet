# LibriSpeech OWSM Miles post-training

This recipe adds a WER GRPO stage to an existing autoregressive OWSM checkpoint.
It imports Miles as a pinned optional library; OWSM policy, export, rollout and
evaluation adapters live in `espnet2/s2t/rl/`.

The default is **one H100**, with the learner and a persistent SGLang engine
sharing GPU 0. `conf/rl_miles_wer_disagg_async.yaml` uses **two H100s**, one for
the learner and one for rollout, driven by Miles' async loop. Both optimize
32 prompts × 8 candidates per update. The YAML files are relative symlinks to
the canonical profiles in the OWSM recipe. Public recipes focus on one and two
GPUs; custom layouts should stay within four GPUs.

Prepare LibriSpeech with the existing `egs2/librispeech/asr1` data stage. From
this directory, select explicit model and environment paths:

```bash
./run.sh \
  --wav-scp ../asr1/data/train_clean_100/wav.scp \
  --text ../asr1/data/train_clean_100/text \
  --train-config /path/to/owsm/config.yaml \
  --checkpoint /path/to/owsm/checkpoint.pth \
  --work-dir exp/owsm_miles_wer \
  --rollout-python /path/to/optional-sglang-env/bin/python \
  --sglang-source /path/to/pinned-sglang-fork \
  --miles-source /path/to/pinned-miles-library \
  --steps 5
```

For two-GPU async, add `--config conf/rl_miles_wer_disagg_async.yaml`,
`--ray-address auto --train-node RAY_NODE_IP --rollout-node RAY_NODE_IP`
with a dedicated Ray cluster exposing both GPUs. The templates default to five updates. Increase `steps` in the selected
YAML for a deliberate longer run. Data preparation accepts user `wav.scp` and `text` files with
matching utterance IDs. Text provides the reference for WER. Preparation
writes an audited manifest and training freezes the effective configuration.

The same entry is available as stage 18 of the existing OWSM recipe:
`bash egs2/owsm_v4/s2t1/run.sh --stage 18` with the same arguments. See the
[stage documentation](../../owsm_v4/s2t1/README.md#optional-miles-rl-stage) and
[shared recipe guide](../../../espnet2/rl/SHORT_RUN_RECIPES.md) for combined
supervised/RL stages, environment pins and data admission.

`OWSMEngine` keeps the SGLang model loaded across updates and loads each native
export in place. ESPnet owns the encoder-decoder policy, scoring and optimizer;
SGLang owns scheduling and sampling. The training directory retains sampled
actions, probability checks, rewards, native checkpoints and metrics.
OWSM-CTC requires a separate non-autoregressive policy.

Use a disjoint manifest for independent native greedy evaluation:

```bash
python -m espnet2.s2t.rl.evaluate \
  --config exp/owsm_miles_wer/rl_config.yaml \
  --checkpoint exp/owsm_miles_wer/train/step-000005.native.pth \
  --manifest /path/to/disjoint-heldout.jsonl --output heldout.json
```

The historical two-update engineering result is retained in
[RESULTS.md](RESULTS.md). A completed optimizer loop establishes engineering
correctness; it does not by itself establish a WER improvement.

An explicit optional vLLM backend remains in
`egs2/owsm_v4/s2t1/conf/rl_miles_wer_vllm.yaml`. Set that YAML's `vllm_source`
and `rollout_python` to the reviewed ESPnet vLLM fork and its environment,
then pass `--config /path/to/rl_miles_wer_vllm.yaml`. The wrapper has no
`--vllm-source` option. This backend also declares 32×8 candidates per update; its GPU
training acceptance is separate from the SGLang profiles.
