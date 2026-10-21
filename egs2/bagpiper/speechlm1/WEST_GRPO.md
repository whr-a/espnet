# Advanced four-H100 WEST-aligned Bagpiper example

[conf/advanced/rl_west_4gpu.yaml](conf/advanced/rl_west_4gpu.yaml) is an advanced
synchronous-disaggregated example: two native learners and two TP=1 SGLang
replicas. The main public targets remain the one-/two-GPU templates documented
in the [shared RL guide](../../../espnet2/rl/README.md). Those templates retain
their own loss/reward settings and are not validated by a four-GPU run.

This example retains the existing 32× 8 batch, native final-answer correctness 1
plus format 0.1 reward, BF16 native/FA3 inference, temperature 0.7/top-k 50,
2048-frame cap, 8192-token scoring microbatches, AdamW 1e-6 and linear 1000-update
schedule, fixed-reference k3 KL 0.04 and gradient clip 1. The language model and
audio adaptor train; audio encoder/codec remain frozen according to the native
configuration. Prepare QA with `--qa-template bagpiper`.

WEST detached-current loss scores the full primary vocabulary at temperature 1.
Its within-group advantages use sample standard deviation plus 1e-4, and loss
reduction gives each response equal weight after valid-token averaging. Forced
DETECT/padding frames are masked; sampled EOS remains a loss action. There is
no PPO clipping, stale-policy correction or truncated importance denominator
in this synchronous objective. Protocol scores at temperature 0.7 serve the
integrity audit and are separate from the raw loss scores.

The mean-error bound 0.02 and retained maximum telemetry do not establish exact
inference/learner distribution equality. The public example does not activate
a private periodic evaluation plan, resource admission script, auditor or
observer. Use the ordinary evaluator on saved checkpoints; the generic optional
callback hook accepts an explicitly supplied external implementation.

Resolve all paths/assets, start a dedicated Ray cluster on existing resources,
then use `python -m espnet2.bin.rl_train --config conf/advanced/rl_west_4gpu.yaml`
with explicit `--ray-address`, `--train-node` and `--rollout-node` selectors.
Validate the configuration before GPU execution. Historical attempt logs and
quality results are archived separately and are not public recipe defaults.
