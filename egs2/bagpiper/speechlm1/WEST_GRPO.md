# WEST-aligned Bagpiper QA training

[conf/rl_west_4gpu.yaml](conf/rl_west_4gpu.yaml) provides a synchronous, four-GPU
Bagpiper QA profile. Its source and CPU contracts are implemented; actual GPU
memory/probability validation and MMAU improvement are not yet established for
this profile. Results from the older 100-step asynchronous FP32 profile do not
validate this new configuration.

The reference is WEST commit
[`b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb`](https://github.com/wenet-e2e/west/tree/b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb/examples/grpo),
with its pinned Transformers 4.52.3 defaults. The model remains native Bagpiper;
Miles is imported as a library for rollout orchestration and training components.
The recipe retains its existing native audio encoder and codec freeze, while
updating the language model and audio adaptor. This model-specific parameter
scope differs from WEST's Qwen entry point.

| Setting | This profile |
| --- | --- |
| Hardware | Two native all-reduce learners and two independent TP=1 SGLang replicas |
| Policy batches | 32 audio questions × eight candidates = 256 responses per update |
| Updates | 1,000; fresh run from the pre-RL checkpoint |
| Precision and offload | BF16; CPU AdamW, reference model and activation offload |
| AdamW | Initial LR 1e-6; betas 0.9/0.999; epsilon 1e-8; weight decay zero |
| Scheduler | Linear decay over all 1,000 updates; zero LR warmup |
| Gradient clipping | Global norm 1.0 after accumulating the complete update |
| Generation | Temperature 0.7; top-k 50; top-p 1.0; at most 2,048 response frames |
| Reference KL | Fixed pre-RL model; coefficient 0.04; k3 estimator |
| Checkpoints and MMAU Mini | Every 100 updates; identical evaluation contract versus pre-training |

WEST's shell uses four candidates by default. The 32 × eight batch, Bagpiper
terminal-answer parser, correctness 1.0 plus terminal-format 0.1 reward, and
periodic MMAU evaluation are deliberate overrides. The 2,048-frame response
limit is a Bagpiper adaptation to WEST's 1,024-token default: captions can
otherwise consume the budget before the final answer. Context is 4,096 frames;
64 concurrent inference requests and an 8,192-token native microbatch budget
are engineering settings that require actual memory validation. Microbatch
accumulation preserves the complete 32 × eight update. Captions and intermediate
reasoning are allowed; the parser selects a terminal answer before consulting
its ground truth. Format reward is excluded from benchmark accuracy.

## Loss and inference integrity

Within each group of eight, advantages use the reward mean and sample standard
deviation with an added 1e-4. The loss uses raw, unfiltered full-primary-vocabulary
model log probabilities at temperature one, for both actor and fixed reference.
It follows WEST's detached-current policy-gradient factor and k3 KL; it has no
PPO clipping, stale-policy correction or truncated importance weights. Loss is
averaged over valid sampled action frames within each response and then equally
over the 256 responses, so long answers do not receive more batch weight.

Bagpiper's forced DETECT control frame has no sampled action and remains masked
from the loss. Sampled EOS counts; padding after EOS does not. Text-only QA never
activates secondary codec streams. The response limit includes native control
frames, so it is not identical to a standard text tokenizer's completion count.

Rollout integrity is audited separately. SGLang returns the temperature-0.7,
protocol-masked log probability **before** top-k filtering; the native integrity
pass checks the same values. These are not renormalized top-k sampling
probabilities and never enter the raw WEST objective. Recorded sampling metadata
names this domain explicitly. Different HF/SGLang top-k boundary tie handling
and random-number streams can produce different texts from the same seed.

The BF16 integrity limits in this profile are an absolute per-action maximum
of 1.5 and a mean absolute difference of 0.02, checked together. They are
operational numerical bounds, not loss clipping or reward settings. On the
validated H100 control, changing only the numerical CUDA kernels of an unchanged
native model produced a mean difference of 0.01175 and a maximum of 1.02786;
ordinary native versus rollout scoring gave mean 0.01163 and maximum 0.43654.
The earlier FP32 maximum of 0.03 did not describe this BF16 runtime. The controls
used four questions and 38,180 sampled frames, so the first complete 32 × eight
batch must still pass both bounds. Checkpoint, policy-version, prepared-input,
action-mask and support checks remain separate requirements. Calibrate these
bounds again for a different precision or kernel environment.

Each update waits for the exact current policy snapshot on both rollout
replicas, generates one batch and performs one optimization update. Staleness is
zero. Candidate seeds depend on the experiment seed, absolute rollout ID, source
ID and candidate index, so an evaluation restart does not reset the training
candidate sequence. This requires SGLang deterministic inference and is limited
to text-only QA; codec sampling is not stateless. The opted-in worker reuses
SGLang's seed/position hash with open-interval uniform values before the Gumbel
transform. This prevents the uint32 maximum from creating infinity plus a
filtered negative infinity, which can yield NaN and an action outside top-k.
The compatibility helper lives under ESPnet's Bagpiper RL package; the pinned
SGLang checkout is retained.

This profile preserves the native decoder's FP32 RoPE frequency buffers when
converting model parameters to BF16. Its explicit `match_native_bf16` option
also selects native residual, RMSNorm, SiLU and rotary arithmetic in the Bagpiper
SGLang model. These model-local operations retain the native BF16 rounding
boundaries. The audio encoder keeps SDPA and the native CPU initialization of its sinusoidal
position table. The separate MMAU
evaluation profile leaves these opt-in training flags disabled, preserving its
fixed pre-training generation contract. Actual rollout/native agreement remains
a required runtime check; these options do not establish it by themselves.

The first update uses LR 1e-6. Update 100 uses 9.01e-7 and leaves 9.0e-7 for the
next update; update 1,000 uses 1e-9 and leaves zero. Evaluation and checkpoint
export do not advance this schedule.

## Configure and launch

Follow the shared [environment guide](../../../espnet2/rl/ENVIRONMENT.md), prepare
native AVQA manifests with the existing recipe data stage, and copy the portable
configuration to a local experiment file. Replace its checkpoint, model assets,
manifest, pinned Miles/SGLang source and environment paths. The native training
YAML must enable activation checkpointing for the memory profile.

Periodic evaluation requires a reviewed, hash-bound evaluation plan, a matching
MMAU baseline/configuration, an audit script and a host-specific resource
admission command. These files are site-specific and are not fabricated by this
portable template. Set `periodic_eval_options` to those real files before
launching. Its family callback closes rollout replica0's engine, evaluates on
that same assigned GPU, and restores the current training engine. It adds no
GPU reservation, overlaps no training/evaluation, and does not automatically
retry a failed evaluation. Learner/optimizer/LR/data-cursor state remains resident.

Run on an existing four-GPU allocation with a Ray head on that same node. Reuse
its verified Ray address and active node ID; do not create an extra allocation.
Both selectors name the same host, while Ray reserves disjoint two-GPU actor
groups. The template requests learner physical GPUs 2/3 and inference 0/1;
verify actual actor GPU IDs before training because device numbering is local to
the host. Pass all three selectors through the standard recipe entry point:

```bash
RAY_ADDRESS=127.0.0.1:6379   # Existing head on the assigned four-GPU host.
RAY_NODE_ID=your_verified_active_node_id
./rl.sh --stage 2 --stop-stage 2 --rl-config conf/rl_west_local.yaml \
    --ray-address "$RAY_ADDRESS" --train-node "$RAY_NODE_ID" \
    --rollout-node "$RAY_NODE_ID"
./rl.sh --stage 3 --stop-stage 3 --rl-config conf/rl_west_local.yaml \
    --ray-address "$RAY_ADDRESS" --train-node "$RAY_NODE_ID" \
    --rollout-node "$RAY_NODE_ID"
```

Before the long run, require a real audio rollout and finite nonzero parameter
update, matching inference/native integrity scores, current-policy version
identity, correct reward extraction, and actual BF16 memory acceptance. Keep all
1,000 updates and the ten complete 1,000-item MMAU Mini evaluations under
supervision. Report both the established benchmark answer-mapping view and the
strict final-choice view against the same pre-training baseline. A training
reward increase alone is not MMAU improvement.
