# Native data parallel training

Launch eight training processes on the training node with one rank per GPU.
Each rank executes native scoring and backward for 32 of the 256 candidates in
a 32-prompt, eight-candidate rollout. GRPO advantages and the sampled-action
token denominator are computed on the complete batch before sharding.

`distributed_strategy: ddp` uses PyTorch DDP. Gradients accumulate under
`no_sync` until the final local microbatch, followed by one reduction per PPO
epoch. `gradient_as_bucket_view` reduces gradient duplication after the first
iteration. DDP's full-model bucket memory must still fit during the first update.

`distributed_strategy: native_allreduce` keeps the native module unwrapped and
averages accumulated gradients in their existing storage. Each collective is
bounded by `allreduce_chunk_bytes` (64 MiB by default). A small usage collective
handles a head used on only some ranks; globally unused parameters keep a None
gradient and do not acquire Adam moments. This mode avoids a second complete
gradient buffer and still runs forward/backward on every training GPU.

Both modes verify replica names, shapes, dtypes and trainable flags, then
synchronize native parameters and registered buffers before copying the frozen
reference. Buffers keep their original devices and storage; CPU buffers use a
bounded staging chunk when the collective backend is NCCL.

For large FP32 speech models, declare these fields explicitly:

```yaml
distributed_strategy: native_allreduce
allreduce_chunk_bytes: 67108864
optimizer_device: cpu
reference_device: cpu
prompts_per_step: 32
group_size: 8
micro_batch_size: 1
```

CPU Adam offload with multiple ranks keeps shadow parameters and both Adam
moments only on rank zero. All ranks finish gradient averaging and clipping;
rank zero updates the CPU shadows, copies them into the original parameter
storage and broadcasts bounded weight chunks. Only rank zero saves the standard
complete Adam state and native checkpoint. A failed owner update is reported to
every rank before weight broadcasts. A single-rank run retains the existing
optimizer implementation and checkpoint interface.

The measured saved Bagpiper FP32 checkpoint contains 33.88 GiB of tensors;
parameters changed by its accepted first update account for at least 30.80 GiB.
Actor plus gradients therefore require at least 64.68 GiB per training GPU,
before activations, NCCL and temporary tensors. One CPU shadow plus two moments
requires at least 92.41 GiB on rank zero. These metadata bounds do not prove a
particular H100 prompt length fits. Preserve activation checkpointing/offload
and check the actual first optimizer update before declaring GPU acceptance.

`test/espnet2/rl/test_parallel_optimizer.py` runs both strategies with eight CPU
Gloo ranks and two actual optimizer updates. It compares masked global GRPO
loss, gradients, parameters, Adam moments and native checkpoint against a
single-process oracle. It also checks rank-local unused paths, frozen reference
and buffers, exact cross-rank parameters, stable Parameter/storage identity,
owner error propagation and standard optimizer-state restoration. Run it with
the optional runtime's Python and the pinned Miles library on `PYTHONPATH`;
the standalone unittest entry does not require pytest or CUDA.
