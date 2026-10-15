Bagpiper rollout uses a candidate-local Hugging Face `DynamicCache` and one
audio encoder/adaptor feature Tensor. The SGLang scheduler still receives the
complete primary sequence for each one-token request and owns primary sampling.
The native secondary sampler, action supports, log-probabilities, seed, phase
transitions and stop conditions retain their existing paths. The learner never
receives these inference caches.

The model defaults `espnet_rl_inference_cache` to `true`, and new exports record
that flag explicitly. For an independent full-prefix comparison, set
`sglang_engine_kwargs.json_model_override_args` to
`{"espnet_rl_inference_cache": false}`. The rollout merges these overrides with
the resolved payload path. Exported `espnet_rl_cache_max_frames` is the effective
text context size; `espnet_rl_cache_max_audio_bytes` defaults to 67108864.
Sequences above the frame bound use the original uncached calculation without
shortening generation. An oversized audio feature Tensor is not retained.

State belongs to one model instance and one `(source_id, sample, step, mode)`.
Its key also contains model load generation, parameter identities/versions,
geometry, device, dtype and a hash of the complete native input, including audio
features and spans. A new identity starts at position zero and receives a fresh
mutable HF KV object. Requests within that candidate must advance by exactly one
position and append exactly one complete multistream frame, preserving every
primary and secondary prefix token. The HF cache length must match that prefix
before and after its extension. Invalid or partially failed requests clear state
and fail, rather than reuse a suspect cache.

The cache retains at most one candidate and is cleared on a candidate reset,
weight loading, model dtype/device movement, explicit cleanup, invalid request,
or the final action at the generation token budget. EOS is sampled by SGLang
after model forward; the completed candidate can remain temporarily until the
next candidate reset. The final Engine process shutdown releases its state.
No scheduler hook or extra generation request is needed for cleanup.

`espnet_token_trace.inference_cache` reports prefill/decode input frames, KV
length, retained audio features and actual audio encoder call counts. These are
HF cache diagnostics; SGLang `cached_tokens` describes its separate prefix cache.

Run CPU contracts explicitly from the repository root:

```sh
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -m pytest -o addopts= \
  espnet2/speechlm/model/speechlm/rl/bagpiper/test_inference_cache.py
```

The tests use real random tiny HF Qwen3 and Qwen3 audio encoders with nine-stream
multimodal inputs. They check full-prefix/cached hidden states and per-stream
log-probabilities, exact action support and token/RNG equivalence, identity and
complete-extension rejection, bounded state and failure cleanup. Primary CPU
sampling is a reference; real checkpoint/SGLang GPU alignment and throughput
require an independent frozen experiment before production use.
