# Bagpiper native QA post-training protocol

**New profile: unvalidated for training.**
[conf/rl_native_qa_4gpu_async.yaml](conf/rl_native_qa_4gpu_async.yaml) is a
separate candidate for four H100 GPUs: two native learners and two independent
SGLang inference replicas. This change prepares native-prompt evaluation and a
future reward contract; it does not establish a new training result.

## Restore the SFT task before comparing RL results

The [official SFT model card](https://huggingface.co/espnet/bagpiper-sft)
reports **74.5% MMAU-Mini** and **73.1% full MMAU**. Bagpiper generates a rich
caption and reasoning before its final answer; its native QA style does not
require the entire completion to be an `<answer>` block.

The earlier 28.9% → 31.0% result remains a comparison under the frozen WEST
prompt and extraction protocol. It is **not an official-SFT reproduction**.
CPU comparison now establishes that its public-HF reconstruction
`790bb1d4fdaafd3cce3d44eb40b9e77a989d322c6e4dc17b0172c62b3d2610fa`
differs in **402 tensors** from the pinned official SFT checkpoint
`50980756fad38bbefeb326ac0012e39f431ec5af8c0caaac8bd254f179f33746`.
The shape and dtype matches do not make these weights equivalent. Reproduce
the official checkpoint's behavior before claiming a gain over its SFT score.

## Native prompts and final answers

Prepare a **new** manifest with `--qa-template bagpiper`. Each record carries
the verbatim native SFT system instruction and user text:

```text
question: {question}
 Options: {choice 1}; {choice 2}; {choice 3}; {choice 4}
```

The system instruction asks for a detailed caption, reasoning and then the
chosen option. Its implementation is
[qa_contract.py](../../../espnet2/speechlm/model/speechlm/rl/bagpiper/qa_contract.py),
with system-prompt SHA256
`c32b7ed5afe9418833a402d8c42abae1240e999d0ab16f079d6ea9061b51cca8`.
The native final marker is `the correct option is:`. Do not append a WEST
answer-only instruction or drop the system message while keeping this name.

For a local AVQA training parquet, the public preparation entry is:

```bash
python -m espnet2.rl.qa_data \
    --input /path/avqa/train.parquet \
    --source-dataset gijs/avqa-processed \
    --source-revision EXACT_DATASET_COMMIT --source-split train \
    --qa-template bagpiper --train-count 3200 --heldout-count 32 \
    --seed 42 --output-dir /path/avqa/bagpiper
```

This writes `train.jsonl`, `dev.jsonl` and `admission.json`. Add benchmark and
project exclusions through `--exclude-manifest`; the public receipt does not
claim semantic/component closure or absence of historical SFT exposure.
MMAU evaluation requires its own newly prepared native QA manifest.

## Reward contract

`qa_bagpiper_v2_reward` gives **1.0 for the correct final choice plus 0.1 for
valid terminal structure**. Captions and reasoning may precede the answer.
Both the native marker followed by an option and a single final `<answer>`
block qualify for the structural bonus. A correct exact plain option can earn
1.0 without the bonus. The format component is independent of correctness.

The parser receives only the response and available choices. It resolves a
unique normalized exact option or an unambiguous option label; the reference
is compared afterwards. Harmless Unicode, case, spacing and decoration are
normalized. Musical answers such as `a minor` and `b flat` retain their first
word. There is no substring/Jaccard scoring, whole-caption choice search or
ground-truth-directed fallback. Malformed answer tags and contradictory
terminal choices fail closed.

The original profiles retain their original semantics:

| Profile | Reward |
| --- | --- |
| WEST default | Accuracy 1.0 + whole-completion format 1.0 |
| WEST think | Accuracy 0.8 + one answer block 0.1 + one think block 0.1 |
| New Bagpiper native QA | Final-choice accuracy 1.0 + native/tag terminal format 0.1 |

## Evaluation and separate metrics

With asset paths filled in and a prepared native MMAU manifest, run evaluation
in a fresh output directory:

```bash
python -m espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate \
    --config egs2/bagpiper/speechlm1/conf/rl_native_qa_4gpu_async.yaml \
    --checkpoint /path/to/official/model.pt \
    --manifest /path/to/native/mmau_mini.jsonl \
    --output /path/to/fresh/native-sft-mmau \
    --qa-protocol bagpiper_native --max-new-tokens 4096 \
    --audio-seconds 60 --temperature 0 --samples 1
```

The native evaluator reports historical SFT choice-mapping compatibility as
`native_choice_accuracy`, strict reward-compatible extraction as
`native_strict_accuracy`, and its structural rate separately. The historical
mapper's substring/Jaccard/full-response fallbacks are **evaluation
compatibility diagnostics**, not the new training reward. WEST and V1
extraction results also remain separate fields. Preserve complete responses,
truncation counts, prompt/checkpoint identities and per-item extraction paths;
do not silently substitute one metric for another.

The 4096-token, greedy setting comes from the historical local native
evaluation. It is not a uniquely established official-paper launcher: the
[released inference YAML](https://huggingface.co/espnet/bagpiper-sft/blob/5646600302bf2670214c601721e44f6d36145c26/inference_text.yaml)
uses 2048 text steps, temperature 0.6 and top-k 20, whereas the paper prose
describes greedy decoding. Label these protocol differences explicitly.

## Training candidate and validation boundary

The candidate keeps **32 prompts × 8 candidates, 100 updates**, learning rate
1e-6, KL 0.04, FP32, CPU optimizer/reference/activation offloads, the existing
probability/support gates and two-plus-two placement. It increases the response
cap to **4096**, context to **8192**, and limits concurrent requests to **12**
per inference replica. The product 12 × 8192 = 98304 is below the previously
observed 121275-token FP32 KV pool; it is only a sizing bound. The new context,
response lengths and official weights still require fresh probability,
generation-completion and learner-memory preflights before training.

Configuration validation alone does not perform those GPU checks. No long
training run has been accepted for this new profile. Use the shared
[environment instructions](../../../espnet2/rl/ENVIRONMENT.md), and retain all
original configs and frozen-run evidence when creating the next experiment.
