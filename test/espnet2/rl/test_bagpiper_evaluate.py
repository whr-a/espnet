"""Held-out evaluation preserves data, model identity and both MMAU scorings."""

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import save_file

from espnet2.rl.data import collate_rollouts
from espnet2.rl.types import TokenScores
from espnet2.speechlm.model.speechlm.rl.bagpiper import evaluate
from espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate import (
    score_mmau,
    score_tts,
    validate_manifests,
    validate_responses,
    verify_export,
    verify_native_log_probs,
)


def fixtures():
    sources = [{"id": "heldout", "task": "qa", "reference": "answer"}]
    generation = {
        "source_id": "heldout",
        "token_ids": [1],
        "log_probs": [-0.5],
        "prompt_token_ids": [0],
        "text": "answer",
    }
    result = {
        "native_sha256": "native",
        "policy_step": -1,
        "rollout_backend": "sglang",
        "generations": [dict(generation), dict(generation)],
    }
    return sources, result


def test_manifest_split_and_response_identity_are_required():
    sources, result = fixtures()
    validate_manifests([{"id": "train"}], sources)
    with pytest.raises(ValueError, match="overlap"):
        validate_manifests(sources, sources)
    assert len(validate_responses(result, sources, 2, "native", -1)) == 2
    for field, value in (
        ("native_sha256", "wrong"),
        ("policy_step", 0),
        ("rollout_backend", "vllm"),
    ):
        with pytest.raises(ValueError, match="identity"):
            validate_responses({**result, field: value}, sources, 2, "native", -1)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_responses(result, sources, 3, "native", -1)
    stray = {**result["generations"][0], "source_id": "other"}
    with pytest.raises(ValueError, match="order"):
        validate_responses(
            {**result, "generations": [stray, stray]}, sources, 2, "native", -1
        )


def test_native_action_probability_and_mask_gate():
    sources, result = fixtures()
    batch = collate_rollouts(sources, result["generations"], 2)

    class Policy:
        def __init__(self, probability=-0.5, mask=1):
            self.probability, self.mask = probability, mask

        def __call__(self, item):
            shape = item.batch["response_mask"].shape
            return TokenScores(
                torch.full(shape, self.probability),
                torch.ones(shape),
                torch.full(shape, self.mask),
            )

    gate = Namespace(score_tolerance=0.03, score_mean_tolerance=None)
    errors, drift = verify_native_log_probs(Policy(), batch, gate)
    assert errors == [0, 0] and drift == {"max": 0, "mean": 0, "p99": 0}
    with pytest.raises(RuntimeError, match="max logprob disagreement"):
        verify_native_log_probs(Policy(probability=-1), batch, gate)
    # The training gate's mean bound applies too.
    gate.score_tolerance, gate.score_mean_tolerance = 1.0, 0.005
    with pytest.raises(RuntimeError, match="mean logprob disagreement"):
        verify_native_log_probs(Policy(probability=-0.51), batch, gate)
    with pytest.raises(ValueError, match="mask"):
        verify_native_log_probs(Policy(mask=0), batch, gate)


def export_tensors(native, root):
    """What the exporter writes: every policy tensor, in one shard."""
    state = torch.load(native, weights_only=True)
    root.mkdir()
    save_file({"weight": state["weight"]}, root / "model.safetensors")
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    return index


def test_export_verifies_complete_tensor_values_and_shard_paths(tmp_path):
    native = tmp_path / "native.pt"
    state = {"weight": torch.tensor([1.0, 2.0]), "vocab_weight": torch.ones(2)}
    torch.save(state, native)
    root = tmp_path / "hf"
    index = export_tensors(native, root)
    assert verify_export(native, root)["exact_native_export_tensors"] == 1
    save_file({"weight": torch.zeros(2)}, root / "model.safetensors")
    with pytest.raises(ValueError, match="tensor differs"):
        verify_export(native, root)
    index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="outside the audited"):
        verify_export(native, root)


@pytest.fixture
def sharded_export(tmp_path):
    native, root = tmp_path / "native.pt", tmp_path / "hf"
    state = {
        "decoder.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "decoder.bias": torch.tensor([3.0], dtype=torch.bfloat16),
        "codec.weight": torch.tensor([4, 5], dtype=torch.int64),
        "vocab_weight": torch.ones(2),
    }
    torch.save({"module": state}, native)
    root.mkdir()
    save_file(
        {key: state[key] for key in ("decoder.weight", "decoder.bias")},
        root / "decoder.safetensors",
    )
    save_file({"codec.weight": state["codec.weight"]}, root / "codec.safetensors")
    mapping = {
        "decoder.weight": "decoder.safetensors",
        "codec.weight": "codec.safetensors",
        "decoder.bias": "decoder.safetensors",
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    return native, root


def test_export_reads_each_shard_once_and_checks_frozen_tensors(
    sharded_export, monkeypatch
):
    native, root = sharded_export
    reads, read_bytes = [], Path.read_bytes

    def record_read(path):
        if path.suffix == ".safetensors":
            reads.append(path.name)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    receipt = verify_export(native, root)
    assert receipt["exact_native_export_tensors"] == 3
    assert receipt["excluded_nonpolicy"] == ["vocab_weight"]
    assert set(receipt["hf_files"]) == {
        "decoder.safetensors",
        "codec.safetensors",
        "model.safetensors.index.json",
    }
    # Multiple indexed tensors in a shard share one buffered deserialization.
    assert sorted(reads) == ["codec.safetensors", "decoder.safetensors"]
    save_file({"codec.weight": torch.tensor([4, 6])}, root / "codec.safetensors")
    with pytest.raises(ValueError, match="tensor differs.*codec.weight"):
        verify_export(native, root)


@pytest.mark.parametrize("damage", ["missing_shard", "corrupt_shard", "missing_key"])
def test_export_rejects_missing_or_corrupt_sharded_weights(sharded_export, damage):
    from safetensors import SafetensorError

    native, root = sharded_export
    shard = root / "decoder.safetensors"
    if damage == "missing_shard":
        shard.unlink()
        error, message = ValueError, "outside the audited"
    elif damage == "corrupt_shard":
        shard.write_bytes(b"not a safetensors archive")
        error, message = SafetensorError, None
    else:
        save_file(
            {"decoder.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)},
            shard,
        )
        error, message = KeyError, "decoder.bias"
    with pytest.raises(error, match=message):
        verify_export(native, root)


QUESTIONS = [
    {
        "id": "q-sound",
        "task": "qa",
        "reference": "Man",
        "choices": ["Man", "Woman", "Robot"],
        "categories": {"task": "sound"},
    },
    {
        "id": "q-music",
        "task": "qa",
        "reference": "Piano",
        "choices": ["Piano", "Guitar"],
        "categories": {"task": "music"},
    },
]
# Saved responses: WEST's extraction scores the whole untagged response, which
# names a wrong option; the v1 extraction reads the stated answer.
ANSWERS = [
    "<answer>Man</answer>",
    "A woman? No. The correct answer is: Man",
    "Not a guitar.\n**Final Answer:** Piano",
    "<answer>Guitar</answer>",
]


def test_mmau_scores_both_extractions_per_category():
    generations = [
        {"source_id": QUESTIONS[i // 2]["id"], "text": text, "finish_reason": "stop"}
        for i, text in enumerate(ANSWERS)
    ]
    generations[3]["finish_reason"] = "length"
    items, summary = score_mmau(QUESTIONS, generations, 2)
    assert [item["west_correct"] for item in items] == [True, False, False, False]
    assert [item["v1_correct"] for item in items] == [True, True, True, False]
    assert [item["v1_path"] for item in items] == ["tag", "cue", "cue", "tag"]
    assert [item["sample"] for item in items] == [0, 1, 0, 1]
    assert summary["count"] == 4
    assert (summary["west_correct"], summary["v1_correct"]) == (1, 3)
    assert (summary["west_accuracy"], summary["v1_accuracy"]) == (0.25, 0.75)
    assert summary["one_answer_block"] == 2 and summary["truncated"] == 1
    assert summary["v1_paths"] == {"tag": 2, "cue": 2}
    assert summary["by_category"]["task"]["sound"] == {
        "count": 2,
        "west_correct": 1,
        "west_accuracy": 0.5,
        "v1_correct": 2,
        "v1_accuracy": 1.0,
    }
    assert summary["by_category"]["task"]["music"]["v1_correct"] == 1


def test_tts_scores_the_training_reward_wer():
    class Reward:
        def transcribe(self, sources, generations):
            assert [source["id"] for source in sources] == ["a", "a", "b", "b"]
            heard = ["the cat sat", "the cat", "a dog", None]
            return [
                {
                    "source_id": source["id"],
                    "reference": source["reference"],
                    "hypothesis": hypothesis,
                    "wer": 1.0 if hypothesis is None else 0.0,  # not recomputed
                    "status": "ok" if hypothesis else "no_audio",
                    "codec_frames": 0,
                }
                for source, hypothesis in zip(sources, heard)
            ]

    sources = [
        {"id": "a", "reference": "The cat sat."},
        {"id": "b", "reference": "a dog"},
    ]
    generations = [{"finish_reason": "stop"}] * 3 + [{"finish_reason": "length"}]
    items, summary = score_tts(Reward(), sources, generations, 2)
    assert [item["errors"] for item in items] == [0, 1, 0, 2]
    assert summary["corpus_wer"] == pytest.approx(3 / 10)
    assert summary["wer_mean"] == pytest.approx(0.25)
    assert summary["statuses"] == {"ok": 3, "no_audio": 1}
    assert summary["truncated"] == 1


class FakeEngine:
    """The family engine contract: built from an export, one rollout per call."""

    calls = []

    def __init__(self, config, model_path, *, directory, replica=0):
        assert (model_path / "model.safetensors.index.json").is_file()
        assert directory.is_dir() and replica == 0
        self.config = config

    def generate(self, sources, rollout_id, sampling):
        FakeEngine.calls.append(
            ([source["id"] for source in sources], rollout_id, sampling)
        )
        return [
            {
                "source_id": source["id"],
                "text": ANSWERS[index],
                "token_ids": [5],
                "log_probs": [-0.1],
                "prompt_token_ids": [1],
                "finish_reason": "stop",
                "prepared_native_source": {
                    "source_id": source["id"],
                    "data": {"native_model_inputs": [0.5] * 4},
                    "sha256": "f" * 64,
                },
            }
            for index, source in enumerate(sources)
            for _ in range(self.config.group_size)
        ]

    def close(self):
        FakeEngine.calls.append("closed")


@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("qa_protocol", ["west", "bagpiper_native"])
def test_greedy_mmau_evaluation_through_the_rollout_engine(
    tmp_path, monkeypatch, temperature, qa_protocol
):
    from espnet2.speechlm.model.speechlm.rl.bagpiper.qa_contract import (
        BAGPIPER_QA_SYSTEM_PROMPT,
        native_qa_prompt,
    )

    clips = []
    for index, seconds in enumerate((0.5, 2.0)):
        clips.append(tmp_path / f"clip-{index}.wav")
        sf.write(clips[-1], np.linspace(-1, 1, int(16000 * seconds)), 16000)
    heldout = tmp_path / "heldout.jsonl"
    heldout_rows = [
        {**question, "audio": str(clip)} for question, clip in zip(QUESTIONS, clips)
    ]
    if qa_protocol == "bagpiper_native":
        for question in heldout_rows:
            question.update(
                question="Which choice?",
                qa_template="bagpiper",
                system_prompt=BAGPIPER_QA_SYSTEM_PROMPT,
                prompt=native_qa_prompt("Which choice?", question["choices"]),
            )
    heldout.write_text(
        "".join(json.dumps(question) + "\n" for question in heldout_rows)
    )
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"id": "train", "task": "qa"}) + "\n")
    checkpoint = tmp_path / "native.pt"
    torch.save({"weight": torch.ones(3), "vocab_weight": torch.ones(1)}, checkpoint)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "family": "bagpiper",
                "adapter_factory": "unused:factory",
                "exporter": "unused:exporter",
                "sglang_engine": f"{__name__}:FakeEngine",
                "manifest": str(train),
                "group_size": 8,
                "temperature": temperature,
                "max_tokens": 1536,
                "sglang_engine_kwargs": {
                    "context_length": 2048,
                    "max_running_requests": 32,
                },
            }
        )
    )

    def export(command, env, **kwargs):
        assert "WORLD_SIZE" not in env
        if "-c" in command:  # the rollout interpreter's environment
            return SimpleNamespace(stdout=prefix + "\n")
        assert command[command.index("--checkpoint") + 1] == str(checkpoint)
        export_tensors(
            checkpoint, Path(command[command.index("--output") + 1]).parent / "hf"
        )

    prefix = sys.prefix

    def arguments(output, **changes):
        return Namespace(
            **{
                "config": str(config),
                "checkpoint": str(checkpoint),
                "output": str(tmp_path / output),
                "manifest": str(heldout),
                "policy_step": 50,
                "temperature": 0.0,
                "max_new_tokens": 2048,
                "samples": 1,
                "audio_seconds": 1.0,
                "batch_size": 1,
                "device": "cpu",
                "verify_native": False,
                "responses": None,
                "qa_protocol": qa_protocol,
                **changes,
            }
        )

    monkeypatch.setattr(os, "environ", {**os.environ, "WORLD_SIZE": "8"})
    monkeypatch.setattr(evaluate.subprocess, "run", export)
    FakeEngine.calls = []
    output = tmp_path / "evaluation"
    summary = evaluate.run(arguments("evaluation"))
    # One greedy candidate per prompt, one rollout per batch of prompts.
    greedy = {"temperature": 0.0, "max_new_tokens": 2048}
    assert FakeEngine.calls == [
        (["q-sound"], 0, greedy),
        (["q-music"], 1, greedy),
        "closed",
    ]
    assert (summary["west_correct"], summary["v1_correct"]) == (1, 1)
    assert summary["clipped_audio"] == 1 and summary["policy_step"] == 50
    assert summary["by_category"]["task"]["music"]["count"] == 1
    frozen = json.loads((output / "frozen_config.json").read_text())
    assert (frozen["group_size"], frozen["max_tokens"]) == (1, 2048)
    assert frozen["temperature"] == temperature
    assert summary["sampling"] == greedy
    assert summary["primary_metric"] == (
        "native_choice_accuracy"
        if qa_protocol == "bagpiper_native"
        else "west_accuracy"
    )
    # 512 more new tokens than the recipe's on its KV budget of 32 x 2048.
    assert frozen["sglang_engine_kwargs"] == {
        "context_length": 2560,
        "max_running_requests": 25,
    }
    # The 2 s clip was cut to its first second; the short one is untouched.
    requests = json.loads((output / "requests.json").read_text())
    assert requests[1]["audio"] == str(clips[1])
    cut, rate = sf.read(output / "audio" / "000001.wav", dtype="float32")
    assert rate == 16000 and len(cut) == 16000
    assert np.array_equal(cut, sf.read(clips[1], dtype="float32")[0][:16000])
    items = [json.loads(line) for line in (output / "items.jsonl").open()]
    assert [item["source_id"] for item in items] == ["q-sound", "q-music"]
    saved = json.loads((output / "responses.json").read_text())
    # The audio features stay in the manifest's audio; only their hash is kept.
    assert saved["generations"][0]["prepared_native_source"] == {
        "source_id": "q-sound",
        "sha256": "f" * 64,
    }
    assert json.loads((output / "completion.json").read_text())["status"] == (
        "Succeeded"
    )
    # Saved responses are scored again without generating, under the decoding
    # they record.
    saved = str(output / "responses.json")
    again = evaluate.run(arguments("again", responses=saved))
    assert again["v1_correct"] == 1 and len(FakeEngine.calls) == 3
    assert again["sampling"] == greedy and again["audio_seconds"] == 1.0
    for index, changes in enumerate(({"max_new_tokens": 1536}, {"audio_seconds": 3})):
        with pytest.raises(ValueError, match="responses were decoded with"):
            evaluate.run(arguments(f"other-{index}", responses=saved, **changes))
    with pytest.raises(ValueError, match="sampled actions"):
        evaluate.run(arguments("never", verify_native=True))
    assert not (tmp_path / "never").exists()
    # Sampled responses saved without their features cannot be verified.
    evaluate.run(arguments("sampled", temperature=temperature))
    with pytest.raises(ValueError, match="no native features"):
        evaluate.run(
            arguments(
                "verified",
                temperature=temperature,
                verify_native=True,
                responses=str(tmp_path / "sampled" / "responses.json"),
            )
        )
    # The engine runs in the evaluator's own process: its environment and
    # SGLang must be the recipe's.
    prefix = "/another/environment"
    with pytest.raises(ValueError, match="rollout_python"):
        evaluate.run(arguments("elsewhere"))


def test_the_engine_process_must_import_the_recipe_sglang(tmp_path):
    import sglang

    config = SimpleNamespace(rollout_python=sys.executable, sglang_source="")
    evaluate.check_interpreter(config, dict(os.environ))
    config.sglang_source = str(Path(sglang.__file__).parents[2])
    evaluate.check_interpreter(config, dict(os.environ))
    config.sglang_source = str(tmp_path)
    with pytest.raises(ValueError, match="sglang_source"):
        evaluate.check_interpreter(config, dict(os.environ))
