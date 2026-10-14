"""Independent evaluation must preserve data, sampled actions and model identity."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from espnet2.rl.types import TokenScores
from espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate import (
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
    return sources, result, SimpleNamespace(group_size=2)


def test_manifest_split_and_response_identity_are_required():
    sources, result, config = fixtures()
    validate_manifests([{"id": "train"}], sources)
    with pytest.raises(ValueError, match="overlap"):
        validate_manifests(sources, sources)
    _, batch = validate_responses(result, sources, config, "native", -1)
    assert len(batch) == 2
    for field, value in (
        ("native_sha256", "wrong"),
        ("policy_step", 0),
        ("rollout_backend", "vllm"),
    ):
        with pytest.raises(ValueError, match="identity"):
            validate_responses({**result, field: value}, sources, config, "native", -1)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_responses(
            {**result, "generations": result["generations"][:1]},
            sources,
            config,
            "native",
            -1,
        )


def test_native_action_probability_and_mask_gate():
    sources, result, config = fixtures()
    _, batch = validate_responses(result, sources, config, "native", -1)

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

    assert verify_native_log_probs(Policy(), batch, 0.03) == [0, 0]
    with pytest.raises(ValueError, match="logprob mismatch"):
        verify_native_log_probs(Policy(probability=-1), batch, 0.03)
    with pytest.raises(ValueError, match="mask"):
        verify_native_log_probs(Policy(mask=0), batch, 0.03)


def test_export_verifies_complete_tensor_values_and_shard_paths(tmp_path):
    native = tmp_path / "native.pt"
    state = {"weight": torch.tensor([1.0, 2.0]), "vocab_weight": torch.ones(2)}
    torch.save(state, native)
    root = tmp_path / "hf"
    root.mkdir()
    save_file({"weight": state["weight"]}, root / "model.safetensors")
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    assert verify_export(native, root)["exact_native_export_tensors"] == 1
    save_file({"weight": torch.zeros(2)}, root / "model.safetensors")
    with pytest.raises(ValueError, match="tensor differs"):
        verify_export(native, root)
    index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="outside the audited"):
        verify_export(native, root)
