"""WER denominators and completeness for independent OpusLM evaluation."""

from types import SimpleNamespace

import pytest
import torch

from espnet2.rl.data import collate_rollouts
from espnet2.rl.rollout import sha256
from espnet2.rl.types import TokenScores
from espnet2.speechlm.rl.evaluate import (
    summarize_wer,
    validate_export_files,
    verify_log_probs,
)


def generation(source_id, text):
    return {
        "source_id": source_id,
        "text": text,
        "token_ids": [1],
        "log_probs": [-1.0],
        "prompt_token_ids": [0],
    }


def test_sample_mean_differs_from_word_weighted_corpus():
    sources = [
        {"id": "short", "reference": "Ａ"},
        {"id": "long", "reference": "a b c"},
    ]
    outputs = [
        generation("short", "a"),
        generation("short", "wrong"),
        generation("long", "a b c"),
        generation("long", "a b"),
    ]
    result = summarize_wer(sources, outputs, 2)
    assert result["wer_mean"] == pytest.approx(1 / 3)
    assert result["wer_corpus"] == pytest.approx(1 / 4)
    assert result["total_reference_words"] == 8
    assert result["total_word_errors"] == 2
    assert result["completions"] == 4
    assert result["records"][0]["wer"] == 0
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_wer(sources, outputs[:-1], 2)
    with pytest.raises(ValueError, match="identity"):
        summarize_wer(sources, list(reversed(outputs)), 2)


def test_native_verification_applies_both_training_drift_bounds():
    sources = [{"id": "a", "reference": "a"}]
    batch = collate_rollouts(sources, [generation("a", "a")] * 2, 2)

    def policy(offset):
        def score(item):
            mask = item.batch["response_mask"]
            return TokenScores(
                item.batch["rollout_log_probs"] + offset, torch.zeros_like(mask), mask
            )

        return score

    gate = SimpleNamespace(score_tolerance=0.03, score_mean_tolerance=None)
    errors, drift = verify_log_probs(policy(0.02), batch, gate)
    assert errors == pytest.approx([0.02, 0.02])
    assert drift["max"] == drift["mean"] == pytest.approx(0.02)
    gate.score_mean_tolerance = 0.01
    with pytest.raises(RuntimeError, match="mean logprob disagreement"):
        verify_log_probs(policy(0.02), batch, gate)


def test_empty_reference_is_explicit_and_punctuation_is_not_an_error():
    result = summarize_wer(
        [{"id": "empty", "reference": ""}],
        [generation("empty", ""), generation("empty", "hello!")],
        2,
    )
    assert result["wer_mean"] == 0.5
    assert result["wer_corpus"] is None
    assert result["empty_reference_completions"] == 2
    assert result["total_reference_words"] == 0
    result = summarize_wer(
        [{"id": "punctuation", "reference": "hello!"}],
        [generation("punctuation", "HELLO!"), generation("punctuation", "hello")],
        2,
    )
    # The reward's normalization: case and punctuation are not word errors.
    assert result["wer_mean"] == 0
    assert result["normalization"].startswith("NFKC, casefold, punctuation removed")


@pytest.mark.parametrize(
    "relative",
    [
        "/absolute",
        "../outside",
        "nested/../../outside",
        "./config.json",
        "nested//config.json",
    ],
)
def test_export_rejects_escaping_or_noncanonical_recorded_paths(tmp_path, relative):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Invalid recorded"):
        validate_export_files(tmp_path, {relative: "0" * 64})


def test_export_requires_complete_unchanged_regular_files(tmp_path):
    root = tmp_path / "hf"
    root.mkdir()
    config = root / "config.json"
    config.write_text("{}")
    recorded = {"config.json": sha256(config)}
    validate_export_files(root, recorded)
    weights = root / "model.safetensors"
    weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="complete file set"):
        validate_export_files(root, recorded)
    recorded["model.safetensors"] = sha256(weights)
    validate_export_files(root, recorded)
    weights.write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="file changed"):
        validate_export_files(root, recorded)


def test_export_rejects_symlink_escape_even_with_matching_hash(tmp_path):
    root = tmp_path / "hf"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (root / "model.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        validate_export_files(root, {"model.safetensors": sha256(outside)})
