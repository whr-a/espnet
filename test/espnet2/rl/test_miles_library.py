"""Independent analytical checks of the native library boundary."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import espnet2.rl.miles_library as miles_library
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.miles_library import (
    compute_grpo_outcome_advantage,
    compute_policy_loss_vanilla,
    kl_penalty,
)
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import TokenScores


@pytest.fixture
def pinned_source(tmp_path, monkeypatch):
    """Use an isolated tiny Git repository; never modify the real Miles pin."""
    relative_paths = [
        "miles/backends/training_utils/loss/hub/math_utils.py",
        "miles/ray/rollout/train_data_conversion.py",
        "miles/backends/training_utils/loss/hub/corrections.py",
        "miles/backends/training_utils/weight_update/protocols/cuda_ipc.py",
        "miles/backends/training_utils/weight_update/protocols/broadcast.py",
        "miles/backends/training_utils/weight_update/session.py",
        "train_async.py",
        "other.py",
    ]
    for relative in relative_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Frozen source fixture.\n")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "Freeze fixture",
    )
    revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(miles_library, "PIN", revision)
    monkeypatch.setattr(
        miles_library,
        "_algorithms",
        lambda: SimpleNamespace(__file__=str(tmp_path / relative_paths[0])),
    )
    return tmp_path, SimpleNamespace(miles_commit=revision)


def test_miles_runtime_verification_retains_algorithm_and_driver_evidence(
    pinned_source,
):
    root, config = pinned_source
    algorithms = miles_library.verify_miles(config)
    runtime = miles_library.verify_miles(config, runtime=True)
    assert runtime["source"] == str(root)
    assert algorithms["algorithms"] == runtime["algorithms"]
    assert len(runtime["algorithms"]) == 3
    assert len(runtime["async_driver_sha256"]) == 64
    # Untracked files do not modify pinned source or widen the tracked gate.
    (root / "untracked.txt").write_text("Untracked evidence.\n")
    assert miles_library.verify_miles(config, runtime=True) == runtime


@pytest.mark.parametrize(
    "relative, message",
    [
        ("miles/backends/training_utils/loss/hub/math_utils.py", "algorithm"),
        ("miles/ray/rollout/train_data_conversion.py", "algorithm"),
        ("miles/backends/training_utils/loss/hub/corrections.py", "algorithm"),
        ("train_async.py", "async driver"),
        ("other.py", "unchanged tracked source"),
    ],
)
def test_miles_runtime_rejects_modified_tracked_source(
    pinned_source, relative, message
):
    root, config = pinned_source
    (root / relative).write_text("# Modified source fixture.\n")
    with pytest.raises(ValueError, match=message):
        miles_library.verify_miles(config, runtime=True)


def test_miles_verification_rejects_unreviewed_recipe_pin(pinned_source):
    _, config = pinned_source
    config.miles_commit = "0" * 40
    with pytest.raises(ValueError, match="reviewed Miles pin"):
        miles_library.verify_miles(config, runtime=True)


def test_fast_weight_transfer_checks_transport_and_all_tracked_source(pinned_source):
    root, config = pinned_source
    config.weight_sync_backend = "miles"
    evidence = miles_library.verify_miles(config)
    assert len(evidence["algorithms"]) == 6
    (root / "other.py").write_text("# Modified source fixture.\n")
    with pytest.raises(ValueError, match="unchanged tracked source"):
        miles_library.verify_miles(config)


def test_frozen_json_configuration_retains_scientific_numeric_types(tmp_path):
    config = ESPnetRLConfig(
        learning_rate=1e-6,
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
    )
    path = tmp_path / "frozen_config.json"
    path.write_text(json.dumps(config.to_dict()))
    restored = ESPnetRLConfig.load(path)
    assert isinstance(restored.learning_rate, float)
    assert restored.to_dict() == config.to_dict()


def test_group_baselines_and_forced_padding():
    rewards = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    mask = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    advantage, returns = compute_grpo_outcome_advantage(
        rewards, mask, np.array(["a", "a", "b", "b"]), False
    )
    expected = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected)


def test_ppo_clipping_and_gradient_against_analytical_oracle():
    current = torch.tensor(
        [[np.log(1.4), np.log(0.6)]], dtype=torch.float32, requires_grad=True
    )
    config = SimpleNamespace(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        global_batch_info={"batch_num_tokens": 2, "dp_size": 1},
    )
    loss, _ = compute_policy_loss_vanilla(
        torch.zeros_like(current),
        current,
        torch.tensor([[1.0, -1.0]]),
        torch.ones_like(current),
        "token-mean",
        config,
    )
    # Positive advantage caps at 1.2 and negative advantage floors at 0.8.
    torch.testing.assert_close(loss, torch.tensor(-0.2))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
    current2 = torch.tensor([[0.05, -0.05]], requires_grad=True)
    loss2, _ = compute_policy_loss_vanilla(
        torch.zeros_like(current2),
        current2,
        torch.tensor([[1.0, -1.0]]),
        torch.ones_like(current2),
        "token-mean",
        config,
    )
    loss2.backward()
    torch.testing.assert_close(
        current2.grad,
        torch.tensor([[-0.5 * np.exp(0.05), 0.5 * np.exp(-0.05)]], dtype=torch.float32),
    )


def test_kl_k3_against_formula():
    current = torch.tensor([-0.3, -0.9], requires_grad=True)
    reference = torch.tensor([-0.5, -0.7])
    expected = torch.exp(reference - current) - 1 - (reference - current)
    torch.testing.assert_close(kl_penalty(current, reference, "k3"), expected)


def test_multistream_batch_retains_audio_reward_evidence():
    sources = [{"id": "u", "reference": "hello"}]
    generations = [
        {
            "source_id": "u",
            "prompt_token_ids": [[1, 0]],
            "token_ids": [[2, 3], [0, 4]],
            "log_probs": [-0.2, -0.3],
            "stream_mask": [[1, 1], [0, 1]],
            "action_mask": [1, 1],
            "audio": "generated.wav",
        }
        for _ in range(2)
    ]
    batch = collate_rollouts(sources, generations, 2)
    assert batch.batch["responses"].shape == (2, 2, 2)
    assert (
        batch.select_idxs([1]).non_tensor_batch["generation"][0]["audio"]
        == "generated.wav"
    )
    assert batch.batch["response_stream_mask"][0, 1, 0].item() == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.5])
@pytest.mark.parametrize("field", ["action_mask", "stream_mask"])
def test_rollout_rejects_nonbinary_sampling_masks(field, invalid):
    generation = {
        "source_id": "u",
        "prompt_token_ids": [[1, 0]],
        "token_ids": [[2, 3], [0, 0]],
        "log_probs": [-0.2, 0.0],
        "stream_mask": [[1, 1], [0, 0]],
        "action_mask": [1, 0],
    }
    generation[field][0] = invalid if field == "action_mask" else [1, invalid]
    with pytest.raises(ValueError, match="finite binary"):
        collate_rollouts([{"id": "u"}], [generation] * 2, 2)


def test_forced_multistream_frame_cannot_enter_reward_denominator():
    generation = {
        "source_id": "u",
        "prompt_token_ids": [[1, 0]],
        "token_ids": [[2, 3], [0, 0]],
        "log_probs": [-0.2, 0.0],
        "stream_mask": [[1, 1], [0, 0]],
        "action_mask": [1, 1],
    }
    with pytest.raises(ValueError, match="differs from sampled streams"):
        collate_rollouts([{"id": "u"}], [generation] * 2, 2)
    generation["action_mask"] = [1, 0]
    batch = collate_rollouts([{"id": "u"}], [generation] * 2, 2)
    assert batch.batch["response_mask"].sum().item() == 2


@pytest.mark.parametrize("changed_mask", [False, True])
def test_trainer_rejects_native_scorer_action_identity_drift(changed_mask):
    generation = {
        "source_id": "u",
        "prompt_token_ids": [1],
        "token_ids": [2, 0],
        "log_probs": [-0.2, 0.0],
        "action_mask": [1, 0],
    }
    batch = collate_rollouts([{"id": "u"}], [generation] * 2, 2)

    class Scorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, part):
            mask = part.batch["response_mask"]
            native_mask = torch.ones_like(mask) if changed_mask else mask.clone()
            return TokenScores(
                torch.zeros_like(mask), torch.zeros_like(mask), native_mask
            )

    trainer = NativeGRPOTrainer.__new__(NativeGRPOTrainer)
    trainer.config = SimpleNamespace(micro_batch_size=1, micro_batch_tokens=None)
    if changed_mask:
        with pytest.raises(ValueError, match="differs from the sampled-action mask"):
            trainer._score_chunks(Scorer(), batch)
    else:
        assert trainer._score_chunks(Scorer(), batch).shape == (2, 2)


def test_clip_fraction_includes_all_microbatches():
    generations = [
        {
            "source_id": "u",
            "prompt_token_ids": [1],
            "token_ids": [2],
            "log_probs": [0.0],
        }
        for _ in range(2)
    ]
    batch = collate_rollouts([{"id": "u"}], generations, 2)
    batch.non_tensor_batch["source"][0]["coefficient"] = 0.0
    batch.non_tensor_batch["source"][1]["coefficient"] = np.log(1.5)
    batch.batch.update(
        old_log_probs=torch.zeros(2, 1),
        ref_log_probs=torch.zeros(2, 1),
        advantages=torch.ones(2, 1),
    )

    class Scorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, part):
            coefficients = torch.tensor(
                [source["coefficient"] for source in part.non_tensor_batch["source"]],
                dtype=torch.float32,
            )
            mask = part.batch["response_mask"]
            return TokenScores(
                (self.weight * coefficients).unsqueeze(1),
                torch.zeros_like(mask),
                mask,
            )

    config = ESPnetRLConfig(
        micro_batch_size=1,
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        kl_coef=0.0,
    )
    metrics = NativeGRPOTrainer(Scorer(), config).update_actor(batch)
    assert metrics["actor/clip_fraction"] == pytest.approx(0.5)
