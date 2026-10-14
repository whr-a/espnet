"""Independent analytical checks of the native library boundary."""

from types import SimpleNamespace

import numpy as np
import torch

from espnet2.rl.data import collate_rollouts
from espnet2.rl.miles_library import (
    compute_grpo_outcome_advantage, compute_policy_loss_vanilla, kl_penalty,
)


def test_group_baselines_and_forced_padding():
    rewards = torch.tensor([[0., 1., 0.], [0., 3., 0.], [2., 0., 0.], [2., 0., 0.]])
    mask = torch.tensor([[1., 1., 0.], [1., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
    advantage, returns = compute_grpo_outcome_advantage(rewards, mask, np.array(["a", "a", "b", "b"]), False)
    expected = torch.tensor([[-1., -1., 0.], [1., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected)


def test_ppo_clipping_and_gradient_against_analytical_oracle():
    current = torch.tensor([[np.log(1.4), np.log(0.6)]], dtype=torch.float32, requires_grad=True)
    config = SimpleNamespace(clip_ratio_low=.2, clip_ratio_high=.2, global_batch_info={"batch_num_tokens": 2, "dp_size": 1})
    loss, _ = compute_policy_loss_vanilla(torch.zeros_like(current), current, torch.tensor([[1., -1.]]), torch.ones_like(current), "token-mean", config)
    # Positive advantage caps at 1.2 and negative advantage floors at 0.8.
    torch.testing.assert_close(loss, torch.tensor(-.2))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
    current2 = torch.tensor([[.05, -.05]], requires_grad=True)
    loss2, _ = compute_policy_loss_vanilla(torch.zeros_like(current2), current2, torch.tensor([[1., -1.]]), torch.ones_like(current2), "token-mean", config)
    loss2.backward()
    torch.testing.assert_close(current2.grad, torch.tensor([[-.5*np.exp(.05), .5*np.exp(-.05)]], dtype=torch.float32))


def test_kl_k3_against_formula():
    current = torch.tensor([-.3, -.9], requires_grad=True)
    reference = torch.tensor([-.5, -.7])
    expected = torch.exp(reference-current)-1-(reference-current)
    torch.testing.assert_close(kl_penalty(current, reference, "k3"), expected)


def test_multistream_batch_retains_audio_reward_evidence():
    sources = [{"id": "u", "reference": "hello"}]
    generations = [{"source_id": "u", "prompt_token_ids": [[1, 0]], "token_ids": [[2, 3], [0, 4]], "log_probs": [-.2, -.3], "stream_mask": [[1, 1], [0, 1]], "action_mask": [1, 1], "audio": "generated.wav"} for _ in range(2)]
    batch = collate_rollouts(sources, generations, 2)
    assert batch.batch["responses"].shape == (2, 2, 2)
    assert batch.select_idxs([1]).non_tensor_batch["generation"][0]["audio"] == "generated.wav"
    assert batch.batch["response_stream_mask"][0, 1, 0].item() == 0
