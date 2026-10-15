"""The WER every family's reward and heldout evaluation share, and the reward hook."""

import pytest

from espnet2.rl.data import collate_rollouts
from espnet2.rl.rewards import (
    outcome_rewards,
    word_error_rate,
    word_error_reward,
    words,
)


def test_punctuation_and_case_are_not_word_errors():
    # A real OWSM v4 dev-clean hypothesis against its LibriSpeech reference:
    # every former "error" was punctuation or the hyphen in to-morrow.
    reference = (
        "HE WOULD SEE HOW THINGS TURNED OUT TO MORROW AND THEN HE WOULD TALK TO"
        " HER THEY WERE GOING TO MEET AS USUAL"
    )
    hypothesis = (
        "He would see how things turned out to-morrow, and then he would talk to"
        " her. They were going to meet as usual."
    )
    assert word_error_rate(reference, hypothesis) == 0
    assert word_error_reward({"reference": reference}, {"text": hypothesis}) == 0


def test_real_word_errors_still_count():
    assert word_error_rate("THE CAT SAT", "the cat sat down") == pytest.approx(1 / 3)
    assert word_error_rate("THE CAT SAT", "a cat") == pytest.approx(2 / 3)
    assert word_error_rate("WIFE'S ACTION", "wife's action.") == 0
    # An empty reference counts every hypothesis word against one word.
    assert word_error_rate("", "anything else") == 2


def test_words_keep_letters_and_digits_of_any_script():
    assert words("Café, naïve 42!") == ["café", "naïve", "42"]
    assert words("你好，世界") == ["你好", "世界"]


def candidates(texts, group_size=2):
    sources = [
        {"id": f"prompt-{index}", "reference": "the cat"}
        for index in range(len(texts) // group_size)
    ]
    generations = [
        {
            "source_id": sources[index // group_size]["id"],
            "token_ids": [7] * (index + 1),
            "log_probs": [-1.0] * (index + 1),
            "prompt_token_ids": [1],
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    return collate_rollouts(sources, generations, group_size)


def scored(source, generation):
    correct = float(generation["text"] == source["reference"])
    return {
        "reward": correct - 0.5,
        "acc": correct,
        "length": len(generation["token_ids"]),
    }


def test_reward_components_are_logged_as_batch_means():
    batch = candidates(["the cat", "a cat", "the cat", "dog"])
    rewards, token_rewards, metrics = outcome_rewards(batch, f"{__name__}:scored")
    assert rewards.tolist() == [0.5, -0.5, 0.5, -0.5]
    # Each reward sits on its candidate's last action.
    assert token_rewards[2].tolist() == [0, 0, 0.5, 0]
    assert metrics == {
        "reward/mean": 0.0,
        "reward/min": -0.5,
        "reward/max": 0.5,
        "reward/acc": 0.5,
        "reward/length": 2.5,
    }
    # A plain float reward logs only its own statistics.
    _, _, metrics = outcome_rewards(batch)
    assert set(metrics) == {"reward/mean", "reward/min", "reward/max"}


def scored_together(sources, generations):
    scored_together.calls.append(len(generations))
    return [-float(len(generation["text"])) for generation in generations]


scored_together.batched = True
scored_together.calls = []


def unequal(source, generation):
    return {"reward": 0.0, **({"acc": 1.0} if generation["text"] == "a" else {})}


def test_batched_rewards_see_every_candidate_at_once_and_components_must_agree():
    batch = candidates(["a", "bb", "ccc", "dddd"])
    rewards, _, _ = outcome_rewards(batch, f"{__name__}:scored_together")
    assert scored_together.calls == [4]
    assert rewards.tolist() == [-1, -2, -3, -4]
    with pytest.raises(ValueError, match="same reward components"):
        outcome_rewards(batch, f"{__name__}:unequal")
