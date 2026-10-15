"""Check WEST think examples, malformed-tag boundaries and MMAU scoring."""

import pytest

from espnet2.speechlm.model.speechlm.rl.bagpiper.data import OPTION_LABEL
from espnet2.speechlm.model.speechlm.rl.bagpiper.rewards import (
    mmau_match,
    qa_think_reward,
)


@pytest.mark.parametrize(
    "prediction,expected",
    [
        ("<answer>Car</answer>", 0.9),
        ("caption <think>reason</think><answer>Car</answer>", 1.0),
        ("<answer>Car</answer><answer>Car</answer>", 0.8),
        ("<answer>\nCar\n</answer>", 0.1),
        ("<think>reason</think>", 0.1),
        ("Car", 0.8),
        ("<think>x</think><think>y</think><answer>Car</answer>", 0.9),
        ("<answer>Car", 0.0),
    ],
)
def test_west_think_reward(prediction, expected):
    assert qa_think_reward({"reference": "Car"}, {"text": prediction}) == pytest.approx(
        expected
    )


VOICE = {
    "reference": "Man",
    # The local MMAU parquet keeps option letters; the official set does not.
    "choices": ["(A) Man", "(B) Woman", "(C) Child", "(D) Robot"],
}


@pytest.mark.parametrize(
    "prediction,expected",
    [
        # Saved Bagpiper responses that the previous whole-text matcher scored 0.
        ("caption... a woman? no.</think>\n\n<answer>(A) Man</answer>", True),
        ("<answer> Man </answer>", True),
        ("Option elimination: not Woman, not Robot.\n<answer>\nMan\n</answer>", True),
        # The official matcher has no letter mapping and rejects wrong options.
        ("<answer>\nA\n</answer>", False),
        ("<answer>(B) Woman</answer>", False),
        ("<answer>Man or Woman</answer>", False),
        # Without a tag the whole response is scored, as the official pipeline does.
        ("The final answer is (A) Man", True),
        ("It is a man, not a woman.", False),
        ("", False),
    ],
)
def test_mmau_scores_the_extracted_answer_against_label_free_options(
    prediction, expected
):
    assert mmau_match(VOICE, {"text": prediction}) is expected


def test_mmau_choices_that_contain_an_article_still_match():
    source = {"reference": "A woman", "choices": ["(A) A man", "(B) A woman"]}
    assert mmau_match(source, {"text": "<answer>(B) A woman</answer>"})
    assert not mmau_match(source, {"text": "<answer>(A) A man</answer>"})


@pytest.mark.parametrize(
    "value,expected",
    [("(A) Man", "Man"), ("D. Train", "Train"), ("Dove", "Dove"), ("(AB) x", "(AB) x")],
)
def test_option_labels_are_removed_once(value, expected):
    assert OPTION_LABEL.sub("", value) == expected
