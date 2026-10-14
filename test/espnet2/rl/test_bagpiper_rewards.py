"""Check WEST think examples and malformed-tag reward boundaries."""

import pytest

from espnet2.speechlm.model.speechlm.rl.bagpiper.rewards import qa_think_reward


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
