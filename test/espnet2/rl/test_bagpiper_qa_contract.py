"""Native SFT final-answer boundaries, independent of a reference answer."""

import hashlib

import pytest

from espnet2.speechlm.model.speechlm.rl.bagpiper.qa_contract import (
    BAGPIPER_QA_SYSTEM_PROMPT,
    match_choice,
    native_qa_prompt,
    parse_native_qa_answer,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.rewards import qa_bagpiper_v2_reward

CHOICES = ["Car", "Dog", "Bird", "Train"]


def test_prompt_preserves_native_sft_format():
    # SHA of SYSTEM_PROMPT in the preexisting Bagpiper eval/prompt_common.py.
    assert hashlib.sha256(BAGPIPER_QA_SYSTEM_PROMPT.encode()).hexdigest() == (
        "c32b7ed5afe9418833a402d8c42abae1240e999d0ab16f079d6ea9061b51cca8"
    )
    assert native_qa_prompt("  What is audible?  ", [" Car ", "Dog"]) == (
        "question: What is audible?\n Options: Car; Dog"
    )


@pytest.mark.parametrize(
    "text,index,path,answer_format",
    [
        (
            "Caption: a car is not audible, but a dog barks.\n"
            "Based on this analysis, the correct option is: Dog",
            1,
            "native_cue",
            True,
        ),
        (
            "Caption: not a car. The correct option is:\n\n**“(B) Dog.”**\n",
            1,
            "native_cue",
            True,
        ),
        (
            "<think>Car and Train are distractors.</think>\n\n"
            "<answer>\n Dog \n</answer>\n",
            1,
            "answer_tag",
            True,
        ),
        ("Reasoning from a renderer prefill.</think>\nDog", 1, "post_think", False),
        ("Dog", 1, "plain", False),
        ("The correct option is: (B)", 1, "native_cue", True),
        ("The correct option is: B.", 1, "native_cue", True),
        ("<answer>Ｄｏｇ</answer>", 1, "answer_tag", True),
        (
            "The correct option is: <answer>Dog</answer>",
            1,
            "answer_tag",
            True,
        ),
        (
            "The correct option is: Dog\nIt barks.",
            1,
            "native_cue",
            False,
        ),
    ],
)
def test_native_final_answer_ignores_prefix_distractors(
    text, index, path, answer_format
):
    parsed = parse_native_qa_answer(text, CHOICES)
    assert parsed["status"] == "ok"
    assert parsed["predicted_index"] == index
    assert parsed["predicted_choice"] == CHOICES[index]
    assert parsed["path"] == path
    assert parsed["format"] is answer_format


@pytest.mark.parametrize(
    "text",
    [
        "<answer>Dog</answer><answer>Car</answer>",
        "<answer>Dog</answer><answer>Dog</answer>",
        "<answer><answer>Dog</answer></answer>",
        "<answer>Dog",
        "Dog</answer>",
        "<answer class='final'>Dog</answer>",
        "<think>A dog barks. The correct option is: Dog",
        "<think>Dog</think><think>Car</think><answer>Dog</answer>",
        "The correct option is: Dog\nThe correct option is: Car",
        "The correct option is: Car\n<answer>Dog</answer>",
        "<answer>Dog</answer>\nCar",
        "The correct option is: Dog\nCar",
        "The correct option is:",
        "<answer> </answer>",
    ],
)
def test_malformed_and_contradictory_answers_fail_closed(text):
    parsed = parse_native_qa_answer(text, CHOICES)
    assert parsed["predicted_index"] is None
    assert parsed["predicted_choice"] is None
    assert parsed["format"] is False
    assert parsed["status"] != "ok"


@pytest.mark.parametrize(
    "answer,index",
    [("a minor", 0), ("A minor.", 0), ("b flat", 1), ("**B flat**", 1)],
)
def test_musical_option_text_is_not_a_bare_letter_label(answer, index):
    choices = ["a minor", "b flat", "c sharp", "d major"]
    parsed = parse_native_qa_answer("The correct option is: " + answer, choices)
    assert parsed["predicted_index"] == index
    assert parsed["format"] is True


@pytest.mark.parametrize(
    "answer,index,method",
    [
        ("(B) Dog", 1, "label_text"),
        ("b", 1, "label"),
        ("(A) Dog", None, "contradictory_label"),
        ("Dog or Car", None, "unknown_answer"),
        ("A dog barks", None, "unknown_answer"),
    ],
)
def test_mapping_is_exact_and_labels_must_agree(answer, index, method):
    assert match_choice(answer, CHOICES) == (index, method)


def test_explicit_choice_labels_are_not_assumed_to_be_positional():
    assert match_choice("B", ["(B) Dog", "(A) Car"]) == (0, "label")
    assert match_choice("(A) Dog", ["(B) Dog", "(A) Car"]) == (
        None,
        "contradictory_label",
    )


def test_caption_does_not_supply_an_answer_without_a_terminal_declaration():
    parsed = parse_native_qa_answer("Caption: a dog barks in the distance.", CHOICES)
    assert parsed["predicted_index"] is None
    assert parsed["format"] is False


def test_unmapped_choice_does_not_get_fuzzy_credit():
    parsed = parse_native_qa_answer("The correct option is: Dog barking", CHOICES)
    assert parsed["predicted_index"] is None
    assert parsed["status"] == "unmapped_answer"


def test_duplicate_normalized_choices_are_ambiguous():
    parsed = parse_native_qa_answer("The correct option is: Dog", ["Dog", "dog."])
    assert parsed["status"] == "invalid_choices"
    assert parsed["predicted_index"] is None
    assert parsed["format"] is False


@pytest.mark.parametrize(
    "text,accuracy,answer_format,total",
    [
        ("Caption: not Car. The correct option is: Dog", 1.0, 1.0, 1.1),
        ("<think>Not Car.</think><answer>Dog</answer>", 1.0, 1.0, 1.1),
        ("The correct option is: Car", 0.0, 1.0, 0.1),
        ("Dog", 1.0, 0.0, 1.0),
        ("<answer>Dog</answer><answer>Car</answer>", 0.0, 0.0, 0.0),
        ("The correct option is: (A) Dog", 0.0, 0.0, 0.0),
    ],
)
def test_native_reward_weights_and_independent_format(
    text, accuracy, answer_format, total
):
    reward = qa_bagpiper_v2_reward(
        {"choices": CHOICES, "reference": "Dog"}, {"text": text}
    )
    assert reward["acc"] == accuracy
    assert reward["format"] == answer_format
    assert reward["reward"] == pytest.approx(total)


def test_reference_cannot_change_extraction_or_format():
    text = "Caption: a dog barks. The correct option is: Car"
    parsed = parse_native_qa_answer(text, CHOICES)
    correct = qa_bagpiper_v2_reward(
        {"choices": CHOICES, "reference": "Car"}, {"text": text}
    )
    wrong = qa_bagpiper_v2_reward(
        {"choices": CHOICES, "reference": "Dog"}, {"text": text}
    )
    assert parsed["predicted_index"] == 0
    assert parse_native_qa_answer(text, CHOICES) == parsed
    assert correct["acc"] == 1.0
    assert wrong["acc"] == 0.0
    assert {
        key: value for key, value in correct.items() if key not in ("acc", "reward")
    } == {key: value for key, value in wrong.items() if key not in ("acc", "reward")}
