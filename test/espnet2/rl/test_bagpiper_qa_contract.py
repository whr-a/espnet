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
    "declaration",
    [
        "**Correct Option:** Dog",
        "**Correct Option**: Dog",
        "**Correct Option: **Dog",
        "Correct Option:\n\n**Dog**\n",
        "### Correct Option: (B) Dog",
        "  **The Correct Option:** B.",
    ],
)
def test_markdown_option_header_accepts_exact_inline_or_next_line_answer(declaration):
    text = "Caption: a dog barks, not a car or a train.\n" + declaration
    parsed = parse_native_qa_answer(text, CHOICES)
    assert parsed["predicted_index"] == 1
    assert parsed["path"] == "option_header"
    assert parsed["format"] is True
    reward = qa_bagpiper_v2_reward(
        {"choices": CHOICES, "reference": "Dog"}, {"text": text}
    )
    assert reward["reward"] == pytest.approx(1.1)
    assert reward["path_option_header"] == 1.0
    assert sum(value for key, value in reward.items() if key.startswith("path_")) == 1.0


def test_option_header_must_start_a_line_not_a_caption_substring():
    text = "Caption: the words Correct Option: Car are spoken.\nCorrect Option: Dog"
    parsed = parse_native_qa_answer(text, CHOICES)
    assert parsed["predicted_index"] == 1
    assert parsed["format"] is True
    assert (
        parse_native_qa_answer(text.splitlines()[0], CHOICES)["predicted_index"] is None
    )


def test_option_header_inside_closed_reasoning_is_not_a_final_declaration():
    prefix = "<think>Consider the distractor.\n**Correct Option:** Car\n</think>"
    parsed = parse_native_qa_answer(prefix + "\n**Correct Option:** Dog", CHOICES)
    assert parsed["predicted_index"] == 1
    assert parsed["format"] is True
    assert parse_native_qa_answer(prefix, CHOICES)["predicted_index"] is None


@pytest.mark.parametrize(
    "text,status",
    [
        ("<think>Still thinking.\n**Correct Option:** Dog", "malformed_reasoning"),
        ("Correct Option: Dog\nCorrect Option: Car", "conflicting_answers"),
        ("Correct Option: Dog\nThe correct option is: Car", "conflicting_answers"),
        ("The correct option is: Car\nCorrect Option: Dog", "conflicting_answers"),
        ("<answer>Car</answer>\nCorrect Option: Dog", "conflicting_answers"),
        ("Correct Option: Dog\n<answer>Car</answer>", "conflicting_answers"),
        ("Correct Option: Dog\nCorrect Option: Not an option", "ambiguous_answers"),
        ("Correct Option: (A) Dog", "unmapped_answer"),
        ("Correct Option: <answer>Dog", "malformed_answer_tags"),
        ("Correct Option:\n** **", "empty_option_header"),
        ("Correct Option: Dog\nCar", "conflicting_answers"),
    ],
)
def test_option_headers_preserve_reasoning_conflict_and_tag_safeguards(text, status):
    parsed = parse_native_qa_answer(text, CHOICES)
    assert parsed["status"] == status
    assert parsed["predicted_index"] is None
    assert parsed["format"] is False


def test_repeated_agreeing_option_headers_have_no_structural_bonus():
    parsed = parse_native_qa_answer("Correct Option: Dog\nCorrect Option: B", CHOICES)
    assert parsed["predicted_index"] == 1
    assert parsed["format"] is False


def test_option_header_can_introduce_one_answer_block_without_double_counting():
    parsed = parse_native_qa_answer("**Correct Option:** <answer>Dog</answer>", CHOICES)
    assert parsed["predicted_index"] == 1
    assert parsed["path"] == "answer_tag"
    assert parsed["format"] is True


def test_option_header_uses_duplicate_choice_canonical_aliases():
    choices = ["four", "nine", "fourteen", "fourteen"]
    parsed = parse_native_qa_answer("Correct Option: (D) fourteen", choices)
    assert parsed["predicted_index"] == 2
    assert parsed["format"] is True


@pytest.mark.parametrize("answer", ["Car", "Dog barking"])
def test_option_header_format_and_extraction_do_not_use_reference_or_caption(answer):
    text = "Caption: Dog\n**Correct Option:** " + answer
    parsed = parse_native_qa_answer(text, CHOICES)
    assert parsed["predicted_index"] == (0 if answer == "Car" else None)
    rewards = [
        qa_bagpiper_v2_reward({"choices": CHOICES, "reference": ref}, {"text": text})
        for ref in CHOICES
    ]
    assert all(reward["format"] == 1.0 for reward in rewards)
    structural = [
        {key: value for key, value in reward.items() if key not in ("acc", "reward")}
        for reward in rewards
    ]
    assert all(value == structural[0] for value in structural)
    assert rewards[1]["acc"] == 0.0
    assert rewards[1]["reward"] == pytest.approx(0.1)


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


@pytest.mark.parametrize(
    "answer,index",
    [
        ("B", 1),
        ("D", 3),
        ("F", None),
        ("J", None),
        ("F. Scott Fitzgerald", 1),
        ("J. D. Salinger", 3),
        ("(D) J. D. Salinger", 3),
    ],
)
def test_author_initials_do_not_replace_positional_option_labels(answer, index):
    choices = [
        "Mark Twain",
        "F. Scott Fitzgerald",
        "Ernest Hemingway",
        "J. D. Salinger",
    ]
    parsed = parse_native_qa_answer("The correct option is: " + answer, choices)
    assert parsed["predicted_index"] == index


@pytest.mark.parametrize("name", ["J.D. Salinger", "J. D. Salinger"])
def test_repeated_author_initials_are_distinct_literal_options(name):
    choices = [name, "J. K. Rowling"]
    assert match_choice(name, choices) == (0, "exact")
    assert match_choice("J. K. Rowling", choices) == (1, "exact")
    assert match_choice("B", choices) == (1, "label")
    assert match_choice("J", choices) == (None, "unknown_label")


@pytest.mark.parametrize(
    "choices",
    [
        ["A. Car", "B. Dog", "C. Bird", "D. Train"],
        ["(A) Car", "(B) Dog", "(C) Bird", "(D) Train"],
    ],
)
def test_coherent_explicit_catalog_labels_and_contradictions(choices):
    assert match_choice("B", choices) == (1, "label")
    assert match_choice("Dog", choices) == (1, "exact")
    assert match_choice(choices[1], choices) == (1, "exact")
    assert match_choice("(A) Dog", choices) == (None, "contradictory_label")
    assert match_choice("E", choices) == (None, "unknown_label")


def test_coherent_e_to_h_catalog_keeps_its_explicit_labels():
    choices = ["E. Car", "F. Dog", "G. Bird", "H. Train"]
    assert match_choice("F", choices) == (1, "label")
    assert match_choice("B", choices) == (None, "unknown_label")
    assert match_choice("(E) Dog", choices) == (None, "contradictory_label")


def test_repeated_parenthesized_labels_are_invalid():
    assert match_choice("Dog", ["(A) Dog", "(A) Car"]) == (
        None,
        "invalid_choices",
    )


def test_literal_musical_note_precedes_positional_letter_interpretation():
    choices = ["C", "G", "A", "F"]
    assert match_choice("C", choices) == (0, "exact")
    assert match_choice("A", choices) == (2, "exact")


@pytest.mark.parametrize("answer,accuracy", [("B", 1.0), ("F", 0.0)])
def test_author_reference_cannot_turn_an_initial_into_a_valid_label(answer, accuracy):
    source = {
        "choices": [
            "Mark Twain",
            "F. Scott Fitzgerald",
            "Ernest Hemingway",
            "J.D. Salinger",
        ],
        "reference": "F. Scott Fitzgerald",
    }
    reward = qa_bagpiper_v2_reward(source, {"text": "The correct option is: " + answer})
    assert reward["acc"] == accuracy
    assert reward["format"] == 1.0
    assert reward["reward"] == pytest.approx(accuracy + 0.1)


def test_caption_does_not_supply_an_answer_without_a_terminal_declaration():
    parsed = parse_native_qa_answer("Caption: a dog barks in the distance.", CHOICES)
    assert parsed["predicted_index"] is None
    assert parsed["format"] is False


def test_unmapped_choice_does_not_get_fuzzy_credit():
    parsed = parse_native_qa_answer("The correct option is: Dog barking", CHOICES)
    assert parsed["predicted_index"] is None
    assert parsed["status"] == "unmapped_answer"


def test_duplicate_normalized_choices_are_canonical_aliases():
    parsed = parse_native_qa_answer("The correct option is: Dog", ["Dog", "dog."])
    assert parsed["status"] == "ok"
    assert parsed["predicted_index"] == 0
    assert parsed["format"] is True
    assert match_choice("B", ["Dog", "dog."]) == (0, "label")


def test_duplicate_distractor_preserves_a_unique_correct_answer():
    source = {"choices": ["four", "nine", "fourteen", "fourteen"], "reference": "nine"}
    parsed = parse_native_qa_answer("The correct option is: nine", source["choices"])
    assert parsed["predicted_index"] == 1
    reward = qa_bagpiper_v2_reward(source, {"text": "The correct option is: nine"})
    assert reward["acc"] == 1.0
    assert reward["reward"] == pytest.approx(1.1)


@pytest.mark.parametrize("answer", ["C", "D", "fourteen", "(D) fourteen"])
@pytest.mark.parametrize("reference", ["fourteen", "C", "D"])
def test_duplicate_correct_options_and_labels_share_one_canonical_answer(
    answer, reference
):
    source = {
        "choices": ["four", "nine", "fourteen", "fourteen"],
        "reference": reference,
    }
    parsed = parse_native_qa_answer(
        "The correct option is: " + answer, source["choices"]
    )
    assert parsed["predicted_index"] == 2
    assert parsed["predicted_choice"] == "fourteen"
    reward = qa_bagpiper_v2_reward(source, {"text": "The correct option is: " + answer})
    assert reward["acc"] == 1.0
    assert reward["reward"] == pytest.approx(1.1)


@pytest.mark.parametrize("answer", ["C", "D", "fourteen", "(D) fourteen"])
def test_explicit_labels_with_identical_bodies_are_canonical_aliases(answer):
    choices = ["(A) four", "(B) nine", "(C) fourteen", "(D) fourteen"]
    assert match_choice(answer, choices)[0] == 2
    source = {"choices": choices, "reference": "(D) fourteen"}
    reward = qa_bagpiper_v2_reward(source, {"text": "The correct option is: " + answer})
    assert reward["reward"] == pytest.approx(1.1)


def test_explicit_literal_duplicate_returns_the_first_body_index():
    choices = ["A. four", "B. nine", "C. fourteen", "D. fourteen"]
    assert match_choice("D. fourteen", choices) == (2, "exact")


def test_duplicate_bodies_do_not_rescue_a_contradictory_label():
    choices = ["(A) four", "(B) nine", "(C) fourteen", "(D) fourteen"]
    text = "The correct option is: (A) fourteen"
    parsed = parse_native_qa_answer(text, choices)
    assert parsed["predicted_index"] is None
    assert parsed["format"] is False
    assert (
        qa_bagpiper_v2_reward(
            {"choices": choices, "reference": "fourteen"}, {"text": text}
        )["reward"]
        == 0.0
    )


@pytest.mark.parametrize(
    "choices", [[], [""], ["Dog", ""], ["Dog", None], ["(A) Dog", "(A) Dog"]]
)
def test_malformed_choices_and_repeated_explicit_labels_still_fail(choices):
    parsed = parse_native_qa_answer("The correct option is: Dog", choices)
    assert parsed["status"] == "invalid_choices"
    assert parsed["predicted_index"] is None


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
