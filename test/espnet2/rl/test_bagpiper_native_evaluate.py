"""Native QA scoring keeps historical evaluation separate from RL extraction."""

import copy
import inspect

import pytest

from espnet2.speechlm.model.speechlm.rl.bagpiper.evaluate import (
    score_mmau,
    validate_qa_protocol,
    validate_saved_qa_protocol,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.qa_contract import (
    BAGPIPER_QA_SYSTEM_PROMPT,
    native_qa_prompt,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.qa_legacy_eval import parse_prediction


def sources():
    rows = [
        {
            "id": "wind",
            "task": "qa",
            "question": "Which is louder?",
            "choices": ["Wind", "Rain"],
            "reference": "Rain",
            "categories": {"task": "sound"},
        },
        {
            "id": "music",
            "task": "qa",
            "question": "Which instrument?",
            "choices": ["Piano", "Guitar"],
            "reference": "Piano",
            "categories": {"task": "music"},
        },
    ]
    for row in rows:
        row.update(
            qa_template="bagpiper",
            system_prompt=BAGPIPER_QA_SYSTEM_PROMPT,
            prompt=native_qa_prompt(row["question"], row["choices"]),
        )
    return rows


def test_historical_parser_keeps_last_marker_and_mapping_behaviors():
    assert tuple(inspect.signature(parse_prediction).parameters) == ("text", "choices")
    choices = ["Wind", "Rain"]
    assert parse_prediction(
        "The correct option is: Wind\nThe correct option is: Rain", choices
    ) == ("Rain", "Rain", "exact")
    assert parse_prediction("The correct option is: Rain is louder", choices) == (
        "Rain",
        "Rain is louder",
        "substring",
    )
    assert parse_prediction("A caption mentions rain only", choices) == (
        "Rain",
        None,
        "fulltext-unique",
    )
    assert parse_prediction("A caption mentions wind and rain", choices) == (
        None,
        None,
        None,
    )
    # Whole-caption substring mapping is retained for historical comparability,
    # including behavior unsuitable for the stricter training reward.
    assert parse_prediction("A woman speaks", ["Man", "Woman"])[0] is None


def test_native_primary_and_strict_reward_diagnostic_are_distinct():
    rows = sources()
    texts = [
        "Wind and rain are audible.\n<think>Compare the levels.</think>\n"
        "The correct option is: Rain",
        "The piano dominates, unlike the guitar.\n<answer>\nPiano\n</answer>",
    ]
    generations = [
        {"source_id": row["id"], "text": text, "finish_reason": "stop"}
        for row, text in zip(rows, texts)
    ]
    items, summary = score_mmau(rows, generations, 1, "bagpiper_native")
    assert summary["primary_metric"] == "native_choice_accuracy"
    assert summary["native_choice_correct"] == 1
    assert summary["native_choice_accuracy"] == 0.5
    assert summary["native_strict_correct"] == 2
    assert summary["native_strict_accuracy"] == 1.0
    assert summary["native_strict_format_count"] == 2
    assert summary["native_strict_format_rate"] == 1.0
    assert items[0]["native_choice"] == "Rain"
    assert items[0]["native_raw_answer"] == "Rain"
    assert items[0]["native_choice_path"] == "exact"
    assert items[1]["native_choice"] is None
    assert items[1]["native_strict"]["predicted_choice"] == "Piano"
    assert summary["by_category"]["task"]["sound"]["native_choice_accuracy"] == 1
    assert summary["by_category"]["task"]["music"]["native_choice_accuracy"] == 0
    # The original WEST and V1 metrics remain explicit diagnostics.
    _, west = score_mmau(rows, generations, 1)
    assert west["primary_metric"] == "west_accuracy"
    for key in ("west_correct", "west_accuracy", "v1_correct", "v1_accuracy"):
        assert summary[key] == west[key]
    assert "native_choice_accuracy" not in west


def test_ground_truth_does_not_change_any_extracted_prediction():
    rows = sources()
    generations = [
        {"text": "The correct option is: Rain", "finish_reason": "stop"},
        {"text": "<answer>Piano</answer>", "finish_reason": "stop"},
    ]
    first, _ = score_mmau(rows, generations, 1, "bagpiper_native")
    changed = copy.deepcopy(rows)
    changed[0]["reference"] = "Wind"
    changed[1]["reference"] = "Guitar"
    second, _ = score_mmau(changed, generations, 1, "bagpiper_native")
    for before, after in zip(first, second):
        for key in (
            "native_choice",
            "native_raw_answer",
            "native_choice_path",
            "native_strict",
        ):
            assert before[key] == after[key]
    assert first[0]["native_choice_correct"]
    assert not second[0]["native_choice_correct"]


def test_native_prompt_identity_and_saved_generation_cannot_be_relabelled():
    rows = sources()
    digest = validate_qa_protocol(rows, "bagpiper_native")
    assert len(digest) == 64
    for field, value in (
        ("qa_template", "default"),
        ("prompt", "Answer in a tag"),
        ("system_prompt", "You are helpful"),
        ("question", "A different question"),
    ):
        damaged = copy.deepcopy(rows)
        damaged[0][field] = value
        with pytest.raises(ValueError, match="exact bagpiper"):
            validate_qa_protocol(damaged, "bagpiper_native")
    with pytest.raises(ValueError, match="requires --qa-protocol"):
        validate_qa_protocol(rows, "west")
    with pytest.raises(ValueError, match="QA protocol differs"):
        validate_saved_qa_protocol({}, "bagpiper_native", digest)
    for result in (
        {"qa_protocol": "bagpiper_native"},
        {"qa_protocol": "bagpiper_native", "prompt_contract_sha256": "wrong"},
    ):
        with pytest.raises(ValueError, match="matching prompt contract"):
            validate_saved_qa_protocol(result, "bagpiper_native", digest)
    validate_saved_qa_protocol(
        {"qa_protocol": "bagpiper_native", "prompt_contract_sha256": digest},
        "bagpiper_native",
        digest,
    )
    # Old frozen WEST response files have no new metadata and remain readable.
    validate_saved_qa_protocol({}, "west", "unused")
    with pytest.raises(ValueError, match="QA protocol differs"):
        validate_saved_qa_protocol({"qa_protocol": "bagpiper_native"}, "west", digest)
