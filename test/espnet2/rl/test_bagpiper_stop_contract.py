"""Bagpiper keeps every action when SGLang's length cap and EOS coincide."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from espnet2.rl.data import collate_rollouts
from espnet2.speechlm.model.speechlm.rl.bagpiper.integration import (
    prepared_native_metadata,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.rollout import BagpiperEngine
from espnet2.speechlm.model.speechlm.rl.bagpiper.tokens import TEXT


def record(tokens, finish, cap=2048, mutate=None):
    engine = BagpiperEngine.__new__(BagpiperEngine)
    engine.geometry = {
        "eos_token_id": 2,
        "num_stream": 4,
        "text_token_offset": 16,
        "text_token_end": 48,
    }
    engine.config = SimpleNamespace(training_objective="west_grpo", top_k=50)
    engine.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "final")
    source = {"id": "qa", "mode": "text_only"}
    output = {
        "output_ids": tokens,
        "meta_info": {
            "id": "15:3:7",
            "finish_reason": {
                "type": finish,
                **({"length": cap} if finish == "length" else {}),
            },
            "output_token_logprobs": [[-0.2, token, None] for token in tokens],
            "espnet_bagpiper": [[TEXT, 1] for _ in tokens],
        },
    }
    if mutate is not None:
        mutate(output)
    before = copy.deepcopy(output)
    result = engine._record(
        source,
        [[6, 0, 0, 0]],
        output,
        prepared_native_metadata(source),
        {"max_new_tokens": cap, "temperature": 0.7},
    )
    assert output == before
    return source, result


def test_exact_cap_eos_is_complete_and_keeps_the_sampled_loss_action():
    source, result = record([18] * 2047 + [2], "length")
    assert result["finish_reason"] == "stop"
    assert result["backend_finish_reason"] == {"type": "length", "length": 2048}
    assert len(result["token_ids"]) == 2048
    assert result["token_ids"][-1] == [2, 0, 0, 0]
    assert result["log_probs"][-1] == -0.2
    assert result["stream_log_probs"][-1] == [-0.2, 0.0, 0.0, 0.0]
    assert result["action_mask"][-1]
    assert result["stream_mask"][-1] == [True, False, False, False]
    assert result["support"][-1] == TEXT
    batch = collate_rollouts([source], [result], 1)
    assert batch.batch["responses"][0, -1, 0] == 2
    assert batch.batch["response_lengths"].tolist() == [2048]
    assert batch.batch["response_mask"][0, -1] == 1
    assert batch.batch["response_stream_mask"][0, -1].tolist() == [1, 0, 0, 0]
    torch.testing.assert_close(
        batch.batch["rollout_log_probs"][0, -1], torch.tensor(-0.2)
    )


@pytest.mark.parametrize(
    "tokens,finish,cap",
    [([18, 2], "stop", 4), ([18, 18, 18, 2], "stop", 4), ([18] * 4, "length", 4)],
)
def test_ordinary_stop_and_real_truncation_keep_their_original_schema(
    tokens, finish, cap
):
    _, result = record(tokens, finish, cap)
    assert result["finish_reason"] == finish
    assert "backend_finish_reason" not in result
    assert [frame[0] for frame in result["token_ids"]] == tokens
    assert result["action_mask"] == [True] * len(tokens)


@pytest.mark.parametrize(
    "tokens,finish,cap",
    [
        ([18, 2], "length", 4),
        ([18, 18], "length", 4),
        ([18, 2, 18, 2], "length", 4),
        ([18, 2, 18, 2], "stop", 4),
        ([18] * 4, "stop", 4),
        ([18] * 4 + [2], "stop", 4),
        ([18] * 5, "length", 4),
        ([18, 2], "abort", 4),
    ],
)
def test_invalid_stop_contract_reports_request_and_boundary_evidence(
    tokens, finish, cap
):
    with pytest.raises(RuntimeError, match="invalid Bagpiper request: ") as caught:
        record(tokens, finish, cap)
    diagnostic = json.loads(str(caught.value).split(": ", 1)[1])
    assert diagnostic["id"] == "15:3:7"
    assert diagnostic["finish_reason"]["type"] == finish
    assert diagnostic["frames"] == len(tokens)
    assert diagnostic["max_new_tokens"] == cap
    assert diagnostic["eos_positions"] == [
        i for i, token in enumerate(tokens) if token == 2
    ]
    assert diagnostic["tail_tokens"] == tokens[-8:]


@pytest.mark.parametrize("problem", ["missing", "short", "logprobs", "empty"])
def test_cap_eos_does_not_bypass_incomplete_trace_rejection(problem):
    def mutate(output):
        if problem == "missing":
            output["meta_info"].pop("espnet_bagpiper")
        elif problem == "short":
            output["meta_info"]["espnet_bagpiper"].pop()
        elif problem == "logprobs":
            output["meta_info"]["output_token_logprobs"].pop()
        else:
            output["output_ids"] = []

    with pytest.raises(RuntimeError, match="incomplete Bagpiper action trace"):
        record([18, 18, 18, 2], "length", 4, mutate)


@pytest.mark.parametrize("problem", ["token", "nan", "inactive", "retracted"])
def test_cap_eos_keeps_primary_action_integrity_guards(problem):
    def mutate(output):
        info = output["meta_info"]
        if problem == "token":
            info["output_token_logprobs"][-1][1] = 18
        elif problem == "nan":
            info["output_token_logprobs"][-1][0] = float("nan")
        elif problem == "inactive":
            info["espnet_bagpiper"][-1][1] = 0
        else:
            info["num_retractions"] = 1

    with pytest.raises(RuntimeError, match="primary action|retracted"):
        record([18, 18, 18, 2], "length", 4, mutate)
