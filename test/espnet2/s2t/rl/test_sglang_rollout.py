"""The OWSM rollout engine's requests, without starting SGLang."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from espnet2.rl.config import ESPnetRLConfig
from espnet2.s2t.rl.sglang_rollout import OWSMEngine

TOKENS = ["<blank>", "<sos>", "<eos>", "<eng>", "<asr>", "<notimestamps>", "a"]


class Recorder:
    """Stands in for sglang.Engine and answers every request in order."""

    def __init__(self, reorder=False):
        self.calls, self.reorder = [], reorder

    def generate(self, **request):
        self.calls.append(request)
        outputs = [
            {
                "output_ids": [6, 2],
                "meta_info": {
                    "id": rid,
                    "output_token_logprobs": [[-0.5, 6, None], [-0.1, 2, None]],
                    "finish_reason": {"type": "stop"},
                },
            }
            for rid in request["rid"]
        ]
        return outputs[::-1] if self.reorder else outputs


def engine(tmp_path, recorder=None, replica=0):
    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="espnet2.s2t.rl.sglang_rollout:OWSMEngine",
        group_size=2,
        rollout_mem_fraction=0.2,
        sglang_engine_kwargs={"json_model_override_args": '{"keep": 1}'},
    )
    result = OWSMEngine.__new__(OWSMEngine)
    result.config, result.directory, result.replica = config, tmp_path, replica
    result.engine = recorder or Recorder()
    result.metadata = {"espnet_config": {"token_list": TOKENS}, "eos_token_id": 2}
    result.text = lambda ids: " ".join(TOKENS[token] for token in ids)
    return result


def test_settings_name_the_sources_file_and_the_replica_seed(tmp_path):
    settings = engine(tmp_path, replica=3).settings(tmp_path / "hf")
    assert settings["model_path"] == str(tmp_path / "hf")
    assert settings["random_seed"] == 42 + 3
    assert settings["mem_fraction_static"] == 0.2
    assert json.loads(settings["json_model_override_args"]) == {
        "keep": 1,
        "espnet_rollout_sources": str(tmp_path / "sources.json"),
    }


def test_requests_carry_real_prefixes_and_name_their_source(tmp_path):
    recorder = Recorder()
    sources = [{"id": "x"}, {"id": "y", "prompt_token_ids": [1, 3, 4]}]
    generations = engine(tmp_path, recorder).generate(sources, 7)
    (request,) = recorder.calls
    assert request["rid"] == ["7:0:0", "7:0:1", "7:1:0", "7:1:1"]
    assert request["input_ids"] == [[1, 3, 4, 5]] * 2 + [[1, 3, 4]] * 2
    assert request["logprob_start_len"] == [3, 3, 2, 2]
    assert json.loads((tmp_path / "sources.json").read_text()) == {
        "rollout_id": 7,
        "sources": sources,
    }
    assert not (tmp_path / "sources.tmp").exists()
    assert [row["source_id"] for row in generations] == ["x", "x", "y", "y"]
    assert generations[2]["prompt_token_ids"] == [1, 3, 4]
    assert generations[0]["log_probs"] == [-0.5, -0.1]
    assert generations[0]["text"] == "a <eos>"


def test_release_empties_the_idle_engine_cache(tmp_path):
    flushes = []

    async def flush_cache(timeout_s=None):
        flushes.append(timeout_s)
        return SimpleNamespace(success=len(flushes) == 1, message="busy")

    recorder = Recorder()
    recorder.loop = asyncio.new_event_loop()
    recorder.tokenizer_manager = SimpleNamespace(flush_cache=flush_cache)
    try:
        engine(tmp_path, recorder).release()
        with pytest.raises(RuntimeError, match="busy"):
            engine(tmp_path, recorder).release()
    finally:
        recorder.loop.close()
    # Each release lets the scheduler retire its last batch before flushing.
    assert flushes == [60, 60]


def test_answers_out_of_request_order_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="incomplete OWSM prompt groups"):
        engine(tmp_path, Recorder(reorder=True)).generate([{"id": "x"}], 0)
