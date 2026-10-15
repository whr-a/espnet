"""The OpusLM rollout engine's batched requests, without starting SGLang."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.rewards import word_error_rate
from espnet2.speechlm.rl.evaluate import summarize_wer
from espnet2.speechlm.rl.integration import (
    TOKENIZER_FILES,
    OpusLMEngine,
    asr_wer_reward,
)

NQ, VOCAB, TEXT = 9, 100, 90
METADATA = {
    "rl_inference_semantics": "native_pad_cleared",
    "nq": NQ,
    "pad_token_id": 0,
    "eos_token_id": 5,
    "text_bpe_start_end_token_id": 35,
    "text_token_start": TEXT,
    "text_token_end": VOCAB,
}


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 4 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{index}" for index in ids)


class Recorder:
    """Stands in for sglang.Engine; ``answer`` may tamper with its replies."""

    def __init__(self, answer=None):
        self.calls, self.updates = [], []
        self.answer = answer or (lambda outputs: outputs)

    def generate(self, **request):
        self.calls.append(request)
        outputs = [
            {
                "output_ids": [TEXT + 1 + index % 2, 5],
                "meta_info": {
                    "id": rid,
                    "output_token_logprobs": [
                        [-0.5, TEXT + 1 + index % 2, None],
                        [-0.1, 5, None],
                    ],
                    "finish_reason": {"type": "stop", "matched": 5},
                },
            }
            for index, rid in enumerate(request["rid"])
        ]
        return self.answer(outputs)

    def update_weights_from_disk(self, path):
        self.updates.append(path)
        return True, "Succeeded", None


def config(**overrides):
    return ESPnetRLConfig(
        family="opuslm_dialogue",
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="espnet2.speechlm.rl.integration:OpusLMEngine",
        group_size=3,
        max_tokens=7,
        **overrides,
    )


def engine(tmp_path, recorder=None, replica=0, **overrides):
    result = OpusLMEngine.__new__(OpusLMEngine)
    result.config, result.directory = config(**overrides), tmp_path
    result.replica = replica
    result.engine = recorder or Recorder()
    result.metadata = dict(METADATA)
    result.tokenizer, result.tokenizer_files = Tokenizer(), {}
    result.embedding = torch.randn(VOCAB, 4, generator=torch.Generator().manual_seed(0))
    result.context_length = 64
    return result


def frames(*rows):
    return [list(row) + [0] * (NQ - len(row)) for row in rows]


def test_one_call_sends_every_candidate_with_its_own_sampling(tmp_path):
    recorder = Recorder()
    rollout = engine(tmp_path, recorder)
    audio = {
        "id": "a",
        "audio": "a.wav",
        "mode": "audio_text",
        # Pad entries (0) of every stream must contribute nothing.
        "prompt_token_ids": frames([5], [80, 3], [0, 4, 6], [35]),
    }
    text = {"id": "t", "text": "ab"}
    generations = rollout.generate([audio, text], 4)
    (request,) = recorder.calls
    assert request["rid"] == ["4:0:0", "4:0:1", "4:0:2", "4:1:0", "4:1:1", "4:1:2"]
    assert request["return_logprob"] is True
    # Candidates of one prompt share its embedding list instead of copies.
    embeds = request["input_embeds"]
    assert embeds[0] is embeds[1] is embeds[2] and embeds[3] is embeds[5]
    expected = [
        sum(rollout.embedding[token] for token in row if token != 0)
        for row in audio["prompt_token_ids"]
    ]
    assert torch.allclose(torch.tensor(embeds[0]), torch.stack(expected))
    # Dialogue's native text-task layout around the shifted text tokens.
    primary = [5, 88, 9, 35, TEXT + 1, TEXT + 2, *([0] * 8), 10, 35]
    assert generations[3]["prompt_token_ids"] == [[i] + [0] * 8 for i in primary]
    assert len(embeds[3]) == len(primary)
    sampling = request["sampling_params"]
    assert len(sampling) == 6 and not any("n" in params for params in sampling)
    assert {params["max_new_tokens"] for params in sampling} == {7}
    assert {tuple(params["stop_token_ids"]) for params in sampling} == {(5,)}
    # Distinct seeds keep a group's candidates apart under seeded sampling.
    assert len({params["sampling_seed"] for params in sampling}) == 6
    assert [row["source_id"] for row in generations] == ["a"] * 3 + ["t"] * 3
    first = generations[0]
    assert first["token_ids"] == frames([TEXT + 1], [5])
    assert first["log_probs"] == [-0.5, -0.1]
    assert first["text"] == "w1" and generations[1]["text"] == "w2"
    assert first["stream_mask"] == [[True] + [False] * 8] * 2
    assert first["sampling_contracts"][0][0]["allowed_ranges"] == [[TEXT, VOCAB]]
    assert first["sampling_contracts"][0][0]["allowed_token_ids"] == [5, 35]
    assert first["prepared_native_source"]["data"]["mode"] == "audio_text"
    assert generations[3]["prepared_native_source"]["data"]["mode"] == "text_text"
    # The trainer accepts the batch as it is.
    batch = collate_rollouts([audio, text], generations, 3)
    assert batch.batch["responses"].shape == (6, 2, NQ)


@pytest.mark.parametrize(
    "answer, message",
    [
        (lambda outputs: outputs[::-1], "incomplete OpusLM prompt groups"),
        (lambda outputs: outputs[:-1], "incomplete OpusLM prompt groups"),
        (
            lambda outputs: [
                {
                    **outputs[0],
                    "meta_info": {
                        **outputs[0]["meta_info"],
                        "finish_reason": {"type": "abort", "message": "too long"},
                    },
                },
                *outputs[1:],
            ],
            "aborted 0:0:0: too long",
        ),
        (
            lambda outputs: [{**outputs[0], "output_ids": [TEXT, 5]}, *outputs[1:]],
            "logprob IDs differ",
        ),
    ],
)
def test_answers_that_do_not_match_their_requests_are_refused(
    tmp_path, answer, message
):
    source = {"id": "a", "prompt_token_ids": frames([5], [35])}
    with pytest.raises(RuntimeError, match=message):
        engine(tmp_path, Recorder(answer)).generate([source], 0)


def test_settings_size_from_the_recipe_and_freeze_the_native_contract(tmp_path):
    settings = engine(
        tmp_path,
        replica=2,
        rollout_mem_fraction=0.45,
        sglang_engine_kwargs={"context_length": 2048, "max_running_requests": 44},
    ).settings(tmp_path / "hf")
    assert settings["model_path"] == str(tmp_path / "hf")
    assert settings["dtype"] == "float32" and settings["random_seed"] == 42 + 2
    assert settings["mem_fraction_static"] == 0.45
    assert (settings["context_length"], settings["max_running_requests"]) == (
        2048,
        44,
    )
    assert settings["disable_radix_cache"] and settings["skip_tokenizer_init"]
    assert settings["attention_backend"] == "triton"
    for frozen in (
        {"disable_radix_cache": False},
        {"disable_cuda_graph": False},
        {"dtype": "bfloat16"},
    ):
        with pytest.raises(ValueError, match="native execution contract"):
            engine(tmp_path, sglang_engine_kwargs=frozen).settings(tmp_path)


def export(directory, embedding, tokenizer="tokenizer", **metadata):
    from safetensors.torch import save_file

    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({**METADATA, **metadata}))
    for name in TOKENIZER_FILES:
        (directory / name).write_text(f"{tokenizer} {name}")
    save_file({"model.embed_tokens.weight": embedding}, str(directory / "w.st"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "w.st"}})
    )
    return directory


def test_updates_reload_the_embedding_and_only_a_changed_tokenizer(
    tmp_path, monkeypatch
):
    loads = []
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda path: loads.append(path) or Tokenizer(),
    )
    first, second = torch.zeros(VOCAB, 4), torch.ones(VOCAB, 4)
    recorder = Recorder()
    rollout = engine(tmp_path, recorder)
    rollout.tokenizer = rollout.tokenizer_files = rollout.embedding = None
    rollout._load_prompt_assets(export(tmp_path / "policy-0", first))
    assert len(loads) == 1 and torch.equal(rollout.embedding, first)
    rollout.update(export(tmp_path / "policy-1", second))
    assert recorder.updates == [str(tmp_path / "policy-1")]
    assert len(loads) == 1 and torch.equal(rollout.embedding, second)
    rollout.update(export(tmp_path / "policy-2", second.clone(), tokenizer="new"))
    assert len(loads) == 2 and torch.equal(rollout.embedding, second)
    with pytest.raises(ValueError, match="vocabulary"):
        rollout.update(export(tmp_path / "policy-3", second, eos_token_id=6))
    assert len(recorder.updates) == 2


def test_requests_and_running_batches_must_fit_the_kv_pool(tmp_path, monkeypatch):
    class Engine(Recorder):
        def __init__(self, **settings):
            super().__init__()
            self.running = settings.get("max_running_requests", 2048)

        def get_server_info(self):
            return {
                "max_total_num_tokens": 44 * 1280,
                "page_size": 1,
                "internal_states": [
                    {"effective_max_running_requests_per_dp": self.running}
                ],
            }

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda path: Tokenizer()
    )
    hf = export(tmp_path / "hf", torch.zeros(VOCAB, 4))
    sized = {"context_length": 1280, "max_running_requests": 44}
    rollout = OpusLMEngine(config(sglang_engine_kwargs=sized), hf, directory=tmp_path)
    # SGLang would silently shorten the budget of a longer embedded prompt.
    assert len(
        rollout.generate([{"id": "a", "prompt_token_ids": frames(*[[5]] * 1271)}], 0)
    )
    with pytest.raises(ValueError, match="exceed context_length"):
        rollout.generate([{"id": "a", "prompt_token_ids": frames(*[[5]] * 1272)}], 1)
    for kwargs in ({**sized, "max_running_requests": 45}, {"context_length": 1280}):
        with pytest.raises(ValueError, match="never retracts"):
            OpusLMEngine(config(sglang_engine_kwargs=kwargs), hf, directory=tmp_path)
    with pytest.raises(ValueError, match="bound context_length"):
        OpusLMEngine(
            config(sglang_engine_kwargs={"max_running_requests": 4}),
            hf,
            directory=tmp_path,
        )


def test_training_reward_normalizes_words_as_evaluation_does():
    source = {"id": "u", "reference": "HELLO WORLD AGAIN", "mode": "audio_text"}
    assert asr_wer_reward(source, {"text": "Hello, world... again!"}) == 0
    hypotheses = ["Hello, word again.", "hello"]
    rewards = [asr_wer_reward(source, {"text": text}) for text in hypotheses]
    assert rewards == [
        -word_error_rate(source["reference"], text) for text in hypotheses
    ]
    generations = [
        {
            "source_id": "u",
            "text": text,
            "token_ids": [1],
            "log_probs": [-1.0],
            "prompt_token_ids": [0],
        }
        for text in hypotheses
    ]
    records = summarize_wer([source], generations, 2)["records"]
    assert [-row["wer"] for row in records] == pytest.approx(rewards)
    with pytest.raises(ValueError, match="text output"):
        asr_wer_reward({**source, "mode": "audio_audio"}, {"text": ""})
