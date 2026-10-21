"""Check the QA rewards' extraction and scoring, MMAU matching and the TTS
reward's failure policy and audit."""

import json
from types import SimpleNamespace

import pytest
import torch

from espnet2.speechlm.model.speechlm.rl.bagpiper.data import OPTION_LABEL
from espnet2.speechlm.model.speechlm.rl.bagpiper.rewards import (
    ASRWERReward,
    aligned_codes,
    mmau_answer,
    mmau_match,
    qa_answer,
    qa_reward,
    qa_think_reward,
    qa_v1_reward,
    qa_west_default_reward,
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
    reward = qa_think_reward({"reference": "Car"}, {"text": prediction})
    assert reward["reward"] == pytest.approx(expected)
    assert reward["reward"] == pytest.approx(
        0.8 * reward["acc"] + 0.1 * reward["format"] + 0.1 * reward["think_format"]
    )


@pytest.mark.parametrize(
    "prediction,accuracy,answer_format",
    [
        ("<answer>Car</answer>", 1, 1),
        ("Car", 1, 0),
        # WEST's training parser: no DOTALL, an exact string, the whole response.
        ("<answer>\nCar\n</answer>", 0, 1),
        ("<answer>car</answer>", 0, 1),
        ("The car. <answer>Car</answer>", 1, 0),
    ],
)
def test_strict_west_reward(prediction, accuracy, answer_format):
    assert qa_reward({"reference": "Car"}, {"text": prediction}) == {
        "reward": 0.5 * (accuracy + answer_format),
        "acc": accuracy,
        "format": answer_format,
    }


@pytest.mark.parametrize(
    "prediction,accuracy,answer_format",
    [
        ("<answer>Car</answer>", 1, 1),
        ("Car", 1, 0),
        ("<answer>car</answer>", 0, 1),
        ("<answer>\nCar\n</answer>", 0, 1),
        ("caption <answer>Car</answer>", 1, 0),
        ("<think>reason</think><answer>Car</answer>", 1, 0),
        # Upstream's fullmatch regex spans adjacent answer blocks; preserve it.
        ("<answer>Car</answer><answer>Other</answer>", 1, 1),
        ("<answer>Other</answer>", 0, 1),
        ("<answer>Car</answer>\n", 1, 0),
        ("<answer>Car", 0, 0),
        ("", 0, 0),
    ],
)
def test_west_default_original_scale(prediction, accuracy, answer_format):
    # Expected components checked against WEST rewards.py and train_grpo.py
    # at b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb, including regex edge cases.
    assert qa_west_default_reward({"reference": "Car"}, {"text": prediction}) == {
        "reward": accuracy + answer_format,
        "acc": accuracy,
        "format": answer_format,
    }


@pytest.mark.parametrize(
    "text,answer,path",
    [
        (
            "caption\n<answer>\n motorcycle \n</answer> then <answer>car</answer>",
            "motorcycle",
            "tag",
        ),
        ("<answer></answer> the answer is car", "", "tag"),
        # Saved Bagpiper batch-0 endings.
        (
            "A loud engine passes. Thus the correct answer is: motorcycle",
            "motorcycle",
            "cue",
        ),
        ("Reasoning...\n\n**Final Answer:** motorcycle", "motorcycle", "cue"),
        ("so the most accurate choice is: by the car.", "by the car", "cue"),
        ('The answer is "on the tree".', "on the tree", "cue"),
        (
            "the most correct answer from the list is:\n\nOn the road",
            "On the road",
            "cue",
        ),
        ("The correct answer is car because: it honks", "car because: it honks", "cue"),
        (
            "I thought the answer is car.\nFinal answer:\n\n__Dog__\nbecause it barks",
            "Dog",
            "cue",
        ),
        (
            "To answer this, listen: a dog barks.",
            "To answer this, listen: a dog barks.",
            "text",
        ),
        ("  motorcycle\n", "motorcycle", "text"),
    ],
)
def test_v1_answer_extraction(text, answer, path):
    assert qa_answer(text) == (answer, path)


CHOICES = {
    "reference": "on the tree",
    "choices": ["Chicken coop", "on the tree", "cage"],
}


@pytest.mark.parametrize(
    "text,acc,answer_format,path",
    [
        ("A bird sings.\n<answer>on the tree</answer>", 1, 1, "tag"),
        ("<answer>\nOn the tree.\n</answer>", 1, 1, "tag"),
        ("<answer>on the tree</answer><answer>cage</answer>", 1, 0, "tag"),
        ("<answer>cage</answer>", 0, 1, "tag"),
        ("Birds chirp above. **Final Answer:** On the tree", 1, 0, "cue"),
        ("It is not in a cage. The answer is: on the tree", 1, 0, "cue"),
        # The whole response mentions a wrong option, as captions usually do.
        ("Not a cage: the birds are on the tree.", 0, 0, "text"),
        ("on the tree", 1, 0, "text"),
    ],
)
def test_v1_reward(text, acc, answer_format, path):
    reward = qa_v1_reward(CHOICES, {"text": text})
    assert reward == {
        "reward": 0.5 * acc + 0.5 * answer_format,
        "acc": acc,
        "format": answer_format,
        "path_tag": float(path == "tag"),
        "path_cue": float(path == "cue"),
        "path_text": float(path == "text"),
    }


def test_v1_reward_strips_option_labels_from_reference_and_choices():
    source = {"reference": "(B) A woman", "choices": ["(A) A man", "(B) A woman"]}
    assert qa_v1_reward(source, {"text": "<answer>A woman</answer>"})["acc"] == 1
    assert qa_v1_reward(source, {"text": "<answer>A man</answer>"})["acc"] == 0


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
    assert mmau_match(VOICE, mmau_answer(prediction)) is expected


def test_mmau_choices_that_contain_an_article_still_match():
    source = {"reference": "A woman", "choices": ["(A) A man", "(B) A woman"]}
    assert mmau_match(source, mmau_answer("<answer>(B) A woman</answer>"))
    assert not mmau_match(source, mmau_answer("<answer>(A) A man</answer>"))


@pytest.mark.parametrize(
    "value,expected",
    [("(A) Man", "Man"), ("D. Train", "Train"), ("Dove", "Dove"), ("(AB) x", "(AB) x")],
)
def test_option_labels_are_removed_once(value, expected):
    assert OPTION_LABEL.sub("", value) == expected


BASE, SIZE = 152192, 1025


def codec_frames(codes):
    """Delay-render [T, 8] codes as the native model emits them."""
    frames = []
    for row in range(len(codes) + 7):
        frame = []
        for stream in range(8):
            index = row - stream
            frame.append(
                BASE + stream * SIZE + 1 + codes[index][stream]
                if 0 <= index < len(codes)
                else BASE + stream * SIZE  # the stream's pad id
            )
        frames.append(frame)
    return frames


def test_aligned_codes_undo_the_delay_and_report_failures():
    codes = [[(3 * t + s) % 1024 for s in range(8)] for t in range(3)]
    text = [[300, 0, 0, 0, 0, 0, 0, 0]]
    aligned, status = aligned_codes(text + codec_frames(codes) + [[2] + [0] * 7])
    assert status == "ok"
    assert aligned.tolist() == torch.tensor(codes).T.tolist()
    assert aligned_codes(text + codec_frames(codes)[:7]) == (None, "no_audio")
    broken = codec_frames(codes)
    broken[4][2] = BASE + 2 * SIZE  # a pad id inside stream 2's real frames
    assert aligned_codes(broken) == (None, "invalid_codec")


class FakeCodec:
    """Decodes T aligned frames to T * 320 samples, as Xcodec does."""

    def decode(self, codes):
        return SimpleNamespace(audio_values=torch.ones(1, 1, codes.shape[-1] * 320) / 4)


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeASR:
    """Hears "the cat sat" in 960 samples, three times over in 320 and "the
    cat" otherwise; the processor's features are each waveform's length."""

    generation_config = SimpleNamespace(is_multilingual=False)

    def __call__(self, waveforms, sampling_rate, return_tensors):
        assert sampling_rate == 16000 and return_tensors == "pt"
        return FakeInputs(input_features=torch.tensor([len(w) for w in waveforms]))

    def generate(self, input_features, max_new_tokens):
        self.batches.append(len(input_features))
        return input_features

    def batch_decode(self, ids, skip_special_tokens):
        heard = {960: "the cat sat", 320: "the cat sat " * 3}
        return [heard.get(n, "the cat") for n in ids.tolist()]


def fake_reward(tmp_path, audits, batch_size=2):
    reward = ASRWERReward.__new__(ASRWERReward)
    asr = FakeASR()
    asr.batches = []
    reward.__dict__.update(
        torch=torch,
        audits=audits,
        calls=0,
        device=torch.device("cpu"),
        batch_size=batch_size,
        directory=tmp_path,
        codec=FakeCodec(),
        rate=16000,
        processor=asr,
        asr=asr,
        asr_tag="fake-asr",
        asr_revision="0" * 40,
        generate_kwargs={"max_new_tokens": 256},
    )
    return reward


def test_tts_failures_score_wer_one_and_are_counted_and_audited(tmp_path):
    codes = [[1] * 8, [2] * 8, [3] * 8]
    invalid = codec_frames(codes)
    invalid[3][1] = BASE + 5  # another layer's id in stream 1
    tokens = [
        codec_frames(codes),  # 3 frames: "the cat sat"
        codec_frames(codes[:2]),  # 2 frames: "the cat"
        codec_frames(codes)[:6],  # no decodable audio
        invalid,
        [[300] + [0] * 7, [2] + [0] * 7],  # text only
        codec_frames(codes[:1]),  # 1 frame: six inserted words
    ]
    sources = [{"id": f"utt-{i}", "reference": "the cat sat"} for i in range(6)]
    generations = [{"token_ids": value} for value in tokens]
    reward = fake_reward(tmp_path, audits=2)
    values = reward(sources, generations)
    wer = [0, pytest.approx(1 / 3), 1, 1, 1, 2]
    assert [value["wer"] for value in values] == wer
    # A failure always scores lowest: no audio is never better than audio.
    assert [value["reward"] for value in values] == [0, -1 / 3, -1, -1, -1, -1]
    assert [value["no_audio"] for value in values] == [0, 0, 1, 0, 1, 0]
    assert [value["invalid_codec"] for value in values] == [0, 0, 0, 1, 0, 0]
    # Only decodable candidates reach the ASR, two at a time.
    assert reward.asr.batches == [2, 1]
    rows = [
        json.loads(line)
        for line in (tmp_path / "batch-000000.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in rows] == [
        "ok",
        "ok",
        "no_audio",
        "invalid_codec",
        "no_audio",
        "ok",
    ]
    assert [row["codec_frames"] for row in rows] == [3, 2, 0, 0, 0, 1]
    # Evenly spaced audits: a WAV where there is audio, the full JSON always.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "batch-000000.jsonl",
        "utt-0-000000-0000.json",
        "utt-0-000000-0000.wav",
        "utt-3-000000-0003.json",
    ]
    audit = json.loads((tmp_path / "utt-3-000000-0003.json").read_text())
    assert audit["status"] == "invalid_codec" and audit["reward"] == -1
    assert audit["generated_tokens"] == invalid
    # The next call is a new audit batch.
    reward(sources[:1], generations[:1])
    assert (tmp_path / "batch-000001.jsonl").is_file()
