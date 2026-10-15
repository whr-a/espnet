"""Bagpiper manifests: word limits, seeded shuffles and MMAU categories."""

import io
import json
import random
import sys

import numpy as np
import pandas as pd
import soundfile as sf

from espnet2.speechlm.model.speechlm.rl.bagpiper import data


def test_selection_filters_words_then_shuffles_then_limits():
    rows = [{"id": i, "reference": " ".join(["word"] * i)} for i in range(1, 9)]
    assert data.select(list(rows)) == rows
    # Words are counted as the WER counts them: hyphens and punctuation split.
    hyphen = {"id": 0, "reference": "you-a little girl."}
    assert data.select([hyphen], max_words=3) == []
    kept = data.select(list(rows), max_words=5)
    assert [row["id"] for row in kept] == [1, 2, 3, 4, 5]
    shuffled = data.select(list(rows), max_words=5, shuffle_seed=42, limit=3)
    expected = rows[:5]
    random.Random(42).shuffle(expected)
    assert shuffled == expected[:3]


def test_libritts_manifest_records_its_selection(tmp_path, monkeypatch):
    source = tmp_path / "LibriTTS" / "19" / "198"
    source.mkdir(parents=True)
    for index, words in enumerate((3, 30, 12, 25, 26)):
        stem = source / f"19_198_{index:06d}_000000"
        stem.with_suffix(".wav").write_bytes(b"")
        stem.with_suffix(".normalized.txt").write_text(" ".join(["a"] * words))
    output = tmp_path / "train" / "libritts.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data.py",
            "--dataset=libritts",
            f"--input={tmp_path / 'LibriTTS'}",
            "--task=tts_asr_wer",
            "--max-words=25",
            "--shuffle-seed=42",
            "--limit=2",
            f"--output={output}",
        ],
    )
    data.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    eligible = ["19_198_000000_000000", "19_198_000002_000000", "19_198_000003_000000"]
    random.Random(42).shuffle(eligible)
    assert [record["id"] for record in records] == eligible[:2]
    assert all(record["task"] == "tts_asr_wer" for record in records)
    audit = json.loads((output.parent / "libritts.jsonl.audit.json").read_text())
    assert (audit["prepared"], audit["count"]) == (5, 2)
    assert (audit["max_words"], audit["shuffle_seed"], audit["limit"]) == (25, 42, 2)


def test_mmau_categories_reach_the_manifest(tmp_path):
    cached = pd.DataFrame(
        {
            "reward_model": [{"ground_truth": "(A) Man"}],
            "extra_info": [
                {
                    "audio_path": "/audio/0.wav",
                    "question": "Who speaks?",
                    "choices": ["(A) Man", "(B) Woman"],
                    "id": "uuid-0",
                    "task": "speech",
                    "dataset": "voxceleb",
                }
            ],
        }
    )
    cached.to_parquet(tmp_path / "cached.parquet")
    (record,) = data.prepare_parquet(tmp_path / "cached.parquet", "mmau", tmp_path)
    assert record["categories"] == {"task": "speech"}
    assert (record["reference"], record["choices"]) == ("Man", ["Man", "Woman"])
    wav = io.BytesIO()
    sf.write(wav, np.zeros(160), 16000, format="WAV")
    official = pd.DataFrame(
        {
            "instruction": ["Who speaks?"],
            "choices": [["(A) Man", "(B) Woman"]],
            "answer": ["(A) Man"],
            "context": [{"bytes": wav.getvalue(), "path": None}],
            "other_attributes": [
                json.dumps(
                    {
                        "id": "uuid-0",
                        "task": "speech",
                        "category": "Reasoning",
                        "sub-category": "Speaker Identification",
                        "difficulty": "easy",
                        "split": "test-mini",
                    }
                )
            ],
        }
    )
    official.to_parquet(tmp_path / "official.parquet")
    (record,) = data.prepare_parquet(tmp_path / "official.parquet", "mmau", tmp_path)
    assert record["categories"] == {
        "task": "speech",
        "category": "Reasoning",
        "sub-category": "Speaker Identification",
        "difficulty": "easy",
    }
    # AVQA rows have no categories and keep their former fields.
    avqa = pd.DataFrame(
        {
            "reward_model": [{"ground_truth": "Car"}],
            "extra_info": [
                {"audio_path": "/a.wav", "question": "What?", "choices": ["Car"]}
            ],
        }
    )
    avqa.to_parquet(tmp_path / "avqa.parquet")
    (record,) = data.prepare_parquet(tmp_path / "avqa.parquet", "avqa", tmp_path)
    assert "categories" not in record
