"""Public raw AVQA preparation and its existing runtime admission contract."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from espnet2.rl.data import read_manifest, verify_data_admission
from espnet2.rl.qa_data import prepare, sha256


def source_parquet(path, *, split=None):
    rows = []
    for index, value in enumerate((0.125, 0.125, 0.25, 0.375, 0.5, 0.625)):
        stream = io.BytesIO()
        sf.write(stream, np.full(32, value), 16000, format="WAV", subtype="PCM_16")
        row = {
            "id": f"question-{index}",
            "audio": {"bytes": stream.getvalue(), "path": None},
            "instruction": "What is heard?\nChoices:\nA. Dog\nB. Door",
            "answer": "A. Dog",
        }
        if split is not None:
            row["other_attributes"] = json.dumps({"split": split})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path)


def arguments(tmp_path, output="prepared", **overrides):
    return SimpleNamespace(
        **{
            "input": str(tmp_path / "train.parquet"),
            "output_dir": str(tmp_path / output),
            "source_dataset": "public/avqa",
            "source_revision": "user-attested-revision",
            "source_split": "train",
            "train_count": 3,
            "heldout_count": 1,
            "qa_template": "default",
            "seed": 42,
            "exclude_manifest": [],
            **overrides,
        }
    )


@pytest.mark.parametrize("qa_template", ["default", "bagpiper"])
def test_raw_parquet_split_excludes_equivalent_pcm_and_passes_runtime_admission(
    tmp_path, qa_template
):
    source_parquet(tmp_path / "train.parquet")
    benchmark_audio = tmp_path / "benchmark.wav"
    # Different file encoding, identical decoded audio to source question-2.
    sf.write(benchmark_audio, np.full(32, 0.25), 16000, subtype="FLOAT")
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps({"id": "other-question", "audio": "benchmark.wav"}) + "\n"
    )
    args = arguments(
        tmp_path, exclude_manifest=[str(benchmark)], qa_template=qa_template
    )
    receipt = prepare(args)
    train = read_manifest(tmp_path / "prepared/train.jsonl")
    dev = read_manifest(tmp_path / "prepared/dev.jsonl")
    assert len(train) == 3 and len(dev) == 1
    assert "question-2" not in {row["id"] for row in train + dev}
    assert len({row["decoded_audio_sha256"] for row in train + dev}) == 4
    assert all(
        row["qa_template"] == qa_template and row["reference"] == "Dog"
        for row in train + dev
    )
    if qa_template == "bagpiper":
        assert all(
            row["prompt"] == "question: What is heard?\n Options: Dog; Door"
            for row in train + dev
        )
        assert all("detailed caption" in row["system_prompt"] for row in train + dev)
        assert "native_qa_contract" in receipt["preparation_sha256"]
    assert all(row["provenance"]["split"] == "train" for row in train + dev)
    assert receipt["source_split"] == {"train": "train", "heldout": "train"}
    assert "not official" in receipt["heldout_scope"]
    assert receipt["full_component_admission"] == "NOT_ASSERTED"
    assert receipt["source_parquet"]["sha256"] == sha256(tmp_path / "train.parquet")
    config = SimpleNamespace(
        manifest=str(tmp_path / "prepared/train.jsonl"),
        eval_manifest=str(tmp_path / "prepared/dev.jsonl"),
    )
    assert (
        verify_data_admission(config, tmp_path / "prepared/admission.json") == receipt
    )
    second = prepare(
        arguments(
            tmp_path,
            output="repeated",
            exclude_manifest=[str(benchmark)],
            qa_template=qa_template,
        )
    )
    assert second["counts"] == receipt["counts"]
    for filename in ("train.jsonl", "dev.jsonl"):
        assert [
            row["id"] for row in read_manifest(tmp_path / "repeated" / filename)
        ] == [row["id"] for row in read_manifest(tmp_path / "prepared" / filename)]
    # Admission is live integrity checking, not a fabricated clearance flag.
    with open(train[0]["audio"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="audio changed"):
        verify_data_admission(config, tmp_path / "prepared/admission.json")


def test_train_split_attestation_cannot_contradict_available_source_metadata(tmp_path):
    source_parquet(tmp_path / "train.parquet", split="test")
    with pytest.raises(ValueError, match="Nonempty source"):
        prepare(arguments(tmp_path, source_revision=""))
    with pytest.raises(ValueError, match="contradicts"):
        prepare(arguments(tmp_path))
    assert not (tmp_path / "prepared").exists()
    with pytest.raises(ValueError, match="actual train split"):
        prepare(arguments(tmp_path, source_split="test"))


def test_insufficient_unique_audio_and_existing_outputs_are_rejected(tmp_path):
    source_parquet(tmp_path / "train.parquet", split="train")
    with pytest.raises(ValueError, match="Only 5 eligible"):
        prepare(arguments(tmp_path, train_count=5))
    assert not (tmp_path / "prepared/admission.json").exists()
    with pytest.raises(FileExistsError, match="fresh preparation"):
        prepare(arguments(tmp_path))
