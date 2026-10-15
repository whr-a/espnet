import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from espnet2.rl.short_run_data import audit_record, select_records


KINDS = ("case", "group", "component", "srid", "oih", "media", "sq_text", "sample_id")


def keys(**values):
    return {kind: set(values.get(kind, [])) for kind in KINDS}


def record(identity, **values):
    return {
        "id": identity,
        "decoded_audio_sha256": f"decoded-{identity}",
        "boundary_keys": {
            kind: sorted(v) for kind, v in keys(media=[identity], **values).items()
        },
    }


def test_protected_alias_blocks_full_transitive_component():
    rows = [
        record("a", component=["C1"], srid=["R1"]),
        record("b", component=["C2"], srid=["R1"]),
        record("protected", component=["C2"]),
        record("clean"),
    ]
    selected, _ = select_records(
        rows, keys(media=["protected"]), keys(), 1, {}, "train"
    )
    assert [row["id"] for row in selected] == ["clean"]


def test_same_decoded_audio_cannot_pad_independent_batch():
    rows = [record("a"), record("b")]
    rows[1]["decoded_audio_sha256"] = rows[0]["decoded_audio_sha256"]
    with pytest.raises(ValueError, match="Only 1 independent"):
        select_records(rows, keys(), keys(), 2, {}, "train")


def test_train_audio_cannot_enter_heldout_after_id_change():
    row = record("new-id")
    row["decoded_audio_sha256"] = "same-pcm"
    with pytest.raises(ValueError, match="Only 0 independent"):
        select_records(
            [row], keys(), keys(), 1, {"train_decoded": ["same-pcm"]}, "heldout"
        )


def test_recipe_audio_fields_are_adapted_before_extraction(tmp_path):
    audio = tmp_path / "source.wav"
    soundfile.write(audio, np.arange(20, dtype="float32") / 20, 16000, subtype="FLOAT")
    observed = {}

    def extract(row):
        observed.update(row)
        return keys(media=row["extra_info"]["media_sha256"])

    row = audit_record(
        {
            "id": "sample",
            "audio": str(audio),
            "reference": "text",
            "native_preparation": {"audio_sha256": "original-hash"},
            "mode": "audio_text",
        },
        tmp_path / "candidate.jsonl",
        SimpleNamespace(extract=extract),
    )
    assert row["boundary_keys"]["media"] == sorted(
        [hashlib.sha256(audio.read_bytes()).hexdigest(), "original-hash"]
    )
    assert observed["extra_info"]["transcript"] == "text"
    assert observed["extra_info"]["sample_id"] == "sample"


def test_changed_audio_hash_fails_before_extractor(tmp_path):
    audio = tmp_path / "source.wav"
    soundfile.write(audio, np.ones(20), 16000)
    with pytest.raises(ValueError, match="Audio changed"):
        audit_record(
            {"id": "a", "audio": str(audio), "audio_sha256": "stale"},
            tmp_path / "candidate.jsonl",
            SimpleNamespace(extract=lambda r: keys()),
        )


def test_no_non_audio_payload_path_is_opened(tmp_path, monkeypatch):
    def forbidden_open(*args, **kwargs):
        pytest.fail("A nonaudio payload was opened")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="nonsealed audio"):
        audit_record(
            {"id": "sealed", "audio": str(tmp_path / "test.parquet")},
            tmp_path / "candidate.jsonl",
            SimpleNamespace(extract=lambda r: keys()),
        )
