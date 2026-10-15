import json

import pytest

from espnet2.speechlm.model.speechlm.rl.bagpiper.events import (
    freeze_native_inputs,
    request_record,
    restore_request,
)


def test_full_native_requests_remain_reconstructible_without_repeated_features(
    tmp_path,
):
    native = {
        "continuous_audio_feats": [[[float(i) / 100 for i in range(5000)]]],
        "continuous_audio_lengths": [5000],
        "continuous_audio_indices": [[0, 12, 625]],
    }
    reference = freeze_native_inputs(tmp_path, native)
    events = []
    original_size = 0
    for sample in range(8):
        for position in range(12):
            request = {
                "source_id": "audio-source",
                "native_model_inputs": native,
                "sequence": [[1, 0, 0]] + [[256 + i, 0, 0] for i in range(position)],
                "mode": "text_only",
                "phase": "text",
                "sample": sample,
                "position": position,
                "step": 2,
            }
            event = json.loads(json.dumps(request_record(request, reference)))
            assert restore_request(tmp_path, event) == request
            events.append(event)
            original_size += len(json.dumps(request))
    assert len(list((tmp_path / "event_native_inputs").iterdir())) == 1
    assert len(json.dumps(events)) + reference["bytes"] < original_size / 10
    assert freeze_native_inputs(tmp_path, native) == reference
    # Existing saved events remain directly readable without rewriting them.
    assert restore_request(tmp_path, request) == request


def test_event_reconstruction_rejects_changed_missing_and_external_native_inputs(
    tmp_path,
):
    reference = freeze_native_inputs(tmp_path, {"features": [[0.0, 1.0]]})
    event = request_record({"native_model_inputs": {}, "sample": 1}, reference)
    file = tmp_path / reference["path"]
    file.write_text('{"features":[[0.0,9.0]]}')
    with pytest.raises(ValueError, match="hash or byte"):
        restore_request(tmp_path, event)
    file.unlink()
    with pytest.raises(FileNotFoundError):
        restore_request(tmp_path, event)
    event["native_model_inputs_ref"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="leaves"):
        restore_request(tmp_path, event)
    with pytest.raises(ValueError, match="lacks"):
        restore_request(tmp_path, {"sample": 1})
