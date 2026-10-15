"""Native feature arrays are hashed by their bytes and shared, never copied."""

import copy
import hashlib
import json

import numpy as np
import pytest

from espnet2.rl.data import collate_rollouts, prepared_sha256
from espnet2.rl.scoring import _identity_digest


def generations(feats):
    data = {
        "native_model_inputs": {
            "continuous_audio_feats": feats,
            "continuous_audio_lengths": [3],
        },
        "mode": "text_only",
    }
    prepared = {"source_id": "u", "data": data, "sha256": prepared_sha256("u", data)}
    return [
        dict(
            source_id="u",
            prompt_token_ids=[1],
            token_ids=[2, 3],
            log_probs=[-0.1, -0.2],
            prepared_native_source=copy.deepcopy(prepared, {id(feats): feats}),
        )
        for _ in range(2)
    ]


def test_prepared_arrays_are_verified_by_bytes_and_shared_by_candidates():
    feats = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    batch = collate_rollouts([{"id": "u"}], generations(feats), 2)
    inputs = [s["native_model_inputs"] for s in batch.non_tensor_batch["source"]]
    assert inputs[0] is not inputs[1]
    assert all(item["continuous_audio_feats"] is feats for item in inputs)
    # Array-free inputs keep the canonical JSON digest of earlier records.
    data = {"native_model_inputs": {"continuous_audio_feats": feats.tolist()}}
    encoded = json.dumps(
        {"source_id": "u", "data": data},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert prepared_sha256("u", data) == hashlib.sha256(encoded).hexdigest()
    changed = feats.copy()
    changed[0, 1, 1] += 1
    for other in (changed, feats.astype(np.float64), feats.reshape(1, 2, 3)):
        records = generations(feats)
        inputs = records[1]["prepared_native_source"]["data"]["native_model_inputs"]
        inputs["continuous_audio_feats"] = other
        with pytest.raises(ValueError, match="hash differs"):
            collate_rollouts([{"id": "u"}], records, 2)


def test_identity_digest_reads_lists_and_arrays_by_value():
    feats = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    batch = collate_rollouts([{"id": "u"}], generations(feats), 2)
    digest = _identity_digest(batch)
    assert _identity_digest(copy.deepcopy(batch)) == digest
    changed = copy.deepcopy(batch)
    changed.non_tensor_batch["generation"][1]["log_probs"][1] = -0.25
    assert _identity_digest(changed) != digest
    changed = copy.deepcopy(batch)
    changed.non_tensor_batch["source"][0]["native_model_inputs"][
        "continuous_audio_feats"
    ][0, 0, 0] = 9
    assert _identity_digest(changed) != digest
