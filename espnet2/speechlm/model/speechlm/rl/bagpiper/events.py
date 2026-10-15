"""Lossless request records with immutable native inputs shared across actions."""

import hashlib
import json
from pathlib import Path


def freeze_native_inputs(directory, native_inputs):
    """Write each exact native input once and return its auditable file identity."""
    directory = Path(directory)
    encoded = json.dumps(
        native_inputs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    relative = Path("event_native_inputs", f"{digest}.json")
    path = directory / relative
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Frozen native event input differs from its identity")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    return {"path": str(relative), "sha256": digest, "bytes": len(encoded)}


def request_record(payload, native_reference):
    """Keep every dynamic request field and bind the omitted immutable field."""
    if "native_model_inputs_ref" in payload or "native_model_inputs" not in payload:
        raise ValueError("Expected a complete native inference request")
    return {
        **{
            key: value for key, value in payload.items() if key != "native_model_inputs"
        },
        "native_model_inputs_ref": dict(native_reference),
    }


def restore_request(directory, record):
    """Recover the complete actual request; reject missing or changed inputs."""
    if "native_model_inputs_ref" not in record:
        if "native_model_inputs" not in record:
            raise ValueError("Inference event lacks native input identity")
        return dict(record)
    if "native_model_inputs" in record:
        raise ValueError("Inference event has conflicting native inputs")
    directory = Path(directory).resolve()
    reference = record["native_model_inputs_ref"]
    path = (directory / reference["path"]).resolve()
    if path.parent != directory / "event_native_inputs":
        raise ValueError("Native event reference leaves its frozen input directory")
    encoded = path.read_bytes()
    if (
        len(encoded) != reference["bytes"]
        or hashlib.sha256(encoded).hexdigest() != reference["sha256"]
    ):
        raise ValueError("Native event input hash or byte count differs")
    return {
        **{
            key: value
            for key, value in record.items()
            if key != "native_model_inputs_ref"
        },
        "native_model_inputs": json.loads(encoded),
    }
