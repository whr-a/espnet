"""Freeze independent short-run prompts against registered protection metadata.

The scoped gate covers all registered protection and source-component identities;
it does not assert unknown semantic links. It never follows a sealed payload.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Candidate manifests require nonempty, unique source IDs")
    return rows


def extractor(path, expected):
    if file_hash(path) != expected:
        raise ValueError("Canonical boundary extractor hash changed")
    spec = importlib.util.spec_from_file_location("frozen_boundary_keys", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge(target, source):
    for kind, values in source.items():
        if kind in target:
            target[kind].update(str(value) for value in values)


def normalized_audio_hash(path):
    """Content identity after deterministic mono/16k conversion, without GPU."""
    import math

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Nonempty finite audio required")
    waveform = waveform.mean(-1)
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        waveform = resample_poly(waveform, 16000 // divisor, int(rate) // divisor)
    return hashlib.sha256(
        np.ascontiguousarray(waveform, dtype="<f4").tobytes()
    ).hexdigest()


def build_protection(args):
    """Read explicitly authorized metadata only, never registry payload paths."""
    current, specialist = load_json(args.current), load_json(args.specialist_current)
    cross = load_json(args.cross_protection)
    module = extractor(args.boundary_extractor, cross["extractor_sha256"])
    keys = {kind: set() for kind in module.KINDS}
    bindings = {}

    def bind(path, expected=None):
        actual = file_hash(path)
        if expected and actual != expected:
            raise ValueError(f"Protection binding changed: {path}")
        bindings[str(Path(path).resolve())] = actual
        return load_json(path)

    bind(args.current)
    bind(args.specialist_current)
    bindings[str(Path(args.boundary_extractor).resolve())] = cross["extractor_sha256"]
    registry = bind(current["registry"], current["registry_sha256"])
    # Bind the full 64+12 metadata registry without opening any dataset/payload.
    general = registry.get("general_release", {}).get("metadata", {})
    if isinstance(general, dict) and general.get("path"):
        bind(general["path"], general.get("sha256"))
    elif isinstance(general, str):
        bind(general)
    manifest = bind(specialist["manifest"], specialist["manifest_sha256"])
    protection = bind(
        manifest["training_protection"]["path"],
        manifest["training_protection"]["sha256"],
    )
    if len(protection["records"]) != specialist["records"]:
        raise ValueError("Incomplete specialist protection")
    by_id = {row["id"]: row for row in protection["records"]}
    allowed_audio = {}
    corpus_keys = set()
    for row in protection["records"]:
        keys["sample_id"].add(row["id"])
        keys["component"].add(row["component"])
        keys["media"].update(row["audio_sha256"])
    for panel in manifest["panels"].values():
        samples = panel["sample_manifest.jsonl"]
        dialogues = panel["native_prompt_dialogues.jsonl"]
        for item in (samples, dialogues):
            actual = file_hash(item["path"])
            if actual != item["sha256"]:
                raise ValueError("Specialist panel binding changed")
            bindings[str(Path(item["path"]).resolve())] = actual
        sample_rows = load_rows(samples["path"])
        for row in sample_rows:
            if row["id"] not in by_id:
                raise ValueError("Specialist panel identity differs from protection")
            for audio in row["audios"]:
                allowed_audio[audio["sha256"]] = audio["path"]
                for identity in re.findall(
                    r"(?<![0-9])([0-9]+[-_][0-9]+[-_][0-9]+)(?![0-9])", audio["path"]
                ):
                    corpus_keys.add(identity.replace("_", "-"))
        native = [
            json.loads(line)
            for line in Path(dialogues["path"]).read_text().splitlines()
        ]
        if len(native) != len(sample_rows):
            raise ValueError("Incomplete specialist dialogue panel")
        for row in native:
            identity = row["example_id"]
            if (
                identity not in by_id
                or hashlib.sha256(canonical(row)).hexdigest()
                != by_id[identity]["prompt_sha256"]
            ):
                raise ValueError("Specialist prompt identity/hash changed")
            prompt = []
            for role, modality, value in row["messages"]:
                part = (
                    {"type": "audio", "audio_url": value}
                    if modality == "audio"
                    else {"type": "text", "text": value}
                )
                prompt.append({"role": role, "content": [part]})
            merge(keys, module.extract({"prompt": prompt}))
    merge(keys, bind(args.test_keys)["keys"])
    bind(args.cross_protection)
    for section in ("test", "val", "natural_dev"):
        merge(keys, cross["sets"][section])
    closure = bind(args.human_closure)
    for field, kind in (
        ("components", "component"),
        ("annotation_content_components", "component"),
        ("case_ids", "case"),
        ("sample_component_closure_rows", "sample_id"),
    ):
        keys[kind].update(closure.get(field, []))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    normalized = []
    if args.decode_registered_audio:
        for expected, path in sorted(allowed_audio.items()):
            if file_hash(path) != expected:
                raise ValueError(f"Registered allowed audio identity changed: {path}")
            normalized.append(normalized_audio_hash(path))
    result = {
        "schema": "espnet-short-run-protection-v1",
        "bindings": bindings,
        "extractor_sha256": cross["extractor_sha256"],
        "keys": {kind: sorted(values) for kind, values in keys.items()},
        "specialist_records": len(protection["records"]),
        "registered_audio_count": len(allowed_audio),
        "registered_audio_decoded": len(normalized),
        "normalized_audio_sha256": sorted(set(normalized)),
        "corpus_utterance_ids": sorted(corpus_keys),
        "scoped_component_admission": (
            "VERIFIED_REGISTERED_PROTECTION"
            if args.decode_registered_audio
            else "METADATA_EXACT_GATE_ONLY"
        ),
        "full_component_admission": "NOT_ASSERTED_UNREGISTERED_SEMANTIC_LINKS",
        "limitations": [
            "Exact registered keys do not prove complete audio/document/conversation closure.",
            "Full general64+native12 per-item component mapping is not established.",
            "Sealed Test450+guard keys are aggregate; no sealed payload was opened.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "protection": str(output),
                "sha256": file_hash(output),
                "key_counts": {k: len(v) for k, v in keys.items()},
                "full_component_admission": result["full_component_admission"],
            }
        )
    )


def audit_record(row, manifest, module):
    """Adapt recipe-native fields before invoking the frozen canonical extractor."""
    import numpy as np
    import soundfile as sf

    value = dict(row)
    audio = Path(value["audio"])
    if not audio.is_absolute():
        audio = Path(manifest).resolve().parent / audio
    if audio.suffix.lower() not in {".wav", ".flac", ".ogg", ".mp3"}:
        raise ValueError("Only explicit nonsealed audio files can be audited")
    actual = file_hash(audio)
    if value.get("audio_sha256") and value["audio_sha256"] != actual:
        raise ValueError(f"Audio changed: {value['id']}")
    waveform, rate = sf.read(audio, dtype="float32", always_2d=True)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("Nonempty finite audio required")
    decoded = hashlib.sha256(
        canonical({"rate": rate, "channels": waveform.shape[1]})
        + np.ascontiguousarray(waveform, dtype="<f4").tobytes()
    ).hexdigest()
    native = value.get("native_preparation", {})
    hashes = {actual}
    for h in (value.get("source_audio_sha256"), native.get("audio_sha256")):
        if h:
            hashes.add(h)
    extra = dict(value.get("extra_info", {}))
    extra["media_sha256"] = sorted(hashes)
    for source, dest in (
        ("component", "component_id"),
        ("component_id", "component_id"),
        ("case_id", "case_id"),
        ("group_id", "group_id"),
        ("source_record_id", "source_record_id"),
        ("original_input_hash", "original_input_hash"),
    ):
        if value.get(source):
            extra[dest] = value[source]
    # The source ID and raw supervision text are separate from generic templates.
    extra["sample_id"] = str(value["id"])
    if value.get("question"):
        extra["question"] = value["question"]
    if (
        value.get("task") == "tts_asr_wer"
        or value.get("reward_task") == "asr_wer"
        or value.get("mode") == "audio_text"
    ):
        extra["transcript"] = value["reference"]
    keys = module.extract({"extra_info": extra})
    value.update(
        audio=str(audio.resolve()),
        audio_sha256=actual,
        source_audio_sha256=value.get(
            "source_audio_sha256", native.get("audio_sha256", actual)
        ),
        decoded_audio_sha256=decoded,
        normalized_audio_sha256=normalized_audio_hash(audio),
        sample_rate=int(rate),
        boundary_keys={k: sorted(v) for k, v in keys.items()},
    )
    return value


def select_records(rows, protected, forbidden, count, audit, role):
    selected, used, rejected = [], {k: set() for k in protected}, []
    blocked = {k: set(protected[k]) | set(forbidden[k]) for k in protected}
    # Scan all intermediate rows before selection, so a linked hit blocks the component.
    changed = True
    while changed:
        changed = False
        for row in rows:
            keys = row["boundary_keys"]
            if any(set(values) & blocked[kind] for kind, values in keys.items()):
                for kind in (
                    "case",
                    "group",
                    "component",
                    "srid",
                    "oih",
                    "media",
                    "sample_id",
                ):
                    new = set(keys[kind]) - blocked[kind]
                    if new:
                        changed = True
                        blocked[kind].update(new)
    decoded = set(audit.get("train_decoded", []))
    for row in sorted(rows, key=lambda item: str(item["id"])):
        keys = row["boundary_keys"]
        hits = {
            k: sorted(set(v) & (protected[k] | forbidden[k] | blocked[k]))
            for k, v in keys.items()
        }
        duplicate = (
            any(
                set(keys[k]) & used[k]
                for k in ("media", "case", "group", "component", "srid", "oih")
            )
            or row["decoded_audio_sha256"] in decoded
        )
        if any(hits.values()) or duplicate:
            rejected.append(
                {
                    "id": row["id"],
                    "hits": {k: v for k, v in hits.items() if v},
                    "duplicate_audio_or_component": duplicate,
                }
            )
            continue
        if len(selected) < count:
            selected.append(row)
            merge(used, keys)
            decoded.add(row["decoded_audio_sha256"])
    audit[role + "_rejections"] = rejected
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} independent eligible {role} records; need {count}"
        )
    return selected, used


def freeze_selection(args):
    protection = load_json(args.protection)
    for path, expected in protection["bindings"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Protection input changed: {path}")
    module = extractor(args.boundary_extractor, protection["extractor_sha256"])
    if not re.match(r"^train(?:$|[-_.])", args.train_split.lower()):
        raise ValueError("New training prompts require a genuine train split")
    if args.train_split == args.heldout_split and not args.heldout_is_train_partition:
        raise ValueError("Train and heldout must use different source splits")
    if args.heldout_is_train_partition and args.train_split != args.heldout_split:
        raise ValueError(
            "A train-partition heldout must retain its true native train split"
        )
    if args.train_count != 32 or args.heldout_count < 1:
        raise ValueError("This short-run contract requires exactly 32 training prompts")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Refusing to replace a frozen data directory")
    train = [
        audit_record(r, args.train_candidates, module)
        for r in load_rows(args.train_candidates)
    ]
    heldout = [
        audit_record(r, args.heldout_candidates, module)
        for r in load_rows(args.heldout_candidates)
    ]
    protected = {k: set(v) for k, v in protection["keys"].items()}
    for row in train:
        if row["normalized_audio_sha256"] in protection.get(
            "normalized_audio_sha256", []
        ):
            protected["media"].add(row["audio_sha256"])
        if str(row["id"]).replace("_", "-") in protection.get(
            "corpus_utterance_ids", []
        ):
            protected["sample_id"].add(str(row["id"]))
    empty = {k: set() for k in protected}
    audit = {
        "schema": "espnet-short-run-data-v1",
        "exact_key_gate": "PASSED",
        "full_component_admission": protection["full_component_admission"],
        "scoped_component_admission": protection["scoped_component_admission"],
        "training_ready": protection["scoped_component_admission"]
        == "VERIFIED_REGISTERED_PROTECTION",
        "limitations": protection["limitations"],
        "protection_sha256": file_hash(args.protection),
        "source_dataset": args.source_dataset,
        "source_revision": args.source_revision,
        "source_split": {"train": args.train_split, "heldout": args.heldout_split},
        "heldout_scope": (
            "deterministic disjoint partition of native train; not official validation/test"
            if args.heldout_is_train_partition
            else "independent native heldout split"
        ),
        "candidate_sha256": {
            "train": file_hash(args.train_candidates),
            "heldout": file_hash(args.heldout_candidates),
        },
        "selection": "sorted source ID; exclude all registered keys and linked components; independent raw/decoded audio",
        "verified_protection_scope": {
            "specialist_records": protection["specialist_records"],
            "registered_audio_count": protection["registered_audio_count"],
            "registered_audio_decoded": protection["registered_audio_decoded"],
            "keys": {k: len(v) for k, v in protection["keys"].items()},
            "checks": [
                "entire Gemini945+human444 registered components and native-dialogue text",
                "all permitted registered raw and normalized audio",
                "Test+guard aggregate, enhanced test/val/natural_dev metadata",
                "transitive candidate case/group/component/source/media closure",
                "actual native source split and independent train/heldout raw+decoded audio",
            ],
        },
    }
    selected, train_keys = select_records(train, protected, empty, 32, audit, "train")
    audit["train_decoded"] = [r["decoded_audio_sha256"] for r in selected]
    for row in heldout:
        if row["normalized_audio_sha256"] in {
            r["normalized_audio_sha256"] for r in selected
        }:
            train_keys["media"].add(row["audio_sha256"])
    evaluation, _ = select_records(
        heldout, empty, train_keys, args.heldout_count, audit, "heldout"
    )
    output.mkdir(parents=True)
    hashes = {}
    for role, rows in (("train32", selected), ("heldout", evaluation)):
        for row in rows:
            row["provenance"] = {
                "dataset": args.source_dataset,
                "revision": args.source_revision,
                "split": args.train_split if role == "train32" else args.heldout_split,
                "source_id": row["id"],
            }
        path = output / (role + ".jsonl")
        path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        hashes[role] = file_hash(path)
    audit.update(
        manifest_sha256=hashes,
        counts={"train": 32, "heldout": len(evaluation)},
        protection_bindings=protection["bindings"],
    )
    (output / "admission.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "counts": audit["counts"],
                "exact_key_gate": "PASSED",
                "training_ready": audit["training_ready"],
                "full_component_admission": audit["full_component_admission"],
            }
        )
    )


def prepare_avqa(args):
    """Keep cached AVQA's true train split and freeze a disjoint local partition."""
    import pandas as pd

    # Load the CPU-only preparer directly; importing the model package would
    # unnecessarily load training kernels and CUDA dependencies.
    preparer = (
        Path(args.bagpiper_preparer)
        if args.bagpiper_preparer
        else (
            Path(__file__).resolve().parents[1]
            / "speechlm/model/speechlm/rl/bagpiper/data.py"
        )
    )
    spec = importlib.util.spec_from_file_location("bagpiper_data_preparer", preparer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qa_prompt = module.qa_prompt

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(args.input)
    train, heldout = [], []
    for index, value in frame.iterrows():
        source = value.to_dict()
        extra = source.get("extra_info", {})
        if extra.get("split") != "train":
            raise ValueError("Cached AVQA source must attest its actual train split")
        identity = str(extra.get("id", f"avqa-{int(extra['index']):06d}"))
        choices = list(extra.get("choices", []))
        question = extra["question"].replace("video", "audio")
        answer = re.sub(
            r"^(?:[A-Z]\.|\([A-Z]\))\s*", "", source["reward_model"]["ground_truth"]
        ).strip()
        row = {
            "id": identity,
            "audio": extra["audio_path"],
            "reference": answer,
            "question": question,
            "choices": choices,
            "prompt": qa_prompt(question, choices, args.qa_template),
            "qa_template": args.qa_template,
            "task": "qa",
            "dataset": "avqa",
            "extra_info": {"source_record_id": f"avqa/train/{extra['index']}"},
            "source_native_split": "train",
            "source_row_index": int(extra["index"]),
        }
        target = (
            heldout
            if int(hashlib.sha256(identity.encode()).hexdigest(), 16) % 5 == 0
            else train
        )
        if len(target) < args.pool_count:
            target.append(row)
        if len(train) == args.pool_count and len(heldout) == args.pool_count:
            break
    if min(len(train), len(heldout)) < args.pool_count:
        raise ValueError(
            "Not enough unique source rows for frozen candidate partitions"
        )
    output.mkdir(parents=True)
    for name, rows in (("train_candidates", train), ("heldout_candidates", heldout)):
        (output / (name + ".jsonl")).write_bytes(
            b"".join(canonical(row) + b"\n" for row in rows)
        )
    audit = {
        "input": str(Path(args.input).resolve()),
        "input_sha256": file_hash(args.input),
        "source_native_split": "train",
        "source_rows": len(frame),
        "partition": "sha256(source_id) mod5 ==0 heldout; others train; first pool_count per partition",
        "candidate_rows": {"train": len(train), "heldout": len(heldout)},
        "qa_template": args.qa_template,
        "heldout_scope": "disjoint local native-train partition, not official validation/test",
        "preparer_sha256": file_hash(__file__),
        "bagpiper_preparer_sha256": file_hash(preparer),
    }
    (output / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    protect = sub.add_parser("protect")
    for name in (
        "current",
        "specialist-current",
        "boundary-extractor",
        "test-keys",
        "cross-protection",
        "human-closure",
        "output",
    ):
        protect.add_argument("--" + name, required=True)
    protect.add_argument("--decode-registered-audio", action="store_true")
    select = sub.add_parser("select")
    for name in (
        "train-candidates",
        "heldout-candidates",
        "train-split",
        "heldout-split",
        "source-dataset",
        "source-revision",
        "protection",
        "boundary-extractor",
        "output-dir",
    ):
        select.add_argument("--" + name, required=True)
    select.add_argument("--train-count", type=int, default=32)
    select.add_argument("--heldout-count", type=int, default=8)
    select.add_argument("--heldout-is-train-partition", action="store_true")
    avqa = sub.add_parser("avqa")
    avqa.add_argument("--input", required=True)
    avqa.add_argument("--output-dir", required=True)
    avqa.add_argument("--pool-count", type=int, default=128)
    avqa.add_argument("--bagpiper-preparer")
    avqa.add_argument("--qa-template", choices=("default", "think"), default="default")
    args = parser.parse_args()
    {"protect": build_protection, "select": freeze_selection, "avqa": prepare_avqa}[
        args.operation
    ](args)


if __name__ == "__main__":
    main()
