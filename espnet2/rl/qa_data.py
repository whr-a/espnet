# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Prepare public AVQA train/local-dev manifests with scoped media admission.

This CPU-only entry reuses Bagpiper's raw-parquet conversion. It does not
download data, establish semantic/component closure, or audit SFT exposure.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def check_split(path, declared):
    """Reject contradictory available metadata; absent metadata stays attested."""
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = sorted(names & {"split", "extra_info", "other_attributes"})
    observed = set()
    for row in pq.read_table(path, columns=columns).to_pylist() if columns else []:
        for value in (row, row.get("extra_info"), row.get("other_attributes")):
            if isinstance(value, str):
                value = json.loads(value)
            if isinstance(value, dict) and value.get("split") is not None:
                observed.add(str(value["split"]))
    if observed - {declared}:
        raise ValueError(f"Source split metadata contradicts {declared}: {observed}")
    return sorted(observed)


def audio_record(row, base, cache):
    import numpy as np
    import soundfile as sf

    path = Path(row["audio"])
    path = (path if path.is_absolute() else base / path).resolve(strict=True)
    if path not in cache:
        waveform, rate = sf.read(path, dtype="float32", always_2d=True)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"Nonempty finite audio required: {path}")
        pcm = hashlib.sha256(
            json_bytes({"sample_rate": rate, "channels": waveform.shape[1]})
            + np.ascontiguousarray(waveform, dtype="<f4").tobytes()
        ).hexdigest()
        cache[path] = sha256(path), pcm, rate
    file_hash, pcm, rate = cache[path]
    if row.get("audio_sha256") not in (None, file_hash):
        raise ValueError(f"Recorded audio hash differs: {path}")
    return {
        **row,
        "audio": str(path),
        "audio_sha256": file_hash,
        "decoded_audio_sha256": pcm,
        "sample_rate": rate,
    }


def identity_keys(row):
    return {
        f"id:{row['id']}",
        f"file:{row['audio_sha256']}",
        f"pcm:{row['decoded_audio_sha256']}",
    }


def prepare(args):
    if not args.source_dataset.strip() or not args.source_revision.strip():
        raise ValueError("Nonempty source dataset and revision attestations required")
    if not re.match(r"^train(?:$|[-_.])", args.source_split):
        raise ValueError(
            "Use an actual train split; local dev retains this native split"
        )
    if args.train_count < 1 or args.heldout_count < 1:
        raise ValueError("Both train and heldout counts must be positive")
    source = Path(args.input).resolve(strict=True)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh preparation directory: {output}")
    observed = check_split(source, args.source_split)
    # Avoid importing model packages/training kernels for CPU data preparation.
    path = (
        Path(__file__).resolve().parents[1]
        / "speechlm/model/speechlm/rl/bagpiper/data.py"
    )
    spec = importlib.util.spec_from_file_location("public_bagpiper_data", path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    cache, excluded, exclusion_bindings = {}, set(), []
    for name in args.exclude_manifest:
        manifest = Path(name).resolve(strict=True)
        for line in manifest.read_text().splitlines():
            if line.strip():
                excluded.update(
                    identity_keys(
                        audio_record(json.loads(line), manifest.parent, cache)
                    )
                )
        exclusion_bindings.append({"path": str(manifest), "sha256": sha256(manifest)})
    output.mkdir(parents=True)
    records = converter.prepare_parquet(source, "avqa", output / "prepared")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Source parquet has duplicate question IDs")
    for index, row in enumerate(records):
        row.update(
            source_row_index=index,
            source_native_split=args.source_split,
            qa_template=args.qa_template,
        )
        row["prompt"] = converter.qa_prompt(
            row["question"], row["choices"], args.qa_template
        )
        if args.qa_template == "bagpiper":
            row["system_prompt"] = converter.qa_system_prompt(args.qa_template)
        row["provenance"] = {
            "dataset": args.source_dataset,
            "revision": args.source_revision,
            "split": args.source_split,
            "source_id": row["id"],
        }
    records.sort(
        key=lambda row: hashlib.sha256(f"{args.seed}\0{row['id']}".encode()).digest()
    )
    selected, used, rejected = [], set(), 0
    for row in records:
        if (
            not row.get("question")
            or len(row.get("choices", [])) < 2
            or row["reference"] not in row["choices"]
        ):
            raise ValueError(
                f"QA question, choices and exact reference required: {row['id']}"
            )
        row = audio_record(row, source.parent, cache)
        keys = identity_keys(row)
        if keys & (used | excluded):
            rejected += 1
            continue
        selected.append(row)
        used.update(keys)
        if len(selected) == args.train_count + args.heldout_count:
            break
    if len(selected) != args.train_count + args.heldout_count:
        raise ValueError(
            f"Only {len(selected)} eligible independent questions; "
            f"need {args.train_count + args.heldout_count}"
        )
    splits = {
        "heldout": selected[: args.heldout_count],
        "train": selected[args.heldout_count :],
    }
    hashes = {}
    for role, rows in splits.items():
        file = output / ("dev.jsonl" if role == "heldout" else "train.jsonl")
        file.write_bytes(b"".join(json_bytes(row) + b"\n" for row in rows))
        hashes[role] = sha256(file)
    admission = {
        "schema": "espnet-public-qa-data-v1",
        "training_ready": True,
        "manifest_sha256": hashes,
        "counts": {role: len(rows) for role, rows in splits.items()},
        "source_dataset": args.source_dataset,
        "source_revision": args.source_revision,
        "preparation_sha256": {
            "helper": sha256(__file__),
            "bagpiper_converter": sha256(path),
            **(
                {"native_qa_contract": sha256(path.with_name("qa_contract.py"))}
                if args.qa_template == "bagpiper"
                else {}
            ),
            "qa_template": hashlib.sha256(
                (converter.qa_template_description(args.qa_template)).encode()
            ).hexdigest(),
        },
        "source_parquet": {
            "path": str(source),
            "sha256": sha256(source),
            "rows": len(records),
        },
        "source_split": {role: args.source_split for role in splits},
        "source_split_attestation": (
            "User-declared native split; available metadata checked for contradictions"
        ),
        "observed_source_split_metadata": observed,
        "heldout_scope": (
            "deterministic disjoint local train partition; not official validation/test"
        ),
        "selection": {
            "seed": args.seed,
            "order": "SHA256(seed NUL source_id); heldout first, then train",
            "rejected_visited_rows": rejected,
            "qa_template": args.qa_template,
        },
        "excluded_manifests": exclusion_bindings,
        "verified_scope": (
            "unique/disjoint question IDs, prepared audio file bytes and decoded PCM; "
            "supplied exclusion manifests only"
        ),
        "full_component_admission": "NOT_ASSERTED",
        "limitations": [
            "Dataset revision/split provenance is attested, not fetched or "
            "authenticated by this helper.",
            "No resampling, perceptual/semantic, case/group/component or "
            "SFT-exposure closure is established.",
            "This public receipt does not replace additional project-specific "
            "protected-data admission.",
        ],
    }
    (output / "admission.json").write_bytes(json_bytes(admission) + b"\n")
    return admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input",
        "output-dir",
        "source-dataset",
        "source-revision",
        "source-split",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--train-count", type=int, default=3200)
    parser.add_argument("--heldout-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qa-template", choices=("default", "think", "bagpiper"), default="default"
    )
    parser.add_argument("--exclude-manifest", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
