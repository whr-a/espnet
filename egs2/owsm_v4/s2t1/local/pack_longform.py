#!/usr/bin/env python3
"""Pack consecutive same-session utterances into the longest audio OWSM ingests.

``espnet2.rl.task_data.prepare_longform_manifest`` groups by a fixed
``--concatenate`` count, which cannot be made safe for a fixed-window
frontend: OWSM v4's preprocessor declares ``speech_length: 30`` and
``espnet2/s2t/rl/policy.py:load_audio`` RAISES on a longer waveform instead of
truncating it, so ``--concatenate 8`` (about 102 s of train audio) is
unfeedable at every single sample. This packer writes the same record schema
and audit fields but groups by a DURATION budget, which is what makes every
group both legal and as long as the window allows.

Provenance kept per group: every source utterance id, every source audio
sha256, the concatenated audio sha256, and the sha256 of the wav.scp/text that
the group was derived from.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def session_of(identity):
    """Same rule as espnet2.rl.task_data.session_of."""
    return identity.rsplit("-", 1)[0]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(records, budget_seconds, min_items, sample_rate):
    """Greedy consecutive packing inside each session under a duration budget."""
    grouped = {}
    for record in records:
        grouped.setdefault(session_of(record["id"]), []).append(record)
    groups = []
    for session, items in sorted(grouped.items()):
        # Keep the corpus order the session was prepared in.
        items = sorted(items, key=lambda r: r["id"])
        current, total = [], 0.0
        for item in items:
            if current and total + item["duration"] > budget_seconds:
                if len(current) >= min_items:
                    groups.append((session, current))
                current, total = [], 0.0
            if item["duration"] > budget_seconds:
                # A single utterance already over the window can never be used.
                current, total = [], 0.0
                continue
            current.append(item)
            total += item["duration"]
        if len(current) >= min_items:
            groups.append((session, current))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-seconds", type=float, default=29.0)
    parser.add_argument("--min-items", type=int, default=2)
    parser.add_argument("--min-reference-words", type=int, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit-groups", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source = [
        json.loads(line)
        for line in Path(args.source_manifest).read_text().splitlines()
        if line.strip()
    ]

    # Kaldi-shaped provenance inputs, written before anything is packed so the
    # audit can hash exactly what the grouping read.
    scp_path = output.parent / (output.stem + ".wav.scp")
    text_path = output.parent / (output.stem + ".text")
    ordered = sorted(source, key=lambda r: r["id"])
    scp_path.write_text("".join(f"{r['id']} {r['audio']}\n" for r in ordered))
    text_path.write_text("".join(f"{r['id']} {r['reference']}\n" for r in ordered))

    for record in ordered:
        info = sf.info(record["audio"])
        if info.samplerate != args.sample_rate:
            raise ValueError(f"{record['id']} is {info.samplerate} Hz")
        record["duration"] = info.frames / info.samplerate

    groups = pack(ordered, args.budget_seconds, args.min_items, args.sample_rate)
    if args.limit_groups:
        groups = groups[: args.limit_groups]
    if not groups:
        raise ValueError("No session produced a group under the duration budget")

    directory = output.parent / (output.stem + "_audio")
    directory.mkdir(parents=True, exist_ok=True)
    records, rejected = [], []
    for session, chunk in groups:
        reference = " ".join(item["reference"] for item in chunk)
        word_count = len(reference.split())
        identity = f"{session}-pack{chunk[0]['id'].rsplit('-', 1)[1]}x{len(chunk)}"
        if word_count < args.min_reference_words:
            rejected.append((identity, word_count))
            continue
        waveform = np.concatenate(
            [sf.read(item["audio"], dtype="float32")[0] for item in chunk]
        )
        target = directory / (
            hashlib.sha256(identity.encode()).hexdigest()[:20] + ".wav"
        )
        sf.write(str(target), waveform, args.sample_rate, subtype="FLOAT")
        records.append(
            {
                "id": identity,
                "audio": str(target),
                "reference": reference,
                "sample_rate": args.sample_rate,
                "audio_sha256": digest(target),
                "concatenated_ids": [item["id"] for item in chunk],
                "concatenated_sha256": [
                    item.get("audio_sha256") or digest(item["audio"]) for item in chunk
                ],
                "duration_seconds": len(waveform) / args.sample_rate,
                "reward_task": "longform_wer",
                "min_reference_words": args.min_reference_words,
            }
        )

    with output.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    durations = np.array([r["duration_seconds"] for r in records])
    counts = np.array([len(r["concatenated_ids"]) for r in records])
    word_counts = np.array([len(r["reference"].split()) for r in records])
    audit = {
        "task": "longform_wer",
        "packer": "local/pack_longform.py duration-budget greedy",
        "reason_not_fixed_concatenate": (
            "OWSM v4 preprocessor_conf.speech_length = 30 s and policy.load_audio "
            "raises above that window, so --concatenate 8 (~102 s) is unfeedable"
        ),
        "min_reference_words": args.min_reference_words,
        "budget_seconds": args.budget_seconds,
        "min_items": args.min_items,
        "sample_rate": args.sample_rate,
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_manifest_sha256": digest(args.source_manifest),
        "wav_scp": str(scp_path),
        "text": str(text_path),
        "wav_scp_sha256": digest(scp_path),
        "text_sha256": digest(text_path),
        "source_utterances": len(ordered),
        "groups_formed": len(groups),
        "groups_written": len(records),
        "groups_rejected_below_floor": len(rejected),
        "rejected_examples": rejected[:10],
        "utterances_consumed": int(counts.sum()),
        "sessions": len({r["id"].rsplit("-pack", 1)[0] for r in records}),
        "duration_seconds": {
            "mean": float(durations.mean()),
            "p50": float(np.percentile(durations, 50)),
            "p90": float(np.percentile(durations, 90)),
            "max": float(durations.max()),
            "min": float(durations.min()),
            "total_hours": float(durations.sum() / 3600),
        },
        "items_per_group": {
            "mean": float(counts.mean()),
            "min": int(counts.min()),
            "max": int(counts.max()),
        },
        "reference_words": {
            "mean": float(word_counts.mean()),
            "p50": float(np.percentile(word_counts, 50)),
            "p10": float(np.percentile(word_counts, 10)),
            "p90": float(np.percentile(word_counts, 90)),
            "max": int(word_counts.max()),
            "min": int(word_counts.min()),
        },
        "reference_words_per_id": {
            r["id"]: len(r["reference"].split()) for r in records
        },
    }
    Path(str(output) + ".audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    summary = {k: v for k, v in audit.items() if k not in ("reference_words_per_id", "rejected_examples")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
