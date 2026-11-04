#!/usr/bin/env python3
"""Split one speaker inventory's Kaldi inputs into train/heldout BY UTTERANCE.

A closed-set speaker task needs the same speakers on both sides, so the split
runs inside each speaker: every speaker appears in train and in heldout, with
disjoint utterances. The draw is seeded per speaker, so it is reproducible.

This is the split design ``espnet2.rl.task_data``'s default
``choice_key="utterance"`` exists for. Pair the two: a per-speaker candidate
draw over a split like this one makes the candidate list a function of the
label, and ``espnet2.rl.rewards.speaker._validate_speaker_id`` refuses such a
manifest because a model can then score it without listening.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def kaldi(path):
    rows = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in rows:
            raise ValueError(f"Malformed/duplicate row at {path}:{number}")
        rows[fields[0]] = fields[1]
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--min-heldout-per-speaker", type=int, default=6)
    parser.add_argument("--min-train-per-speaker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    waves, speakers = kaldi(args.wav_scp), kaldi(args.utt2spk)
    if waves.keys() != speakers.keys():
        raise ValueError("wav.scp and utt2spk must cover the same utterances")

    by_speaker = {}
    for utt, speaker in speakers.items():
        by_speaker.setdefault(speaker, []).append(utt)

    split = {"train": [], "heldout": []}
    per_speaker = {}
    for speaker in sorted(by_speaker):
        utterances = sorted(by_speaker[speaker])
        generator = random.Random(f"{args.seed}:{speaker}")
        generator.shuffle(utterances)
        count = max(args.min_heldout_per_speaker, round(len(utterances) * args.heldout_fraction))
        if len(utterances) - count < args.min_train_per_speaker:
            raise ValueError(
                f"Speaker {speaker} has only {len(utterances)} utterances; a "
                f"{count}-utterance heldout would leave too few for training"
            )
        held, trained = utterances[:count], utterances[count:]
        split["heldout"].extend(held)
        split["train"].extend(trained)
        per_speaker[speaker] = {"total": len(utterances), "train": len(trained), "heldout": len(held)}

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = {}
    for role, utterances in split.items():
        utterances = sorted(utterances)
        for name, table in (("wav.scp", waves), ("utt2spk", speakers)):
            path = output / f"{role}.{name}"
            path.write_text("".join(f"{utt} {table[utt]}\n" for utt in utterances))
            written[f"{role}.{name}"] = {"path": str(path), "sha256": digest(path), "rows": len(utterances)}

    overlap = set(split["train"]) & set(split["heldout"])
    train_speakers = {speakers[u] for u in split["train"]}
    heldout_speakers = {speakers[u] for u in split["heldout"]}
    audit = {
        "source_wav_scp": str(Path(args.wav_scp).resolve()),
        "source_wav_scp_sha256": digest(args.wav_scp),
        "source_utt2spk": str(Path(args.utt2spk).resolve()),
        "source_utt2spk_sha256": digest(args.utt2spk),
        "seed": args.seed,
        "heldout_fraction": args.heldout_fraction,
        "min_heldout_per_speaker": args.min_heldout_per_speaker,
        "utterance_overlap": len(overlap),
        "speakers_total": len(by_speaker),
        "speakers_in_train": len(train_speakers),
        "speakers_in_heldout": len(heldout_speakers),
        "speakers_shared": len(train_speakers & heldout_speakers),
        "counts": {role: len(utts) for role, utts in split.items()},
        "files": written,
        "per_speaker": per_speaker,
    }
    if overlap:
        raise ValueError(f"{len(overlap)} utterances landed in both splits")
    if train_speakers != heldout_speakers:
        raise ValueError("A closed-set split must carry the same speakers on both sides")
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in (
        "utterance_overlap", "speakers_total", "speakers_shared", "counts")}, indent=2))


if __name__ == "__main__":
    main()
