# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Kaldi recipe inputs and auditable manifests shared by the task preparers."""

import hashlib
import json
import subprocess
from pathlib import Path

from espnet2.rl.utils import sha256


def read_kaldi(path):
    rows = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in rows:
            raise ValueError(f"Malformed/duplicate Kaldi row at {path}:{number}")
        rows[fields[0]] = fields[1]
    return rows


def input_files(**paths):
    """Audit entries for a preparer's inputs: every path, then every SHA256."""
    entries = {name: str(Path(path).resolve()) for name, path in paths.items()}
    entries.update({f"{name}_sha256": sha256(path) for name, path in paths.items()})
    return entries


def materialize_kaldi_audio(wav_scp, text, audio_dir, sample_rate=None, channel="mean"):
    """Resolve paths and pipes into auditable waveforms joined by utterance ID.

    Keep the source rate when ``sample_rate`` is None. SpeechLM frontends can
    then apply their own native resampler after selecting the first channel.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    waves, references = read_kaldi(wav_scp), read_kaldi(text)
    if waves.keys() != references.keys():
        raise ValueError("wav.scp and text must contain the same utterance IDs")
    if channel not in ("mean", "first"):
        raise ValueError("channel must be mean or first")
    audio_dir = Path(audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for identity, source in waves.items():
        target = audio_dir / (
            hashlib.sha256(identity.encode()).hexdigest()[:20] + ".wav"
        )
        if source.rstrip().endswith("|"):
            # Pipe commands are part of the user's recipe input. Preserve the
            # actual returned payload alongside the normalized waveform.
            import io

            process = subprocess.run(
                source.rstrip()[:-1], shell=True, capture_output=True, check=True
            )
            target.with_suffix(".source").write_bytes(process.stdout)
            waveform, rate = sf.read(io.BytesIO(process.stdout), dtype="float32")
            source_hash = hashlib.sha256(process.stdout).hexdigest()
        else:
            path = Path(source)
            if not path.is_absolute():
                path = Path(wav_scp).resolve().parent / path
            waveform, rate = sf.read(path, dtype="float32")
            source_hash = sha256(path)
        if waveform.ndim == 2:
            waveform = waveform.mean(-1) if channel == "mean" else waveform[:, 0]
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid audio for {identity}")
        target_rate = int(sample_rate) if sample_rate is not None else rate
        if rate != target_rate:
            from math import gcd

            divisor = gcd(rate, target_rate)
            waveform = resample_poly(waveform, target_rate // divisor, rate // divisor)
        sf.write(target, waveform, target_rate, subtype="FLOAT")
        records.append(
            {
                "id": identity,
                "audio": str(target),
                "reference": references[identity],
                "sample_rate": target_rate,
                "source_sample_rate": rate,
                "channel_rule": channel,
                "audio_sha256": sha256(target),
                "source_audio_sha256": source_hash,
                "source_audio": source,
            }
        )
    return records


def audio_directory(output):
    """Where a preparer writes the waveforms its manifest ``output`` points to."""
    return Path(output).parent / (Path(output).stem + "_audio")


def write_manifest(output, records, audit):
    """Write ``records`` as JSON lines and ``audit`` beside them; never overwrite."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifest: {output}")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(
            {**audit, "rows": len(records), "manifest_sha256": sha256(output)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return records
