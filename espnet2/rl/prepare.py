"""Build auditable recipe manifests from Kaldi wav.scp and supervision."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _kaldi(path):
    rows = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in rows:
            raise ValueError(f"Malformed/duplicate Kaldi row at {path}:{number}")
        rows[fields[0]] = fields[1]
    return rows


def materialize_kaldi_audio(wav_scp, text, audio_dir, sample_rate=None, channel="mean"):
    """Resolve paths and pipes into auditable waveforms joined by utterance ID.

    Keep the source rate when ``sample_rate`` is None. SpeechLM frontends can
    then apply their own native resampler after selecting the first channel.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    waves, references = _kaldi(wav_scp), _kaldi(text)
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
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
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
                "audio_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_audio_sha256": source_hash,
                "source_audio": source,
            }
        )
    return records


def prepare_kaldi_manifest(wav_scp, text, output, task="asr_wer", **kwargs):
    """Materialize pipe audio, normalize rate, and join references by utterance ID."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifest: {output}")
    sample_rate = int(kwargs.get("sample_rate", 16000))
    records = materialize_kaldi_audio(
        wav_scp, text, output.parent / (output.stem + "_audio"), sample_rate
    )
    for record in records:
        record["reward_task"] = task
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(
            {
                "rows": len(records),
                "task": task,
                "sample_rate": sample_rate,
                "wav_scp": str(Path(wav_scp).resolve()),
                "text": str(Path(text).resolve()),
                "wav_scp_sha256": hashlib.sha256(
                    Path(wav_scp).read_bytes()
                ).hexdigest(),
                "text_sha256": hashlib.sha256(Path(text).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="asr_wer")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()
    prepare_kaldi_manifest(
        args.wav_scp, args.text, args.output, args.task, sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()
