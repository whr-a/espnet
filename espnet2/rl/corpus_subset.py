"""Stream a genuine LibriSpeech/LibriTTS split into an auditable CPU subset."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


class RecordedStream:
    def __init__(self, stream):
        self.stream, self.bytes_read = stream, 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.bytes_read += len(value)
        self.digest.update(value)
        return value


def download_subset(dataset, split, output_dir, minimum_pairs):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Refusing to replace an existing corpus subset")
    if minimum_pairs < 1:
        raise ValueError("minimum_pairs must be positive")
    resource = {"LibriSpeech": "12", "LibriTTS": "60"}[dataset]
    url = f"https://www.openslr.org/resources/{resource}/{split}.tar.gz"
    prefix = PurePosixPath(dataset) / split
    output.mkdir(parents=True)
    members, audio, references = [], {}, {}
    with urllib.request.urlopen(url, timeout=60) as response:
        stream = RecordedStream(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if not member.isfile() or not name.is_relative_to(prefix):
                    continue
                relative = name.relative_to(prefix)
                if ".." in relative.parts:
                    raise ValueError("Unsafe archive member")
                is_audio = name.suffix in (".wav", ".flac")
                is_text = (
                    name.name.endswith(".trans.txt")
                    if dataset == "LibriSpeech"
                    else name.name.endswith(".normalized.txt")
                )
                if not is_audio and not is_text:
                    continue
                payload = archive.extractfile(member).read()
                target = output / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                members.append(
                    {
                        "archive_member": member.name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                if is_audio:
                    audio[name.stem] = str(target.resolve())
                elif dataset == "LibriSpeech":
                    for line in payload.decode().splitlines():
                        identity, text = line.split(maxsplit=1)
                        references[identity] = text
                else:
                    references[name.name.removesuffix(".normalized.txt")] = (
                        payload.decode().strip()
                    )
                complete = audio.keys() & references.keys()
                if len(complete) >= minimum_pairs:
                    break
        receipt = {
            "dataset": dataset,
            "source_native_split": split,
            "source_url": url,
            "http_status": response.status,
            "http_etag": response.headers.get("ETag"),
            "http_last_modified": response.headers.get("Last-Modified"),
            "archive_content_length": response.headers.get("Content-Length"),
            "compressed_bytes_read": stream.bytes_read,
            "compressed_read_prefix_sha256": stream.digest.hexdigest(),
            "full_archive_downloaded": False,
            "full_archive_hash_verified": False,
            "integrity_scope": "HTTPS source and exact retained tar member payload hashes; partial stream, not full archive checksum",
            "members": members,
        }
    complete = sorted(audio.keys() & references.keys())
    if len(complete) < minimum_pairs:
        receipt["status"] = "FAILED_INSUFFICIENT_PAIRS"
        (output / "source_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        raise ValueError(f"Only {len(complete)} complete pairs")
    (output / "wav.scp").write_text(
        "".join(f"{key} {audio[key]}\n" for key in complete)
    )
    (output / "text").write_text(
        "".join(f"{key} {references[key]}\n" for key in complete)
    )
    receipt.update(
        status="PREPARED_CPU_SUBSET",
        pairs=len(complete),
        wav_scp_sha256=hashlib.sha256((output / "wav.scp").read_bytes()).hexdigest(),
        text_sha256=hashlib.sha256((output / "text").read_bytes()).hexdigest(),
    )
    (output / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in receipt.items() if key != "members"}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("LibriSpeech", "LibriTTS"), required=True)
    parser.add_argument(
        "--split", choices=("train-clean-100", "dev-clean"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-pairs", type=int, default=128)
    args = parser.parse_args()
    download_subset(args.dataset, args.split, args.output_dir, args.minimum_pairs)


if __name__ == "__main__":
    main()
