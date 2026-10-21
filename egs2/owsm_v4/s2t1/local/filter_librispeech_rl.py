#!/usr/bin/env python3
"""Keep LibriSpeech RL utterances that cannot leak the protected test sets.

The general benchmark registers test-clean and test-other as protected, and
the boundary-key admission only binds those sets by file hash. This filter
removes what that check cannot see before an RL manifest is built:

* utterances whose transcript equals a protected test transcript,
* chapters whose book or LibriVox project also appears in test-clean or
  test-other (the speaker and chapter never overlap, the source text can),
* audio longer than the model's input window.

It reads a Kaldi ``wav.scp``/``text`` pair whose keys are LibriSpeech
utterance IDs, writes the kept pair and a receipt with every exclusion.
"""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

TEST_SUBSETS = ("test-clean", "test-other")


def read_kaldi(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            key, value = line.split(maxsplit=1)
            rows[key] = value
    return rows


def canonical(text):
    return " ".join(text.upper().split())


def protected_sources(chapters_path):
    """Book and project IDs of every test chapter, and each chapter's own."""
    chapter_source, test_books, test_projects = {}, set(), set()
    for line in Path(chapters_path).read_text().splitlines():
        if line.startswith(";") or not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        chapter, subset, project, book = fields[0], fields[3], fields[4], fields[5]
        chapter_source[chapter] = (book, project)
        if subset in TEST_SUBSETS:
            test_books.add(book)
            test_projects.add(project)
    return chapter_source, test_books, test_projects


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--chapters", required=True, help="LibriSpeech CHAPTERS.TXT")
    parser.add_argument(
        "--test-transcripts",
        required=True,
        help="JSONL of the protected test references, one 'answer' per line",
    )
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument(
        "--keep-shared-books",
        action="store_true",
        help="Skip the book/project rule, e.g. for an evaluation-only split",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    audio, text = read_kaldi(args.wav_scp), read_kaldi(args.text)
    if audio.keys() != text.keys():
        raise ValueError("wav.scp and text cover different utterances")
    protected_text = {
        canonical(json.loads(line)["answer"])
        for line in Path(args.test_transcripts).read_text().splitlines()
        if line.strip()
    }
    chapter_source, test_books, test_projects = protected_sources(args.chapters)

    kept, excluded = [], {"test_transcript": [], "test_book_or_project": [], "long": []}
    for key in sorted(audio):
        speaker, chapter, _ = key.split("-")
        if canonical(text[key]) in protected_text:
            excluded["test_transcript"].append(key)
            continue
        if not args.keep_shared_books:
            if chapter not in chapter_source:
                raise ValueError(f"Chapter {chapter} of {key} is not in CHAPTERS.TXT")
            book, project = chapter_source[chapter]
            if book in test_books or project in test_projects:
                excluded["test_book_or_project"].append(key)
                continue
        info = sf.info(audio[key])
        if info.frames / info.samplerate > args.max_seconds:
            excluded["long"].append(key)
            continue
        kept.append(key)

    output.mkdir(parents=True)
    (output / "wav.scp").write_text("".join(f"{k} {audio[k]}\n" for k in kept))
    (output / "text").write_text("".join(f"{k} {text[k]}\n" for k in kept))
    receipt = {
        "inputs": {
            "wav_scp": {"path": args.wav_scp, "sha256": sha256(args.wav_scp)},
            "text": {"path": args.text, "sha256": sha256(args.text)},
            "chapters": {"path": args.chapters, "sha256": sha256(args.chapters)},
            "test_transcripts": {
                "path": args.test_transcripts,
                "sha256": sha256(args.test_transcripts),
                "unique": len(protected_text),
            },
        },
        "rules": {
            "test_transcript": "uppercase, whitespace-collapsed transcript equals a protected test reference",
            "test_book_or_project": (
                "skipped"
                if args.keep_shared_books
                else f"chapter shares a BOOK ID or PROJ. with {', '.join(TEST_SUBSETS)}"
            ),
            "long": f"audio longer than {args.max_seconds} s",
        },
        "input_utterances": len(audio),
        "kept": len(kept),
        "kept_speakers": len({key.split("-")[0] for key in kept}),
        "kept_chapters": len({key.split("-")[1] for key in kept}),
        "excluded_counts": {rule: len(keys) for rule, keys in excluded.items()},
        "excluded": excluded,
        "outputs": {
            "wav_scp_sha256": sha256(output / "wav.scp"),
            "text_sha256": sha256(output / "text"),
        },
    }
    (output / "filter_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "excluded"}, indent=2))


if __name__ == "__main__":
    main()
