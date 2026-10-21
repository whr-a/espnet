"""Native greedy heldout WER audit for autoregressive OWSM RL checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from espnet2.rl.rewards import edit_distance, words
from espnet2.s2t.rl.policy import OWSMPolicy, load_audio


@torch.no_grad()
def greedy_decode(policy, sources, config, max_tokens=256, batch_size=32):
    """Greedy native decoding of every source, ``batch_size`` utterances at once.

    OWSM pads every waveform to its fixed input window, so a batch of
    utterances runs the encoder exactly as one utterance alone does. The
    decoder recomputes each full prefix, padded after its true length.
    """
    device = next(policy.parameters()).device
    eos = policy.model.eos
    hypotheses = []
    for start in range(0, len(sources), batch_size):
        chunk = sources[start : start + batch_size]
        speech = torch.stack(
            [
                torch.as_tensor(
                    load_audio(source, policy.sample_rate, policy.window_samples)
                )
                for source in chunk
            ]
        ).to(device)
        lengths = torch.full((len(chunk),), speech.size(1), device=device)
        memory, memory_lengths = policy._encode(speech, lengths)
        prefixes = [policy.build_prefix(source, config) for source in chunk]
        generated = [[] for _ in chunk]
        active = list(range(len(chunk)))
        for _ in range(max_tokens):
            if not active:
                break
            rows = [
                torch.tensor(prefixes[i] + generated[i], device=device) for i in active
            ]
            tokens = pad_sequence(rows, batch_first=True, padding_value=eos)
            token_lengths = torch.tensor([len(row) for row in rows], device=device)
            index = torch.tensor(active, device=device)
            logits, _ = policy.model.decoder(
                memory[index], memory_lengths[index], tokens, token_lengths
            )
            chosen = logits[torch.arange(len(active)), token_lengths - 1].argmax(-1)
            for i, token in zip(active, chosen.tolist()):
                generated[i].append(token)
            active = [i for i in active if generated[i][-1] != eos]
        hypotheses.extend(generated)
    return hypotheses


def evaluate(config, manifest, output, max_tokens=256, batch_size=32):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace heldout evidence: {output}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = OWSMPolicy.from_config(config, config.device).eval()
    sources = [
        json.loads(line)
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    ]
    native = yaml.safe_load(Path(config.train_config).read_text())
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=native["bpemodel"])
    results = []
    generations = greedy_decode(policy, sources, config, max_tokens, batch_size)
    for source, generated in zip(sources, generations, strict=True):
        pieces = [
            policy.token_list[token]
            for token in generated
            if not (
                policy.token_list[token].startswith("<")
                and policy.token_list[token].endswith(">")
            )
        ]
        text = processor.decode(pieces)
        reference, hypothesis = words(source["reference"]), words(text)
        errors = edit_distance(reference, hypothesis)
        results.append(
            dict(
                id=source["id"],
                reference=source["reference"],
                hypothesis=text,
                errors=errors,
                reference_words=len(reference),
                wer=errors / max(1, len(reference)),
                token_ids=generated,
                eos_reached=generated[-1] == policy.model.eos,
                audio_sha256=hashlib.sha256(
                    Path(source["audio"]).read_bytes()
                ).hexdigest(),
            )
        )
    total_errors = sum(row["errors"] for row in results)
    total_words = sum(row["reference_words"] for row in results)
    audit = dict(
        backend="native_espnet_greedy",
        max_tokens=max_tokens,
        checkpoint=str(config.checkpoint),
        checkpoint_sha256=hashlib.file_digest(
            Path(config.checkpoint).open("rb"), "sha256"
        ).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        sample_count=len(results),
        total_errors=total_errors,
        total_reference_words=total_words,
        corpus_wer=total_errors / max(1, total_words),
        rows=results,
    )
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = SimpleNamespace(**yaml.safe_load(Path(args.config).read_text()))
    config.checkpoint = args.checkpoint
    config.device = args.device
    audit = evaluate(
        config, args.manifest, args.output, args.max_tokens, args.batch_size
    )
    print(
        json.dumps(
            {
                key: audit[key]
                for key in [
                    "sample_count",
                    "total_errors",
                    "total_reference_words",
                    "corpus_wer",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
