"""Native greedy heldout WER audit for autoregressive OWSM RL checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from espnet2.rl.rewards import edit_distance, words
from espnet2.s2t.rl.policy import OWSMPolicy, load_audio


def evaluate(config, manifest, output, max_tokens=256):
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
    device = next(policy.parameters()).device
    results = []
    with torch.no_grad():
        for source in sources:
            audio = load_audio(source, policy.sample_rate, policy.window_samples)
            speech = torch.tensor(audio, device=device).unsqueeze(0)
            memory, lengths = policy._encode(
                speech, torch.tensor([speech.size(1)], device=device)
            )
            prefix = policy.build_prefix(source, config)
            generated = []
            ended = False
            for _ in range(max_tokens):
                context = torch.tensor([[*prefix, *generated]], device=device)
                logits, _ = policy.model.decoder(
                    memory,
                    lengths,
                    context,
                    torch.tensor([context.size(1)], device=device),
                )
                token = int(logits[0, -1].argmax())
                generated.append(token)
                if token == policy.model.eos:
                    ended = True
                    break
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
                    eos_reached=ended,
                    audio_sha256=hashlib.sha256(
                        Path(source["audio"]).read_bytes()
                    ).hexdigest(),
                )
            )
    errors = sum(row["errors"] for row in results)
    words = sum(row["reference_words"] for row in results)
    audit = dict(
        backend="native_espnet_greedy",
        max_tokens=max_tokens,
        checkpoint=str(config.checkpoint),
        checkpoint_sha256=hashlib.file_digest(
            Path(config.checkpoint).open("rb"), "sha256"
        ).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        sample_count=len(results),
        total_errors=errors,
        total_reference_words=words,
        corpus_wer=errors / max(1, words),
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
    args = parser.parse_args()
    config = SimpleNamespace(**yaml.safe_load(Path(args.config).read_text()))
    config.checkpoint = args.checkpoint
    config.device = args.device
    audit = evaluate(config, args.manifest, args.output, args.max_tokens)
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
