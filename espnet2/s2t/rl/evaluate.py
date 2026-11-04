"""Native heldout WER audit for autoregressive OWSM RL checkpoints.

Two decode modes share one audit format:

``native_espnet_greedy``
    The original argmax path, unchanged, that produced every existing
    ``heldout_step*.json``.
``native_espnet_sampled``
    Temperature sampling over the full support, which is the decode the GRPO
    reward was actually computed on (``temperature 1.0``, ``top_p 1.0``,
    ``top_k -1``). Greedy returns the mode of the policy, so a policy that
    tightens around a mode it already held moves the sampled WER without
    moving the greedy WER; only the sampled number can see that.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from espnet2.rl.rewards.text import edit_distance, words
from espnet2.rl.types import tempered
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


def row_seed(seed, index, draw):
    """A stable seed for one draw of one prompt, independent of batching.

    Keying on the prompt's position and the draw, rather than on one stream
    advanced per step, is what lets a sampled run be reproduced at a different
    ``batch_size``: a row samples the same tokens whichever rows ride with it.
    """
    digest = hashlib.sha256(f"{seed}:{index}:{draw}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def restrict(probabilities, top_p, top_k):
    """Zero the tail that SGLang's top-k / top-p filters would remove.

    The OWSM RL runs sample the full support (``top_p 1.0``, ``top_k -1``), so
    both filters are off and this returns its input untouched.
    """
    vocabulary = probabilities.size(-1)
    if top_k is not None and 0 < top_k < vocabulary:
        kept = probabilities.topk(top_k, dim=-1).values[:, -1:]
        probabilities = probabilities.masked_fill(probabilities < kept, 0.0)
    if top_p is not None and 0 < top_p < 1:
        ordered, index = probabilities.sort(dim=-1, descending=True)
        # Keep the shortest prefix reaching top_p: a token stays when the mass
        # strictly before it still falls short of the budget.
        keep = (ordered.cumsum(-1) - ordered) < top_p
        restored = torch.zeros_like(keep).scatter(-1, index, keep)
        probabilities = probabilities * restored.to(probabilities.dtype)
    return probabilities


def sample_tokens(logits, temperature, top_p, top_k, uniforms):
    """Inverse-CDF sampling of one token per row from a pre-drawn uniform."""
    probabilities = torch.softmax(tempered(logits.float(), temperature), dim=-1)
    probabilities = restrict(probabilities, top_p, top_k)
    cumulative = probabilities.cumsum(-1)
    # Scaling by the final mass absorbs both the top-p/top-k truncation and
    # the last-bit error a 50k-wide cumulative sum carries.
    targets = (uniforms * cumulative[:, -1]).unsqueeze(-1)
    return (
        torch.searchsorted(cumulative.contiguous(), targets, right=True)
        .squeeze(-1)
        .clamp_max(probabilities.size(-1) - 1)
    )


@torch.no_grad()
def sampled_decode(
    policy,
    sources,
    config,
    max_tokens=256,
    batch_size=32,
    temperature=1.0,
    samples=1,
    seed=0,
    top_p=1.0,
    top_k=-1,
):
    """Draw ``samples`` continuations per source at ``temperature``.

    Returns one generation per (source, draw) pair in source-major order, so
    row ``index * samples + draw`` belongs to ``sources[index]``. One encoder
    pass per source feeds all of its draws.

    ``temperature`` 0 falls back to argmax and reproduces ``greedy_decode``
    token for token.
    """
    if samples < 1:
        raise ValueError("Sampling needs at least one draw per prompt")
    device = next(policy.parameters()).device
    eos = policy.model.eos
    per_chunk = max(1, batch_size // samples)
    hypotheses = []
    for start in range(0, len(sources), per_chunk):
        chunk = sources[start : start + per_chunk]
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
        memory = memory.repeat_interleave(samples, dim=0)
        memory_lengths = memory_lengths.repeat_interleave(samples, dim=0)
        prefixes = [
            policy.build_prefix(source, config)
            for source in chunk
            for _ in range(samples)
        ]
        streams = [
            torch.Generator().manual_seed(row_seed(seed, start + offset, draw))
            for offset in range(len(chunk))
            for draw in range(samples)
        ]
        generated = [[] for _ in prefixes]
        active = list(range(len(prefixes)))
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
            logits = logits[torch.arange(len(active)), token_lengths - 1]
            if temperature == 0:
                chosen = logits.argmax(-1)
            else:
                uniforms = torch.tensor(
                    [torch.rand((), generator=streams[i]).item() for i in active],
                    device=device,
                )
                chosen = sample_tokens(logits, temperature, top_p, top_k, uniforms)
            for i, token in zip(active, chosen.tolist()):
                generated[i].append(token)
            active = [i for i in active if generated[i][-1] != eos]
        hypotheses.extend(generated)
    return hypotheses


def read_manifest(manifest):
    return [
        json.loads(line)
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    ]


def detokenizer(config):
    native = yaml.safe_load(Path(config.train_config).read_text())
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=native["bpemodel"])


def hypothesis_text(policy, processor, generated):
    """Detokenize one generation, dropping OWSM's angle-bracket control tokens."""
    pieces = [
        policy.token_list[token]
        for token in generated
        if not (
            policy.token_list[token].startswith("<")
            and policy.token_list[token].endswith(">")
        )
    ]
    return processor.decode(pieces)


def score_samples(sources, texts, generations, eos_reached, samples):
    """Per-draw rows, per-prompt aggregates, and the corpus totals.

    ``corpus_wer`` pools every draw: its denominator counts each prompt's
    reference ``samples`` times, which makes it the mean over draws of the
    corpus WER -- the quantity the GRPO reward averages over a group.
    ``corpus_wer_best`` and ``corpus_wer_worst`` instead keep each prompt's
    extreme draw over the single-pass denominator, so all three are directly
    comparable with a greedy corpus WER.
    """
    rows, prompts = [], []
    for index, source in enumerate(sources):
        reference = words(source["reference"])
        errors = []
        for draw in range(samples):
            row = index * samples + draw
            count = edit_distance(reference, words(texts[row]))
            errors.append(count)
            rows.append(
                dict(
                    id=source["id"],
                    draw=draw,
                    reference=source["reference"],
                    hypothesis=texts[row],
                    errors=count,
                    reference_words=len(reference),
                    wer=count / max(1, len(reference)),
                    token_ids=generations[row],
                    eos_reached=eos_reached[row],
                )
            )
        prompts.append(
            dict(
                id=source["id"],
                reference_words=len(reference),
                draws=samples,
                errors=errors,
                mean_errors=sum(errors) / samples,
                best_errors=min(errors),
                worst_errors=max(errors),
                eos_failures=sum(
                    1
                    for draw in range(samples)
                    if not eos_reached[index * samples + draw]
                ),
            )
        )
    reference_words = sum(prompt["reference_words"] for prompt in prompts)
    denominator = max(1, reference_words)
    totals = dict(
        prompt_count=len(prompts),
        samples_per_prompt=samples,
        sample_count=len(rows),
        reference_words_per_draw=reference_words,
        total_reference_words=reference_words * samples,
        total_errors=sum(row["errors"] for row in rows),
        eos_failures=sum(1 for row in rows if not row["eos_reached"]),
        corpus_wer=sum(prompt["mean_errors"] for prompt in prompts) / denominator,
        corpus_wer_best=sum(prompt["best_errors"] for prompt in prompts) / denominator,
        corpus_wer_worst=sum(prompt["worst_errors"] for prompt in prompts)
        / denominator,
    )
    return rows, prompts, totals


def select_subset(sources, subset_stride, subset_limit):
    """A fixed, reproducible slice: every ``subset_stride``-th prompt, capped.

    Striding rather than truncating keeps the 40 heldout speakers represented,
    because the manifest is ordered by speaker.
    """
    if subset_stride < 1:
        raise ValueError("subset_stride must be at least 1")
    selected = sources[::subset_stride]
    if subset_limit:
        selected = selected[:subset_limit]
    if not selected:
        raise ValueError("The requested subset selected no prompts")
    return selected


def evaluate(config, manifest, output, max_tokens=256, batch_size=32):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace heldout evidence: {output}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = OWSMPolicy.from_config(config, config.device).eval()
    sources = read_manifest(manifest)
    processor = detokenizer(config)
    results = []
    generations = greedy_decode(policy, sources, config, max_tokens, batch_size)
    for source, generated in zip(sources, generations, strict=True):
        text = hypothesis_text(policy, processor, generated)
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


def evaluate_sampled(
    config,
    manifest,
    output,
    max_tokens=256,
    batch_size=32,
    temperature=1.0,
    samples=1,
    seed=0,
    top_p=1.0,
    top_k=-1,
    subset_stride=1,
    subset_limit=0,
):
    """Audit the sampled WER of one checkpoint on one heldout manifest."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace heldout evidence: {output}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = OWSMPolicy.from_config(config, config.device).eval()
    selected = select_subset(
        read_manifest(manifest), subset_stride, subset_limit
    )
    processor = detokenizer(config)
    generations = sampled_decode(
        policy,
        selected,
        config,
        max_tokens,
        batch_size,
        temperature,
        samples,
        seed,
        top_p,
        top_k,
    )
    texts = [hypothesis_text(policy, processor, row) for row in generations]
    eos_reached = [row[-1] == policy.model.eos for row in generations]
    rows, prompts, totals = score_samples(
        selected, texts, generations, eos_reached, samples
    )
    audit = dict(
        backend="native_espnet_sampled",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        max_tokens=max_tokens,
        batch_size=batch_size,
        checkpoint=str(config.checkpoint),
        checkpoint_sha256=hashlib.file_digest(
            Path(config.checkpoint).open("rb"), "sha256"
        ).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        subset_stride=subset_stride,
        subset_limit=subset_limit,
        # Proves two checkpoints were scored on the very same prompts.
        subset_sha256=hashlib.sha256(
            "\n".join(source["id"] for source in selected).encode()
        ).hexdigest(),
        **totals,
        prompts=prompts,
        rows=rows,
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sample at this temperature instead of decoding greedily. "
        "The OWSM RL reward was computed at 1.0 over the full support.",
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Draws per prompt; needs --temperature"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--subset-stride", type=int, default=1)
    parser.add_argument("--subset-limit", type=int, default=0)
    args = parser.parse_args()
    config = SimpleNamespace(**yaml.safe_load(Path(args.config).read_text()))
    config.checkpoint = args.checkpoint
    config.device = args.device
    if args.temperature is None:
        # Fail closed rather than silently decode greedily under flags that
        # only the sampled path honours.
        sampling = dict(samples=1, seed=0, top_p=1.0, top_k=-1, subset_stride=1)
        supplied = {
            name: getattr(args, name)
            for name, default in sampling.items()
            if getattr(args, name) != default
        }
        if supplied or args.subset_limit:
            raise SystemExit(f"--temperature is required by {sorted(supplied)}")
        audit = evaluate(
            config, args.manifest, args.output, args.max_tokens, args.batch_size
        )
        reported = [
            "sample_count",
            "total_errors",
            "total_reference_words",
            "corpus_wer",
        ]
    else:
        audit = evaluate_sampled(
            config,
            args.manifest,
            args.output,
            args.max_tokens,
            args.batch_size,
            args.temperature,
            args.samples,
            args.seed,
            args.top_p,
            args.top_k,
            args.subset_stride,
            args.subset_limit,
        )
        reported = [
            "temperature",
            "seed",
            "prompt_count",
            "samples_per_prompt",
            "sample_count",
            "total_errors",
            "total_reference_words",
            "corpus_wer",
            "corpus_wer_best",
            "corpus_wer_worst",
            "eos_failures",
            "subset_sha256",
        ]
    print(json.dumps({key: audit[key] for key in reported}))


if __name__ == "__main__":
    main()
