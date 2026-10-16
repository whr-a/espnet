# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""AVQA choice rewards and audited ASR-WER on generated audio.

Each reward returns its value as ``"reward"`` together with the components it
was made of, which ``espnet2.rl.rewards.outcome_rewards`` logs separately.
"""

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path

from espnet2.rl.rewards import word_error_rate

from .data import OPTION_LABEL, validate_native_qa_sources
from .qa_contract import match_choice, parse_native_qa_answer

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
# Where Bagpiper's untagged responses state their choice, as in "the correct
# answer is: motorcycle", "**Final Answer:** motorcycle", "the most accurate
# choice is: by the car" or "the most correct answer from the given list is:".
# The last cue counts.
CUE = re.compile(
    r"(?:(?:final|correct) answer(?:[^.:\n]{0,40}?(?:\bis\b|:))?"
    r"|answer is|choice is|answer\s*:)\s*:?",
    re.I,
)
# Markdown emphasis, quotes and punctuation around a stated answer.
DECORATION = " \t*_`\"'“”‘’.,;:!"
# Bagpiper's codec layer 0 starts here; each of the 8 layers holds a pad id
# followed by the 1024 Xcodec codes.
CODEC_BASE, CODEC_SIZE, CODEC_STREAMS = 152192, 1025, 8


def mmau_answer(text):
    """The prediction WEST's decode_mmau.py scores: the content of the first
    answer tag, across lines, or the whole response when there is no tag."""
    match = ANSWER.search(text)
    return (match.group(1) if match else text).strip()


def qa_answer(text):
    """The prediction the v1 reward scores, and where it was found.

    The first answer tag (as WEST's evaluation reads it), else the line after
    the last answer cue, else the whole response: ``(answer, path)`` with
    ``path`` one of ``"tag"``, ``"cue"`` and ``"text"``.
    """
    match = ANSWER.search(text)
    if match:
        return match.group(1).strip(), "tag"
    cues = list(CUE.finditer(text))
    if cues:
        lines = text[cues[-1].end() :].splitlines()
        stated = next((line for line in lines if line.strip(DECORATION)), "")
        return stated.strip(DECORATION), "cue"
    return text.strip(), "text"


def mmau_match(source, prediction):
    """Official MMAU evaluation.py token-set match of an extracted prediction.

    All answer words must be present and no word unique to a wrong choice may
    be. Option letters are removed from the reference and the choices, as in
    the official test set, so that "a", "b", ... never count as wrong-option
    words.
    """
    tokenize = lambda value: set(re.findall(r"\b\w+\b", value.lower()))
    answer = tokenize(OPTION_LABEL.sub("", source["reference"]))
    predicted = tokenize(prediction)
    incorrect = set()
    for choice in source["choices"]:
        tokens = tokenize(OPTION_LABEL.sub("", str(choice)))
        if tokens != answer:
            incorrect.update(tokens - answer)
    return bool(predicted) and answer <= predicted and predicted.isdisjoint(incorrect)


def west_accuracy(source, text):
    """WEST's training accuracy: the first answer tag, without DOTALL, else the
    whole response, must equal the reference string exactly."""
    match = re.search(r"<answer>(.*?)</answer>", text)
    return float((match.group(1) if match else text).strip() == source["reference"])


def qa_reward(source, generation):
    """Historical half-scaled WEST default reward, as earlier runs used it:
    exact-string accuracy and a response that is nothing but one answer tag,
    with equal weights."""
    text = generation["text"]
    accuracy = west_accuracy(source, text)
    answer_format = float(re.fullmatch(r"<answer>.*?</answer>", text, re.S) is not None)
    return {
        "reward": 0.5 * (accuracy + answer_format),
        "acc": accuracy,
        "format": answer_format,
    }


def qa_west_default_reward(source, generation):
    """WEST's default accuracy-plus-format reward, with its original scale.

    Matches west/utils/rewards.py and west/bin/train_grpo.py at upstream
    b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb: exact, case-sensitive answer
    accuracy plus the whole-completion answer-format check, each weighted 1.
    Keep ``qa_reward`` unchanged for runs frozen with its historical scale.
    """
    components = qa_reward(source, generation)
    return {**components, "reward": components["acc"] + components["format"]}


def qa_v1_reward(source, generation):
    """Protocol qa_west_bagpiper_v1: WEST's equal accuracy and format weights
    with an answer Bagpiper's free-form responses can earn.

    Accuracy is the official MMAU match of ``qa_answer``; format requires
    exactly one answer block anywhere, as WEST's think mode does, so captions
    and reasoning around it are allowed.
    """
    text = generation["text"]
    answer, path = qa_answer(text)
    accuracy = float(mmau_match(source, answer))
    answer_format = float(len(ANSWER.findall(text)) == 1)
    return {
        "reward": 0.5 * accuracy + 0.5 * answer_format,
        "acc": accuracy,
        "format": answer_format,
        **{f"path_{name}": float(path == name) for name in ("tag", "cue", "text")},
    }


def qa_think_reward(source, generation):
    """WEST think mode for native caption-reasoning-answer responses.

    Keep WEST's exact-string answer check, including its newline behavior.
    Structural rewards require exactly one complete answer/think block.
    Select this protocol only before freezing a new independent round.
    """
    text = generation["text"]
    accuracy = west_accuracy(source, text)
    answer_format = float(len(re.findall(r"<answer>.*?</answer>", text, re.S)) == 1)
    think_format = float(len(re.findall(r"<think>.*?</think>", text, re.S)) == 1)
    return {
        "reward": 0.8 * accuracy + 0.1 * answer_format + 0.1 * think_format,
        "acc": accuracy,
        "format": answer_format,
        "think_format": think_format,
    }


def qa_bagpiper_v2_reward(source, generation):
    """Native Bagpiper QA: correct final choice 1.0 plus terminal format 0.1.

    Caption and reasoning may precede a native final-option marker, Markdown
    option header or answer block. Parse without the reference, then compare
    canonical choice indices. Identical normalized option texts share their
    first index.
    This is a new protocol; the historical WEST and V1 rewards stay unchanged.
    """
    parsed = parse_native_qa_answer(generation["text"], source["choices"])
    reference_index, _ = match_choice(source["reference"], source["choices"])
    accuracy = float(
        parsed["predicted_index"] is not None
        and reference_index is not None
        and parsed["predicted_index"] == reference_index
    )
    answer_format = float(parsed["format"])
    return {
        "reward": accuracy + 0.1 * answer_format,
        "acc": accuracy,
        "format": answer_format,
        "parsed": float(parsed["predicted_index"] is not None),
        **{
            f"path_{path}": float(parsed["path"] == path)
            for path in (
                "native_cue",
                "option_header",
                "answer_tag",
                "post_think",
                "plain",
                "none",
            )
        },
    }


qa_bagpiper_v2_reward.validate_sources = validate_native_qa_sources


def aligned_codes(token_ids):
    """The [8, T] Xcodec codes of a candidate's codec frames, or why it has none.

    The native delay renderer shifts stream s by s frames. Reserved codec pad
    tokens are removed only where this fixed delay discards them; an invalid
    token in an actual aligned codebook frame is never clamped. Returns
    ``(codes, "ok")``, ``(None, "no_audio")`` for fewer than 8 codec frames
    or ``(None, "invalid_codec")``.
    """
    import torch

    matrix = torch.as_tensor(token_ids, dtype=torch.long)
    frames = matrix[
        (matrix[:, 0] >= CODEC_BASE) & (matrix[:, 0] < CODEC_BASE + CODEC_SIZE)
    ]
    if len(frames) < CODEC_STREAMS:
        return None, "no_audio"
    length = len(frames) - CODEC_STREAMS + 1
    aligned = []
    for stream in range(CODEC_STREAMS):
        tokens = frames[stream : stream + length, stream]
        offset = CODEC_BASE + stream * CODEC_SIZE + 1
        if ((tokens < offset) | (tokens >= offset + CODEC_SIZE - 1)).any():
            return None, "invalid_codec"
        aligned.append(tokens - offset)
    return torch.stack(aligned), "ok"


@contextmanager
def _ieee_float32(torch):
    # The codec's convolutions and the ASR's matmuls compute exactly as on CPU.
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


class ASRWERReward:
    """Negative WER of a fixed independent ASR on the actual generated audio.

    Codec actions are decoded one candidate at a time on ``reward_device``
    (default: the learner's device), since the decoder's convolutions would
    see past the end of a padded batch. The frozen ASR then transcribes
    ``asr_batch_size`` waveforms at once; each is padded to its own 30 s
    window, as it is alone. A candidate without decodable audio or with an
    invalid codec token gets WER 1.0 and is counted. The reward is
    -min(WER, 1) for every candidate: audio, however many words the ASR
    inserts, never scores below a failure. Each call
    writes one line per candidate to ``reward_audit/batch-NNNNNN.jsonl`` and
    the WAV and full JSON of ``audits`` evenly spaced candidates (all if None).
    """

    def __init__(self, config, audits=4):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            XcodecModel,
        )

        self.torch = torch
        self.config, self.audits, self.calls = config, audits, 0
        self.options = config.adapter_options
        self.device = torch.device(self.options.get("reward_device", config.device))
        self.batch_size = int(self.options.get("asr_batch_size", 64))
        self.directory = Path(config.output_dir) / "reward_audit"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.codec = XcodecModel(
            AutoConfig.from_pretrained(self.options["codec_config_path"])
        )
        self.rate = int(self.codec.config.sample_rate)
        # Only the frozen codec is read from the native checkpoint.
        native = torch.load(
            config.checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        native = native.get("module", native)
        prefix = "multimodal_io_dict.discrete_audio.codec_model."
        self.codec.load_state_dict(
            {
                key[len(prefix) :]: value
                for key, value in native.items()
                if key.startswith(prefix)
            },
            strict=True,
        )
        del native
        self.codec.eval().to(self.device)
        self.asr_tag = self.options["asr_model"]
        self.asr_revision = self.options.get("asr_revision")
        if not Path(self.asr_tag).is_dir() and not self.asr_revision:
            raise ValueError("ASR reward must freeze a local model or revision")
        self.processor = AutoProcessor.from_pretrained(
            self.asr_tag, revision=self.asr_revision
        )
        self.asr = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                self.asr_tag, revision=self.asr_revision
            )
            .eval()
            .to(self.device)
        )
        self.generate_kwargs = {"max_new_tokens": 256}
        if getattr(self.asr.generation_config, "is_multilingual", False):
            self.generate_kwargs.update(language="en", task="transcribe")

    def transcribe(self, sources, generations):
        """One row per candidate: its transcript, WER and decoding status."""
        import numpy as np

        torch = self.torch
        waveforms, statuses, lengths = [], [], []
        with _ieee_float32(torch), torch.inference_mode():
            for generation in generations:
                codes, status = aligned_codes(generation["token_ids"])
                waveform = None
                if codes is not None:
                    waveform = (
                        self.codec.decode(codes[None].to(self.device))
                        .audio_values.squeeze()
                        .float()
                        .cpu()
                        .numpy()
                    )
                    if (
                        waveform.ndim != 1
                        or not waveform.size
                        or not np.isfinite(waveform).all()
                    ):
                        raise ValueError("Invalid generated WAV")
                waveforms.append(waveform)
                statuses.append(status)
                lengths.append(0 if codes is None else codes.shape[1])
            audible = [
                i for i, waveform in enumerate(waveforms) if waveform is not None
            ]
            hypotheses = {}
            for start in range(0, len(audible), self.batch_size):
                chunk = audible[start : start + self.batch_size]
                inputs = self.processor(
                    [waveforms[i] for i in chunk],
                    sampling_rate=self.rate,
                    return_tensors="pt",
                )
                ids = self.asr.generate(
                    **inputs.to(self.device), **self.generate_kwargs
                )
                hypotheses.update(
                    zip(
                        chunk,
                        self.processor.batch_decode(ids, skip_special_tokens=True),
                    )
                )
        rows = []
        for index, (source, status) in enumerate(zip(sources, statuses, strict=True)):
            hypothesis = hypotheses.get(index)
            rows.append(
                {
                    "source_id": source["id"],
                    "reference": source["reference"],
                    "hypothesis": hypothesis,
                    "wer": (
                        1.0
                        if hypothesis is None
                        else word_error_rate(source["reference"], hypothesis)
                    ),
                    "status": status,
                    "codec_frames": lengths[index],
                }
            )
        self._audit(rows, waveforms, generations)
        return rows

    def _audit(self, rows, waveforms, generations):
        import soundfile as sf

        with (self.directory / f"batch-{self.calls:06d}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        step = 1 if self.audits is None else -(-len(rows) // self.audits)
        for index in range(0, len(rows), step):
            row, waveform = rows[index], waveforms[index]
            name = f"{row['source_id']}-{self.calls:06d}-{index:04d}"
            digest = None
            if waveform is not None:
                digest = hashlib.sha256(waveform.tobytes()).hexdigest()
                sf.write(str(self.directory / f"{name}.wav"), waveform, self.rate)
            (self.directory / f"{name}.json").write_text(
                json.dumps(
                    {
                        **row,
                        "reward": -min(row["wer"], 1.0),
                        "waveform_sha256": digest,
                        "sample_rate": self.rate,
                        "asr_model": self.asr_tag,
                        "asr_revision": self.asr_revision,
                        "generated_tokens": generations[index]["token_ids"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        self.calls += 1

    def __call__(self, sources, generations):
        return [
            {
                "reward": -min(row["wer"], 1.0),
                "wer": row["wer"],
                "no_audio": float(row["status"] == "no_audio"),
                "invalid_codec": float(row["status"] == "invalid_codec"),
            }
            for row in self.transcribe(sources, generations)
        ]


_REWARD = None


def tts_asr_wer(sources, generations):
    global _REWARD
    if _REWARD is None:
        from espnet2.rl.config import ESPnetRLConfig

        _REWARD = ASRWERReward(ESPnetRLConfig.load(sources[0]["reward_config"]))
    return _REWARD(sources, generations)


tts_asr_wer.batched = True
