# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Audited ASR-WER on generated audio and AVQA choice rewards."""

import hashlib
import json
import re
from pathlib import Path

from espnet2.rl.rewards import word_error_rate, words

from .data import OPTION_LABEL


def qa_reward(source, generation):
    """WEST default GRPO: equal weights for exact accuracy and answer format."""
    text = generation["text"]
    match = re.search(r"<answer>(.*?)</answer>", text)
    answer = match.group(1).strip() if match else text.strip()
    accuracy = float(answer == source["reference"])
    format_score = float(re.fullmatch(r"<answer>.*?</answer>", text, re.S) is not None)
    return 0.5 * (accuracy + format_score)


def qa_think_reward(source, generation):
    """WEST think mode for native caption-reasoning-answer responses.

    Keep WEST's exact-string answer check, including its newline behavior.
    Structural rewards require exactly one complete answer/think block.
    Select this protocol only before freezing a new independent round.
    """
    text = generation["text"]
    match = re.search(r"<answer>(.*?)</answer>", text)
    answer = match.group(1).strip() if match else text.strip()
    accuracy = float(answer == source["reference"])
    answer_format = float(len(re.findall(r"<answer>.*?</answer>", text, re.S)) == 1)
    think_format = float(len(re.findall(r"<think>.*?</think>", text, re.S)) == 1)
    return 0.8 * accuracy + 0.1 * answer_format + 0.1 * think_format


def mmau_answer(text):
    """The prediction the official MMAU pipeline scores.

    As in WEST's decode_mmau.py: the content of the answer tag, across lines,
    or the whole response when the model wrote no tag.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return (match.group(1) if match else text).strip()


def mmau_match(source, generation):
    """Official MMAU evaluation.py token-set match, with all rows retained.

    Scores the extracted answer, never the caption or reasoning around it,
    whose discussion of wrong options would otherwise always fail the match.
    Option letters are removed from the reference and the choices, as in the
    official test set, so that "a", "b", ... never count as wrong-option words.
    """
    tokenize = lambda value: set(re.findall(r"\b\w+\b", value.lower()))
    answer = tokenize(OPTION_LABEL.sub("", source["reference"]))
    prediction = tokenize(mmau_answer(generation["text"]))
    incorrect = set()
    for choice in source["choices"]:
        tokens = tokenize(OPTION_LABEL.sub("", str(choice)))
        if tokens != answer:
            incorrect.update(tokens - answer)
    return (
        bool(prediction)
        and answer.issubset(prediction)
        and prediction.isdisjoint(incorrect)
    )


class ASRWERReward:
    """Load a fixed independent ASR and decode actual generated codec actions."""

    def __init__(self, config):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            XcodecModel,
        )

        self.torch = torch
        self.config = config
        self.options = config.adapter_options
        self.directory = Path(config.output_dir) / "reward_audit"
        self.directory.mkdir(exist_ok=True)
        self.codec = XcodecModel(
            AutoConfig.from_pretrained(self.options["codec_config_path"])
        )
        native = torch.load(config.checkpoint, map_location="cpu", weights_only=True)
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
        self.codec.eval().to(self.options.get("reward_device", "cpu"))
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
            .to(self.options.get("reward_device", "cpu"))
        )

    def __call__(self, source, generation):
        import numpy as np
        import soundfile as sf

        torch = self.torch
        matrix = torch.tensor(generation["token_ids"], dtype=torch.long)
        base = 152192
        size = 1025
        q = 8
        frames = matrix[(matrix[:, 0] >= base) & (matrix[:, 0] < base + size)]
        if len(frames) < q:
            raise ValueError(
                "TTS candidate has no decodable audio; preserve as runtime failure"
            )
        # Native delay renderer shifts stream s by s frames. Reserved codec
        # pad tokens are removed only where this fixed delay discards them;
        # an invalid token in an actual aligned codebook frame is an error.
        length = len(frames) - q + 1
        aligned = []
        for stream in range(q):
            tokens = frames[stream : stream + length, stream]
            offset = base + stream * size + 1
            if ((tokens < offset) | (tokens >= offset + 1024)).any():
                raise ValueError(
                    f"Invalid sampled codec token in aligned stream {stream}; no clamping"
                )
            aligned.append(tokens - offset)
        aligned = torch.stack(aligned, 0)[None].to(self.codec.device)
        with torch.inference_mode():
            waveform = (
                self.codec.decode(aligned).audio_values.squeeze().cpu().float().numpy()
            )
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError("Invalid generated WAV")
        rate = int(self.codec.config.sample_rate)
        digest = hashlib.sha256(waveform.tobytes()).hexdigest()
        target = self.directory / f"{source['id']}-{digest[:16]}"
        sf.write(str(target) + ".wav", waveform, rate)
        inputs = self.processor(waveform, sampling_rate=rate, return_tensors="pt")
        kwargs = {"max_new_tokens": 256}
        if getattr(self.asr.generation_config, "is_multilingual", False):
            kwargs.update(language="en", task="transcribe")
        with torch.inference_mode():
            ids = self.asr.generate(**inputs.to(self.asr.device), **kwargs)
        hypothesis = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        value = word_error_rate(source["reference"], hypothesis)
        Path(str(target) + ".json").write_text(
            json.dumps(
                {
                    "source_id": source["id"],
                    "reference": source["reference"],
                    "hypothesis": hypothesis,
                    "wer": value,
                    "reward": -value,
                    "waveform_sha256": digest,
                    "sample_rate": rate,
                    "asr_model": self.asr_tag,
                    "asr_revision": self.asr_revision,
                    "codec_frames": length,
                    "generated_tokens": generation["token_ids"],
                },
                indent=2,
            )
            + "\n"
        )
        return -value


_REWARD = None


def tts_asr_wer(source, generation):
    global _REWARD
    if _REWARD is None:
        from espnet2.rl.config import ESPnetRLConfig

        config = ESPnetRLConfig.load(source["reward_config"])
        _REWARD = ASRWERReward(config)
    return _REWARD(source, generation)
