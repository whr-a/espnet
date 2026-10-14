# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Prepare native nine-stream OpusLM RL conditions from Kaldi wav.scp/text.

Audio tokenization uses the released ESPnet DAC and XEUS layer / k-means. The
complete delayed codec/SSL prompt is recorded, so training and rollout consume
the same condition. References are supervision only and never enter the prompt.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_kaldi(path):
    data = {}
    for line in Path(path).read_text().splitlines():
        key, value = line.split(maxsplit=1)
        if key in data:
            raise ValueError(f"Duplicate utterance: {key}")
        data[key] = value
    return data


class NativeAudioTokenizer:
    def __init__(
        self,
        metadata,
        dac_config,
        dac_checkpoint,
        xeus_config,
        xeus_checkpoint,
        kmeans,
        device="cpu",
    ):
        import joblib

        from espnet2.gan_codec.dac import DAC
        from espnet2.gan_codec.espnet_model import ESPnetGANCodecModel
        from espnet2.tasks.ssl import SSLTask

        self.metadata, self.device = metadata, device
        recipe = yaml.safe_load(Path(dac_config).read_text())
        self.codec = ESPnetGANCodecModel(
            codec=DAC(**recipe["codec_conf"]), **recipe["model_conf"]
        )
        codec_state = torch.load(dac_checkpoint, map_location="cpu", weights_only=True)
        # Published v1.4 DAC used a constant Snake alpha=1 (not a state tensor).
        # Current ESPnet registers it as a trainable parameter. Reconstruct the
        # declared old architecture without inventing checkpoint weights.
        if not any(key.endswith(".alpha") for key in codec_state):
            from espnet2.gan_codec.shared.encoder.snake_activation import Snake1d

            for module in self.codec.modules():
                if isinstance(module, Snake1d):
                    alpha = module.alpha.detach()
                    del module._parameters["alpha"]
                    module.register_buffer("alpha", alpha, persistent=False)
        self.codec.load_state_dict(codec_state, strict=True)
        self.codec.eval().float().to(device)
        self.ssl, _ = SSLTask.build_model_from_file(
            xeus_config, xeus_checkpoint, device=device
        )
        self.ssl.eval().float()
        if hasattr(self.ssl, "util_attributes"):
            self.ssl.util_attributes.discard("mask")
            self.ssl.util_attributes.discard("block_mask")
        self.kmeans = joblib.load(kmeans)
        self.artifacts = {
            str(Path(path).resolve()): sha256(path)
            for path in (
                dac_config,
                dac_checkpoint,
                xeus_config,
                xeus_checkpoint,
                kmeans,
            )
        }
        self.layer = int(metadata.get("xeus_layer", 18))

    @torch.inference_mode()
    def encode(self, waveform):
        wav = torch.as_tensor(waveform.copy(), device=self.device, dtype=torch.float32)
        codec = (
            self.codec.encode(wav[None, None])
            .permute(1, 2, 0)[0, :, :8]
            .long()
            .cpu()
            .numpy()
        )
        output = self.ssl.encode(
            wav[None], torch.tensor([len(wav)], device=self.device)
        )
        features = (
            output["encoder_output"]
            if isinstance(output, dict)
            else output[0] if isinstance(output, tuple) else output
        )
        if isinstance(features, (tuple, list)):
            features = features[min(self.layer, len(features) - 1)]
        ssl = self.kmeans.predict(features[0].float().cpu().numpy()).astype(np.int64)
        length = min(len(ssl), len(codec))
        if length < 1:
            raise ValueError("Native tokenizer produced no audio frames")
        ssl_size = self.metadata["ssl_token_end"] - self.metadata["ssl_token_start"]
        if ((ssl < 0) | (ssl >= ssl_size)).any():
            raise ValueError("XEUS kmeans label outside model vocabulary")
        if (
            (codec < 0) | (codec >= self.metadata.get("codec_per_stream_size", 1024))
        ).any():
            raise ValueError("DAC code outside model vocabulary")
        frames = np.column_stack(
            [
                ssl[:length] + self.metadata["ssl_token_start"],
                codec[:length]
                + self.metadata["codec_token_start"]
                + np.arange(8)[None] * self.metadata.get("codec_per_stream_size", 1024),
            ]
        )
        return frames


def asr_prompt(frames, metadata, family):
    """Native ASR segment with delayed codec streams and explicit flush gap."""
    nq = int(metadata.get("nq", 9))
    if frames.ndim != 2 or frames.shape[1] != nq:
        raise ValueError("Opus audio frames must retain nine streams")
    zero = [0] * nq
    # Speech task layout is checkpoint bound. Dialogue can still use its shared
    # ASR task: the published vocabulary and model support this native task.
    prefix = [
        [metadata["eos_token_id"]] + [0] * (nq - 1),
        [metadata.get("codec_ssl_asr_task_token_id", 80)] + [0] * (nq - 1),
        [metadata["codec_ssl_start_end_token_id"]] + [0] * (nq - 1),
    ]
    delayed = np.zeros((len(frames) + nq, nq), dtype=np.int64)
    for stream in range(nq):
        delayed[stream : stream + len(frames), stream] = frames[:, stream]
    delayed[len(frames), 0] = metadata["codec_ssl_start_end_token_id"]
    prefix += delayed.tolist()
    prefix += [[metadata["text_bpe_start_end_token_id"]] + [0] * (nq - 1)]
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "wav-scp",
        "text",
        "model-assets",
        "output",
        "dac-config",
        "dac-checkpoint",
        "xeus-config",
        "xeus-checkpoint",
        "kmeans",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--family", choices=("opuslm", "opuslm_dialogue"), default="opuslm"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen Opus manifest {output}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    import soundfile
    import torchaudio

    wavs, refs = read_kaldi(args.wav_scp), read_kaldi(args.text)
    if set(wavs) != set(refs):
        raise ValueError("wav.scp and text utterance IDs must match exactly")
    metadata = json.loads((Path(args.model_assets) / "config.json").read_text())
    tokenizer = NativeAudioTokenizer(
        metadata,
        args.dac_config,
        args.dac_checkpoint,
        args.xeus_config,
        args.xeus_checkpoint,
        args.kmeans,
        args.device,
    )
    rows = []
    for i, (key, audio) in enumerate(wavs.items()):
        if args.limit is not None and i >= args.limit:
            break
        if audio.endswith("|"):
            raise ValueError(
                "Materialize wav.scp pipe entries before freezing Opus RL audio"
            )
        audio = (
            str((Path(args.wav_scp).resolve().parent / audio).resolve())
            if not Path(audio).is_absolute()
            else audio
        )
        source = {
            "id": key,
            "audio": str(Path(audio).resolve()),
            "reference": refs[key],
            "sample_rate": 16000,
        }
        wav, sample_rate = soundfile.read(
            source["audio"], dtype="float32", always_2d=True
        )
        wav = wav[:, 0]
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav.copy()), sample_rate, 16000
            ).numpy()
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError("Nonempty finite waveform required")
        frames = tokenizer.encode(wav)
        source.update(
            prompt_token_ids=asr_prompt(frames, metadata, args.family),
            prompt_codec_mask=[False] * 3 + [True] * (len(frames) + 9) + [False],
            native_model_inputs={},
            mode="audio_text",
            native_preparation={
                "family": args.family,
                "audio_sha256": sha256(audio),
                "waveform_sha256": hashlib.sha256(wav.tobytes()).hexdigest(),
                "native_audio_frames": len(frames),
                "native_frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                "sample_rate": 16000,
                "encoder_layer": tokenizer.layer,
                "artifacts_sha256": tokenizer.artifacts,
                "condition": "original wav; XEUS kmeans + DAC; all nine delayed token streams",
            },
        )
        rows.append(source)
        print(
            json.dumps(
                {
                    "utterance": key,
                    "native_audio_frames": len(frames),
                    "prompt_frames": len(source["prompt_token_ids"]),
                }
            ),
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    output.with_suffix(output.suffix + ".audit.json").write_text(
        json.dumps(
            {
                "wav_scp_sha256": sha256(args.wav_scp),
                "text_sha256": sha256(args.text),
                "manifest_sha256": sha256(output),
                "rows": len(rows),
                "family": args.family,
                "artifacts_sha256": tokenizer.artifacts,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
