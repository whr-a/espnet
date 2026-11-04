# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""OpusLM's own rewards: the shape of a sampled nine-stream codec block.

Its text rewards are the model-agnostic ones in :mod:`espnet2.rl.rewards`.
"""

from espnet2.rl.rewards import require_fields

from .multistream import deinterleave_codec


def codec_block_reward(source, generation):
    """Shape-only reward for a nine-stream codec rollout: did it make audio?

    This is the smallest honest reward for OpusLM's generation direction. It
    scores the block the engine actually sampled -- how much of it decodes
    under the fixed ARDelay alignment, and whether the model closed the block
    itself instead of running into the token cap -- and nothing about how the
    audio sounds. It loads no model, so it runs on CPU with every other
    reward.

    A TTS quality reward (ASR WER on the decoded waveform, as
    bagpiper/rewards.py:ASRWERReward does) needs a codec decoder and a frozen
    ASR on a GPU; this one deliberately does not, so a training loop can be
    brought up and its train/inference consistency measured before any of that
    is wired.
    """
    metadata = source.get("codec_metadata")
    if not metadata:
        raise ValueError(
            "codec_block_reward requires codec_metadata on every manifest record"
        )
    frames = generation["token_ids"]
    if not frames or not isinstance(frames[0], (list, tuple)):
        raise ValueError("codec_block_reward requires a multistream [T,nq] rollout")
    codes, status = deinterleave_codec(frames, metadata)
    closed = bool(
        any(
            int(frame[0]) == int(metadata["codec_ssl_start_end_token_id"])
            for frame in frames
        )
    )
    decodable = float(codes.shape[1]) if status == "ok" else 0.0
    # Normalised by the cap so the reward stays in [0, 1] whatever max_tokens is.
    covered = min(1.0, decodable / max(1, len(frames)))
    return {
        "reward": covered + (0.25 if closed else 0.0),
        "decodable_frames": decodable,
        "closed": float(closed),
        "invalid_codec": float(status == "invalid_codec"),
        "no_audio": float(status == "no_audio"),
    }


codec_block_reward.validate_sources = lambda records: require_fields(
    records, "codec_block", ("codec_metadata",)
)
