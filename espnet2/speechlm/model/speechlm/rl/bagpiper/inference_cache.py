# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Bounded, single-candidate state for the Bagpiper inference model.

This state is private to one model instance. In particular, a mutable HF cache
is never copied to another candidate or to the learner.
"""

from dataclasses import dataclass
from typing import Any


class CacheContractError(ValueError):
    """A request cannot safely extend the currently cached native sequence."""


@dataclass
class CandidateState:
    key: tuple
    sequence: tuple = ()
    position: int = -1
    prefix_length: int = 0
    past_key_values: Any = None
    audio_features: Any = None


class CandidateInferenceCache:
    """Retain at most one candidate, one feature Tensor and a bounded HF KV."""

    def __init__(self, max_frames, max_audio_bytes):
        if (
            type(max_frames) is not int
            or type(max_audio_bytes) is not int
            or max_frames < 1
            or max_audio_bytes < 1
        ):
            raise ValueError("Inference cache bounds must be positive")
        self.max_frames = max_frames
        self.max_audio_bytes = max_audio_bytes
        self.state = None
        self.counters = {
            "prefills": 0,
            "decodes": 0,
            "audio_encoder_calls": 0,
            "resets": 0,
            "uncached_forwards": 0,
        }
        self.last_trace = {"operation": "empty", "kv_frames": 0}

    def clear(self, reason):
        if self.state is not None:
            self.counters["resets"] += 1
        self.state = None
        self.last_trace = {"operation": "cleared", "reason": reason, "kv_frames": 0}

    def prepare(self, key, sequence, position, *, reset=False):
        """Return the first uncached frame; require exact nine-stream extension."""
        if type(position) is not int or position < 0:
            raise CacheContractError("Invalid candidate position")
        if type(reset) is not bool:
            raise CacheContractError("Candidate reset must be a boolean")
        if reset and position != 0:
            raise CacheContractError(
                "An explicit candidate reset requires position zero"
            )
        if len(sequence) > self.max_frames:
            raise CacheContractError("prepare called above the cache frame bound")
        if reset or self.state is None or self.state.key != key:
            if position != 0:
                raise CacheContractError(
                    "A new candidate identity requires position zero"
                )
            self.clear("candidate_start")
            self.state = CandidateState(key=key, prefix_length=len(sequence))
            return 0
        state = self.state
        if position != state.position + 1:
            raise CacheContractError("Candidate position must increase by exactly one")
        if len(sequence) != len(state.sequence) + 1:
            raise CacheContractError(
                "Native sequence must append exactly one full frame"
            )
        if sequence[:-1] != state.sequence:
            raise CacheContractError(
                "Cached native prefix differs in a primary or secondary stream"
            )
        if len(sequence) != state.prefix_length + position:
            raise CacheContractError(
                "Native sequence length and candidate position differ"
            )
        if state.past_key_values is None:
            raise CacheContractError("Candidate has no committed HF KV cache")
        if int(state.past_key_values.get_seq_length()) != len(state.sequence):
            raise CacheContractError(
                "HF KV length differs from the complete native prefix"
            )
        return len(state.sequence)

    def commit(self, sequence, position, past_key_values, start):
        if int(past_key_values.get_seq_length()) != len(sequence):
            raise CacheContractError(
                "HF KV did not consume the complete native extension"
            )
        self.state.sequence = sequence
        self.state.position = position
        self.state.past_key_values = past_key_values
        operation = "prefill" if start == 0 else "decode"
        self.counters["prefills" if start == 0 else "decodes"] += 1
        self.last_trace = {
            "operation": operation,
            "input_frames": len(sequence) - start,
            "kv_frames": len(sequence),
            "audio_features_retained": self.state.audio_features is not None,
        }

    def retain_audio(self, features):
        if features.numel() * features.element_size() <= self.max_audio_bytes:
            self.state.audio_features = features.detach()

    def snapshot(self):
        return {
            **self.last_trace,
            "counters": dict(self.counters),
            "max_frames": self.max_frames,
            "max_audio_bytes": self.max_audio_bytes,
            "active_candidates": int(self.state is not None),
        }
