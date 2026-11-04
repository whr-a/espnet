# Copyright 2026 Individual Contributor: Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Attach a new task's supervision to an already prepared manifest.

A family whose prompt is a prepared token matrix (OpusLM) can then reuse an
audited audio condition and change only the reward.
"""

import json
from pathlib import Path

from espnet2.rl.utils import sha256

from .kaldi import write_manifest


def append_text_instruction(prompt_token_ids, text_ids, text_token_start):
    """Append tokenized instruction text to a prepared multi-stream prompt.

    A prepared OpusLM prompt is ``[frames, nq]``; its primary stream carries
    native control and text ids while the secondary streams carry codec/SSL
    conditions. An instruction is text, so it occupies the primary stream only
    and pads the rest, exactly as ``_native_prompt`` builds a text prompt.
    """
    if not prompt_token_ids or not all(
        len(frame) == len(prompt_token_ids[0]) for frame in prompt_token_ids
    ):
        raise ValueError("A prepared prompt must be a nonempty rectangular [P, nq]")
    width = len(prompt_token_ids[0])
    return [list(frame) for frame in prompt_token_ids] + [
        [int(token) + int(text_token_start)] + [0] * (width - 1) for token in text_ids
    ]


def derive_task_manifest(
    source_manifests,
    output,
    *,
    task,
    supervision,
    instruction_tokens=None,
    text_token_start=None,
    drop=(),
):
    """Attach a new task's supervision to already prepared records.

    ``supervision`` maps a record id to the fields that task's reward needs.
    The audio condition — ``prompt_token_ids``, ``prompt_codec_mask``,
    ``native_preparation`` — is copied unchanged, so the derived manifest
    reuses an audited preparation instead of re-encoding audio. When
    ``instruction_tokens`` is given, those text ids are appended to each
    prompt and the original prompt hash is kept beside the new one, because
    the prompt is no longer the one ``native_preparation`` describes.
    """
    records, seen = [], set()
    for path in source_manifests:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                raise ValueError(f"Duplicate source record {record['id']!r}")
            seen.add(record["id"])
            fields = supervision.get(record["id"])
            if fields is None:
                raise ValueError(f"No {task} supervision for record {record['id']!r}")
            for name in drop:
                record.pop(name, None)
            if instruction_tokens is not None:
                if text_token_start is None:
                    raise ValueError(
                        "Appending an instruction needs the export's text_token_start"
                    )
                original = record["prompt_token_ids"]
                record["prompt_token_ids"] = append_text_instruction(
                    original, instruction_tokens, text_token_start
                )
                record["prompt_codec_mask"] = list(
                    record.get("prompt_codec_mask", [False] * len(original))
                ) + [False] * len(instruction_tokens)
                record["derived_prompt"] = {
                    "appended_text_tokens": list(instruction_tokens),
                    "text_token_start": int(text_token_start),
                    "original_frames": len(original),
                    "frames": len(record["prompt_token_ids"]),
                }
            # A mixed manifest's supervision names each record's own task, so
            # its reward_task wins over the one this call was given.
            record.update({"reward_task": task, **fields})
            records.append(record)
    if not records:
        raise ValueError("No source record was derived")
    return write_manifest(
        output,
        records,
        {
            "task": task,
            "derived_from": [str(Path(path).resolve()) for path in source_manifests],
            "derived_from_sha256": [sha256(path) for path in source_manifests],
            "instruction_tokens": (
                None if instruction_tokens is None else list(instruction_tokens)
            ),
            "text_token_start": text_token_start,
            "dropped_fields": list(drop),
            "audio_condition": "copied unchanged from the source manifest",
        },
    )
