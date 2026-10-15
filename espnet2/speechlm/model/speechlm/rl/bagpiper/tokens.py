# Copyright 2025 Individual Contributor: Haoran Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bagpiper frame supports, decoding phases and codec delays."""

import torch

# The primary-stream support of a Bagpiper frame. The rollout engine records
# it per frame (``support``) and the learner scores the frame under it. The
# seven secondary codec streams are active exactly when the raw primary argmax
# is a codec id in text_audio mode (``stream_mask``); text_only frames never
# have codec support.
TEXT, CODEC, DETECT, DETECT_AUDIO, ASSISTANT, EOS = range(6)
SUPPORTS = ("text", "codec", "detect", "detect_audio", "assistant", "eos")
# The text_audio decoding phase is a function of the sampled primaries:
# text -(eot)-> transition (forced assistant) -> audio -(eot)-> audio_stop
# (forced eos). text_only stays in text.
PHASES = ("text", "transition", "audio", "audio_stop")


def support_masks(geometry, vocab_size, device=None):
    """[len(SUPPORTS), vocab] allowed primary ids of every support."""
    masks = torch.zeros(len(SUPPORTS), vocab_size, dtype=torch.bool, device=device)
    masks[TEXT, geometry["text_token_offset"] : geometry["text_token_end"]] = True
    base = geometry["codec_base_offset"]
    masks[CODEC, base : base + geometry["codec_layer_size"]] = True
    for support in (TEXT, CODEC):
        masks[support, geometry["eos_token_id"]] = True
        masks[support, geometry["eot_token_id"]] = True
    masks[DETECT, geometry["text_token_id"]] = True
    masks[DETECT_AUDIO, geometry["text_token_id"]] = True
    masks[DETECT_AUDIO, geometry["audio_token_id"]] = True
    masks[ASSISTANT, geometry["assistant_token_id"]] = True
    masks[EOS, geometry["eos_token_id"]] = True
    return masks


def frame_supports(raw, phase, text_audio, geometry):
    """Every row's support and secondary activity from its raw primary logits.

    ``raw`` [B, V] are unmasked primary logits, ``phase`` [B] indexes PHASES and
    ``text_audio`` [B] is each row's mode. Forced phases outrank the argmax
    categories; a codec argmax still activates the secondary streams there.
    """
    tentative = raw.argmax(-1)
    detect = (tentative == geometry["text_token_id"]) | (
        tentative == geometry["audio_token_id"]
    )
    codec = text_audio & (tentative >= geometry["codec_base_offset"]) & ~detect
    support = torch.where(codec, CODEC, TEXT)
    audio = text_audio & (phase == 2)
    support = torch.where(detect, torch.where(audio, DETECT_AUDIO, DETECT), support)
    support = torch.where(text_audio & (phase == 1), ASSISTANT, support)
    support = torch.where(text_audio & (phase == 3), EOS, support)
    return support, codec


def next_phase(phase, token, text_audio, eot_token_id):
    """The phase of the next frame after the sampled primary ``token``."""
    eot = token == eot_token_id
    step = torch.where(
        (phase == 0) & eot,
        1,
        torch.where(phase == 1, 2, torch.where((phase == 2) & eot, 3, phase)),
    )
    return torch.where(text_audio, step, phase).to(phase.dtype)


def delay_interleave(codes, lengths, pad_ids):
    """Interleave [B,T,Q] codes and return masks for real, sampled positions.

    Stream q is delayed q frames. The prefix/suffix padding added here belongs
    to the environment and is not an RL action, even if its ID is nonzero.
    """
    if codes.ndim != 3 or lengths.shape != (codes.shape[0],):
        raise ValueError("Expected codes [B,T,Q] and lengths [B]")
    b, t, q = codes.shape
    if ((lengths < 0) | (lengths > t)).any():
        raise ValueError("Invalid codec lengths")
    pads = torch.as_tensor(pad_ids, device=codes.device, dtype=codes.dtype)
    if pads.shape != (q,):
        raise ValueError("One padding token per codebook is required")
    output = pads.view(1, 1, q).expand(b, t + q - 1, q).clone()
    mask = torch.zeros_like(output, dtype=torch.bool)
    valid = torch.arange(t, device=codes.device)[None, :] < lengths[:, None]
    for stream in range(q):
        output[:, stream : stream + t, stream] = torch.where(
            valid, codes[:, :, stream], pads[stream]
        )
        mask[:, stream : stream + t, stream] = valid
    return output, mask


def delay_deinterleave(codes, lengths):
    if codes.ndim != 3 or codes.shape[1] < codes.shape[2] - 1:
        raise ValueError("Invalid delayed codec shape")
    q = codes.shape[2]
    t = codes.shape[1] - q + 1
    if ((lengths < 0) | (lengths > t)).any():
        raise ValueError("Invalid original codec lengths")
    output = torch.stack(
        [codes[:, stream : stream + t, stream] for stream in range(q)], -1
    )
    valid = torch.arange(t, device=codes.device)[None, :] < lengths[:, None]
    return output, valid
