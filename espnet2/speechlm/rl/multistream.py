# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""OpusLM's nine-stream ARDelay frame state, shared by engine and learner.

The released layout puts the SSL semantic token and every control symbol in
stream 0 and DAC layer q-1 in stream q. The renderer delays stream q by q
frames, so frame t of a codec block holds audio frame t-q in stream q: stream
q carries a real action from frame q until q frames after stream 0 closed the
block, and pad (id 0) everywhere else. Pad is environment, never an action.

Both halves of the loop must agree on this, so the supports and the ramp live
here rather than being written twice.
"""


def codec_supports(metadata, temperature=1.0, top_k=0):
    """Per-stream sampling contract of a codec frame, in learner order.

    Stream 0 may emit any SSL token or the marker that closes the block;
    stream q in 1..nq-1 may emit any id of its own DAC layer. The learner reads
    exactly these specs out of ``sampling_contracts`` (policy.py:_scores), so
    the engine must mask the identical support when it samples.
    """
    nq = int(metadata["nq"])
    size = int(metadata["codec_per_stream_size"])
    base = int(metadata["codec_token_start"])
    if base + (nq - 1) * size != int(metadata["codec_token_end"]):
        raise ValueError("Codec stream slices must tile codec_token_start..end")
    supports = [
        {
            "allowed_ranges": [
                [int(metadata["ssl_token_start"]), int(metadata["ssl_token_end"])]
            ],
            "allowed_token_ids": [int(metadata["codec_ssl_start_end_token_id"])],
            "temperature": temperature,
            "top_k": top_k,
        }
    ]
    for stream in range(1, nq):
        low = base + (stream - 1) * size
        supports.append(
            {
                "allowed_ranges": [[low, low + size]],
                "temperature": temperature,
                "top_k": top_k,
            }
        )
    return supports


def active_streams(nq, step, closed_at):
    """Which streams carry a real action at frame ``step`` of a codec block.

    ``closed_at`` is the frame at which stream 0 emitted the closing marker, or
    None while the block is still open.
    """
    if closed_at is None:
        return [True] + [step >= stream for stream in range(1, nq)]
    return [False] + [
        step >= stream and step - closed_at < stream for stream in range(1, nq)
    ]


def block_finished(nq, step, closed_at):
    """True once stream nq-1 has flushed its last delayed code."""
    return closed_at is not None and step - closed_at >= nq - 1


def deinterleave_codec(token_ids, metadata):
    """The [nq-1, T] DAC codes of a sampled block, or why it has none.

    Undoes the fixed delay: layer q-1 of audio frame j sits in stream q of
    frame j+q, so audio frame j is complete only once every stream has reached
    it. A closed block of T frames and a block truncated at T both yield
    T-(nq-1) complete frames. Returns ``(codes, "ok")``, ``(None, "no_audio")``
    when the block is shorter than the delay ramp, or ``(None,
    "invalid_codec")`` when an aligned position holds an id outside its layer.
    """
    import torch

    nq = int(metadata["nq"])
    size = int(metadata["codec_per_stream_size"])
    base = int(metadata["codec_token_start"])
    close = int(metadata["codec_ssl_start_end_token_id"])
    matrix = torch.as_tensor(token_ids, dtype=torch.long)
    if matrix.ndim != 2 or matrix.shape[1] != nq:
        raise ValueError("A codec rollout must keep all nine streams")
    # Audio frame j is complete only once the slowest stream reached it, which
    # is frame j+nq-1. A block closed at frame c declares c audio frames, but a
    # block cut off mid-flush has only as many as arrived, so take the smaller
    # of the two rather than discarding a short but real block.
    closed = (matrix[:, 0] == close).nonzero()
    reached = len(matrix) - (nq - 1)
    length = min(int(closed[0]), reached) if len(closed) else reached
    if length < 1:
        return None, "no_audio"
    aligned = []
    for stream in range(1, nq):
        tokens = matrix[stream : stream + length, stream]
        low = base + (stream - 1) * size
        if ((tokens < low) | (tokens >= low + size)).any():
            return None, "invalid_codec"
        aligned.append(tokens - low)
    return torch.stack(aligned), "ok"
