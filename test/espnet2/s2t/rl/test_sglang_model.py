"""Batched native decoding in the OWSM SGLang model.

Every running request goes through one padded native decoder call. These
checks compare it with the per-request algorithm it replaced, using a small
real ESPnet TransformerDecoder and memories of different lengths.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("sglang")

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder  # noqa: E402
from espnet2.s2t.rl import sglang_model  # noqa: E402

VOCAB, EOS = 31, 30


class Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind == "extend"

    def is_idle(self):
        return False


def owsm_model(tmp_path):
    torch.manual_seed(0)
    decoder = TransformerDecoder(
        VOCAB,
        16,
        attention_heads=2,
        linear_units=32,
        num_blocks=2,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        use_flash_attn=False,
    ).eval()
    model = sglang_model.ESPnetOWSMForCausalLM.__new__(
        sglang_model.ESPnetOWSMForCausalLM
    )
    nn.Module.__init__(model)
    model.policy = nn.Module()
    model.policy.model = nn.Module()
    model.policy.model.decoder = decoder
    model.policy.model.eos = EOS
    model.config = SimpleNamespace(vocab_size=VOCAB, torch_dtype="float32")
    model.target_device = torch.device("cpu")
    model.memory, model.request_prefixes = {}, {}
    model.rollout_id = model.sources = model.source_sha256 = None
    model.source_path = tmp_path / "sources.json"
    write_sources(model, 0)
    # Encoder memories of different lengths, as different utterances produce.
    memories = {
        index: (torch.randn(1, length, 16), torch.tensor([length]))
        for index, length in enumerate((5, 9, 3, 12))
    }
    model._memory = lambda source, device: memories[source]
    return model, decoder, memories


def write_sources(model, rollout_id):
    sources = [{"id": f"utt{index}"} for index in range(4)]
    model.source_path.write_text(
        json.dumps({"rollout_id": rollout_id, "sources": sources})
    )


def prefill(model, rids, prefixes, return_logprob=True):
    return model.forward(
        torch.tensor(sum(prefixes, [])),
        None,
        SimpleNamespace(
            req_pool_indices=torch.arange(len(rids)),
            rids=rids,
            forward_mode=Mode("extend"),
            extend_seq_lens_cpu=[len(prefix) for prefix in prefixes],
            extend_prefix_lens_cpu=[0] * len(prefixes),
            return_logprob=return_logprob,
        ),
    )


def per_request(decoder, memories, rows):
    """The previous algorithm: one native decoder call per running request."""
    logits, prompt = [], []
    for source, prefix in rows:
        memory, lengths = memories[source]
        out, _ = decoder(
            memory, lengths, torch.tensor([prefix]), torch.tensor([len(prefix)])
        )
        logits.append(out[0, -1].float())
        prompt.append(torch.log_softmax(out[0, -2].float(), -1)[prefix[-1]])
    return torch.stack(logits), torch.stack(prompt)


def generate(model, sources, steps, monkeypatch, budget):
    monkeypatch.setattr(sglang_model, "LOGITS_BUDGET_BYTES", budget)
    model.request_prefixes = {}
    pool = list(range(len(sources)))
    # Request IDs name the source; the input IDs are only the real prefix.
    rids = [f"0:{source}:{sample}" for sample, source in enumerate(sources)]
    first = prefill(model, rids, [[3, 4, 5] for _ in sources])
    outputs = [first.next_token_logits]
    generator = torch.Generator().manual_seed(1)
    for step in range(steps):
        # Requests finish at different times, so later steps run fewer rows
        # whose prefixes differ in length from the rows that remain.
        active = [p for p in pool if not (step >= 2 and p % 2)]
        tokens = torch.randint(0, EOS, (len(active),), generator=generator)
        decoded = model.forward(
            tokens,
            None,
            SimpleNamespace(
                req_pool_indices=torch.tensor(active),
                rids=[rids[p] for p in active],
                forward_mode=Mode("decode"),
                return_logprob=False,
            ),
        )
        outputs.append((active, tokens.tolist(), decoded.next_token_logits))
    return first, outputs


def replay(decoder, memories, sources, outputs):
    rows = {p: (source, [3, 4, 5]) for p, source in enumerate(sources)}
    expected = [per_request(decoder, memories, list(rows.values()))]
    for active, tokens, _ in outputs[1:]:
        for p, token in zip(active, tokens):
            rows[p] = (rows[p][0], rows[p][1] + [token])
        expected.append(per_request(decoder, memories, [rows[p] for p in active]))
    return expected


def test_one_row_per_call_reproduces_the_per_request_algorithm(monkeypatch, tmp_path):
    model, decoder, memories = owsm_model(tmp_path)
    sources = [0, 1, 2, 3, 1, 0]
    with torch.no_grad():
        prefill, outputs = generate(model, sources, 4, monkeypatch, budget=1)
        expected = replay(decoder, memories, sources, outputs)
    assert torch.equal(prefill.next_token_logits, expected[0][0])
    assert torch.equal(prefill.input_token_logprobs, expected[0][1])
    for (_, _, logits), (want, _) in zip(outputs[1:], expected[1:]):
        assert torch.equal(logits, want)


def test_batched_call_matches_up_to_fp32_reduction_order(monkeypatch, tmp_path):
    model, decoder, memories = owsm_model(tmp_path)
    sources = [0, 1, 2, 3, 1, 0]
    with torch.no_grad():
        prefill, outputs = generate(model, sources, 4, monkeypatch, budget=2**30)
        expected = replay(decoder, memories, sources, outputs)
    torch.testing.assert_close(prefill.next_token_logits, expected[0][0])
    torch.testing.assert_close(prefill.input_token_logprobs, expected[0][1])
    for (_, _, logits), (want, _) in zip(outputs[1:], expected[1:]):
        torch.testing.assert_close(
            torch.log_softmax(logits, -1), torch.log_softmax(want, -1)
        )


def test_decode_without_prefill_is_refused(tmp_path):
    model, _, _ = owsm_model(tmp_path)
    with pytest.raises(RuntimeError, match="missing native prefill"):
        model.forward(
            torch.tensor([1]),
            None,
            SimpleNamespace(
                req_pool_indices=torch.tensor([7]),
                rids=["0:0:0"],
                forward_mode=Mode("decode"),
                return_logprob=False,
            ),
        )


def test_each_rollout_reads_its_own_sources(tmp_path):
    """A new rollout ID reloads the sources file and forgets old memories."""
    model, _, _ = owsm_model(tmp_path)
    prefill(model, ["0:1:0", "0:2:1"], [[3, 4, 5], [3, 4]])
    assert model.rollout_id == 0 and model.request_prefixes[1] == (2, [3, 4])
    model.memory["stale"] = "encoded with rollout 0"
    write_sources(model, 1)
    prefill(model, ["1:3:0"], [[3, 4, 6]])
    assert model.rollout_id == 1 and model.sources[3] == {"id": "utt3"}
    assert "stale" not in model.memory and set(model.request_prefixes) == {0}
    # The file must belong to the rollout the requests name.
    with pytest.raises(ValueError, match="different rollout"):
        prefill(model, ["2:0:0"], [[3, 4, 5]])
    with pytest.raises(ValueError, match="mixes OWSM rollouts"):
        prefill(model, ["1:0:0", "3:0:1"], [[3, 4, 5], [3, 4, 5]])


def test_repeated_full_updates_load_strictly_and_reset_request_state(tmp_path):
    model, decoder, _ = owsm_model(tmp_path)
    prefill(model, ["0:0:0"], [[3, 4, 5]])
    model.memory[0] = "encoded by the previous policy"
    state = {
        name: tensor + 1 for name, tensor in model.policy.model.state_dict().items()
    }
    assert model.load_weights(state.items()) == set(state)
    for name, tensor in model.policy.model.state_dict().items():
        assert torch.equal(tensor, state[name])
    # The trained encoder changed, so nothing computed before may survive.
    assert model.memory == {} and model.request_prefixes == {}
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_weights(missing.items())
