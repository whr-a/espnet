"""CPU contracts of the Bagpiper SGLang model, its rollout records and learner.

The SGLang model runs with stand-ins for SGLang's Qwen3 body; the pipeline
test shares one real tiny Qwen3 backbone and Qwen3-Omni audio encoder between
the engine (each request's history recomputed in full) and the learner's
native ParallelLLM embedding. No Engine, CUDA or download.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from torch import nn
from transformers import Qwen3Config

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.speechlm.model.speechlm.rl.bagpiper import rollout as rollout_module
from espnet2.speechlm.model.speechlm.rl.bagpiper.policy import NativeSpeechLMPolicy
from espnet2.speechlm.model.speechlm.rl.bagpiper.sglang_model import (
    ESPnetBagpiperForCausalLM,
)
from espnet2.speechlm.model.speechlm.rl.bagpiper.tokens import (
    ASSISTANT,
    CODEC,
    DETECT,
    DETECT_AUDIO,
    EOS,
    PHASES,
    SUPPORTS,
    TEXT,
    frame_supports,
    support_masks,
)

VOCAB, LAYER, BASE = 80, 8, 48
GEOMETRY = dict(
    num_stream=4,
    eos_token_id=2,
    eot_token_id=3,
    assistant_token_id=6,
    text_token_id=7,
    audio_token_id=8,
    text_token_offset=16,
    text_token_end=BASE,
    codec_base_offset=BASE,
    codec_layer_size=LAYER,
    adaptor_input_dim=24,
    audio_config=dict(
        num_mel_bins=8,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        d_model=32,
        output_dim=24,
        max_source_positions=64,
        n_window=8,
        n_window_infer=32,
        conv_chunksize=8,
        downsample_hidden_size=4,
        dropout=0.0,
        attention_dropout=0.0,
    ),
)


class Core(nn.Module):
    def __init__(self, hidden=7):
        super().__init__()
        self.model = nn.Identity()
        self._embed = lambda *args: None
        self.num_stream = GEOMETRY["num_stream"]
        self.stream_emb = nn.Embedding(self.num_stream, hidden)
        self.lm_head = nn.Linear(hidden, VOCAB, bias=False)


def batch(mode, rids, slots, lengths=None, prefixes=None):
    return SimpleNamespace(
        forward_mode=mode,
        rids=rids,
        req_pool_indices=torch.tensor(slots),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=prefixes or [0] * len(rids),
    )


def engine_model(directory, hidden, backbone, lm_head):
    """The SGLang model class around a stand-in body and head."""
    model = ESPnetBagpiperForCausalLM.__new__(ESPnetBagpiperForCausalLM)
    nn.Module.__init__(model)
    model.model, model.lm_head = backbone, lm_head
    model.setup(
        SimpleNamespace(
            espnet_bagpiper=GEOMETRY,
            hidden_size=hidden,
            vocab_size=VOCAB,
            espnet_rollout_sources=str(directory / "sources.pt"),
        )
    )
    return model.eval()


def deployed(policy, logits, stream, phase, mode, target, active, support, codec):
    """The learner's scored row of one frame and stream under a support."""
    return policy._bagpiper_deployed_row(
        logits,
        logits,
        stream,
        phase,
        mode,
        torch.tensor([[target]]),
        torch.tensor([[active]]),
        support,
        codec,
    )[0, 0]


def test_engine_and_learner_give_every_frame_the_same_support():
    policy = NativeSpeechLMPolicy(
        Core(), bagpiper_config=GEOMETRY, score_checkpointing=False
    )
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(64, VOCAB, generator=generator)
    # Every argmax class, then exact ties between classes (first index wins).
    for row, token in enumerate((16, 47, 2, 3, 0, 6, 7, 8, BASE, BASE + 7, 60, 79)):
        raw[row, token] = 9
    for row, (first, second) in enumerate(((7, BASE), (20, 60), (8, 30)), 12):
        raw[row, [first, second]] = 9
    masks, seen = support_masks(GEOMETRY, VOCAB), set()
    for mode in ("text_only", "text_audio"):
        for index, phase in enumerate(PHASES):
            supports, codecs = frame_supports(
                raw,
                torch.full((len(raw),), index, dtype=torch.int8),
                torch.full((len(raw),), mode == "text_audio"),
                GEOMETRY,
            )
            for row, support, codec in zip(raw, supports.tolist(), codecs.tolist()):
                seen.add(support)
                logits = row[None, None]
                own = policy._bagpiper_support(logits, phase, mode)
                assert own == (support, codec) and (mode == "text_audio" or not codec)
                allowed = masks[support]
                active = bool((allowed & torch.isfinite(row)).sum() > 1)
                target = int(allowed.nonzero()[0])
                frame = (phase, mode, target, active, support, codec)
                with torch.no_grad():
                    scored = deployed(policy, logits, 0, *frame)
                assert torch.equal(torch.isfinite(scored), allowed)
                for stream in range(1, GEOMETRY["num_stream"]):
                    lo = BASE + stream * LAYER
                    frame = (phase, mode, lo, codec, support, codec)
                    scored = deployed(policy, logits, stream, *frame)
                    finite = torch.isfinite(scored).nonzero().flatten().tolist()
                    assert finite == list(range(lo, lo + LAYER))
    assert seen == set(range(len(SUPPORTS)))
    assert (policy.support_mismatches, policy.support_frames) == (0, 8 * len(raw))
    # A recorded support the current logits do not give is counted.
    with torch.no_grad():
        deployed(policy, raw[:1, None], 0, "text", "text_audio", 2, True, CODEC, False)
    assert policy.support_mismatches == 1
    with pytest.raises(ValueError, match="zero likelihood"):
        deployed(policy, raw[:1, None], 0, "text", "text_only", BASE, True, TEXT, False)
    with pytest.raises(ValueError, match="stream mask"):
        deployed(policy, raw[:1, None], 0, "text", "text_only", 7, True, DETECT, False)


class ScriptedBody(nn.Module):
    """SGLang's Qwen3 body for the state machine: each request's hidden state
    makes its scripted (argmax, best allowed) primary ids the largest logits;
    every input embedding the body receives is kept."""

    def __init__(self, scripts):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, VOCAB)
        self.scripts, self.steps, self.inputs = scripts, {}, []

    def forward(self, input_ids, positions, forward_batch, input_embeds):
        hidden = torch.zeros(len(input_ids), VOCAB)
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            ends = torch.tensor(forward_batch.extend_seq_lens_cpu).cumsum(0)
            chunks = input_embeds.split(forward_batch.extend_seq_lens_cpu)
        else:
            ends, chunks = torch.arange(1, len(input_ids) + 1), input_embeds.split(1)
        for rid, end, chunk in zip(forward_batch.rids, ends, chunks):
            step = self.steps.get(rid, 0)
            self.steps[rid] = step + 1
            self.inputs.append((rid, step, chunk.clone()))
            argmax, best = self.scripts[rid.split(":", 1)[1]][step]
            hidden[end - 1, argmax] = 30
            hidden[end - 1, best] += 10
        return hidden


def write_sources(directory, batch_id, sources, temperature=0.0):
    torch.save(
        dict(batch=batch_id, rollout_id=0, temperature=temperature, sources=sources),
        directory / "sources.pt",
    )


def test_model_state_tables_phases_secondary_inputs_and_guards(tmp_path):
    torch.manual_seed(3)
    t, eot, eos, audio = 20, 3, 2, 8
    scripts = {
        # text_audio: text, eot, forced assistant (a codec argmax still samples
        # secondary streams), detect {text, audio}, codec, codec, eot, forced eos.
        "0:0": [(t, t), (eot, eot), (50, 50), (audio, audio), (49, 49), (51, 51)]
        + [(eot, eot), (t, t)],
        # Ends on its token budget with active secondary streams.
        "0:1": [(t, t), (eot, eot), (52, 52)],
        # Takes over the slot of 0:1 and must start from a clean state.
        "0:2": [(t, t), (eot, eot), (53, 53), (audio, audio), (54, 54)],
        # text_only with audio: detect, codec argmax under text support, eos.
        "1:0": [(7, 7), (BASE, 25), (30, 30), (eos, eos)],
    }
    budget = {"0:0:0": 8, "0:0:1": 3, "0:0:2": 5, "0:1:0": 4}
    body = ScriptedBody(scripts)
    model = engine_model(
        tmp_path, VOCAB, body, SimpleNamespace(weight=torch.eye(VOCAB))
    )
    secondary = [BASE + stream * LAYER + stream for stream in range(1, 4)]
    with torch.no_grad():
        model.stream_emb.weight.zero_()
        for stream, token in enumerate(secondary, 1):
            model.stream_emb.weight[stream, token] = 5
    tts = torch.tensor([[1, 0, 0, 0], [5, 0, 0, 0], [49, 57, 0, 73], [6, 0, 0, 0]])
    qa = torch.tensor([[1, 0, 0, 0], [5, 0, 0, 0], *[[0, 0, 0, 0]] * 3, [6, 0, 0, 0]])
    feats = torch.randn(1, 24, 8)
    sources = [
        dict(id="tts", prompt=tts, text_audio=True, samples=3, audio=None),
        dict(
            id="qa",
            prompt=qa,
            text_audio=False,
            samples=1,
            audio=dict(feats=feats, lengths=torch.tensor([24]), start=2, length=3),
        ),
    ]
    write_sources(tmp_path, 0, sources)
    records, slots, done = {}, {}, set()

    def run(mode, rids, inputs, lengths=None):
        forward_batch = batch(mode, rids, [slots[rid] for rid in rids], lengths)
        output = model(inputs, None, forward_batch)
        tokens = output.next_token_logits.argmax(-1).tolist()
        for rid, token, row in zip(
            rids, tokens, output.customized_info["espnet_bagpiper"]
        ):
            records.setdefault(rid, []).append((token, row))
            if token == eos or len(records[rid]) == budget[rid]:
                done.add(rid)

    slots.update({"0:0:0": 1, "0:0:1": 2, "0:1:0": 3})
    run(ForwardMode.EXTEND, list(slots), torch.cat((tts, tts, qa))[:, 0], [4, 4, 6])
    while active := [rid for rid in slots if rid not in done]:
        if "0:0:1" in done and "0:0:2" not in slots:
            slots["0:0:2"] = 2
            run(ForwardMode.EXTEND, ["0:0:2"], tts[:, 0], [4])
            continue
        last = torch.tensor([records[rid][-1][0] for rid in active])
        run(ForwardMode.DECODE, active, last)
    tokens = {rid: [token for token, _ in rows] for rid, rows in records.items()}
    supports = {rid: [row[0] for _, row in rows] for rid, rows in records.items()}
    assert tokens["0:0:0"] == [t, eot, 6, audio, 49, 51, eot, eos]
    assert supports["0:0:0"] == [
        TEXT,
        TEXT,
        ASSISTANT,
        DETECT_AUDIO,
        CODEC,
        CODEC,
        TEXT,
        EOS,
    ]
    assert [row[1] for _, row in records["0:0:0"]] == [1, 1, 0, 1, 1, 1, 1, 0]
    assert tokens["0:0:1"] == [t, eot, 6]
    assert tokens["0:0:2"] == [t, eot, 6, audio, 54]
    assert tokens["0:1:0"] == [7, 25, 30, eos]
    assert supports["0:1:0"] == [DETECT, TEXT, TEXT, TEXT]
    assert all(len(row) == 2 for _, row in records["0:1:0"])
    codec_rows = {
        rid: [step for step, (_, row) in enumerate(rows) if len(row) > 2]
        for rid, rows in records.items()
    }
    assert codec_rows == {
        "0:0:0": [2, 4, 5],
        "0:0:1": [2],
        "0:0:2": [2, 4],
        "0:1:0": [],
    }
    for rid, steps in codec_rows.items():
        for step in steps:
            row = records[rid][step][1]
            assert row[2:5] == secondary and all(lp <= 0 for lp in row[5:])
    embed = body.embed_tokens.weight
    inputs = {(rid, step): chunk for rid, step, chunk in body.inputs}
    # Prompt: native stream sum (zero ids cleared) and the adapted audio span.
    expected = embed[tts[:, 0]].clone()
    expected[2] = embed[49] + embed[57] + embed[73]
    torch.testing.assert_close(inputs["0:0:0", 0], expected)
    audio_rows = model.audio_adaptor(
        model.audio_tower(feats[0].T, feature_lens=torch.tensor([24])).last_hidden_state
    )
    torch.testing.assert_close(inputs["0:1:0", 0][2:5], audio_rows)
    torch.testing.assert_close(inputs["0:1:0", 0][[0, 1, 5]], embed[[1, 5, 6]])
    # Decode inputs: the last primary plus the FP32 sum of its secondaries,
    # including the request that reused a slot left with pending secondaries.
    for rid, rows in records.items():
        for step in range(1, len(rows)):
            token, row = rows[step - 1]
            value = embed[token] + sum(embed[i] for i in row[2:5]) * (len(row) > 2)
            torch.testing.assert_close(inputs[rid, step][0], value)
    for rids, lengths, prefixes, inputs_, message in (
        (["0:0:1"], [4], None, tts[:, 0], "prefilled once"),  # retracted
        (["0:0:5"], [3], [1], tts[1:, 0], "prefilled once"),  # prefix reuse
        (["0:0:6"], [3], None, tts[:3, 0], "prefilled once"),  # chunked
        (["0:0:7", "1:0:0"], [4, 4], None, tts[:, 0].repeat(2), "mixes"),
        (["1:0:0"], [4], None, tts[:, 0], "holds batch 0"),
    ):
        forward_batch = batch(
            ForwardMode.EXTEND, rids, [3] * len(rids), lengths, prefixes
        )
        with pytest.raises(RuntimeError, match=message):
            model(inputs_, None, forward_batch)
    write_sources(tmp_path, 1, sources)
    with pytest.raises(RuntimeError, match="input ids differ"):
        model(qa[:4, 0], None, batch(ForwardMode.EXTEND, ["1:0:0"], [3], [4]))


def test_load_weights_is_strict_and_complete():
    model = ESPnetBagpiperForCausalLM.__new__(ESPnetBagpiperForCausalLM)
    nn.Module.__init__(model)
    calls = []

    def fused(param, tensor, shard):
        calls.append(shard)
        param.data[{"q": 0, "k": 1, "v": 2, 0: 0, 1: 1}[shard]] = tensor

    model.model = nn.Module()
    for name, rows in (("qkv_proj", 3), ("gate_up_proj", 2), ("o_proj", 1)):
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(rows, 2))
        if rows > 1:
            module.weight.weight_loader = fused
        model.model.add_module(name, module)
    model.stream_emb = nn.Embedding(2, 2)
    # The audio encoder keeps its own unfused attention projections.
    model.audio_tower = nn.Module()
    model.audio_tower.q_proj = nn.Linear(2, 2, bias=False)
    weights = [
        ("model.q_proj.weight", torch.ones(2)),
        ("model.k_proj.weight", torch.ones(2)),
        ("model.v_proj.weight", torch.ones(2)),
        ("model.gate_proj.weight", torch.ones(2)),
        ("model.up_proj.weight", torch.ones(2)),
        ("model.o_proj.weight", torch.ones(1, 2)),
        ("stream_emb.weight", torch.ones(2, 2)),
        (
            "multimodal_io_dict.continuous_audio.model.audio_tower.q_proj.weight",
            torch.ones(2, 2),
        ),
        ("multimodal_io_dict.discrete_audio.codec_model.decoder.weight", torch.ones(3)),
    ]
    for _ in range(2):  # every update is a complete policy
        model.batch = "0"
        assert len(model.load_weights(iter(weights))) == 5
        assert model.batch is None and model.audio_tower.q_proj.weight.eq(1).all()
    assert calls == ["q", "k", "v", 0, 1] * 2
    for broken, message in (
        (weights[1:], "incomplete"),
        (weights + weights[:1], "repeated"),
        (weights + [("model.extra.weight", torch.ones(2))], "Unexpected"),
    ):
        with pytest.raises(ValueError, match=message):
            model.load_weights(iter(broken))


class HistoryBody(nn.Module):
    """SGLang's Qwen3 body for CPU tests: every request's whole history is
    recomputed by the shared HF Qwen3 model instead of paged attention."""

    def __init__(self, qwen):
        super().__init__()
        self.qwen, self.embed_tokens, self.history = qwen, qwen.embed_tokens, {}

    def forward(self, input_ids, positions, forward_batch, input_embeds):
        slots = forward_batch.req_pool_indices.tolist()
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            chunks = input_embeds.split(forward_batch.extend_seq_lens_cpu)
            self.history.update(zip(slots, chunks))
            return torch.cat([self._run(slot) for slot in slots])
        for slot, row in zip(slots, input_embeds.split(1)):
            self.history[slot] = torch.cat((self.history[slot], row))
        return torch.cat([self._run(slot)[-1:] for slot in slots])

    def _run(self, slot):
        rows = self.history[slot]
        return self.qwen(
            inputs_embeds=rows[None],
            position_ids=torch.arange(len(rows))[None],
            use_cache=False,
        ).last_hidden_state[0]


class AudioIO(nn.Module):
    """The native continuous-audio handler's tensor path."""

    is_discrete = False

    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def encode_batch(self, feats, lengths):
        return [self.tower(feats[0].T, feature_lens=lengths).last_hidden_state]


@pytest.fixture
def native(tmp_path):
    """A tiny native ParallelLLM and the SGLang engine wired to the same weights."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from espnet2.speechlm.model.speechlm.lm.parallel import build_parallel_hf_class

    torch.manual_seed(11)
    text = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        architectures=["Qwen3ForCausalLM"],
    )
    (tmp_path / "backbone").mkdir()
    text.save_pretrained(tmp_path / "backbone")
    text._attn_implementation = "sdpa"
    core = build_parallel_hf_class(str(tmp_path / "backbone"))(text).eval()
    audio = Qwen3OmniMoeAudioEncoderConfig(**GEOMETRY["audio_config"])
    audio._attn_implementation = "sdpa"
    core.num_stream = GEOMETRY["num_stream"]
    core.stream_emb = nn.Embedding(core.num_stream, 32)
    core.multimodal_io_dict = nn.ModuleDict(
        {"continuous_audio": AudioIO(Qwen3OmniMoeAudioEncoder(audio))}
    )
    core.adaptor = nn.ModuleDict({"continuous_audio": nn.Linear(24, 32)})
    with torch.no_grad():
        # Peakier logits and a frequent eot: candidates pass every phase.
        core.lm_head.weight.mul_(4)
        core.lm_head.weight[3] += 34 * core.lm_head.weight[16:48].mean(0)
    core.eval()
    export = tmp_path / "hf"
    export.mkdir()
    (export / "config.json").write_text(json.dumps({"espnet_bagpiper": GEOMETRY}))
    return core, export


class FakeEngine:
    """sglang.Engine for CPU tests: schedules the Bagpiper model like SGLang
    (new requests prefilled together, then decoded together; a finished
    request's slot is reused) and samples the primary from its logits."""

    core = None
    running = 3

    def __init__(self, **settings):
        self.settings = settings
        overrides = json.loads(settings["json_model_override_args"])
        directory = Path(overrides["espnet_rollout_sources"]).parent
        self.model = engine_model(
            directory, 32, HistoryBody(self.core.model), self.core.lm_head
        )
        self.model.stream_emb = self.core.stream_emb
        self.model.audio_tower = self.core.multimodal_io_dict["continuous_audio"].tower
        self.model.audio_adaptor = self.core.adaptor["continuous_audio"]

    def get_server_info(self):
        return {
            "max_total_num_tokens": 4096,
            "page_size": 1,
            "internal_states": [{"effective_max_running_requests_per_dp": 3}],
        }

    def generate(self, input_ids, sampling_params, rid, **kwargs):
        waiting = list(zip(rid, input_ids, sampling_params))
        free, running, results = [1, 2, 3], {}, {}
        while waiting or running:
            if waiting and len(running) < self.running:
                new = waiting[: self.running - len(running)]
                waiting = waiting[len(new) :]
                for name, ids, params in new:
                    running[name] = dict(slot=free.pop(0), params=params, ids=ids)
                    results[name] = dict(output_ids=[], logprobs=[], trace=[])
                names = [name for name, _, _ in new]
                inputs = torch.tensor([i for _, ids, _ in new for i in ids])
                forward_batch = batch(
                    ForwardMode.EXTEND,
                    names,
                    [running[n]["slot"] for n in names],
                    [len(ids) for _, ids, _ in new],
                )
            else:
                names = list(running)
                inputs = torch.tensor([results[n]["output_ids"][-1] for n in names])
                forward_batch = batch(
                    ForwardMode.DECODE, names, [running[n]["slot"] for n in names]
                )
            output = self.model(inputs, None, forward_batch)
            logits = output.next_token_logits
            for row, name in enumerate(names):
                params = running[name]["params"]
                if params["temperature"] == 0:
                    token = int(logits[row].argmax())
                else:
                    token = int(torch.multinomial(logits[row].softmax(-1), 1))
                result = results[name]
                result["output_ids"].append(token)
                result["logprobs"].append(
                    (float(logits[row].log_softmax(-1)[token]), token, None)
                )
                result["trace"].append(output.customized_info["espnet_bagpiper"][row])
                stop = token in params["stop_token_ids"]
                if stop or len(result["output_ids"]) == params["max_new_tokens"]:
                    result["finish"] = "stop" if stop else "length"
                    free.append(running.pop(name)["slot"])
        return [
            dict(
                output_ids=results[name]["output_ids"],
                meta_info=dict(
                    id=name,
                    finish_reason=dict(type=results[name]["finish"]),
                    num_retractions=0,
                    output_token_logprobs=results[name]["logprobs"],
                    espnet_bagpiper=results[name]["trace"],
                ),
            )
            for name in rid
        ]


def rollout_sources():
    """A QA prompt with an audio span and a TTS prompt with a codec frame."""
    feats = torch.randn(1, 24, 8, generator=torch.Generator().manual_seed(5))
    native = dict(
        continuous_audio_feats=feats.tolist(),
        continuous_audio_lengths=[24],
        continuous_audio_indices=[[0, 2, 3]],
    )
    frames = lambda *ids: [[i, 0, 0, 0] for i in ids]  # noqa: E731
    return [
        dict(
            id="qa",
            task="qa",
            prompt_token_ids=frames(1, 5, 0, 0, 0, 20, 2, 6),
            native_model_inputs=native,
        ),
        dict(
            id="tts",
            task="tts_asr_wer",
            prompt_token_ids=[*frames(1, 5, 7), [49, 57, 0, 73], *frames(2, 6)],
        ),
    ]


def build_engine(native, tmp_path, monkeypatch, **overrides):
    import sglang
    import transformers

    core, export = native
    FakeEngine.core = core
    for name in ("SGLANG_RETURN_ORIGINAL_LOGPROB", "TRITON_F32_DEFAULT"):
        monkeypatch.delenv(name, raising=False)  # restored after the test
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.setattr(sglang, "Engine", FakeEngine)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            decode=lambda ids, **_: " ".join(map(str, ids))
        ),
    )
    config = ESPnetRLConfig(
        family="bagpiper",
        adapter_factory="a:b",
        exporter="c:d",
        sglang_engine="e:f",
        group_size=4,
        max_tokens=14,
        **overrides,
    )
    directory = tmp_path / "replica-0"
    directory.mkdir()
    return rollout_module.BagpiperEngine(config, export, directory=directory)


def test_records_are_scored_by_the_learner_under_their_supports(
    native, tmp_path, monkeypatch
):
    core, _ = native
    engine = build_engine(native, tmp_path, monkeypatch)
    settings = engine.engine.settings
    assert settings["dtype"] == "float32" and settings["attention_backend"] == "triton"
    assert settings["chunked_prefill_size"] == -1 and settings["skip_tokenizer_init"]
    assert settings["disable_radix_cache"] and settings["disable_overlap_schedule"]
    sources = rollout_sources()
    torch.manual_seed(0)
    generations = engine.generate(sources, 7)
    assert [g["source_id"] for g in generations] == ["qa"] * 4 + ["tts"] * 4
    assert (tmp_path / "replica-0" / "sources.pt").is_file()
    for generation in generations:
        frames = len(generation["token_ids"])
        assert len(generation["support"]) == frames == len(generation["log_probs"])
        assert generation["finish_reason"] in ("stop", "length")
        for support, mask, ids in zip(
            generation["support"], generation["stream_mask"], generation["token_ids"]
        ):
            assert mask[1:] == [mask[1]] * 3 and (ids[1] != 0) == mask[1]
            assert mask[0] == (support in (TEXT, CODEC, DETECT_AUDIO))
        if generation["source_id"] == "qa":
            assert not any(any(mask[1:]) for mask in generation["stream_mask"])
            assert CODEC not in generation["support"]
    supports = {s for g in generations for s in g["support"]}
    assert {TEXT, CODEC, ASSISTANT, EOS} <= supports
    batch_ = collate_rollouts(sources, generations, 4)
    policy = NativeSpeechLMPolicy(
        core, token_chunk_size=4, bagpiper_config=GEOMETRY, score_checkpointing=False
    )
    with torch.no_grad():
        scores = policy(batch_)
    mask = batch_.batch["response_mask"].bool()
    error = (scores.log_probs - batch_.batch["rollout_log_probs"]).abs()[mask].max()
    assert error < 1e-4
    assert policy.support_mismatches == 0
    assert policy.support_frames == int(batch_.batch["response_lengths"].sum())
    recorded = torch.zeros_like(scores.stream_log_probs)
    for row, generation in enumerate(generations):
        values = torch.tensor(generation["stream_log_probs"])
        recorded[row, : len(values)] = values
    torch.testing.assert_close(scores.stream_log_probs, recorded, atol=1e-4, rtol=0)
    # The update pass scores under the same recorded supports with gradients.
    policy.score_checkpointing = True
    policy(batch_).log_probs.sum().backward()
    assert core.lm_head.weight.grad is not None
    # A policy whose own supports differ still scores the recorded ones; the
    # frames it disagrees on are counted and gated by their rate.
    with torch.no_grad():
        core.lm_head.weight.neg_()
        policy.support_mismatch_rate, before = 1.0, policy.support_mismatches
        policy(batch_)
        assert policy.support_mismatches > before
        policy.support_mismatch_rate = 0.0
        with pytest.raises(ValueError, match="another support"):
            policy(batch_)


def test_sampling_override_and_retraction_free_sizing(native, tmp_path, monkeypatch):
    engine = build_engine(native, tmp_path, monkeypatch, dtype="bfloat16")
    assert engine.engine.settings["attention_backend"] == "fa3"
    override = {"temperature": 0.0, "max_new_tokens": 6, "samples": 1}
    greedy = [engine.generate(rollout_sources(), 1, override) for _ in range(2)]
    assert greedy[0] == greedy[1]
    assert [g["source_id"] for g in greedy[0]] == ["qa", "tts"]
    assert all(len(g["token_ids"]) <= 6 for g in greedy[0])
    assert {
        contract["temperature"]
        for g in greedy[0]
        for frame in g["sampling_contracts"]
        for contract in frame
    } == {0.0}
    with pytest.raises(ValueError, match="overrides temperature"):
        engine.generate(rollout_sources(), 2, {"top_p": 0.5})
    with pytest.raises(ValueError, match="never retracts"):
        engine.generate(rollout_sources(), 3, {"max_new_tokens": 2000})
