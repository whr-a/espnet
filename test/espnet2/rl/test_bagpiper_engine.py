"""CPU contracts of the Bagpiper SGLang model, its rollout records and learner.

The SGLang model runs with stand-ins for SGLang's Qwen3 body; the pipeline
test shares one real tiny Qwen3 backbone and Qwen3-Omni audio encoder between
the engine (each request's history recomputed in full) and the learner's
native ParallelLLM embedding. No Engine, CUDA or download.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sglang.srt.layers import sampler as sglang_sampler
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import OutputLogprobProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.utils.common import get_bool_env_var
from torch import nn
from transformers import Qwen3Config

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import collate_rollouts
from espnet2.rl.trainer import NativeGRPOTrainer
from espnet2.rl.types import support_counts, support_mismatch_rate
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


def reference_support(raw, phase, mode):
    """The rule of a frame's support and secondary activity, one row at a time."""
    tentative = int(raw.argmax())
    detect = tentative in (GEOMETRY["text_token_id"], GEOMETRY["audio_token_id"])
    codec = mode == "text_audio" and tentative >= BASE and not detect
    if mode == "text_audio" and phase in ("transition", "audio_stop"):
        return (ASSISTANT if phase == "transition" else EOS), codec
    if detect:
        audio = mode == "text_audio" and phase == "audio"
        return (DETECT_AUDIO if audio else DETECT), codec
    return (CODEC if codec else TEXT), codec


def reference_allowed(support):
    """The primary ids a support allows, one support at a time."""
    allowed = torch.zeros(VOCAB, dtype=torch.bool)
    if support in (TEXT, CODEC):
        lo, hi = (16, BASE) if support == TEXT else (BASE, BASE + LAYER)
        allowed[lo:hi] = allowed[2] = allowed[3] = True
    else:
        ids = {DETECT: [7], DETECT_AUDIO: [7, 8], ASSISTANT: [6], EOS: [2]}
        allowed[ids[support]] = True
    return allowed


def test_engine_and_learner_give_every_frame_the_same_support():
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(64, VOCAB, generator=generator)
    # Every argmax class, then exact ties between classes (first index wins).
    for row, token in enumerate((16, 47, 2, 3, 0, 6, 7, 8, BASE, BASE + 7, 60, 79)):
        raw[row, token] = 9
    for row, (first, second) in enumerate(((7, BASE), (20, 60), (8, 30)), 12):
        raw[row, [first, second]] = 9
    # The learner's head on hidden states that are these raw logits.
    core = Core(VOCAB)
    with torch.no_grad():
        core.lm_head.weight.copy_(torch.eye(VOCAB))
        core.stream_emb.weight.zero_()
    policy = NativeSpeechLMPolicy(core, GEOMETRY, token_chunk_size=5)
    masks, seen = support_masks(GEOMETRY, VOCAB), set()
    layers = BASE + LAYER * torch.arange(1, GEOMETRY["num_stream"])
    for mode in ("text_only", "text_audio"):
        for index, phase in enumerate(PHASES):
            phases = torch.full((len(raw),), index)
            text_audio = torch.full((len(raw),), mode == "text_audio")
            supports, codecs = frame_supports(raw, phases, text_audio, GEOMETRY)
            for row, support, codec in zip(raw, supports.tolist(), codecs.tolist()):
                assert reference_support(row, phase, mode) == (support, codec)
                assert torch.equal(masks[support], reference_allowed(support))
                assert mode == "text_audio" or not codec
                seen.add(support)
            # The first id each support and codec layer allows.
            labels = torch.cat(
                (masks[supports].int().argmax(-1, keepdim=True), layers.expand(64, -1)),
                -1,
            )
            active = torch.cat(
                ((masks[supports].sum(-1) > 1)[:, None], codecs[:, None].expand(-1, 3)),
                -1,
            )
            with torch.no_grad():
                logp, entropy, streams = policy._score_frames(
                    raw, labels, active, supports, phases, text_audio
                )
            primary = raw.masked_fill(~masks[supports], -torch.inf).log_softmax(-1)
            expected = torch.zeros(64, GEOMETRY["num_stream"])
            expected[:, 0] = primary.gather(-1, labels[:, :1])[:, 0] * active[:, 0]
            for stream, lo in enumerate(layers.tolist(), 1):
                layer = raw[:, lo : lo + LAYER].log_softmax(-1)[:, 0]
                expected[:, stream] = layer * codecs
            torch.testing.assert_close(streams, expected)
            torch.testing.assert_close(logp, expected.sum(-1))
            assert bool((entropy[~active.any(-1)] == 0).all())
    assert seen == set(range(len(SUPPORTS)))
    assert (policy.support_mismatches, policy.support_frames) == (0, 8 * len(raw))
    # A recorded support the current logits do not give is counted.
    phase, text_audio = torch.zeros(1, dtype=torch.long), torch.ones(
        1, dtype=torch.bool
    )
    frame = raw[8:9]  # a codec argmax
    with torch.no_grad():
        policy._score_frames(
            frame,
            torch.tensor([[2, 0, 0, 0]]),
            torch.tensor([[True, False, False, False]]),
            torch.tensor([TEXT]),
            phase,
            text_audio,
        )
    assert policy.support_mismatches == 1
    for labels, active, message in (
        ([BASE, 0, 0, 0], [True, False, False, False], "zero likelihood"),
        ([20, BASE, 0, 0], [True, True, True, True], "zero likelihood"),
        ([20, 0, 0, 0], [False, False, False, False], "stream mask"),
        ([20, *layers.tolist()], [True, True, False, True], "stream mask"),
    ):
        with pytest.raises(ValueError, match=message):
            policy._score_frames(
                frame,
                torch.tensor([labels]),
                torch.tensor([active]),
                torch.tensor([TEXT]),
                phase,
                text_audio,
            )


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


def test_load_weights_is_strict_and_complete(tmp_path):
    model = ESPnetBagpiperForCausalLM.__new__(ESPnetBagpiperForCausalLM)
    nn.Module.__init__(model)
    model.sources_path = tmp_path / "sources.pt"
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


def sglang_sample(logits, params):
    """SGLang's own sampler on one batch: the sampled ids and the log
    probabilities it returns for them."""
    requests = [SamplingParams(**p) for p in params]
    sampler = sglang_sampler.Sampler.__new__(sglang_sampler.Sampler)
    nn.Module.__init__(sampler)
    sampler.rl_on_policy_target = None
    sampler.enable_deterministic = sampler.use_log_softmax_logprob = False
    sampler.use_ascend_backend = False
    sampler.output_logprob_processor = OutputLogprobProcessor()
    info = SimpleNamespace(
        temperatures=torch.tensor([[r.temperature] for r in requests]),
        is_all_greedy=all(r.top_k <= 1 for r in requests),
        need_top_p_sampling=any(r.top_p != 1 for r in requests),
        need_top_k_sampling=any(r.top_k != TOP_K_ALL for r in requests),
        need_min_p_sampling=any(r.min_p > 0 for r in requests),
        has_custom_logit_processor=False,
        return_sampling_masks=None,
        sampling_seed=None,
        grammars=None,
    )
    output = LogitsProcessorOutput(next_token_logits=logits)
    ids = sampler(output, info, True, [0] * len(params), [None] * len(params), None)
    return ids.tolist(), output.next_token_logprobs.tolist()


class FakeEngine:
    """sglang.Engine for CPU tests: schedules the Bagpiper model like SGLang
    (new requests prefilled together, then decoded together; a finished
    request's slot is reused) and samples the primary with SGLang's sampler."""

    core = None
    running = 3
    # Three requests of the default 2,048-token context.
    pool = 3 * 2048

    def __init__(self, **settings):
        self.settings = settings
        # The scheduler reads this once, when it imports its sampler.
        sglang_sampler.SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var(
            "SGLANG_RETURN_ORIGINAL_LOGPROB"
        )
        overrides = json.loads(settings["json_model_override_args"])
        directory = Path(overrides["espnet_rollout_sources"]).parent
        self.model = engine_model(
            directory, 32, HistoryBody(self.core.model), self.core.lm_head
        )
        self.model.stream_emb = self.core.stream_emb
        self.model.audio_tower = self.core.multimodal_io_dict["continuous_audio"].tower
        self.model.audio_adaptor = self.core.adaptor["continuous_audio"]

    def shutdown(self):
        pass

    def get_server_info(self):
        return {
            "max_total_num_tokens": self.pool,
            "page_size": 1,
            "internal_states": [
                {"effective_max_running_requests_per_dp": self.running}
            ],
        }

    def generate(self, input_ids, sampling_params, rid, **kwargs):
        self.params = sampling_params
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
            tokens, logprobs = sglang_sample(
                output.next_token_logits, [running[n]["params"] for n in names]
            )
            for row, name in enumerate(names):
                params, token = running[name]["params"], tokens[row]
                result = results[name]
                result["output_ids"].append(token)
                result["logprobs"].append((logprobs[row], token, None))
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


def fake_sglang(native, monkeypatch):
    """Serve BagpiperEngine from FakeEngine on the tiny native model."""
    import sglang
    import transformers

    FakeEngine.core = native[0]
    for name in ("SGLANG_RETURN_ORIGINAL_LOGPROB", "TRITON_F32_DEFAULT"):
        monkeypatch.delenv(name, raising=False)  # restored after the test
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.setattr(sglang_sampler, "SGLANG_RETURN_ORIGINAL_LOGPROB", False)
    monkeypatch.setattr(sglang, "Engine", FakeEngine)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            decode=lambda ids, **_: " ".join(map(str, ids))
        ),
    )


def build_engine(native, tmp_path, monkeypatch, **overrides):
    fake_sglang(native, monkeypatch)
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
    return rollout_module.BagpiperEngine(config, native[1], directory=directory)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_records_are_scored_by_the_learner_under_their_supports(
    native, tmp_path, monkeypatch, temperature
):
    core, _ = native
    engine = build_engine(native, tmp_path, monkeypatch, temperature=temperature)
    settings = engine.engine.settings
    assert settings["dtype"] == "float32" and settings["attention_backend"] == "triton"
    assert settings["chunked_prefill_size"] == -1 and settings["skip_tokenizer_init"]
    assert settings["disable_radix_cache"] and settings["disable_overlap_schedule"]
    sources = rollout_sources()
    torch.manual_seed(2)  # candidates that pass every support at both temperatures
    generations = engine.generate(sources, 7)
    assert [g["source_id"] for g in generations] == ["qa"] * 4 + ["tts"] * 4
    # The model divides every stream's logits by T; SGLang samples them at its
    # own temperature 1 and returns their log_softmax.
    prepared = torch.load(tmp_path / "replica-0" / "sources.pt", weights_only=True)
    assert prepared["temperature"] == temperature
    assert {params["temperature"] for params in engine.engine.params} == {1.0}
    assert sglang_sampler.SGLANG_RETURN_ORIGINAL_LOGPROB
    for generation in generations:
        assert generation["sampling_contract"] == {
            "temperature": temperature,
            "top_k": 0,
        }
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
        core,
        token_chunk_size=4,
        bagpiper_config=GEOMETRY,
        score_checkpointing=False,
        temperature=temperature,
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
    # At unit temperature every stream's actions have other probabilities.
    unit = {"temperature": 1.0, "top_k": 0}
    untempered = collate_rollouts(
        sources, [{**g, "sampling_contract": unit} for g in generations], 4
    )
    with torch.no_grad():
        other = NativeSpeechLMPolicy(core, GEOMETRY, score_checkpointing=False)(
            untempered
        )
    gaps = (other.stream_log_probs - recorded).abs().flatten(0, 1).max(0).values
    assert ((gaps < 1e-4) if temperature == 1 else (gaps > 1e-2)).all()
    # The update pass scores under the same recorded supports with gradients.
    policy.score_checkpointing = True
    policy(batch_).log_probs.sum().backward()
    assert core.lm_head.weight.grad is not None
    # A policy whose own supports differ still scores the recorded ones; the
    # frames it disagrees on are counted, and a scoring pass gated by their rate.
    with torch.no_grad():
        core.lm_head.weight.neg_()
        before = support_counts(policy)
        policy(batch_)
    counts = [now - then for now, then in zip(support_counts(policy), before)]
    assert counts[0] > 0 and counts[1] == int(batch_.batch["response_lengths"].sum())
    with pytest.raises(RuntimeError, match="another support"):
        support_mismatch_rate(policy, counts, "current")


def test_batched_learner_equals_the_learner_per_candidate(
    native, tmp_path, monkeypatch
):
    """Right-padded micro-batches score and differentiate as single candidates.

    DESIGN test 4: QA prompts with their own audio spans at different
    positions and a TTS prompt with codec frames, mixed in one batch.
    """
    core, _ = native
    audio = core.multimodal_io_dict["continuous_audio"]
    frames = lambda *ids: [[i, 0, 0, 0] for i in ids]  # noqa: E731
    sources = rollout_sources()[1:]
    for identity, count, lead in (("qa-a", 24, 1), ("qa-b", 40, 3)):
        feats = torch.randn(1, count, 8, generator=torch.Generator().manual_seed(count))
        with torch.no_grad():
            span = len(audio.encode_batch(feats, torch.tensor([count]))[0])
        sources.append(
            dict(
                id=identity,
                task="qa",
                prompt_token_ids=frames(1, *[5] * lead, *[0] * span, 20, 2, 6),
                native_model_inputs=dict(
                    continuous_audio_feats=feats.numpy(),
                    continuous_audio_lengths=[count],
                    continuous_audio_indices=[[0, 1 + lead, span]],
                ),
            )
        )
    engine = build_engine(native, tmp_path, monkeypatch)
    torch.manual_seed(1)
    batch_ = collate_rollouts(sources, engine.generate(sources, 0), 4)
    feats = [
        source.get("native_model_inputs", {}).get("continuous_audio_feats")
        for source in batch_.non_tensor_batch["source"]
    ]
    # The candidates of one prompt share its feature array.
    assert all(item is feats[4] for item in feats[4:8]) and feats[8] is not feats[4]
    calls = []
    encode = audio.encode_batch
    monkeypatch.setattr(
        audio, "encode_batch", lambda *args: calls.append(1) or encode(*args)
    )
    policy = NativeSpeechLMPolicy(core, GEOMETRY, token_chunk_size=3)
    count = len(batch_)
    with torch.no_grad():
        together = policy(batch_)
        assert len(calls) == 2  # once per prompt, not per candidate
        apart = [policy(batch_.select_idxs([i])) for i in range(count)]
    for name in ("log_probs", "entropy", "response_mask", "stream_log_probs"):
        torch.testing.assert_close(
            getattr(together, name),
            torch.cat([getattr(scores, name) for scores in apart]),
            atol=1e-6,
            rtol=0,
        )
    weights = torch.randn(
        batch_.batch["response_mask"].shape, generator=torch.Generator().manual_seed(9)
    )

    def gradients(parts):
        core.zero_grad(set_to_none=True)
        for part in parts:
            scores = policy(batch_.select_idxs(part))
            loss = (scores.log_probs + 0.1 * scores.entropy) * weights[part]
            loss.sum().backward()
        return {
            name: parameter.grad.clone()
            for name, parameter in core.named_parameters()
            if parameter.grad is not None
        }

    together = gradients([list(range(count))])
    apart = gradients([[i] for i in range(count)])
    assert together.keys() == apart.keys()
    assert {"adaptor.continuous_audio.weight", "stream_emb.weight"} <= set(together)
    for name, gradient in together.items():
        # FP32 sums over frames in another order: 1e-6 of the gradient scale.
        scale = max(1.0, float(gradient.abs().max()))
        torch.testing.assert_close(gradient, apart[name], atol=1e-6 * scale, rtol=0)
    # Token-budget micro-batches: candidates by length, padded size in budget.
    trainer = NativeGRPOTrainer.__new__(NativeGRPOTrainer)
    trainer.config = SimpleNamespace(micro_batch_size=3, micro_batch_tokens=60)
    parts = trainer._micro_batches(batch_)
    lengths = [
        len(prompt) + length
        for prompt, length in zip(
            batch_.non_tensor_batch["prompt_token_ids"],
            batch_.batch["response_lengths"].tolist(),
        )
    ]
    order = np.concatenate(parts).tolist()
    assert sorted(order) == list(range(count)) and len(parts) > 3
    assert [lengths[i] for i in order] == sorted(lengths)
    for part in parts:
        assert len(part) <= 3
        assert len(part) == 1 or len(part) * max(lengths[i] for i in part) <= 60
    with torch.no_grad():
        torch.testing.assert_close(
            trainer._score_chunks(policy, batch_),
            torch.cat(
                [policy(batch_.select_idxs([i])).log_probs for i in range(count)]
            ),
            atol=1e-6,
            rtol=0,
        )


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_sampling_override_and_retraction_free_sizing(
    native, tmp_path, monkeypatch, temperature
):
    engine = build_engine(
        native, tmp_path, monkeypatch, dtype="bfloat16", temperature=temperature
    )
    assert engine.engine.settings["attention_backend"] == "fa3"
    override = {"temperature": 0.0, "max_new_tokens": 6, "samples": 1}
    greedy = [engine.generate(rollout_sources(), 1, override) for _ in range(2)]
    assert greedy[0] == greedy[1]
    assert [g["source_id"] for g in greedy[0]] == ["qa", "tts"]
    assert all(len(g["token_ids"]) <= 6 for g in greedy[0])
    assert {g["sampling_contract"]["temperature"] for g in greedy[0]} == {0.0}
    assert engine.engine.model.temperature == 0.0
    assert {params["temperature"] for params in engine.engine.params} == {0.0}
    with pytest.raises(ValueError, match="overrides temperature"):
        engine.generate(rollout_sources(), 2, {"top_p": 0.5})
    # SGLang serves an 8-frame prompt whole with at most 2,038 new tokens.
    engine.check_request(8, 2038)
    with pytest.raises(ValueError, match="exceed context_length"):
        engine.generate(rollout_sources(), 3, {"max_new_tokens": 2039})
    # Every running request at its longest must fit the KV pool at startup.
    monkeypatch.setattr(FakeEngine, "pool", 3 * 2048 - 1)
    (tmp_path / "small").mkdir()
    with pytest.raises(ValueError, match="never retracts"):
        build_engine(native, tmp_path / "small", monkeypatch)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_held_out_evaluation_decodes_through_the_engine(
    native, tmp_path, monkeypatch, temperature
):
    """The evaluator's generate on BagpiperEngine itself: greedy and sampled
    candidates in rollouts of one prompt each, records that MMAU scoring and
    the native verification accept."""
    from espnet2.speechlm.model.speechlm.rl.bagpiper import evaluate

    core, export = native
    fake_sglang(native, monkeypatch)
    config = ESPnetRLConfig(
        family="bagpiper",
        adapter_factory="a:b",
        exporter="c:d",
        sglang_engine="espnet2.speechlm.model.speechlm.rl.bagpiper.rollout:"
        "BagpiperEngine",
        group_size=1,
        max_tokens=10,
        temperature=temperature,
    )
    sources = rollout_sources()
    sources[0].update(reference="20 2", choices=["20 2", "7"])
    greedy = evaluate.generate(
        config, sources, export.parent, {"temperature": 0.0, "max_new_tokens": 10}, 1
    )
    assert [g["source_id"] for g in greedy] == ["qa", "tts"]
    for generation in greedy:
        assert generation["sampling_contract"] == {"temperature": 0.0, "top_k": 0}
        assert len(generation["support"]) == len(generation["token_ids"]) <= 10
    items, summary = evaluate.score_mmau(sources[:1], greedy[:1], 1)
    assert summary["count"] == 1 and items[0]["text"] == greedy[0]["text"]
    # Sampled candidates are the behavior policy the native checkpoint verifies.
    sampled = tmp_path / "sampled"
    sampled.mkdir()
    (sampled / "hf").symlink_to(export, target_is_directory=True)
    config.group_size = 4
    torch.manual_seed(2)
    generations = evaluate.generate(
        config, sources, sampled, {"temperature": temperature, "max_new_tokens": 10}, 1
    )
    policy = NativeSpeechLMPolicy(
        core, GEOMETRY, token_chunk_size=4, temperature=temperature
    )
    errors, drift = evaluate.verify_native_log_probs(
        policy, collate_rollouts(sources, generations, 4), config
    )
    assert len(errors) == 8 and drift["max"] < 1e-4
