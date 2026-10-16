"""OWSM rollouts sampled at a temperature, from the engine to the update.

OWSMEngine drives a stand-in for sglang.Engine that schedules the real
ESPnetOWSMForCausalLM, on a tiny ESPnet TransformerDecoder, as SGLang does and
samples every step with SGLang's own sampler. The native learner then scores
the recorded actions, and the trainer prepares and applies one update.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("sglang")

from sglang.srt.layers import sampler as sglang_sampler  # noqa: E402
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.layers.logprob_processor import OutputLogprobProcessor  # noqa: E402
from sglang.srt.sampling.sampling_params import (  # noqa: E402
    TOP_K_ALL,
    SamplingParams,
)
from sglang.srt.utils.common import get_bool_env_var  # noqa: E402

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder  # noqa: E402
from espnet2.rl.config import ESPnetRLConfig  # noqa: E402
from espnet2.rl.data import collate_rollouts  # noqa: E402
from espnet2.rl.trainer import NativeGRPOTrainer  # noqa: E402
from espnet2.s2t.rl import sglang_rollout  # noqa: E402
from espnet2.s2t.rl.policy import OWSMPolicy  # noqa: E402
from espnet2.s2t.rl.sglang_model import ESPnetOWSMForCausalLM  # noqa: E402

VOCAB, EOS, WIDTH = 31, 30, 16


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


class Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind == "extend"

    def is_idle(self):
        return False


class FakeEngine:
    """sglang.Engine on CPU: prefills every request together, then decodes the
    unfinished ones together, sampling each step with SGLang's sampler."""

    decoder = None

    def __init__(self, **settings):
        # The scheduler reads this once, when it imports its sampler.
        sglang_sampler.SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var(
            "SGLANG_RETURN_ORIGINAL_LOGPROB"
        )
        overrides = json.loads(settings["json_model_override_args"])
        model = ESPnetOWSMForCausalLM.__new__(ESPnetOWSMForCausalLM)
        nn.Module.__init__(model)
        model.policy = nn.Module()
        model.policy.model = nn.Module()
        model.policy.model.decoder = self.decoder
        model.policy.model.eos = EOS
        model.config = SimpleNamespace(vocab_size=VOCAB, torch_dtype="float32")
        model.target_device = torch.device("cpu")
        model.memory, model.request_prefixes = {}, {}
        model.rollout_id = model.sources = model.source_sha256 = None
        model.source_path = Path(overrides["espnet_rollout_sources"])

        def memory(index, device):
            features = torch.tensor(model.sources[index]["encoder_features"])
            return features[None], torch.tensor([len(features)])

        model._memory = memory
        self.model = model

    def generate(self, input_ids, sampling_params, rid, **kwargs):
        self.params = sampling_params
        rows, outputs = list(range(len(rid))), {}
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor(rows),
            rids=rid,
            forward_mode=Mode("extend"),
            extend_seq_lens_cpu=[len(ids) for ids in input_ids],
            extend_prefix_lens_cpu=[0] * len(rid),
            return_logprob=True,
        )
        inputs = torch.tensor(sum(input_ids, []))
        while rows:
            with torch.no_grad():
                logits = self.model.forward(inputs, None, batch).next_token_logits
            tokens, logprobs = sglang_sample(logits, [sampling_params[r] for r in rows])
            for row, token, logprob in zip(rows, tokens, logprobs):
                outputs.setdefault(row, ([], []))
                outputs[row][0].append(token)
                outputs[row][1].append([logprob, token, None])
            rows = [
                row
                for row in rows
                if outputs[row][0][-1] not in sampling_params[row]["stop_token_ids"]
                and len(outputs[row][0]) < sampling_params[row]["max_new_tokens"]
            ]
            inputs = torch.tensor([outputs[row][0][-1] for row in rows])
            batch = SimpleNamespace(
                req_pool_indices=torch.tensor(rows),
                rids=[rid[row] for row in rows],
                forward_mode=Mode("decode"),
                return_logprob=True,
            )
        return [
            {
                "output_ids": outputs[row][0],
                "meta_info": {
                    "id": name,
                    "output_token_logprobs": outputs[row][1],
                    "finish_reason": {
                        "type": "stop" if outputs[row][0][-1] == EOS else "length"
                    },
                },
            }
            for row, name in enumerate(rid)
        ]


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_engine_and_learner_agree_on_the_tempered_policy(
    tmp_path, monkeypatch, temperature
):
    import sglang

    torch.manual_seed(0)
    decoder = TransformerDecoder(
        VOCAB,
        WIDTH,
        attention_heads=2,
        linear_units=32,
        num_blocks=2,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        use_flash_attn=False,
    ).eval()
    FakeEngine.decoder = decoder
    monkeypatch.setattr(sglang, "Engine", FakeEngine)
    monkeypatch.setattr(
        sglang_rollout, "_detokenizer", lambda path: lambda ids: " ".join(map(str, ids))
    )
    # An inherited setting must not reach the scheduler; both are restored.
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    monkeypatch.setattr(sglang_sampler, "SGLANG_RETURN_ORIGINAL_LOGPROB", True)
    export = tmp_path / "hf"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps({"espnet_config": {"token_list": []}, "eos_token_id": EOS})
    )
    config = ESPnetRLConfig(
        adapter_factory="a:b",
        exporter="c:d",
        sglang_engine="e:f",
        prompts_per_step=3,
        group_size=4,
        max_tokens=6,
        temperature=temperature,
    )
    engine = sglang_rollout.OWSMEngine(config, export, directory=tmp_path)
    sources = [
        {
            "id": f"utt{index}",
            "prompt_token_ids": [27, 28, 29],
            "encoder_features": torch.randn(length, WIDTH).tolist(),
            "reference": "1 2",
        }
        for index, length in enumerate((5, 9, 3))
    ]
    batch = collate_rollouts(sources, engine.generate(sources, 0), 4)
    # SGLang samples at T and returns the log probability after it.
    assert {params["temperature"] for params in engine.engine.params} == {temperature}
    assert not sglang_sampler.SGLANG_RETURN_ORIGINAL_LOGPROB
    model = nn.Module()
    model.decoder, model.eos = decoder, EOS
    policy = OWSMPolicy(model, temperature=temperature)
    mask = batch.batch["response_mask"].bool()
    rollout = batch.batch["rollout_log_probs"]
    with torch.no_grad():
        assert (policy(batch).log_probs - rollout).abs()[mask].max() < 1e-5
        untempered = OWSMPolicy(model)(batch).log_probs
    assert (temperature == 1) == bool((untempered - rollout).abs()[mask].max() < 1e-5)
    # The reference scores at the same temperature: no KL at the first update.
    trainer = NativeGRPOTrainer(policy, config)
    metrics = trainer.prepare_batch(batch)
    assert metrics["rollout/logprob_max_error"] < 1e-5
    assert torch.equal(batch.batch["ref_log_probs"], batch.batch["old_log_probs"])
    metrics = trainer.update_actor(batch)
    assert metrics["actor/kl_loss"] < 1e-9 and metrics["actor/grad_norm"] > 0
