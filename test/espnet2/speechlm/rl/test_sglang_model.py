"""Complete weight loads for the OpusLM SGLang models, without a GPU."""

from types import SimpleNamespace

import pytest
from torch import nn

pytest.importorskip("sglang")

from espnet2.speechlm.rl import sglang_model  # noqa: E402

EXPORT = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
    "head_emb.weight",
]


class Backbone(nn.Module):
    """SGLang's parameter layout and a loader that, like the real ones,
    takes whatever names it is given without checking coverage."""

    def __init__(self, tie=False):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(6, 2)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(2, 6, bias=False)
        layer.self_attn.o_proj = nn.Linear(2, 2, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Linear(2, 8, bias=False)
        layer.mlp.down_proj = nn.Linear(4, 2, bias=False)
        self.model.layers = nn.ModuleList([layer])
        self.model.norm = nn.LayerNorm(2, bias=False)
        self.lm_head = self.model.embed_tokens if tie else nn.Linear(2, 6, bias=False)
        self.head_emb = nn.Embedding(12, 2)
        self.loads = []

    def load_weights(self, weights):
        self.loads.append([name for name, _ in weights])


class Model(sglang_model._CompleteLoad, Backbone):
    pass


def weights(names):
    return ((name, None) for name in names)


def test_every_load_must_write_the_whole_policy():
    model = Model()
    # Built from one export, then updated in place by the next.
    model.load_weights(weights(EXPORT))
    model.load_weights(weights(EXPORT))
    assert model.loads == [EXPORT, EXPORT]
    for names, message in (
        (EXPORT[:-1], "missing.*head_emb"),
        (EXPORT + ["model.layers.1.mlp.down_proj.weight"], "unexpected.*layers.1"),
        (EXPORT + EXPORT[:1], "exactly once"),
    ):
        with pytest.raises(ValueError, match=message):
            model.load_weights(weights(names))


def test_a_tied_head_is_carried_by_the_embedding():
    model = Model(tie=True)
    model.load_weights(weights(EXPORT))
    model.load_weights(weights(name for name in EXPORT if name != "lm_head.weight"))


def test_both_architectures_load_through_the_complete_check():
    for cls in (
        sglang_model.OpusLMForConditionalGeneration,
        sglang_model.OpusLMDialogueForConditionalGeneration,
    ):
        assert cls.load_weights is sglang_model._CompleteLoad.load_weights
