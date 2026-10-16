"""Native text tokenization survives the metadata-only export vocabulary shift."""

import json
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import AutoTokenizer

from espnet2.speechlm.model.speechlm.rl.bagpiper.prepare_assets import SOURCES, prepare


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("bagpiper-metadata-input")
    for name in SOURCES:
        (root / name).mkdir()
    text_config = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000,
    }
    write(root / "text/config.json", text_config)
    # Dense checkpoint-size geometry with a small byte-level language. The
    # placeholder vocabulary occupies the same original IDs as public Qwen;
    # no real model or network access is required by this regression test.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: index for index, token in enumerate(alphabet)}
    vocabulary.update({f"<base_{index}>": index for index in range(256, 151643)})
    backend = Tokenizer(models.BPE(vocabulary, []))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    raw = json.loads(backend.to_str())
    tail = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ]
    entries = [
        {
            "id": 151643 + index,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": index < 14,
        }
        for index, token in enumerate(tail)
    ]
    raw["added_tokens"] = entries[:22]
    write(root / "text/tokenizer.json", raw)
    write(
        root / "text/tokenizer_config.json",
        {
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "clean_up_tokenization_spaces": False,
            "added_tokens_decoder": {
                str(item["id"]): {
                    key: value for key, value in item.items() if key != "id"
                }
                for item in entries
            },
        },
    )
    write(
        root / "audio/config.json",
        {
            "thinker_config": {
                "audio_config": {
                    "num_mel_bins": 128,
                    "encoder_layers": 32,
                    "encoder_attention_heads": 20,
                    "d_model": 1280,
                    "output_dim": 2048,
                    "n_window": 50,
                    "n_window_infer": 800,
                }
            }
        },
    )
    write(
        root / "audio/preprocessor_config.json",
        {"sampling_rate": 16000, "feature_size": 128, "hop_length": 160, "n_fft": 400},
    )
    write(
        root / "codec/config.json",
        {
            "model_type": "xcodec",
            "codebook_size": 1024,
            "sample_rate": 16000,
            "acoustic_model_config": {"hop_length": 320},
        },
    )
    recipe = {
        "job_type": "speechlm",
        "multimodal_io": {
            "text": {"tokenizer_name": SOURCES["text"][0]},
            "discrete_audio": {
                "codec_choice": "Xcodec",
                "codec_hf_model_tag": SOURCES["codec"][0],
                "ssl_choice": None,
                "delay_interleave": True,
                "stream_weights": [0.125] * 8,
            },
            "continuous_audio": {"encoder_hf_model_tag": SOURCES["audio"][0]},
        },
        "model": {"model_choice": "parallel", "model_hf_tag": SOURCES["text"][0]},
    }
    (root / "native.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    return root


@pytest.fixture(scope="module")
def prepared(inputs, tmp_path_factory):
    output = tmp_path_factory.mktemp("bagpiper-metadata-output") / "new"
    receipt = prepare(
        inputs / "native.yaml", output, *(inputs / name for name in SOURCES)
    )
    return output, receipt


def test_native_vs_export_tokenizer_roundtrip_including_config_only_tokens(
    inputs, prepared
):
    output, _ = prepared
    native = AutoTokenizer.from_pretrained(
        inputs / "text", use_fast=True, local_files_only=True
    )
    exported = AutoTokenizer.from_pretrained(
        output / "reference_assets", use_fast=True, local_files_only=True
    )
    assert len(native) == 151669 and len(exported) == 160392
    assert (
        json.loads(native.backend_tokenizer.to_str())["pre_tokenizer"]
        == json.loads(exported.backend_tokenizer.to_str())["pre_tokenizer"]
    )
    for text in (
        "Hello, audio!",
        "\n你好 — café 🎵\t",
        "<think>try</think>",
        "<tool_response>42</tool_response>",
        "Don't change 151936.",
    ):
        before = native.encode(text, add_special_tokens=False)
        after = exported.encode(text, add_special_tokens=False)
        assert after == [token + 256 for token in before]
        for skip in (False, True):
            assert exported.decode(after, skip_special_tokens=skip) == native.decode(
                before, skip_special_tokens=skip
            )
    for index, token in enumerate(
        ("<tool_response>", "</tool_response>", "<think>", "</think>"), 151665
    ):
        assert native.convert_tokens_to_ids(token) == index
        assert exported.convert_tokens_to_ids(token) == index + 256


def test_native_special_reserved_text_and_all_codec_boundaries(prepared):
    output, receipt = prepared
    tokenizer = AutoTokenizer.from_pretrained(
        output / "reference_assets", use_fast=True, local_files_only=True
    )
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (
        0,
        1,
        2,
    )
    assert tokenizer.convert_tokens_to_ids("<unused_151669>") == 151925
    assert tokenizer.convert_tokens_to_ids("<unused_151935>") == 152191
    for stream in range(8):
        assert (
            tokenizer.convert_tokens_to_ids(f"<codec_layer{stream}_pad>")
            == 152192 + stream * 1025
        )
        assert (
            tokenizer.convert_tokens_to_ids(f"<codec_layer{stream}_0>")
            == 152193 + stream * 1025
        )
        assert (
            tokenizer.convert_tokens_to_ids(f"<codec_layer{stream}_1023>")
            == 153216 + stream * 1025
        )
    assert (
        tokenizer.decode(
            [0, 1, 2, 151925, 152191, 152192, 160391], skip_special_tokens=True
        )
        == ""
    )
    assert receipt["model_weights_downloaded_or_loaded"] is False
    assert not any(
        path.suffix in (".pt", ".pth", ".safetensors") for path in output.rglob("*")
    )
    recipe = yaml.safe_load((output / "native.yaml").read_text())
    assert list(recipe["multimodal_io"]) == [
        "text",
        "discrete_audio",
        "continuous_audio",
    ]
    assert recipe["multimodal_io"]["text"]["tokenizer_name"] == str(output / "text")
    assert recipe["model"]["model_hf_tag"] == str(output / "text")


def test_wrong_config_vocabulary_is_rejected_before_any_output(inputs, tmp_path):
    text = tmp_path / "wrong-text"
    text.mkdir()
    config = json.loads((inputs / "text/config.json").read_text())
    config["vocab_size"] = (
        151669  # Actual tokenizer length must not replace checkpoint vocabulary.
    )
    write(text / "config.json", config)
    with pytest.raises(ValueError, match="text geometry: vocab_size"):
        prepare(
            inputs / "native.yaml",
            tmp_path / "output",
            text,
            inputs / "audio",
            inputs / "codec",
        )
    assert not (tmp_path / "output").exists()


def test_wrong_modality_order_is_rejected(inputs, tmp_path):
    recipe = yaml.safe_load((inputs / "native.yaml").read_text())
    io = recipe["multimodal_io"]
    recipe["multimodal_io"] = {
        name: io[name] for name in ("discrete_audio", "text", "continuous_audio")
    }
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    with pytest.raises(ValueError, match="modality order"):
        prepare(path, tmp_path / "output", *(inputs / name for name in SOURCES))


def test_conflicting_config_only_tail_token_is_rejected(inputs, tmp_path):
    from espnet2.speechlm.model.speechlm.rl.bagpiper.prepare_assets import (
        shifted_tokenizer,
    )

    config = json.loads((inputs / "text/tokenizer_config.json").read_text())
    config["added_tokens_decoder"]["151665"]["content"] = "<wrong>"
    with pytest.raises(ValueError, match="config-only Qwen tail"):
        shifted_tokenizer(
            json.loads((inputs / "text/tokenizer.json").read_text()), config
        )


def test_existing_asset_directory_is_never_overwritten(inputs, prepared):
    output, _ = prepared
    before = (output / "asset_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(inputs / "native.yaml", output, *(inputs / name for name in SOURCES))
    assert (output / "asset_manifest.json").read_bytes() == before
