"""CPU contract checks; these do not establish H100 training or placement."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from espnet2.bin.rl_train_async import resolve_nodes
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.data import verify_data_admission
from espnet2.rl.engine import engine_kwargs, fit_context
from espnet2.rl.rollout import sha256
from espnet2.rl.types import tempered


def configuration(**kwargs):
    defaults = dict(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        prompts_per_step=32,
        group_size=8,
        steps=5,
        warmup_steps=1,
        train_gpus="0,1,2,3",
        rollout_gpus="4,5,6,7",
    )
    return ESPnetRLConfig(**{**defaults, **kwargs})


def admission_files(tmp_path, train_count=32, heldout_count=8):
    rows = []
    for index in range(train_count + heldout_count):
        audio = tmp_path / f"{index}.wav"
        audio.write_bytes(f"unique audio {index}".encode())
        rows.append(
            {"id": str(index), "audio": str(audio), "audio_sha256": sha256(audio)}
        )
    train, heldout = tmp_path / "train32.jsonl", tmp_path / "heldout.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows[:train_count]))
    heldout.write_text("".join(json.dumps(row) + "\n" for row in rows[train_count:]))
    receipt = {
        "training_ready": True,
        "manifest_sha256": {"train": sha256(train), "heldout": sha256(heldout)},
        "counts": {"train": train_count, "heldout": heldout_count},
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(receipt))
    return configuration(manifest=str(train), eval_manifest=str(heldout)), path


@pytest.mark.parametrize(
    "changed", ["audio", "train_manifest", "heldout_manifest", "counts", "not_ready"]
)
def test_admission_detects_live_file_or_receipt_changes(tmp_path, changed):
    config, path = admission_files(tmp_path)
    assert verify_data_admission(config, path)["counts"]["train"] == 32
    if changed == "audio":
        (tmp_path / "0.wav").write_bytes(b"different audio")
    elif changed == "train_manifest":
        Path(config.manifest).write_text("changed manifest")
    elif changed == "heldout_manifest":
        Path(config.eval_manifest).write_text("changed manifest")
    else:
        receipt = json.loads(path.read_text())
        if changed == "counts":
            receipt["counts"]["train"] = 31
        else:
            receipt["training_ready"] = False
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_data_admission(config, path)


@pytest.mark.parametrize("training_key", ["train", "train32"])
def test_admission_rejects_swapped_equal_sized_manifests(tmp_path, training_key):
    config, path = admission_files(tmp_path, train_count=4, heldout_count=4)
    receipt = json.loads(path.read_text())
    receipt["manifest_sha256"][training_key] = receipt["manifest_sha256"].pop("train")
    path.write_text(json.dumps(receipt))
    assert verify_data_admission(config, path)["counts"] == {"train": 4, "heldout": 4}

    config.manifest, config.eval_manifest = config.eval_manifest, config.manifest
    with pytest.raises(ValueError, match="train manifest differs"):
        verify_data_admission(config, path)


@pytest.mark.parametrize(
    "train, rollout, ranks, replicas, collocated",
    [
        ("0", "0", 1, 1, True),
        ("0", "1", 1, 1, False),
        ("0,1,2,3", "4,5,6,7", 4, 4, False),
        ("0,1,2,3,4,5,6,7", "0,1,2,3,4,5,6,7", 8, 8, True),
    ],
)
def test_geometry_comes_from_the_declared_devices(
    train, rollout, ranks, replicas, collocated
):
    config = configuration(train_gpus=train, rollout_gpus=rollout)
    config.validate()
    assert config.train_ranks == ranks
    assert config.rollout_replicas == replicas
    assert config.collocated is collocated


@pytest.mark.parametrize("name", ["train_gpus", "rollout_gpus"])
def test_repeated_or_empty_device_lists_are_rejected(name):
    for value in ("0,0", ""):
        config = configuration(**{name: value})
        with pytest.raises(ValueError, match="distinct physical GPU ids"):
            config.validate()


def test_actions_are_sampled_from_their_whole_support_at_most_at_unit_temperature():
    for family in ("owsm", "bagpiper"):
        for temperature in (0.7, 1, 1.0):
            configuration(family=family, temperature=temperature).validate()
    for changes in (
        {"temperature": 0.0},
        {"temperature": -0.7},
        {"temperature": 1.3},
        {"temperature": float("nan")},
        {"temperature": "0.7"},
        {"top_p": 0.9},
        {"top_k": 50},
    ):
        with pytest.raises(ValueError, match="whole support"):
            configuration(**changes).validate()
    # Engines and learners that do not apply a temperature sample at 1.
    vllm = dict(rollout_backend="vllm", rollout_gpus="4", vllm_generate_factory="a:b")
    for changes in ({"family": "opuslm"}, {"family": "opuslm_dialogue"}, vllm):
        configuration(**changes).validate()
        with pytest.raises(ValueError, match="temperature 1 only"):
            configuration(temperature=0.7, **changes).validate()


def test_tempered_logits_are_the_ones_sglang_samples_from():
    logits = torch.randn(4, 9).masked_fill(torch.rand(4, 9) < 0.3, -torch.inf)
    # Unit temperature is the logits themselves: nothing is recomputed.
    assert tempered(logits, 1.0) is logits
    # SGLang's sampler divides by a [B, 1] float32 temperature tensor in place.
    expected = logits.clone().div_(torch.full((4, 1), 0.7))
    assert torch.equal(tempered(logits, 0.7), expected)


def test_a_config_with_the_replaced_engine_key_names_its_successor(tmp_path):
    for path, text in (
        (tmp_path / "recipe.yaml", "sglang_generate_factory: a:b\n"),
        (tmp_path / "frozen_config.json", '{"sglang_generate_factory": "a:b"}'),
    ):
        path.write_text(text)
        with pytest.raises(ValueError, match="replaced by sglang_engine"):
            ESPnetRLConfig.load(path)


def test_longer_evaluation_requests_keep_the_recipe_kv_budget():
    config = configuration(sglang_engine_kwargs={"max_running_requests": 44})
    defaults = {"context_length": 1280, "max_running_requests": 8}
    fit_context(config, defaults, 1280)
    assert config.sglang_engine_kwargs == {"max_running_requests": 44}
    fit_context(config, defaults, 1753)
    assert config.sglang_engine_kwargs == {
        "context_length": 1753,
        "max_running_requests": 44 * 1280 // 1753,
    }
    # An engine that bounds neither is left to its own checks.
    unbounded = configuration()
    fit_context(unbounded, {}, 4096)
    assert unbounded.sglang_engine_kwargs == {}


def test_engine_kwargs_sizes_the_engine_and_keeps_frozen_settings():
    """rollout_mem_fraction is the one way a recipe sizes its engine."""
    sized = configuration(rollout_mem_fraction=0.2)
    sized.validate()
    settings = engine_kwargs(sized, 3, defaults={"mem_fraction_static": 0.65})
    assert settings["mem_fraction_static"] == 0.2
    # One seed per replica for the engine's whole life.
    assert settings["random_seed"] == sized.seed + 3
    assert settings["tp_size"] == 1
    assert settings["disable_cuda_graph"] is True
    assert settings["disable_radix_cache"] is True
    # Left unset, the family's own default survives untouched.
    assert (
        engine_kwargs(configuration(), 0, defaults={"mem_fraction_static": 0.65})[
            "mem_fraction_static"
        ]
        == 0.65
    )
    # Model identity, dtype, one device per replica and the replica's seed
    # are frozen for every family, not per family.
    repeated = configuration(sglang_engine_kwargs={"dtype": "float32", "tp_size": 1})
    assert (
        engine_kwargs(repeated, 0, defaults={"dtype": "float32"})["dtype"] == "float32"
    )
    changed = configuration(sglang_engine_kwargs={"dtype": "bfloat16"})
    with pytest.raises(ValueError, match="native execution contract"):
        engine_kwargs(changed, 0, defaults={"dtype": "float32"})
    # A recipe-wide seed would make every replica repeat the same candidates.
    pinned = configuration(sglang_engine_kwargs={"random_seed": 7})
    with pytest.raises(ValueError, match="native execution contract"):
        engine_kwargs(pinned, 3)
    # A setting a family does not freeze stays a recipe decision.
    loud = configuration(sglang_engine_kwargs={"log_level": "debug"})
    assert engine_kwargs(loud, 0)["log_level"] == "debug"
    # The family's own frozen settings compose with the shared ones.
    unsafe = configuration(sglang_engine_kwargs={"max_running_requests": 4})
    with pytest.raises(ValueError, match="native execution contract"):
        engine_kwargs(
            unsafe,
            0,
            defaults={"max_running_requests": 1},
            frozen=("max_running_requests",),
        )
    # A memory fraction is still a fraction, whichever way it arrives.
    for kwargs in ({"mem_fraction_static": 1.4}, {"mem_fraction_static": 0}):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            engine_kwargs(configuration(sglang_engine_kwargs=kwargs), 0)
    # The two ways to size the engine may not disagree.
    conflict = configuration(
        rollout_mem_fraction=0.2, sglang_engine_kwargs={"mem_fraction_static": 0.5}
    )
    with pytest.raises(ValueError, match="disagree"):
        engine_kwargs(conflict, 0)
    agreed = configuration(
        rollout_mem_fraction=0.2, sglang_engine_kwargs={"mem_fraction_static": 0.2}
    )
    assert engine_kwargs(agreed, 0)["mem_fraction_static"] == 0.2


def test_node_resolution_allows_one_node_that_holds_both_roles():
    nodes = [
        {
            "NodeID": str(i),
            "NodeManagerAddress": f"10.0.0.{i}",
            "Alive": True,
            "Resources": {"GPU": 8},
        }
        for i in range(2)
    ]
    assert resolve_nodes(
        nodes, "0", "10.0.0.1", train_ranks=8, rollout_replicas=8
    ) == tuple(nodes)
    # One node may host both roles when it exposes enough GPUs for each.
    first, second = resolve_nodes(
        nodes, "0", "10.0.0.0", train_ranks=1, rollout_replicas=1
    )
    assert first["NodeID"] == second["NodeID"] == "0"
    with pytest.raises(ValueError, match="both roles"):
        resolve_nodes(nodes, "0", "10.0.0.0", train_ranks=4, rollout_replicas=8)
    with pytest.raises(ValueError, match="fewer than 8"):
        resolve_nodes(
            [nodes[0], {**nodes[1], "Resources": {"GPU": 6}}],
            "0",
            "1",
            train_ranks=8,
            rollout_replicas=8,
        )


def test_ray_object_store_roundtrip_uses_actual_reference_and_cpu_batch():
    import ray
    import torch
    from miles.utils import object_store
    from miles.utils.data import RolloutDataPack, remove_rollout_data_refs

    previous = object_store._INSTANCE
    ray.init(
        address="local",
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=80 * 1024 * 1024,
    )
    try:
        object_store._INSTANCE = None
        store = object_store.init_instance(
            SimpleNamespace(object_store_backend="ray", worker_comm_backend="ray")
        )
        ref = store.put({"actions": torch.tensor([1, 2]), "behavior_version": 0})
        pack = RolloutDataPack(data_ref=ref, sample_indices=[0, 1])
        with store.get(pack.data_ref) as payload:
            assert payload["behavior_version"] == 0
            torch.testing.assert_close(payload["actions"], torch.tensor([1, 2]))
        remove_rollout_data_refs(SimpleNamespace(), pack)
        # Ray wire mode uses normal reference lifetime, not force-free.
        assert store._frees_objects is False
    finally:
        object_store._INSTANCE = previous
        ray.shutdown()
