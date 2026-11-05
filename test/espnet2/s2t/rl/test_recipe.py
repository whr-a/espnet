"""The OWSM stage-18 builder keeps a frozen manifest and the recipe geometry."""

import sys

import yaml

from espnet2.rl.config import ESPnetRLConfig


def four_rank_config(tmp_path):
    """A collocated recipe of four training ranks and four rollout replicas."""
    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        execution_mode="sync",
        prompts_per_step=32,
        group_size=8,
        steps=5,
        warmup_steps=1,
        reference_device="cpu",
        manifest="train32.jsonl",
        train_gpus="0,1,2,3",
        rollout_gpus="4,5,6,7",
    )
    path = tmp_path / "sync.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    return path


def test_owsm_recipe_builder_preserves_five_updates_and_existing_manifest(
    tmp_path, monkeypatch
):
    from espnet2.s2t.rl import recipe

    config = four_rank_config(tmp_path)
    manifest = tmp_path / "train32.jsonl"
    manifest.write_text("already frozen data")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("already frozen heldout")
    observed = []
    monkeypatch.setattr(
        "espnet2.rl.tasks.manifest.verify_data_admission",
        lambda config, path: observed.append((config.manifest, path)),
    )
    work = tmp_path / "work"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recipe",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--eval-manifest",
            str(heldout),
            "--data-admission",
            "audit.json",
            "--train-config",
            "native.yaml",
            "--checkpoint",
            "native.pt",
            "--work-dir",
            str(work),
        ],
    )
    recipe.main()
    effective = ESPnetRLConfig.load(work / "rl_config.yaml")
    assert effective.steps == 5 and effective.warmup_steps == 1
    assert effective.train_gpus == "0,1,2,3" and effective.rollout_gpus == "4,5,6,7"
    assert effective.train_ranks == 4 and effective.rollout_replicas == 4
    assert effective.reference_device == "cpu"
    assert (
        effective.manifest == str(manifest)
        and manifest.read_text() == "already frozen data"
    )
    assert observed == [(str(manifest), "audit.json")]
