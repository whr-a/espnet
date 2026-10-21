"""Recipe routes and effective config checks without launching training."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from espnet2.rl.cli import launch_command, parser
from espnet2.rl.config import ESPnetRLConfig

ROOT = Path(__file__).resolve().parents[3]


def frozen_config(tmp_path, mode, *, small=False, name=None, **overrides):
    """Write a recipe yaml whose geometry is only its two device lists.

    The small profile is the one-GPU recipe: a single training rank sharing
    its device with the rollout engine. The full profiles keep the original
    four collocated training ranks with four rollout replicas, and eight of
    each for the disaggregated async node pair.
    """
    devices = (
        dict(train_gpus="0", rollout_gpus="0")
        if small
        else dict(
            train_gpus="0,1,2,3,4,5,6,7" if mode == "disagg_async" else "0,1,2,3",
            rollout_gpus=("0,1,2,3,4,5,6,7" if mode == "disagg_async" else "4,5,6,7"),
        )
    )
    config = ESPnetRLConfig(
        adapter_factory="unused:factory",
        exporter="unused:exporter",
        sglang_engine="unused:Engine",
        execution_mode=mode,
        prompts_per_step=2 if small else 32,
        group_size=4 if small else 8,
        steps=5,
        warmup_steps=0 if small else 1,
        reference_device="cpu",
        manifest="train32.jsonl",
        **{**devices, **overrides},
    )
    path = tmp_path / f"{name or mode}.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    return path


def test_shared_launcher_routes_async_and_collocate_and_keeps_small_sync(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    sync = frozen_config(tmp_path, "sync")
    args = parser().parse_args(
        ["--config", str(sync), "--data-admission", "audit.json"]
    )
    command, environment = launch_command(args, python="pinned-python")
    assert command[:7] == [
        "pinned-python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "espnet2.bin.rl_train",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    args.validate_only = True
    assert launch_command(args, python="pinned-python")[0][:3] == [
        "pinned-python",
        "-m",
        "espnet2.bin.rl_train",
    ]
    async_path = frozen_config(tmp_path, "disagg_async")
    args = parser().parse_args(
        [
            "--config",
            str(async_path),
            "--data-admission",
            "audit.json",
            "--ray-address",
            "ray-address",
            "--train-node",
            "train-node",
            "--rollout-node",
            "rollout-node",
        ]
    )
    assert launch_command(args, python="pinned-python")[0][:3] == [
        "pinned-python",
        "-m",
        "espnet2.bin.rl_train",
    ]
    args.rollout_node = None
    with pytest.raises(ValueError, match="rollout-node"):
        launch_command(args)
    # A one-GPU recipe trains in this process and keeps the pod's device
    # list, so its own `device` selects the GPU.
    small = frozen_config(tmp_path, "sync", small=True, name="small")
    args = parser().parse_args(["--config", str(small)])
    command, environment = launch_command(args, python="pinned-python")
    assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
    assert "torch.distributed.run" not in command
    assert environment.get("CUDA_VISIBLE_DEVICES") == os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )


@pytest.mark.parametrize(
    "train_gpus, rollout_gpus, ranks",
    [
        ("0", "0", 1),
        ("3", "3", 1),
        ("0", "1", 1),
        ("0,1", "2,3", 2),
        ("0,1,2,3", "4,5,6,7", 4),
        ("0,1,2,3,4,5,6,7", "0,1,2,3,4,5,6,7", 8),
    ],
)
def test_launcher_geometry_follows_the_recipe_train_gpus(
    tmp_path, monkeypatch, train_gpus, rollout_gpus, ranks
):
    """One training rank per declared train GPU; torchrun only above one."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    path = frozen_config(
        tmp_path, "sync", train_gpus=train_gpus, rollout_gpus=rollout_gpus
    )
    assert ESPnetRLConfig.load(path).train_ranks == ranks
    args = parser().parse_args(["--config", str(path)])
    command, environment = launch_command(args, python="pinned-python")
    if ranks == 1:
        assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
        assert "torch.distributed.run" not in command
        assert environment.get("CUDA_VISIBLE_DEVICES") == os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
    else:
        assert environment["CUDA_VISIBLE_DEVICES"] == train_gpus
        assert command[:7] == [
            "pinned-python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            "-m",
            "espnet2.bin.rl_train",
        ]


def test_single_rank_launches_inside_a_platform_distributed_environment(
    tmp_path, monkeypatch
):
    """A pytorch job pod exports WORLD_SIZE; one rank must still launch.

    The stale variables are dropped from the child environment, so only a
    nested torchrun is refused.
    """
    for name, value in (
        ("WORLD_SIZE", "2"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "10.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        monkeypatch.setenv(name, value)
    one = frozen_config(tmp_path, "sync", train_gpus="0", rollout_gpus="0", name="one")
    command, environment = launch_command(
        parser().parse_args(["--config", str(one)]), python="pinned-python"
    )
    assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
    assert "torch.distributed.run" not in command
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert name not in environment
    # Several ranks still launch: this toolkit's torchrun builds its own
    # rendezvous from the scrubbed environment and pins the devices.
    many = frozen_config(
        tmp_path, "sync", train_gpus="0,1", rollout_gpus="0", name="many"
    )
    command, environment = launch_command(
        parser().parse_args(["--config", str(many)]), python="pinned-python"
    )
    assert "--nproc-per-node=2" in command
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    for name in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
        assert name not in environment


def test_shipped_one_and_two_gpu_owsm_recipes_declare_their_devices(
    tmp_path, monkeypatch
):
    """The two new recipes carry the same geometry the launcher reads back."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    conf = ROOT / "egs2" / "owsm_v4" / "s2t1" / "conf"
    single = ESPnetRLConfig.load(conf / "rl_miles_wer_1gpu.yaml")
    assert (single.execution_mode, single.train_gpus, single.rollout_gpus) == (
        "sync",
        "0",
        "0",
    )
    assert single.train_ranks == 1 and single.rollout_replicas == 1
    assert single.collocated is True
    assert (single.prompts_per_step, single.group_size) == (32, 8)
    args = parser().parse_args(["--config", str(conf / "rl_miles_wer_1gpu.yaml")])
    command, environment = launch_command(args, python="pinned-python")
    assert command[:3] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
    assert "torch.distributed.run" not in command
    assert environment.get("CUDA_VISIBLE_DEVICES") == os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    pair = ESPnetRLConfig.load(conf / "rl_miles_wer_2gpu_async.yaml")
    assert (pair.execution_mode, pair.train_gpus, pair.rollout_gpus) == (
        "disagg_async",
        "0",
        "1",
    )
    assert pair.train_ranks == 1 and pair.rollout_replicas == 1
    assert pair.collocated is False
    assert (pair.prompts_per_step, pair.group_size) == (32, 8)
    selectors = [
        "--config",
        str(conf / "rl_miles_wer_2gpu_async.yaml"),
        "--ray-address",
        "ray-address",
        "--train-node",
        "train-node",
        "--rollout-node",
        "rollout-node",
    ]
    args = parser().parse_args(selectors + ["--data-admission", "audit.json"])
    assert launch_command(args, python="pinned-python")[0][:3] == [
        "pinned-python",
        "-m",
        "espnet2.bin.rl_train",
    ]
    # An admission receipt is optional for both launchers; the Ray selectors
    # are not.
    assert launch_command(parser().parse_args(selectors), python="pinned-python")[0][
        :3
    ] == ["pinned-python", "-m", "espnet2.bin.rl_train"]
    for dropped in ("--ray-address", "--train-node", "--rollout-node"):
        trimmed = list(selectors)
        index = trimmed.index(dropped)
        del trimmed[index : index + 2]
        with pytest.raises(ValueError, match=dropped.lstrip("-")):
            launch_command(parser().parse_args(trimmed), python="pinned-python")


@pytest.mark.parametrize(
    "name, mode, train_gpus, rollout_gpus",
    [
        ("rl_opuslm_dialogue_librispeech_1gpu.yaml", "sync", "0", "0"),
        ("rl_opuslm_dialogue_librispeech_2gpu_async.yaml", "disagg_async", "0", "1"),
        ("rl_opuslm_librispeech_2gpu_async.yaml", "disagg_async", "0", "1"),
    ],
)
def test_shipped_opuslm_recipes_resolve_their_geometry_and_engine(
    monkeypatch, name, mode, train_gpus, rollout_gpus
):
    from espnet2.rl.utils import resolve_callable
    from espnet2.speechlm.rl.integration import OpusLMEngine

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    path = ROOT / "egs2" / "libritts" / "speechlm1" / "conf" / name
    config = ESPnetRLConfig.load(path)
    assert (config.execution_mode, config.train_gpus, config.rollout_gpus) == (
        mode,
        train_gpus,
        rollout_gpus,
    )
    assert config.train_ranks == 1 and config.rollout_replicas == 1
    assert config.collocated is (mode == "sync")
    assert (config.prompts_per_step, config.group_size) == (32, 8)
    assert resolve_callable(config.sglang_engine) is OpusLMEngine
    # The recipe's sizing reaches the engine without touching its contract.
    rollout = OpusLMEngine.__new__(OpusLMEngine)
    rollout.config, rollout.replica = config, 0
    settings = rollout.settings("hf")
    assert settings["mem_fraction_static"] == config.rollout_mem_fraction
    assert settings["dtype"] == "float32" and settings["disable_radix_cache"]
    for key in ("context_length", "max_running_requests"):
        assert settings[key] == config.sglang_engine_kwargs[key]
    selectors = (
        ["--ray-address", "a", "--train-node", "t", "--rollout-node", "r"]
        if mode == "disagg_async"
        else []
    )
    args = parser().parse_args(["--config", str(path), *selectors])
    command = launch_command(args, python="pinned-python")[0]
    assert command[2] == "espnet2.bin.rl_train"
    assert command[command.index("--worker") + 1] == ("ray" if selectors else "local")


def mock_python(tmp_path):
    path = tmp_path / "python-route"
    path.write_text(
        "#!" + sys.executable + "\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['ROUTE_LOG']).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if 'espnet2.s2t.rl.recipe' in sys.argv: print(os.environ['GENERATED_CONFIG'])\n"
    )
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("recipe", ["bagpiper", "bagpiper_tts"])
def test_bagpiper_recipe_stages_forward_dual_mode_parameters(tmp_path, recipe):
    command_python = mock_python(tmp_path)
    log = tmp_path / "route.jsonl"
    config = frozen_config(tmp_path, "disagg_async")
    env = {**os.environ, "ROUTE_LOG": str(log), "GENERATED_CONFIG": str(config)}
    args = [
        "bash",
        str(ROOT / "egs2" / recipe / "speechlm1/rl.sh"),
        "--stage",
        "2",
        "--stop-stage",
        "3",
        "--python",
        str(command_python),
        "--rl-config",
        str(config),
        "--manifest",
        "train manifest.jsonl",
        "--eval-manifest",
        "heldout.jsonl",
        "--data-admission",
        "audit.json",
        "--ray-address",
        "ray-address",
        "--train-node",
        "train-node",
        "--rollout-node",
        "rollout-node",
    ]
    subprocess.run(args, env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert all(call[:2] == ["-m", "espnet2.bin.rl_train"] for call in calls)
    assert "--validate-only" in calls[0] and "--validate-only" not in calls[1]
    assert all(
        call[call.index("--manifest") + 1] == "train manifest.jsonl" for call in calls
    )
    assert all(
        call[call.index("--rollout-node") + 1] == "rollout-node" for call in calls
    )


def test_owsm_and_librispeech_recipe_routes_keep_template_step_budget(tmp_path):
    command_python = mock_python(tmp_path)
    config = frozen_config(tmp_path, "disagg_async")
    for index, relative in enumerate(
        ("egs2/owsm_v4/s2t1/rl.sh", "egs2/librispeech/owsm_rl1/run.sh")
    ):
        log = tmp_path / f"route{index}.jsonl"
        env = {
            **os.environ,
            "PYTHON": str(command_python),
            "ROUTE_LOG": str(log),
            "GENERATED_CONFIG": str(config),
        }
        subprocess.run(
            [
                "bash",
                str(ROOT / relative),
                "--config",
                str(config),
                "--manifest",
                "train32.jsonl",
                "--eval-manifest",
                "heldout.jsonl",
                "--data-admission",
                "audit.json",
                "--train-config",
                "native.yaml",
                "--checkpoint",
                "native.pt",
                "--work-dir",
                str(tmp_path / f"work{index}"),
                "--ray-address",
                "ray-address",
                "--train-node",
                "train-node",
                "--rollout-node",
                "rollout-node",
            ],
            env=env,
            check=True,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert calls[0][:2] == ["-m", "espnet2.s2t.rl.recipe"]
        assert "--steps" not in calls[0]
        assert calls[1][:2] == ["-m", "espnet2.bin.rl_train"]
        assert calls[2][:2] == ["-m", "espnet2.bin.rl_train"]
        assert calls[2][calls[2].index("--rollout-node") + 1] == "rollout-node"


def test_owsm_recipe_builder_preserves_five_updates_and_existing_manifest(
    tmp_path, monkeypatch
):
    from espnet2.s2t.rl import recipe

    config = frozen_config(tmp_path, "sync")
    manifest = tmp_path / "train32.jsonl"
    manifest.write_text("already frozen data")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("already frozen heldout")
    observed = []
    monkeypatch.setattr(
        "espnet2.rl.data.verify_data_admission",
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


def test_opus_stage10_routes_before_unrelated_supervised_requirements(tmp_path):
    command_python = mock_python(tmp_path)
    config = frozen_config(tmp_path, "disagg_async")
    log = tmp_path / "opus_route.jsonl"
    env = {**os.environ, "ROUTE_LOG": str(log), "GENERATED_CONFIG": str(config)}
    subprocess.run(
        [
            "bash",
            "speechlm.sh",
            "--stage",
            "10",
            "--stop_stage",
            "10",
            "--python",
            str(command_python),
            "--rl_config",
            str(config),
            "--rl_manifest",
            "train32.jsonl",
            "--rl_eval_manifest",
            "heldout.jsonl",
            "--rl_data_admission",
            "audit.json",
            "--rl_ray_address",
            "ray-address",
            "--rl_train_node",
            "train-node",
            "--rl_rollout_node",
            "rollout-node",
            "--rl_validate_only",
            "true",
        ],
        cwd=ROOT / "egs2/libritts/speechlm1",
        env=env,
        check=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1 and calls[0][:2] == ["-m", "espnet2.bin.rl_train"]
    assert "--validate-only" in calls[0]
    assert calls[0][calls[0].index("--rollout-node") + 1] == "rollout-node"


@pytest.mark.parametrize(
    "path",
    sorted(ROOT.glob("egs2/*/*/conf/rl*.yaml")),
    ids=lambda path: str(path.relative_to(ROOT / "egs2")),
)
def test_every_shipped_recipe_names_an_existing_engine_and_reward(path):
    from espnet2.rl.rollout_worker import engine_class
    from espnet2.rl.utils import resolve_callable

    config = ESPnetRLConfig.load(path)
    engine_class(config)
    if config.reward_function:
        resolve_callable(config.reward_function)
