"""Every committed task recipe validates and resolves the reward it names."""

from pathlib import Path

import pytest
import yaml

from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.rewards import REWARDS, get_reward

ROOT = Path(__file__).resolve().parents[3]

# recipe -> (family, execution mode, reward name)
RECIPES = {
    "egs2/owsm_v4/s2t1/conf/rl_miles_longform_wer_1gpu.yaml": (
        "owsm",
        "sync",
        "longform_wer",
    ),
    "egs2/owsm_v4/s2t1/conf/rl_miles_longform_wer_2gpu_async.yaml": (
        "owsm",
        "disagg_async",
        "longform_wer",
    ),
    "egs2/libritts/speechlm1/conf/rl_opuslm_dialogue_speaker_id_1gpu.yaml": (
        "opuslm_dialogue",
        "sync",
        "speaker_id",
    ),
    "egs2/libritts/speechlm1/conf/rl_opuslm_dialogue_longform_wer_1gpu.yaml": (
        "opuslm_dialogue",
        "sync",
        "longform_wer",
    ),
    "egs2/libritts/speechlm1/conf/rl_opuslm_dialogue_mixed_1gpu.yaml": (
        "opuslm_dialogue",
        "sync",
        "mixed",
    ),
    "egs2/libritts/speechlm1/conf/rl_opuslm_longform_wer_2gpu_async.yaml": (
        "opuslm",
        "disagg_async",
        "longform_wer",
    ),
    "egs2/libritts/speechlm1/conf/rl_opuslm_speaker_id_2gpu_async.yaml": (
        "opuslm",
        "disagg_async",
        "speaker_id",
    ),
    "egs2/bagpiper/speechlm1/conf/advanced/rl_speaker_id.yaml": (
        "bagpiper",
        "sync",
        "speaker_id",
    ),
    "egs2/bagpiper/speechlm1/conf/advanced/rl_multispeaker_cpwer.yaml": (
        "bagpiper",
        "sync",
        "multispeaker_cpwer",
    ),
    "egs2/bagpiper/speechlm1/conf/advanced/rl_multispeaker_cpwer_shaped.yaml": (
        "bagpiper",
        "sync",
        "multispeaker_cpwer_shaped",
    ),
    "egs2/bagpiper/speechlm1/conf/advanced/rl_speaker_verification.yaml": (
        "bagpiper",
        "sync",
        "speaker_verification",
    ),
    "egs2/libritts/speechlm1/conf/rl_opuslm_dialogue_multistream_tts_1gpu.yaml": (
        "opuslm_dialogue",
        "sync",
        "codec_block",
    ),
}


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_recipe_validates_and_names_a_resolvable_task_reward(recipe):
    family, mode, name = RECIPES[recipe]
    config = ESPnetRLConfig.from_yaml(ROOT / recipe)
    assert (config.family, config.execution_mode) == (family, mode)
    assert config.reward_function == name
    reward = get_reward(config.reward_function)
    # Every task reward fails a manifest closed before models or Ray start.
    assert callable(getattr(reward, "validate_sources", None))


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_recipe_keeps_the_update_size_and_whole_support_sampling(recipe):
    config = ESPnetRLConfig.from_yaml(ROOT / recipe)
    assert config.prompts_per_step * config.group_size == 256
    assert (config.temperature, config.top_p, config.top_k) == (1.0, 1.0, -1)
    assert config.score_tolerance == 0.03
    assert config.dtype == "float32" or config.optimizer_device == "cpu"


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_recipe_declares_disjoint_or_shared_devices_as_its_mode_requires(recipe):
    config = ESPnetRLConfig.from_yaml(ROOT / recipe)
    if config.execution_mode == "sync":
        assert config.collocated, "a one-GPU profile shares its device"
    else:
        assert not config.collocated, "a disaggregated profile needs disjoint devices"
    assert config.train_ranks == 1


def test_one_gpu_engines_can_hold_every_running_request_in_their_kv_pool():
    """A declared context_length x max_running_requests must stay in the pool.

    The recipe comments carry the pool sizes these numbers were chosen
    against; this check only keeps the two engine settings consistent with
    the response budget, so raising max_tokens cannot silently exceed the
    context length the engine was sized for.
    """
    for recipe in sorted(RECIPES):
        config = ESPnetRLConfig.from_yaml(ROOT / recipe)
        settings = config.sglang_engine_kwargs
        if "context_length" not in settings:
            continue
        assert settings["context_length"] >= config.max_tokens, recipe
        assert settings["max_running_requests"] >= 1, recipe


def test_task_recipes_do_not_change_the_existing_profiles():
    """The committed ASR/QA recipes keep their own rewards.

    A new task is a new file. Repointing an existing recipe's reward would
    change a frozen contract, which the shared guide forbids.
    """
    unchanged = {
        "egs2/owsm_v4/s2t1/conf/rl_miles_wer_1gpu.yaml": None,
        "egs2/libritts/speechlm1/conf/rl_opuslm_librispeech_2gpu_async.yaml": (
            "wer"
        ),
        "egs2/libritts/speechlm1/conf/rl_opuslm_dialogue_librispeech_1gpu.yaml": (
            "wer"
        ),
        "egs2/bagpiper/speechlm1/conf/rl.yaml": "bagpiper_qa_v1",
        "egs2/bagpiper_tts/speechlm1/conf/rl.yaml": "bagpiper_tts_asr_wer",
    }
    for recipe, reward in unchanged.items():
        assert yaml.safe_load((ROOT / recipe).read_text())["reward_function"] == reward


def test_every_registered_reward_trains_in_a_committed_recipe():
    """A registered reward is one some committed recipe actually trains."""
    used = {
        yaml.safe_load(path.read_text()).get("reward_function") or "wer"
        for path in ROOT.glob("egs2/*/*/conf/**/rl*.yaml")
    }
    # The mixed recipe's manifests name asr_wer per record; no recipe names it.
    assert set(REWARDS) - {"asr_wer"} <= used
    assert "mixed" in used
