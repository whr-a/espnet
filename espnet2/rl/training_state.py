"""Native learner checkpoint validation and rank-local random-state recovery."""

import random

import numpy as np
import torch


def capture_rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        if "torch_cuda" not in state:
            raise ValueError("GPU resume requires this rank's CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def validate_resume_checkpoint(checkpoint, config, world_size, source_count=None):
    """Reject a changed objective, horizon, sampler, architecture or rank count."""
    state = checkpoint.get("trainer_state", {})
    step = checkpoint.get("step")
    if (
        state.get("version") != 1
        or not isinstance(step, int)
        or not 0 <= step <= config.steps
    ):
        raise ValueError("Resume requires a complete native trainer checkpoint")
    if state.get("completed_updates") != step or state.get("next_update") != step:
        raise ValueError("Checkpoint optimizer update and data cursor disagree")
    if state.get("consumed_prompts") != step * config.prompts_per_step:
        raise ValueError("Checkpoint prompt cursor differs from optimizer updates")
    expected_order = (
        "torch_randperm_seed_plus_epoch_v1"
        if config.shuffle_train_data
        else "cyclic_v1"
    )
    if (
        state.get("data_seed") != config.data_seed
        or state.get("data_order") != expected_order
    ):
        raise ValueError("Checkpoint data ordering differs from the configured sampler")
    if (
        state.get("world_size") != world_size
        or len(state.get("rank_rng_states", [])) != world_size
    ):
        raise ValueError("Resume requires the checkpoint's original learner rank count")
    if source_count is not None and state.get("source_count") != source_count:
        raise ValueError("Checkpoint source count differs from the training manifest")
    if (
        "scheduler" not in checkpoint
        or checkpoint["scheduler"].get("last_epoch") != step
    ):
        raise ValueError("Checkpoint scheduler and optimizer updates disagree")
    # Output locations and role-local device names may change during a clean
    # recovery; algorithm, initial reference and input locations remain fixed.
    allowed_changes = {"output_dir", "resume_checkpoint", "device", "reference_device"}
    current = config.to_dict()
    original = checkpoint.get("config", {})
    for name, value in current.items():
        if name not in allowed_changes and original.get(name) != value:
            raise ValueError(f"Resume changes frozen configuration: {name}")
    return state


def load_native_model_state(policy, state):
    """Load the same native key space emitted by checkpoint_state_dict()."""
    if set(state) == set(policy.state_dict()):
        policy.load_state_dict(state, strict=True)
    elif hasattr(policy, "model") and set(state) == set(policy.model.state_dict()):
        policy.model.load_state_dict(state, strict=True)
    else:
        raise ValueError("Checkpoint keys differ from the native policy architecture")
