"""Bind native ESPnet components to the unchanged pinned Miles async loop.

Miles currently exposes orchestration factories as module globals. A private
function namespace supplies ESPnet implementations without changing the module,
its function, or any Miles source. The exact loop code and its safe-point weight
update ordering remain Miles' implementation. This boundary is pinned and audited.
"""

import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import FunctionType, SimpleNamespace

from .miles_library import PIN, verify_miles


def load_async_loop(config, components):
    evidence = verify_miles(config)
    root = Path(evidence["source"])
    path = root / "train_async.py"
    original = subprocess.run(
        ["git", "-C", str(root), "show", f"{PIN}:train_async.py"],
        check=True,
        capture_output=True,
    ).stdout
    if path.read_bytes() != original:
        raise ValueError("Pinned Miles async driver has local changes")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("The Miles runtime requires an unchanged tracked source tree")
    spec = importlib.util.spec_from_file_location("espnet_pinned_miles_async", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factories = {
        "init_orchestration_script": components.initialize,
        "create_rollout_components": components.create_rollout_components,
        "create_training_models": components.create_training_models,
        "update_weights": components.update_weights,
    }
    namespace = {**module.train.__globals__, **factories}
    driver = FunctionType(
        module.train.__code__,
        namespace,
        module.train.__name__,
        module.train.__defaults__,
        module.train.__closure__,
    )
    driver.__kwdefaults__ = module.train.__kwdefaults__
    evidence["runtime"] = {
        "entrypoint": "train_async.py:train",
        "sha256": hashlib.sha256(original).hexdigest(),
        "bindings": sorted(factories),
        "source_modified": False,
        "original_function_modified": False,
        "same_code_object": driver.__code__ is module.train.__code__,
    }
    return driver, evidence


def orchestration_args(config):
    """The first profile has no critic, in-training eval, or recovery."""
    config.validate()
    if config.execution_mode != "disagg_async":
        raise ValueError("Miles async orchestration requires disagg_async")
    return SimpleNamespace(
        colocate=False,
        use_critic=False,
        use_rollout_logprobs=False,
        use_tis=True,
        keep_old_actor=True,
        fully_async=False,
        num_rollout=config.steps,
        start_rollout_id=0,
        update_weights_interval=config.update_weights_interval,
        offload_train=False,
        num_critic_only_steps=0,
        check_weight_update_equal=False,
        eval_interval=None,
        skip_eval_before_train=True,
        hf_checkpoint=None,
        save_trigger_sentinel=None,
        save_interval=config.save_interval,
        api_server_port=None,
        mini_ft_controller_enable=False,
        debug_exit_after_rollout=None,
        eval_uses_snapshots=False,
    )
