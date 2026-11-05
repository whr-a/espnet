"""`python -m espnet2.bin.rl_train` must run, not import-and-exit.

A module that holds `main()` but no `__main__` guard imports and exits 0
without parsing arguments, validating anything or training.
`--validate-only` returns 0 the same way, which reads as "validation
passed" — the worst possible failure for a gate.
"""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("module", ["espnet2.bin.rl_train"])
def test_running_either_module_reaches_argument_parsing(module, monkeypatch):
    """A missing config must be reported, not silently succeed."""
    monkeypatch.setattr(sys, "argv", [module])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(module, run_name="__main__")
    # argparse exits 2 on a missing required argument; 0 would mean the module
    # body ran and fell off the end without doing anything.
    assert exit_info.value.code == 2, f"{module} exited {exit_info.value.code}"


@pytest.mark.execution_timeout(20)
@pytest.mark.parametrize("module", ["espnet2.bin.rl_train"])
def test_a_nonexistent_config_is_never_a_silent_success(module):
    """End to end, in a real subprocess: exit 0 here would be a false green."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--config",
            "/nonexistent/does-not-exist.yaml",
            "--validate-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode != 0, (
        f"{module} --validate-only exited 0 on a missing config; that reads as "
        f"'validation passed'. stdout={done.stdout[-400:]!r}"
    )
