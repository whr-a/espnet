"""Real bounded CPU TCPStore oracle for independent torchrun child rollouts."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from espnet2.rl.rollout import CheckpointRollout

TRAIN_IDENTITY = {
    "RANK": "0",
    "LOCAL_RANK": "0",
    "WORLD_SIZE": "4",
    "LOCAL_WORLD_SIZE": "4",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "19530",
    "GROUP_RANK": "0",
    "ROLE_RANK": "0",
    "ROLE_WORLD_SIZE": "4",
    "TORCHELASTIC_USE_AGENT_STORE": "True",
    "TORCHELASTIC_RUN_ID": "none",
    "TORCHELASTIC_RESTART_COUNT": "0",
    "TORCHELASTIC_MAX_RESTARTS": "0",
    "TORCHELASTIC_ERROR_FILE": "/training/owned-error.json",
    "TORCHELASTIC_FUTURE_IDENTITY": "remove-by-prefix",
}

STORE_ORACLE = r"""
from datetime import timedelta
import importlib,json,sys,torch
rdzv=importlib.import_module('torch.distributed.rendezvous')
store=rdzv._create_c10d_store('127.0.0.1',int(sys.argv[1]),rank=0,world_size=1,timeout=timedelta(seconds=1))
store.set('independent-rollout','server-ready')
assert store.get('independent-rollout')==b'server-ready'
print(json.dumps({'rank':0,'world_size':1,'server_ready':True,'agent_store':rdzv._torchelastic_use_agent_store(),'cuda_initialized':torch.cuda.is_initialized()}))
"""


def config(root):
    return SimpleNamespace(
        output_dir=str(root),
        rollout_backend="sglang",
        sglang_source="",
        sglang_engine_kwargs={},
        vllm_source="",
        miles_source="",
        rollout_gpus="7",
        dtype="float32",
    )


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.parametrize(
    "backend,memory_saver", [("sglang", True), ("sglang", False), ("vllm", True)]
)
def test_allocator_configuration_belongs_to_the_learner_with_memory_saver(
    tmp_path, monkeypatch, backend, memory_saver
):
    aliases = {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    }
    for name, value in aliases.items():
        monkeypatch.setenv(name, value)
    settings = config(tmp_path)
    settings.rollout_backend = backend
    settings.sglang_engine_kwargs = {"enable_memory_saver": memory_saver}
    parent = dict(os.environ)
    child = CheckpointRollout(settings)._environment()
    assert dict(os.environ) == parent
    for name, value in aliases.items():
        if backend == "sglang" and memory_saver:
            assert name not in child
        else:
            assert child[name] == value


class RolloutEnvironment(unittest.TestCase):
    def test_removes_complete_torchrun_identity_preserving_other_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            inherited = {
                **TRAIN_IDENTITY,
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "OMP_NUM_THREADS": "1",
                "TASK_MARKER": "preserve",
            }
            with patch.dict(os.environ, inherited, clear=False):
                before = dict(os.environ)
                child = CheckpointRollout(config(temporary))._environment()
                self.assertEqual(dict(os.environ), before)
            self.assertTrue(all(key not in child for key in TRAIN_IDENTITY))
            self.assertFalse(any(key.startswith("TORCHELASTIC_") for key in child))
            self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual(child["OMP_NUM_THREADS"], "1")
            self.assertEqual(child["TASK_MARKER"], "preserve")

    def test_actual_cpu_tcpstore_agent_client_fails_but_clean_rank_zero_starts(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, TRAIN_IDENTITY, clear=False),
        ):
            inherited = dict(os.environ)
            inherited["CUDA_VISIBLE_DEVICES"] = ""
            failed = subprocess.run(
                [sys.executable, "-c", STORE_ORACLE, str(unused_port())],
                env=inherited,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("timed out", failed.stderr.lower())
            child = CheckpointRollout(config(temporary))._environment()
            child["CUDA_VISIBLE_DEVICES"] = ""
            ready = subprocess.run(
                [sys.executable, "-c", STORE_ORACLE, str(unused_port())],
                env=child,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(ready.returncode, 0, ready.stderr)
            result = json.loads(ready.stdout)
            self.assertTrue(result["server_ready"])
            self.assertFalse(result["agent_store"])
            self.assertFalse(result["cuda_initialized"])


if __name__ == "__main__":
    unittest.main()
