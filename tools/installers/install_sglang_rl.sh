#!/usr/bin/env bash
# Build a pinned SGLang fork in a dedicated inference environment.
set -euo pipefail
python_cmd=${PYTHON:-python}
source_dir=${SGLANG_SOURCE:-}
revision=${SGLANG_REVISION:-}
if [[ -z ${source_dir} || -z ${revision} ]]; then
    echo "Set SGLANG_SOURCE and the reviewed SGLANG_REVISION from the recipe README" >&2
    exit 2
fi
[[ $(git -C "${source_dir}" rev-parse HEAD) == "${revision}" ]] || {
    echo "SGLang checkout differs from the requested pin" >&2; exit 1;
}
"${python_cmd}" -m pip install -e "${source_dir}/python"
"${python_cmd}" - "${source_dir}" <<'PYCODE'
import subprocess
import sys
from pathlib import Path
import sglang
from sglang.srt.entrypoints.engine import Engine

root = Path(sys.argv[1]).resolve()
actual = Path(sglang.__file__).resolve()
expected = root / "python/sglang/__init__.py"
if actual != expected:
    raise RuntimeError(f"SGLang import is shadowed: {actual}, expected {expected}")
for relative in (
    "python/sglang/srt/entrypoints/engine.py",
    "python/sglang/srt/model_executor/model_runner.py",
    "python/sglang/srt/models/espnet_owsm.py",
    "python/sglang/srt/models/espnet_bagpiper.py",
    "python/sglang/srt/models/espnet_opuslm.py",
):
    source = root / relative
    original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"])
    if original != source.read_bytes():
        raise RuntimeError(f"Pinned SGLang source changed: {source}")
print("SGLang inference engine ready", Engine, root)
PYCODE
