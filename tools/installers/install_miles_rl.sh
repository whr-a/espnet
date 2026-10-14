#!/usr/bin/env bash
# Install the reviewed Miles library without modifying its repository.
set -euo pipefail
python_cmd=${PYTHON:-python}
source_dir=${MILES_SOURCE:-$(pwd)/miles-rl-library}
revision=a22fd824191c64ba7c5d6c531f125406e4ce9cc6
if ! git -C "${source_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git clone --filter=blob:none --no-checkout https://github.com/radixark/miles.git "${source_dir}"
    git -C "${source_dir}" checkout --detach "${revision}"
fi
[[ $(git -C "${source_dir}" rev-parse HEAD) == "${revision}" ]] || {
    echo "Miles checkout differs from the reviewed pin: ${source_dir}" >&2; exit 1;
}
"${python_cmd}" -m pip install numpy PyYAML soundfile scipy safetensors 'ray[default]' msgpack pybase64
"${python_cmd}" -m pip install --no-deps --no-build-isolation -e "${source_dir}"
"${python_cmd}" - "${source_dir}" "${revision}" <<'PYCODE'
import hashlib
import subprocess
import sys
from pathlib import Path
from miles.backends.training_utils.loss.hub import math_utils
from miles.ray.rollout import train_data_conversion

root = Path(sys.argv[1]).resolve()
revision = sys.argv[2]
for module, relative in (
    (math_utils, "miles/backends/training_utils/loss/hub/math_utils.py"),
    (train_data_conversion, "miles/ray/rollout/train_data_conversion.py"),
):
    actual = Path(module.__file__).resolve()
    expected = root / relative
    if actual != expected:
        raise RuntimeError(f"Miles import is shadowed: {actual}, expected {expected}")
    original = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{relative}"])
    if original != actual.read_bytes():
        raise RuntimeError(f"Pinned Miles algorithm changed: {actual}")
    print(relative, hashlib.sha256(original).hexdigest())
print("Miles native RL library ready", root, revision)
PYCODE

