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
"${python_cmd}" -c 'from miles.backends.training_utils.loss.hub.math_utils import compute_policy_loss; from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout; print("Miles native RL library ready")'

