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
"${python_cmd}" -m pip install "${source_dir}/python"
"${python_cmd}" -c 'from sglang.srt.entrypoints.engine import Engine; print("SGLang inference engine ready", Engine)'

