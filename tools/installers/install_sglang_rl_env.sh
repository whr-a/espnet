#!/usr/bin/env bash
# Create the reviewed SGLang Python environment for ESPnet RL.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
espnet_source=$(cd -- "${script_dir}/../.." && pwd)
sglang_source=
sglang_revision=4c3f163fe54634f20ef846e1ec0c507b27ae2430
python_cmd=${PYTHON:-python3}
requirements=${script_dir}/sglang_rl/cp312_linux_x86_64_cu130/requirements.txt
profile=${script_dir}/sglang_rl/cp312_linux_x86_64_cu130/profile.json
venv_dir=

usage() {
    cat <<'HELP'
Usage: install_sglang_rl_env.sh --venv NEW_PATH --sglang-source CHECKOUT [options]
  --espnet-source CHECKOUT  ESPnet source (default: this checkout)
  --sglang-revision COMMIT  Expected SGLang revision (default: reviewed pin)
  --python EXECUTABLE      CPython 3.12 executable (default: PYTHON or python3)
  --requirements FILE      Reviewed hash-locked dependencies (default: cu130 profile)

Creates a new venv on Linux x86_64, installs hash-locked dependencies and source
editables, and checks CPU imports and package requirements. The default
SGLANG_BUILD_RUST_EXTS=none selects the validated Python Engine profile;
set another documented build mode externally when a Rust toolchain is available.
HELP
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h) usage; exit 0 ;;
        --venv|--sglang-source|--espnet-source|--sglang-revision|--python|--requirements)
            [[ $# -ge 2 && -n $2 && $2 != --* ]] || { usage >&2; exit 2; }
            case $1 in
                --venv) venv_dir=$2 ;;
                --sglang-source) sglang_source=$2 ;;
                --espnet-source) espnet_source=$2 ;;
                --sglang-revision) sglang_revision=$2 ;;
                --python) python_cmd=$2 ;;
                --requirements) requirements=$2 ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n ${venv_dir} && -n ${sglang_source} ]] || { usage >&2; exit 2; }
[[ ! -e ${venv_dir} && ! -L ${venv_dir} ]] || {
    echo "Refusing existing environment path: ${venv_dir}" >&2; exit 1;
}
[[ -f ${profile} && -f ${requirements} && -f ${espnet_source}/pyproject.toml && -f ${sglang_source}/python/pyproject.toml ]] || {
    echo "Profile metadata, requirements file or source checkout is missing" >&2; exit 1;
}
espnet_source=$(cd -- "${espnet_source}" && pwd)
sglang_source=$(cd -- "${sglang_source}" && pwd)
requirements=$("${python_cmd}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${requirements}")
"${python_cmd}" - <<'PYCODE'
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("This profile requires CPython 3.12 on Linux x86_64")
PYCODE
[[ $(git -C "${sglang_source}" rev-parse HEAD) == "${sglang_revision}" ]] || {
    echo "SGLang source differs from the requested revision" >&2; exit 1;
}
"${python_cmd}" - "${profile}" "${requirements}" "${espnet_source}" "${sglang_source}" "${sglang_revision}" <<'PYCODE'
import hashlib
import json
import sys
import tomllib
from pathlib import Path

profile_path, requirements, espnet_source, sglang_source = map(Path, sys.argv[1:5])
sglang_revision = sys.argv[5]
profile = json.loads(profile_path.read_text())
if requirements.resolve() == (profile_path.parent / "requirements.txt").resolve():
    if sglang_revision != profile["source_pins"]["sglang"]:
        raise SystemExit("Default dependency profile requires its validated SGLang revision")
    if hashlib.sha256(requirements.read_bytes()).hexdigest() != profile["requirements_sha256"]:
        raise SystemExit("Default dependency lock changed without profile review")
    for name, source in (("espnet", espnet_source / "pyproject.toml"), ("sglang", sglang_source / "python/pyproject.toml")):
        dependencies = tomllib.loads(source.read_text())["project"]["dependencies"]
        if sorted(dependencies) != sorted(profile["dependency_metadata"][name]):
            raise SystemExit(f"{name} dependency metadata differs from the reviewed profile")
PYCODE
export SGLANG_BUILD_RUST_EXTS=${SGLANG_BUILD_RUST_EXTS:-none}
if [[ ${SGLANG_BUILD_RUST_EXTS} != none ]] && ! command -v cargo >/dev/null; then
    echo "Rust build mode needs cargo; select SGLANG_BUILD_RUST_EXTS=none for the Python profile" >&2; exit 1;
fi
echo "SGLang revision: ${sglang_revision}; Rust build mode: ${SGLANG_BUILD_RUST_EXTS}"

# Keep package resolution and validation independent of an ambient Python stack.
unset PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
unset PIP_FIND_LINKS PIP_NO_INDEX PIP_REQUIREMENT PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
unset PIP_PREFIX PIP_TARGET PIP_USER PIP_CERT PIP_CLIENT_CERT
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=
"${python_cmd}" -m venv "${venv_dir}"
venv_dir=$(cd -- "${venv_dir}" && pwd)
python_cmd=${venv_dir}/bin/python
"${python_cmd}" -m pip install --only-binary=:all: --index-url https://pypi.org/simple pip==26.1.1
"${python_cmd}" -m pip install --no-deps --require-hashes --only-binary=:all: \
    --no-binary=antlr4-python3-runtime --index-url https://pypi.org/simple -r "${requirements}"
"${python_cmd}" -m pip install --no-deps --index-url https://pypi.org/simple -e "${espnet_source}"
"${python_cmd}" -m pip install --no-deps --index-url https://pypi.org/simple -e "${sglang_source}/python"
"${python_cmd}" -m pip check
"${python_cmd}" -I - "${espnet_source}" "${sglang_source}" <<'PYCODE'
import importlib
import inspect
import sys
from pathlib import Path

import espnet2
import sglang
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.models.registry import ModelRegistry

for module, source in ((espnet2, sys.argv[1]), (sglang, sys.argv[2])):
    if not Path(module.__file__).resolve().is_relative_to(Path(source).resolve()):
        raise RuntimeError(f"Source import is shadowed: {module.__file__}")
for name, module_name in (
    ("ESPnetOWSMForCausalLM", "espnet2.s2t.rl.sglang_model"),
    ("ESPnetBagpiperForCausalLM", "espnet2.speechlm.model.speechlm.rl.bagpiper.sglang_model"),
    ("OpusLMForConditionalGeneration", "espnet2.speechlm.rl.sglang_model"),
    ("OpusLMDialogueForConditionalGeneration", "espnet2.speechlm.rl.sglang_model"),
):
    expected = getattr(importlib.import_module(module_name), name)
    actual, architecture = ModelRegistry.resolve_model_cls([name])
    if actual is not expected or architecture != name:
        raise RuntimeError(f"Architecture registration differs: {name}")
    print(name, inspect.getfile(actual))
print("SGLang Python Engine import ready", inspect.getfile(Engine))
PYCODE
echo "SGLang RL CPU environment ready: ${venv_dir}"
