#!/usr/bin/env bash
# Install a reviewed native ESPnet/Miles asynchronous source environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile_dir=${script_dir}/miles_async/cp312_linux_x86_64_cu130
espnet_source=$(cd -- "${script_dir}/../.." && pwd)
sglang_source=
miles_source=
venv_dir=
python_cmd=${PYTHON:-python3}
with_tests=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv|--sglang-source|--miles-source|--espnet-source|--python)
            [[ $# -ge 2 && -n $2 && $2 != --* ]] || exit 2
            case $1 in
                --venv) venv_dir=$2 ;;
                --sglang-source) sglang_source=$2 ;;
                --miles-source) miles_source=$2 ;;
                --espnet-source) espnet_source=$2 ;;
                --python) python_cmd=$2 ;;
            esac
            shift 2 ;;
        --with-tests) with_tests=true; shift ;;
        --help|-h)
            echo 'Usage: install_miles_async_env.sh --venv NEW_PATH --sglang-source CHECKOUT --miles-source CHECKOUT [--espnet-source CHECKOUT] [--python PYTHON3.12] [--with-tests]'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n ${venv_dir} && -n ${sglang_source} && -n ${miles_source} ]] || exit 2
[[ ! -e ${venv_dir} && ! -L ${venv_dir} ]] || { echo 'Refusing an existing environment' >&2; exit 1; }
espnet_source=$(cd -- "${espnet_source}" && pwd)
sglang_source=$(cd -- "${sglang_source}" && pwd)
miles_source=$(cd -- "${miles_source}" && pwd)
"${python_cmd}" - "${profile_dir}" "${espnet_source}" "${sglang_source}" "${miles_source}" <<'PY'
import hashlib,json,platform,subprocess,sys
from pathlib import Path
profile_dir,espnet,sglang,miles=map(Path,sys.argv[1:])
profile=json.loads((profile_dir/'profile.json').read_text())
if sys.version_info[:2]!=(3,12) or platform.system()!='Linux' or platform.machine()!='x86_64':
    raise SystemExit('Requires CPython3.12 on Linux x86_64')
libc,version=platform.libc_ver()
if libc!='glibc' or tuple(map(int,version.split('.')[:2]))<(2,39):
    raise SystemExit('Official Miles router wheel requires glibc2.39 or later')
for name in ('requirements','test_requirements'):
    if hashlib.sha256((profile_dir/(name+'.txt')).read_bytes()).hexdigest()!=profile[name+'_sha256']:
        raise SystemExit('Reviewed dependency lock changed: '+name)
for name,path in [('espnet',espnet),('sglang',sglang),('miles',miles)]:
    head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if name in profile['source_pins'] and head!=profile['source_pins'][name]:
        raise SystemExit('Unexpected source pin: '+name)
    if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():
        raise SystemExit('Source checkout is not clean: '+name)
PY
unset PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
unset PIP_FIND_LINKS PIP_NO_INDEX PIP_REQUIREMENT PIP_CONSTRAINT PIP_BUILD_CONSTRAINT
unset PIP_PREFIX PIP_TARGET PIP_USER PIP_CERT PIP_CLIENT_CERT
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=
export SGLANG_BUILD_RUST_EXTS=none
"${python_cmd}" -m venv "${venv_dir}"
venv_dir=$(cd -- "${venv_dir}" && pwd)
python_cmd=${venv_dir}/bin/python
"${python_cmd}" -m pip install --no-deps --require-hashes --only-binary=:all: \
    --no-binary=antlr4-python3-runtime,pylatexenc --index-url https://pypi.org/simple \
    -r "${profile_dir}/requirements.txt"
if [[ ${with_tests} == true ]]; then
    "${python_cmd}" -m pip install --no-deps --require-hashes --only-binary=:all: \
        --index-url https://pypi.org/simple -r "${profile_dir}/test_requirements.txt"
fi
"${python_cmd}" -m pip install --no-deps --index-url https://pypi.org/simple \
    -e "${espnet_source}" -e "${sglang_source}/python"
"${python_cmd}" -m pip check
PYTHONPATH="${miles_source}" "${python_cmd}" \
    "${script_dir}/verify_miles_async_env.py" --profile-dir "${profile_dir}" \
    --espnet-source "${espnet_source}" --sglang-source "${sglang_source}" \
    --miles-source "${miles_source}" --output "${venv_dir}.acceptance.json"
echo "Native Miles async environment ready: ${venv_dir}"
echo "Use config.miles_source=${miles_source}; GPU execution requires its own acceptance."
