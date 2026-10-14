#!/usr/bin/env bash
# Stage 18: optional Miles WER post-training of an existing OWSM checkpoint.
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
espnet_root=$(cd -- "${recipe_dir}/../../.." && pwd)
export PYTHONPATH="${espnet_root}${PYTHONPATH:+:${PYTHONPATH}}"
python_cmd=${PYTHON:-python}
config=${recipe_dir}/conf/rl_miles_wer.yaml
work_dir=exp/rl_miles_wer
wav_scp=
text=
train_config=
checkpoint=
steps=100
device=cuda:0
reference_device=cuda:1
rollout_gpus=2
rollout_python=${python_cmd}
sglang_source=
miles_source=
language='<eng>'
while (($#)); do
    key=${1#--}; key=${key//-/_}
    case ${key} in
        stage|stop_stage) shift 2 ;;
        wav_scp|text|train_config|checkpoint|config|work_dir|steps|device|reference_device|rollout_gpus|rollout_python|sglang_source|miles_source|language)
            printf -v "${key}" '%s' "$2"; shift 2 ;;
        *) echo "Unknown RL option: $1" >&2; exit 2 ;;
    esac
done
for key in wav_scp text train_config checkpoint; do
    [[ -n ${!key} ]] || { echo "--${key//_/-} is required" >&2; exit 2; }
done
config_path=$("${python_cmd}" -m espnet2.s2t.rl.recipe \
    --wav-scp "${wav_scp}" --text "${text}" \
    --train-config "${train_config}" --checkpoint "${checkpoint}" \
    --config "${config}" --work-dir "${work_dir}" --steps "${steps}" \
    --device "${device}" --reference-device "${reference_device}" \
    --rollout-gpus "${rollout_gpus}" --rollout-python "${rollout_python}" \
    --sglang-source "${sglang_source}" --miles-source "${miles_source}" --language "${language}")
"${python_cmd}" -m espnet2.bin.rl_train --config "${config_path}" --validate-only
exec "${python_cmd}" -m espnet2.bin.rl_train --config "${config_path}"
