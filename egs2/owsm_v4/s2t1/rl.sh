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
steps=
device=
reference_device=
rollout_gpus=
train_gpus=
rollout_python=
sglang_source=
miles_source=
language=
manifest=
eval_manifest=
data_admission=
ray_address=
train_node=
rollout_node=
validate_only=false
while (($#)); do
    if [[ $1 == --validate-only ]]; then validate_only=true; shift; continue; fi
    [[ $# -ge 2 && $2 != --* ]] || { echo "$1 requires a value" >&2; exit 2; }
    key=${1#--}; key=${key//-/_}
    case ${key} in
        stage|stop_stage) shift 2 ;;
        wav_scp|text|train_config|checkpoint|config|work_dir|steps|device|reference_device|train_gpus|rollout_gpus|rollout_python|sglang_source|miles_source|language|manifest|eval_manifest|data_admission|ray_address|train_node|rollout_node)
            printf -v "${key}" '%s' "$2"; shift 2 ;;
        *) echo "Unknown RL option: $1" >&2; exit 2 ;;
    esac
done
for key in train_config checkpoint; do
    [[ -n ${!key} ]] || { echo "--${key//_/-} is required" >&2; exit 2; }
done
[[ -n ${manifest} || ( -n ${wav_scp} && -n ${text} ) ]] || {
    echo "Provide --manifest or both --wav-scp and --text" >&2; exit 2;
}
prepare_args=(--train-config "${train_config}" --checkpoint "${checkpoint}" --config "${config}" --work-dir "${work_dir}")
for key in wav_scp text manifest eval_manifest data_admission steps device reference_device train_gpus rollout_gpus rollout_python sglang_source miles_source language; do
    [[ -z ${!key} ]] || prepare_args+=(--${key//_/-} "${!key}")
done
config_path=$("${python_cmd}" -m espnet2.s2t.rl.recipe "${prepare_args[@]}")
launch_args=(--config "${config_path}")
for key in data_admission ray_address train_node rollout_node; do
    [[ -z ${!key} ]] || launch_args+=(--${key//_/-} "${!key}")
done
"${python_cmd}" -m espnet2.bin.rl_train "${launch_args[@]}" --validate-only
if ! ${validate_only}; then
    exec "${python_cmd}" -m espnet2.bin.rl_train "${launch_args[@]}"
fi
