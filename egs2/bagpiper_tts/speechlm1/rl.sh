#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
espnet_root=$(cd -- "${recipe_dir}/../../.." && pwd)
export PYTHONPATH="${espnet_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${recipe_dir}"
python_cmd=${PYTHON:-python}
stage=1
stop_stage=3
rl_config=conf/rl.yaml
manifest=
reuse_manifest=false
dataset=libritts
data_input=
wav_scp=
text=
questions=
choices=
hf_revision=
limit=
eval_manifest=
data_admission=
ray_address=
train_node=
rollout_node=
output_dir=
checkpoint=
train_config=
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
    case "$1" in
        --stage) stage=$2 ;;
        --stop-stage) stop_stage=$2 ;;
        --rl-config) rl_config=$2 ;;
        --manifest) manifest=$2; reuse_manifest=true ;;
        --data-input) data_input=$2 ;;
        --wav-scp) wav_scp=$2 ;;
        --text) text=$2 ;;
        --questions) questions=$2 ;;
        --choices) choices=$2 ;;
        --hf-revision) hf_revision=$2 ;;
        --limit) limit=$2 ;;
        --python) python_cmd=$2 ;;
        --eval-manifest) eval_manifest=$2 ;;
        --data-admission) data_admission=$2 ;;
        --ray-address) ray_address=$2 ;;
        --train-node) train_node=$2 ;;
        --rollout-node) rollout_node=$2 ;;
        --output-dir) output_dir=$2 ;;
        --checkpoint) checkpoint=$2 ;;
        --train-config) train_config=$2 ;;
        *) echo "Unknown RL option $1" >&2; exit 2 ;;
    esac
    shift 2
done
if [[ -z "$manifest" ]]; then
    manifest=$("${python_cmd}" - "$rl_config" <<'PYCONFIG'
import sys
import yaml
with open(sys.argv[1]) as stream:
    print(yaml.safe_load(stream)["manifest"])
PYCONFIG
)
fi
if (( stage <= 1 && stop_stage >= 1 )); then
    if [[ "$reuse_manifest" == true && -z "$wav_scp" && -z "$data_input" ]]; then
        [[ -f "$manifest" ]] || { echo "Prepared manifest does not exist: $manifest" >&2; exit 2; }
    else
        args=(--dataset "$dataset" --output "$manifest" --task tts_asr_wer)
        if [[ -n "$wav_scp" ]]; then
            args=(--dataset wav_scp --wav-scp "$wav_scp" --text "$text" --output "$manifest" --task tts_asr_wer)
            [[ -z "$questions" ]] || args+=(--questions "$questions")
            [[ -z "$choices" ]] || args+=(--choices "$choices")
        else
            [[ -n "$data_input" ]] || { echo "--data-input or --wav-scp is required" >&2; exit 2; }
            args+=(--input "$data_input")
        fi
        [[ -z "$hf_revision" ]] || args+=(--hf-revision "$hf_revision")
        [[ -z "$limit" ]] || args+=(--limit "$limit")
        "${python_cmd}" -m espnet2.speechlm.model.speechlm.rl.bagpiper.data "${args[@]}"
    fi
fi
launch_args=(--config "$rl_config" --manifest "$manifest")
for key in eval_manifest data_admission ray_address train_node rollout_node output_dir checkpoint train_config; do
    [[ -z ${!key} ]] || launch_args+=(--${key//_/-} "${!key}")
done
if (( stage <= 2 && stop_stage >= 2 )); then
    "${python_cmd}" -m espnet2.bin.rl_train "${launch_args[@]}" --validate-only
fi
if (( stage <= 3 && stop_stage >= 3 )); then
    "${python_cmd}" -m espnet2.bin.rl_train "${launch_args[@]}"
fi
