#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
stage=1
stop_stage=3
rl_config=conf/rl.yaml
manifest=data/rl/train.jsonl
dataset=avqa
data_input=
wav_scp=
text=
questions=
choices=
forbidden=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) stage=$2 ;;
        --stop-stage) stop_stage=$2 ;;
        --rl-config) rl_config=$2 ;;
        --manifest) manifest=$2 ;;
        --data-input) data_input=$2 ;;
        --wav-scp) wav_scp=$2 ;;
        --text) text=$2 ;;
        --questions) questions=$2 ;;
        --choices) choices=$2 ;;
        *) echo "Unknown RL option $1" >&2; exit 2 ;;
    esac
    shift 2
done
if (( stage <= 1 && stop_stage >= 1 )); then
    args=(--dataset "$dataset" --output "$manifest" --task qa)
    if [[ -n "$wav_scp" ]]; then
        args=(--dataset wav_scp --wav-scp "$wav_scp" --text "$text" --output "$manifest" --task qa)
        [[ -z "$questions" ]] || args+=(--questions "$questions")
        [[ -z "$choices" ]] || args+=(--choices "$choices")
    else
        args+=(--input "$data_input")
    fi
    python -m espnet2.speechlm.model.speechlm.rl.bagpiper.data "${args[@]}"
fi
if (( stage <= 2 && stop_stage >= 2 )); then
    python -m espnet2.bin.rl_train --config "$rl_config" --validate-only
fi
if (( stage <= 3 && stop_stage >= 3 )); then
    python -m espnet2.bin.rl_train --config "$rl_config"
fi
