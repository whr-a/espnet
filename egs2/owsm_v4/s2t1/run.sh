#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Keep RL options separate from the ordinary S2T recipe's options.
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stage=1
stop_stage=10000
stop_stage_explicit=false
rl_requested=false
help_requested=false
s2t_args=()
rl_args=()
standalone_rl_args=()

die() { echo "$0: $*" >&2; exit 2; }
require_value() {
    [[ $# -ge 2 && $2 != --* ]] || die "$1 requires a value"
}
while (($#)); do
    case $1 in
        --stage|--stop-stage|--stop_stage)
            require_value "$@"
            [[ $2 =~ ^[0-9]+$ ]] || die "$1 requires a nonnegative integer"
            [[ ${#2} -le 9 ]] || die "$1 must use at most 9 digits"
            if [[ $1 == --stage ]]; then
                stage=$((10#$2))
            else
                stop_stage=$((10#$2))
                stop_stage_explicit=true
            fi
            s2t_args+=("$1" "$2")
            shift 2 ;;
        --rl-*|--rl_*)
            key=${1#--rl}; key=${key#[-_]}; key=${key//-/_}
            case ${key} in
                wav_scp|text|train_config|checkpoint|config|work_dir|steps|device|reference_device|rollout_gpus|rollout_python|sglang_source|miles_source|language|manifest|eval_manifest|data_admission|ray_address|train_node|rollout_node) ;;
                *) die "Unknown RL option: $1" ;;
            esac
            require_value "$@"
            rl_args+=("--${key//_/-}" "$2")
            standalone_rl_args+=("--${key//_/-}" "$2")
            rl_requested=true
            shift 2 ;;
        --help|-h)
            help_requested=true
            s2t_args+=("$1")
            standalone_rl_args+=("$1")
            shift ;;
        --*)
            # Preserve each ordinary option and its value without parsing it.
            s2t_args+=("$1")
            standalone_rl_args+=("$1")
            shift
            if (($#)); then
                s2t_args+=("$1")
                standalone_rl_args+=("$1")
                shift
            fi ;;
        *)
            s2t_args+=("$1")
            standalone_rl_args+=("$1")
            shift ;;
    esac
done

validate_rl_args() {
    local key option wav_scp= text= train_config= checkpoint= manifest=
    local config=${recipe_dir}/conf/rl_miles_wer.yaml
    while (($#)); do
        option=$1
        if [[ ${option} == --validate-only ]]; then shift; continue; fi
        key=${option#--}; key=${key//-/_}
        case ${key} in
            wav_scp|text|train_config|checkpoint|config|work_dir|steps|device|reference_device|rollout_gpus|rollout_python|sglang_source|miles_source|language|manifest|eval_manifest|data_admission|ray_address|train_node|rollout_node) ;;
            *) die "Unknown RL option: ${option}" ;;
        esac
        require_value "$@"
        case ${key} in
            wav_scp|text|train_config|checkpoint|config|manifest) printf -v "${key}" '%s' "$2" ;;
            steps) [[ $2 =~ ^[1-9][0-9]*$ ]] || die "${option} requires a positive integer" ;;
        esac
        shift 2
    done
    for key in train_config checkpoint; do
        [[ -n ${!key} ]] || die "RL requires --${key//_/-} (or --rl-${key//_/-} for a combined run)"
    done
    [[ -n ${manifest} || ( -n ${wav_scp} && -n ${text} ) ]] || die "RL requires --manifest or both --wav-scp and --text"
    [[ -f ${config} && -r ${config} ]] || die "RL config is not a readable file: ${config}"
}

run_rl=false
if ! ${help_requested}; then
    if (( stage == 18 )); then
        if (( stop_stage < 18 )); then
            echo "$0: Stage range excludes 18; RL was not executed." >&2
            exit 0
        fi
        validate_rl_args "${standalone_rl_args[@]}"
        exec "${recipe_dir}/rl.sh" "${standalone_rl_args[@]}"
    fi
    if ${rl_requested}; then
        ${stop_stage_explicit} || die "A combined RL run requires explicit --stop_stage or --stop-stage including stage 18"
        if (( stage <= 18 && stop_stage >= 18 )); then
            validate_rl_args "${rl_args[@]}"
            run_rl=true
        else
            echo "$0: Stage range excludes 18; RL options were ignored and RL was not executed." >&2
        fi
    fi
fi

train_set=train
valid_set=dev
test_sets=dev

nbpe=50000
s2t_config=conf/train_conv2d8_size1024_e18_d18_mel128.yaml

./s2t.sh \
    --use_lm false \
    --num_nodes 4 \
    --ngpu 4 \
    --nj 512 \
    --gpu_inference true \
    --inference_nj 64 \
    --num_splits_s2t 1 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 100000000 \
    --s2t_config "${s2t_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" "${s2t_args[@]}"

if ${run_rl}; then
    "${recipe_dir}/rl.sh" "${rl_args[@]}"
fi
