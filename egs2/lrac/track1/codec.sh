#!/usr/bin/env bash

# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0
# Modified from the original template for LRAC challenge

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=7         # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=6               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=1                 # The number of parallel jobs.
inference_nj=1       # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw             # Input feature type.
audio_format=wav          # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=2.9         # Minimum duration in second.
max_wav_duration=240       # Maximum duration in second.
fs=24000                   # Sampling rate.

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
codec_exp=""         # Specify the directory path for experiment. If this option is specified, tag is ignored.
codec_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for codec corpus.

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding (e.g., "--threshold 0.75").
                    # Note that it will overwrite args in inference config.
target_bandwidths='1 6' # We are separately handling the target bandwidths here in decoding
                        # instead of adding them to the inference config or args

inference_tag=""    # Suffix for decoding directory.
inference_model=valid.mel_loss.best.pth
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
download_model=""  # Download a model from Model Zoo and use it for decoding.

# Scoring related
scoring_config="" # Config for scoring.
scoring_args=""   # Arguments for scoring.
                  # Note that it will overwrite args in scoring config.
scoring_tag=""    # Suffix for scoring directory.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
audio_fold_length=256000 # fold_length for audio data.

# Upload model related
hf_repo=

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>" --srctexts "<srctexts>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --fs               # Sampling rate (default="${fs}").

    # Training related
    --train_config  # Config for training (default="${train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --codec_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${codec_exp}").
    --codec_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${codec_stats_dir}").
    --num_splits    # Number of splitting for codec corpus (default="${num_splits}").

    # Decoding related
    --inference_config  # Config for decoding (default="${inference_config}").
    --inference_args    # Arguments for decoding, (default="${inference_args}").
                        # e.g., --inference_args "--threshold 0.75"
                        # Note that it will overwrite args in inference config.
    --inference_tag     # Suffix for decoding directory (default="${inference_tag}").
    --inference_model   # Model path for decoding (default=${inference_model}).
    --download_model    # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # Scoring related
    --scoring_config     # Config for scoring (default="${scoring_config}").
    --scoring_args       # Arguments for scoring (default="${scoring_args}").
    --scoring_tag        # Suffix for scoring directory (default="${scoring_tag}").


    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set          # Name of training set (required).
    --valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --test_sets          # Names of test sets (required).
                         # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --audio_fold_length  # Fold length for audio data (default="${audio_fold_length}").

    # Upload model related
    ---hf_repo          # Huggingface model tag for huggingface model upload
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: only supported: --feats_type raw"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_fs${fs}"
    else
        tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi
if [ -z "${scoring_tag}" ]; then
    if [ -n "${scoring_config}" ]; then
        scoring_tag="$(basename "${scoring_config}" .yaml)"
    else
        scoring_tag=scoring
    fi
    # Add overwritten arg's info
    if [ -n "${scoring_args}" ]; then
        scoring_tag+="$(echo "${scoring_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${codec_stats_dir}" ]; then
    codec_stats_dir="${expdir}/codec_stats_${feats_type}"
fi
# The directory used for training commands
if [ -z "${codec_exp}" ]; then
    codec_exp="${expdir}/codec_${tag}"
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        log "This is handled in the lrac data generation repo. Run the recipe there and copy the data folder here!!!"
        DIRS=("./data" "./data/speech" "./data/speech_validation" "./data/open_testset_track1_clean" "./data/open_testset_track1_noisy" "./data/open_testset_track1_reverb")
        for DIR in "${DIRS[@]}"; do
            if [ ! -d "$DIR" ]; then
                echo "Assertion failed: Directory '$DIR' does not exist." >&2
                exit 1
            fi
        done
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(kamo): Change kaldi-ark to npy or HDF5?
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            mkdir -p "${data_feats}${_suf}/${dset}"
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done

        for dset in ${test_sets}; do
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" --out_filename "reference.scp" ${_opts} \
                "data/${dset}/reference.scp" "${data_feats}/${dset}"
        done
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            mkdir -p "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/wav.scp" "${data_feats}/${dset}/wav.scp"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"
        done
    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation has finished here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Neural codec collect stats: We are skipping this stage as we use raw audio in this recipe!"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: Codec Training: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.gan_codec_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        log "Generate '${codec_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${codec_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${codec_exp}/run.sh"; chmod +x "${codec_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "Neural codec training started... log: '${codec_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${codec_exp})"
        else
            jobname="${codec_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${codec_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${codec_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.gan_codec_train" \
                --use_preprocessor true \
                --resume true \
                --fold_length "${audio_fold_length}" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},audio,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},audio,${_type}" \
                --output_dir "${codec_exp}" \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi

# ========================== Training has finished here. ==========================

if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Decoding: training_dir=${codec_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        for tbw in $target_bandwidths; do
            final_inference_tag=$inference_tag 
            final_inference_tag+="_${tbw}kbps"
            log "Generate '${codec_exp}/${final_inference_tag}/run.sh'. You can resume the process from stage 6 using this script"
            mkdir -p "${codec_exp}/${final_inference_tag}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${codec_exp}/${final_inference_tag}/run.sh"; chmod +x "${codec_exp}/${final_inference_tag}/run.sh"

            for dset in ${test_sets}; do
                _data="${data_feats}/${dset}"
                _dir="${codec_exp}/${final_inference_tag}/${dset}"
                _logdir="${_dir}/log"
                mkdir -p "${_logdir}"

                # 0. Copy feats_type
                cp "${_data}/feats_type" "${_dir}/feats_type"

                # 1. Split the key file
                key_file=${_data}/wav.scp
                split_scps=""
                _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
                for n in $(seq "${_nj}"); do
                    split_scps+=" ${_logdir}/keys.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                # 2. Submit decoding jobs
                log "Decoding started... log: '${_logdir}/codec_inference.*.log'"
                # shellcheck disable=SC2046,SC2086
                ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/codec_inference.JOB.log \
                    ${python} -m espnet2.bin.gan_codec_inference \
                        --ngpu "${_ngpu}" \
                        --data_path_and_name_and_type ${_data}/${_scp},audio,${_type} \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --model_file "${codec_exp}"/"${inference_model}" \
                        --train_config "${codec_exp}"/config.yaml \
                        --output_dir "${_logdir}"/output.JOB \
                        --target_bandwidth "${tbw}" \
                        ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/codec_inference.*.log) ; exit 1; }

                # 3. Concatenates the output files from each jobs
                if [ -e "${_logdir}/output.${_nj}/codes" ]; then
                    mkdir -p "${_dir}"/codes
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/codes/feats.scp"
                    done | LC_ALL=C sort -k1 > "${_dir}/codes/feats.scp"
                fi
                if [ -e "${_logdir}/output.${_nj}/wav" ]; then
                    mkdir -p "${_dir}"/wav
                    for i in $(seq "${_nj}"); do
                        mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                        rm -rf "${_logdir}/output.${i}"/wav
                    done
                    find "${_dir}/wav" -name "*.wav" | while read -r line; do
                        echo "$(basename "${line}" .wav) ${line}"
                    done | LC_ALL=C sort -k1 > "${_dir}/wav/wav.scp"
                fi
            done
        done
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Scoring"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        for tbw in $target_bandwidths; do
            final_inference_tag=$inference_tag 
            final_inference_tag+="_${tbw}kbps"

            for dset in ${test_sets}; do
                _data="${data_feats}/${dset}"
                _gt_wavscp="${_data}/reference.scp"
                _dir="${codec_exp}/${final_inference_tag}/${dset}"
                _gen_wavscp="${_dir}/wav/wav.scp"

                log "Begin evaluation on ${dset}, results are written under ${_dir}"

                # 1. Split the key file
                _scoredir="${_dir}/${scoring_tag}_${tbw}kbps"
                _logdir="${_scoredir}/log"
                mkdir -p ${_scoredir}
                mkdir -p ${_logdir}

                # Get the minimum number among ${nj} and the number lines of input files
                _nj=$(min "${inference_nj}" "$(<${_gen_wavscp} wc -l)" )

                key_file=${_gen_wavscp}
                split_scps=""
                for n in $(seq "${_nj}"); do
                    split_scps+=" ${_logdir}/test.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                # 2. Generate run.sh
                log "Generate '${_scoredir}/run.sh'. You can resume the process from stage 7 using this script"
                echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${_scoredir}/run.sh"; chmod +x "${_scoredir}/run.sh"

                # 3. Submit jobs
                log "Evaluation started... log: '${_logdir}/codec_evaluate.*.log'"
                # shellcheck disable=SC2046,SC2086
                ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/codec_evaluate.JOB.log \
                    ${python} -m versa.bin.scorer \
                        --pred "${_logdir}"/test.JOB.scp \
                        --gt "${_gt_wavscp}" \
                        --output_file "${_logdir}/result.JOB.txt" \
                        --score_config "${scoring_config}" \
                        ${scoring_args} || { cat $(grep -l -i error "${_logdir}"/codec_evaluate.*.log) ; exit 1; }

                # 4. Aggregate the results
                ${python} pyscripts/utils/aggregate_eval.py \
                    --logdir "${_logdir}" \
                    --scoredir "${_scoredir}" \
                    --nj "${_nj}"

                # 5. Show results
                echo "Result saved at ${_scoredir}/avg_result.txt"
                cat "${_scoredir}/avg_result.txt"

            done
        done
    fi
else
    log "Skip the evaluation stages"
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
