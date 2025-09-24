#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000

opts="--audio_format wav "

train_set=speech
valid_set=speech_validation
test_sets="open_testset_track2_clean_enh open_testset_track2_noisy_enh open_testset_track2_reverb_enh"
# test_sets="test_all"

train_config=conf/train.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml

./codec.sh \
    --local_data_opts "--trim_all_silence false" \
    --fs ${fs} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --gpu_inference true\
    --target_bandwidths 1\
    --inference_model LRAC2025_Track2_baseline_model.pth\
    --stage 6 \
    --stop_stage 6\
    --ngpu 1 \
    --nj 12 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts} "$@"
