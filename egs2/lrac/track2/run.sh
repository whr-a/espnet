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
test_sets="open_testset_track2_clean open_testset_track2_noisy open_testset_track2_reverb"

train_config=conf/train.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml

./codec.sh \
    --local_data_opts "--trim_all_silence false" \
    --fs ${fs} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    ${opts} "$@"
