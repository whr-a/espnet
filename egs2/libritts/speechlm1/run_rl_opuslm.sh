#!/usr/bin/env bash
# Copyright 2026 Haoran Wang
# Apache 2.0
set -euo pipefail
./speechlm.sh --stage 10 --stop_stage 10 --skip_data_prep true \
    --skip_train true --skip_eval true --task asr --tag rl_opuslm \
    --rl_config conf/rl_opuslm_librispeech.yaml "$@"
