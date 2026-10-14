#!/usr/bin/env bash
# Use data/train_clean_100/{wav.scp,text} produced by the LibriSpeech recipe.
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${recipe_dir}/../../owsm_v4/s2t1/rl.sh" --stage 18 "$@"
