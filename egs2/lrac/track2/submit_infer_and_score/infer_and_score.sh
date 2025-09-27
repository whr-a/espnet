#!/bin/bash

# ==================== CONFIGURATION ====================
#              MODIFY YOUR EXPERIMENT HERE
# =======================================================
CONFIG_VERSION="v1"
EPOCH=253
# 如果有多个 test_set，用引号包起来，用空格隔开
target_bandwidths=1
# =======================================================


# --- Script Paths ---
INFER_SCRIPT="/work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2/submit_infer_and_score/infer.sh"
SCORE_SCRIPT="/work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2/submit_infer_and_score/score.sh"

# --- Script Logic ---

# 1. 提交推理任务，并将配置作为参数传递
echo "Submitting GPU inference job with config: ${CONFIG_VERSION}, epoch: ${EPOCH}'"
INFER_JOB_ID=$(sbatch --parsable "$INFER_SCRIPT" "$CONFIG_VERSION" "$EPOCH" "$target_bandwidths")

if [ -z "$INFER_JOB_ID" ]; then
    echo "Error: Failed to submit infer.sh!"
    exit 1
fi
echo "Inference job submitted with ID: $INFER_JOB_ID"


# 2. 提交评分任务，传递相同的配置，并设置依赖
echo "Submitting CPU scoring job with the same config..."
SCORE_JOB_ID=$(sbatch --parsable --dependency=afterok:"$INFER_JOB_ID" "$SCORE_SCRIPT" "$CONFIG_VERSION" "$EPOCH" "$target_bandwidths")

if [ -z "$SCORE_JOB_ID" ]; then
    echo "Error: Failed to submit score.sh!"
    scancel "$INFER_JOB_ID"
    echo "Cancelled job $INFER_JOB_ID."
    exit 1
fi
echo "Scoring job submitted with ID: $SCORE_JOB_ID. It will run after job $INFER_JOB_ID completes."
echo "Pipeline successfully submitted."