#!/bin/bash
#SBATCH --job-name=reserve1_cpu1        # 作业名字
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu          # GPU 分区
##SBATCH --nodes=1                      # 1 个节点
##SBATCH --ntasks=1                     # 1 个任务
#SBATCH --cpus-per-task=16              # 每任务 8 核 CPU
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00              # 最长运行 2 天
#SBATCH --output=/u/hwang41/reserve/reserve4_gpu_%j.out   # 输出文件

source ~/.bashrc
conda activate espnet

cd /work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2

echo "[$(date '+%F %T')] 分配到节点：$SLURM_NODELIST"

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
    --target_bandwidths 6\
    --inference_model LRAC2025_Track2_baseline_model.pth\
    --python /work/nvme/bbjs/hwang41/miniconda3/envs/versa_v2_lrac/bin/python\
    --stage 7 \
    --stop_stage 7\
    --nj 8 \
    --inference_nj 8 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts} "$@"