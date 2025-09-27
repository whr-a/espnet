#!/bin/bash
#SBATCH --job-name=reserve1_cpu1        # 作业名字
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu          # GPU 分区
##SBATCH --nodes=1                      # 1 个节点
##SBATCH --ntasks=1                     # 1 个任务
#SBATCH --cpus-per-task=64              # 每任务 8 核 CPU
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00              # 最长运行 2 天
#SBATCH --output=/u/hwang41/reserve/reserve4_gpu_%j.out   # 输出文件

echo "[$(date '+%F %T')] 分配到节点：$SLURM_NODELIST"

source ~/.bashrc
cd /work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track1
conda activate versa_v2_lrac

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters. Usage: sbatch infer.sh <config_version> <epoch> <test_set>"
    exit 1
fi
config_version=$1
epoch=$2
target_bandwidths=$3

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi


train_set=speech
valid_set=speech_validation
test_sets="open_testset_track1_clean open_testset_track1_noisy open_testset_track1_reverb"

train_config=conf/config_universa/universa_${config_version}.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml


./codec.sh \
    --local_data_opts "--trim_all_silence false" \
    --fs ${fs} \
    --nj 8\
    --stage 7\
    --stop_stage 7\
    --target_bandwidths ${target_bandwidths}\
    --expdir /work/nvme/bbjs/someki1/deltaai/20_other_project/06_LRAC_Challenge/egs2/lrac/track1/exp\
    --python /work/nvme/bbjs/hwang41/miniconda3/envs/versa_v2_lrac/bin/python\
    --inference_model ${epoch}epoch.pth\
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --inference_nj 8\
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts}
