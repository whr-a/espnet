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
cd /u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1
conda activate versa_v2

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000

config_version=v6_7
epoch=132
test_set="test_music"
visqol=false
mel_stft=true
opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi


train_set=train_all
valid_set=dev_sub
test_sets=${test_set}

train_config=conf/tuning/pretrain_gan_music/${config_version}.yaml
inference_config=conf/decode.yaml
score_config=versa/test/visqol.yaml
if [ "$visqol" = "true" ]; then
    ./codec.sh \
        --local_data_opts "--trim_all_silence false" \
        --fs ${fs} \
        --nj 16\
        --stage 7\
        --stop_stage 7\
        --expdir /work/nvme/bbjs/shi3/codec_haoran/espnet/egs_band/bandcodec/codec1/exp_music\
        --python /work/nvme/bbjs/hwang41/miniconda3/envs/versa_v2/bin/python\
        --inference_model ${epoch}epoch.pth\
        --train_config "${train_config}" \
        --inference_config "${inference_config}" \
        --scoring_config "${score_config}" \
        --inference_nj 16\
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" ${opts} "$@"
fi    

if [ "$mel_stft" = "true" ]; then
    conda activate espnet

    for current_test in ${test_set}; do
        
        echo "--> Processing test set: ${current_test}"

        # 1. 根据当前的测试集名称，动态生成 wav 文件的路径
        wav_path="/work/nvme/bbjs/shi3/codec_haoran/espnet/egs_band/bandcodec/codec1/exp_music/codec_${config_version}_raw_fs24000/decode_${epoch}epoch/${current_test}/wav"

        # 2. (推荐) 检查路径是否存在，如果不存在就跳过，避免报错
        if [ ! -d "${wav_path}" ]; then
            echo "    Warning: Directory not found, skipping. Path: ${wav_path}"
            continue # 继续下一次循环
        fi

        echo "    Running score script for path: ${wav_path}"
        python /work/nvme/bbjs/hwang41/3ai/espnet/egs_band/bandcodec/codec1/test/test2.py "${wav_path}"

    done
fi