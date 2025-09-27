#!/bin/bash
#SBATCH --job-name=reserve1_gpu1        # 作业名字
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4-interactive         # GPU 分区
#SBATCH --nodes=1                      # 1 个节点
#SBATCH --ntasks=1                     # 1 个任务
#SBATCH --gres=gpu:1                   # 请求 1 张 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00 
#SBATCH --output=/u/hwang41/reserve/reserve4_gpu_%j.out   # 输出文件

echo "[$(date '+%F %T')] 分配到节点：$SLURM_NODELIST"

source ~/.bashrc
cd /work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2
conda activate espnet

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters. Usage: sbatch infer.sh <config_version> <epoch> <test_set>"
    exit 1
fi
config_version=$1
epoch=$2
target_bandwidths=$3
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi


train_set=speech
valid_set=speech_validation
test_sets="open_testset_track2_clean_enh open_testset_track2_noisy_enh open_testset_track2_reverb_enh"

train_config=conf/codec_back/train_${config_version}.yaml
inference_config=conf/decode.yaml
score_config=conf/score.yaml

./codec.sh \
    --local_data_opts "--trim_all_silence false" \
    --fs ${fs} \
    --ngpu 1 \
    --nj 64\
    --stage 6\
    --stop_stage 6\
    --target_bandwidths ${target_bandwidths}\
    --inference_model ${epoch}epoch.pth\
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --scoring_config "${score_config}" \
    --inference_nj 1 \
    --expdir /work/nvme/bbjs/someki1/deltaai/20_other_project/06_LRAC_Challenge/egs2/lrac/track2/exp\
    --gpu_inference true\
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts}
