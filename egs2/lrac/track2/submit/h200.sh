#!/bin/bash
#SBATCH --job-name=reserve1_gpu1        # 作业名字
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuH200x8         # GPU 分区
#SBATCH --nodes=1                      # 1 个节点
#SBATCH --ntasks=1                     # 1 个任务
#SBATCH --gres=gpu:1                   # 请求 1 张 GPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=01:00:00 
#SBATCH --output=/u/hwang41/reserve/reserve4_gpu_%j.out  # 输出文件

echo "[$(date '+%F %T')] 分配到节点：$SLURM_NODELIST"

source ~/.bashrc
conda activate espnet

export PYTHONPATH=/u/hwang41/hwang41/lrac/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder:/u/hwang41/hwang41/lrac/espnet_mylrac:/u/hwang41/hwang41/3ai/versa:/u/hwang41/hwang41/lrac/Longcat-Codec:$PYTHONPATH

cd /work/nvme/bbjs/hwang41/lrac/espnet_mylrac/egs2/lrac/track2

bash run.sh
