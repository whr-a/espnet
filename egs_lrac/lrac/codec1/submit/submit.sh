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
source ~/l
conda activate espnet
cd /work/nvme/bbjs/hwang41/lrac/espnet/egs_lrac/lrac/codec1
bash run.sh
