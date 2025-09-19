#!/bin/bash
#SBATCH --job-name=arecho_loss_lrac_a40
#SBATCH --output=logs/arecho_loss_lrac_a40_%A_%a.out
#SBATCH --error=logs/arecho_loss_lrac_a40_%A_%a.err
#SBATCH --account=bbjs-delta-gpu     # Your project account
#SBATCH --partition=gpuA40x4        # Or another GPU partition if needed
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --cpus-per-task=4            # Number of CPU cores
#SBATCH --mem=64G                    # Memory
#SBATCH --time=02-00:00:00              # Job time limit (hh:mm:ss)

# Initialize conda (only needed once per session if not in .bashrc)
source /work/nvme/bbjs/bsu5/miniconda3/etc/profile.d/conda.sh
source /u/bsu5/.bashrc
# Activate your environment
conda activate espnet

export PYTHONPATH=/work/nvme/bbjs/bsu5/lrac_espnet/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder:/work/nvme/bbjs/bsu5/lrac_espnet/espnet:$PYTHONPATH
# Change directory to your project
cd /work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs2/lrac/track1

bash run_arecho.sh