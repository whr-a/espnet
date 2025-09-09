#!/bin/bash
#SBATCH --job-name=canary_encoder_distill
#SBATCH --output=logs/canary_encoder_distill_%A_%a.out
#SBATCH --error=logs/canary_encoder_distill_%A_%a.err
#SBATCH --account=bbjs-delta-gpu     # Your project account
#SBATCH --partition=gpuA40x4        # Or another GPU partition if needed
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --cpus-per-task=4            # Number of CPU cores
#SBATCH --mem=128G                    # Memory
#SBATCH --time=01-00:00:00              # Job time limit (hh:mm:ss)

# Initialize conda (only needed once per session if not in .bashrc)
source /work/nvme/bbjs/bsu5/miniconda3/etc/profile.d/conda.sh
source /u/bsu5/.bashrc
# Activate your environment
conda activate espnet

export PYTHONPATH=/work/nvme/bbjs/bsu5/lrac_espnet/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder:/work/nvme/bbjs/bsu5/lrac_espnet/espnet:$PYTHONPATH
# Change directory to your project
cd /work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs_lrac/lrac/codec1


# python kmeans_cluster.py --wav_scp /work/nvme/bbjs/hwang41/lrac/espnet/egs_lrac/lrac/codec1/dump/raw/train_all/wav.scp --out_dir /work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs_lrac/lrac/codec1
python distill_canary.py --wav_scp /work/nvme/bbjs/hwang41/lrac/espnet/egs_lrac/lrac/codec1/dump/raw/train_all/wav.scp --emb_dir /work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs_lrac/lrac/codec1/canary_embedding --student_cfg ./1b_student_encoder.yaml --preproc_pt /work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/preprocessor_standalone.pt --out_dir ./distill_runs --epochs 10 --batch_size 16 --lr 2e-4 --amp