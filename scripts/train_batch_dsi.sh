#!/bin/bash
#SBATCH -J exatrkx
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=30G
#SBATCH --open-mode=append # So that outcomes are appended, not rewritten
#SBATCH --signal=SIGUSR1@90 # So that job requeues 90 seconds before timeout

# Optional SBATCH arguments. Modify as needed.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@rcc.uchicago.edu
#SBATCH --output=/home/%u/%A.out
#SBATCH --error=/home/%u/%A.err

# If there are existing checkpoint files, specify ckpt_dir
ckpt_dir=/net/projects/fermi-gnn/jiheeyou/vertex/attentional-mlp-64-sementic-filter/checkpoints/
files=$(ls -t $ckpt_dir)
latest_file=$(echo "$files" | head -n 1) # Get the first (latest) file
ckpt_path=$ckpt_dir$latest_file # Construct the full path to the most recent checkpoint file
echo $ckpt_path # This will print ckpt_path in .out file

# Add either of these two flag combinations to srun command below
# Choose based on if you have ckpt files or not
# Choice 1: --logdir /net/projects/fermi-gnn/%u --name vertex
# Choice 2: --resume $ckpt_path
srun python scripts/train.py --data-path /net/projects/fermi-gnn/CHEP2023.gnn.h5 --vertex $@

# This is an examble submission command
# You submit through the terminal after activating numl-dsi
# sbatch scripts/train_batch_dsi.sh --version attentional-mlp-64-sementic-filter --vertex-aggr attn --vertex-mlp-feats 64 --semantic --filter