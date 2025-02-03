#!/bin/bash
#SBATCH -J exatrkx
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=30G
#SBATCH --open-mode=append # So that outcomes are appended, not rewritten
#SBATCH --signal=SIGUSR1@90

srun python scripts/train.py $@