#!/bin/bash
#SBATCH -J exatrkx
#SBATCH -t 1440
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=30G

srun python scripts/train.py --data-path /net/projects/fermi-gnn/CHEP2023.gnn.h5 --vertex $@
