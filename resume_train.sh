#!/bin/bash
#SBATCH -J exatrkx
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=Kat-Run
#SBATCH --output=logs/32ftestktrain_%j.out
#SBATCH --error=logs/32ftestktrain_%j.err
#SBATCH --signal=SIGUSR1@90
source setup-env.sh
python ./scripts/train.py \
  --device 0 \
  --data-path /net/projects/fermi-gnn/data/uboone-opendata/uboone-opendata-19be46d8.gnn.h5 \
  --semantic \
  --instance \
  --name FKFinal_Test32_Train \
  --batch-size 32 \
  --epochs 10 \
  --resume /net/projects/fermi-gnn/logs/kartavya/FKFinal_Test32_Train/nugraph3/h64qebu5/checkpoints/epoch=6-step=86912.ckpt
