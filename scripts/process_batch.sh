#!/bin/bash
#SBATCH -J exatrkx_process
#SBATCH -p cpu_gce
#SBATCH -t 240
#SBATCH -n 64
#SBATCH -A fwk
#SBATCH -q regular

# process in parallel and then merge output
mpiexec -l -n $SLURM_NPROCS scripts/process.py -i $1 -o $2 --label-vertex
scripts/merge.py -f $2
