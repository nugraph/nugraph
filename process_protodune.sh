#!/bin/bash
#SBATCH -J exatrkx_process
#SBATCH -p cpu_gce
#SBATCH -t 240
#SBATCH -n 64
#SBATCH -A fwk
#SBATCH -q regular

# process in parallel and then merge output
#mpiexec -l -n 32 scripts/process_protodune.py -i $1 -o $2 --label-vertex
mpiexec -l -n 16 scripts/process_protodune.py -i $1 -o $2
scripts/merge.py -f $2
