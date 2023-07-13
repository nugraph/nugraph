#!/bin/bash
#SBATCH -J exatrkx_process
#SBATCH -p cpu_gce
#SBATCH -n 64
#SBATCH -A fwk
#SBATCH -q regular

infile=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/enhanced.evt.h5
outfile=/wclustre/fwk/exatrkx/data/uboone/CHEP2023/test.gnn.h5

# process in parallel and then merge output
mpiexec -l -n 64 scripts/process.py -i $infile -o $outfile
scripts/merge.py -f $outfile