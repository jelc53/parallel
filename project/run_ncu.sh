#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=fp
#SBATCH --output=ncompute.out
#SBATCH --error=ncompute.err

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------
echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $SLURM_SUBMIT_DIR`"
echo
echo Output from code
echo ----------------
### end of information preamble

cd $SLURM_SUBMIT_DIR

make

ncu -f -o ncompute --set full ./main -g 3
# ncu -f -o ncompute --target-process all mpirun -n 4 ./main -n 100 -b 10000 -l 0.001 -e 1
