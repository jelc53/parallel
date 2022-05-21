#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=fp
#SBATCH --output=fp-%j.out
#SBATCH --error=fp-%j.err

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

# Name of executable to use; modify as needed
EXE=$SLURM_SUBMIT_DIR/main

# Select the number of MPI processes to use
N=4

# This script will run all 4 modes with your choice of N

# Runs mode 1, 2, and 3
for mode in 1 2 3; do

  echo -e "\n* Mode ${mode} *"
  echo mpirun -np ${N} $EXE -g ${mode}  
  mpirun -np ${N} $EXE -g ${mode}

done

echo -e "\n*** Summary ***\n"

for mode in 1 2 3; do
  tail -n 1 Outputs/CpuGpuDiff-${N}-${mode}.txt
done

echo -e "\n*** Grading mode 4 ***\n"

echo "main -g 4"
$EXE -g 4

echo -e "\n*** Tests are complete ***\n"
