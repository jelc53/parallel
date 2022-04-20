#!/bin/bash
#SBATCH -o slurm.sh.out
#SBATCH -p CME
#SBATCH --cpus-per-task=16

OMP_NUM_THREADS=4
### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main_q1
echo ----------------
./main_q1

echo
echo Output from main_q2
echo ----------------
./main_q2

echo
echo Output from main_q2_part6
echo ----------------
./main_q2_part6
