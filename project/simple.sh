#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=fp
#SBATCH --output=fp-%j.out
#SBATCH --error=fp-%j.err

make
time mpirun -n 4 ./main -sd -n 1000 -b 800 -l 0.001 -e 2
