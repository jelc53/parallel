#!/bin/bash
#SBATCH -o slurm.sh.out
#SBATCH -p CME
#SBATCH --gres gpu:1

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main
echo ----------------
./main -bgs
