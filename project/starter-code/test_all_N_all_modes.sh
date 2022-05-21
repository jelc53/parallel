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

# Choose single or double to select the precision
TAG=single

# Edit the lines below
# CPU_SAVE_DIR=/home/XXX
# OUTPUT_DIR=/home/XXX/Outputs_$TAG
# EXE=/home/XXX/main

DEBUG_DIR=Debug_$TAG
MAKEFILE_NAME=Makefile_$TAG

cd $SLURM_SUBMIT_DIR

make clean
make -f ${MAKEFILE_NAME} -j

if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

for mode in 1 2 3; do

  if [ ! -d Outputs/CPUmats ]; then 
    mkdir Outputs/CPUmats
  fi

  if [ -f Outputs/CPUmats/Sequentialb0-0.mat ]; then
    rm Outputs/CPUmats/*
  fi  

  # copy files
  if [ -f ${CPU_SAVE_DIR}/seq_nn-b0-${mode}.mat ]; then
    cp ${CPU_SAVE_DIR}/* Outputs/CPUmats
  fi

  for N in 1 2 3 4; do 

    echo mpirun -np ${N} $EXE -g ${mode}  
    mpirun -np ${N} $EXE -g ${mode}

    if [ ! -f ${CPU_SAVE_DIR}/seq_nn-b0-${mode}.mat ]; then
      mkdir ${OUTPUT_DIR}/mode_${mode}    
      cp Outputs/CPUmats/* ${OUTPUT_DIR}/mode_${mode}
    fi   

    if [ -d ${DEBUG_DIR}_${N}_${mode} ]; then
        rm -r ${DEBUG_DIR}_${N}_${mode}
    fi 

    mkdir ${DEBUG_DIR}_${N}_${mode}
    cp Outputs/{CpuGpuDiff-${N}-${mode}.txt,NNErrors-${N}-${mode}.txt} ${DEBUG_DIR}_${N}_${mode}/   
  done 
done

echo -e "\n*** Summary ***\n"

for mode in 1 2 3; do  
  for N in 1 2 3 4; do 
    tail -n 1 ${DEBUG_DIR}_${N}_${mode}/CpuGpuDiff-${N}-${mode}.txt
  done 
done

echo -e "\n*** Grading mode 4 ***\n"

echo "main -g 4"
$EXE -g 4

echo -e "\n*** Tests are complete ***\n"
