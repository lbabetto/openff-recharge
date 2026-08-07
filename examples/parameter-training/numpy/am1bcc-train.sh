#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition g100_usr_prod 
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --output slurm-%x-%j.out
#SBATCH --error slurm-%x-%j.err
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 48
#SBATCH --mail-user l.babetto@cineca.it
#SBATCH --mail-type ALL

source /g100_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

time python train-bcc-parameters.py $1 $2
