#!/bin/bash
#SBATCH -j openff_train
#SBATCH -a cin_staff
#SBATCH -p g100_usr_prod 
#SBATCH -t 0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

source /g100_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python train-bcc-parameters.py spice-pubchem.smi SMARTS.txt
