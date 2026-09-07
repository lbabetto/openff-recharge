#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition dcgp_usr_prod
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node 1
#SBATCH --gres tmpfs:300G
#SBATCH --output slurm-%j.out
#SBATCH --error slurm-%j.out
#SBATCH --mail-user l.babetto@cineca.it
##SBATCH --mail-type ALL

source /leonardo_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

export PSI_SCRATCH=$TMPDIR

SMILES_FILE=$1

time python 1-precompute-QM-data.py $SMILES_FILE
