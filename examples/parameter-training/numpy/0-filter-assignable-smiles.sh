#!/bin/bash
#SBATCH --job-name filter-smiles
#SBATCH --account cin_staff
#SBATCH --partition dcgp_usr_prod
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node 1
#SBATCH --gres tmpfs:300G
#SBATCH --output slurm-%x-%A_%a.out
#SBATCH --error slurm-%x-%A_%a.err
#SBATCH --mail-user l.babetto@cineca.it
##SBATCH --mail-type ALL

source /leonardo_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

SMILES_FILE=$1
SMARTS_FILE=$2

time python filter-assignable-smiles.py $SMILES_FILE "${SMILES_FILE%.smi}"-filtered.smi $SMARTS_FILE
