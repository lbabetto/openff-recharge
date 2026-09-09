#!/bin/bash
#SBATCH --job-name merge-qm-esp
#SBATCH --account cin_staff
#SBATCH --partition dcgp_usr_prod
#SBATCH --time 02:00:00
#SBATCH --nodes 1
#SBATCH --output slurm-%x-%j.out
#SBATCH --error slurm-%x-%j.err
#SBATCH --mail-user l.babetto@cineca.it
##SBATCH --mail-type ALL

source /leonardo_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

SMILES_FILE=$1
OUTPUT_FILE="${SMILES_FILE%.smi}.sqlite"

time python merge-esp-stores.py "${SMILES_FILE%.smi}"-*.sqlite --output "$OUTPUT_FILE"

