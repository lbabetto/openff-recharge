#!/bin/bash
#SBATCH --job-name precompute-qm-esp
#SBATCH --account cin_staff
#SBATCH --partition dcgp_usr_prod
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node 1
#SBATCH --gres tmpfs:300G
#SBATCH --array 1-10
#SBATCH --output slurm-%x-%A_%a.out
#SBATCH --error slurm-%x-%A_%a.err
#SBATCH --mail-user l.babetto@cineca.it
##SBATCH --mail-type ALL

source /leonardo_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

export PSI_SCRATCH=$TMPDIR

SMILES_FILE=$1

# Working with a temporary file with the SMILES chunk to process, which gets removed at exit.
# The corresponding .sqlite database is instead kept and can be merged later with
# 1b-merge-QM-ESP.sh
CHUNK_FILE="${SMILES_FILE%.smi}-${SLURM_ARRAY_TASK_ID}.smi"
trap 'rm -f "$CHUNK_FILE"' EXIT

# NOTE: make sure to use a range 1-N for the job arrays, otherwise the split WILL get messed up.
split -n "l/$((SLURM_ARRAY_TASK_ID))/${SLURM_ARRAY_TASK_COUNT}" "$SMILES_FILE" > "$CHUNK_FILE"

time python precompute-QM-ESP.py "$CHUNK_FILE"
