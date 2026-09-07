#!/bin/bash
#SBATCH --job-name train-bcc-parameters
#SBATCH --account cin_staff
#SBATCH --partition dcgp_usr_prod
#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --exclusive
#SBATCH --output slurm-%x-%j.out
#SBATCH --error slurm-%x-%j.err
#SBATCH --mail-user l.babetto@cineca.it
##SBATCH --mail-type ALL

source /leonardo_work/cin_staff/lbabetto/miniforge3/etc/profile.d/conda.sh
conda activate openff-recharge

QM_ESP_FILE=$1
SMARTS_FILE=$2

time python 2-train-bcc-parameters.py $QM_ESP_FILE $SMARTS_FILE
