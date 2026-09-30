#!/usr/bin/env bash
#SBATCH --job-name=bcc-random-validation
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --output=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%A_%a.out
#SBATCH --error=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca"
PYTHON="/leonardo_work/cin_staff/lquerci1/MLFF/.micromamba/envs/recharge-condensation/bin/python"
MANIFEST="/leonardo_work/cin_staff/lquerci1/MLFF/recharge-condensation-work/resonance-fit/manifests/validation_covered_manifest.csv"
CONFORMER_SEED=20260930

TRAIN_CONFORMERS=(1 2 5 10)
N_TRAIN_CONFORMERS="${TRAIN_CONFORMERS[$SLURM_ARRAY_TASK_ID]}"

BCC_COLLECTION="$REPO_ROOT/partial-charge-condensation/results/fit_resonance_random-${N_TRAIN_CONFORMERS}-conf/fitted_bcc_collection.json"
OUTPUT_DIR="$REPO_ROOT/partial-charge-condensation/results/validation_resonance_random-${N_TRAIN_CONFORMERS}-trainconf"

if [[ ! -f "$BCC_COLLECTION" ]]; then
    echo "BCC collection non trovata: $BCC_COLLECTION" >&2
    exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output già esistente, non sovrascrivo: $OUTPUT_DIR" >&2
    exit 1
fi

echo "Training conformers: $N_TRAIN_CONFORMERS"
echo "Validation conformers: 10"
echo "Conformer seed: $CONFORMER_SEED"
echo "BCC collection: $BCC_COLLECTION"
echo "Output: $OUTPUT_DIR"

cd "$REPO_ROOT"

"$PYTHON" partial-charge-condensation/03_evaluate_bcc.py     --manifest "$MANIFEST"     --bcc-collection "$BCC_COLLECTION"     --output-dir "$OUTPUT_DIR"     --max-conformers 10     --conformer-seed "$CONFORMER_SEED"     --progress-every 100
