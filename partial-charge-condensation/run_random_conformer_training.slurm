#!/usr/bin/env bash
#SBATCH --job-name=bcc-random-conformers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-3
#SBATCH --output=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%A_%a.out
#SBATCH --error=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%A_%a.err

set -euo pipefail

REPO="/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca"
PYTHON="/leonardo_work/cin_staff/lquerci1/MLFF/.micromamba/envs/recharge-condensation/bin/python"
MANIFEST="/leonardo_work/cin_staff/lquerci1/MLFF/recharge-condensation-work/resonance-fit/manifests/train_covered_manifest.csv"
CONFORMER_SEED=20260930

CONFORMER_COUNTS=(1 2 5 10)
N_CONFORMERS="${CONFORMER_COUNTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$REPO/partial-charge-condensation/results/fit_resonance_random-${N_CONFORMERS}-conf"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output directory already exists: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$REPO/partial-charge-condensation/logs"
cd "$REPO"

echo "Conformers per molecule: $N_CONFORMERS"
echo "Conformer seed: $CONFORMER_SEED"
echo "Output directory: $OUTPUT_DIR"

"$PYTHON" \
    partial-charge-condensation/02_fit_global_bcc.py \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --max-conformers "$N_CONFORMERS" \
    --conformer-seed "$CONFORMER_SEED" \
    --checkpoint-every 100
