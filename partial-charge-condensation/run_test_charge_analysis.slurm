#!/usr/bin/env bash
#SBATCH --job-name=bcc-test-charge-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%j.out
#SBATCH --error=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca"
PYTHON="/leonardo_work/cin_staff/lquerci1/MLFF/.micromamba/envs/recharge-condensation/bin/python"
MANIFEST="/leonardo_work/cin_staff/lquerci1/MLFF/recharge-condensation-work/resonance-fit/manifests/test_covered_manifest.csv"
BCC_COLLECTION="$REPO_ROOT/partial-charge-condensation/results/fit_resonance_random-5-conf/fitted_bcc_collection.json"
OUTPUT_DIR="$REPO_ROOT/partial-charge-condensation/results/test_resonance_random-5-trainconf/atomic-charge-analysis"

cd "$REPO_ROOT"

"$PYTHON"     partial-charge-condensation/04_analyze_test_charge_predictions.py     --manifest "$MANIFEST"     --bcc-collection "$BCC_COLLECTION"     --output-dir "$OUTPUT_DIR"     --expected-charge-rmse 0.10864688306866589     --top-n 10     --progress-every 100
