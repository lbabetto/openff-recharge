#!/usr/bin/env bash
#SBATCH --job-name=bcc-selected-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --output=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%j.out
#SBATCH --error=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca"
PYTHON="/leonardo_work/cin_staff/lquerci1/MLFF/.micromamba/envs/recharge-condensation/bin/python"
MANIFEST="/leonardo_work/cin_staff/lquerci1/MLFF/recharge-condensation-work/resonance-fit/manifests/test_covered_manifest.csv"
BCC_COLLECTION="$REPO_ROOT/partial-charge-condensation/results/fit_resonance_random-5-conf/fitted_bcc_collection.json"
OUTPUT_DIR="$REPO_ROOT/partial-charge-condensation/results/test_resonance_random-5-trainconf"
CONFORMER_SEED=20260930

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest non trovato: $MANIFEST" >&2
    exit 1
fi

if [[ ! -f "$BCC_COLLECTION" ]]; then
    echo "BCC collection non trovata: $BCC_COLLECTION" >&2
    exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output già esistente, non sovrascrivo: $OUTPUT_DIR" >&2
    exit 1
fi

echo "Modello selezionato: 5 conformeri di training"
echo "Conformeri di test: 10"
echo "Conformer seed: $CONFORMER_SEED"
echo "Test manifest: $MANIFEST"
echo "BCC collection: $BCC_COLLECTION"
echo "Output: $OUTPUT_DIR"

cd "$REPO_ROOT"

"$PYTHON" partial-charge-condensation/03_evaluate_bcc.py     --manifest "$MANIFEST"     --bcc-collection "$BCC_COLLECTION"     --output-dir "$OUTPUT_DIR"     --max-conformers 10     --conformer-seed "$CONFORMER_SEED"     --progress-every 100
