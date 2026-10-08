#!/usr/bin/env bash
#SBATCH --job-name=bcc-apply-5conf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --account=cin_staff
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%j.out
#SBATCH --error=/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca/partial-charge-condensation/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/leonardo_work/cin_staff/lquerci1/MLFF/openff-recharge-luca"
DATASET_ROOT="/leonardo_work/cin_staff/lquerci1/MLFF/GRAPPA/grappa-v.1.4.1/data/datasets"
RECHARGE_PY="/leonardo_work/cin_staff/lquerci1/MLFF/.micromamba/envs/recharge-condensation/bin/python"

BCC_COLLECTION="${REPO_ROOT}/partial-charge-condensation/results/fit_resonance_random-5-conf/fitted_bcc_collection.json"
REPORT_DIR="${REPO_ROOT}/partial-charge-condensation/results/application_resonance_random-5-conf-to-grappa"
SUFFIX="-recharge-condensed-5-confs-esp"

DATASETS=(
    gen2
    gen2-torsion
    pepconf-dlc
    protein-torsion
    spice-pubchem
    spice-dipeptide
    spice-des-monomers
    rna-nucleoside
)

cd "${REPO_ROOT}"

if [[ ! -f "${BCC_COLLECTION}" ]]; then
    echo "BCC collection not found: ${BCC_COLLECTION}" >&2
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    source_dir="${DATASET_ROOT}/${dataset}"
    target_dir="${DATASET_ROOT}/${dataset}${SUFFIX}"
    staging_dir="${DATASET_ROOT}/.${dataset}${SUFFIX}.incomplete"

    if [[ ! -d "${source_dir}" ]]; then
        echo "Source dataset not found: ${source_dir}" >&2
        exit 1
    fi

    if [[ -e "${target_dir}" ]]; then
        echo "Target already exists: ${target_dir}" >&2
        exit 1
    fi

    if [[ -e "${staging_dir}" ]]; then
        echo "Incomplete staging directory exists: ${staging_dir}" >&2
        exit 1
    fi
done

if [[ -e "${REPORT_DIR}" ]]; then
    echo "Report directory already exists: ${REPORT_DIR}" >&2
    exit 1
fi

echo "BCC collection: ${BCC_COLLECTION}"
echo "Dataset root:   ${DATASET_ROOT}"
echo "Output suffix:  ${SUFFIX}"
echo "Report:         ${REPORT_DIR}"
echo

"${RECHARGE_PY}" \
    partial-charge-condensation/07_apply_fitted_bcc_to_grappa.py \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${DATASET_ROOT}" \
    --bcc-collection "${BCC_COLLECTION}" \
    --dataset "${DATASETS[@]}" \
    --suffix="${SUFFIX}" \
    --report-dir "${REPORT_DIR}" \
    --progress-every 100

echo
echo "Application completed."
echo "Report: ${REPORT_DIR}"
