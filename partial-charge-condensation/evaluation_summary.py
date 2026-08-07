import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def count_manifest_rows(path):
    with path.open(newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


parser = argparse.ArgumentParser()
parser.add_argument("--input-summary", required=True)
parser.add_argument("--original-manifest", required=True)
parser.add_argument("--reference-charge-model", required=True)
parser.add_argument("--output-summary", required=True)
parser.add_argument("--reference-esp-source")
args = parser.parse_args()

input_path = Path(args.input_summary).resolve()
original_manifest_path = Path(args.original_manifest).resolve()
output_path = Path(args.output_summary).resolve()

summary = json.loads(input_path.read_text())

selected_molecules = summary.get(
    "selected_molecules",
    summary.get("molecules"),
)

if selected_molecules is None:
    raise KeyError(
        "Il riepilogo non contiene selected_molecules o molecules."
    )

selected_molecules = int(selected_molecules)
original_molecules = count_manifest_rows(original_manifest_path)

if selected_molecules > original_molecules:
    raise ValueError(
        "Il numero di molecole valutate è maggiore di quello presente "
        "nel manifest originale."
    )

reference_esp_source = args.reference_esp_source

if reference_esp_source is None:
    reference_esp_source = (
        "point-charge ESP reconstructed from "
        f"{args.reference_charge_model} partial charges"
    )

summary.update(
    {
        "reference_charge_model": args.reference_charge_model,
        "reference_esp_source": reference_esp_source,
        "original_test_manifest": str(original_manifest_path),
        "original_test_manifest_sha256": sha256(
            original_manifest_path
        ),
        "original_test_molecules": original_molecules,
        "excluded_uncovered_molecules": (
            original_molecules - selected_molecules
        ),
        "test_coverage_fraction": (
            selected_molecules / original_molecules
        ),
        "aggregation": {
            "charge_rmse": "global RMSE over atoms",
            "esp_rmse": "global RMSE over ESP grid points",
        },
    }
)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(summary, indent=2) + "\n")

print("Molecole nel test originale:", original_molecules)
print("Molecole valutate:", selected_molecules)
print(
    "Molecole escluse perché non coperte:",
    original_molecules - selected_molecules,
)
print(
    "Copertura del test:",
    f"{100.0 * selected_molecules / original_molecules:.2f}%",
)
print("Output:", output_path)
