import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = [
    "gen2",
    "gen2-torsion",
    "pepconf-dlc",
    "protein-torsion",
    "spice-pubchem",
    "spice-dipeptide",
    "spice-des-monomers",
    "rna-nucleoside",
    "rna-diverse",
]


def file_sort_key(path):
    if path.stem.isdigit():
        return 0, int(path.stem)
    return 1, path.stem


def determine_split(mol_id, smiles, test_smiles, validation_smiles):
    matches = set()

    for identifier in {mol_id, smiles}:
        if identifier in test_smiles:
            matches.add("test")
        if identifier in validation_smiles:
            matches.add("validation")

    if len(matches) > 1:
        raise ValueError(
            f"Identificatore assegnato a split incompatibili: {mol_id}"
        )

    if matches:
        return matches.pop()

    return "train"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", required=True)
parser.add_argument("--split-source", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
args = parser.parse_args()

dataset_root = Path(args.dataset_root).resolve()
split_source = Path(args.split_source).resolve()
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

with (split_source / "te_smiles.json").open() as file:
    test_smiles = set(json.load(file))

with (split_source / "vl_smiles.json").open() as file:
    validation_smiles_raw = set(json.load(file))

validation_test_overlap = validation_smiles_raw & test_smiles
validation_smiles = validation_smiles_raw - test_smiles

rows_by_split = {
    "train": [],
    "validation": [],
    "test": [],
}

found_test_smiles = set()
found_validation_smiles = set()

for dataset_name in args.datasets:
    dataset_path = dataset_root / dataset_name

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    npz_paths = sorted(dataset_path.glob("*.npz"), key=file_sort_key)

    if not npz_paths:
        raise RuntimeError(f"Nessun file NPZ trovato in {dataset_path}")

    print(f"Analisi {dataset_name}: {len(npz_paths)} molecole")

    for molecule_index, npz_path in enumerate(npz_paths, start=1):
        with np.load(npz_path, allow_pickle=False) as data:
            mol_id = str(data["mol_id"].item())
            smiles = str(data["smiles"].item())
            n_atoms = int(data["atomic_numbers"].shape[0])
            n_conformers = int(data["xyz"].shape[0])

        split = determine_split(
            mol_id,
            smiles,
            test_smiles,
            validation_smiles,
        )

        for identifier in {mol_id, smiles}:
            if identifier in test_smiles:
                found_test_smiles.add(identifier)
            if identifier in validation_smiles:
                found_validation_smiles.add(identifier)

        rows_by_split[split].append(
            {
                "path": str(npz_path),
                "dataset": dataset_name,
                "file_name": npz_path.name,
                "mol_id": mol_id,
                "smiles": smiles,
                "n_atoms": n_atoms,
                "n_conformers": n_conformers,
            }
        )

        if molecule_index % 1000 == 0:
            print(f"  completate {molecule_index} molecole")

fieldnames = [
    "path",
    "dataset",
    "file_name",
    "mol_id",
    "smiles",
    "n_atoms",
    "n_conformers",
]

for split_name, rows in rows_by_split.items():
    manifest_path = output_dir / f"{split_name}_manifest.csv"

    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

mol_ids_by_split = {
    split_name: {row["mol_id"] for row in rows}
    for split_name, rows in rows_by_split.items()
}

if mol_ids_by_split["train"] & mol_ids_by_split["test"]:
    raise RuntimeError("Sono presenti mol_id sia nel training sia nel test.")

if mol_ids_by_split["train"] & mol_ids_by_split["validation"]:
    raise RuntimeError(
        "Sono presenti mol_id sia nel training sia nella validation."
    )

if mol_ids_by_split["validation"] & mol_ids_by_split["test"]:
    raise RuntimeError(
        "Sono presenti mol_id sia nella validation sia nel test."
    )

dataset_summary = {}

for dataset_name in args.datasets:
    counts = Counter()

    for split_name, rows in rows_by_split.items():
        counts[split_name] = sum(
            row["dataset"] == dataset_name
            for row in rows
        )

    total = sum(counts.values())

    dataset_summary[dataset_name] = {
        "total": total,
        "train": counts["train"],
        "validation": counts["validation"],
        "test": counts["test"],
        "train_fraction": counts["train"] / total,
        "validation_fraction": counts["validation"] / total,
        "test_fraction": counts["test"] / total,
    }

total_rows = sum(len(rows) for rows in rows_by_split.values())

summary = {
    "dataset_root": str(dataset_root),
    "split_source": str(split_source),
    "datasets": args.datasets,
    "total_molecules": total_rows,
    "train_molecules": len(rows_by_split["train"]),
    "validation_molecules": len(rows_by_split["validation"]),
    "test_molecules": len(rows_by_split["test"]),
    "train_fraction": len(rows_by_split["train"]) / total_rows,
    "validation_fraction": len(rows_by_split["validation"]) / total_rows,
    "test_fraction": len(rows_by_split["test"]) / total_rows,
    "published_test_smiles": len(test_smiles),
    "published_validation_smiles_before_overlap_removal": len(
        validation_smiles_raw
    ),
    "validation_test_overlap_removed": len(validation_test_overlap),
    "published_test_smiles_found": len(found_test_smiles),
    "published_validation_smiles_found": len(found_validation_smiles),
    "published_test_smiles_not_found": len(
        test_smiles - found_test_smiles
    ),
    "published_validation_smiles_not_found": len(
        validation_smiles - found_validation_smiles
    ),
    "per_dataset": dataset_summary,
}

with (output_dir / "split_summary.json").open("w") as file:
    json.dump(summary, file, indent=2)

with (output_dir / "unmatched_published_smiles.json").open("w") as file:
    json.dump(
        {
            "test": sorted(test_smiles - found_test_smiles),
            "validation": sorted(
                validation_smiles - found_validation_smiles
            ),
        },
        file,
        indent=2,
    )

print()
print("Molecole totali:", total_rows)
print("Training:", len(rows_by_split["train"]))
print("Validation:", len(rows_by_split["validation"]))
print("Test:", len(rows_by_split["test"]))
print("Overlap validation/test rimosso:", len(validation_test_overlap))
print(
    "Test SMILES pubblicati non trovati:",
    len(test_smiles - found_test_smiles),
)
print(
    "Validation SMILES pubblicati non trovati:",
    len(validation_smiles - found_validation_smiles),
)
print()
print("Riepilogo per dataset:")

for dataset_name, values in dataset_summary.items():
    print(
        dataset_name,
        f"train={values['train']}",
        f"validation={values['validation']}",
        f"test={values['test']}",
    )

print()
print("Output:", output_dir)
