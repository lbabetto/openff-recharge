import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.recharge.grids import GridGenerator, MSKGridSettings
from openff.toolkit import Molecule
from openff.units import unit


BOHR_PER_ANGSTROM = 1.8897261254578281


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_manifest(path, max_molecules):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if max_molecules is not None:
        rows = rows[:max_molecules]

    return rows


def conformer_indices(n_conformers, max_conformers):
    if max_conformers is None or max_conformers >= n_conformers:
        return np.arange(n_conformers, dtype=int)

    return np.linspace(
        0,
        n_conformers - 1,
        num=max_conformers,
        dtype=int,
    )


def load_molecule_data(path):
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=float)
        atomic_numbers = np.asarray(data["atomic_numbers"], dtype=int)
        q_reference = np.asarray(data["partial_charges"], dtype=float)
        mapped_smiles = str(data["mapped_smiles"].item())

    molecule = Molecule.from_mapped_smiles(
        mapped_smiles,
        allow_undefined_stereo=True,
    )

    molecule_atomic_numbers = np.array(
        [atom.atomic_number for atom in molecule.atoms],
        dtype=int,
    )

    if not np.array_equal(molecule_atomic_numbers, atomic_numbers):
        raise ValueError(f"Ordine atomico incoerente in {path}")

    q_base = np.array(
        [
            atom.formal_charge.m_as(unit.elementary_charge)
            for atom in molecule.atoms
        ],
        dtype=float,
    )

    return molecule, xyz, q_reference, q_base


def new_metric_state():
    return {
        "molecules": 0,
        "conformers": 0,
        "atoms": 0,
        "grid_points": 0,
        "esp_ss_before": 0.0,
        "esp_ss_after": 0.0,
        "charge_ss_before": 0.0,
        "charge_ss_after": 0.0,
        "maximum_total_charge_error": 0.0,
    }


def add_metrics(target, source):
    for key in [
        "molecules",
        "conformers",
        "atoms",
        "grid_points",
        "esp_ss_before",
        "esp_ss_after",
        "charge_ss_before",
        "charge_ss_after",
    ]:
        target[key] += source[key]

    target["maximum_total_charge_error"] = max(
        target["maximum_total_charge_error"],
        source["maximum_total_charge_error"],
    )


def finalized_metrics(state):
    return {
        "molecules": state["molecules"],
        "conformers": state["conformers"],
        "atoms": state["atoms"],
        "grid_points": state["grid_points"],
        "esp_rmse_before_au": float(
            np.sqrt(state["esp_ss_before"] / state["grid_points"])
        ),
        "esp_rmse_after_au": float(
            np.sqrt(state["esp_ss_after"] / state["grid_points"])
        ),
        "charge_rmse_before_e": float(
            np.sqrt(state["charge_ss_before"] / state["atoms"])
        ),
        "charge_rmse_after_e": float(
            np.sqrt(state["charge_ss_after"] / state["atoms"])
        ),
        "maximum_total_charge_error_e": float(
            state["maximum_total_charge_error"]
        ),
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--bcc-collection", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-molecules", type=int)
parser.add_argument("--max-conformers", type=int, default=1)
parser.add_argument("--progress-every", type=int, default=100)
args = parser.parse_args()

manifest_path = Path(args.manifest).resolve()
collection_path = Path(args.bcc_collection).resolve()
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

rows = load_manifest(manifest_path, args.max_molecules)

if not rows:
    raise RuntimeError("Il manifest non contiene molecole.")

bcc_collection = BCCCollection.parse_raw(collection_path.read_text())
parameter_values = np.array(
    [parameter.value for parameter in bcc_collection.parameters],
    dtype=float,
)
n_parameters = len(parameter_values)
grid_settings = MSKGridSettings()

overall_state = new_metric_state()
dataset_states = defaultdict(new_metric_state)
per_molecule_rows = []
pattern_molecule_counts = np.zeros(n_parameters, dtype=np.int64)
pattern_nonzero_atom_counts = np.zeros(n_parameters, dtype=np.int64)

for position, row in enumerate(rows, start=1):
    npz_path = Path(row["path"])
    dataset = row.get("dataset") or npz_path.parent.name
    molecule, xyz, q_reference, q_base = load_molecule_data(npz_path)

    try:
        assignment_matrix = BCCGenerator.build_assignment_matrix(
            molecule,
            bcc_collection,
        )
    except ChargeAssignmentError as error:
        raise RuntimeError(
            f"Molecola non coperta nel manifest di valutazione: {npz_path}"
        ) from error

    active_indices = np.flatnonzero(
        np.any(np.abs(assignment_matrix) > 0.0, axis=0)
    )
    pattern_molecule_counts[active_indices] += 1
    pattern_nonzero_atom_counts += np.count_nonzero(
        assignment_matrix,
        axis=0,
    )

    q_predicted = q_base + assignment_matrix @ parameter_values
    selected_conformers = conformer_indices(
        len(xyz),
        args.max_conformers,
    )

    molecule_state = new_metric_state()
    molecule_state["molecules"] = 1
    molecule_state["atoms"] = molecule.n_atoms
    molecule_state["charge_ss_before"] = float(
        np.sum((q_base - q_reference) ** 2)
    )
    molecule_state["charge_ss_after"] = float(
        np.sum((q_predicted - q_reference) ** 2)
    )
    molecule_state["maximum_total_charge_error"] = abs(
        float(q_predicted.sum() - q_base.sum())
    )

    for conformer_index in selected_conformers:
        coordinates = xyz[conformer_index]
        grid = GridGenerator.generate(
            molecule,
            coordinates * unit.angstrom,
            grid_settings,
        )

        grid_angstrom = grid.m_as(unit.angstrom)
        displacements_bohr = (
            grid_angstrom[:, None, :] - coordinates[None, :, :]
        ) * BOHR_PER_ANGSTROM
        distances_bohr = np.linalg.norm(displacements_bohr, axis=2)
        inverse_distance_matrix = 1.0 / distances_bohr

        esp_reference = inverse_distance_matrix @ q_reference
        esp_base = inverse_distance_matrix @ q_base
        esp_predicted = inverse_distance_matrix @ q_predicted

        molecule_state["esp_ss_before"] += float(
            np.sum((esp_base - esp_reference) ** 2)
        )
        molecule_state["esp_ss_after"] += float(
            np.sum((esp_predicted - esp_reference) ** 2)
        )
        molecule_state["grid_points"] += len(grid_angstrom)
        molecule_state["conformers"] += 1

    add_metrics(overall_state, molecule_state)
    add_metrics(dataset_states[dataset], molecule_state)

    per_molecule_rows.append(
        {
            "dataset": dataset,
            "path": str(npz_path),
            "mol_id": row.get("mol_id", ""),
            "smiles": row.get("smiles", ""),
            "n_atoms": molecule_state["atoms"],
            "n_conformers": molecule_state["conformers"],
            "n_grid_points": molecule_state["grid_points"],
            "esp_rmse_before_au": np.sqrt(
                molecule_state["esp_ss_before"]
                / molecule_state["grid_points"]
            ),
            "esp_rmse_after_au": np.sqrt(
                molecule_state["esp_ss_after"]
                / molecule_state["grid_points"]
            ),
            "charge_rmse_before_e": np.sqrt(
                molecule_state["charge_ss_before"]
                / molecule_state["atoms"]
            ),
            "charge_rmse_after_e": np.sqrt(
                molecule_state["charge_ss_after"]
                / molecule_state["atoms"]
            ),
            "total_charge_error_e": molecule_state[
                "maximum_total_charge_error"
            ],
        }
    )

    if args.progress_every > 0 and position % args.progress_every == 0:
        print(
            f"Valutate {position}/{len(rows)} molecole",
            flush=True,
        )

dataset_rows = []

for dataset in sorted(dataset_states):
    dataset_rows.append(
        {
            "dataset": dataset,
            **finalized_metrics(dataset_states[dataset]),
        }
    )

pattern_rows = []
test_active_parameters = 0
test_active_untrained_parameters = 0

for index, parameter in enumerate(bcc_collection.parameters):
    provenance = parameter.provenance or {}
    trained = bool(provenance.get("trained", False))
    active_in_test = pattern_molecule_counts[index] > 0

    if active_in_test:
        test_active_parameters += 1

    if active_in_test and not trained:
        test_active_untrained_parameters += 1

    pattern_rows.append(
        {
            "parameter_index": index,
            "smirks": parameter.smirks,
            "fitted_value": float(parameter.value),
            "trained": trained,
            "active_in_test": bool(active_in_test),
            "test_molecule_count": int(
                pattern_molecule_counts[index]
            ),
            "test_nonzero_atom_count": int(
                pattern_nonzero_atom_counts[index]
            ),
        }
    )

overall_metrics = finalized_metrics(overall_state)
summary = {
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "bcc_collection": str(collection_path),
    "bcc_collection_sha256": sha256(collection_path),
    "selected_molecules": len(rows),
    "max_molecules": args.max_molecules,
    "max_conformers": args.max_conformers,
    "base_charges": "formal_charges",
    "reference_charges": "partial_charges",
    "grid": "MSKGridSettings_default",
    "bcc_parameters": n_parameters,
    "test_active_parameters": test_active_parameters,
    "test_active_untrained_parameters": (
        test_active_untrained_parameters
    ),
    **overall_metrics,
}

write_csv(
    output_dir / "test_metrics_by_molecule.csv",
    [
        "dataset",
        "path",
        "mol_id",
        "smiles",
        "n_atoms",
        "n_conformers",
        "n_grid_points",
        "esp_rmse_before_au",
        "esp_rmse_after_au",
        "charge_rmse_before_e",
        "charge_rmse_after_e",
        "total_charge_error_e",
    ],
    per_molecule_rows,
)

write_csv(
    output_dir / "test_metrics_by_dataset.csv",
    [
        "dataset",
        "molecules",
        "conformers",
        "atoms",
        "grid_points",
        "esp_rmse_before_au",
        "esp_rmse_after_au",
        "charge_rmse_before_e",
        "charge_rmse_after_e",
        "maximum_total_charge_error_e",
    ],
    dataset_rows,
)

write_csv(
    output_dir / "test_pattern_coverage.csv",
    [
        "parameter_index",
        "smirks",
        "fitted_value",
        "trained",
        "active_in_test",
        "test_molecule_count",
        "test_nonzero_atom_count",
    ],
    pattern_rows,
)

summary_path = output_dir / "evaluation_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print()
print("Molecole valutate:", overall_metrics["molecules"])
print("Conformeri valutati:", overall_metrics["conformers"])
print("Punti ESP:", overall_metrics["grid_points"])
print("Parametri attivi nel test:", test_active_parameters)
print(
    "Parametri attivi nel test ma non allenati:",
    test_active_untrained_parameters,
)
print(
    "Test ESP RMSE prima:",
    overall_metrics["esp_rmse_before_au"],
    "a.u.",
)
print(
    "Test ESP RMSE dopo:",
    overall_metrics["esp_rmse_after_au"],
    "a.u.",
)
print(
    "Test charge RMSE prima:",
    overall_metrics["charge_rmse_before_e"],
    "e",
)
print(
    "Test charge RMSE dopo:",
    overall_metrics["charge_rmse_after_e"],
    "e",
)
print(
    "Errore massimo sulla carica totale:",
    overall_metrics["maximum_total_charge_error_e"],
    "e",
)
print("Riepilogo:", summary_path)
