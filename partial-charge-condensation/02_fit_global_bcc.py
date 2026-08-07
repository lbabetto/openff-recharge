import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    BCCParameter,
    original_am1bcc_corrections,
)
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


def save_checkpoint(path, state, run_configuration):
    np.savez_compressed(
        path,
        hessian=state["hessian"],
        gradient=state["gradient"],
        target_ss=np.array(state["target_ss"]),
        n_grid_points=np.array(state["n_grid_points"]),
        processed_molecules=np.array(state["processed_molecules"]),
        processed_conformers=np.array(state["processed_conformers"]),
        processed_atoms=np.array(state["processed_atoms"]),
        pattern_molecule_counts=state["pattern_molecule_counts"],
        pattern_nonzero_atom_counts=state["pattern_nonzero_atom_counts"],
        run_configuration=np.array(json.dumps(run_configuration)),
    )


def load_checkpoint(path, run_configuration):
    with np.load(path, allow_pickle=False) as data:
        saved_configuration = json.loads(
            str(data["run_configuration"].item())
        )

        if saved_configuration != run_configuration:
            raise RuntimeError(
                "Il checkpoint è stato creato con una configurazione diversa."
            )

        return {
            "hessian": np.asarray(data["hessian"], dtype=float),
            "gradient": np.asarray(data["gradient"], dtype=float),
            "target_ss": float(data["target_ss"].item()),
            "n_grid_points": int(data["n_grid_points"].item()),
            "processed_molecules": int(
                data["processed_molecules"].item()
            ),
            "processed_conformers": int(
                data["processed_conformers"].item()
            ),
            "processed_atoms": int(data["processed_atoms"].item()),
            "pattern_molecule_counts": np.asarray(
                data["pattern_molecule_counts"],
                dtype=np.int64,
            ),
            "pattern_nonzero_atom_counts": np.asarray(
                data["pattern_nonzero_atom_counts"],
                dtype=np.int64,
            ),
        }


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-molecules", type=int)
parser.add_argument("--max-conformers", type=int)
parser.add_argument("--checkpoint-every", type=int, default=100)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

manifest_path = Path(args.manifest).resolve()
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

rows = load_manifest(manifest_path, args.max_molecules)

if not rows:
    raise RuntimeError("Il manifest non contiene molecole.")

bcc_template = original_am1bcc_corrections()
n_parameters = len(bcc_template.parameters)
grid_settings = MSKGridSettings()
checkpoint_path = output_dir / "fit_checkpoint.npz"

run_configuration = {
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "selected_molecules": len(rows),
    "max_molecules": args.max_molecules,
    "max_conformers": args.max_conformers,
    "n_parameters": n_parameters,
    "base_charges": "formal_charges",
    "reference_charges": "partial_charges",
    "grid": "MSKGridSettings_default",
}

if args.resume:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    state = load_checkpoint(checkpoint_path, run_configuration)
    print(
        "Ripresa dal checkpoint dopo",
        state["processed_molecules"],
        "molecole",
    )
else:
    state = {
        "hessian": np.zeros(
            (n_parameters, n_parameters),
            dtype=float,
        ),
        "gradient": np.zeros(n_parameters, dtype=float),
        "target_ss": 0.0,
        "n_grid_points": 0,
        "processed_molecules": 0,
        "processed_conformers": 0,
        "processed_atoms": 0,
        "pattern_molecule_counts": np.zeros(
            n_parameters,
            dtype=np.int64,
        ),
        "pattern_nonzero_atom_counts": np.zeros(
            n_parameters,
            dtype=np.int64,
        ),
    }

start_index = state["processed_molecules"]

for row_index in range(start_index, len(rows)):
    row = rows[row_index]
    npz_path = Path(row["path"])

    molecule, xyz, q_reference, q_base = load_molecule_data(npz_path)

    assignment_matrix = BCCGenerator.build_assignment_matrix(
        molecule,
        bcc_template,
    )

    active_indices = np.flatnonzero(
        np.any(np.abs(assignment_matrix) > 0.0, axis=0)
    )

    state["pattern_molecule_counts"][active_indices] += 1
    state["pattern_nonzero_atom_counts"] += np.count_nonzero(
        assignment_matrix,
        axis=0,
    )

    selected_conformers = conformer_indices(
        len(xyz),
        args.max_conformers,
    )

    charge_difference = q_reference - q_base

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
        target_vector = inverse_distance_matrix @ charge_difference

        state["target_ss"] += float(target_vector @ target_vector)
        state["n_grid_points"] += len(target_vector)
        state["processed_conformers"] += 1

        if len(active_indices) > 0:
            local_assignment_matrix = assignment_matrix[:, active_indices]
            design_matrix = (
                inverse_distance_matrix @ local_assignment_matrix
            )

            state["hessian"][np.ix_(active_indices, active_indices)] += (
                design_matrix.T @ design_matrix
            )
            state["gradient"][active_indices] += (
                design_matrix.T @ target_vector
            )

    state["processed_molecules"] += 1
    state["processed_atoms"] += molecule.n_atoms

    if (
        state["processed_molecules"] % args.checkpoint_every == 0
        or state["processed_molecules"] == len(rows)
    ):
        save_checkpoint(
            checkpoint_path,
            state,
            run_configuration,
        )
        print(
            f"Completate {state['processed_molecules']}/{len(rows)} "
            f"molecole, {state['processed_conformers']} conformeri"
        )

active_training_indices = np.flatnonzero(
    np.diag(state["hessian"]) > 0.0
)

if len(active_training_indices) == 0:
    raise RuntimeError("Nessun parametro BCC è stato attivato.")

local_hessian = state["hessian"][
    np.ix_(active_training_indices, active_training_indices)
]
local_gradient = state["gradient"][active_training_indices]

local_delta, residuals, rank, singular_values = np.linalg.lstsq(
    local_hessian,
    local_gradient,
    rcond=None,
)

delta = np.zeros(n_parameters, dtype=float)
delta[active_training_indices] = local_delta

training_sse_before = state["target_ss"]
training_sse_after = (
    training_sse_before
    - 2.0 * float(delta @ state["gradient"])
    + float(delta @ state["hessian"] @ delta)
)
training_sse_after = max(training_sse_after, 0.0)

training_esp_rmse_before = np.sqrt(
    training_sse_before / state["n_grid_points"]
)
training_esp_rmse_after = np.sqrt(
    training_sse_after / state["n_grid_points"]
)

charge_ss_before = 0.0
charge_ss_after = 0.0
charge_count = 0
maximum_total_charge_error = 0.0

for row in rows:
    molecule, xyz, q_reference, q_base = load_molecule_data(
        Path(row["path"])
    )
    assignment_matrix = BCCGenerator.build_assignment_matrix(
        molecule,
        bcc_template,
    )
    q_predicted = q_base + assignment_matrix @ delta

    charge_ss_before += float(
        np.sum((q_base - q_reference) ** 2)
    )
    charge_ss_after += float(
        np.sum((q_predicted - q_reference) ** 2)
    )
    charge_count += len(q_reference)
    maximum_total_charge_error = max(
        maximum_total_charge_error,
        abs(float(q_predicted.sum() - q_base.sum())),
    )

charge_rmse_before = np.sqrt(charge_ss_before / charge_count)
charge_rmse_after = np.sqrt(charge_ss_after / charge_count)

fitted_parameters = []

for parameter_index, template_parameter in enumerate(
    bcc_template.parameters
):
    provenance = {
        "source": "formal-charge BCC condensation",
        "reference_charge_key": "partial_charges",
        "training_manifest_sha256": run_configuration[
            "manifest_sha256"
        ],
        "template_parameter_index": parameter_index,
        "template_value": float(template_parameter.value),
        "template_provenance": template_parameter.provenance,
        "trained": bool(parameter_index in active_training_indices),
    }

    fitted_parameters.append(
        BCCParameter(
            smirks=template_parameter.smirks,
            value=float(delta[parameter_index]),
            provenance=provenance,
        )
    )

fitted_collection = BCCCollection(
    parameters=fitted_parameters,
    aromaticity_model=bcc_template.aromaticity_model,
)

collection_path = output_dir / "fitted_bcc_collection.json"
collection_path.write_text(fitted_collection.json(indent=2))

coverage_path = output_dir / "training_pattern_coverage.csv"

with coverage_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "parameter_index",
            "smirks",
            "template_value",
            "fitted_value",
            "trained",
            "molecule_count",
            "nonzero_atom_count",
        ],
    )
    writer.writeheader()

    for parameter_index, template_parameter in enumerate(
        bcc_template.parameters
    ):
        writer.writerow(
            {
                "parameter_index": parameter_index,
                "smirks": template_parameter.smirks,
                "template_value": float(template_parameter.value),
                "fitted_value": float(delta[parameter_index]),
                "trained": parameter_index in active_training_indices,
                "molecule_count": int(
                    state["pattern_molecule_counts"][parameter_index]
                ),
                "nonzero_atom_count": int(
                    state["pattern_nonzero_atom_counts"][parameter_index]
                ),
            }
        )

if len(singular_values) > 0 and singular_values[-1] > 0.0:
    condition_number = float(
        singular_values[0] / singular_values[-1]
    )
else:
    condition_number = None

summary = {
    **run_configuration,
    "processed_molecules": state["processed_molecules"],
    "processed_conformers": state["processed_conformers"],
    "processed_atoms": state["processed_atoms"],
    "n_grid_points": state["n_grid_points"],
    "active_training_parameters": len(active_training_indices),
    "normal_equation_rank": int(rank),
    "condition_number": condition_number,
    "training_esp_rmse_before_au": float(
        training_esp_rmse_before
    ),
    "training_esp_rmse_after_au": float(training_esp_rmse_after),
    "training_charge_rmse_before_e": float(charge_rmse_before),
    "training_charge_rmse_after_e": float(charge_rmse_after),
    "maximum_total_charge_error_e": maximum_total_charge_error,
    "fitted_collection": str(collection_path),
    "pattern_coverage": str(coverage_path),
    "checkpoint": str(checkpoint_path),
}

summary_path = output_dir / "fit_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print()
print("Molecole processate:", state["processed_molecules"])
print("Conformeri processati:", state["processed_conformers"])
print("Punti ESP:", state["n_grid_points"])
print("Parametri attivi:", len(active_training_indices))
print("Rango:", rank)
print("Numero di condizionamento:", condition_number)
print("Training ESP RMSE prima:", training_esp_rmse_before, "a.u.")
print("Training ESP RMSE dopo:", training_esp_rmse_after, "a.u.")
print("Training charge RMSE prima:", charge_rmse_before, "e")
print("Training charge RMSE dopo:", charge_rmse_after, "e")
print("Errore massimo sulla carica totale:", maximum_total_charge_error, "e")
print("Tabella BCC:", collection_path)
print("Riepilogo:", summary_path)
