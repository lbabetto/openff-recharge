import argparse
import json
from pathlib import Path

import numpy as np
from openff.recharge.charges.bcc import (
    BCCGenerator,
    original_am1bcc_corrections,
)
from openff.recharge.grids import GridGenerator, MSKGridSettings
from openff.toolkit import Molecule
from openff.units import unit


BOHR_PER_ANGSTROM = 1.8897261254578281


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output-prefix", required=True)
args = parser.parse_args()

input_path = Path(args.input)
output_prefix = Path(args.output_prefix)
output_prefix.parent.mkdir(parents=True, exist_ok=True)

with np.load(input_path, allow_pickle=False) as data:
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
    raise ValueError("L'ordine atomico della molecola non coincide con quello del file NPZ.")

q_base = np.array(
    [
        atom.formal_charge.m_as(unit.elementary_charge)
        for atom in molecule.atoms
    ],
    dtype=float,
)

bcc_collection = original_am1bcc_corrections()
assignment_matrix_full = BCCGenerator.build_assignment_matrix(
    molecule,
    bcc_collection,
)

active_indices = np.flatnonzero(
    np.any(np.abs(assignment_matrix_full) > 0.0, axis=0)
)

assignment_matrix = assignment_matrix_full[:, active_indices]
grid_settings = MSKGridSettings()

design_matrices = []
target_vectors = []
inverse_distance_matrices = []
reference_potentials = []
base_potentials = []
grid_counts = []

for conformer_index, coordinates in enumerate(xyz):
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

    reference_potential = inverse_distance_matrix @ q_reference
    base_potential = inverse_distance_matrix @ q_base

    design_matrix = inverse_distance_matrix @ assignment_matrix
    target_vector = reference_potential - base_potential

    inverse_distance_matrices.append(inverse_distance_matrix)
    reference_potentials.append(reference_potential)
    base_potentials.append(base_potential)
    design_matrices.append(design_matrix)
    target_vectors.append(target_vector)
    grid_counts.append(len(grid_angstrom))

design_matrix_global = np.vstack(design_matrices)
target_vector_global = np.concatenate(target_vectors)

delta, residuals, rank, singular_values = np.linalg.lstsq(
    design_matrix_global,
    target_vector_global,
    rcond=None,
)

q_predicted = q_base + assignment_matrix @ delta

rmse_before = []
rmse_after = []

for inverse_distance_matrix, reference_potential, base_potential in zip(
    inverse_distance_matrices,
    reference_potentials,
    base_potentials,
):
    predicted_potential = inverse_distance_matrix @ q_predicted

    rmse_before.append(
        np.sqrt(np.mean((base_potential - reference_potential) ** 2))
    )
    rmse_after.append(
        np.sqrt(np.mean((predicted_potential - reference_potential) ** 2))
    )

reference_potential_global = np.concatenate(reference_potentials)
base_potential_global = np.concatenate(base_potentials)

predicted_potential_global = np.concatenate(
    [
        matrix @ q_predicted
        for matrix in inverse_distance_matrices
    ]
)

global_rmse_before = np.sqrt(
    np.mean(
        (base_potential_global - reference_potential_global) ** 2
    )
)

global_rmse_after = np.sqrt(
    np.mean(
        (predicted_potential_global - reference_potential_global) ** 2
    )
)

charge_rmse = np.sqrt(
    np.mean((q_predicted - q_reference) ** 2)
)

fitted_parameters = []

for parameter_index, fitted_value in zip(active_indices, delta):
    parameter = bcc_collection.parameters[parameter_index]

    fitted_parameters.append(
        {
            "parameter_index": int(parameter_index),
            "smirks": parameter.smirks,
            "original_value": float(parameter.value),
            "fitted_value": float(fitted_value),
            "provenance": parameter.provenance,
        }
    )

json_path = Path(f"{output_prefix}-fitted-bcc.json")
results_path = Path(f"{output_prefix}-fit.npz")

with json_path.open("w") as file:
    json.dump(fitted_parameters, file, indent=2)

np.savez_compressed(
    results_path,
    active_indices=active_indices,
    delta=delta,
    singular_values=singular_values,
    q_reference=q_reference,
    q_base=q_base,
    q_predicted=q_predicted,
    grid_counts=np.asarray(grid_counts),
    rmse_before=np.asarray(rmse_before),
    rmse_after=np.asarray(rmse_after),
    global_rmse_before=global_rmse_before,
    global_rmse_after=global_rmse_after,
    charge_rmse=charge_rmse,
)

print("Molecola:", input_path)
print("Conformeri:", len(xyz))
print("Punti ESP totali:", sum(grid_counts))
print("Parametri BCC totali:", len(bcc_collection.parameters))
print("Parametri attivi:", len(active_indices))
print("Rango della matrice:", rank)
print("Somma q_reference:", q_reference.sum())
print("Somma q_base:", q_base.sum())
print("Somma q_predicted:", q_predicted.sum())
print("ESP RMSE globale prima:", global_rmse_before, "a.u.")
print("ESP RMSE globale dopo:", global_rmse_after, "a.u.")
print("ESP RMSE medio per conformero prima:", np.mean(rmse_before), "a.u.")
print("ESP RMSE medio per conformero dopo:", np.mean(rmse_after), "a.u.")
print("Charge RMSE:", charge_rmse, "e")
print()
print("Delta ottimizzati:")

for parameter in fitted_parameters:
    print(parameter["smirks"], parameter["fitted_value"])

print()
print("Tabella:", json_path)
print("Risultati:", results_path)
