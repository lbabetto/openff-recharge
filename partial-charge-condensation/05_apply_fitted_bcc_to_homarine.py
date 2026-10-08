import argparse
import csv
from pathlib import Path

import numpy as np
from openff.nagl.features.atoms import AtomAverageFormalCharge
from openff.recharge.charges.bcc import BCCCollection
from openff.toolkit import Molecule
from openff.units import unit

from fast_bcc_apply import build_assignment_matrix_fast, compile_bcc_queries


HOMARINE_SMILES = "C[n+]1ccccc1C(=O)[O-]"


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_collection = (
        script_dir
        / "results"
        / "fit_resonance_random-5-conf"
        / "fitted_bcc_collection.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Apply the fitted resonance-aware BCC model to homarine."
        )
    )
    parser.add_argument(
        "--bcc-collection",
        type=Path,
        default=default_collection,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "homarine_fitted_bcc",
    )
    parser.add_argument("--smiles", default=HOMARINE_SMILES)
    return parser.parse_args()


def resonance_seed(molecule):
    tensor = AtomAverageFormalCharge().encode(molecule)
    return (
        tensor.detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(float)
    )


def formal_charges(molecule):
    return np.array(
        [
            atom.formal_charge.m_as(unit.elementary_charge)
            for atom in molecule.atoms
        ],
        dtype=float,
    )


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    collection_path = args.bcc_collection.resolve()
    output_dir = args.output_dir.resolve()

    if not collection_path.is_file():
        raise FileNotFoundError(collection_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    molecule = Molecule.from_smiles(
        args.smiles,
        allow_undefined_stereo=True,
    )
    mapped_smiles = molecule.to_smiles(
        mapped=True,
        explicit_hydrogens=True,
    )

    collection = BCCCollection.model_validate_json(
        collection_path.read_text()
    )
    compiled_queries = compile_bcc_queries(collection)
    assignment_matrix = build_assignment_matrix_fast(
        molecule,
        collection,
        compiled_queries,
    )

    parameter_values = np.array(
        [parameter.value for parameter in collection.parameters],
        dtype=float,
    )
    q_formal = formal_charges(molecule)
    q_seed = resonance_seed(molecule)
    q_bcc = assignment_matrix @ parameter_values
    q_predicted = q_seed + q_bcc

    if not np.isclose(q_seed.sum(), q_formal.sum(), atol=1.0e-8):
        raise ValueError("The resonance seed does not conserve total charge.")
    if not np.isclose(q_predicted.sum(), q_seed.sum(), atol=1.0e-8):
        raise ValueError("The BCC corrections do not conserve total charge.")

    atom_rows = []
    contribution_rows = []

    for atom_index, atom in enumerate(molecule.atoms):
        neighbors = ";".join(
            f"{neighbor.symbol}{neighbor.molecule_atom_index}"
            for neighbor in atom.bonded_atoms
        )
        atom_rows.append(
            {
                "atom_index": atom_index,
                "element": atom.symbol,
                "neighbors": neighbors,
                "formal_charge_e": q_formal[atom_index],
                "resonance_seed_e": q_seed[atom_index],
                "bcc_shift_e": q_bcc[atom_index],
                "predicted_charge_e": q_predicted[atom_index],
            }
        )

        active_parameters = np.flatnonzero(
            np.abs(assignment_matrix[atom_index]) > 0.0
        )
        for parameter_index in active_parameters:
            coefficient = assignment_matrix[atom_index, parameter_index]
            parameter = collection.parameters[parameter_index]
            contribution_rows.append(
                {
                    "atom_index": atom_index,
                    "element": atom.symbol,
                    "parameter_index": int(parameter_index),
                    "smirks": parameter.smirks,
                    "coefficient": float(coefficient),
                    "parameter_value_e": float(parameter.value),
                    "contribution_e": float(
                        coefficient * parameter.value
                    ),
                }
            )

    atom_path = output_dir / "homarine_atomic_charges.csv"
    contribution_path = output_dir / "homarine_bcc_contributions.csv"
    smiles_path = output_dir / "homarine_mapped_smiles.txt"

    write_csv(atom_path, list(atom_rows[0]), atom_rows)
    write_csv(
        contribution_path,
        [
            "atom_index",
            "element",
            "parameter_index",
            "smirks",
            "coefficient",
            "parameter_value_e",
            "contribution_e",
        ],
        contribution_rows,
    )
    smiles_path.write_text(mapped_smiles + "\n")

    print("Homarine SMILES:", args.smiles)
    print("Mapped SMILES:", mapped_smiles)
    print("BCC collection:", collection_path)
    print()
    print(
        f"{'atom':>6} {'el':>3} {'formal':>11} {'seed':>11} "
        f"{'BCC':>11} {'predicted':>11}  neighbors"
    )
    for row in atom_rows:
        print(
            f"{row['atom_index']:6d} {row['element']:>3} "
            f"{row['formal_charge_e']:+11.6f} "
            f"{row['resonance_seed_e']:+11.6f} "
            f"{row['bcc_shift_e']:+11.6f} "
            f"{row['predicted_charge_e']:+11.6f}  "
            f"{row['neighbors']}"
        )

    print()
    print(f"Formal total:    {q_formal.sum():+.10f} e")
    print(f"Seed total:      {q_seed.sum():+.10f} e")
    print(f"BCC shift total: {q_bcc.sum():+.10f} e")
    print(f"Predicted total: {q_predicted.sum():+.10f} e")
    print("Atomic charges:", atom_path)
    print("BCC contributions:", contribution_path)


if __name__ == "__main__":
    main()
