import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openff.recharge.charges.bcc import (
    BCCGenerator,
    original_am1bcc_corrections,
)
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.toolkit import Molecule
from openff.units import unit
from rdkit import Chem


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path, max_molecules):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if max_molecules is not None:
        rows = rows[:max_molecules]

    return rows


def load_molecule(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        mapped_smiles = str(data["mapped_smiles"].item())
        atomic_numbers = np.asarray(data["atomic_numbers"], dtype=int)

    molecule = Molecule.from_mapped_smiles(
        mapped_smiles,
        allow_undefined_stereo=True,
    )

    molecule_atomic_numbers = np.array(
        [atom.atomic_number for atom in molecule.atoms],
        dtype=int,
    )

    if not np.array_equal(molecule_atomic_numbers, atomic_numbers):
        raise ValueError(f"Ordine atomico incoerente in {npz_path}")

    return molecule, mapped_smiles


def formal_charge(atom):
    return int(
        round(atom.formal_charge.m_as(unit.elementary_charge))
    )


def atom_map_number(atom):
    metadata = getattr(atom, "metadata", {})
    return metadata.get("atom_map", "")


def bond_indices(molecule, bond):
    return (
        molecule.atom_index(bond.atom1),
        molecule.atom_index(bond.atom2),
    )


def build_adjacency(molecule):
    adjacency = defaultdict(list)
    bonds = list(molecule.bonds)

    for bond_index, bond in enumerate(bonds):
        atom1_index, atom2_index = bond_indices(molecule, bond)
        adjacency[atom1_index].append(
            (atom2_index, bond_index, bond)
        )
        adjacency[atom2_index].append(
            (atom1_index, bond_index, bond)
        )

    return adjacency, bonds


def atom_descriptor(molecule, atom_index, adjacency, periodic_table):
    atom = molecule.atoms[atom_index]

    return {
        "atom_index": atom_index,
        "atom_map": atom_map_number(atom),
        "element": periodic_table.GetElementSymbol(atom.atomic_number),
        "atomic_number": atom.atomic_number,
        "degree": len(adjacency[atom_index]),
        "formal_charge": formal_charge(atom),
        "aromatic": bool(atom.is_aromatic),
    }


def compact_atom_key(descriptor):
    return (
        f"Z{descriptor['atomic_number']}"
        f"_X{descriptor['degree']}"
        f"_q{descriptor['formal_charge']}"
        f"_a{int(descriptor['aromatic'])}"
    )


def bond_symbol(bond):
    if bond.is_aromatic:
        return ":"

    order = float(bond.bond_order)

    if np.isclose(order, 1.0):
        return "-"
    if np.isclose(order, 2.0):
        return "="
    if np.isclose(order, 3.0):
        return "#"

    return f"~{order:g}~"


def canonical_bond_environment(
    molecule,
    atom1_index,
    atom2_index,
    bond,
    adjacency,
    periodic_table,
):
    atom1 = atom_descriptor(
        molecule,
        atom1_index,
        adjacency,
        periodic_table,
    )
    atom2 = atom_descriptor(
        molecule,
        atom2_index,
        adjacency,
        periodic_table,
    )

    endpoint1 = compact_atom_key(atom1)
    endpoint2 = compact_atom_key(atom2)
    left, right = sorted((endpoint1, endpoint2))

    return f"{left}{bond_symbol(bond)}{right}"


def parse_missing_atom_indices(error):
    message = str(error)
    match = re.search(r"Atoms? (.*?) could not", message)

    if match is None:
        return []

    return sorted(set(int(value) for value in re.findall(r"\d+", match.group(1))))


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-molecules", type=int)
parser.add_argument("--progress-every", type=int, default=250)
args = parser.parse_args()

manifest_path = Path(args.manifest).resolve()
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

rows = load_manifest(manifest_path, args.max_molecules)

if not rows:
    raise RuntimeError("Il manifest non contiene molecole.")

bcc_collection = original_am1bcc_corrections()
n_parameters = len(bcc_collection.parameters)
periodic_table = Chem.GetPeriodicTable()

molecule_rows = []
uncovered_atom_rows = []
dataset_counts = defaultdict(Counter)
missing_atom_occurrences = Counter()
missing_atom_molecules = Counter()
missing_atom_examples = {}
missing_bond_occurrences = Counter()
missing_bond_molecules = Counter()
missing_bond_examples = {}
pattern_molecule_counts = np.zeros(n_parameters, dtype=np.int64)
pattern_nonzero_atom_counts = np.zeros(n_parameters, dtype=np.int64)

for position, row in enumerate(rows, start=1):
    npz_path = Path(row["path"])
    dataset = row.get("dataset") or npz_path.parent.name
    molecule, mapped_smiles = load_molecule(npz_path)
    adjacency, bonds = build_adjacency(molecule)

    dataset_counts[dataset]["total_molecules"] += 1

    try:
        assignment_matrix = BCCGenerator.build_assignment_matrix(
            molecule,
            bcc_collection,
        )

        active_indices = np.flatnonzero(
            np.any(np.abs(assignment_matrix) > 0.0, axis=0)
        )
        pattern_molecule_counts[active_indices] += 1
        pattern_nonzero_atom_counts += np.count_nonzero(
            assignment_matrix,
            axis=0,
        )

        dataset_counts[dataset]["covered_molecules"] += 1
        coverage_status = "covered"
        error_message = ""
        missing_indices = []
    except ChargeAssignmentError as error:
        dataset_counts[dataset]["uncovered_molecules"] += 1
        coverage_status = "uncovered"
        error_message = str(error)
        missing_indices = parse_missing_atom_indices(error)

        missing_atom_keys_in_molecule = set()
        missing_bond_keys_in_molecule = set()
        incident_bond_indices = set()

        for atom_index in missing_indices:
            descriptor = atom_descriptor(
                molecule,
                atom_index,
                adjacency,
                periodic_table,
            )
            atom_key = compact_atom_key(descriptor)

            missing_atom_occurrences[atom_key] += 1
            missing_atom_keys_in_molecule.add(atom_key)
            missing_atom_examples.setdefault(
                atom_key,
                {
                    "example_path": str(npz_path),
                    "example_atom_index": atom_index,
                    "example_atom_map": descriptor["atom_map"],
                },
            )

            neighbors = []

            for neighbor_index, bond_index, bond in adjacency[atom_index]:
                neighbor = atom_descriptor(
                    molecule,
                    neighbor_index,
                    adjacency,
                    periodic_table,
                )
                neighbors.append(
                    {
                        **neighbor,
                        "bond_order": float(bond.bond_order),
                        "bond_aromatic": bool(bond.is_aromatic),
                    }
                )
                incident_bond_indices.add(bond_index)

            uncovered_atom_rows.append(
                {
                    "dataset": dataset,
                    "path": str(npz_path),
                    "atom_index": atom_index,
                    "atom_map": descriptor["atom_map"],
                    "element": descriptor["element"],
                    "atomic_number": descriptor["atomic_number"],
                    "degree": descriptor["degree"],
                    "formal_charge": descriptor["formal_charge"],
                    "aromatic": descriptor["aromatic"],
                    "atom_environment": atom_key,
                    "neighbors": json.dumps(
                        sorted(
                            neighbors,
                            key=lambda item: item["atom_index"],
                        ),
                        sort_keys=True,
                    ),
                    "mapped_smiles": mapped_smiles,
                }
            )

        for atom_key in missing_atom_keys_in_molecule:
            missing_atom_molecules[atom_key] += 1

        for bond_index in incident_bond_indices:
            bond = bonds[bond_index]
            atom1_index, atom2_index = bond_indices(molecule, bond)
            bond_key = canonical_bond_environment(
                molecule,
                atom1_index,
                atom2_index,
                bond,
                adjacency,
                periodic_table,
            )

            missing_bond_occurrences[bond_key] += 1
            missing_bond_keys_in_molecule.add(bond_key)
            missing_bond_examples.setdefault(
                bond_key,
                {
                    "example_path": str(npz_path),
                    "example_atom1_index": atom1_index,
                    "example_atom2_index": atom2_index,
                },
            )

        for bond_key in missing_bond_keys_in_molecule:
            missing_bond_molecules[bond_key] += 1

    molecule_rows.append(
        {
            "manifest_position": position,
            "dataset": dataset,
            "path": str(npz_path),
            "mol_id": row.get("mol_id", ""),
            "smiles": row.get("smiles", ""),
            "n_atoms": molecule.n_atoms,
            "status": coverage_status,
            "missing_atom_indices": ";".join(
                map(str, missing_indices)
            ),
            "error": error_message,
        }
    )

    if args.progress_every > 0 and position % args.progress_every == 0:
        uncovered_so_far = sum(
            counts["uncovered_molecules"]
            for counts in dataset_counts.values()
        )
        print(
            f"Analizzate {position}/{len(rows)} molecole; "
            f"non coperte: {uncovered_so_far}",
            flush=True,
        )

total_molecules = len(rows)
covered_molecules = sum(
    counts["covered_molecules"]
    for counts in dataset_counts.values()
)
uncovered_molecules = sum(
    counts["uncovered_molecules"]
    for counts in dataset_counts.values()
)

dataset_rows = []

for dataset in sorted(dataset_counts):
    counts = dataset_counts[dataset]
    total = counts["total_molecules"]
    covered = counts["covered_molecules"]
    uncovered = counts["uncovered_molecules"]

    dataset_rows.append(
        {
            "dataset": dataset,
            "total_molecules": total,
            "covered_molecules": covered,
            "uncovered_molecules": uncovered,
            "coverage_fraction": covered / total,
        }
    )

missing_atom_environment_rows = []

for atom_key, occurrence_count in missing_atom_occurrences.most_common():
    missing_atom_environment_rows.append(
        {
            "atom_environment": atom_key,
            "occurrence_count": occurrence_count,
            "molecule_count": missing_atom_molecules[atom_key],
            **missing_atom_examples[atom_key],
        }
    )

missing_bond_environment_rows = []

for bond_key, occurrence_count in missing_bond_occurrences.most_common():
    missing_bond_environment_rows.append(
        {
            "bond_environment": bond_key,
            "occurrence_count": occurrence_count,
            "molecule_count": missing_bond_molecules[bond_key],
            **missing_bond_examples[bond_key],
        }
    )

pattern_usage_rows = []

for index, parameter in enumerate(bcc_collection.parameters):
    pattern_usage_rows.append(
        {
            "parameter_index": index,
            "smirks": parameter.smirks,
            "original_value": parameter.value,
            "covered_molecule_count": int(
                pattern_molecule_counts[index]
            ),
            "nonzero_atom_count": int(
                pattern_nonzero_atom_counts[index]
            ),
        }
    )

write_csv(
    output_dir / "coverage_by_molecule.csv",
    [
        "manifest_position",
        "dataset",
        "path",
        "mol_id",
        "smiles",
        "n_atoms",
        "status",
        "missing_atom_indices",
        "error",
    ],
    molecule_rows,
)

write_csv(
    output_dir / "uncovered_atoms.csv",
    [
        "dataset",
        "path",
        "atom_index",
        "atom_map",
        "element",
        "atomic_number",
        "degree",
        "formal_charge",
        "aromatic",
        "atom_environment",
        "neighbors",
        "mapped_smiles",
    ],
    uncovered_atom_rows,
)

write_csv(
    output_dir / "missing_atom_environments.csv",
    [
        "atom_environment",
        "occurrence_count",
        "molecule_count",
        "example_path",
        "example_atom_index",
        "example_atom_map",
    ],
    missing_atom_environment_rows,
)

write_csv(
    output_dir / "candidate_missing_bond_environments.csv",
    [
        "bond_environment",
        "occurrence_count",
        "molecule_count",
        "example_path",
        "example_atom1_index",
        "example_atom2_index",
    ],
    missing_bond_environment_rows,
)

write_csv(
    output_dir / "coverage_by_dataset.csv",
    [
        "dataset",
        "total_molecules",
        "covered_molecules",
        "uncovered_molecules",
        "coverage_fraction",
    ],
    dataset_rows,
)

write_csv(
    output_dir / "pattern_usage_covered_molecules.csv",
    [
        "parameter_index",
        "smirks",
        "original_value",
        "covered_molecule_count",
        "nonzero_atom_count",
    ],
    pattern_usage_rows,
)

summary = {
    "manifest": str(manifest_path),
    "selected_molecules": total_molecules,
    "covered_molecules": covered_molecules,
    "uncovered_molecules": uncovered_molecules,
    "coverage_fraction": covered_molecules / total_molecules,
    "uncovered_atom_occurrences": len(uncovered_atom_rows),
    "unique_missing_atom_environments": len(
        missing_atom_environment_rows
    ),
    "unique_candidate_bond_environments": len(
        missing_bond_environment_rows
    ),
    "built_in_bcc_parameters": n_parameters,
    "max_molecules": args.max_molecules,
}

with (output_dir / "coverage_summary.json").open("w") as file:
    json.dump(summary, file, indent=2)

print()
print("Molecole analizzate:", total_molecules)
print("Molecole coperte:", covered_molecules)
print("Molecole non coperte:", uncovered_molecules)
print("Copertura:", f"{100.0 * covered_molecules / total_molecules:.2f}%")
print("Atomi non assegnati:", len(uncovered_atom_rows))
print(
    "Ambienti atomici mancanti unici:",
    len(missing_atom_environment_rows),
)
print(
    "Ambienti di legame candidati unici:",
    len(missing_bond_environment_rows),
)
print("Output:", output_dir)
