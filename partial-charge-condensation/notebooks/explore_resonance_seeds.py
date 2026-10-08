# %%
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from openff.units import unit


# %%
DATASET_DIR = Path(
    "/leonardo_work/cin_staff/lquerci1/MLFF/"
    "GRAPPA/grappa-v.1.4.1/data/datasets/gen2"
)

CARBOXYLATE_SMIRKS = (
    "[#6X3:1](~[#8X1:2])~[#8X1-1:3]"
)


# %%
def scalar_string(value):
    item = np.asarray(value).reshape(-1)[0]

    if isinstance(item, bytes):
        return item.decode("utf-8")

    return str(item)

# function to load molecule data from a .npz file
def load_molecule_data(path):
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=float)
        atomic_numbers = np.asarray(
            data["atomic_numbers"],
            dtype=int,
        ).reshape(-1)
        q_reference = np.asarray(
            data["partial_charges"],
            dtype=float,
        ).reshape(-1)
        mapped_smiles = scalar_string(data["mapped_smiles"])

    molecule = Molecule.from_mapped_smiles(
        mapped_smiles,
        allow_undefined_stereo=True,
    )

    molecule_atomic_numbers = np.array(
        [atom.atomic_number for atom in molecule.atoms],
        dtype=int,
    )

    if not np.array_equal(
        molecule_atomic_numbers,
        atomic_numbers,
    ):
        raise ValueError(
            f"Ordine atomico incoerente in {path}"
        )

    q_base = np.array(
        [
            atom.formal_charge.m_as(
                unit.elementary_charge
            )
            for atom in molecule.atoms
        ],
        dtype=float,
    )

    return molecule, xyz, q_reference, q_base


# %%
# search for a molecule with a carboxylate group in the dataset
carboxylate_path = None
carboxylate_smiles = None
carboxylate_matches = None

for candidate_path in sorted(DATASET_DIR.glob("*.npz")):
    with np.load(
        candidate_path,
        allow_pickle=False,
    ) as data:
        mapped_smiles = scalar_string(
            data["mapped_smiles"]
        )

    candidate_molecule = Molecule.from_mapped_smiles(
        mapped_smiles,
        allow_undefined_stereo=True,
    )

    matches = (
        candidate_molecule.chemical_environment_matches(
            CARBOXYLATE_SMIRKS
        )
    )

    if matches:
        carboxylate_path = candidate_path
        carboxylate_smiles = mapped_smiles
        carboxylate_matches = matches
        break

if carboxylate_path is None:
    raise RuntimeError(
        f"Nessun carbossilato trovato in {DATASET_DIR}"
    )

NPZ_PATH = carboxylate_path

print("Molecola trovata:", NPZ_PATH)
print("Mapped SMILES:", carboxylate_smiles)
print("Match (C, O neutro, O anionico):", carboxylate_matches)


# %%
# load the molecule data from the .npz file
molecule, xyz, q_reference, q_base = (
    load_molecule_data(NPZ_PATH)
)

print()
print("Numero di atomi:", molecule.n_atoms)
print("Numero di conformeri:", len(xyz))
print("Cariche base:", q_base)
print("Carica totale base:", q_base.sum())
print("Carica totale di riferimento:", q_reference.sum())


# %%
# ottenere gli indici degli atomi del gruppo carbossilato
carbon_index, neutral_oxygen_index, anionic_oxygen_index = (
    carboxylate_matches[0]
)

print()
print("Primo gruppo carbossilato:")
print(
    "C:",
    carbon_index,
    "q_base =",
    q_base[carbon_index],
    "q_reference =",
    q_reference[carbon_index],
)
print(
    "O neutro:",
    neutral_oxygen_index,
    "q_base =",
    q_base[neutral_oxygen_index],
    "q_reference =",
    q_reference[neutral_oxygen_index],
)
print(
    "O anionico:",
    anionic_oxygen_index,
    "q_base =",
    q_base[anionic_oxygen_index],
    "q_reference =",
    q_reference[anionic_oxygen_index],
)
print()
print("Cariche parziali AM1-BCC-ELF10:")
print(
    np.array2string(
        q_reference,
        precision=6,
        separator=", ",
        threshold=np.inf,
    )
)
