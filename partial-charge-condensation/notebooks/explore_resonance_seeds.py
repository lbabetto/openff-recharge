# %%
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from openff.units import unit

# %%
NPZ_PATH = Path(
    input("Percorso del file NPZ: ").strip()
).expanduser().resolve()

# %%
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


molecule, xyz, q_reference, q_base = load_molecule_data("/mnt/c/Users/f.rizza/Desktop/dompe/datasets-grappa/gen2/328.npz")

print("Numero di atomi:", molecule.n_atoms)
print("Numero di conformeri:", len(xyz))
print("Cariche base:", q_base)
print("Carica totale:", q_base.sum())
# %%
