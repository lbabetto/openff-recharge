"""A conservative, RDKit-only fast path for OpenFF Recharge BCC assignment.

This module preserves the matching order and validation used by
``BCCGenerator.build_assignment_matrix`` while avoiding repeated conversion of
the same OpenFF molecule and repeated compilation of the same SMIRKS patterns.

It is intentionally separate from the OpenFF Recharge source tree so that it
can first be validated as a benchmark prototype without modifying the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

from openff.recharge.aromaticity import AromaticityModel
from openff.recharge.charges.bcc import BCCGenerator

if TYPE_CHECKING:
    from openff.recharge.charges.bcc import BCCCollection
    from openff.toolkit import Molecule


@dataclass(frozen=True)
class CompiledBCCQuery:
    """A compiled RDKit query and its mapped query-atom positions."""

    query: Chem.Mol
    mapped_positions: tuple[tuple[int, int], ...]


def compile_bcc_queries(bcc_collection: "BCCCollection") -> list[CompiledBCCQuery]:
    """Compile every BCC SMIRKS exactly once, preserving collection order."""

    compiled = []

    for parameter in bcc_collection.parameters:
        query = Chem.MolFromSmarts(parameter.smirks)
        if query is None:
            raise ValueError(f"Failed to parse BCC SMIRKS: {parameter.smirks}")

        mapped_positions = tuple(
            (query_index, query_atom.GetAtomMapNum() - 1)
            for query_index, query_atom in enumerate(query.GetAtoms())
            if query_atom.GetAtomMapNum() != 0
        )

        compiled.append(
            CompiledBCCQuery(
                query=query,
                mapped_positions=mapped_positions,
            )
        )

    return compiled


def _bond_key(atom_index_a: int, atom_index_b: int) -> tuple[int, int]:
    return tuple(sorted((atom_index_a, atom_index_b)))


def _prepare_rdkit_molecule(
    molecule: "Molecule",
    bcc_collection: "BCCCollection",
) -> Chem.Mol:
    """Convert and prepare one molecule exactly as Recharge's RDKit matcher."""

    is_atom_aromatic, is_bond_aromatic = AromaticityModel.apply(
        molecule,
        bcc_collection.aromaticity_model,
    )

    rd_molecule: Chem.Mol = molecule.to_rdkit()
    Chem.SanitizeMol(
        rd_molecule,
        Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY,
    )

    rd_atoms = {atom.GetIdx(): atom for atom in rd_molecule.GetAtoms()}
    rd_bonds = {
        _bond_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()): bond
        for bond in rd_molecule.GetBonds()
    }

    for atom_index, is_aromatic in is_atom_aromatic.items():
        rd_atoms[atom_index].SetIsAromatic(is_aromatic)

    for atom_indices, is_aromatic in is_bond_aromatic.items():
        rd_bonds[_bond_key(*atom_indices)].SetIsAromatic(is_aromatic)

    return rd_molecule


def build_assignment_matrix_fast(
    molecule: "Molecule",
    bcc_collection: "BCCCollection",
    compiled_queries: list[CompiledBCCQuery],
) -> np.ndarray:
    """Build a BCC assignment matrix with the same semantics as Recharge."""

    parameters = bcc_collection.parameters
    if len(compiled_queries) != len(parameters):
        raise ValueError(
            "The number of compiled queries does not match the BCC collection"
        )

    rd_molecule = _prepare_rdkit_molecule(molecule, bcc_collection)
    max_matches = np.iinfo(np.uintc).max

    assignment_matrix = np.zeros((molecule.n_atoms, len(parameters)))
    bcc_counts_matrix = np.zeros((molecule.n_atoms, len(parameters)))
    matched_bonds: set[tuple[int, int]] = set()

    for parameter_index, compiled_query in enumerate(compiled_queries):
        full_matches = rd_molecule.GetSubstructMatches(
            compiled_query.query,
            uniquify=False,
            maxMatches=max_matches,
            useChirality=True,
        )

        for match in full_matches:
            matched_indices = {
                map_index: match[query_index]
                for query_index, map_index in compiled_query.mapped_positions
            }

            forward_matched_bond = (matched_indices[0], matched_indices[1])
            reverse_matched_bond = (matched_indices[1], matched_indices[0])

            if (
                forward_matched_bond in matched_bonds
                or reverse_matched_bond in matched_bonds
            ):
                continue

            assignment_matrix[matched_indices[0], parameter_index] += 1
            assignment_matrix[matched_indices[1], parameter_index] -= 1

            bcc_counts_matrix[matched_indices[0], parameter_index] += 1
            bcc_counts_matrix[matched_indices[1], parameter_index] += 1

            matched_bonds.add(forward_matched_bond)
            matched_bonds.add(reverse_matched_bond)

    BCCGenerator._validate_assignment_matrix(
        molecule,
        assignment_matrix,
        bcc_counts_matrix,
        bcc_collection,
    )

    return assignment_matrix
