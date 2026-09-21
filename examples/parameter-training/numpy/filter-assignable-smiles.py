#!/usr/bin/env python
"""Filter SMILES to only include molecules where all atoms can be assigned BCC types."""

import argparse
import logging
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from openff.recharge.charges.bcc import BCCCollection, BCCGenerator, BCCParameter
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.toolkit import Molecule


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def can_assign_all_atoms(smiles: str, bcc_collection: BCCCollection) -> bool:
    """Check if all atoms in a molecule can be assigned BCC types.

    Parameters
    ----------
    smiles
        The SMILES string of the molecule.
    bcc_collection
        The BCC collection to use for assignment.

    Returns
    -------
        True if all atoms can be assigned, False otherwise.
    """
    try:
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        # Try to build the assignment matrix - this will raise if atoms can't be assigned
        BCCGenerator.build_assignment_matrix(molecule, bcc_collection)
        return True

    except ChargeAssignmentError as e:
        logging.warning(f"Failed to assign BCC types for {smiles}\n{e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error for {smiles}\n{e}")
        return False


def load_bcc_collection(smarts_file: Path) -> BCCCollection:
    """Load BCC collection from a SMARTS file.

    Parameters
    ----------
    smarts_file
        Path to file containing one SMARTS pattern per line.

    Returns
    -------
        BCCCollection with the specified SMARTS patterns.
    """
    with open(smarts_file) as f:
        smarts = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(smarts)} SMARTS patterns from {smarts_file}")
    return BCCCollection(parameters=[BCCParameter(smirks=smarts, value=0.0) for smarts in smarts])


def filter_smiles(
    input_file: Path,
    output_file: Path,
    bcc_collection: BCCCollection,
    n_workers: int = cpu_count(),
):
    """Filter SMILES file to only include molecules with assignable BCC types.

    Parameters
    ----------
    input_file
        Path to input file containing one SMILES per line.
    output_file
        Path to output file for filtered SMILES.
    bcc_collection
        BCC collection to use for assignment testing.
    n_workers
        Number of worker processes to use.
    verbose
        If True, log information about skipped molecules.
    """
    logging.info(f"Reading SMILES from {input_file}")

    with open(input_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    logging.info(f"Found {len(smiles_list)} molecules to filter")
    logging.info(f"Using {n_workers} worker processes")

    # Create a worker function with the BCC collection pre-bound
    worker = partial(can_assign_all_atoms, bcc_collection=bcc_collection)

    # Process in parallel
    results = []
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(worker, smiles_list),
                total=len(smiles_list),
                desc="Filtering SMILES",
                unit="molecule",
                file=sys.stdout,
            )
        )

    # Separate successful and failed molecules
    successful_smiles = [s for s, result in zip(smiles_list, results) if result]
    failed_smiles = [s for s, result in zip(smiles_list, results) if not result]

    # Write filtered SMILES
    with open(output_file, "w") as f:
        for smiles in successful_smiles:
            f.write(smiles + "\n")

    # Write failed SMILES to separate file
    if failed_smiles:
        failed_file = output_file.with_suffix(".failed" + output_file.suffix)
        with open(failed_file, "w") as f:
            for smiles in failed_smiles:
                f.write(smiles + "\n")
        logging.info(f"Wrote {len(failed_smiles)} failed molecules to {failed_file}")

    logging.info(
        f"Successfully filtered {len(successful_smiles)}/{len(smiles_list)} molecules "
        f"({100*len(successful_smiles)/len(smiles_list):.1f}%)"
    )
    logging.info(f"Output written to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter SMILES to only include molecules where all atoms can be assigned BCC types."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input file containing one SMILES per line.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output file for filtered SMILES.",
    )
    parser.add_argument(
        "bcc_smarts_file",
        type=Path,
        help="File containing one SMARTS pattern per line defining BCC types.",
    )
    parser.add_argument(
        "-n",
        "--n-workers",
        type=int,
        default=cpu_count(),
        help=f"Number of worker processes to use (default: all available CPUs, {cpu_count()}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.info("Loading BCC collection")
    bcc_collection = load_bcc_collection(args.bcc_smarts_file)

    logging.info("Filtering SMILES")
    filter_smiles(
        args.input_file,
        args.output_file,
        bcc_collection,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
