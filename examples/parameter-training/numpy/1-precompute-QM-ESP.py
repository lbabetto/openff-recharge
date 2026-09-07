import argparse
import logging
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

from openff.toolkit import Quantity
from openff.toolkit.topology import Molecule
from tqdm import tqdm

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.grids import LatticeGridSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def generate_conformers(smiles: str) -> list[tuple[Molecule, Quantity]]:
    try:
        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        conformers = ConformerGenerator.generate(
            molecule,
            ConformerSettings(
                method="rdkit",
                max_conformers=10,
            ),
        )

    except Exception as error:
        logging.error(f"Exception occurred for SMILES {smiles}:\n{error}")
        return []

    return [(molecule, conformer) for conformer in conformers]


def compute_esp(
    molecule_conformer: tuple[Molecule, Quantity], esp_settings: ESPSettings
) -> MoleculeESPRecord | None:
    molecule, conformer = molecule_conformer

    try:
        conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
            molecule=molecule,
            conformer=conformer,
            settings=esp_settings,
            # Minimize the input conformer prior to evaluating the ESP / EF
            minimize=True,
            n_threads=1,
        )
        return MoleculeESPRecord.from_molecule(molecule, conformer, grid, esp, electric_field, esp_settings)

    except (Exception, Psi4Error) as error:
        logging.error(f"Exception occurred for a conformer of {molecule.to_smiles()}:\n{error}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute reference QM ESP data for a set of molecules.")
    parser.add_argument(
        "training_data_file",
        type=Path,
        help="A file containing one SMILES pattern per line.",
    )
    parser.add_argument(
        "-n",
        "--n-workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes to use (default: all available CPUs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.training_data_file) as f:
        training_set = [line.strip() for line in f if line.strip()]

    # Define the grid that the electrostatic properties will be trained on and the
    # level of theory to compute the properties at.
    grid_settings = LatticeGridSettings(
        type="fcc", spacing=0.5, inner_vdw_scale=1.4, outer_vdw_scale=2.0
    )

    # Generate reference QC data for each molecule in the set.
    esp_settings = ESPSettings(
        method="hf",
        basis="6-31G*",
        grid_settings=grid_settings,
    )

    # Conformer generation is cheap, so do it up front and flatten the
    # (molecule, conformer) pairs so every conformer of every molecule gets
    # its own task in the pool below, rather than parallelizing only over
    # molecules and looping over their conformers serially within a worker.
    conformer_tasks = [
        task
        for smiles in tqdm(training_set, desc="Generating conformers", file=sys.stdout)
        for task in generate_conformers(smiles)
    ]

    worker = partial(compute_esp, esp_settings=esp_settings)

    with Pool(processes=args.n_workers) as pool:
        records = list(
            tqdm(
                pool.imap(worker, conformer_tasks),
                total=len(conformer_tasks),
                desc="Computing ESPs",
                file=sys.stdout,
            )
        )

    qc_data_records = [record for record in records if record is not None]
    logging.info(
        f"Successfully computed ESPs for {len(qc_data_records)} / {len(conformer_tasks)} conformers "
        f"({len(training_set)} molecules)"
    )

    output_file = args.training_data_file.with_suffix(".sqlite")
    output_file.unlink(missing_ok=True)

    qc_data_store = MoleculeESPStore(str(output_file))
    qc_data_store.store(*qc_data_records)
    logging.info(f"Saved QM ESP data to {output_file}")


if __name__ == "__main__":
    main()
