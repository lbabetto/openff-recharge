import argparse
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
from openff.toolkit.topology import Molecule
from tqdm import tqdm

from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.exceptions import Psi4Error
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import LatticeGridSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def process_molecule(smiles: str, esp_settings: ESPSettings) -> MoleculeESPRecord | None:
    try:
        molecule = Molecule.from_smiles(smiles)

        conformers = ConformerGenerator.generate(
            molecule,
            ConformerSettings(
                method="rdkit",
                max_conformers=5,
            ),
        )

    except Exception as error:
        logging.error(f"Exception occurred for SMILES {smiles}:\n{error}")
        return None

    for conformer in conformers:
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
            logging.error(f"Exception occurred for a conformer of {smiles}:\n{error}")

    return None


def save_records(records: list[MoleculeESPRecord], output_file: Path) -> None:
    with h5py.File(output_file, "w") as h5_file:
        for i, record in enumerate(records):
            group = h5_file.create_group(f"molecule_{i:06d}")
            group.attrs["tagged_smiles"] = record.tagged_smiles
            group.attrs["esp_settings"] = record.esp_settings.model_dump_json()

            group.create_dataset("conformer", data=record.conformer)
            group.create_dataset("grid_coordinates", data=record.grid_coordinates)
            group.create_dataset("esp", data=record.esp)

            if record.electric_field is not None:
                group.create_dataset("electric_field", data=record.electric_field)


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

    # Generate reference QC data for each molecule in the set.
    esp_settings = ESPSettings(
        method="hf",
        basis="6-31G*",
        grid_settings=LatticeGridSettings(spacing=0.7),
    )
    worker = partial(process_molecule, esp_settings=esp_settings)

    with Pool(processes=args.n_workers) as pool:
        records = list(
            tqdm(
                pool.imap(worker, training_set),
                total=len(training_set),
                desc="Computing ESPs",
            )
        )

    qc_data_records = [record for record in records if record is not None]
    logging.info(f"Successfully computed ESPs for {len(qc_data_records)} / {len(training_set)} molecules")

    output_file = args.training_data_file.with_suffix(".hdf5")
    save_records(qc_data_records, output_file)
    logging.info(f"Saved QM data to {output_file}")


if __name__ == "__main__":
    main()
