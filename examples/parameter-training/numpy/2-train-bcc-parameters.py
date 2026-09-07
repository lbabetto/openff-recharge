import argparse
import logging
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy
from tqdm import tqdm

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.qc import QCChargeSettings
from openff.recharge.esp.storage import MoleculeESPRecord, MoleculeESPStore
from openff.recharge.optimize import ESPObjective, ESPObjectiveTerm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_objective_term(
    esp_record: MoleculeESPRecord,
    bcc_collection: BCCCollection,
    bcc_parameter_keys: list[str],
):
    # Each esp_record is independent, so run compute_objective_terms on a single
    # record at a time to parallelize the (comparatively expensive) AM1 charge
    # calculation it performs internally for every esp_record it is given.
    return next(
        ESPObjective.compute_objective_terms(
            esp_records=[esp_record],
            charge_collection=QCChargeSettings(theory="am1"),
            bcc_collection=bcc_collection,
            bcc_parameter_keys=bcc_parameter_keys,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BCC parameters against previously precomputed QM ESP data.")
    parser.add_argument(
        "qc_data_file",
        type=Path,
        help="The MoleculeESPStore SQLite file produced by 1-precompute-QM-ESP.py.",
    )
    parser.add_argument(
        "bcc_smarts_file",
        type=Path,
        help="A file containing one SMARTS pattern per line defining the BCC parameters to train.",
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

    qc_data_store = MoleculeESPStore(str(args.qc_data_file))
    qc_data_records = qc_data_store.retrieve()
    logging.info(f"Loaded {len(qc_data_records)} QC data records from {args.qc_data_file}")

    # Define a set of parameters to train
    with open(args.bcc_smarts_file) as f:
        bcc_smarts = [line.strip() for line in f if line.strip()]

    bcc_collection = BCCCollection(parameters=[BCCParameter(smirks=smarts, value=0.0) for smarts in bcc_smarts])
    bcc_parameters_to_train = list(bcc_smarts)

    # Construct the terms in our objective function that we will aim to minimize. See
    # also the ``ElectricFieldObjective`` objective class. Each term only depends on
    # its own esp_record, so compute them in parallel across conformers.
    worker = partial(
        compute_objective_term,
        bcc_collection=bcc_collection,
        bcc_parameter_keys=bcc_parameters_to_train,
    )

    with Pool(processes=args.n_workers) as pool:
        objective_terms = list(
            tqdm(
                pool.imap(worker, qc_data_records),
                total=len(qc_data_records),
                desc="Computing objective terms",
                file=sys.stdout,
            )
        )

    # Combine all the terms in our objective function (i.e. the difference between
    # the reference and predicted ESP values for each molecule in each conformer) into
    # a single object.
    objective_term = ESPObjectiveTerm.combine(*objective_terms)

    # Train the parameters.
    trained_values, *_ = numpy.linalg.lstsq(
        objective_term.atom_charge_design_matrix,
        objective_term.reference_values,
        rcond=None,
    )

    print("TRAINED PARAMETERS".center(80, "-"))

    for parameter_smirks, trained_value in zip(bcc_parameters_to_train, trained_values):
        print(f"{parameter_smirks:<48}  INITIAL=0.0000  FINAL={float(trained_value[0]):.4f}")


if __name__ == "__main__":
    main()
