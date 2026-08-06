import numpy
from openff.toolkit.topology import Molecule
from tqdm import tqdm

from openff.recharge.charges.bcc import BCCCollection, BCCParameter
from openff.recharge.charges.qc import QCChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import LatticeGridSettings
from openff.recharge.optimize import ESPObjective, ESPObjectiveTerm

import sys
import os
import logging


def main():
    # Load in the molecules to train
    training_set = []
    training_data_file = sys.argv[1]
    with open(training_data_file, "r") as f:
        for line in f:
            training_set.append(line.strip())

    # Generate reference QC data for each molecule in the set.
    qc_data_settings = ESPSettings(
        method="hf",
        basis="6-31G*",
        grid_settings=LatticeGridSettings(spacing=0.7),
    )
    qc_data_records = []

    for smiles in tqdm(training_set):
        try:
            molecule = Molecule.from_smiles(smiles)

            conformers = ConformerGenerator.generate(
                molecule,
                ConformerSettings(
                    method="rdkit",
                    max_conformers=5,
                ),
            )

        except Exception as e:
            logging.error(f"Exception occurred for SMILES {smiles}:\n {e}")
            continue

        for conformer in tqdm(conformers):
            try:
                conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
                    molecule=molecule,
                    conformer=conformer,
                    settings=qc_data_settings,
                    # Minimize the input conformer prior to evaluating the ESP / EF
                    minimize=True,
                    n_threads=os.cpu_count(),
                )
                qc_data_record = MoleculeESPRecord.from_molecule(
                    molecule, conformer, grid, esp, electric_field, qc_data_settings
                )

                qc_data_records.append(qc_data_record)

            except Exception as e:
                logging.error(f"Exception occurred for conformer {conformer}:\n {e}")
                continue

    # Define a set of parameters to train
    bcc_smarts_file = sys.argv[2]
    bcc_smarts = []
    with open(bcc_smarts_file, "r") as f:
        for line in f:
            bcc_smarts.append(line.strip())

    bcc_collection = BCCCollection(parameters=[BCCParameter(smirks=smarts, value=0.0) for smarts in bcc_smarts])
    bcc_parameters_to_train = [smarts for smarts in bcc_smarts]

    # Construct the terms in our objective function that we will aim to minimize. See
    # also the ``ElectricFieldObjective`` objective class.
    objective_terms_generator = ESPObjective.compute_objective_terms(
        esp_records=qc_data_records,
        # Here we use AM1-mulliken charges as the base charges to correct.
        charge_collection=QCChargeSettings(theory="am1"),
        bcc_collection=bcc_collection,
        bcc_parameter_keys=bcc_parameters_to_train,
    )
    # Combine all the terms in our objective function (i.e. the difference between
    # the reference and predicted ESP values for each molecule in each conformer) into
    # a single object.
    objective_term = ESPObjectiveTerm.combine(*objective_terms_generator)

    # Train the parameters.
    trained_values, *_ = numpy.linalg.lstsq(
        objective_term.atom_charge_design_matrix,
        objective_term.reference_values,
        rcond=None,
    )

    print("TRAINED PARAMETERS".center(80, "-"))

    for parameter_smirks, trained_value in zip(bcc_parameters_to_train, trained_values):
        print(
            f"{parameter_smirks:<48}".format("left aligned"),
            f"  INITIAL={0.0:.4f}".format("left aligned"),
            f"  FINAL={float(trained_value[0]):.4f}".format("left aligned"),
        )


if __name__ == "__main__":
    main()
