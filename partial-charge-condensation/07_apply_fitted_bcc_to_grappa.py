import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from openff.recharge.charges.bcc import BCCCollection, BCCGenerator
from openff.recharge.charges.exceptions import ChargeAssignmentError
from openff.toolkit import Molecule
from openff.units import unit


DEFAULT_DATASETS = [
    "gen2",
    "gen2-torsion",
    "pepconf-dlc",
    "protein-torsion",
    "spice-pubchem",
    "spice-dipeptide",
    "spice-des-monomers",
    "rna-nucleoside",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_collection = (
        script_dir / "results" / "fit" / "fitted_bcc_collection.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Apply the fitted formal-charge BCC condensation model to "
            "Grappa-format NPZ datasets. Every input array is preserved and "
            "only partial_charges is replaced."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory containing the input Grappa dataset directories.",
    )
    parser.add_argument(
        "--output-root",
        help="Output parent directory. Default: the input dataset root.",
    )
    parser.add_argument(
        "--bcc-collection",
        default=str(default_collection),
        help="Fitted BCCCollection JSON.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names. Default: the eight datasets used in the fit workflow.",
    )
    parser.add_argument(
        "--suffix",
        default="-recharge-condensed",
        help="Suffix added to each output dataset directory.",
    )
    parser.add_argument(
        "--report-dir",
        help=(
            "Directory for application_manifest.csv and application_summary.json. "
            "Default: a sibling of output-root."
        ),
    )
    parser.add_argument(
        "--neutral-only",
        action="store_true",
        help="Write only molecules with zero net formal charge.",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        help="Process at most this many NPZ files per dataset (for testing).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N input molecules.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def natural_sort_key(path):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def scalar_string(value):
    array = np.asarray(value)

    if array.size != 1:
        raise ValueError(f"Expected one string value, found shape {array.shape}.")

    item = array.reshape(-1)[0]

    if isinstance(item, bytes):
        return item.decode("utf-8")

    return str(item)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.array(data[key], copy=True) for key in data.files}

    required_keys = {"mapped_smiles", "atomic_numbers", "partial_charges"}
    missing_keys = sorted(required_keys - payload.keys())

    if missing_keys:
        raise KeyError(f"{path} is missing keys: {', '.join(missing_keys)}")

    return payload


def molecule_id(payload, fallback):
    if "mol_id" not in payload:
        return fallback

    return scalar_string(payload["mol_id"])


def smiles(payload):
    if "smiles" not in payload:
        return ""

    return scalar_string(payload["smiles"])


def build_molecule(payload, path):
    mapped_smiles = scalar_string(payload["mapped_smiles"])
    molecule = Molecule.from_mapped_smiles(
        mapped_smiles,
        allow_undefined_stereo=True,
    )
    expected_atomic_numbers = np.asarray(
        payload["atomic_numbers"],
        dtype=int,
    ).reshape(-1)
    actual_atomic_numbers = np.array(
        [atom.atomic_number for atom in molecule.atoms],
        dtype=int,
    )

    if not np.array_equal(actual_atomic_numbers, expected_atomic_numbers):
        raise ValueError(f"Atomic order mismatch in {path}")

    q_seed = np.array(
        [
            atom.formal_charge.m_as(unit.elementary_charge)
            for atom in molecule.atoms
        ],
        dtype=float,
    )
    return molecule, mapped_smiles, q_seed


def trained_flags(collection):
    flags = []

    for index, parameter in enumerate(collection.parameters):
        provenance = parameter.provenance or {}

        if "trained" not in provenance:
            raise ValueError(
                f"BCC parameter {index} has no provenance.trained flag. "
                "Use the fitted collection produced by 02_fit_global_bcc.py."
            )

        flags.append(bool(provenance["trained"]))

    return np.asarray(flags, dtype=bool)


def charge_output_dtype(original):
    dtype = np.asarray(original).dtype

    if np.issubdtype(dtype, np.floating):
        return dtype

    return np.dtype(np.float64)


def write_manifest_row(writer, report_file, row):
    writer.writerow(row)
    report_file.flush()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else dataset_root
    )
    collection_path = Path(args.bcc_collection).resolve()
    suffix = args.suffix

    if not suffix:
        raise ValueError("--suffix cannot be empty.")

    if args.max_molecules is not None and args.max_molecules <= 0:
        raise ValueError("--max-molecules must be positive.")

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not collection_path.is_file():
        raise FileNotFoundError(f"BCC collection not found: {collection_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else output_root.parent / f"{suffix.lstrip('-')}-report"
    )

    if report_dir.exists():
        raise FileExistsError(
            f"Report directory already exists: {report_dir}. "
            "Choose a new --report-dir or remove it after inspection."
        )

    dataset_jobs = []

    for dataset in args.datasets:
        source_dir = dataset_root / dataset
        target_dir = output_root / f"{dataset}{suffix}"
        staging_dir = output_root / f".{dataset}{suffix}.incomplete"

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Input dataset not found: {source_dir}")

        if target_dir.exists():
            raise FileExistsError(f"Output dataset already exists: {target_dir}")

        if staging_dir.exists():
            raise FileExistsError(
                f"Incomplete output already exists: {staging_dir}. "
                "Inspect it before removing or renaming it."
            )

        npz_paths = sorted(source_dir.glob("*.npz"), key=natural_sort_key)

        if args.max_molecules is not None:
            npz_paths = npz_paths[: args.max_molecules]

        if not npz_paths:
            raise RuntimeError(f"No NPZ files found in {source_dir}")

        dataset_jobs.append(
            (dataset, source_dir, target_dir, staging_dir, npz_paths)
        )

    bcc_collection = BCCCollection.parse_raw(collection_path.read_text())
    parameter_values = np.array(
        [parameter.value for parameter in bcc_collection.parameters],
        dtype=float,
    )
    parameter_is_trained = trained_flags(bcc_collection)

    report_dir.mkdir(parents=True)
    manifest_path = report_dir / "application_manifest.csv"
    summary_path = report_dir / "application_summary.json"
    manifest_fields = [
        "dataset",
        "file_name",
        "mol_id",
        "smiles",
        "mapped_smiles",
        "n_atoms",
        "net_formal_charge_e",
        "bcc_covered",
        "all_active_parameters_trained",
        "untrained_parameter_indices",
        "status",
        "exclusion_reasons",
        "maximum_total_charge_error_e",
        "output_path",
        "coverage_error",
    ]
    total_counts = Counter()
    dataset_summaries = {}

    with manifest_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=manifest_fields)
        writer.writeheader()

        for dataset, source_dir, target_dir, staging_dir, npz_paths in dataset_jobs:
            staging_dir.mkdir()
            counts = Counter()

            for position, npz_path in enumerate(npz_paths, start=1):
                payload = load_npz(npz_path)
                molecule, mapped_smiles, q_seed = build_molecule(
                    payload,
                    npz_path,
                )
                net_formal_charge = float(q_seed.sum())
                is_neutral = np.isclose(net_formal_charge, 0.0, atol=1.0e-8)
                assignment_matrix = None
                coverage_error = ""
                untrained_indices = np.array([], dtype=int)

                try:
                    assignment_matrix = BCCGenerator.build_assignment_matrix(
                        molecule,
                        bcc_collection,
                    )
                    bcc_covered = True
                except ChargeAssignmentError as error:
                    bcc_covered = False
                    coverage_error = str(error)

                if bcc_covered:
                    active_indices = np.flatnonzero(
                        np.any(np.abs(assignment_matrix) > 0.0, axis=0)
                    )
                    untrained_indices = active_indices[
                        ~parameter_is_trained[active_indices]
                    ]
                    all_active_parameters_trained = len(untrained_indices) == 0
                else:
                    all_active_parameters_trained = False

                exclusion_reasons = []

                if args.neutral_only and not is_neutral:
                    exclusion_reasons.append("charged")

                if not bcc_covered:
                    exclusion_reasons.append("uncovered")
                elif not all_active_parameters_trained:
                    exclusion_reasons.append("active_untrained_parameter")

                counts["input_molecules"] += 1
                counts["charged_molecules"] += int(not is_neutral)
                counts["uncovered_molecules"] += int(not bcc_covered)
                counts["active_untrained_molecules"] += int(
                    bcc_covered and not all_active_parameters_trained
                )

                maximum_total_charge_error = ""
                output_path = ""

                if exclusion_reasons:
                    status = "excluded"
                    counts["excluded_molecules"] += 1
                else:
                    q_predicted = q_seed + assignment_matrix @ parameter_values
                    maximum_total_charge_error = abs(
                        float(q_predicted.sum() - net_formal_charge)
                    )

                    if not np.all(np.isfinite(q_predicted)):
                        raise ValueError(f"Non-finite predicted charges in {npz_path}")

                    if maximum_total_charge_error > 1.0e-8:
                        raise ValueError(
                            "Total charge was not conserved in "
                            f"{npz_path}: error={maximum_total_charge_error} e"
                        )

                    original_charges = np.asarray(payload["partial_charges"])

                    if original_charges.shape != q_predicted.shape:
                        raise ValueError(
                            f"partial_charges shape mismatch in {npz_path}: "
                            f"stored={original_charges.shape}, "
                            f"predicted={q_predicted.shape}"
                        )

                    payload["partial_charges"] = q_predicted.astype(
                        charge_output_dtype(original_charges),
                        copy=False,
                    )
                    output_npz = staging_dir / npz_path.name
                    np.savez_compressed(output_npz, **payload)
                    output_path = str(target_dir / npz_path.name)
                    status = "written"
                    counts["written_molecules"] += 1

                row = {
                    "dataset": dataset,
                    "file_name": npz_path.name,
                    "mol_id": molecule_id(payload, npz_path.stem),
                    "smiles": smiles(payload),
                    "mapped_smiles": mapped_smiles,
                    "n_atoms": molecule.n_atoms,
                    "net_formal_charge_e": net_formal_charge,
                    "bcc_covered": bcc_covered,
                    "all_active_parameters_trained": (
                        all_active_parameters_trained if bcc_covered else ""
                    ),
                    "untrained_parameter_indices": ";".join(
                        str(index) for index in untrained_indices
                    ),
                    "status": status,
                    "exclusion_reasons": ";".join(exclusion_reasons),
                    "maximum_total_charge_error_e": maximum_total_charge_error,
                    "output_path": output_path,
                    "coverage_error": coverage_error,
                }
                write_manifest_row(writer, report_file, row)

                if position % args.progress_every == 0 or position == len(npz_paths):
                    print(
                        f"{dataset}: {position}/{len(npz_paths)} input, "
                        f"{counts['written_molecules']} written, "
                        f"{counts['excluded_molecules']} excluded",
                        flush=True,
                    )

            staging_dir.rename(target_dir)
            total_counts.update(counts)
            dataset_summaries[dataset] = {
                **dict(counts),
                "source_directory": str(source_dir),
                "output_directory": str(target_dir),
            }

    summary = {
        "method": "formal-charge BCC condensation",
        "charge_equation": "q = q_formal + assignment_matrix @ fitted_delta",
        "bcc_collection": str(collection_path),
        "bcc_collection_sha256": sha256(collection_path),
        "bcc_parameters": len(parameter_values),
        "trained_bcc_parameters": int(parameter_is_trained.sum()),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "output_suffix": suffix,
        "neutral_only": args.neutral_only,
        "max_molecules_per_dataset": args.max_molecules,
        "npz_policy": (
            "all arrays preserved; only partial_charges replaced; legacy "
            "charge_model feature preserved for Grappa compatibility"
        ),
        "counting_note": (
            "charged, uncovered, and active-untrained counts are independent "
            "and may overlap; excluded_molecules is the unique excluded total"
        ),
        "totals": dict(total_counts),
        "datasets": dataset_summaries,
        "manifest": str(manifest_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Input molecules: {total_counts['input_molecules']}")
    print(f"Written molecules: {total_counts['written_molecules']}")
    print(f"Excluded molecules: {total_counts['excluded_molecules']}")
    print(f"Charged molecules: {total_counts['charged_molecules']}")
    print(f"Uncovered molecules: {total_counts['uncovered_molecules']}")
    print(
        "Molecules using untrained parameters: "
        f"{total_counts['active_untrained_molecules']}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
