import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from openff.nagl.features.atoms import AtomAverageFormalCharge
from openff.recharge.charges.bcc import BCCCollection
from openff.toolkit import Molecule
from openff.units import unit

from fast_bcc_apply import (
    build_assignment_matrix_fast,
    compile_bcc_queries,
)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("Il manifest non contiene molecole.")

    return rows


def formal_charge_vector(molecule):
    return np.array(
        [
            atom.formal_charge.m_as(unit.elementary_charge)
            for atom in molecule.atoms
        ],
        dtype=float,
    )


def build_resonance_averaged_seed(molecule):
    charge_tensor = AtomAverageFormalCharge().encode(molecule)

    q_resonance = (
        charge_tensor.detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(float)
    )
    q_formal = formal_charge_vector(molecule)

    if q_resonance.shape != q_formal.shape:
        raise ValueError(
            "Il seed resonance-aware ha una dimensione incompatibile."
        )

    if not np.all(np.isfinite(q_resonance)):
        raise ValueError(
            "Il seed resonance-aware contiene valori non finiti."
        )

    if not np.isclose(
        q_resonance.sum(),
        q_formal.sum(),
        atol=1.0e-8,
    ):
        raise ValueError(
            "Il seed resonance-aware non conserva la carica totale."
        )

    return q_resonance


def scalar_string(value):
    array = np.asarray(value)

    if array.size != 1:
        raise ValueError(
            f"Expected one string value, found shape {array.shape}."
        )

    item = array.reshape(-1)[0]

    if isinstance(item, bytes):
        return item.decode("utf-8")

    return str(item)


def load_molecule_data(path):
    with np.load(path, allow_pickle=False) as data:
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
        raise ValueError(f"Ordine atomico incoerente in {path}")

    try:
        q_seed = build_resonance_averaged_seed(molecule)
    except Exception as error:
        raise RuntimeError(
            f"Impossibile costruire il seed per {path}"
        ) from error

    if q_reference.shape != q_seed.shape:
        raise ValueError(
            f"Cariche incompatibili con la molecola in {path}"
        )

    return molecule, mapped_smiles, q_reference, q_seed


def local_environment(rd_molecule, atom_index):
    atom = rd_molecule.GetAtomWithIdx(atom_index)
    neighbors = []

    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        neighbors.append(
            f"{other.GetSymbol()}{other.GetIdx()}:"
            f"{str(bond.GetBondType())}"
        )

    return ";".join(sorted(neighbors))


def write_csv(path, fieldnames, rows):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bcc-collection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-charge-rmse", type=float)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    collection_path = Path(args.bcc_collection).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Non sovrascrivo una directory esistente: {output_dir}"
        )

    if args.top_n < 1:
        raise ValueError("top-n deve essere maggiore di zero.")

    rows = load_manifest(manifest_path)

    bcc_collection = BCCCollection.model_validate_json(
        collection_path.read_text()
    )
    compiled_bcc_queries = compile_bcc_queries(bcc_collection)
    parameter_values = np.array(
        [
            parameter.value
            for parameter in bcc_collection.parameters
        ],
        dtype=float,
    )

    atom_rows = []
    maximum_total_charge_error = 0.0

    for position, row in enumerate(rows, start=1):
        npz_path = Path(row["path"])
        dataset = row.get("dataset") or npz_path.parent.name

        (
            molecule,
            mapped_smiles,
            q_reference,
            q_seed,
        ) = load_molecule_data(npz_path)

        assignment_matrix = build_assignment_matrix_fast(
            molecule,
            bcc_collection,
            compiled_bcc_queries,
        )

        q_bcc = assignment_matrix @ parameter_values
        q_predicted = q_seed + q_bcc

        if q_predicted.shape != q_reference.shape:
            raise ValueError(
                f"Predizione incompatibile con {npz_path}"
            )

        total_charge_error = abs(
            float(q_predicted.sum() - q_seed.sum())
        )
        maximum_total_charge_error = max(
            maximum_total_charge_error,
            total_charge_error,
        )

        rd_molecule = molecule.to_rdkit()
        q_formal = formal_charge_vector(molecule)

        for atom_index in range(molecule.n_atoms):
            rd_atom = rd_molecule.GetAtomWithIdx(atom_index)
            active_indices = np.flatnonzero(
                np.abs(assignment_matrix[atom_index]) > 0.0
            )

            bcc_terms = []

            for parameter_index in active_indices:
                coefficient = float(
                    assignment_matrix[
                        atom_index,
                        parameter_index,
                    ]
                )
                value = float(
                    parameter_values[parameter_index]
                )
                contribution = coefficient * value
                parameter = bcc_collection.parameters[
                    parameter_index
                ]

                bcc_terms.append(
                    {
                        "parameter_index": int(parameter_index),
                        "smirks": parameter.smirks,
                        "coefficient": coefficient,
                        "parameter_value_e": value,
                        "contribution_e": contribution,
                    }
                )

            reconstructed_bcc = sum(
                term["contribution_e"]
                for term in bcc_terms
            )

            if not np.isclose(
                reconstructed_bcc,
                q_bcc[atom_index],
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "Decomposizione BCC incoerente per "
                    f"{npz_path}, atomo {atom_index}"
                )

            residual = float(
                q_predicted[atom_index]
                - q_reference[atom_index]
            )

            atom_rows.append(
                {
                    "dataset": dataset,
                    "file_name": npz_path.name,
                    "path": str(npz_path),
                    "mol_id": row.get("mol_id", ""),
                    "smiles": row.get("smiles", ""),
                    "mapped_smiles": mapped_smiles,
                    "atom_index": atom_index,
                    "element": rd_atom.GetSymbol(),
                    "atomic_number": rd_atom.GetAtomicNum(),
                    "formal_charge_e": float(
                        q_formal[atom_index]
                    ),
                    "local_environment": local_environment(
                        rd_molecule,
                        atom_index,
                    ),
                    "reference_charge_e": float(
                        q_reference[atom_index]
                    ),
                    "resonance_seed_charge_e": float(
                        q_seed[atom_index]
                    ),
                    "resonance_seed_shift_e": float(
                        q_seed[atom_index]
                        - q_formal[atom_index]
                    ),
                    "bcc_correction_e": float(
                        q_bcc[atom_index]
                    ),
                    "predicted_charge_e": float(
                        q_predicted[atom_index]
                    ),
                    "residual_e": residual,
                    "absolute_error_e": abs(residual),
                    "bcc_term_count": len(bcc_terms),
                    "bcc_terms_json": json.dumps(
                        bcc_terms,
                        separators=(",", ":"),
                    ),
                    "molecule_total_charge_error_e": (
                        total_charge_error
                    ),
                }
            )

        if (
            args.progress_every > 0
            and position % args.progress_every == 0
        ):
            print(
                f"Analizzate {position}/{len(rows)} molecole",
                flush=True,
            )

    reference = np.array(
        [
            row["reference_charge_e"]
            for row in atom_rows
        ],
        dtype=float,
    )
    predicted = np.array(
        [
            row["predicted_charge_e"]
            for row in atom_rows
        ],
        dtype=float,
    )
    residuals = predicted - reference

    charge_rmse = float(
        np.sqrt(np.mean(residuals**2))
    )
    charge_mae = float(np.mean(np.abs(residuals)))
    maximum_absolute_error = float(
        np.max(np.abs(residuals))
    )
    pearson_r = float(
        np.corrcoef(reference, predicted)[0, 1]
    )

    reference_ss = float(
        np.sum((reference - np.mean(reference)) ** 2)
    )
    residual_ss = float(np.sum(residuals**2))
    r_squared = float(1.0 - residual_ss / reference_ss)

    if (
        args.expected_charge_rmse is not None
        and not np.isclose(
            charge_rmse,
            args.expected_charge_rmse,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "Charge RMSE ricostruita diversa dal test: "
            f"attesa={args.expected_charge_rmse:.15f}, "
            f"ottenuta={charge_rmse:.15f}"
        )

    sorted_rows = sorted(
        atom_rows,
        key=lambda row: row["absolute_error_e"],
        reverse=True,
    )
    top_rows = sorted_rows[: args.top_n]

    output_dir.mkdir(parents=True, exist_ok=False)

    all_fields = list(atom_rows[0])
    predictions_path = (
        output_dir / "test_atomic_charge_predictions.csv"
    )
    outliers_path = (
        output_dir / "test_top10_charge_outliers.csv"
    )
    scatter_path = (
        output_dir / "test_charge_scatter.png"
    )
    decomposition_path = (
        output_dir / "test_top10_charge_decomposition.png"
    )
    summary_path = output_dir / "charge_analysis_summary.json"

    write_csv(
        predictions_path,
        all_fields,
        atom_rows,
    )

    ranked_top_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(top_rows, start=1)
    ]
    write_csv(
        outliers_path,
        ["rank", *all_fields],
        ranked_top_rows,
    )

    element_counts = Counter(
        row["element"] for row in atom_rows
    )
    element_rows = defaultdict(list)

    for row in atom_rows:
        element_rows[row["element"]].append(row)

    metrics_by_element = {}

    for element in sorted(element_rows):
        current = element_rows[element]
        current_residuals = np.array(
            [row["residual_e"] for row in current],
            dtype=float,
        )
        metrics_by_element[element] = {
            "count": len(current),
            "rmse_e": float(
                np.sqrt(np.mean(current_residuals**2))
            ),
            "mae_e": float(
                np.mean(np.abs(current_residuals))
            ),
            "maximum_absolute_error_e": float(
                np.max(np.abs(current_residuals))
            ),
        }

    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    color_map = plt.get_cmap("tab10")
    elements = [
        element
        for element, _ in element_counts.most_common()
    ]

    for element_index, element in enumerate(elements):
        mask = np.array(
            [
                row["element"] == element
                for row in atom_rows
            ],
            dtype=bool,
        )
        axis.scatter(
            reference[mask],
            predicted[mask],
            s=7,
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
            color=color_map(
                element_index % color_map.N
            ),
            label=f"{element} (n={np.count_nonzero(mask)})",
        )

    lower = float(min(reference.min(), predicted.min()))
    upper = float(max(reference.max(), predicted.max()))
    padding = 0.05 * max(upper - lower, 1.0)
    lower -= padding
    upper += padding

    axis.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linewidth=1.2,
        linestyle="--",
        #label="q predetta = q riferimento",
    )

    top_reference = [
        row["reference_charge_e"] for row in top_rows
    ]
    top_predicted = [
        row["predicted_charge_e"] for row in top_rows
    ]

    axis.scatter(
        top_reference,
        top_predicted,
        s=42,
        facecolors="none",
        edgecolors="red",
        linewidths=1.2,
        zorder=5,
        label=f"Top {len(top_rows)} outlier",
    )

    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Carica AM1-BCC-ELF10 di riferimento (e)")
    axis.set_ylabel("Carica predetta seed + BCC (e)")
    axis.set_title(
        "Cariche atomiche predette sul test set"
    )
    axis.grid(alpha=0.18)

    metrics_text = (
        f"N atomi = {len(atom_rows):,}\n"
        f"RMSE = {charge_rmse:.6f} e\n"
        f"MAE = {charge_mae:.6f} e\n"
        #f"R² = {r_squared:.5f}\n"
        f"Pearson r = {pearson_r:.5f}"
    )
    axis.text(
        0.03,
        0.97,
        metrics_text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "0.7",
        },
    )
    axis.legend(
        loc="lower right",
        fontsize=7,
        frameon=False,
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(scatter_path, dpi=300)
    plt.close(figure)

    y_positions = np.arange(len(top_rows))
    figure_height = max(5.5, 0.68 * len(top_rows) + 2.0)
    figure, axis = plt.subplots(
        figsize=(11.5, figure_height)
    )

    for position, row in enumerate(top_rows):
        axis.plot(
            [
                row["resonance_seed_charge_e"],
                row["predicted_charge_e"],
            ],
            [position, position],
            color="0.55",
            linewidth=2.0,
            zorder=1,
        )

    axis.scatter(
        [
            row["reference_charge_e"]
            for row in top_rows
        ],
        y_positions,
        marker="o",
        s=55,
        color="black",
        label="AM1-BCC-ELF10",
        zorder=4,
    )
    axis.scatter(
        [
            row["resonance_seed_charge_e"]
            for row in top_rows
        ],
        y_positions,
        marker="s",
        s=55,
        color="tab:orange",
        label="Seed di risonanza",
        zorder=3,
    )
    axis.scatter(
        [
            row["predicted_charge_e"]
            for row in top_rows
        ],
        y_positions,
        marker="D",
        s=48,
        color="tab:blue",
        label="Seed + BCC",
        zorder=5,
    )

    labels = [
        (
            f"{rank}. {row['element']}{row['atom_index']}  "
            f"{row['dataset']}/{row['file_name']}  "
            f"|errore|={row['absolute_error_e']:.3f} e"
        )
        for rank, row in enumerate(top_rows, start=1)
    ]

    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.axvline(
        0.0,
        color="0.65",
        linewidth=0.8,
        linestyle="--",
    )
    axis.set_xlabel("Carica atomica (e)")
    axis.set_title(
        "Decomposizione delle cariche per i principali outlier"
    )
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="x", alpha=0.2)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        loc="best",
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(decomposition_path, dpi=300)
    plt.close(figure)

    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "bcc_collection": str(collection_path),
        "bcc_collection_sha256": sha256(collection_path),
        "molecules": len(rows),
        "atoms": len(atom_rows),
        "bcc_parameters": len(parameter_values),
        "charge_rmse_e": charge_rmse,
        "charge_mae_e": charge_mae,
        "maximum_absolute_error_e": maximum_absolute_error,
        "r_squared": r_squared,
        "pearson_r": pearson_r,
        "maximum_total_charge_error_e": (
            maximum_total_charge_error
        ),
        "top_n": len(top_rows),
        "metrics_by_element": metrics_by_element,
        "atomic_predictions": str(predictions_path),
        "top_outliers": str(outliers_path),
        "scatter_plot": str(scatter_path),
        "decomposition_plot": str(decomposition_path),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Molecole:", len(rows))
    print("Atomi:", len(atom_rows))
    print("Charge RMSE:", charge_rmse, "e")
    print("Charge MAE:", charge_mae, "e")
    print("Errore massimo:", maximum_absolute_error, "e")
    print("R²:", r_squared)
    print("Pearson r:", pearson_r)
    print(
        "Errore massimo sulla carica totale:",
        maximum_total_charge_error,
        "e",
    )

    print("\nTop outlier:")

    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:2d}. "
            f"{row['dataset']}/{row['file_name']} "
            f"{row['element']}{row['atom_index']} "
            f"ref={row['reference_charge_e']:+.6f} "
            f"seed={row['resonance_seed_charge_e']:+.6f} "
            f"BCC={row['bcc_correction_e']:+.6f} "
            f"pred={row['predicted_charge_e']:+.6f} "
            f"errore={row['residual_e']:+.6f} "
            f"ambiente={row['local_environment']}"
        )

    print("\nOutput:", output_dir)


if __name__ == "__main__":
    main()
