#!/usr/bin/env python3
"""
Compare trained BCC parameters with original AM1-BCC values and visualize the results.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openff.toolkit.topology import Molecule
from tqdm import tqdm


def load_trained_parameters(trained_file: Path) -> dict[str, float]:
    """Load trained BCC parameters from TSV file."""
    df = pd.read_csv(trained_file, sep="\t", header=None, names=["smarts", "value"])
    return dict(zip(df["smarts"], df["value"]))


def load_original_parameters(original_file: Path) -> dict[str, float]:
    """Load original AM1-BCC parameters from JSON file."""
    with open(original_file) as f:
        data = json.load(f)
    return {entry["smirks"]: entry["value"] for entry in data}


def calculate_match_counts(smiles_file: Path, smarts_patterns: list[str]) -> dict[str, int]:
    """Calculate SMARTS pattern match counts by loading molecules from SMILES file."""
    match_counts = {pattern: 0 for pattern in smarts_patterns}

    print(f"Loading molecules from {smiles_file}...")
    with open(smiles_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smiles_list)} SMILES strings")
    print(f"Calculating match counts for {len(smarts_patterns)} SMARTS patterns...")

    for smiles in tqdm(smiles_list, desc="Processing molecules"):
        try:
            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        except Exception as e:
            print(f"Warning: Failed to parse SMILES '{smiles}': {e}")
            continue

        for pattern in smarts_patterns:
            try:
                matches = molecule.chemical_environment_matches(pattern)
                match_counts[pattern] += len(matches)
            except Exception as e:
                print(f"Warning: Failed to match pattern '{pattern}': {e}")
                continue

    return match_counts


def main():
    parser = argparse.ArgumentParser(description="Compare trained BCC parameters with original values.")
    parser.add_argument(
        "trained_file",
        type=Path,
        help="TSV file with trained BCC parameters (output from training).",
    )
    parser.add_argument(
        "original_file",
        type=Path,
        help="JSON file with original AM1-BCC parameters.",
    )
    parser.add_argument(
        "smiles_file",
        type=Path,
        help="SMILES file for calculating match counts.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("bcc-comparison"),
        help="Prefix for output files (default: bcc-comparison).",
    )
    args = parser.parse_args()

    print("Loading parameters...")
    trained = load_trained_parameters(args.trained_file)
    original = load_original_parameters(args.original_file)

    print("Calculating SMARTS match counts...")
    coverage = calculate_match_counts(args.smiles_file, list(original.keys()))

    # Compile comparison data
    print("Compiling comparison data...")
    comparison_data = []

    for smarts in original.keys():
        orig_val = original[smarts]
        train_val = trained.get(smarts, np.nan)
        match_count = coverage.get(smarts, 0)

        comparison_data.append(
            {
                "smarts_pattern": smarts,
                "original_value": orig_val,
                "trained_value": train_val,
                "match_count": match_count,
            }
        )

    df_comparison = pd.DataFrame(comparison_data)

    # Save comparison TSV
    output_tsv = args.output_prefix.with_suffix(".tsv")
    df_comparison.to_csv(output_tsv, sep="\t", index=False)
    print(f"Comparison data saved to {output_tsv}")

    # Create correlation plot
    print("Creating correlation plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter out NaN values for plotting
    mask = ~(df_comparison["original_value"].isna() | df_comparison["trained_value"].isna())
    orig_vals = df_comparison.loc[mask, "original_value"].values
    train_vals = df_comparison.loc[mask, "trained_value"].values

    # Scatter plot
    ax.scatter(orig_vals, train_vals, alpha=0.6, s=30)

    # Diagonal line (y=x)
    min_val = min(orig_vals.min(), train_vals.min())
    max_val = max(orig_vals.max(), train_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x (perfect match)", linewidth=2)

    # Labels and title
    ax.set_xlabel("Original AM1-BCC Value", fontsize=12)
    ax.set_ylabel("Trained BCC Value", fontsize=12)
    ax.set_title("BCC Parameter Comparison: Original vs Trained", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add correlation coefficient
    correlation = np.corrcoef(orig_vals, train_vals)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {correlation:.4f}\nn = {len(orig_vals)}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_plot = args.output_prefix.with_suffix(".png")
    plt.savefig(output_plot, dpi=150)
    print(f"Correlation plot saved to {output_plot}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total SMARTS patterns: {len(df_comparison)}")
    print(f"Patterns with trained values: {mask.sum()}")
    print(f"Patterns without trained values: {(~mask).sum()}")
    print(f"\nPearson correlation: {correlation:.4f}")
    print(f"Mean original value: {orig_vals.mean():.6f}")
    print(f"Mean trained value: {train_vals.mean():.6f}")
    print(f"Std original value: {orig_vals.std():.6f}")
    print(f"Std trained value: {train_vals.std():.6f}")
    print(f"Max change: {np.abs(orig_vals - train_vals).max():.6f}")
    print(f"Mean absolute change: {np.abs(orig_vals - train_vals).mean():.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
