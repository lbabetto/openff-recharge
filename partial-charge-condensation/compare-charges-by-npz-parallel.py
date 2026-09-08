#!/usr/bin/env python3

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "spice-pubchem",
    "spice-des-monomers",
    "spice-dipeptide",
    "rna-diverse",
    "rna-trinucleotide",
    "rna-nucleoside",
    "gen2-torsion",
    "protein-torsion",
    "gen2",
    "pepconf-dlc",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare partial charges between two Grappa .npz dataset trees. "
            "Methods are identified by dataset directory suffixes."
        )
    )

    p.add_argument(
        "--method1",
        required=True,
        help=(
            "First method suffix, e.g. nagl, espaloma, "
            "espaloma-charges2, am1"
        ),
    )

    p.add_argument(
        "--root1",
        required=True,
        help="Root containing first method dataset folders",
    )

    p.add_argument(
        "--method2",
        required=True,
        help=(
            "Second method suffix, e.g. nagl, gasteiger, am1"
        ),
    )

    p.add_argument(
        "--root2",
        required=True,
        help="Root containing second method dataset folders",
    )

    p.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        default=None,
        help="Dataset(s) to compare. Default: infer common datasets.",
    )

    p.add_argument(
        "--charge-key",
        default="partial_charges",
        help="Charge array key inside .npz. Default: partial_charges",
    )

    p.add_argument(
        "--suffix1",
        default=None,
        help=(
            "Override directory suffix for method1. "
            "Default: method1. Use empty string '' for no suffix."
        ),
    )

    p.add_argument(
        "--suffix2",
        default=None,
        help=(
            "Override directory suffix for method2. "
            "Default: method2. Use empty string '' for no suffix."
        ),
    )

    p.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV summary.",
    )

    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on missing files or mismatched atom counts "
            "instead of skipping."
        ),
    )

    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel workers for .npz loading/comparison. "
            "Default: min(32, os.cpu_count()). "
            "Use 1 for serial execution."
        ),
    )

    p.add_argument(
        "--outdir",
        default="compare_npz_charge_methods",
        help=(
            "Output directory. If relative, it is created in the "
            "current working directory. "
            "Default: compare_npz_charge_methods"
        ),
    )

    p.add_argument(
        "--scatterplot",
        action="store_true",
        help=(
            "Create one scatterplot per dataset plus one "
            "global scatterplot."
        ),
    )

    p.add_argument(
        "--nohexbin",
        "--no-hexbin",
        dest="nohexbin",
        action="store_true",
        help=(
            "Disable hexbin density plots and use plain "
            "scatter points instead."
        ),
    )

    p.add_argument(
        "--qq",
        action="store_true",
        help=(
            "Create one normal QQ plot of charge differences "
            "per dataset plus one global QQ plot."
        ),
    )

    return p.parse_args()


class Reporter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w", encoding="utf-8")

    def log(self, msg=""):
        print(msg)
        self.handle.write(str(msg) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def resolve_outdir(outdir_arg):
    outdir = Path(outdir_arg).expanduser()

    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir

    return outdir.resolve()


def safe_name(s):
    s = str(s)
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in s
    )


def method_to_suffix(method, suffix_override=None):
    """
    Convert method name to directory suffix.

    method='nagl'
        -> suffix='nagl'
        -> dataset-nagl

    method='espaloma-charges2'
        -> suffix='espaloma-charges2'
        -> dataset-espaloma-charges2

    method='am1'
        -> suffix=''
        -> dataset
    """
    if suffix_override is not None:
        return suffix_override

    if method.lower() in {
        "am1",
        "am1-bcc",
        "am1-bcc-elf10",
    }:
        return ""

    return method


def dataset_dir(root, dataset, suffix):
    if suffix == "":
        return root / dataset

    return root / f"{dataset}-{suffix}"


def infer_datasets_from_root(root, suffix):
    root = Path(root)

    if not root.is_dir():
        raise SystemExit(
            f"Missing root directory: {root}"
        )

    datasets = []

    if suffix == "":
        for dataset in DEFAULT_DATASETS:
            if (root / dataset).is_dir():
                datasets.append(dataset)

        return sorted(datasets)

    marker = f"-{suffix}"

    for path in root.iterdir():
        if not path.is_dir():
            continue

        name = path.name

        if name.endswith(marker):
            dataset = name[:-len(marker)]
            datasets.append(dataset)

    return sorted(set(datasets))


def get_npz_files(dataset_path):
    if not dataset_path.is_dir():
        return {}

    return {
        path.relative_to(dataset_path): path
        for path in sorted(
            dataset_path.rglob("*.npz")
        )
    }


def load_charges(npz_path, charge_key):
    with np.load(npz_path) as data:
        if charge_key not in data:
            available = ", ".join(data.files)

            raise KeyError(
                f"{npz_path} does not contain key "
                f"'{charge_key}'. "
                f"Available keys: {available}"
            )

        q = np.asarray(
            data[charge_key],
            dtype=float,
        ).reshape(-1)

    return q


def compare_one_npz_pair(
    dataset,
    relpath,
    p1,
    p2,
    charge_key,
):
    try:
        q1 = load_charges(
            p1,
            charge_key,
        )

        q2 = load_charges(
            p2,
            charge_key,
        )

        if len(q1) != len(q2):
            return {
                "status": "SKIPPED",
                "dataset": dataset,
                "relpath": str(relpath),
                "message": (
                    f"{dataset}/{relpath}: "
                    f"different number of charges: "
                    f"{len(q1)} vs {len(q2)}"
                ),
                "q1": None,
                "q2": None,
            }

        return {
            "status": "OK",
            "dataset": dataset,
            "relpath": str(relpath),
            "message": "",
            "q1": q1,
            "q2": q2,
        }

    except Exception as exc:
        return {
            "status": "ERROR",
            "dataset": dataset,
            "relpath": str(relpath),
            "message": (
                f"{dataset}/{relpath}: "
                f"{type(exc).__name__}: {exc}"
            ),
            "q1": None,
            "q2": None,
        }


def get_workers(requested_workers):
    if requested_workers == 1:
        return 1

    if requested_workers > 1:
        return requested_workers

    n = os.cpu_count() or 1

    return min(32, n)


def pearson_r(x, y):
    if len(x) == 0:
        return np.nan

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    x0 = x - x.mean()
    y0 = y - y.mean()

    denom = np.sqrt(
        np.sum(x0 * x0)
        * np.sum(y0 * y0)
    )

    if denom == 0.0:
        return np.nan

    return float(
        np.sum(x0 * y0) / denom
    )


def mae(x, y):
    if len(x) == 0:
        return np.nan

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(x - y)
        )
    )


def rmse(x, y):
    if len(x) == 0:
        return np.nan

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (x - y) ** 2
            )
        )
    )


def fmt_float(x):
    if pd.isna(x):
        return "nan"

    return f"{x:.6g}"


def plot_scatter(
    x,
    y,
    row,
    out_png,
    method1,
    method2,
    title,
    nohexbin,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[finite]
    y = y[finite]

    plt.figure(
        figsize=(7.5, 6.5)
    )

    if x.size == 0:
        plt.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
        )

        plt.title(title)

        plt.savefig(
            out_png,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()
        return

    lo = float(
        min(
            np.min(x),
            np.min(y),
        )
    )

    hi = float(
        max(
            np.max(x),
            np.max(y),
        )
    )

    pad = (
        0.05 * (hi - lo)
        if hi > lo
        else 1.0
    )

    lo -= pad
    hi += pad

    if nohexbin:
        plt.scatter(
            x,
            y,
            s=1.0,
            alpha=0.8,
            linewidths=0,
        )

    else:
        hb = plt.hexbin(
            x,
            y,
            gridsize=200,
            mincnt=1,
        )

        cb = plt.colorbar(hb)
        cb.set_label("counts")

    plt.plot(
        [lo, hi],
        [lo, hi],
        linewidth=1.0,
    )

    txt = (
        #f"n_atoms = {int(row['n_atoms'])}\n"
        f"R       = {fmt_float(row['pearson_r'])}\n"
        f"MAE     = {fmt_float(row['mae'])}\n"
        f"RMSE    = {fmt_float(row['rmse'])}"
    )

    plt.gca().text(
        0.02,
        0.98,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.25",
            "alpha": 1.0,
        },
    )

    plt.xlim(
        lo,
        hi,
    )

    plt.ylim(
        lo,
        hi,
    )

    plt.xlabel(
        f"{method1} charges / e"
    )

    plt.ylabel(
        f"{method2} charges / e"
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        out_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_qq(
    q1,
    q2,
    out_png,
    method1,
    method2,
    title,
):
    """
    Create a normal QQ plot of:

        delta_q = q2 - q1

    where q1 contains method1 charges
    and q2 contains method2 charges.
    """
    try:
        from scipy import stats as scipy_stats

    except ImportError as exc:
        raise RuntimeError(
            "The --qq option requires scipy. "
            "Install scipy in the active Python environment."
        ) from exc

    out_png = Path(out_png)

    out_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    q1 = np.asarray(
        q1,
        dtype=float,
    )

    q2 = np.asarray(
        q2,
        dtype=float,
    )

    finite = (
        np.isfinite(q1)
        & np.isfinite(q2)
    )

    delta_q = (
        q2[finite]
        - q1[finite]
    )

    plt.figure(
        figsize=(6.5, 6.5)
    )

    if delta_q.size == 0:
        plt.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
        )

        plt.title(title)

        plt.savefig(
            out_png,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()
        return

    (
        theoretical_quantiles,
        ordered_delta_q,
    ), (
        slope,
        intercept,
        qq_r,
    ) = scipy_stats.probplot(
        delta_q,
        dist="norm",
    )

    theoretical_quantiles = np.asarray(
        theoretical_quantiles,
        dtype=float,
    )

    ordered_delta_q = np.asarray(
        ordered_delta_q,
        dtype=float,
    )

    plt.scatter(
        theoretical_quantiles,
        ordered_delta_q,
        s=4.0,
        alpha=0.8,
        linewidths=0,
    )

    fitted_values = (
        slope * theoretical_quantiles
        + intercept
    )

    plt.plot(
        theoretical_quantiles,
        fitted_values,
        linewidth=1.0,
    )

    txt = (
        #f"n_atoms = {delta_q.size}\n"
        f"mean Δq = {np.mean(delta_q):.6g}\n"
        f"std Δq  = {np.std(delta_q):.6g}\n"
        f"QQ R    = {qq_r:.6g}"
    )

    plt.gca().text(
        0.02,
        0.98,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.25",
            "alpha": 1.0,
        },
    )

    plt.xlabel(
        "Theoretical normal quantiles"
    )

    plt.ylabel(
        f"Ordered Δq = "
        f"{method2} − {method1} / e"
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        out_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main():
    args = parse_args()

    root1 = Path(
        args.root1
    ).resolve()

    root2 = Path(
        args.root2
    ).resolve()

    suffix1 = method_to_suffix(
        args.method1,
        args.suffix1,
    )

    suffix2 = method_to_suffix(
        args.method2,
        args.suffix2,
    )

    workers = get_workers(
        args.workers
    )

    outdir = resolve_outdir(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir = (
        outdir / "plots"
    )

    report_path = (
        outdir / "report.txt"
    )

    reporter = Reporter(
        report_path
    )

    log = reporter.log

    if args.datasets is None:
        datasets1 = set(
            infer_datasets_from_root(
                root1,
                suffix1,
            )
        )

        datasets2 = set(
            infer_datasets_from_root(
                root2,
                suffix2,
            )
        )

        datasets = sorted(
            datasets1 & datasets2
        )

    else:
        datasets = args.datasets

    if not datasets:
        reporter.close()

        raise SystemExit(
            "No datasets found to compare."
        )

    rows = []

    global_q1 = []
    global_q2 = []

    log("Comparison setup")
    log("=" * 80)
    log(f"method1     : {args.method1}")
    log(f"root1       : {root1}")
    log(f"suffix1     : {suffix1!r}")
    log(f"method2     : {args.method2}")
    log(f"root2       : {root2}")
    log(f"suffix2     : {suffix2!r}")
    log(f"charge_key  : {args.charge_key}")
    log(f"workers     : {workers}")
    log(f"outdir      : {outdir}")
    log(f"report      : {report_path}")
    log(f"scatterplot : {args.scatterplot}")
    log(f"qq          : {args.qq}")
    log(f"hexbin      : {not args.nohexbin}")
    log(f"datasets    : {' '.join(datasets)}")
    log("=" * 80)
    log()

    for dataset in datasets:
        dir1 = dataset_dir(
            root1,
            dataset,
            suffix1,
        )

        dir2 = dataset_dir(
            root2,
            dataset,
            suffix2,
        )

        if not dir1.is_dir():
            msg = (
                "Missing dataset directory "
                f"for method1: {dir1}"
            )

            if args.strict:
                reporter.close()

                raise SystemExit(msg)

            log(f"WARNING: {msg}")
            continue

        if not dir2.is_dir():
            msg = (
                "Missing dataset directory "
                f"for method2: {dir2}"
            )

            if args.strict:
                reporter.close()

                raise SystemExit(msg)

            log(f"WARNING: {msg}")
            continue

        files1 = get_npz_files(
            dir1
        )

        files2 = get_npz_files(
            dir2
        )

        common = sorted(
            set(files1)
            & set(files2)
        )

        only1 = len(
            set(files1)
            - set(files2)
        )

        only2 = len(
            set(files2)
            - set(files1)
        )

        q1_dataset = []
        q2_dataset = []

        skipped = 0
        errors = 0

        if (
            workers == 1
            or len(common) <= 1
        ):
            for relpath in common:
                p1 = files1[relpath]
                p2 = files2[relpath]

                result = compare_one_npz_pair(
                    dataset=dataset,
                    relpath=relpath,
                    p1=p1,
                    p2=p2,
                    charge_key=args.charge_key,
                )

                if result["status"] == "OK":
                    q1_dataset.extend(
                        result["q1"].tolist()
                    )

                    q2_dataset.extend(
                        result["q2"].tolist()
                    )

                elif result["status"] == "SKIPPED":
                    if args.strict:
                        reporter.close()

                        raise SystemExit(
                            result["message"]
                        )

                    log(
                        "WARNING: skipped "
                        f"{result['message']}"
                    )

                    skipped += 1

                else:
                    if args.strict:
                        reporter.close()

                        raise RuntimeError(
                            result["message"]
                        )

                    log(
                        "WARNING: skipped "
                        f"{result['message']}"
                    )

                    skipped += 1
                    errors += 1

        else:
            with ProcessPoolExecutor(
                max_workers=workers
            ) as executor:
                futures = []

                for relpath in common:
                    p1 = files1[relpath]
                    p2 = files2[relpath]

                    futures.append(
                        executor.submit(
                            compare_one_npz_pair,
                            dataset,
                            relpath,
                            p1,
                            p2,
                            args.charge_key,
                        )
                    )

                for future in as_completed(
                    futures
                ):
                    result = future.result()

                    if result["status"] == "OK":
                        q1_dataset.extend(
                            result["q1"].tolist()
                        )

                        q2_dataset.extend(
                            result["q2"].tolist()
                        )

                    elif result["status"] == "SKIPPED":
                        if args.strict:
                            reporter.close()

                            raise SystemExit(
                                result["message"]
                            )

                        log(
                            "WARNING: skipped "
                            f"{result['message']}"
                        )

                        skipped += 1

                    else:
                        if args.strict:
                            reporter.close()

                            raise RuntimeError(
                                result["message"]
                            )

                        log(
                            "WARNING: skipped "
                            f"{result['message']}"
                        )

                        skipped += 1
                        errors += 1

        row = {
            "dataset": dataset,
            "method1": args.method1,
            "method2": args.method2,
            "dir1": str(dir1),
            "dir2": str(dir2),
            "n_common_npz": len(common),
            "n_atoms": len(q1_dataset),
            "pearson_r": pearson_r(
                q1_dataset,
                q2_dataset,
            ),
            "mae": mae(
                q1_dataset,
                q2_dataset,
            ),
            "rmse": rmse(
                q1_dataset,
                q2_dataset,
            ),
            "only_method1_npz": only1,
            "only_method2_npz": only2,
            "skipped": skipped,
            "errors": errors,
        }

        rows.append(row)

        if args.scatterplot:
            plot_path = (
                plots_dir
                / (
                    f"{safe_name(dataset)}__"
                    f"{safe_name(args.method1)}_vs_"
                    f"{safe_name(args.method2)}.png"
                )
            )

            plot_scatter(
                x=q1_dataset,
                y=q2_dataset,
                row=row,
                out_png=plot_path,
                method1=args.method1,
                method2=args.method2,
                title=(
                    f"{dataset}: "
                    f"{args.method1} vs "
                    f"{args.method2}"
                ),
                nohexbin=args.nohexbin,
            )

        if args.qq:
            qq_path = (
                plots_dir
                / (
                    f"{safe_name(dataset)}__"
                    f"{safe_name(args.method2)}_minus_"
                    f"{safe_name(args.method1)}__qq.png"
                )
            )

            plot_qq(
                q1=q1_dataset,
                q2=q2_dataset,
                out_png=qq_path,
                method1=args.method1,
                method2=args.method2,
                title=(
                    f"{dataset}: normal QQ plot of "
                    f"Δq = {args.method2} − "
                    f"{args.method1}"
                ),
            )

        global_q1.extend(
            q1_dataset
        )

        global_q2.extend(
            q2_dataset
        )

    global_row = {
        "dataset": "GLOBAL",
        "method1": args.method1,
        "method2": args.method2,
        "dir1": str(root1),
        "dir2": str(root2),
        "n_common_npz": sum(
            row["n_common_npz"]
            for row in rows
        ),
        "n_atoms": len(global_q1),
        "pearson_r": pearson_r(
            global_q1,
            global_q2,
        ),
        "mae": mae(
            global_q1,
            global_q2,
        ),
        "rmse": rmse(
            global_q1,
            global_q2,
        ),
        "only_method1_npz": sum(
            row["only_method1_npz"]
            for row in rows
        ),
        "only_method2_npz": sum(
            row["only_method2_npz"]
            for row in rows
        ),
        "skipped": sum(
            row["skipped"]
            for row in rows
        ),
        "errors": sum(
            row["errors"]
            for row in rows
        ),
    }

    rows.append(
        global_row
    )

    if args.scatterplot:
        plot_path = (
            plots_dir
            / (
                f"GLOBAL__"
                f"{safe_name(args.method1)}_vs_"
                f"{safe_name(args.method2)}.png"
            )
        )

        plot_scatter(
            x=global_q1,
            y=global_q2,
            row=global_row,
            out_png=plot_path,
            method1=args.method1,
            method2=args.method2,
            title=(
                #f"GLOBAL: "
                f"{args.method1} vs "
                f"{args.method2}"
            ),
            nohexbin=args.nohexbin,
        )

    if args.qq:
        qq_path = (
            plots_dir
            / (
                f"GLOBAL__"
                f"{safe_name(args.method2)}_minus_"
                f"{safe_name(args.method1)}__qq.png"
            )
        )

        plot_qq(
            q1=global_q1,
            q2=global_q2,
            out_png=qq_path,
            method1=args.method1,
            method2=args.method2,
            title=(
                #"GLOBAL: normal QQ plot of "
                "Normal QQ plot of "
                f"Δq = {args.method2} − "
                f"{args.method1}"
            ),
        )

    df = pd.DataFrame(
        rows
    )

    header = (
        f"{'dataset':<22} "
        f"{'n_npz':>8} "
        f"{'n_atoms':>10} "
        f"{'R':>10} "
        f"{'MAE':>12} "
        f"{'RMSE':>12} "
        f"{'only_m1':>8} "
        f"{'only_m2':>8} "
        f"{'skipped':>8}"
    )

    log(header)
    log("-" * len(header))

    for _, row in df.iterrows():
        log(
            f"{row['dataset']:<22} "
            f"{int(row['n_common_npz']):>8} "
            f"{int(row['n_atoms']):>10} "
            f"{fmt_float(row['pearson_r']):>10} "
            f"{fmt_float(row['mae']):>12} "
            f"{fmt_float(row['rmse']):>12} "
            f"{int(row['only_method1_npz']):>8} "
            f"{int(row['only_method2_npz']):>8} "
            f"{int(row['skipped']):>8}"
        )

    if args.output_csv:
        out = Path(
            args.output_csv
        )

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            out,
            index=False,
        )

        log()
        log(f"CSV written: {out}")

    log()
    log(f"Report written: {report_path}")

    if args.scatterplot:
        log(
            f"Scatterplots written in: "
            f"{plots_dir}"
        )

    if args.qq:
        log(
            f"QQ plots written in: "
            f"{plots_dir}"
        )

    reporter.close()


if __name__ == "__main__":
    main()
