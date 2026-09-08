import argparse
import logging
from pathlib import Path

from openff.recharge.esp.storage import MoleculeESPStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge several MoleculeESPStore .sqlite files produced by separate "
        "chunks of a multi-node run into a single .sqlite file."
    )
    parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="The per-chunk .sqlite files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path of the merged output .sqlite file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.output.unlink(missing_ok=True)
    merged_store = MoleculeESPStore(str(args.output))

    n_records = 0

    for input_file in args.input_files:
        chunk_store = MoleculeESPStore(str(input_file))
        records = chunk_store.retrieve()

        merged_store.store(*records)
        n_records += len(records)

        logging.info(f"Merged {len(records)} records from {input_file}")

    logging.info(f"Merged {n_records} records from {len(args.input_files)} files into {args.output}")


if __name__ == "__main__":
    main()
