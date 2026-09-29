from __future__ import annotations
import argparse
import glob
import logging
from pathlib import Path
import pandas as pd
from oracle_hybrid_builder import build_oracle_for_cell

DEFAULT_GLOB = "exports/full_grid/gamma_*"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB,
                        help=f"glob of cell directories (default: {DEFAULT_GLOB})")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild cells whose oracle_hybrid/ directory exists")
    args = parser.parse_args()

    # Silence library loggers: a NullHandler on the root logger stops Python's
    # last-resort handler from writing records to stderr.
    logging.getLogger().addHandler(logging.NullHandler())

    cell_dirs = sorted(Path(p) for p in glob.glob(args.cells)
                       if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(f"No cell directories match '{args.cells}'.")

    failures: list[dict] = []
    for cell in cell_dirs:
        out_dir = cell / "oracle_hybrid"
        if out_dir.exists() and not args.overwrite:
            continue                          # already built
        try:
            build_oracle_for_cell(cell)
        except Exception as exc:
            failures.append({"cell": cell.name, "error": repr(exc)})

    if failures:
        pd.DataFrame(failures).to_csv(
            cell_dirs[0].parent / "oracle_hybrid_failures.csv", index=False)
        raise SystemExit(1)


if __name__ == "__main__":
    main()