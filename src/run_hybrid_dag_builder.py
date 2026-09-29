from __future__ import annotations
import argparse
import glob
import logging
from pathlib import Path
import pandas as pd
from hybrid_dag_builder import HybridConfig, build_hybrid_for_cell

DEFAULT_GLOB = "exports/full_grid/gamma_*"
SETTINGS_ALL = ["controlled", "uncontrolled"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB,
                        help=f"glob of cell directories (default: {DEFAULT_GLOB})")
    parser.add_argument("--settings", choices=SETTINGS_ALL + ["both"],
                        default="controlled",
                        help="which PCMCI+ setting to consume (default: "
                             "controlled — the main comparison setting)")
    parser.add_argument("--min-lag", type=int, default=1,
                        help="minimum lag for discovered channel edges "
                             "(default: 1 — drop contemporaneous links)")
    parser.add_argument("--keep-reverse-sales-edges", action="store_true",
                        help="keep discovered Sales->channel edges "
                             "(default: drop as implausible for MMM)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Silence library loggers: a NullHandler on the root logger stops Python's
    # last-resort handler from writing records to stderr.
    logging.getLogger().addHandler(logging.NullHandler())

    cell_dirs = sorted(Path(p) for p in glob.glob(args.cells)
                       if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(f"No cell directories match '{args.cells}'.")
    settings = SETTINGS_ALL if args.settings == "both" else [args.settings]

    rows: list[dict] = []
    failures: list[dict] = []
    for cell in cell_dirs:
        for setting in settings:
            if not (cell / "pcmci_plus" / setting).is_dir():
                continue                      # no PCMCI+ output for this cell
            out_dir = cell / "hybrid_dag" / setting
            if out_dir.exists() and not args.overwrite:
                continue                      # already built
            try:
                cfg = HybridConfig(
                    setting=setting, min_lag=args.min_lag,
                    keep_reverse_sales_edges=args.keep_reverse_sales_edges)
                rows.append(build_hybrid_for_cell(cell, cfg))
            except Exception as exc:
                failures.append({"cell": cell.name, "setting": setting,
                                 "error": repr(exc)})

    root = cell_dirs[0].parent
    if rows:
        pd.DataFrame(rows).to_csv(root / "hybrid_dag_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(root / "hybrid_dag_failures.csv",
                                      index=False)
        raise SystemExit(1)


if __name__ == "__main__":
    main()