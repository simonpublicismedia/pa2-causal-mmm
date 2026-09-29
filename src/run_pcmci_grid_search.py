from __future__ import annotations
import argparse
import logging
from pathlib import Path
from pcmci_grid_search import (
    GridSpec, run_grid_search, resolve_tuning_cells,
    DEFAULT_ROOT, DEFAULT_OUT_DIR, DEFAULT_MIN_PRECISION,
    COND_IND_TESTS, TAU_MAX_VALUES, PC_ALPHA_VALUES,
)

def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=None,
                        help="glob overriding the predefined tuning set "
                             "(default: the seven predefined cells under "
                             f"{DEFAULT_ROOT})")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"root of the exported grid (default: {DEFAULT_ROOT})")
    parser.add_argument("--settings", nargs="+",
                        choices=["controlled", "uncontrolled"],
                        default=["controlled"],
                        help="discovery setting(s) (default: controlled; "
                             "uncontrolled is meant as a later sensitivity "
                             "analysis)")
    parser.add_argument("--tests", nargs="+", choices=COND_IND_TESTS,
                        default=list(COND_IND_TESTS))
    parser.add_argument("--tau-max-values", nargs="+", type=int,
                        default=list(TAU_MAX_VALUES))
    parser.add_argument("--pc-alpha-values", nargs="+", type=float,
                        default=list(PC_ALPHA_VALUES))
    parser.add_argument("--min-precision", type=float,
                        default=DEFAULT_MIN_PRECISION,
                        help="precision constraint of the selection rule "
                             f"(default: {DEFAULT_MIN_PRECISION})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into an out-dir that already "
                             "contains grid_search_runs.csv")
    args = parser.parse_args()

    # Silence the library loggers: a NullHandler on the root logger stops
    # Python's last-resort handler from writing records to stderr.
    logging.getLogger().addHandler(logging.NullHandler())

    if (args.out_dir / "grid_search_runs.csv").exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.out_dir} already contains grid search results. "
            "Use --overwrite to re-run, or choose another --out-dir.")

    cells = resolve_tuning_cells(args.root, args.cells)

    spec = GridSpec(settings=args.settings, cond_ind_tests=args.tests,
                    tau_max_values=args.tau_max_values,
                    pc_alpha_values=args.pc_alpha_values)

    return run_grid_search(cells, spec, args.out_dir,
                           min_precision=args.min_precision)


if __name__ == "__main__":
    main()