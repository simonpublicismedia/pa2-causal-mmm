

from __future__ import annotations

import argparse
import glob
import logging
import time
from pathlib import Path

import pandas as pd

from benchmark_mmm import run_benchmark_on_cell

DEFAULT_GLOB = "exports/test_grid/gamma_*"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB,
                        help=f"glob of cell directories (default: {DEFAULT_GLOB})")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Silence PyMC/PyTensor: a NullHandler on the root logger stops Python's
    # last-resort handler from writing records to stderr; the explicit levels
    # mirror the worker setup in run_benchmark_parallel.py.
    logging.getLogger().addHandler(logging.NullHandler())
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    cell_dirs = sorted(p for p in glob.glob(args.cells) if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(
            f"No cell directories match '{args.cells}'. "
            "Run export_mmm_scenarios.py first.")

    rows: list[dict] = []
    failed = 0
    for cell in cell_dirs:
        t0 = time.time()
        try:
            row = run_benchmark_on_cell(
                cell, draws=args.draws, tune=args.tune, chains=args.chains,
                target_accept=args.target_accept, seed=args.seed)
            row["runtime_s"] = round(time.time() - t0, 1)
            rows.append(row)
        except Exception as exc:
            failed += 1
            rows.append({"cell": Path(cell).name, "converged": False,
                         "error": repr(exc),
                         "runtime_s": round(time.time() - t0, 1)})

    out_path = Path(cell_dirs[0]).parent / "benchmark_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()