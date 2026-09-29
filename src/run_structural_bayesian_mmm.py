from __future__ import annotations
import argparse
import glob
import json
import logging
import traceback
from pathlib import Path
import pandas as pd
from structural_bayesian_mmm import (run_structural_mmm_on_cell, GRAPH_TYPES)

DEFAULT_GLOB = "exports/full_grid/gamma_*"
MANY_RUNS_GUARD = 60
SUMMARY_NAME = "structural_bayesian_mmm_summary.csv"
META_KEYS = ["interdependency", "noise", "n_weeks", "dag_density", "seed"]


def _cell_metadata(cell_dir: Path) -> dict:
    path = cell_dir / "scenario_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        meta = json.load(f)
    return {k: meta.get(k) for k in META_KEYS}


def _merge_summary(root: Path, new_rows: pd.DataFrame) -> Path:
    """Merge with an existing summary so incremental runs accumulate."""
    out_path = root / SUMMARY_NAME
    if out_path.exists():
        old = pd.read_csv(out_path)
        keep = ~old.set_index(["cell", "graph_type"]).index.isin(
            new_rows.set_index(["cell", "graph_type"]).index)
        new_rows = pd.concat([old[keep], new_rows], ignore_index=True)
    new_rows.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB,
                        help=f"glob of cell directories (default: {DEFAULT_GLOB})")
    parser.add_argument("--graph-types", nargs="+", choices=GRAPH_TYPES,
                        default=["hybrid"],
                        help="graph variants to fit (default: hybrid)")
    parser.add_argument("--pcmci-setting", default="controlled",
                        choices=["controlled", "uncontrolled"],
                        help="which PCMCI+/hybrid setting to load graphs from")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="mark all outputs as non-final smoke test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help=f"confirm runs exceeding {MANY_RUNS_GUARD} fits")
    args = parser.parse_args()

    # Silence PyMC/PyTensor: a NullHandler on the root logger stops Python's
    # last-resort handler from writing records to stderr.
    logging.getLogger().addHandler(logging.NullHandler())
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    cell_dirs = sorted(Path(p) for p in glob.glob(args.cells)
                       if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(f"No cell directories match '{args.cells}'.")

    n_fits = len(cell_dirs) * len(args.graph_types)
    if n_fits > MANY_RUNS_GUARD and not args.yes:
        raise RuntimeError(
            f"{n_fits} fits requested ({len(cell_dirs)} cells x "
            f"{len(args.graph_types)} graph types). This looks like a "
            f"large (full-grid) run — pass --yes to confirm.")

    rows: list[dict] = []
    failures: list[dict] = []
    for cell in cell_dirs:
        meta = _cell_metadata(cell)
        for gt in args.graph_types:
            out_dir = cell / "structural_bayesian_mmm" / gt
            if out_dir.exists() and not args.overwrite:
                continue                       # already fitted
            try:
                row = run_structural_mmm_on_cell(
                    cell, gt, draws=args.draws, tune=args.tune,
                    chains=args.chains, target_accept=args.target_accept,
                    seed=args.seed, pcmci_setting=args.pcmci_setting,
                    smoke_test=args.smoke_test)
                rows.append({**{"cell": row.pop("cell"),
                                "graph_type": row.pop("graph_type")},
                             **meta, **row})
            except Exception as exc:
                failures.append({"cell": cell.name, "graph_type": gt,
                                 **meta, "error": repr(exc),
                                 "traceback": traceback.format_exc()})

    root = cell_dirs[0].parent
    if rows:
        _merge_summary(root, pd.DataFrame(rows))
    if failures:
        pd.DataFrame(failures).to_csv(root / "structural_failures.csv",
                                      index=False)
        raise SystemExit(1)


if __name__ == "__main__":
    main()