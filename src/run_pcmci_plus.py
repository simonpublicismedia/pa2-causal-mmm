from __future__ import annotations
import argparse
import glob
import json
import logging
import sys
from pathlib import Path
import pandas as pd
from pcmci_plus_pipeline import DiscoveryConfig, run_discovery_on_cell, \
    DEFAULT_TAU_MAX, DEFAULT_PC_ALPHA, DEFAULT_COND_IND_TEST

DEFAULT_GLOB = "exports/test_grid/gamma_*"
SETTINGS_ALL = ["controlled", "uncontrolled"]


def _cell_metadata(cell_dir: Path) -> dict:
    """Read scenario metadata for aggregation (never used in discovery)."""
    meta_path = cell_dir / "scenario_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    keys = ["interdependency", "noise", "n_weeks", "dag_density", "seed"]
    return {k: meta.get(k) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB,
                        help=f"glob of cell directories (default: {DEFAULT_GLOB})")
    parser.add_argument("--settings", choices=SETTINGS_ALL + ["both"],
                        default="both",
                        help="which discovery setting(s) to run (default: both)")
    parser.add_argument("--tau-max", type=int, default=DEFAULT_TAU_MAX)
    parser.add_argument("--pc-alpha", type=float, default=DEFAULT_PC_ALPHA)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run cells whose output directory exists")
    args = parser.parse_args()

    # Silence library loggers: a NullHandler on the root logger stops Python's
    # last-resort handler from writing records to stderr.
    logging.getLogger().addHandler(logging.NullHandler())

    cell_dirs = sorted(Path(p) for p in glob.glob(args.cells)
                       if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(
            f"No cell directories match '{args.cells}'. "
            "Run export_mmm_scenarios.py first.")
    settings = SETTINGS_ALL if args.settings == "both" else [args.settings]

    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for cell in cell_dirs:
        meta = _cell_metadata(cell)
        for setting in settings:
            out_dir = cell / "pcmci_plus" / setting
            tag = f"{cell.name}/{setting}"
            if out_dir.exists() and not args.overwrite:
                summary_rows.append({"cell": cell.name, "setting": setting,
                                     **meta, "status": "skipped"})
                continue
            try:
                cfg = DiscoveryConfig(setting=setting, tau_max=args.tau_max,
                                      pc_alpha=args.pc_alpha,
                                      cond_ind_test=DEFAULT_COND_IND_TEST)
                row = run_discovery_on_cell(cell, cfg)
                summary_rows.append({**row, **meta, "status": "ok"})
                detail = pd.read_csv(out_dir / "edge_recovery_detail.csv")
                detail.insert(0, "cell", cell.name)
                detail.insert(1, "setting", setting)
                for k, v in meta.items():
                    detail[k] = v
                detail_frames.append(detail)
            except Exception as exc:
                failures.append(tag)
                summary_rows.append({"cell": cell.name, "setting": setting,
                                     **meta, "status": "failed",
                                     "error": repr(exc)})

    root = cell_dirs[0].parent
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "pcmci_summary.csv", index=False)
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True) \
          .to_csv(root / "edge_recovery_long.csv", index=False)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()