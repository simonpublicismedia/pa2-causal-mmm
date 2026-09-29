from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import pandas as pd
from mmm_datagen import (
    ScenarioConfig,
    generate_dataset,
    interventional_effects,
    export_discovery_settings,
    GAMMA_LEVELS,
    NOISE_LEVELS,
)

# ===========================================================================
# Configuration
# ===========================================================================

GENERATOR_VERSION = "1.0"        # frozen DGP version (mmm_datagen.py)
RUN_MODE = "test"                # "test" | "full"  (CLI --mode overrides)
EXPORT_ROOT = Path("exports")
INTERVENTION_FACTOR = 1.10       # do(Spend x 1.10) for ground-truth effects
INTERVENTION_BURN_IN = 5         # weeks excluded from effect aggregation

# --- Mini test set (default): 3 cells x 3 seeds = 9 exports ---------------
TEST_GRID = {
    "interdependency": ["weak", "medium", "strong"],
    "noise": ["medium"],
    "n_weeks": [156],
    "dag_density": ["medium"],
    "seeds": [1, 2, 3],
}

# --- Full experimental grid (prepared, NOT run by default) -----------------
# 3 x 3 x 3 x 3 x 15 = 1,215 cells. Adjust seeds to change replicate count.
FULL_GRID = {
    "interdependency": ["weak", "medium", "strong"],
    "noise": ["low", "medium", "high"],
    "n_weeks": [80, 156, 300],
    "dag_density": ["sparse", "medium", "dense"],
    "seeds": list(range(1, 16)),
}

GRIDS = {"test": ("test_grid", TEST_GRID), "full": ("full_grid", FULL_GRID)}


# ===========================================================================
# Grid iteration
# ===========================================================================

def iter_cells(grid: dict):
    """Yield ScenarioConfig for every cell of a grid configuration."""
    for inter, noise, n_weeks, density, seed in product(
            grid["interdependency"], grid["noise"], grid["n_weeks"],
            grid["dag_density"], grid["seeds"]):
        yield ScenarioConfig(n_weeks=n_weeks, interdependency=inter,
                             noise=noise, dag_density=density, seed=seed)


def cell_dirname(cfg: ScenarioConfig) -> str:
    """gamma_weak__noise_medium__T_156__density_medium__seed_001"""
    return (f"gamma_{cfg.interdependency}"
            f"__noise_{cfg.noise}"
            f"__T_{cfg.n_weeks}"
            f"__density_{cfg.dag_density}"
            f"__seed_{cfg.seed:03d}")


# ===========================================================================
# Per-cell export
# ===========================================================================

def _interventional_to_frame(iv: dict) -> pd.DataFrame:
    """Flatten interventional_effects() output to one row per channel."""
    rows = []
    for ch, e in iv["channels"].items():
        rows.append({
            "channel": ch,
            "factor": iv["factor"],
            "burn_in_weeks": iv["burn_in_weeks"],
            "delta_sales_total": e["delta_sales_total"],
            "delta_sales_mean_weekly": e["delta_sales_mean_weekly"],
            "pct_sales_change": e["pct_sales_change"],
            "delta_own_spend": e["delta_own_spend"],
            "delta_total_media_spend": e["delta_total_media_spend"],
            "marginal_roas_own": e["marginal_roas_own"],
            "marginal_roas_total": e["marginal_roas_total"],
            "induced_downstream_spend_json": json.dumps(
                e["induced_downstream_spend"]),
        })
    return pd.DataFrame(rows)


def export_cell(cfg: ScenarioConfig, out_dir: Path, run_mode: str) -> dict:
    """Generate one scenario cell and write all required artifacts."""
    result = generate_dataset(cfg)
    gt = result["ground_truth"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- observational datasets (discovery settings) -----------------------
    settings = export_discovery_settings(result)
    settings["unobserved"].to_parquet(out_dir / "observed_uncontrolled.parquet",
                                      index=False)
    settings["controlled"].to_parquet(out_dir / "observed_controlled.parquet",
                                      index=False)

    # --- full diagnostics (all columns incl. latents, adstock, saturation) -
    result["df"].to_parquet(out_dir / "full_diagnostics.parquet", index=False)

    # --- ground truth ------------------------------------------------------
    adj = pd.DataFrame(gt["adjacency_matrix"])
    # adjacency_matrix dict is column-oriented; restore source-row layout
    adj.index.name = "source"
    adj.to_csv(out_dir / "true_adjacency_matrix.csv")

    pd.DataFrame(gt["lagged_edges"]).to_csv(
        out_dir / "true_lagged_edges.csv", index=False)

    iv = interventional_effects(result, factor=INTERVENTION_FACTOR,
                                burn_in=INTERVENTION_BURN_IN)
    _interventional_to_frame(iv).to_csv(
        out_dir / "true_interventional_effects.csv", index=False)

    true_params = dict(gt["true_parameters"])
    true_params["structural_path_effects_on_sales"] = \
        gt["structural_path_effects_on_sales"]
    with open(out_dir / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "run_mode": run_mode,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "intervention_factor": INTERVENTION_FACTOR,
        "intervention_burn_in_weeks": INTERVENTION_BURN_IN,
        **asdict(cfg),
    }
    with open(out_dir / "scenario_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(result["validation"], f, indent=2)

    # compact per-cell summary for the manifest
    vd = result["validation"]["variance_decomposition"]
    return {
        "cell": cell_dirname(cfg),
        "gamma": GAMMA_LEVELS[cfg.interdependency],
        "noise_target": NOISE_LEVELS[cfg.noise],
        "T": cfg.n_weeks,
        "density": cfg.dag_density,
        "seed": cfg.seed,
        "n_true_edges": len(gt["lagged_edges"]),
        "demand_share": vd["demand_share"],
        "media_share": vd["media_share"],
        "noise_share": vd["noise_share"],
        "lag_sanity_all_pass": all(
            v["lag_dominates"]
            for v in result["validation"]["lag_sanity"].values()),
        "correlations_positive":
            result["validation"]["correlations_positive"],
        "status": "ok",
    }


# ===========================================================================
# Main
# ===========================================================================

def check_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return
    except ImportError:
        raise RuntimeError("No parquet engine found. Install one first, "
                           "e.g.: pip install pyarrow")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(GRIDS), default=RUN_MODE,
                        help=f"grid to export (default: {RUN_MODE})")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-export cells whose directory already exists")
    parser.add_argument("--export-root", type=Path, default=EXPORT_ROOT,
                        help=f"output root directory (default: {EXPORT_ROOT})")
    args = parser.parse_args()

    check_parquet_engine()

    grid_name, grid = GRIDS[args.mode]
    root = args.export_root / grid_name
    cells = list(iter_cells(grid))

    manifest_rows, failures = [], []
    for cfg in cells:
        out_dir = root / cell_dirname(cfg)
        if out_dir.exists() and not args.overwrite:
            manifest_rows.append({"cell": out_dir.name, "status": "skipped"})
            continue
        try:
            manifest_rows.append(export_cell(cfg, out_dir, run_mode=args.mode))
        except Exception as exc:  # keep going, report via manifest + exit code
            failures.append(out_dir.name)
            manifest_rows.append({"cell": out_dir.name, "status": "failed",
                                  "error": repr(exc)})

    manifest = pd.DataFrame(manifest_rows)
    root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(root / "manifest.csv", index=False)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()