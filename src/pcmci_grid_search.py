from __future__ import annotations
import itertools
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import tigramite
from pcmci_plus_pipeline import (
    DiscoveryConfig, SYSTEM_VARS, TARGET,
    load_cell_dataset, run_diagnostics, prepare_arrays, run_pcmci_plus,
    extract_edges, build_adjacency, load_ground_truth,
    evaluate_graph_recovery,
)

log = logging.getLogger("pcmci_grid_search")

# ---------------------------------------------------------------------------
# Predefined tuning set (validation cells for configuration selection).
# The seven cells span the three dimensions most relevant for PCMCI+
# performance (interdependency strength, noise level, series length) around
# the medium anchor cell. They live under exports/full_grid/.
# ---------------------------------------------------------------------------

TUNING_CELLS: list[str] = [
    "gamma_weak__noise_medium__T_156__density_medium__seed_001",
    "gamma_medium__noise_medium__T_156__density_medium__seed_001",
    "gamma_strong__noise_medium__T_156__density_medium__seed_001",
    "gamma_medium__noise_low__T_156__density_medium__seed_001",
    "gamma_medium__noise_high__T_156__density_medium__seed_001",
    "gamma_medium__noise_medium__T_80__density_medium__seed_001",
    "gamma_medium__noise_medium__T_300__density_medium__seed_001",
]

DEFAULT_ROOT = Path("exports/full_grid")
DEFAULT_OUT_DIR = Path("exports/pcmci_grid_search")

# Default grid: 2 tests x 3 tau_max x 5 pc_alpha = 30 configurations.
COND_IND_TESTS: list[str] = ["ParCorr", "RobustParCorr"]
TAU_MAX_VALUES: list[int] = [2, 3, 4]
PC_ALPHA_VALUES: list[float] = [0.05, 0.075, 0.10, 0.15, 0.20]
# alpha_level is not used separately in edge extraction (PCMCI+ links are
# taken directly from the returned graph), so it is fixed and not expanded.
ALPHA_LEVEL_VALUES: list[float] = [0.05]

DEFAULT_MIN_PRECISION: float = 0.70

CONFIG_KEYS = ["setting", "cond_ind_test", "tau_max", "pc_alpha"]
METRIC_KEYS = ["precision", "recall", "f1", "lag_accuracy", "shd"]
SUBGROUP_KEYS = ["precision_channel", "recall_channel", "f1_channel",
                 "precision_sales", "recall_sales", "f1_sales"]


@dataclass
class GridSpec:
    settings: list[str] = field(default_factory=lambda: ["controlled"])
    cond_ind_tests: list[str] = field(default_factory=lambda: list(COND_IND_TESTS))
    tau_max_values: list[int] = field(default_factory=lambda: list(TAU_MAX_VALUES))
    pc_alpha_values: list[float] = field(default_factory=lambda: list(PC_ALPHA_VALUES))

    def configs(self):
        for setting, test, tau, alpha in itertools.product(
                self.settings, self.cond_ind_tests,
                self.tau_max_values, self.pc_alpha_values):
            yield DiscoveryConfig(setting=setting, tau_max=tau,
                                  pc_alpha=alpha, cond_ind_test=test)

    def n_configs(self) -> int:
        return (len(self.settings) * len(self.cond_ind_tests)
                * len(self.tau_max_values) * len(self.pc_alpha_values))


# ---------------------------------------------------------------------------
# Tuning-cell resolution
# ---------------------------------------------------------------------------

def resolve_tuning_cells(root: Path, cells_glob: str | None) -> list[Path]:
    """
    Return the cell directories to tune on. Without an explicit override,
    exactly the predefined seven tuning cells are used (no folder scanning,
    no random sampling); missing cells raise, so the tuning set is never
    silently reduced.
    """
    if cells_glob:
        import glob as _glob
        dirs = sorted(Path(p) for p in _glob.glob(cells_glob)
                      if Path(p).is_dir())
        if not dirs:
            raise FileNotFoundError(f"no cell directories match '{cells_glob}'")
        return dirs
    dirs = [root / name for name in TUNING_CELLS]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(
            "predefined tuning cells missing:\n  " + "\n  ".join(missing)
            + "\nRun the full-grid export first (export_mmm_scenarios.py "
              "--mode full) or override with --cells.")
    return dirs


def _cell_metadata(cell_dir: Path) -> dict:
    meta_path = cell_dir / "scenario_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return {k: meta.get(k) for k in
            ["interdependency", "noise", "n_weeks", "dag_density", "seed"]}


# ---------------------------------------------------------------------------
# Subgroup metrics (channel-to-channel vs. media-to-sales edges)
# ---------------------------------------------------------------------------

def subgroup_metrics(edges: pd.DataFrame, true_adj: pd.DataFrame,
                     nodes: list[str] = SYSTEM_VARS) -> dict:
    """
    Precision/recall/F1 computed separately for the two edge families that
    matter for the later hybrid causal MMM:
      * channel edges:  source and target are media channels (interdependencies)
      * sales edges:    media channel -> Sales (direct sales effects)
    """
    disc_adj = build_adjacency(edges, nodes)
    true = {(s, t) for s in nodes for t in nodes if true_adj.loc[s, t]}
    disc = {(s, t) for s in nodes for t in nodes if disc_adj.loc[s, t]}

    def _prf(t_sub: set, d_sub: set) -> tuple[float, float, float]:
        tp = len(t_sub & d_sub)
        p = tp / len(d_sub) if d_sub else 0.0
        r = tp / len(t_sub) if t_sub else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    is_channel = lambda s, t: s != TARGET and t != TARGET
    is_sales = lambda s, t: t == TARGET and s != TARGET

    pc, rc, fc = _prf({e for e in true if is_channel(*e)},
                      {e for e in disc if is_channel(*e)})
    ps, rs, fs = _prf({e for e in true if is_sales(*e)},
                      {e for e in disc if is_sales(*e)})
    return {"precision_channel": pc, "recall_channel": rc, "f1_channel": fc,
            "precision_sales": ps, "recall_sales": rs, "f1_sales": fs}


# ---------------------------------------------------------------------------
# Single run (in-memory; grid search writes NO files into the cells)
# ---------------------------------------------------------------------------

def run_single(cell_dir: Path, cfg: DiscoveryConfig) -> dict:
    """
    One cell x one configuration. Discovery uses observational data only;
    ground truth enters strictly afterwards for evaluation.
    Returns a flat result row; never raises (errors are recorded).
    """
    meta = _cell_metadata(cell_dir)
    row = {"cell": cell_dir.name, "setting": cfg.setting, **meta,
           "cond_ind_test": cfg.cond_ind_test, "tau_max": cfg.tau_max,
           "pc_alpha": cfg.pc_alpha, "status": "ok", "error_message": ""}
    t0 = time.time()
    try:
        df = load_cell_dataset(cell_dir, cfg.setting)
        run_diagnostics(df, cell_dir, cfg.setting, cfg.tau_max)
        data, var_names = prepare_arrays(df, cfg.setting)
        results = run_pcmci_plus(data, var_names, cfg)
        edges = extract_edges(results)

        true_adj, true_edges = load_ground_truth(cell_dir)   # evaluation only
        metrics, _ = evaluate_graph_recovery(edges, true_adj, true_edges)
        row.update({k: metrics[k] for k in METRIC_KEYS})
        row.update({k: metrics[k] for k in
                    ["n_true_edges", "n_discovered_edges", "true_positives",
                     "false_positives", "false_negatives"]})
        row["fp_edges"] = ";".join(metrics["fp_edges"])
        row["fn_edges"] = ";".join(metrics["fn_edges"])
        row.update(subgroup_metrics(edges, true_adj))
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = repr(exc)
        log.exception("run failed: %s | %s", cell_dir.name, asdict(cfg))
    row["runtime_s"] = round(time.time() - t0, 2)
    return row


# ---------------------------------------------------------------------------
# Aggregation, ranking, recommendation
# ---------------------------------------------------------------------------

def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration, aggregated over tuning cells."""
    rows = []
    for keys, grp in runs.groupby(CONFIG_KEYS):
        ok = grp[grp["status"] == "ok"]
        row = dict(zip(CONFIG_KEYS, keys))
        row.update({"n_runs": len(grp), "n_success": len(ok),
                    "n_failed": int((grp["status"] == "failed").sum())})
        for m in METRIC_KEYS + SUBGROUP_KEYS:
            row[f"mean_{m}"] = round(float(ok[m].mean()), 4) if len(ok) else np.nan
            row[f"std_{m}"] = round(float(ok[m].std(ddof=0)), 4) if len(ok) else np.nan
        for m in ["true_positives", "false_positives", "false_negatives",
                  "n_discovered_edges"]:
            row[f"mean_{m}"] = round(float(ok[m].mean()), 4) if len(ok) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def precision_constrained_order(summary: pd.DataFrame,
                                min_precision: float) -> pd.DataFrame:
    """
    Selection rule: keep configs with mean_precision >= min_precision, then
    sort by highest mean_f1, then lowest mean_shd, then highest
    mean_lag_accuracy. Returns the (possibly empty) ordered candidate set.
    """
    cand = summary[summary["mean_precision"] >= min_precision].copy()
    return cand.sort_values(
        by=["mean_f1", "mean_shd", "mean_lag_accuracy"],
        ascending=[False, True, False])


def build_rankings(summary: pd.DataFrame,
                   min_precision: float) -> pd.DataFrame:
    """Per-configuration rank columns for several criteria."""
    out = summary.copy()
    out["rank_by_f1"] = out["mean_f1"].rank(ascending=False, method="min")
    out["rank_by_precision"] = out["mean_precision"].rank(ascending=False,
                                                          method="min")
    out["rank_by_recall"] = out["mean_recall"].rank(ascending=False,
                                                    method="min")
    out["rank_by_shd"] = out["mean_shd"].rank(ascending=True, method="min")
    ordered = precision_constrained_order(out, min_precision)
    out["rank_by_precision_constrained_f1"] = np.nan
    for r, idx in enumerate(ordered.index, start=1):
        out.loc[idx, "rank_by_precision_constrained_f1"] = r
    rank_cols = [c for c in out.columns if c.startswith("rank_by_")]
    return out.sort_values("rank_by_precision_constrained_f1",
                           na_position="last")[CONFIG_KEYS + rank_cols
                                               + [f"mean_{m}" for m in METRIC_KEYS]]


def recommend_config(summary: pd.DataFrame, min_precision: float,
                     primary_setting: str) -> dict:
    """
    Recommendation for ONE global configuration (per the selection rule),
    computed on the primary setting. This is a recommendation only — the
    final configuration is frozen by the researcher, not by this script.
    """
    subset = summary[summary["setting"] == primary_setting]
    ordered = precision_constrained_order(subset, min_precision)
    rule = (f"mean_precision >= {min_precision}, then highest mean_f1, "
            f"then lowest mean_shd, then highest mean_lag_accuracy")
    note = ("This is a recommendation based on the predefined tuning set. "
            "The final configuration should be frozen before downstream "
            "causal MMM evaluation.")
    if ordered.empty:
        fallback = subset.sort_values(by=["mean_precision", "mean_f1"],
                                      ascending=False).head(1)
        best = fallback.iloc[0] if len(fallback) else None
        return {"recommended_config": None if best is None else
                {k: best[k] for k in CONFIG_KEYS},
                "selection_rule": rule,
                "constraint_satisfied": False,
                "metrics": None if best is None else
                {f"mean_{m}": float(best[f"mean_{m}"]) for m in METRIC_KEYS},
                "note": ("No configuration met the precision constraint on "
                         "the tuning set; the listed configuration is the "
                         "closest fallback (max precision, then F1). ") + note}
    best = ordered.iloc[0]
    return {"recommended_config": {k: (int(best[k]) if k == "tau_max"
                                       else float(best[k]) if k == "pc_alpha"
                                       else str(best[k]))
                                   for k in CONFIG_KEYS},
            "selection_rule": rule,
            "constraint_satisfied": True,
            "metrics": {f"mean_{m}": float(best[f"mean_{m}"])
                        for m in METRIC_KEYS},
            "note": note}


# ---------------------------------------------------------------------------
# Orchestration + export (I/O kept separate from the logic above)
# ---------------------------------------------------------------------------

def run_grid_search(cell_dirs: list[Path], spec: GridSpec, out_dir: Path,
                    min_precision: float = DEFAULT_MIN_PRECISION) -> dict:
    """Execute the full grid, write all outputs, return the recommendation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = list(spec.configs())
    total = len(configs) * len(cell_dirs)
    log.info("grid search: %d configurations x %d cells = %d runs",
             len(configs), len(cell_dirs), total)

    rows: list[dict] = []
    n = 0
    t0 = time.time()
    for cfg in configs:
        for k, cell in enumerate(cell_dirs, start=1):
            n += 1
            log.info("Running cell %d/%d | setting=%s | test=%s | "
                     "tau_max=%d | pc_alpha=%s   [%d/%d total]",
                     k, len(cell_dirs), cfg.setting, cfg.cond_ind_test,
                     cfg.tau_max, cfg.pc_alpha, n, total)
            rows.append(run_single(cell, cfg))

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "grid_search_runs.csv", index=False)

    summary = aggregate_runs(runs)
    summary.to_csv(out_dir / "grid_search_summary.csv", index=False)

    rankings = build_rankings(summary, min_precision)
    rankings.to_csv(out_dir / "grid_search_rankings.csv", index=False)

    primary = "controlled" if "controlled" in spec.settings else spec.settings[0]
    recommendation = recommend_config(summary, min_precision, primary)
    with open(out_dir / "selected_config_recommendation.json", "w") as f:
        json.dump(recommendation, f, indent=2)

    with open(out_dir / "grid_search_config.json", "w") as f:
        json.dump({
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "tuning_cells": [str(c) for c in cell_dirs],
            "grid": asdict(spec),
            "min_precision": min_precision,
            "alpha_level_values": ALPHA_LEVEL_VALUES,
            "runtime_s": round(time.time() - t0, 1),
            "n_runs": len(runs),
            "n_failed": int((runs["status"] == "failed").sum()),
            "tigramite_version": getattr(tigramite, "__version__", "unknown"),
        }, f, indent=2)

    return {"runs": runs, "summary": summary, "rankings": rankings,
            "recommendation": recommendation}