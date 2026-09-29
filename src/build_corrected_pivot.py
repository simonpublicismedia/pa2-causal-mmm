from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CAUSAL_GRAPHS = ["pcmci_dag", "hybrid", "oracle_hybrid"]
GAMMA_ORDER = ["weak", "medium", "strong"]

METRICS = [
    ("diff_mae_effect", "MAE Effect"),
    ("diff_mae_vs_own", "MAE vs Own Effect"),
    ("diff_mae_vs_total_buggy", "MAE vs Total Effect -- BUGGY (zum Vergleich)"),
    ("diff_mae_vs_total_fixed", "MAE vs Total Effect -- FIXED"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True,
                    help="Pfad zu summary_099.csv")
    ap.add_argument("--benchmark", type=Path, required=True,
                    help="Pfad zu benchmark_summary.csv")
    ap.add_argument("--fixed", type=Path, required=True,
                    help="Pfad zu vs_total_fixed.csv (Output von fix_vs_total_metric.py)")
    ap.add_argument("--out-prefix", default="pivot")
    args = ap.parse_args()

    causal = pd.read_csv(args.summary)
    bench = pd.read_csv(args.benchmark)
    fixed = pd.read_csv(args.fixed)

    missing = [c for c in ("cell", "graph_type", "mae_effect", "mae_vs_own",
                           "mae_vs_total", "interdependency", "n_weeks")
               if c not in causal.columns]
    if missing:
        raise RuntimeError(f"{args.summary.name}: fehlende Spalten {missing}")

    # korrigiertes mae_vs_total einmischen
    fixed_small = fixed[["cell", "graph_type", "mae_vs_total_fixed"]]
    causal = causal.rename(columns={"mae_vs_total": "mae_vs_total_buggy"})
    causal = causal.merge(fixed_small, on=["cell", "graph_type"], how="left")
    n_missing_fix = int(causal["mae_vs_total_fixed"].isna().sum())

    bench_small = bench[["cell", "mae_effect", "mae_vs_own", "mae_vs_total"]].rename(
        columns={"mae_effect": "bench_mae_effect",
                 "mae_vs_own": "bench_mae_vs_own",
                 "mae_vs_total": "bench_mae_vs_total"})

    merged = causal.merge(bench_small, on="cell", how="left")
    n_no_bench = int(merged["bench_mae_effect"].isna().sum())

    merged["diff_mae_effect"] = merged["bench_mae_effect"] - merged["mae_effect"]
    merged["diff_mae_vs_own"] = merged["bench_mae_vs_own"] - merged["mae_vs_own"]
    merged["diff_mae_vs_total_buggy"] = (
        merged["bench_mae_vs_total"] - merged["mae_vs_total_buggy"])
    merged["diff_mae_vs_total_fixed"] = (
        merged["bench_mae_vs_total"] - merged["mae_vs_total_fixed"])

    merged.to_csv(Path(f"{args.out_prefix}_merged.csv"), index=False)

    # --- Pivots im Langformat statt auf der Konsole ------------------------
    pivot_rows = []
    for graph in CAUSAL_GRAPHS:
        sub = merged[merged["graph_type"] == graph]
        if sub.empty:
            continue
        for col, label in METRICS:
            pivot = sub.pivot_table(index="n_weeks", columns="interdependency",
                                    values=col, aggfunc="mean")
            cols_order = [c for c in GAMMA_ORDER if c in pivot.columns]
            if cols_order:
                pivot = pivot[cols_order]
            long = (pivot.stack(future_stack=True).rename("value")
                    .reset_index())
            long.insert(0, "graph_type", graph)
            long.insert(1, "metric", label)
            pivot_rows.append(long)
            print(f"\n{graph}: {label}")
            print(pivot.to_string(float_format=lambda v: f"{v:,.2f}"))

    if pivot_rows:
        pd.concat(pivot_rows, ignore_index=True).to_csv(
            Path(f"{args.out_prefix}_pivots.csv"), index=False)

    if n_missing_fix:
        print(f"\nHinweis: {n_missing_fix} Zeilen ohne korrigierten Wert "
              f"in {args.fixed.name}.")
    if n_no_bench:
        print(f"Hinweis: {n_no_bench} Zeilen ohne passende Benchmark-Zelle.")

    report = {
        "n_rows_merged": int(len(merged)),
        "n_rows_without_fixed_value": n_missing_fix,
        "n_rows_without_benchmark_cell": n_no_bench,
        "graph_types_present": sorted(merged["graph_type"].dropna().unique().tolist()),
    }
    with open(f"{args.out_prefix}_merge_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())