
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GRAPH_TYPES = ["pcmci_dag", "hybrid", "oracle_hybrid"]

CELL_NAME_RE = re.compile(
    r"gamma_(?P<gamma>\w+?)__noise_(?P<noise>\w+?)__T_(?P<T>\d+)"
    r"__density_(?P<density>\w+?)__seed_(?P<seed>\d+)"
)

# matches ".../<cell>/structural_bayesian_mmm/<graph>/estimated_effects.csv"
EST_RE = re.compile(
    r"(?P<cell>[^/]+)/structural_bayesian_mmm/(?P<graph>" +
    "|".join(GRAPH_TYPES) + r")/estimated_effects\.csv$")
# matches ".../<cell>/true_interventional_effects.csv"
TRUTH_RE = re.compile(r"(?P<cell>[^/]+)/true_interventional_effects\.csv$")


def parse_cell_name(name: str) -> dict:
    m = CELL_NAME_RE.search(name)
    if not m:
        return {"gamma": None, "noise": None, "T": None,
                "density": None, "seed": None}
    d = m.groupdict()
    d["T"] = int(d["T"])
    d["seed"] = int(d["seed"])
    return d


# ---------------------------------------------------------------------------
# Filesystem abstraction: local path or gs:// -- same code either way.
# ---------------------------------------------------------------------------

class Store:
    """Thin wrapper so the rest of the script doesn't care whether the grid
    lives on local disk or in a GCS bucket."""

    def __init__(self, grid: str):
        self.grid = grid.rstrip("/")
        self.is_gcs = self.grid.startswith("gs://")
        if self.is_gcs:
            try:
                import fsspec
            except ImportError as exc:
                raise RuntimeError(
                    "GCS-Modus braucht 'fsspec' und 'gcsfs': "
                    "pip install fsspec gcsfs") from exc
            self.fs = fsspec.filesystem("gcs")
        else:
            self.fs = None  # plain pathlib/glob

    def find_all(self) -> list[str]:
        """All file paths under the grid root (cheap: metadata/listing only,
        no content is downloaded)."""
        if self.is_gcs:
            return [p if p.startswith("gs://") else f"gs://{p}"
                    for p in self.fs.find(self.grid)]
        return [str(p) for p in Path(self.grid).rglob("*") if p.is_file()]

    def exists(self, path: str) -> bool:
        if self.is_gcs:
            return self.fs.exists(path)
        return Path(path).exists()

    def read_csv(self, path: str) -> pd.DataFrame:
        if self.is_gcs:
            with self.fs.open(path, "rb") as f:
                return pd.read_csv(f)
        return pd.read_csv(path)

    def read_json(self, path: str) -> dict:
        if self.is_gcs:
            with self.fs.open(path, "rb") as f:
                return json.load(f)
        return json.loads(Path(path).read_text())

    def sibling(self, path: str, filename: str) -> str:
        """Replace the last path component with `filename`."""
        return path.rsplit("/", 1)[0] + "/" + filename


def discover_pairs(store: Store, cell_contains: str | None):
    """Walk the grid once, return list of (cell, graph, est_path, truth_path)."""
    all_paths = store.find_all()

    est_by_cell_graph: dict[tuple[str, str], str] = {}
    truth_by_cell: dict[str, str] = {}

    for p in all_paths:
        m = EST_RE.search(p)
        if m:
            est_by_cell_graph[(m["cell"], m["graph"])] = p
            continue
        m = TRUTH_RE.search(p)
        if m:
            truth_by_cell[m["cell"]] = p

    pairs = []
    for (cell, graph), est_path in est_by_cell_graph.items():
        if cell_contains and cell_contains not in cell:
            continue
        truth_path = truth_by_cell.get(cell)
        if truth_path is None:
            continue
        pairs.append((cell, graph, est_path, truth_path))
    return pairs


# ---------------------------------------------------------------------------
# Core fix
# ---------------------------------------------------------------------------

def fixed_roas_total(est: pd.DataFrame) -> np.ndarray:
    """marginal_roas_total = delta_sales_total_mean / delta_total_media_spend_mean.
    NaN wenn kein induzierter Downstream-Spend vorhanden (Nenner ~ 0) --
    z.B. bei pcmci_dag-Zellen ohne entdeckte Kanal-Interdependenzen."""
    denom = est["delta_total_media_spend_mean"].to_numpy(float)
    numer = est["delta_sales_total_mean"].to_numpy(float)
    out = np.full_like(denom, np.nan, dtype=float)
    mask = np.abs(denom) > 1e-9
    out[mask] = numer[mask] / denom[mask]
    return out


def process_pair(store: Store, cell: str, graph: str,
                 est_path: str, truth_path: str) -> dict | None:
    est = store.read_csv(est_path)
    if "delta_total_media_spend_mean" not in est.columns:
        return None  # aelteres Format ohne dieses Feld -- ueberspringen
    truth = store.read_csv(truth_path)

    merged = est.merge(
        truth[["channel", "marginal_roas_total", "marginal_roas_own"]],
        on="channel", how="inner")
    if merged.empty:
        return None

    merged["marginal_roas_total_fixed"] = fixed_roas_total(merged)
    merged["bias_vs_total_fixed"] = (
        merged["marginal_roas_total_fixed"] - merged["marginal_roas_total"])
    merged["bias_vs_total_buggy"] = (
        merged["marginal_roas_mean"] - merged["marginal_roas_total"])

    valid = merged.dropna(subset=["bias_vs_total_fixed"])
    if valid.empty:
        return None

    orig_mae_vs_total = None
    agg_path = store.sibling(est_path, "comparison_aggregates.json")
    if store.exists(agg_path):
        try:
            orig_mae_vs_total = store.read_json(agg_path).get("mae_vs_total")
        except Exception:
            pass

    row = {
        "cell": cell,
        "graph_type": graph,
        "n_channels_used": int(len(valid)),
        "n_channels_skipped_no_downstream": int(len(merged) - len(valid)),
        "mae_vs_total_buggy_saved": orig_mae_vs_total,
        "mae_vs_total_buggy_recomputed": float(valid["bias_vs_total_buggy"].abs().mean()),
        "mae_vs_total_fixed": float(valid["bias_vs_total_fixed"].abs().mean()),
        "rmse_vs_total_fixed": float(np.sqrt((valid["bias_vs_total_fixed"] ** 2).mean())),
        "mean_bias_vs_total_fixed": float(valid["bias_vs_total_fixed"].mean()),
    }
    # Selbstkontrolle: reproduziert die Nachrechnung den gespeicherten Wert?
    # Nur aussagekraeftig, wenn kein Kanal uebersprungen wurde.
    rec = row["mae_vs_total_buggy_recomputed"]
    row["buggy_reproduction_rel_error"] = (
        abs(rec - orig_mae_vs_total) / max(abs(orig_mae_vs_total), 1e-9)
        if orig_mae_vs_total is not None else np.nan)
    row.update(parse_cell_name(cell))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True,
                    help="Lokaler Ordner (z.B. exports/full_grid) ODER "
                         "gs://bucket/prefix")
    ap.add_argument("--cell-contains", default=None,
                    help="Nur Zellen verarbeiten, deren Name diesen String "
                         "enthaelt (z.B. 'T_156')")
    ap.add_argument("--out", type=Path, default=Path("vs_total_fixed.csv"))
    args = ap.parse_args()

    store = Store(args.grid)
    pairs = discover_pairs(store, args.cell_contains)
    if not pairs:
        raise RuntimeError(
            "Keine (Zelle, Graphtyp)-Paare mit estimated_effects.csv + "
            "true_interventional_effects.csv gefunden -- --grid pruefen.")

    rows, skipped = [], []
    for cell, graph, est_path, truth_path in pairs:
        try:
            row = process_pair(store, cell, graph, est_path, truth_path)
        except Exception as exc:
            skipped.append({"cell": cell, "graph_type": graph,
                            "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if row is not None:
            rows.append(row)
        else:
            skipped.append({"cell": cell, "graph_type": graph,
                            "reason": "kein verwertbares Ergebnis"})

    if skipped:
        pd.DataFrame(skipped).to_csv(
            args.out.with_name(args.out.stem + "_skipped.csv"), index=False)
    if not rows:
        return 1

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print(f"{len(out)} Zellen/Graph-Kombinationen -> {args.out}"
          + (f"  ({len(skipped)} uebersprungen)" if skipped else ""))

    rep = out["buggy_reproduction_rel_error"].dropna()
    if len(rep):
        print(f"Reproduktion des gespeicherten Werts: max rel. Abweichung "
              f"{rep.max():.2e}  (muss ~0 sein)")
    n_skip_ch = int(out["n_channels_skipped_no_downstream"].sum())
    if n_skip_ch:
        print(f"Hinweis: {n_skip_ch} Kanal-Zeilen ohne Downstream-Spend "
              f"uebersprungen -- Reproduktionsfehler dort nicht aussagekraeftig.")

    summary = out.groupby("graph_type")[
        ["mae_vs_total_buggy_recomputed", "mae_vs_total_fixed"]].mean()
    print("\nDurchschnitt ueber alle verarbeiteten Zellen (vorher vs. nachher):")
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))

    if out["gamma"].notna().any():
        for label, col in (("VORHER (Bug)", "mae_vs_total_buggy_recomputed"),
                           ("NACHHER (korrigiert)", "mae_vs_total_fixed")):
            pv = out.pivot_table(index="T", columns="gamma", values=col,
                                 aggfunc="mean")
            print(f"\nPivot {label} -- T x gamma:")
            print(pv.to_string(float_format=lambda v: f"{v:,.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())