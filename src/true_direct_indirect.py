from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

BURN_IN_DEFAULT = 5
FACTOR_DEFAULT = 1.1
TOL_TRANSFORM = 1e-3      # Rekonstruktion vs. gespeicherte Spalten
TOL_IDENTITY = 1e-2       # Blattkanal-Identitaet direkt == total


# ---------------------------------------------------------------------------
# Transformationen (Formen an den gespeicherten Spalten verifiziert)
# ---------------------------------------------------------------------------

def adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """Rekursiv, NICHT normalisiert: a[t] = x[t] + theta * a[t-1]."""
    out = np.empty(len(x), dtype=float)
    carry = 0.0
    for t in range(len(x)):
        carry = float(x[t]) + theta * carry
        out[t] = carry
    return out


def hill(x: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """x^alpha / (x^alpha + k^alpha)."""
    xa = np.power(np.maximum(np.asarray(x, dtype=float), 0.0), alpha)
    return xa / (xa + np.power(k, alpha))


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Nach hinten verschieben, vorne mit 0 auffuellen."""
    if lag <= 0:
        return x.copy()
    out = np.zeros_like(x)
    out[lag:] = x[:len(x) - lag]
    return out


# ---------------------------------------------------------------------------
# Struktur
# ---------------------------------------------------------------------------

def outgoing_channels(cell: Path, channels: list[str]) -> set[str]:
    """Kanaele mit mindestens einer ausgehenden Kanal-zu-Kanal-Kante."""
    p = cell / "true_lagged_edges.csv"
    if p.exists():
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        s = cols.get("source") or cols.get("from") or cols.get("parent")
        t = cols.get("target") or cols.get("to") or cols.get("child")
        if s and t:
            return {str(r[s]) for _, r in df.iterrows()
                    if str(r[s]) in channels and str(r[t]) in channels}
    p = cell / "true_adjacency_matrix.csv"
    if p.exists():
        adj = pd.read_csv(p, index_col=0)
        out = set()
        for src in adj.index:
            if str(src) not in channels:
                continue
            for dst in adj.columns:
                if str(dst) in channels and float(adj.loc[src, dst]) != 0:
                    out.add(str(src))
        return out
    return set()


# ---------------------------------------------------------------------------
# Zelle
# ---------------------------------------------------------------------------

def run_cell(cell: Path) -> tuple[pd.DataFrame | None, str]:
    """Gibt (Ergebnis, Fehlergrund) zurueck; Ergebnis None heisst nicht bestanden."""
    diag = pd.read_parquet(cell / "full_diagnostics.parquet")
    params = json.loads((cell / "true_parameters.json").read_text())
    true_eff = pd.read_csv(cell / "true_interventional_effects.csv")

    meta_p = cell / "scenario_metadata.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    factor = float(meta.get("intervention_factor", FACTOR_DEFAULT))
    burn_in = int(meta.get("intervention_burn_in_weeks", BURN_IN_DEFAULT))

    betas = params.get("betas", {})
    thetas = params["adstock_theta"]
    alphas = params["hill_alpha"]
    ks = params["hill_k"]
    lags = params.get("sales_lags", {})
    channels = [c for c in thetas if c in diag.columns]

    leaves = set(channels) - outgoing_channels(cell, channels)

    # --- Validierung 1: Transformationen rekonstruierbar? -------------------
    worst_tr, worst_name = 0.0, ""
    for ch in channels:
        spend = diag[ch].to_numpy(float)
        ad = adstock(spend, thetas[ch])
        sa = hill(ad, alphas[ch], ks[ch])
        for suffix, rec in (("_Adstock", ad), ("_Saturated", sa)):
            col = f"{ch}{suffix}"
            if col not in diag.columns:
                continue
            stored = diag[col].to_numpy(float)
            scale = max(float(np.max(np.abs(stored))), 1e-9)
            err = float(np.max(np.abs(rec - stored)) / scale)
            if err > worst_tr:
                worst_tr, worst_name = err, col
    if worst_tr > TOL_TRANSFORM:
        return None, (f"Transformation weicht ab ({worst_name}, "
                      f"rel. {worst_tr:.2e})")

    # --- Zerlegung ----------------------------------------------------------
    sl = slice(burn_in, None)
    rows = []
    for ch in channels:
        beta = float(betas.get(ch, 0.0))          # Display fehlt in betas -> 0
        lag = int(lags.get(ch, 0))
        spend = diag[ch].to_numpy(float)

        sat_base = hill(adstock(spend, thetas[ch]), alphas[ch], ks[ch])
        sat_cf = hill(adstock(spend * factor, thetas[ch]), alphas[ch], ks[ch])
        contrib = beta * (shift(sat_cf, lag) - shift(sat_base, lag))
        d_direct = float(contrib[sl].sum())

        ref = true_eff.loc[true_eff["channel"] == ch]
        if ref.empty:
            continue
        d_total = float(ref["delta_sales_total"].iloc[0])
        d_own = float(ref["delta_own_spend"].iloc[0])

        is_leaf = ch in leaves
        ident = (abs(d_direct - d_total) / max(abs(d_total), 1e-9)
                 if is_leaf else np.nan)

        rows.append({
            "cell": cell.name,
            "channel": ch,
            "factor": factor,
            "burn_in_weeks": burn_in,
            "true_direct": d_direct,
            "true_indirect": d_total - d_direct,
            "true_total": d_total,
            "true_indirect_share": (d_total - d_direct) / d_total
                                   if abs(d_total) > 1e-9 else np.nan,
            "delta_own_spend": d_own,
            "has_outgoing_edges": not is_leaf,
            "beta_direct_sales": beta,
            "identity_check_leaf": ident,
            "transform_max_rel_error": worst_tr,
        })

    out = pd.DataFrame(rows)
    leaf_err = out["identity_check_leaf"].dropna()
    ok = bool(len(leaf_err) > 0 and leaf_err.max() <= TOL_IDENTITY)
    out["validated"] = ok
    out["leaf_identity_max"] = (float(leaf_err.max()) if len(leaf_err)
                                else np.nan)

    if len(leaf_err) == 0:
        return None, "kein Blattkanal vorhanden, Identitaet nicht pruefbar"
    if not ok:
        return None, f"Blatt-Identitaet verletzt (max {leaf_err.max():.2e})"
    return out, ""


# ---------------------------------------------------------------------------
# Vergleich mit der Modellseite
# ---------------------------------------------------------------------------

def compare(cell: Path, gt: pd.DataFrame, graph: str) -> pd.DataFrame | None:
    """None, wenn der Modell-Output fehlt (direct_indirect_plugin.py zuerst)."""
    p = cell / "structural_bayesian_mmm" / graph / "direct_indirect_effects.csv"
    if not p.exists():
        return None
    mod = pd.read_csv(p)[["channel", "indirect_share"]]
    mod = mod.rename(columns={"indirect_share": f"share_{graph}"})
    m = gt[["channel", "true_indirect_share"]].merge(mod, on="channel", how="left")
    m[f"bias_{graph}"] = m[f"share_{graph}"] - m["true_indirect_share"]
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path)
    ap.add_argument("--grid", type=Path)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", default=None,
                    help="Graphtyp, dessen indirect_share gegengestellt wird")
    args = ap.parse_args()

    if args.cell:
        out, reason = run_cell(args.cell)
        if out is not None:
            print(f"[{args.cell.name}] Transformation rel. "
                  f"{out['transform_max_rel_error'].iloc[0]:.1e} | "
                  f"Blatt-Identitaet max {out['leaf_identity_max'].iloc[0]:.2e}"
                  f" -> OK")
        else:
            print(f"[{args.cell.name}] FEHLGESCHLAGEN: {reason}",
                  file=sys.stderr)
        if out is None:
            pd.DataFrame([{"cell": args.cell.name, "reason": reason}]).to_csv(
                args.cell / "true_direct_indirect_failed.csv", index=False)
            return 1
        dest = args.out or (args.cell / "true_direct_indirect_effects.csv")
        out.to_csv(dest, index=False)
        show = ["channel", "true_direct", "true_indirect", "true_total",
                "true_indirect_share", "has_outgoing_edges"]
        print(out[show].to_string(index=False,
                                  float_format=lambda v: f"{v:,.3f}"))
        if args.compare:
            m = compare(args.cell, out, args.compare)
            if m is not None:
                m.to_csv(dest.with_name(
                    dest.stem + f"_vs_{args.compare}.csv"), index=False)
        return 0

    if args.grid:
        cells = sorted(p for p in glob(str(args.grid / args.pattern))
                       if Path(p).is_dir())
        frames, failed = [], []
        for c in cells:
            try:
                r, reason = run_cell(Path(c))
            except Exception as exc:
                failed.append({"cell": Path(c).name,
                               "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if r is not None:
                frames.append(r)
                print(f"[{Path(c).name}] Transformation rel. "
                      f"{r['transform_max_rel_error'].iloc[0]:.1e} | "
                      f"Blatt-Identitaet max "
                      f"{r['leaf_identity_max'].iloc[0]:.2e} -> OK")
            else:
                failed.append({"cell": Path(c).name, "reason": reason})
                print(f"[{Path(c).name}] FEHLGESCHLAGEN: {reason}",
                      file=sys.stderr)
        dest = args.out or Path("true_direct_indirect.csv")
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
        if failed:
            pd.DataFrame(failed).to_csv(
                dest.with_name(dest.stem + "_failed.csv"), index=False)
        print(f"\n{len(frames)}/{len(cells)} Zellen validiert -> {dest}")
        return 0 if frames and not failed else 1

    raise SystemExit("--cell oder --grid noetig")


if __name__ == "__main__":
    raise SystemExit(main())