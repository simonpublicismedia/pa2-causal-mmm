from __future__ import annotations

import argparse
import inspect as _inspect
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import structural_bayesian_mmm as sbm
from structural_bayesian_mmm import (
    MEDIA_VARS, CONTROL_VARS, INTERVENTION_FACTOR, BURN_IN_WEEKS,
    load_graph, prepare_data, param_var_names,
    counterfactual_sales, _sales_expectation,
)
from benchmark_mmm import load_cell_data

TOL_STRUCT = 0.01      # harte Schranke fuer own/media spend
RATIO_WARN = 0.15      # nur Warnung fuer das Niveau


# ---------------------------------------------------------------------------
# Posterior-Mittelwerte -> draws-Dict mit S = 1
# ---------------------------------------------------------------------------

def read_means(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    name_col = df.columns[0]
    mean_col = next((c for c in ("mean", "Mean", "posterior_mean") if c in df.columns), None)
    if mean_col is None:
        raise RuntimeError(f"Keine mean-Spalte in {path}: {list(df.columns)}")
    return {str(k).strip(): float(v) for k, v in zip(df[name_col], df[mean_col])}


def _vec(means: dict, base: str, labels: list[str]) -> np.ndarray:
    out = []
    for lb in labels:
        key = f"{base}[{lb}]"
        if key not in means:
            raise KeyError(f"Parameter fehlt in posterior_summary.csv: {key}")
        out.append(means[key])
    return np.asarray(out, dtype=float)[None, :]          # (1, n)


def _scalar(means: dict, name: str) -> np.ndarray:
    if name not in means:
        raise KeyError(f"Parameter fehlt in posterior_summary.csv: {name}")
    return np.asarray([means[name]], dtype=float)          # (1,)


def build_draws(means: dict, spec) -> dict:
    """Kuenstlicher Posterior mit einem Draw, Formen wie posterior_draws()."""
    d: dict = {
        "intercept_sales": _scalar(means, "intercept_sales"),
        "sigma_sales": _scalar(means, "sigma_sales"),
        "beta_control_sales": _vec(means, "beta_control_sales", CONTROL_VARS),
    }
    if spec.sales_parents:
        for base in ("beta_media_sales", "adstock_theta", "hill_alpha", "hill_k"):
            d[base] = _vec(means, base, list(spec.sales_parents))
    for node in spec.endogenous_nodes:
        parent_sources = [e["source"] for e in spec.parents_of(node)]
        d[f"intercept_{node}"] = _scalar(means, f"intercept_{node}")
        d[f"delta_{node}"] = _vec(means, f"delta_{node}", parent_sources)
        d[f"beta_control_{node}"] = _vec(means, f"beta_control_{node}", CONTROL_VARS)
        d[f"sigma_{node}"] = _scalar(means, f"sigma_{node}")

    missing = [n for n in param_var_names(spec) if n not in d]
    if missing:
        raise KeyError(f"draws-Dict unvollstaendig, fehlt: {missing}")
    return d


# ---------------------------------------------------------------------------
# Kernrechnung
# ---------------------------------------------------------------------------

def model_dir(cell: Path, graph: str) -> Path:
    p = cell / "structural_bayesian_mmm" / graph
    return p if p.exists() else cell / graph


def _load_graph_compat(cell: Path, graph: str):
    """load_graph hat je nach Version Zusatzargumente mit Defaults."""
    try:
        return load_graph(cell, graph)
    except TypeError as exc:
        sig = _inspect.signature(load_graph)
        raise RuntimeError(
            f"load_graph konnte nicht aufgerufen werden ({exc}). "
            f"Signatur: {sig}") from exc


def run_cell(cell: Path, graph: str) -> tuple[pd.DataFrame | None, str]:
    """Gibt (Ergebnis, Fehlergrund) zurueck; None heisst Struktur-Check nicht bestanden."""
    mdir = model_dir(cell, graph)
    est = pd.read_csv(mdir / "estimated_effects.csv")
    means = read_means(mdir / "posterior_summary.csv")

    df = load_cell_data(cell)
    spec = _load_graph_compat(cell, graph)
    data = prepare_data(df, spec)
    draws = build_draws(means, spec)

    factor = float(est["factor"].iloc[0]) if "factor" in est else INTERVENTION_FACTOR
    burn_in = int(est["burn_in_weeks"].iloc[0]) if "burn_in_weeks" in est else BURN_IN_WEEKS
    sl = slice(burn_in, None)

    base_media = data["media_scaled"]
    mu_base = _sales_expectation(draws, data, spec, base_media)          # (1, T)

    rows = []
    for ch in MEDIA_VARS:
        ci = data["media_index"][ch]

        # --- total: volle strukturelle Propagation -------------------------
        cf = counterfactual_sales(draws, data, spec, ch, factor)
        d_total = float(cf["d_sales_scaled"][:, sl].sum(axis=1)[0] * data["sales_scale"])

        # --- direkt: Mediatoren auf Baseline eingefroren --------------------
        cf_frozen = base_media.copy()
        cf_frozen[:, ci] = cf_frozen[:, ci] * factor
        mu_dir = _sales_expectation(draws, data, spec, cf_frozen)
        d_direct = float((mu_dir - mu_base)[:, sl].sum(axis=1)[0] * data["sales_scale"])

        # --- Spend-Buchhaltung wie im Original ------------------------------
        d_own = float((factor - 1.0) * data["media_raw"][sl, ci].sum())
        diff = (cf["cf_media"][0][sl] - base_media[sl]) * data["media_max"][None, :]
        d_media = float(diff.sum())

        r = est.loc[est["channel"] == ch].iloc[0]
        ref_total = float(r["delta_sales_total_mean"])
        ref_own = float(r["delta_own_spend"])
        ref_media = float(r["delta_total_media_spend_mean"])
        rel = lambda a, b: abs(a - b) / max(abs(b), 1e-9)

        share = (d_total - d_direct) / d_total if abs(d_total) > 1e-12 else np.nan

        rows.append({
            "cell": cell.name, "graph": graph, "channel": ch,
            "factor": factor, "burn_in_weeks": burn_in,
            # primaere Groesse
            "indirect_share": share,
            # auf den gespeicherten Posterior-Mittelwert verankerte Niveaus
            "delta_sales_direct_anchored": (1.0 - share) * ref_total,
            "delta_sales_indirect_anchored": share * ref_total,
            "delta_sales_total_saved": ref_total,
            # rohe Plug-in-Werte
            "delta_sales_direct_plugin": d_direct,
            "delta_sales_indirect_plugin": d_total - d_direct,
            "delta_sales_total_plugin": d_total,
            "ratio_total_plugin": d_total / ref_total if abs(ref_total) > 1e-9 else np.nan,
            # Struktur-Checks
            "delta_own_spend": d_own,
            "induced_downstream_spend": d_media - d_own,
            "check_rel_own_spend": rel(d_own, ref_own),
            "check_rel_media_spend": rel(d_media, ref_media),
        })

    out = pd.DataFrame(rows)
    worst = float(out[["check_rel_own_spend", "check_rel_media_spend"]].to_numpy().max())
    ok = worst <= TOL_STRUCT
    out["validated"] = ok
    out["struct_check_max"] = worst

    # Niveau-Abweichung ist kein Fehler, nur ein Vermerk: Anteile bleiben
    # nutzbar, die rohen Plug-in-Niveaus nicht.
    rmin = float(out["ratio_total_plugin"].min())
    rmax = float(out["ratio_total_plugin"].max())
    out["level_warning"] = not (1 - RATIO_WARN <= rmin and rmax <= 1 + RATIO_WARN)

    if not ok:
        return None, f"Struktur-Check verletzt (max rel. {worst:.2e})"
    return out, ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _report(name: str, graph: str, out: pd.DataFrame | None,
            reason: str) -> None:
    """Eine Zeile je Zelle: Struktur-Check und Niveau-Verhaeltnis."""
    if out is None:
        print(f"[{name}/{graph}] FEHLGESCHLAGEN: {reason}", file=sys.stderr)
        return
    rmin = out["ratio_total_plugin"].min()
    rmax = out["ratio_total_plugin"].max()
    print(f"[{name}/{graph}] Struktur-Check max "
          f"{out['struct_check_max'].iloc[0]:.2e} -> OK | "
          f"Plug-in/gespeichert {rmin:.3f}-{rmax:.3f}")
    if bool(out["level_warning"].iloc[0]):
        print("  Hinweis: Niveau-Abweichung > 15 % (Rundung + Jensen). "
              "Anteile bleiben nutzbar, Plug-in-Niveaus nicht.",
              file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path)
    ap.add_argument("--grid", type=Path)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--graph", default="oracle_hybrid",
                    choices=["pcmci_dag", "hybrid", "oracle_hybrid"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.cell:
        out, reason = run_cell(args.cell, args.graph)
        _report(args.cell.name, args.graph, out, reason)
        dest = args.out or (model_dir(args.cell, args.graph)
                            / "direct_indirect_effects.csv")
        if out is None:
            pd.DataFrame([{"cell": args.cell.name, "graph": args.graph,
                           "reason": reason}]).to_csv(
                dest.with_name(dest.stem + "_failed.csv"), index=False)
            return 1
        out.to_csv(dest, index=False)
        show = ["channel", "indirect_share", "delta_sales_direct_anchored",
                "delta_sales_indirect_anchored", "delta_sales_total_saved",
                "ratio_total_plugin"]
        print(out[show].to_string(index=False,
                                  float_format=lambda v: f"{v:,.3f}"))
        return 0

    if args.grid:
        cells = sorted(p for p in glob(str(args.grid / args.pattern))
                       if Path(p).is_dir())
        frames, failed = [], []
        for c in cells:
            try:
                r, reason = run_cell(Path(c), args.graph)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failed.append({"cell": Path(c).name, "graph": args.graph,
                               "reason": msg})
                _report(Path(c).name, args.graph, None, msg)
                continue
            _report(Path(c).name, args.graph, r, reason)
            if r is not None:
                frames.append(r)
            else:
                failed.append({"cell": Path(c).name, "graph": args.graph,
                               "reason": reason})
        dest = args.out or Path(f"direct_indirect_{args.graph}.csv")
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