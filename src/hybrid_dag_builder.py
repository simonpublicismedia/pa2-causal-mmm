from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import networkx as nx
from pcmci_plus_pipeline import (
    MEDIA, TARGET, SYSTEM_VARS, CONTROLS,
    build_adjacency, load_ground_truth, _shd, _node_color,
)


@dataclass
class HybridConfig:
    setting: str = "controlled"      # which PCMCI+ result to consume
    min_lag: int = 1                 # drop discovered channel edges below this lag
    keep_reverse_sales_edges: bool = False   # Sales -> channel edges (default: drop)


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------

def load_discovered_edges(cell_dir: str | Path, setting: str) -> pd.DataFrame:
    """Load the PCMCI+ edge table of one cell/setting."""
    path = Path(cell_dir) / "pcmci_plus" / setting / "discovered_edges.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing PCMCI+ output: {path} — run run_pcmci_plus.py first")
    return pd.read_csv(path)


def _aggregate_lags(directed: pd.DataFrame) -> pd.DataFrame:
    """One row per (source, target): all lags, min p, primary (min-p) lag."""
    rows = []
    for (s, t), grp in directed.groupby(["source", "target"]):
        grp = grp.sort_values("p_value")
        rows.append({"source": s, "target": t,
                     "lags": sorted(int(l) for l in grp["lag"].unique()),
                     "primary_lag": int(grp.iloc[0]["lag"]),
                     "p_value": float(grp["p_value"].min())})
    return pd.DataFrame(rows)


def build_hybrid_dag(discovered: pd.DataFrame,
                     cfg: HybridConfig) -> tuple[pd.DataFrame, dict]:
    """
    Construct the hybrid edge table from PCMCI+ output.

    Returns (edges, report). Edge columns:
      source, target, origin ("benchmark_prior" | "discovered"),
      lags (json list or None), primary_lag, p_value,
      confirmed_by_discovery (sales edges only).
    """
    report: dict = {"n_discovered_links_total": int(len(discovered)),
                    "dropped_unoriented": 0, "dropped_reverse_sales": 0,
                    "dropped_below_min_lag": 0, "dropped_control_edges": 0,
                    "dropped_autoregressive_self_links": 0,
                    "cycle_removed_edges": []}

    d = discovered.copy()
    # rule 1: oriented links only
    report["dropped_unoriented"] = int((~d["oriented"]).sum())
    d = d[d["oriented"]]
    # controls never enter the structural graph (controlled setting input)
    ctrl_mask = d["source"].isin(CONTROLS) | d["target"].isin(CONTROLS)
    report["dropped_control_edges"] = int(ctrl_mask.sum())
    d = d[~ctrl_mask]

    # split by target
    to_sales = d[d["target"] == TARGET]
    from_sales = d[d["source"] == TARGET]
    channel = d[(d["source"] != TARGET) & (d["target"] != TARGET)]

    # rule 2: reverse sales edges
    if not cfg.keep_reverse_sales_edges:
        report["dropped_reverse_sales"] = int(len(from_sales))
        from_sales = from_sales.iloc[0:0]

    # autoregressive self-links (X -> X): legitimate carryover findings of
    # PCMCI+, but not cross-channel structure — excluded and counted, never
    # treated as "cycles"
    self_mask = channel["source"] == channel["target"]
    report["dropped_autoregressive_self_links"] = int(self_mask.sum())
    channel = channel[~self_mask]

    # rule 3: minimum lag for channel edges
    below = channel["lag"] < cfg.min_lag
    report["dropped_below_min_lag"] = int(below.sum())
    channel = channel[~below]

    channel_agg = _aggregate_lags(channel) if len(channel) else \
        pd.DataFrame(columns=["source", "target", "lags", "primary_lag",
                              "p_value"])

    # rule 5: enforce acyclicity on the aggregated channel graph
    g = nx.DiGraph()
    g.add_nodes_from(MEDIA)
    for _, e in channel_agg.iterrows():
        g.add_edge(e["source"], e["target"], p=e["p_value"])
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        worst = max(cycle, key=lambda uv: g.edges[uv[0], uv[1]]["p"])
        report["cycle_removed_edges"].append(
            {"source": worst[0], "target": worst[1],
             "p_value": g.edges[worst[0], worst[1]]["p"]})
        g.remove_edge(*worst)
        channel_agg = channel_agg[~((channel_agg["source"] == worst[0])
                                    & (channel_agg["target"] == worst[1]))]

    # discovery confirmation of sales edges
    sales_agg = _aggregate_lags(to_sales) if len(to_sales) else \
        pd.DataFrame(columns=["source", "target", "lags", "primary_lag",
                              "p_value"])
    sales_info = {r["source"]: r for _, r in sales_agg.iterrows()}

    rows = []
    # base edges: every media channel -> Sales, always present
    for ch in MEDIA:
        info = sales_info.get(ch)
        has_info = info is not None
        rows.append({"source": ch, "target": TARGET,
                     "origin": "benchmark_prior",
                     "lags": json.dumps(info["lags"]) if has_info else None,
                     "primary_lag": (int(info["primary_lag"])
                                     if has_info else None),
                     "p_value": float(info["p_value"]) if has_info else None,
                     "confirmed_by_discovery": has_info})
    # discovered channel-to-channel edges
    for _, e in channel_agg.iterrows():
        rows.append({"source": e["source"], "target": e["target"],
                     "origin": "discovered",
                     "lags": json.dumps(e["lags"]),
                     "primary_lag": int(e["primary_lag"]),
                     "p_value": float(e["p_value"]),
                     "confirmed_by_discovery": True})
    if cfg.keep_reverse_sales_edges and len(from_sales):
        for _, e in _aggregate_lags(from_sales).iterrows():
            rows.append({"source": e["source"], "target": e["target"],
                         "origin": "discovered",
                         "lags": json.dumps(e["lags"]),
                         "primary_lag": int(e["primary_lag"]),
                         "p_value": float(e["p_value"]),
                         "confirmed_by_discovery": True})

    edges = pd.DataFrame(rows)
    report.update({
        "n_prior_sales_edges": len(MEDIA),
        "n_sales_edges_confirmed_by_discovery":
            int(edges[edges["origin"] == "benchmark_prior"]
                ["confirmed_by_discovery"].sum()),
        "n_discovered_channel_edges_added":
            int((edges["origin"] == "discovered").sum()),
        "is_dag": True,
    })
    return edges, report


def hybrid_adjacency(edges: pd.DataFrame) -> pd.DataFrame:
    """Aggregated directed adjacency of the hybrid graph (system nodes)."""
    adj = pd.DataFrame(0, index=SYSTEM_VARS, columns=SYSTEM_VARS, dtype=int)
    for _, e in edges.iterrows():
        adj.loc[e["source"], e["target"]] = 1
    return adj


# ---------------------------------------------------------------------------
# Optional post-hoc evaluation against ground truth (never used in building)
# ---------------------------------------------------------------------------

def evaluate_hybrid(edges: pd.DataFrame, cell_dir: str | Path) -> dict:
    """
    Aggregated-graph metrics of the hybrid DAG vs. ground truth, overall and
    for the channel-edge subgroup. Note: because all media->Sales edges are
    present by construction, the sales-edge recall is 1 by design; the
    informative part is the channel subgroup and overall precision.
    """
    true_adj, _ = load_ground_truth(cell_dir)
    true_adj = true_adj.loc[SYSTEM_VARS, SYSTEM_VARS]
    adj = hybrid_adjacency(edges)

    def _prf(t_set: set, d_set: set) -> dict:
        tp = len(t_set & d_set)
        p = tp / len(d_set) if d_set else 0.0
        r = tp / len(t_set) if t_set else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(f1, 4)}

    true_set = {(s, t) for s in SYSTEM_VARS for t in SYSTEM_VARS
                if true_adj.loc[s, t]}
    disc_set = {(s, t) for s in SYSTEM_VARS for t in SYSTEM_VARS
                if adj.loc[s, t]}
    overall = _prf(true_set, disc_set)
    overall["shd"] = _shd(true_adj, adj, SYSTEM_VARS)
    ch_true = {e for e in true_set if e[1] != TARGET}
    ch_disc = {e for e in disc_set if e[1] != TARGET}
    channel = _prf(ch_true, ch_disc)
    return {"overall": overall, "channel_edges": channel,
            "fp_edges": sorted(f"{s}->{t}" for s, t in disc_set - true_set),
            "fn_edges": sorted(f"{s}->{t}" for s, t in true_set - disc_set)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_hybrid_dag(edges: pd.DataFrame, out_path: Path, title: str) -> None:
    """Prior edges solid, discovered channel edges dashed with lag labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = nx.DiGraph()
    g.add_nodes_from(SYSTEM_VARS)
    for _, e in edges.iterrows():
        g.add_edge(e["source"], e["target"], origin=e["origin"],
                   label=("" if e["origin"] == "benchmark_prior"
                          else f"lag {int(e['primary_lag'])}"))
    pos = nx.shell_layout(g, nlist=[[TARGET], MEDIA])
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2600,
                           node_color=[_node_color(n) for n in g.nodes])
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_color="white")
    prior = [(u, v) for u, v, d in g.edges(data=True)
             if d["origin"] == "benchmark_prior"]
    disc = [(u, v) for u, v, d in g.edges(data=True)
            if d["origin"] == "discovered"]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=prior, arrows=True,
                           arrowsize=14, edge_color="#90A4AE",
                           min_source_margin=28, min_target_margin=28)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=disc, arrows=True,
                           arrowsize=16, style="dashed", edge_color="#C62828",
                           connectionstyle="arc3,rad=0.15",
                           min_source_margin=28, min_target_margin=28)
    nx.draw_networkx_edge_labels(
        g, pos, ax=ax, font_size=7,
        edge_labels={(u, v): g.edges[u, v]["label"] for u, v in disc})
    ax.set_title(title + "\n(grey solid = benchmark prior, "
                         "red dashed = discovered)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration and export
# ---------------------------------------------------------------------------

def build_hybrid_for_cell(cell_dir: str | Path, cfg: HybridConfig) -> dict:
    """
    Build, evaluate, and export the hybrid DAG for one cell/setting.
    Writes into <cell>/hybrid_dag/<setting>/ and returns a summary row.
    """
    cell_dir = Path(cell_dir)
    out_dir = cell_dir / "hybrid_dag" / cfg.setting
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = load_discovered_edges(cell_dir, cfg.setting)
    edges, report = build_hybrid_dag(discovered, cfg)
    edges.to_csv(out_dir / "hybrid_dag_edges.csv", index=False)
    hybrid_adjacency(edges).to_csv(out_dir / "hybrid_adjacency_matrix.csv")

    evaluation = evaluate_hybrid(edges, cell_dir)
    build_report = {"config": asdict(cfg), **report,
                    "evaluation_vs_ground_truth": evaluation}
    with open(out_dir / "hybrid_build_report.json", "w") as f:
        json.dump(build_report, f, indent=2)

    # machine-readable structure for the causal MMM
    structure = {
        "nodes": SYSTEM_VARS,
        "sales_edges": [e["source"] for _, e in edges.iterrows()
                        if e["origin"] == "benchmark_prior"],
        "channel_edges": [
            {"source": e["source"], "target": e["target"],
             "lag": int(e["primary_lag"]), "p_value": e["p_value"]}
            for _, e in edges.iterrows() if e["origin"] == "discovered"],
    }
    with open(out_dir / "hybrid_dag.json", "w") as f:
        json.dump(structure, f, indent=2)

    plot_error = ""
    try:
        plot_hybrid_dag(edges, out_dir / "hybrid_dag_plot.png",
                        f"Hybrid DAG — {cell_dir.name} ({cfg.setting})")
    except Exception as exc:                     # plotting is best-effort
        plot_error = repr(exc)                   # CSV/JSON exports stay valid

    ev = evaluation["overall"]
    row = {"cell": cell_dir.name, "setting": cfg.setting,
           "n_channel_edges_added": report["n_discovered_channel_edges_added"],
           "n_sales_edges_confirmed":
               report["n_sales_edges_confirmed_by_discovery"],
           "n_cycle_removed": len(report["cycle_removed_edges"]),
           "precision": ev["precision"], "recall": ev["recall"],
           "f1": ev["f1"], "shd": ev["shd"],
           "channel_f1": evaluation["channel_edges"]["f1"],
           "plot_error": plot_error}
    return row