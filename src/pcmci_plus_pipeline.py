from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import tigramite
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite import data_processing as pp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEDIA: list[str] = ["TV", "Video", "Display", "Social", "Search", "Affiliate"]
TARGET: str = "Sales"
SYSTEM_VARS: list[str] = MEDIA + [TARGET]          # nodes of the true graph
CONTROLS: list[str] = ["Seasonality", "Trend", "Event_Effect"]

SETTING_FILES: dict[str, str] = {
    "uncontrolled": "observed_uncontrolled.parquet",
    "controlled": "observed_controlled.parquet",
}
SETTING_VARS: dict[str, list[str]] = {
    "uncontrolled": SYSTEM_VARS,
    "controlled": SYSTEM_VARS + CONTROLS,
}

DEFAULT_TAU_MAX: int = 2
DEFAULT_PC_ALPHA: float = 0.075

# minimum-length heuristics for diagnostics (soft/hard thresholds)
MIN_T_HARD_FACTOR: int = 3     # T <= 3*(tau_max+1)  -> abort
MIN_T_SOFT_FACTOR: int = 10    # T <  10*(tau_max+1) -> warn


DEFAULT_COND_IND_TEST: str = "RobustParCorr"


def make_cond_ind_test(name: str):
    """Factory for the conditional independence test used by PCMCI+."""
    if name == "ParCorr":
        return ParCorr()
    if name == "RobustParCorr":
        return RobustParCorr()
    raise ValueError(f"Unknown conditional independence test: {name}")


@dataclass
class DiscoveryConfig:
    setting: str = "controlled"            # "controlled" | "uncontrolled"
    tau_max: int = DEFAULT_TAU_MAX
    pc_alpha: float = DEFAULT_PC_ALPHA
    cond_ind_test: str = DEFAULT_COND_IND_TEST   # "ParCorr" | "RobustParCorr"


# ---------------------------------------------------------------------------
# Data loading and diagnostics
# ---------------------------------------------------------------------------

def load_cell_dataset(cell_dir: str | Path, setting: str) -> pd.DataFrame:
    """Load the observational dataset of one scenario cell for a setting."""
    if setting not in SETTING_FILES:
        raise ValueError(f"unknown setting '{setting}'")
    path = Path(cell_dir) / SETTING_FILES[setting]
    if not path.exists():
        raise FileNotFoundError(f"input file missing: {path}")
    return pd.read_parquet(path)


def run_diagnostics(df: pd.DataFrame, cell_dir: str | Path, setting: str,
                    tau_max: int) -> dict:
    """
    Pre-run checks. Hard failures raise; soft issues are recorded as
    warnings. Returns the diagnostics report (also meant for JSON export).
    """
    report: dict = {"cell": str(cell_dir), "setting": setting,
                    "tau_max": tau_max, "checks": {}, "warnings": [],
                    "passed": True}

    # 1. input file exists — implicitly verified by load_cell_dataset;
    #    record it for completeness.
    report["checks"]["input_file_exists"] = True

    # 2. required variables exist
    required = SETTING_VARS[setting]
    missing = [v for v in required if v not in df.columns]
    report["checks"]["required_variables_present"] = not missing
    if missing:
        report["passed"] = False
        raise ValueError(f"missing variables in {setting} dataset: {missing}")

    # 3. no missing values
    na_counts = df[required].isna().sum()
    n_na = int(na_counts.sum())
    report["checks"]["no_missing_values"] = n_na == 0
    if n_na > 0:
        report["passed"] = False
        raise ValueError(f"missing values found: "
                         f"{na_counts[na_counts > 0].to_dict()}")

    # 4. series length sufficient for tau_max
    T = len(df)
    report["checks"]["n_timesteps"] = T
    if T <= MIN_T_HARD_FACTOR * (tau_max + 1):
        report["passed"] = False
        raise ValueError(f"time series too short (T={T}) for tau_max={tau_max}")
    if T < MIN_T_SOFT_FACTOR * (tau_max + 1):
        report["warnings"].append(
            f"T={T} is small for tau_max={tau_max}; results may be unstable")
    report["checks"]["length_sufficient"] = True

    # 5. no nearly constant variables
    near_constant = []
    for v in required:
        s = df[v].astype(float)
        if s.nunique() <= 1 or s.std() < 1e-10:
            near_constant.append(v)
        elif s.std() / max(abs(s.mean()), 1e-12) < 0.01:
            report["warnings"].append(f"variable '{v}' has very low relative "
                                      f"variation (CV < 1%)")
    report["checks"]["no_constant_variables"] = not near_constant
    if near_constant:
        report["passed"] = False
        raise ValueError(f"nearly constant variables: {near_constant}")

    # 6. 'Week' must not be treated as a causal variable
    report["checks"]["week_excluded_from_analysis"] = "Week" not in required
    if "Week" in required:
        report["passed"] = False
        raise ValueError("'Week' must not be part of the analysis variables")

    return report


def prepare_arrays(df: pd.DataFrame, setting: str) -> tuple[np.ndarray, list[str]]:
    """
    Select analysis variables (drops 'Week' and everything else) and
    z-standardize each series. ParCorr is scale-invariant, so
    standardization does not change test decisions; it only improves
    numerical conditioning.
    """
    var_names = SETTING_VARS[setting]
    data = df[var_names].to_numpy(float)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data, list(var_names)


# ---------------------------------------------------------------------------
# PCMCI+ execution
# ---------------------------------------------------------------------------

def run_pcmci_plus(data: np.ndarray, var_names: list[str],
                   cfg: DiscoveryConfig) -> dict:
    """
    Run PCMCI+ with a linear partial-correlation test.

    Ground truth is deliberately NOT an argument of this function: discovery
    operates on observed data only.
    """
    dataframe = pp.DataFrame(data, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=make_cond_ind_test(cfg.cond_ind_test),
                  verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=cfg.tau_max,
                                  pc_alpha=cfg.pc_alpha)
    return {"graph": results["graph"], "p_matrix": results["p_matrix"],
            "val_matrix": results["val_matrix"], "var_names": var_names}


# ---------------------------------------------------------------------------
# Edge extraction and adjacency construction
# ---------------------------------------------------------------------------

def extract_edges(results: dict) -> pd.DataFrame:
    """
    Convert the tigramite graph array into a tidy edge table.

    Semantics of graph[i, j, tau]:
      '-->' : var i at t-tau  causes  var j at t   (directed)
      '<--' : reverse direction (only at tau=0)
      'o-o' : contemporaneous, unoriented
      'x-x' : contemporaneous, conflicting orientation
    Undirected/conflicting links are kept in the table (oriented=False) but
    are not part of the directed adjacency used for evaluation.
    """
    graph = results["graph"]
    p = results["p_matrix"]
    val = results["val_matrix"]
    names = results["var_names"]
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            for tau in range(graph.shape[2]):
                link = graph[i, j, tau]
                if link == "" or link == "<--":
                    continue                     # '<--' mirrored as '-->'
                if link == "-->":
                    rows.append({"source": names[i], "target": names[j],
                                 "lag": tau, "link_type": link,
                                 "oriented": True,
                                 "p_value": float(p[i, j, tau]),
                                 "test_stat": float(val[i, j, tau])})
                elif link in ("o-o", "x-x") and i < j and tau == 0:
                    rows.append({"source": names[i], "target": names[j],
                                 "lag": 0, "link_type": link,
                                 "oriented": False,
                                 "p_value": float(p[i, j, tau]),
                                 "test_stat": float(val[i, j, tau])})
    return pd.DataFrame(rows, columns=["source", "target", "lag", "link_type",
                                       "oriented", "p_value", "test_stat"])


def build_adjacency(edges: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """
    Aggregated directed adjacency over all lags, restricted to `nodes`
    (i.e., the system variables; controls are dropped here — see module
    docstring for the rationale).
    """
    adj = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    directed = edges[edges["oriented"]]
    for _, e in directed.iterrows():
        if e["source"] in nodes and e["target"] in nodes \
                and e["source"] != e["target"]:
            adj.loc[e["source"], e["target"]] = 1
    return adj


# ---------------------------------------------------------------------------
# Ground truth loading and graph-recovery evaluation
# ---------------------------------------------------------------------------

def load_ground_truth(cell_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load true aggregated adjacency and true lagged edges of a cell."""
    cell_dir = Path(cell_dir)
    true_adj = pd.read_csv(cell_dir / "true_adjacency_matrix.csv", index_col=0)
    true_edges = pd.read_csv(cell_dir / "true_lagged_edges.csv")
    return true_adj, true_edges


def _shd(true_adj: pd.DataFrame, disc_adj: pd.DataFrame,
         nodes: list[str]) -> int:
    """
    Structural Hamming Distance on the aggregated directed graph:
    per unordered node pair, an edge insertion or deletion costs 1 each and
    a pure reversal costs 1 (not 2).
    """
    shd = 0
    for a_i, a in enumerate(nodes):
        for b in nodes[a_i + 1:]:
            t = {(a, b)} if true_adj.loc[a, b] else set()
            t |= {(b, a)} if true_adj.loc[b, a] else set()
            d = {(a, b)} if disc_adj.loc[a, b] else set()
            d |= {(b, a)} if disc_adj.loc[b, a] else set()
            if t == d:
                continue
            if len(t) == 1 and len(d) == 1:      # pure reversal
                shd += 1
            else:
                shd += len(t.symmetric_difference(d))
    return shd


def evaluate_graph_recovery(edges: pd.DataFrame, true_adj: pd.DataFrame,
                            true_edges: pd.DataFrame,
                            nodes: list[str] = SYSTEM_VARS) -> tuple[dict, pd.DataFrame]:
    """
    Compare the discovered directed graph against ground truth on the
    system variables only. Returns (metrics dict, per-edge detail table).

    Metrics: precision, recall, F1 on aggregated directed edges;
    lag accuracy among recovered true edges; SHD; FP/FN edge lists.
    """
    disc_adj = build_adjacency(edges, nodes)
    true_adj = true_adj.loc[nodes, nodes]

    true_set = {(s, t) for s in nodes for t in nodes if true_adj.loc[s, t]}
    disc_set = {(s, t) for s in nodes for t in nodes if disc_adj.loc[s, t]}
    tp = true_set & disc_set
    fp = disc_set - true_set
    fn = true_set - disc_set

    precision = len(tp) / len(disc_set) if disc_set else 0.0
    recall = len(tp) / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)

    # lag accuracy: of the recovered true edges, how many include the true lag
    directed = edges[edges["oriented"]]
    lags_by_edge: dict[tuple[str, str], set[int]] = {}
    for _, e in directed.iterrows():
        lags_by_edge.setdefault((e["source"], e["target"]), set()).add(int(e["lag"]))

    detail_rows = []
    n_lag_correct = 0
    for _, te in true_edges.iterrows():
        key = (te["source"], te["target"])
        recovered = key in tp
        disc_lags = sorted(lags_by_edge.get(key, set()))
        lag_ok = recovered and int(te["lag"]) in disc_lags
        n_lag_correct += int(lag_ok)
        detail_rows.append({"source": key[0], "target": key[1],
                            "true_lag": int(te["lag"]), "status": "true_edge",
                            "recovered": recovered,
                            "discovered_lags": json.dumps(disc_lags),
                            "lag_correct": lag_ok})
    for (s, t) in sorted(fp):
        detail_rows.append({"source": s, "target": t, "true_lag": None,
                            "status": "false_positive", "recovered": True,
                            "discovered_lags":
                                json.dumps(sorted(lags_by_edge.get((s, t), set()))),
                            "lag_correct": None})
    detail = pd.DataFrame(detail_rows)

    lag_accuracy = n_lag_correct / len(tp) if tp else 0.0
    n_unoriented = int((~edges["oriented"]).sum()) if len(edges) else 0

    metrics = {
        "n_true_edges": len(true_set), "n_discovered_edges": len(disc_set),
        "true_positives": len(tp), "false_positives": len(fp),
        "false_negatives": len(fn),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "lag_accuracy": round(lag_accuracy, 4),
        "shd": _shd(true_adj, disc_adj, nodes),
        "n_unoriented_contemporaneous": n_unoriented,
        "fp_edges": sorted([f"{s}->{t}" for s, t in fp]),
        "fn_edges": sorted([f"{s}->{t}" for s, t in fn]),
    }
    return metrics, detail


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _node_color(name: str) -> str:
    if name == TARGET:
        return "#2E7D32"          # sales: green
    if name in CONTROLS:
        return "#B0BEC5"          # controls: grey
    return "#3F72AF"              # media: blue


def plot_aggregated_graph(edges: pd.DataFrame, var_names: list[str],
                          out_path: Path, title: str) -> None:
    """Aggregated directed graph; edge labels show min lag and p-value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = nx.DiGraph()
    g.add_nodes_from(var_names)
    directed = edges[edges["oriented"]]
    best = directed.sort_values("p_value").drop_duplicates(["source", "target"])
    for _, e in best.iterrows():
        g.add_edge(e["source"], e["target"],
                   label=f"lag {int(e['lag'])}\np={e['p_value']:.3f}")

    pos = nx.circular_layout(g)
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2600,
                           node_color=[_node_color(n) for n in g.nodes])
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_color="white")
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowsize=16,
                           connectionstyle="arc3,rad=0.12",
                           min_source_margin=28, min_target_margin=28)
    nx.draw_networkx_edge_labels(g, pos, ax=ax, font_size=7,
                                 edge_labels=nx.get_edge_attributes(g, "label"))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_lagged_graph(edges: pd.DataFrame, var_names: list[str], tau_max: int,
                      out_path: Path, title: str) -> None:
    """Time-series graph: variables as rows, lags as columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.2 * (tau_max + 1) + 2,
                                    0.75 * len(var_names) + 1.5))
    ypos = {v: len(var_names) - k for k, v in enumerate(var_names)}
    for tau in range(tau_max + 1):
        x = tau_max - tau
        for v in var_names:
            ax.scatter(x, ypos[v], s=340, color=_node_color(v), zorder=3)
        ax.text(x, len(var_names) + 0.7, f"t-{tau}" if tau else "t",
                ha="center", fontsize=9)
    for v in var_names:
        ax.text(-0.7, ypos[v], v, ha="right", va="center", fontsize=9)

    directed = edges[edges["oriented"]]
    for _, e in directed.iterrows():
        x0, x1 = tau_max - int(e["lag"]), tau_max
        y0, y1 = ypos[e["source"]], ypos[e["target"]]
        ax.annotate("", xy=(x1 - 0.12, y1), xytext=(x0 + 0.12, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.1,
                                    color="#37474F", alpha=0.85,
                                    connectionstyle="arc3,rad=0.08"))
    ax.set_xlim(-2.2, tau_max + 0.6)
    ax.set_ylim(0.2, len(var_names) + 1.4)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration and export (file I/O kept separate from the logic above)
# ---------------------------------------------------------------------------

def run_discovery_on_cell(cell_dir: str | Path, cfg: DiscoveryConfig) -> dict:
    """
    Full per-cell, per-setting pipeline:
    diagnostics -> PCMCI+ -> edges/adjacency -> evaluation -> exports.
    Writes into <cell_dir>/pcmci_plus/<setting>/ and returns a summary row.
    """
    cell_dir = Path(cell_dir)
    out_dir = cell_dir / "pcmci_plus" / cfg.setting
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_cell_dataset(cell_dir, cfg.setting)
    diagnostics = run_diagnostics(df, cell_dir, cfg.setting, cfg.tau_max)
    with open(out_dir / "pcmci_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    data, var_names = prepare_arrays(df, cfg.setting)
    results = run_pcmci_plus(data, var_names, cfg)

    edges = extract_edges(results)
    edges.to_csv(out_dir / "discovered_edges.csv", index=False)

    adjacency = build_adjacency(edges, SYSTEM_VARS)
    adjacency.to_csv(out_dir / "adjacency_matrix.csv")

    # evaluation happens strictly AFTER discovery, using stored results only
    true_adj, true_edges = load_ground_truth(cell_dir)
    metrics, detail = evaluate_graph_recovery(edges, true_adj, true_edges)
    metrics["settings"] = {**asdict(cfg),
                           "tigramite_version":
                               getattr(tigramite, "__version__", "unknown")}
    with open(out_dir / "graph_recovery_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    detail.to_csv(out_dir / "edge_recovery_detail.csv", index=False)

    plot_error = ""
    try:
        plot_aggregated_graph(edges, var_names, out_dir / "graph_plot.png",
                              f"PCMCI+ aggregated graph — {cfg.setting}")
        plot_lagged_graph(edges, var_names, cfg.tau_max,
                          out_dir / "lagged_graph_plot.png",
                          f"PCMCI+ lagged graph — {cfg.setting}")
    except Exception as exc:                     # plotting is best-effort
        plot_error = repr(exc)                   # CSV/JSON exports stay valid

    summary = {"cell": cell_dir.name, "setting": cfg.setting,
               **{k: metrics[k] for k in
                  ["precision", "recall", "f1", "lag_accuracy", "shd",
                   "true_positives", "false_positives", "false_negatives",
                   "n_unoriented_contemporaneous"]},
               "plot_error": plot_error}
    return summary