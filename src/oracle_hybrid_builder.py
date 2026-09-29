from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import networkx as nx
from pcmci_plus_pipeline import MEDIA, TARGET, SYSTEM_VARS, load_ground_truth

BUILDER_NAME = "oracle_hybrid"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_oracle_hybrid(true_adj: pd.DataFrame,
                        true_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the oracle hybrid edge table.

    Columns: source, target, edge_type, source_origin, lag
    (lag is taken from true_lagged_edges for interdependencies and for
    sales edges that exist in the true graph; forced sales edges without a
    true counterpart carry no lag).
    """
    true_adj = true_adj.loc[SYSTEM_VARS, SYSTEM_VARS]
    lag_lookup = {(e["source"], e["target"]): int(e["lag"])
                  for _, e in true_edges.iterrows()}

    rows = []
    # 1. forced direct media-to-sales edges (always all six)
    for ch in MEDIA:
        rows.append({"source": ch, "target": TARGET,
                     "edge_type": "direct_sales",
                     "source_origin": "forced_prior",
                     "lag": lag_lookup.get((ch, TARGET))})
    # 2. all true channel-to-channel interdependencies
    for s in SYSTEM_VARS:
        for t in SYSTEM_VARS:
            if t == TARGET or s == TARGET or s == t:
                continue
            if true_adj.loc[s, t] == 1:
                rows.append({"source": s, "target": t,
                             "edge_type": "interdependency",
                             "source_origin": "ground_truth",
                             "lag": lag_lookup.get((s, t))})

    edges = pd.DataFrame(rows)
    # 3. de-duplicate defensively (construction cannot produce duplicates,
    #    but the guarantee is validated, not assumed)
    edges = edges.drop_duplicates(subset=["source", "target"]) \
                 .reset_index(drop=True)
    return edges


def oracle_adjacency(edges: pd.DataFrame) -> pd.DataFrame:
    """Binary directed adjacency over the seven system variables."""
    adj = pd.DataFrame(0, index=SYSTEM_VARS, columns=SYSTEM_VARS, dtype=int)
    for _, e in edges.iterrows():
        adj.loc[e["source"], e["target"]] = 1
    return adj


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_oracle_hybrid(edges: pd.DataFrame, adj: pd.DataFrame,
                           true_adj: pd.DataFrame) -> dict:
    """The six required checks; every result is exported to metadata."""
    checks: dict = {}

    # 1. all six media-to-sales edges exist
    sales_edges = {(e["source"], e["target"]) for _, e in edges.iterrows()
                   if e["target"] == TARGET}
    checks["all_media_to_sales_edges_present"] = \
        sales_edges == {(ch, TARGET) for ch in MEDIA}

    # 2. no duplicate edges
    checks["no_duplicate_edges"] = \
        not edges.duplicated(subset=["source", "target"]).any()

    # 3. graph is acyclic
    g = nx.DiGraph()
    g.add_nodes_from(SYSTEM_VARS)
    g.add_edges_from(edges[["source", "target"]].itertuples(index=False))
    checks["is_dag"] = bool(nx.is_directed_acyclic_graph(g))

    # 4. all interdependency edges originate from ground truth
    inter = edges[edges["edge_type"] == "interdependency"]
    checks["all_interdependencies_from_ground_truth"] = all(
        true_adj.loc[e["source"], e["target"]] == 1
        for _, e in inter.iterrows()) if len(inter) else True

    # 5. Sales has no outgoing edges
    checks["sales_has_no_outgoing_edges"] = \
        not (edges["source"] == TARGET).any()

    # 6. adjacency dimensions correct
    checks["adjacency_dimensions_correct"] = \
        list(adj.index) == SYSTEM_VARS and list(adj.columns) == SYSTEM_VARS

    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


# ---------------------------------------------------------------------------
# Orchestration and export
# ---------------------------------------------------------------------------

def build_oracle_for_cell(cell_dir: str | Path) -> dict:
    """
    Build, validate, and export the oracle hybrid DAG for one cell.
    Writes into <cell>/oracle_hybrid/ and returns a summary row.
    """
    cell_dir = Path(cell_dir)
    out_dir = cell_dir / "oracle_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    true_adj, true_edges = load_ground_truth(cell_dir)
    edges = build_oracle_hybrid(true_adj, true_edges)
    adj = oracle_adjacency(edges)
    validation = validate_oracle_hybrid(edges, adj,
                                        true_adj.loc[SYSTEM_VARS, SYSTEM_VARS])
    if not validation["all_passed"]:
        raise RuntimeError(f"oracle hybrid validation failed for "
                           f"{cell_dir.name}: {validation}")

    # required four output columns first; lag as auxiliary column
    edges[["source", "target", "edge_type", "source_origin", "lag"]] \
        .to_csv(out_dir / "oracle_hybrid_edges.csv", index=False)
    adj.to_csv(out_dir / "oracle_hybrid_adjacency_matrix.csv")

    # machine-readable structure — SAME schema as hybrid_dag.json so the
    # causal MMM consumes oracle and PCMCI hybrid identically
    structure = {
        "nodes": SYSTEM_VARS,
        "sales_edges": MEDIA,
        "channel_edges": [
            {"source": e["source"], "target": e["target"],
             "lag": (int(e["lag"]) if pd.notna(e["lag"]) else None),
             "p_value": None}
            for _, e in edges.iterrows()
            if e["edge_type"] == "interdependency"],
    }
    with open(out_dir / "oracle_hybrid_graph.json", "w") as f:
        json.dump(structure, f, indent=2)

    n_inter = int((edges["edge_type"] == "interdependency").sum())
    n_sales_in_truth = int(sum(true_adj.loc[ch, TARGET] == 1 for ch in MEDIA))
    metadata = {
        "builder": BUILDER_NAME,
        "n_edges_total": int(len(edges)),
        "n_forced_sales_edges": len(MEDIA),
        "n_forced_sales_edges_present_in_ground_truth": n_sales_in_truth,
        "n_ground_truth_interdependencies": n_inter,
        "is_dag": validation["is_dag"],
        "validation": validation,
    }
    with open(out_dir / "oracle_hybrid_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {"cell": cell_dir.name, "n_edges_total": len(edges),
            "n_interdependencies": n_inter,
            "n_forced_sales_in_truth": n_sales_in_truth,
            "validation_passed": validation["all_passed"]}