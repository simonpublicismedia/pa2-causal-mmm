from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from benchmark_mmm import (
    build_lag_cube, adstock_from_cube, hill, _hdi, _az_summary,
    _group_dataset, L_MAX, HDI_PROB,
)

MEDIA_VARS = ["TV", "Video", "Display", "Social", "Search", "Affiliate"]
TARGET_VAR = "Sales"
CONTROL_VARS = ["Seasonality", "Trend", "Event_Effect"]
SYSTEM_VARS = MEDIA_VARS + [TARGET_VAR]

GRAPH_TYPES = ["pcmci_dag", "hybrid", "oracle_hybrid"]
INTERVENTION_FACTOR = 1.10
BURN_IN_WEEKS = 5
DEFAULT_CHANNEL_LAG = 1

PRIORS_SPEC = {
    "intercept_sales": "Normal(0.5, 0.5)  [sales/max scale]",
    "beta_media_sales": "HalfNormal(0.5) per Sales parent",
    "beta_control_sales": "Normal(0, 0.5) per control",
    "adstock_theta": "Beta(2, 2) per Sales media parent",
    "hill_alpha": "Gamma(3, 2) per Sales media parent",
    "hill_k": "Beta(2, 2) per Sales media parent",
    "sigma_sales": "HalfNormal(0.5)",
    "intercept_node": "Normal(0.5, 0.5) per endogenous node [spend/max scale]",
    "delta_channel": "HalfNormal(0.5) per channel-to-channel edge",
    "beta_control_node": "Normal(0, 0.5) per endogenous node and control",
    "sigma_node": "HalfNormal(0.5) per endogenous node",
}


# ===========================================================================
# 1. Graph specification and loading
# ===========================================================================

@dataclass
class GraphSpec:
    graph_type: str
    sales_parents: list[str]                       # media parents of Sales
    channel_edges: list[dict]                      # {source, target, lag}
    source_file: str | None = None
    cycle_removed_edges: list = field(default_factory=list)

    @property
    def endogenous_nodes(self) -> list[str]:
        # topological order among media nodes
        g = nx.DiGraph()
        g.add_nodes_from(MEDIA_VARS)
        g.add_edges_from((e["source"], e["target"]) for e in self.channel_edges)
        order = list(nx.topological_sort(g))
        targets = {e["target"] for e in self.channel_edges}
        return [n for n in order if n in targets]

    def parents_of(self, node: str) -> list[dict]:
        return [e for e in self.channel_edges if e["target"] == node]

    def all_edges(self) -> list[tuple[str, str]]:
        return ([(e["source"], e["target"]) for e in self.channel_edges]
                + [(p, TARGET_VAR) for p in self.sales_parents])


def _resolve_cycles(channel_edges: list[dict]) -> tuple[list[dict], list]:
    """Drop the highest-p edge of each cycle until the graph is acyclic."""
    g = nx.DiGraph()
    g.add_nodes_from(MEDIA_VARS)
    for e in channel_edges:
        g.add_edge(e["source"], e["target"], p=e.get("p_value", 0.0))
    removed = []
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        worst = max(cycle, key=lambda uv: g.edges[uv[0], uv[1]]["p"])
        removed.append({"source": worst[0], "target": worst[1],
                        "p_value": g.edges[worst[0], worst[1]]["p"]})
        g.remove_edge(*worst)
    kept = {(u, v) for u, v in g.edges}
    return ([e for e in channel_edges
             if (e["source"], e["target"]) in kept], removed)


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _norm_lag(value, default: int) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return max(int(value), DEFAULT_CHANNEL_LAG)


def load_graph(cell_dir: Path, graph_type: str,
               pcmci_setting: str = "controlled") -> GraphSpec:
    """
    Build the GraphSpec for a graph type. Paths are robust to the small
    naming differences between the task spec and the actual pipeline files.
    """
    if graph_type == "pcmci_dag":
        path = _first_existing([
            cell_dir / "pcmci_plus" / pcmci_setting / "discovered_edges.csv",
            cell_dir / "pcmci_plus" / pcmci_setting / "estimated_lagged_edges.csv",
        ])
        if path is None:
            raise FileNotFoundError(f"no PCMCI+ edge file in "
                                    f"{cell_dir / 'pcmci_plus' / pcmci_setting}")
        d = pd.read_csv(path)
        d = d[d.get("oriented", True) == True]                     # noqa: E712
        d = d[d["source"].isin(SYSTEM_VARS) & d["target"].isin(SYSTEM_VARS)]
        d = d[d["source"] != d["target"]]
        d = d[d["source"] != TARGET_VAR]                 # no Sales -> X
        sales_parents = sorted(d[d["target"] == TARGET_VAR]["source"].unique())
        ch = d[(d["target"] != TARGET_VAR)]
        ch = ch[ch["lag"] >= DEFAULT_CHANNEL_LAG]        # drop contemporaneous
        # one edge per pair: smallest p-value determines the lag
        ch = ch.sort_values("p_value").drop_duplicates(["source", "target"])
        channel_edges = [{"source": r["source"], "target": r["target"],
                          "lag": _norm_lag(r["lag"], DEFAULT_CHANNEL_LAG),
                          "p_value": float(r["p_value"])}
                         for _, r in ch.iterrows()]
        # the raw discovered graph may contain cycles (e.g. A->B and B->A);
        # resolve by dropping the weakest-evidence edge per cycle, exactly
        # like the hybrid builder, and record the removals
        channel_edges, removed = _resolve_cycles(channel_edges)
        return GraphSpec("pcmci_dag", sales_parents=list(sales_parents),
                         channel_edges=channel_edges, source_file=str(path),
                         cycle_removed_edges=removed)

    if graph_type == "hybrid":
        path = _first_existing([
            cell_dir / "hybrid_dag" / pcmci_setting / "hybrid_dag_edges.csv",
            cell_dir / "hybrid_dag" / "hybrid_dag_edges.csv",
            cell_dir / "hybrid_dag" / "hybrid_edges.csv",
        ])
        if path is None:
            raise FileNotFoundError(f"no hybrid DAG edge file under "
                                    f"{cell_dir / 'hybrid_dag'}")
        d = pd.read_csv(path)
        channel_edges = [
            {"source": r["source"], "target": r["target"],
             "lag": _norm_lag(r.get("primary_lag", r.get("lag")),
                              DEFAULT_CHANNEL_LAG)}
            for _, r in d[d["origin"] == "discovered"].iterrows()
            if r["target"] != TARGET_VAR]
        return GraphSpec("hybrid", sales_parents=list(MEDIA_VARS),
                         channel_edges=channel_edges, source_file=str(path))

    if graph_type == "oracle_hybrid":
        path = _first_existing([
            cell_dir / "oracle_hybrid" / "oracle_hybrid_edges.csv"])
        if path is None:
            raise FileNotFoundError(f"no oracle hybrid edge file under "
                                    f"{cell_dir / 'oracle_hybrid'}")
        d = pd.read_csv(path)
        channel_edges = [
            {"source": r["source"], "target": r["target"],
             "lag": _norm_lag(r.get("lag"), DEFAULT_CHANNEL_LAG)}
            for _, r in d[d["edge_type"] == "interdependency"].iterrows()]
        return GraphSpec("oracle_hybrid", sales_parents=list(MEDIA_VARS),
                         channel_edges=channel_edges, source_file=str(path))

    raise ValueError(f"unknown graph_type '{graph_type}' "
                     f"(valid: {GRAPH_TYPES})")


def validate_graph(spec: GraphSpec, df: pd.DataFrame,
                   cell_dir: Path) -> dict:
    """The ten pre-fit checks; hard failures raise after recording."""
    checks: dict = {"graph_type": spec.graph_type,
                    "graph_source_file": spec.source_file}

    checks["data_file_exists"] = True                      # loaded upstream
    checks["graph_file_exists"] = spec.source_file is not None
    g = nx.DiGraph()
    g.add_nodes_from(SYSTEM_VARS)
    g.add_edges_from(spec.all_edges())
    checks["graph_is_acyclic"] = bool(nx.is_directed_acyclic_graph(g))
    nodes_in_graph = {n for e in spec.all_edges() for n in e}
    checks["all_graph_nodes_in_dataset"] = all(n in df.columns
                                               for n in nodes_in_graph)
    checks["sales_in_dataset"] = TARGET_VAR in df.columns
    checks["sales_has_no_outgoing_edges"] = all(
        s != TARGET_VAR for s, _ in spec.all_edges())
    checks["no_duplicate_edges"] = (len(spec.all_edges())
                                    == len(set(spec.all_edges())))
    used = df[[c for c in SYSTEM_VARS + CONTROL_VARS if c in df.columns]]
    checks["no_missing_values"] = not used.isna().any().any()
    checks["media_non_negative"] = bool((df[MEDIA_VARS] >= 0).all().all())
    max_lag = max([e["lag"] for e in spec.channel_edges] + [0])
    checks["lags_valid_for_length"] = max_lag < len(df) // 3
    checks["graph_type_valid"] = spec.graph_type in GRAPH_TYPES
    checks["n_sales_parents"] = len(spec.sales_parents)
    checks["n_channel_edges"] = len(spec.channel_edges)
    checks["cycle_removed_edges"] = spec.cycle_removed_edges
    if spec.graph_type == "pcmci_dag" and not spec.sales_parents:
        checks.setdefault("warnings", []).append(
            "pcmci_dag graph has no discovered media->Sales edges; the Sales "
            "equation contains controls and intercept only")

    hard = ["graph_is_acyclic", "all_graph_nodes_in_dataset",
            "sales_in_dataset", "sales_has_no_outgoing_edges",
            "no_duplicate_edges", "no_missing_values", "media_non_negative",
            "lags_valid_for_length", "graph_type_valid"]
    checks["all_passed"] = all(bool(checks[k]) for k in hard)
    if not checks["all_passed"]:
        failed = [k for k in hard if not checks[k]]
        raise ValueError(f"graph validation failed for {cell_dir.name}/"
                         f"{spec.graph_type}: {failed}")
    return checks


# ===========================================================================
# 2. Data preparation (benchmark-identical scaling)
# ===========================================================================

def prepare_data(df: pd.DataFrame, spec: GraphSpec) -> dict:
    """
    Media / max, Sales / max, controls z-standardized — identical to the
    benchmark. Additionally stores fixed standardization constants of the
    scaled parent series for the channel-to-channel equations (structural
    parameters: they are NOT recomputed under interventions).
    """
    media_raw = df[MEDIA_VARS].to_numpy(float)
    media_max = media_raw.max(axis=0)
    media_scaled = media_raw / media_max

    sales_raw = df[TARGET_VAR].to_numpy(float)
    sales_scale = float(sales_raw.max())
    sales_scaled = sales_raw / sales_scale

    ctrl_raw = df[CONTROL_VARS].to_numpy(float)
    ctrl_sd = ctrl_raw.std(axis=0)
    ctrl_sd[ctrl_sd < 1e-12] = 1.0
    ctrl_scaled = (ctrl_raw - ctrl_raw.mean(axis=0)) / ctrl_sd

    parent_norm = {ch: (float(media_scaled[:, i].mean()),
                        float(max(media_scaled[:, i].std(), 1e-9)))
                   for i, ch in enumerate(MEDIA_VARS)}

    sales_parent_idx = [MEDIA_VARS.index(ch) for ch in spec.sales_parents]
    return {"n_obs": len(df),
            "media_raw": media_raw, "media_scaled": media_scaled,
            "media_max": media_max, "media_index":
                {ch: i for i, ch in enumerate(MEDIA_VARS)},
            "sales_raw": sales_raw, "sales_scaled": sales_scaled,
            "sales_scale": sales_scale,
            "controls_scaled": ctrl_scaled,
            "parent_norm": parent_norm,
            "sales_parent_idx": sales_parent_idx,
            "lag_cube_sales": build_lag_cube(
                media_scaled[:, sales_parent_idx], L_MAX)
                if sales_parent_idx else np.zeros((0, len(df), L_MAX))}


def _std_lagged(series: np.ndarray, lag: int, norm: tuple) -> np.ndarray:
    """Standardized lagged series with zero-padding before start."""
    mu, sd = norm
    out = np.zeros_like(series)
    out[lag:] = (series[:-lag] - mu) / sd if lag > 0 else (series - mu) / sd
    return out


# ===========================================================================
# 3. Automatic model construction from the graph
# ===========================================================================

def build_model(data: dict, spec: GraphSpec) -> pm.Model:
    coords = {"control": CONTROL_VARS, "obs": np.arange(data["n_obs"])}
    if spec.sales_parents:
        coords["sales_channel"] = spec.sales_parents
    for node in spec.endogenous_nodes:
        coords[f"parents_{node}"] = [e["source"] for e in spec.parents_of(node)]

    with pm.Model(coords=coords) as model:
        ctrl = pm.Data("controls", data["controls_scaled"])

        # ---- Sales equation (benchmark-identical structure) --------------
        intercept = pm.Normal("intercept_sales", 0.5, 0.5)
        b_ctrl = pm.Normal("beta_control_sales", 0.0, 0.5, dims="control")
        sigma_y = pm.HalfNormal("sigma_sales", 0.5)
        mu_y = intercept + pt.dot(ctrl, b_ctrl)

        if spec.sales_parents:
            cube = pm.Data("lag_cube_sales", data["lag_cube_sales"])
            theta = pm.Beta("adstock_theta", 2.0, 2.0, dims="sales_channel")
            alpha = pm.Gamma("hill_alpha", 3.0, 2.0, dims="sales_channel")
            k = pm.Beta("hill_k", 2.0, 2.0, dims="sales_channel")
            beta = pm.HalfNormal("beta_media_sales", 0.5,
                                 dims="sales_channel")
            adstocked = adstock_from_cube(cube, theta, L_MAX)      # (C, T)
            saturated = hill(adstocked.T, alpha, k)                # (T, C)
            media_effect = pm.Deterministic(
                "media_contrib_scaled", saturated * beta,
                dims=("obs", "sales_channel"))
            mu_y = mu_y + media_effect.sum(axis=-1)

        pm.Deterministic("mu_sales", mu_y, dims="obs")
        pm.Normal("sales_obs", mu=mu_y, sigma=sigma_y,
                  observed=data["sales_scaled"], dims="obs")

        # ---- Endogenous media equations ----------------------------------
        for node in spec.endogenous_nodes:
            parents = spec.parents_of(node)
            j = data["media_index"][node]
            X = np.column_stack([
                _std_lagged(data["media_scaled"][:, data["media_index"]
                            [e["source"]]], e["lag"],
                            data["parent_norm"][e["source"]])
                for e in parents])
            Xd = pm.Data(f"parents_data_{node}", X)
            a_n = pm.Normal(f"intercept_{node}", 0.5, 0.5)
            delta = pm.HalfNormal(f"delta_{node}", 0.5,
                                  dims=f"parents_{node}")
            lam = pm.Normal(f"beta_control_{node}", 0.0, 0.5, dims="control")
            sig = pm.HalfNormal(f"sigma_{node}", 0.5)
            mu_n = a_n + pt.dot(Xd, delta) + pt.dot(ctrl, lam)
            pm.Deterministic(f"mu_{node}", mu_n, dims="obs")
            pm.Normal(f"{node}_obs", mu=mu_n, sigma=sig,
                      observed=data["media_scaled"][:, j], dims="obs")
    return model


def param_var_names(spec: GraphSpec) -> list[str]:
    names = ["intercept_sales", "beta_control_sales", "sigma_sales"]
    if spec.sales_parents:
        names += ["beta_media_sales", "adstock_theta", "hill_alpha", "hill_k"]
    for node in spec.endogenous_nodes:
        names += [f"intercept_{node}", f"delta_{node}",
                  f"beta_control_{node}", f"sigma_{node}"]
    return names


def fit(model: pm.Model, draws=1000, tune=1000, chains=4,
        target_accept=0.95, seed=42, cores=None) -> az.InferenceData:
    if cores is None:
        import os
        cores = max(1, min(chains, os.cpu_count() or 1))
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                         target_accept=target_accept, random_seed=seed,
                         progressbar=False)


# ===========================================================================
# 4. Posterior draw extraction
# ===========================================================================

def posterior_draws(idata, spec: GraphSpec, max_draws=500, seed=0) -> dict:
    """Thinned stacked posterior draws for all parameters (draw-first axes)."""
    post = idata.posterior
    if hasattr(post, "dataset"):
        post = post.dataset
    post = post.stack(sample=("chain", "draw"))
    n = post.sizes["sample"]
    rng = np.random.default_rng(seed)
    idx = (np.arange(n) if n <= max_draws
           else np.sort(rng.choice(n, max_draws, replace=False)))

    def get(name):
        return np.moveaxis(np.asarray(post[name]), -1, 0)[idx]

    names = param_var_names(spec)
    return {name: get(name) for name in names}


# ===========================================================================
# 5. Structural predictions and counterfactual propagation
# ===========================================================================

def _sales_expectation(draws: dict, data: dict, spec: GraphSpec,
                       media_scaled: np.ndarray) -> np.ndarray:
    """E[Sales_scaled] per draw, shape (S, T), for a given media matrix."""
    S = draws["intercept_sales"].shape[0]
    T = media_scaled.shape[0]
    mu = (draws["intercept_sales"][:, None]
          + np.einsum("tk,sk->st", data["controls_scaled"],
                      draws["beta_control_sales"]))
    if spec.sales_parents:
        cube = build_lag_cube(media_scaled[:, data["sales_parent_idx"]], L_MAX)
        for s in range(S):
            ad = adstock_from_cube(cube, draws["adstock_theta"][s], L_MAX)
            sat = hill(ad.T, draws["hill_alpha"][s], draws["hill_k"][s])
            mu[s] += (sat * draws["beta_media_sales"][s]).sum(axis=-1)
    return mu


def _node_structural_pred(draws: dict, data: dict, spec: GraphSpec,
                          node: str, media_scaled: np.ndarray) -> np.ndarray:
    """E[node_scaled] per draw, shape (S, T), given (possibly cf) parents."""
    parents = spec.parents_of(node)
    X = np.column_stack([
        _std_lagged(media_scaled[:, data["media_index"][e["source"]]],
                    e["lag"], data["parent_norm"][e["source"]])
        for e in parents])
    return (draws[f"intercept_{node}"][:, None]
            + np.einsum("tp,sp->st", X, draws[f"delta_{node}"])
            + np.einsum("tk,sk->st", data["controls_scaled"],
                        draws[f"beta_control_{node}"]))


def counterfactual_sales(draws: dict, data: dict, spec: GraphSpec,
                         channel: str, factor: float) -> dict:
    """
    do(X_channel := factor * X_channel) with structural downstream
    propagation, per posterior draw, expected values only.

    Abduction-style: the counterfactual series of an endogenous node is the
    OBSERVED series plus the structural shift implied by its equation
    (residuals held fixed) — mirroring the noise-replay semantics of the
    ground truth. Returns per-draw delta sales (S, T) on the SCALED scale
    plus the counterfactual media matrices per draw for spend accounting.
    """
    S = draws["intercept_sales"].shape[0]
    T = data["n_obs"]
    ci = data["media_index"][channel]

    base_media = data["media_scaled"]
    cf_common = base_media.copy()
    cf_common[:, ci] = cf_common[:, ci] * factor      # do(): path fixed

    # propagate structural shifts in topological order, per draw
    cf_stack = np.repeat(cf_common[None, :, :], S, axis=0)   # (S, T, C)
    for node in spec.endogenous_nodes:
        if node == channel:
            continue                                   # do() cuts equations
        j = data["media_index"][node]
        base_pred = _node_structural_pred(draws, data, spec, node, base_media)
        for s in range(S):
            cf_pred_s = _node_structural_pred(
                {k: v[s:s + 1] for k, v in draws.items()},
                data, spec, node, cf_stack[s])[0]
            shift = cf_pred_s - base_pred[s]
            cf_stack[s, :, j] = np.maximum(base_media[:, j] + shift, 0.0)

    mu_base = _sales_expectation(draws, data, spec, base_media)   # (S, T)
    d_sales = np.empty((S, T))
    for s in range(S):
        mu_cf_s = _sales_expectation(
            {k: v[s:s + 1] for k, v in draws.items()},
            data, spec, cf_stack[s])[0]
        d_sales[s] = mu_cf_s - mu_base[s]
    return {"d_sales_scaled": d_sales, "cf_media": cf_stack}


def estimate_interventional_effects(idata, data, spec: GraphSpec,
                                    factor=INTERVENTION_FACTOR,
                                    burn_in=BURN_IN_WEEKS,
                                    max_draws=300) -> pd.DataFrame:
    """Model-implied do(+10%) effects per channel, original scale, with HDI."""
    draws = posterior_draws(idata, spec, max_draws=max_draws)
    sl = slice(burn_in, None)
    rows = []
    for ch in MEDIA_VARS:
        cf = counterfactual_sales(draws, data, spec, ch, factor)
        d_sales = cf["d_sales_scaled"][:, sl].sum(axis=1) * data["sales_scale"]
        ci = data["media_index"][ch]
        d_own = (factor - 1.0) * data["media_raw"][sl, ci].sum()
        # induced downstream spend (original scale), per draw -> mean
        d_total = np.empty(len(d_sales))
        for s in range(len(d_sales)):
            diff = (cf["cf_media"][s][sl] - data["media_scaled"][sl]) \
                * data["media_max"][None, :]
            d_total[s] = diff.sum()
        roas_own = d_sales / d_own
        hdi_s, hdi_r = _hdi(d_sales), _hdi(roas_own)
        rows.append({
            "channel": ch, "factor": factor, "burn_in_weeks": burn_in,
            "delta_sales_total_mean": float(d_sales.mean()),
            "delta_sales_total_sd": float(d_sales.std()),
            "delta_sales_hdi_low": float(hdi_s[0]),
            "delta_sales_hdi_high": float(hdi_s[1]),
            "delta_own_spend": float(d_own),
            "delta_total_media_spend_mean": float(d_total.mean()),
            "marginal_roas_mean": float(roas_own.mean()),
            "marginal_roas_hdi_low": float(hdi_r[0]),
            "marginal_roas_hdi_high": float(hdi_r[1]),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 6. Attribution, ROAS, model fit, node fit
# ===========================================================================

def attribution(idata, data, spec: GraphSpec, max_draws=300) -> pd.DataFrame:
    """
    Direct Sales-equation contribution attribution (posterior mean, original
    scale). NOTE: for DAG models this is NOT the total causal effect —
    indirect paths are reported via the counterfactual evaluation only.
    Channels that are not parents of Sales receive a zero direct
    contribution by definition of this attribution.
    """
    draws = posterior_draws(idata, spec, max_draws=max_draws)
    contrib = {ch: 0.0 for ch in MEDIA_VARS}
    if spec.sales_parents:
        cube = data["lag_cube_sales"]
        S = draws["intercept_sales"].shape[0]
        total = np.zeros((S, data["n_obs"], len(spec.sales_parents)))
        for s in range(S):
            ad = adstock_from_cube(cube, draws["adstock_theta"][s], L_MAX)
            sat = hill(ad.T, draws["hill_alpha"][s], draws["hill_k"][s])
            total[s] = sat * draws["beta_media_sales"][s]
        per_channel = total.sum(axis=1).mean(axis=0) * data["sales_scale"]
        for i, ch in enumerate(spec.sales_parents):
            contrib[ch] = float(per_channel[i])
    ctrl_total = float(
        (np.einsum("tk,sk->st", data["controls_scaled"],
                   draws["beta_control_sales"]).sum(axis=1).mean())
        * data["sales_scale"])
    base_total = float(draws["intercept_sales"].mean() * data["n_obs"]
                       * data["sales_scale"])
    rows = ([{"component": ch, "contribution_abs": contrib[ch]}
             for ch in MEDIA_VARS]
            + [{"component": "Controls", "contribution_abs": ctrl_total},
               {"component": "Baseline_Intercept",
                "contribution_abs": base_total}])
    out = pd.DataFrame(rows)
    total_sum = out["contribution_abs"].sum()
    out["contribution_pct"] = out["contribution_abs"] / total_sum * 100
    out["mean_weekly"] = out["contribution_abs"] / data["n_obs"]
    return out


def average_roas(idata, data, spec: GraphSpec, max_draws=300) -> pd.DataFrame:
    """Average ROAS = direct Sales-equation contribution / total spend."""
    draws = posterior_draws(idata, spec, max_draws=max_draws)
    rows = []
    if spec.sales_parents:
        cube = data["lag_cube_sales"]
        S = draws["intercept_sales"].shape[0]
        contrib = np.zeros((S, len(spec.sales_parents)))
        for s in range(S):
            ad = adstock_from_cube(cube, draws["adstock_theta"][s], L_MAX)
            sat = hill(ad.T, draws["hill_alpha"][s], draws["hill_k"][s])
            contrib[s] = (sat * draws["beta_media_sales"][s]).sum(axis=0)
        contrib *= data["sales_scale"]
        for i, ch in enumerate(spec.sales_parents):
            spend = data["media_raw"][:, data["media_index"][ch]].sum()
            roas = contrib[:, i] / spend
            h = _hdi(roas)
            rows.append({"channel": ch, "avg_roas_mean": float(roas.mean()),
                         "avg_roas_sd": float(roas.std()),
                         "avg_roas_hdi_low": float(h[0]),
                         "avg_roas_hdi_high": float(h[1])})
    for ch in MEDIA_VARS:
        if ch not in spec.sales_parents:
            rows.append({"channel": ch, "avg_roas_mean": 0.0,
                         "avg_roas_sd": 0.0, "avg_roas_hdi_low": 0.0,
                         "avg_roas_hdi_high": 0.0})
    return pd.DataFrame(rows)


def model_fit(idata, data, spec: GraphSpec, max_draws=300) -> dict:
    """Bayesian R^2 (Gelman et al. 2019) etc. for the Sales equation."""
    draws = posterior_draws(idata, spec, max_draws=max_draws)
    mu = _sales_expectation(draws, data, spec, data["media_scaled"])
    y = data["sales_scaled"][None, :]
    resid = y - mu
    r2_draws = mu.var(axis=1) / (mu.var(axis=1) + resid.var(axis=1))
    h = _hdi(r2_draws)
    mu_mean = mu.mean(0)
    ss_res = ((data["sales_scaled"] - mu_mean) ** 2).sum()
    ss_tot = ((data["sales_scaled"] - data["sales_scaled"].mean()) ** 2).sum()
    resid_raw = (data["sales_scaled"] - mu_mean) * data["sales_scale"]
    return {"bayes_r2_mean": float(r2_draws.mean()),
            "bayes_r2_hdi_low": float(h[0]),
            "bayes_r2_hdi_high": float(h[1]),
            "r2_posterior_mean_pred": float(1.0 - ss_res / ss_tot),
            "rmse_sales": float(np.sqrt((resid_raw ** 2).mean())),
            "mae_sales": float(np.abs(resid_raw).mean())}


def node_fit_metrics(idata, data, spec: GraphSpec,
                     max_draws=300) -> pd.DataFrame:
    """R^2 / RMSE / MAE of endogenous media equations (original scale)."""
    rows = []
    if spec.endogenous_nodes:
        draws = posterior_draws(idata, spec, max_draws=max_draws)
        for node in spec.endogenous_nodes:
            j = data["media_index"][node]
            pred = _node_structural_pred(draws, data, spec, node,
                                         data["media_scaled"]).mean(0)
            obs = data["media_scaled"][:, j]
            ss_res = ((obs - pred) ** 2).sum()
            ss_tot = ((obs - obs.mean()) ** 2).sum()
            resid_raw = (obs - pred) * data["media_max"][j]
            rows.append({"node": node,
                         "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
                         "rmse": float(np.sqrt((resid_raw ** 2).mean())),
                         "mae": float(np.abs(resid_raw).mean())})
    return pd.DataFrame(rows, columns=["node", "r2", "rmse", "mae"])


# ===========================================================================
# 7. Ground-truth comparison (evaluation only — never used during fitting)
# ===========================================================================

def compare_to_ground_truth(effects: pd.DataFrame,
                            cell_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Spec-exact comparison table and aggregates (both ROAS references)."""
    truth = pd.read_csv(cell_dir / "true_interventional_effects.csv")
    merged = effects.merge(
        truth[["channel", "delta_sales_total", "marginal_roas_own",
               "marginal_roas_total"]], on="channel")
    out = pd.DataFrame({
        "channel": merged["channel"],
        "estimated_effect": merged["delta_sales_total_mean"],
        "estimated_effect_hdi_low": merged["delta_sales_hdi_low"],
        "estimated_effect_hdi_high": merged["delta_sales_hdi_high"],
        "estimated_roas": merged["marginal_roas_mean"],
        "estimated_roas_hdi_low": merged["marginal_roas_hdi_low"],
        "estimated_roas_hdi_high": merged["marginal_roas_hdi_high"],
        "true_delta_sales_total": merged["delta_sales_total"],
        "true_marginal_roas_own": merged["marginal_roas_own"],
        "true_marginal_roas_total": merged["marginal_roas_total"],
    })
    out["bias_effect"] = out["estimated_effect"] - out["true_delta_sales_total"]
    out["hdi_covers_effect"] = (
        (out["true_delta_sales_total"] >= out["estimated_effect_hdi_low"])
        & (out["true_delta_sales_total"] <= out["estimated_effect_hdi_high"]))
    for ref in ["own", "total"]:
        t = out[f"true_marginal_roas_{ref}"]
        out[f"bias_vs_{ref}"] = out["estimated_roas"] - t
        out[f"hdi_covers_{ref}"] = ((t >= out["estimated_roas_hdi_low"])
                                    & (t <= out["estimated_roas_hdi_high"]))

    def _mae(x): return float(np.abs(x).mean())
    def _rmse(x): return float(np.sqrt((x ** 2).mean()))
    agg = {"mean_bias_effect": float(out["bias_effect"].mean()),
           "mae_effect": _mae(out["bias_effect"]),
           "rmse_effect": _rmse(out["bias_effect"]),
           "hdi_coverage_effect": float(out["hdi_covers_effect"].mean()),
           "mae_vs_own": _mae(out["bias_vs_own"]),
           "rmse_vs_own": _rmse(out["bias_vs_own"]),
           "mae_vs_total": _mae(out["bias_vs_total"]),
           "rmse_vs_total": _rmse(out["bias_vs_total"]),
           "hdi_coverage_vs_own": float(out["hdi_covers_own"].mean()),
           "hdi_coverage_vs_total": float(out["hdi_covers_total"].mean())}
    return out, agg


# ===========================================================================
# 8. Diagnostics (benchmark-style)
# ===========================================================================

def diagnostics_report(idata, model, data, spec: GraphSpec,
                       summary: pd.DataFrame, sampler_cfg: dict,
                       idata_saved: bool, smoke_test: bool,
                       out_dir: Path, seed: int = 1) -> dict:
    ss = _group_dataset(idata, "sample_stats")
    divergences = (int(np.asarray(ss["diverging"]).sum())
                   if "diverging" in ss else None)
    acc_name = next((n for n in ("acceptance_rate", "mean_tree_accept",
                                 "accept") if n in ss), None)
    acceptance = float(np.asarray(ss[acc_name]).mean()) if acc_name else None
    post = _group_dataset(idata, "posterior")

    with model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["sales_obs"], progressbar=False,
            random_seed=seed)
    y_rep = np.asarray(_group_dataset(ppc, "posterior_predictive")
                       ["sales_obs"])
    y_rep = y_rep.reshape(-1, y_rep.shape[-1]) * data["sales_scale"]
    y_obs = data["sales_raw"]
    lo = np.quantile(y_rep, 0.03, axis=0)
    hi = np.quantile(y_rep, 0.97, axis=0)
    ppc_summary = {
        "obs_mean": float(y_obs.mean()),
        "rep_mean": float(y_rep.mean(axis=1).mean()),
        "bayes_p_mean": float((y_rep.mean(axis=1) >= y_obs.mean()).mean()),
        "obs_sd": float(y_obs.std()),
        "rep_sd": float(y_rep.std(axis=1).mean()),
        "bayes_p_sd": float((y_rep.std(axis=1) >= y_obs.std()).mean()),
        "coverage_94_interval": float(((y_obs >= lo) & (y_obs <= hi)).mean())}

    report = {"graph_type": spec.graph_type,
              "r_hat_max": float(summary["r_hat"].max()),
              "r_hat_median": float(summary["r_hat"].median()),
              "ess_bulk_min": float(summary["ess_bulk"].min()),
              "ess_tail_min": float(summary["ess_tail"].min()),
              "divergences": divergences,
              "acceptance_rate_mean": acceptance,
              "n_chains": int(post.sizes["chain"]),
              "n_draws_per_chain": int(post.sizes["draw"]),
              "sampler_settings": sampler_cfg,
              "priors": PRIORS_SPEC,
              "posterior_predictive_check": ppc_summary,
              "idata_saved": bool(idata_saved),
              "smoke_test": bool(smoke_test),
              "versions": {"pymc": pm.__version__, "arviz": az.__version__}}
    with open(out_dir / "diagnostics_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


# ===========================================================================
# 9. Per-cell orchestration
# ===========================================================================

def run_structural_mmm_on_cell(cell_dir: str | Path, graph_type: str,
                               draws=1000, tune=1000, chains=4,
                               target_accept=0.95, seed=42,
                               pcmci_setting: str = "controlled",
                               max_posterior_draws: int = 500,
                               smoke_test: bool = False) -> dict:
    """Fit one graph variant on one cell and export the full output set."""
    t0 = time.time()
    cell_dir = Path(cell_dir)
    out_dir = cell_dir / "structural_bayesian_mmm" / graph_type
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cell_dir / "observed_controlled.parquet")
    spec = load_graph(cell_dir, graph_type, pcmci_setting=pcmci_setting)
    validation = validate_graph(spec, df, cell_dir)
    validation["smoke_test"] = smoke_test
    with open(out_dir / "model_graph_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    # graph exports (model input documentation)
    edges_rows = ([{"source": e["source"], "target": e["target"],
                    "lag": e["lag"], "edge_type": "interdependency"}
                   for e in spec.channel_edges]
                  + [{"source": p, "target": TARGET_VAR, "lag": 0,
                      "edge_type": "direct_sales"}
                     for p in spec.sales_parents])
    pd.DataFrame(edges_rows).to_csv(out_dir / "model_graph_edges.csv",
                                    index=False)
    adj = pd.DataFrame(0, index=SYSTEM_VARS, columns=SYSTEM_VARS, dtype=int)
    for s, t in spec.all_edges():
        adj.loc[s, t] = 1
    adj.to_csv(out_dir / "model_graph_adjacency.csv")

    data = prepare_data(df, spec)
    model = build_model(data, spec)
    idata = fit(model, draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, seed=seed)

    summary = _az_summary(idata, param_var_names(spec))
    summary.to_csv(out_dir / "posterior_summary.csv")
    plot_error = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            az.plot_trace(idata, var_names=param_var_names(spec),
                          compact=True)
        except (TypeError, ValueError):
            az.plot_trace(idata, var_names=param_var_names(spec))
        plt.gcf().suptitle(f"Structural MMM ({graph_type}) — trace plots",
                           y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / "trace_plots.png", dpi=130,
                    bbox_inches="tight")
        plt.close("all")
    except Exception as exc:                     # plotting is best-effort
        plot_error = repr(exc)

    effects = estimate_interventional_effects(
        idata, data, spec, max_draws=max_posterior_draws)
    effects.to_csv(out_dir / "estimated_effects.csv", index=False)

    attribution(idata, data, spec, max_draws=max_posterior_draws) \
        .to_csv(out_dir / "attribution.csv", index=False)
    average_roas(idata, data, spec, max_draws=max_posterior_draws) \
        .to_csv(out_dir / "roas.csv", index=False)

    fit_metrics = model_fit(idata, data, spec,
                            max_draws=max_posterior_draws)
    fit_metrics["smoke_test"] = smoke_test
    with open(out_dir / "model_fit.json", "w") as f:
        json.dump(fit_metrics, f, indent=2)
    node_fit_metrics(idata, data, spec, max_draws=max_posterior_draws) \
        .to_csv(out_dir / "node_fit_metrics.csv", index=False)

    comparison, agg = compare_to_ground_truth(effects, cell_dir)
    comparison.to_csv(out_dir / "ground_truth_comparison.csv", index=False)
    agg["smoke_test"] = smoke_test
    with open(out_dir / "comparison_aggregates.json", "w") as f:
        json.dump(agg, f, indent=2)

    idata_saved = True
    try:
        idata.to_netcdf(out_dir / "idata.nc")
    except Exception:                            # needs h5netcdf + h5py
        idata_saved = False

    diag = diagnostics_report(
        idata, model, data, spec, summary,
        sampler_cfg={"draws": draws, "tune": tune, "chains": chains,
                     "target_accept": target_accept, "seed": seed,
                     "pcmci_setting": pcmci_setting},
        idata_saved=idata_saved, smoke_test=smoke_test, out_dir=out_dir)

    runtime = round(time.time() - t0, 1)
    row = {"cell": cell_dir.name, "graph_type": graph_type,
           "bayes_r2_mean": round(fit_metrics["bayes_r2_mean"], 4),
           "rmse_sales": round(fit_metrics["rmse_sales"], 3),
           "mae_sales": round(fit_metrics["mae_sales"], 3),
           **{k: round(v, 4) for k, v in agg.items()
              if isinstance(v, float)},
           "divergences": diag["divergences"],
           "r_hat_max": diag["r_hat_max"],
           "ess_bulk_min": diag["ess_bulk_min"],
           "runtime_s": runtime, "smoke_test": smoke_test,
           "plot_error": plot_error}
    return row