from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNELS = ["TV", "Video", "Display", "Social", "Search", "Affiliate"]
EXOGENOUS = ["TV", "Video", "Display", "Affiliate"]
ENDOGENOUS = ["Social", "Search"]          # topological order matters
NODES = CHANNELS + ["Sales"]

ADSTOCK_THETA = {
    "TV": 0.70, "Video": 0.60, "Display": 0.40,
    "Social": 0.35, "Search": 0.20, "Affiliate": 0.25,
}
GAMMA_LEVELS = {"weak": 0.10, "medium": 0.30, "strong": 0.60}
NOISE_LEVELS = {"low": 0.10, "medium": 0.20, "high": 0.35}
LENGTH_LEVELS = {"short": 80, "baseline": 156, "long": 300}

CROSS_EFFECT_SCALE = 12.0   # spend units per 1 sd parent deviation at gamma=1


@dataclass
class ScenarioConfig:
    n_weeks: int = 156
    interdependency: str = "medium"
    noise: str = "medium"
    dag_density: str = "medium"
    social_direct_effect: bool = True
    seed: int = 42

    growth_rate: float = 0.0008
    season_A: float = 80.0
    season_B: float = 40.0
    demand_noise_sd: float = 25.0

    target_demand_share: float = 0.40
    target_media_share: float = 0.40

    def gamma(self) -> float:
        return GAMMA_LEVELS[self.interdependency]

    def noise_share(self) -> float:
        return NOISE_LEVELS[self.noise]


# ---------------------------------------------------------------------------
# Step 1: Baseline demand
# ---------------------------------------------------------------------------

def generate_base_demand(cfg: ScenarioConfig, rng: np.random.Generator):
    t = np.arange(cfg.n_weeks)
    trend = 1000.0 * (1.0 + cfg.growth_rate * t)
    seasonality = (cfg.season_A * np.sin(2 * np.pi * t / 52)
                   + cfg.season_B * np.cos(2 * np.pi * t / 52))
    woy = t % 52
    events = np.zeros(cfg.n_weeks)
    for start, end, height in [(46, 48, 160.0), (49, 52, 220.0),
                               (26, 29, 90.0), (34, 37, 70.0)]:
        events += np.where((woy >= start) & (woy < end), height, 0.0)
    market_noise = rng.normal(0.0, cfg.demand_noise_sd, cfg.n_weeks)
    base_demand = trend + seasonality + events + market_noise
    components = pd.DataFrame({"Trend": trend, "Seasonality": seasonality,
                               "Event_Effect": events,
                               "Market_Noise": market_noise})
    return base_demand, components


# ---------------------------------------------------------------------------
# Steps 2-3: Campaign flighting (demand-driven planning)
# ---------------------------------------------------------------------------

FLIGHT_SHAPES = ("flat", "ramp_up", "ramp_down", "triangular")


def _flight_profile(duration: int, shape: str) -> np.ndarray:
    if shape == "flat":
        return np.ones(duration)
    if shape == "ramp_up":
        return np.linspace(0.4, 1.0, duration)
    if shape == "ramp_down":
        return np.linspace(1.0, 0.4, duration)
    if shape == "triangular":
        half = (duration + 1) // 2
        up = np.linspace(0.4, 1.0, half)
        down = np.linspace(1.0, 0.4, duration - half + 1)[1:]
        return np.concatenate([up, down])
    raise ValueError(shape)


def generate_flights(cfg, rng, demand, *, flights_per_year, duration_range,
                     intensity_range, demand_sensitivity=2.0):
    n = cfg.n_weeks
    series = np.zeros(n)
    flight_log = []
    d_norm = (demand - demand.mean()) / demand.std()
    n_years = max(n / 52.0, 1.0)
    n_candidates = int(np.ceil(rng.uniform(*flights_per_year) * n_years * 1.8))
    a, b = -0.4, demand_sensitivity
    occupied = np.zeros(n, dtype=bool)
    for _ in range(n_candidates):
        start = int(rng.integers(0, max(n - duration_range[1], 1)))
        p_accept = 1.0 / (1.0 + np.exp(-(a + b * d_norm[start])))
        if rng.uniform() > p_accept:
            continue
        duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
        end = min(start + duration, n)
        if occupied[start:end].any():
            continue
        occupied[start:end] = True
        intensity = rng.uniform(*intensity_range)
        shape = FLIGHT_SHAPES[rng.integers(0, len(FLIGHT_SHAPES))]
        series[start:end] += intensity * _flight_profile(end - start, shape)
        flight_log.append({"start": start, "duration": int(end - start),
                           "intensity": float(intensity), "shape": shape})
    return series, flight_log


FLIGHT_BEHAVIOR = {
    "TV":        dict(flights_per_year=(2, 4),  duration_range=(3, 6),
                      intensity_range=(120, 200), demand_sensitivity=2.5),
    "Video":     dict(flights_per_year=(3, 5),  duration_range=(2, 5),
                      intensity_range=(80, 150),  demand_sensitivity=2.0),
    "Display":   dict(flights_per_year=(6, 10), duration_range=(1, 3),
                      intensity_range=(30, 70),   demand_sensitivity=1.0),
    "Social":    dict(flights_per_year=(4, 7),  duration_range=(2, 4),
                      intensity_range=(50, 100),  demand_sensitivity=1.5),
    "Search":    dict(flights_per_year=(3, 5),  duration_range=(1, 3),
                      intensity_range=(20, 45),   demand_sensitivity=1.0),
    "Affiliate": dict(flights_per_year=(3, 5),  duration_range=(1, 3),
                      intensity_range=(30, 60),   demand_sensitivity=2.0),
}
CHANNEL_BASELINE = {"TV": 15.0, "Video": 20.0, "Display": 45.0,
                    "Social": 35.0, "Search": 60.0, "Affiliate": 40.0}
CHANNEL_DEMAND_EFFECT = {"TV": 0.010, "Video": 0.010, "Display": 0.015,
                         "Social": 0.015, "Search": 0.030, "Affiliate": 0.012}
CHANNEL_RHO = {"TV": 0.45, "Video": 0.45, "Display": 0.55,
               "Social": 0.50, "Search": 0.60, "Affiliate": 0.55}
CHANNEL_NOISE_SD = {"TV": 6.0, "Video": 6.0, "Display": 5.0,
                    "Social": 5.0, "Search": 5.0, "Affiliate": 4.0}


# ---------------------------------------------------------------------------
# Structural dynamics (deterministic given flights + noise -> replayable)
# ---------------------------------------------------------------------------

def _exogenous_dynamics(name, cfg, demand, flights, noise):
    """Spend_i(t) = base + flight + rho*excess(t-1) + demand effect + eps(t)."""
    n = cfg.n_weeks
    spend = np.zeros(n)
    rho = CHANNEL_RHO[name]
    d_centered = demand - demand.mean()
    for t in range(n):
        prev = spend[t - 1] - CHANNEL_BASELINE[name] if t > 0 else 0.0
        spend[t] = (CHANNEL_BASELINE[name] + flights[t]
                    + rho * max(prev, 0.0) * 0.3
                    + CHANNEL_DEMAND_EFFECT[name] * d_centered[t]
                    + noise[t])
    return np.maximum(spend, 0.0)


def _endogenous_dynamics(name, cfg, demand, parents, gamma, flights, noise,
                         parent_norm):
    """
    X(t) = base + flight + rho*excess(t-1)
           + gamma * sum_p standardized(parent_p(t-1)) * SCALE
           + demand effect + eps(t)

    parent_norm holds FIXED (observational) standardization constants
    {parent: (mean, std)} so interventions are not absorbed by
    re-standardization.
    """
    n = cfg.n_weeks
    spend = np.zeros(n)
    rho = CHANNEL_RHO[name]
    d_centered = demand - demand.mean()
    for t in range(n):
        prev = spend[t - 1] - CHANNEL_BASELINE[name] if t > 0 else 0.0
        cross = 0.0
        if t > 0:
            for p, series in parents.items():
                mu, sd = parent_norm[p]
                cross += gamma * ((series[t - 1] - mu) / sd) * CROSS_EFFECT_SCALE
        spend[t] = (CHANNEL_BASELINE[name] + flights[t]
                    + rho * max(prev, 0.0) * 0.3
                    + cross
                    + CHANNEL_DEMAND_EFFECT[name] * d_centered[t]
                    + noise[t])
    return np.maximum(spend, 0.0)


# ---------------------------------------------------------------------------
# Steps 7-8: Adstock and Hill saturation
# ---------------------------------------------------------------------------

def apply_adstock(spend: np.ndarray, theta: float) -> np.ndarray:
    out = np.zeros_like(spend)
    for t in range(len(spend)):
        out[t] = spend[t] + (theta * out[t - 1] if t > 0 else 0.0)
    return out


def apply_hill_saturation(adstock, alpha, k=None):
    if k is None:
        k = float(np.median(adstock))
    k = max(k, 1e-9)
    xa = np.power(np.maximum(adstock, 0.0), alpha)
    return xa / (xa + k ** alpha), k


RAW_BETA_WEIGHTS = {"Search": 1.00, "TV": 0.75, "Video": 0.55,
                    "Affiliate": 0.45, "Social": 0.15}
SALES_LAG = {"Search": 1, "TV": 2, "Video": 2, "Affiliate": 1, "Social": 1}
HILL_ALPHA = {"TV": 2.0, "Video": 1.8, "Display": 1.5,
              "Social": 1.6, "Search": 1.4, "Affiliate": 1.5}


def _lagged(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(x)
    if lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:] = x
    return out


def _sales_from_saturated(base_demand, saturated, betas, sales_noise):
    """Deterministic sales equation given saturated media and fixed noise."""
    media = np.zeros_like(base_demand)
    for ch, beta in betas.items():
        media += beta * _lagged(saturated[ch], SALES_LAG[ch])
    return base_demand + media + sales_noise, media


def generate_sales(cfg, rng, base_demand, saturated):
    """Calibrate betas + noise sd toward target variance decomposition."""
    direct = {ch: w for ch, w in RAW_BETA_WEIGHTS.items()
              if ch != "Social" or cfg.social_direct_effect}
    raw_media = np.zeros(cfg.n_weeks)
    for ch, w in direct.items():
        raw_media += w * _lagged(saturated[ch], SALES_LAG[ch])
    var_demand = base_demand.var()
    media_scale = np.sqrt((cfg.target_media_share / cfg.target_demand_share)
                          * var_demand / max(raw_media.var(), 1e-12))
    betas = {ch: float(w * media_scale) for ch, w in direct.items()}

    noise_share = cfg.noise_share()
    media_contrib = raw_media * media_scale
    signal_var = var_demand + media_contrib.var()
    noise_sd = float(np.sqrt(noise_share / (1.0 - noise_share) * signal_var))
    sales_noise = rng.normal(0.0, noise_sd, cfg.n_weeks)

    sales, media_contrib = _sales_from_saturated(base_demand, saturated,
                                                 betas, sales_noise)
    return sales, betas, noise_sd, media_contrib, sales_noise


# ---------------------------------------------------------------------------
# Ground truth graph
# ---------------------------------------------------------------------------

def build_ground_truth_graph(cfg: ScenarioConfig):
    edges = [("TV", "Search", 1), ("Video", "Search", 1),
             ("Social", "Search", 1), ("Display", "Social", 1),
             ("Search", "Sales", 1), ("Affiliate", "Sales", 1),
             ("TV", "Sales", 2), ("Video", "Sales", 2)]
    if cfg.social_direct_effect:
        edges.append(("Social", "Sales", 1))
    if cfg.dag_density == "sparse":
        edges = [e for e in edges if e[:2] not in
                 {("Video", "Search"), ("Display", "Social")}]
    elif cfg.dag_density == "dense":
        edges += [("TV", "Social", 1), ("Display", "Search", 1)]

    adj = pd.DataFrame(0, index=NODES, columns=NODES, dtype=int)
    for s, d, _ in edges:
        adj.loc[s, d] = 1
    lagged_edges = [{"source": s, "target": d, "lag": lag} for s, d, lag in edges]
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from([(s, d, {"lag": lag}) for s, d, lag in edges])
    assert nx.is_directed_acyclic_graph(g)
    return adj, lagged_edges, g


def compute_effects(g: nx.DiGraph, direct_strengths: dict) -> dict:
    """Structural path products (NOT marginal effects — see interventions)."""
    effects = {}
    for node in g.nodes:
        if node == "Sales":
            continue
        total = sum(np.prod([direct_strengths.get((u, v), 0.0)
                             for u, v in zip(p[:-1], p[1:])])
                    for p in nx.all_simple_paths(g, node, "Sales"))
        direct = direct_strengths.get((node, "Sales"), 0.0)
        effects[node] = {"direct": round(direct, 5),
                         "indirect": round(float(total) - direct, 5),
                         "total": round(float(total), 5)}
    return effects


# ---------------------------------------------------------------------------
# Interventional ground truth (counterfactual replay)
# ---------------------------------------------------------------------------

def replay_system(result: dict, spend_override: dict[str, np.ndarray] | None = None):
    """
    Deterministically replay the structural system with the stored noise.
    spend_override pins channels to given series (do-operator: incoming
    structural equations of overridden channels are cut).
    Returns (spend dict, sales array, media_contrib).
    """
    st = result["state"]
    cfg = result["config"]
    spend_override = spend_override or {}
    spend = {}

    for ch in EXOGENOUS:
        spend[ch] = (spend_override[ch].copy() if ch in spend_override
                     else st["spend"][ch].copy())

    for ch in ENDOGENOUS:   # topological order: Social before Search
        if ch in spend_override:
            spend[ch] = spend_override[ch].copy()
            continue
        parents = {p: spend[p] for p in st["parents"][ch]}
        spend[ch] = _endogenous_dynamics(ch, cfg, st["base_demand"], parents,
                                         st["gamma"], st["flights"][ch],
                                         st["noise"][ch], st["parent_norm"][ch])

    saturated = {}
    for ch in st["betas"]:
        ad = apply_adstock(spend[ch], ADSTOCK_THETA[ch])
        # Hill k FIXED at observational value (structural parameter)
        saturated[ch], _ = apply_hill_saturation(ad, HILL_ALPHA[ch],
                                                 k=st["hill_k"][ch])
    sales, media = _sales_from_saturated(st["base_demand"], saturated,
                                         st["betas"], st["sales_noise"])
    return spend, sales, media


def interventional_effects(result: dict, factor: float = 1.10,
                           burn_in: int = 5) -> dict:
    """
    Ground-truth marginal effects: for each channel i, simulate
    do(Spend_i := factor * Spend_i^obs) and propagate downstream.
    All noise is held fixed, so the sales delta is the pure causal effect
    including adstock carryover, Hill saturation and funnel spillovers.
    """
    st = result["state"]
    base_spend, base_sales, _ = replay_system(result)          # reference
    assert np.max(np.abs(base_sales - st["sales"])) < 1e-6, \
        "replay must reproduce baseline exactly"

    sl = slice(burn_in, None)
    out = {"factor": factor, "burn_in_weeks": burn_in, "channels": {}}
    for ch in CHANNELS:
        do_spend = {ch: st["spend"][ch] * factor}
        cf_spend, cf_sales, _ = replay_system(result, spend_override=do_spend)

        d_sales = cf_sales[sl] - base_sales[sl]
        d_own = (cf_spend[ch][sl] - base_spend[ch][sl]).sum()
        d_total_spend = sum((cf_spend[c][sl] - base_spend[c][sl]).sum()
                            for c in CHANNELS)
        induced = {c: round(float((cf_spend[c][sl] - base_spend[c][sl]).sum()), 2)
                   for c in CHANNELS if c != ch
                   and np.abs(cf_spend[c] - base_spend[c]).max() > 1e-9}
        out["channels"][ch] = {
            "delta_sales_total": round(float(d_sales.sum()), 2),
            "delta_sales_mean_weekly": round(float(d_sales.mean()), 3),
            "pct_sales_change": round(float(d_sales.sum()
                                            / base_sales[sl].sum() * 100), 3),
            "delta_own_spend": round(float(d_own), 2),
            "delta_total_media_spend": round(float(d_total_spend), 2),
            "induced_downstream_spend": induced,
            "marginal_roas_own": round(float(d_sales.sum() / d_own), 4)
                                 if abs(d_own) > 1e-9 else None,
            "marginal_roas_total": round(float(d_sales.sum() / d_total_spend), 4)
                                   if abs(d_total_spend) > 1e-9 else None,
        }
    return out


# ---------------------------------------------------------------------------
# Discovery settings (unobserved vs controlled demand)
# ---------------------------------------------------------------------------

def export_discovery_settings(result: dict) -> dict[str, pd.DataFrame]:
    """
    'unobserved': Week + media + Sales      (demand fully latent -> confounded)
    'controlled': + Seasonality, Trend, Event_Effect
                  (systematic demand observed; Market_Noise stays latent)
    """
    df = result["df"]
    base_cols = ["Week"] + CHANNELS + ["Sales"]
    controlled_cols = base_cols + ["Seasonality", "Trend", "Event_Effect"]
    return {"unobserved": df[base_cols].copy(),
            "controlled": df[controlled_cols].copy()}


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def run_validation_checks(df, flight_logs, media_contrib, sales_noise,
                          base_demand, lagged_edges=None):
    checks = {}
    checks["spend_variation"] = {
        ch: {"mean": round(float(df[ch].mean()), 2),
             "std": round(float(df[ch].std()), 2),
             "cv": round(float(df[ch].std() / df[ch].mean()), 3),
             "n_flights": len(flight_logs.get(ch, [])),
             "active_weeks": int((df[ch] > df[ch].quantile(0.1) * 1.05).sum())}
        for ch in CHANNELS}

    corr_pairs = [("TV", "Search"), ("Video", "Search"),
                  ("Search", "Sales"), ("TV", "Sales")]
    corr = {f"{a}~{b}": round(float(df[a].corr(df[b])), 3) for a, b in corr_pairs}
    checks["correlations"] = corr
    checks["correlations_positive"] = all(v > 0 for v in corr.values())

    pairs = ([(e["source"], e["target"], e["lag"]) for e in lagged_edges
              if e["target"] != "Sales"] if lagged_edges is not None
             else [("TV", "Search", 1), ("Video", "Search", 1),
                   ("Social", "Search", 1)])
    lag_checks = {}
    for src, dst, lag in pairs:
        c_lag = float(df[src].shift(lag).corr(df[dst]))
        c_syn = float(df[src].corr(df[dst]))
        lag_checks[f"{src}->{dst}"] = {
            "true_lag": lag,
            f"corr_lag{lag}": round(c_lag, 3), "corr_lag0": round(c_syn, 3),
            "lag_dominates": bool(c_lag > c_syn)}
    checks["lag_sanity"] = lag_checks

    v_d, v_m, v_n = base_demand.var(), media_contrib.var(), sales_noise.var()
    v_tot = v_d + v_m + v_n
    checks["variance_decomposition"] = {
        "demand_share": round(float(v_d / v_tot), 3),
        "media_share": round(float(v_m / v_tot), 3),
        "noise_share": round(float(v_n / v_tot), 3)}
    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_dataset(cfg: ScenarioConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    gamma = cfg.gamma()

    base_demand, demand_components = generate_base_demand(cfg, rng)
    _, lagged_edges, g = build_ground_truth_graph(cfg)

    spend, flights_series, flight_logs, noise = {}, {}, {}, {}
    for ch in EXOGENOUS:
        flights_series[ch], flight_logs[ch] = generate_flights(
            cfg, rng, base_demand, **FLIGHT_BEHAVIOR[ch])
        noise[ch] = rng.normal(0.0, CHANNEL_NOISE_SD[ch], cfg.n_weeks)
        spend[ch] = _exogenous_dynamics(ch, cfg, base_demand,
                                        flights_series[ch], noise[ch])

    parents_map, parent_norm = {}, {}
    for ch in ENDOGENOUS:
        flights_series[ch], flight_logs[ch] = generate_flights(
            cfg, rng, base_demand, **FLIGHT_BEHAVIOR[ch])
        noise[ch] = rng.normal(0.0, CHANNEL_NOISE_SD[ch], cfg.n_weeks)
        parents_map[ch] = [p for p in g.predecessors(ch)]
        parents = {p: spend[p] for p in parents_map[ch]}
        parent_norm[ch] = {p: (float(s.mean()), float(max(s.std(), 1e-9)))
                           for p, s in parents.items()}
        spend[ch] = _endogenous_dynamics(ch, cfg, base_demand, parents, gamma,
                                         flights_series[ch], noise[ch],
                                         parent_norm[ch])

    adstock, saturated, hill_k = {}, {}, {}
    for ch in CHANNELS:
        adstock[ch] = apply_adstock(spend[ch], ADSTOCK_THETA[ch])
        saturated[ch], hill_k[ch] = apply_hill_saturation(adstock[ch],
                                                          HILL_ALPHA[ch])

    sales, betas, noise_sd, media_contrib, sales_noise = generate_sales(
        cfg, rng, base_demand, saturated)

    df = pd.DataFrame({"Week": np.arange(cfg.n_weeks)})
    for ch in CHANNELS:
        df[ch] = np.round(spend[ch], 2)
    df["Sales"] = np.round(sales, 2)
    df["Base_Demand"] = np.round(base_demand, 2)
    df["Seasonality"] = np.round(demand_components["Seasonality"], 2)
    df["Trend"] = np.round(demand_components["Trend"], 2)
    df["Event_Effect"] = np.round(demand_components["Event_Effect"], 2)
    for ch in CHANNELS:
        df[f"{ch}_Adstock"] = np.round(adstock[ch], 3)
        df[f"{ch}_Saturated"] = np.round(saturated[ch], 5)

    adj, lagged_edges, g = build_ground_truth_graph(cfg)
    direct_strengths = {}
    for e in lagged_edges:
        s, d = e["source"], e["target"]
        direct_strengths[(s, d)] = betas.get(s, 0.0) if d == "Sales" else gamma
    effects = compute_effects(g, direct_strengths)

    ground_truth = {
        "adjacency_matrix": adj.to_dict(),
        "lagged_edges": lagged_edges,
        "structural_path_effects_on_sales": effects,
        "true_parameters": {
            "betas": {k: round(v, 4) for k, v in betas.items()},
            "gamma": gamma, "adstock_theta": ADSTOCK_THETA,
            "hill_alpha": HILL_ALPHA,
            "hill_k": {k: round(v, 3) for k, v in hill_k.items()},
            "sales_lags": SALES_LAG,
            "sales_noise_sd": round(noise_sd, 3)},
        "scenario_metadata": asdict(cfg)}

    state = {"spend": spend, "flights": flights_series, "noise": noise,
             "base_demand": base_demand, "sales_noise": sales_noise,
             "sales": sales, "betas": betas, "hill_k": hill_k,
             "gamma": gamma, "parents": parents_map,
             "parent_norm": parent_norm}

    result = {"df": df, "ground_truth": ground_truth,
              "flight_logs": flight_logs, "config": cfg, "state": state}
    result["validation"] = run_validation_checks(
        df, flight_logs, media_contrib, sales_noise, base_demand,
        lagged_edges=lagged_edges)
    return result


def scenario_grid(seeds=(42,)):
    for inter in GAMMA_LEVELS:
        for noise in NOISE_LEVELS:
            for _, n_weeks in LENGTH_LEVELS.items():
                for density in ("sparse", "medium", "dense"):
                    for seed in seeds:
                        yield ScenarioConfig(n_weeks=n_weeks,
                                             interdependency=inter,
                                             noise=noise,
                                             dag_density=density, seed=seed)