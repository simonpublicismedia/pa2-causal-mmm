"""
benchmark_mmm.py
================

Correlation-based Bayesian Marketing Mix Model (PyMC) — the BENCHMARK model
for the thesis comparison "causal vs. correlation-based Bayesian MMM".

Model philosophy
----------------
All media channels are assumed to act DIRECTLY on Sales:

    TV, Video, Display, Social, Search, Affiliate  ->  Sales

Channel-to-channel relationships (TV->Search, Display->Social, ...) are
deliberately NOT modeled. Everything else (adstock, Hill saturation,
controls, Bayesian estimation) is identical to the future causal MMM, so
the comparison isolates the value of causal structure modeling.

Mean structure (on scaled data):

    Sales = intercept
          + sum_i  beta_i * Hill_i( Adstock_i( media_i ) )
          + b_seas * Seasonality + b_trend * Trend + b_event * Event_Effect
          + eps

Priors (weakly informative):
    intercept        ~ Normal(0.5, 0.5)         [sales scaled to ~(0,1)]
    beta_i           ~ HalfNormal(0.5)          [non-negative media effects]
    control coefs    ~ Normal(0, 0.5)
    adstock theta_i  ~ Beta(2, 2)               [decay in (0,1)]
    hill alpha_i     ~ Gamma(3, 2)              [slope > 0, mean 1.5]
    hill k_i         ~ Beta(2, 2)               [half-saturation on (0,1) scale]
    sigma            ~ HalfNormal(0.5)

Ground-truth comparison
-----------------------
Estimated channel effects are made comparable to the exported ground truth
(true_interventional_effects.csv) by running the SAME intervention on the
fitted model: spend_i -> factor * spend_i, re-applying adstock + Hill per
posterior draw, and summing the implied sales delta (identical burn-in).
Note: by construction the benchmark holds all OTHER channels fixed under
this intervention — it cannot represent funnel spillovers (e.g. TV-induced
Search spend). The resulting bias for upper-funnel channels is exactly the
phenomenon the thesis comparison is designed to expose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

CHANNELS = ["TV", "Video", "Display", "Social", "Search", "Affiliate"]
CONTROLS = ["Seasonality", "Trend", "Event_Effect"]
TARGET = "Sales"
L_MAX = 12          # adstock convolution window (weeks)
HDI_PROB = 0.94


# --- ArviZ version compatibility (0.x uses hdi_prob, 1.x uses ci_prob) -----

def _hdi(samples: np.ndarray, prob: float = HDI_PROB) -> np.ndarray:
    """
    Highest-density interval, own numpy implementation (shortest interval
    containing `prob` mass) — independent of the installed ArviZ version.
    samples: (S,) -> returns (2,);  (S, K) -> returns (K, 2).
    """
    samples = np.asarray(samples)
    if samples.ndim == 2:
        return np.stack([_hdi(samples[:, j], prob)
                         for j in range(samples.shape[1])])
    x = np.sort(samples)
    n = len(x)
    m = max(int(np.floor(prob * n)), 1)
    widths = x[m:] - x[: n - m]
    j = int(np.argmin(widths))
    return np.array([x[j], x[j + m]])


def _az_summary(idata, var_names, prob: float = HDI_PROB) -> pd.DataFrame:
    try:                                  # ArviZ 0.x
        s = az.summary(idata, var_names=var_names, hdi_prob=prob)
    except TypeError:                     # ArviZ >= 1.0
        s = az.summary(idata, var_names=var_names, ci_prob=prob)
    # ArviZ >= 1.0 returns formatted strings; coerce numerics
    for col in s.columns:
        s[col] = pd.to_numeric(s[col], errors="coerce")
    return s


# ===========================================================================
# 1. Data preparation
# ===========================================================================

def load_cell_data(cell_dir: str | Path) -> pd.DataFrame:
    """Load the controlled observational dataset of one scenario cell."""
    cell_dir = Path(cell_dir)
    df = pd.read_parquet(cell_dir / "observed_controlled.parquet")
    missing = [c for c in CHANNELS + CONTROLS + [TARGET] if c not in df]
    if missing:
        raise ValueError(f"columns missing in {cell_dir}: {missing}")
    return df


def prepare_data(df: pd.DataFrame) -> dict:
    """
    Scale variables for stable NUTS sampling and store all scalers so that
    every downstream quantity can be reported on the original scale.

      media_i : divided by its max            -> [0, 1]
      Sales   : divided by its max            -> (0, 1)
      controls: z-standardized
    """
    media_raw = df[CHANNELS].to_numpy(float)
    media_max = media_raw.max(axis=0)
    media_scaled = media_raw / media_max

    sales_raw = df[TARGET].to_numpy(float)
    sales_scale = sales_raw.max()
    sales_scaled = sales_raw / sales_scale

    ctrl_raw = df[CONTROLS].to_numpy(float)
    ctrl_mean = ctrl_raw.mean(axis=0)
    ctrl_sd = ctrl_raw.std(axis=0)
    ctrl_sd[ctrl_sd < 1e-12] = 1.0
    ctrl_scaled = (ctrl_raw - ctrl_mean) / ctrl_sd

    return {
        "n_obs": len(df),
        "media_raw": media_raw, "media_scaled": media_scaled,
        "media_max": media_max,
        "sales_raw": sales_raw, "sales_scaled": sales_scaled,
        "sales_scale": sales_scale,
        "controls_scaled": ctrl_scaled,
        "lag_cube": build_lag_cube(media_scaled, L_MAX),
    }


def build_lag_cube(media_scaled: np.ndarray, l_max: int) -> np.ndarray:
    """
    Lagged design cube for the adstock convolution.
    Shape (C, T, L): cube[c, t, l] = media_scaled[t - l, c]  (0 before start).
    """
    T, C = media_scaled.shape
    cube = np.zeros((C, T, l_max))
    for l in range(l_max):
        cube[:, l:, l] = media_scaled[: T - l, :].T
    return cube


# ===========================================================================
# 2. Media transformations (symbolic + numpy twin implementations)
# ===========================================================================

def adstock_from_cube(cube, theta, l_max: int):
    """
    Geometric adstock via weighted convolution over the lag cube.
    Weights are normalized (sum to 1) so adstocked spend keeps the spend
    scale: w_l = theta^l / sum_l theta^l.
    Works for pytensor tensors and numpy arrays alike.
    theta shape (C,), cube shape (C, T, L) -> adstock shape (C, T).
    """
    lags = np.arange(l_max)
    if isinstance(theta, np.ndarray):
        w = theta[:, None] ** lags[None, :]
        w = w / w.sum(axis=1, keepdims=True)
        return (cube * w[:, None, :]).sum(axis=-1)
    w = theta[:, None] ** pt.constant(lags)[None, :]
    w = w / w.sum(axis=1, keepdims=True)
    return (cube * w[:, None, :]).sum(axis=-1)


def hill(x, alpha, k):
    """Hill saturation x^a / (x^a + k^a); x >= 0, alpha > 0, k > 0."""
    if isinstance(x, np.ndarray):
        xa = np.power(np.maximum(x, 1e-12), alpha)
        return xa / (xa + np.power(k, alpha))
    xa = pt.maximum(x, 1e-12) ** alpha
    return xa / (xa + k ** alpha)


# ===========================================================================
# 3. Model specification
# ===========================================================================

def build_model(data: dict) -> pm.Model:
    coords = {"channel": CHANNELS, "control": CONTROLS,
              "obs": np.arange(data["n_obs"])}
    with pm.Model(coords=coords) as model:
        cube = pm.Data("lag_cube", data["lag_cube"])
        ctrl = pm.Data("controls", data["controls_scaled"])
        y = pm.Data("sales_scaled", data["sales_scaled"])

        # --- media transformation parameters (channel-specific) -----------
        theta = pm.Beta("adstock_theta", 2.0, 2.0, dims="channel")
        alpha = pm.Gamma("hill_alpha", 3.0, 2.0, dims="channel")
        k = pm.Beta("hill_k", 2.0, 2.0, dims="channel")

        adstocked = adstock_from_cube(cube, theta, L_MAX)      # (C, T)
        saturated = hill(adstocked.T, alpha, k)                # (T, C)

        # --- coefficients --------------------------------------------------
        intercept = pm.Normal("intercept", 0.5, 0.5)
        beta = pm.HalfNormal("beta_media", 0.5, dims="channel")
        b_ctrl = pm.Normal("beta_control", 0.0, 0.5, dims="control")
        sigma = pm.HalfNormal("sigma", 0.5)

        media_effect = pm.Deterministic(
            "media_contrib_scaled", saturated * beta, dims=("obs", "channel"))
        mu = (intercept
              + media_effect.sum(axis=-1)
              + pt.dot(ctrl, b_ctrl))
        pm.Deterministic("mu", mu, dims="obs")
        pm.Normal("sales_obs", mu=mu, sigma=sigma, observed=y, dims="obs")
    return model


# ===========================================================================
# 4. Inference + diagnostics
# ===========================================================================

def fit(model: pm.Model, draws=1000, tune=1000, chains=4,
        target_accept=0.9, seed=42, cores=None) -> az.InferenceData:
    if cores is None:
        import os
        cores = max(1, min(chains, os.cpu_count() or 1))
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=False)
    return idata


PARAM_VARS = ["intercept", "beta_media", "beta_control",
              "adstock_theta", "hill_alpha", "hill_k", "sigma"]


def diagnostics(idata: az.InferenceData, out_dir: Path | None = None) -> pd.DataFrame:
    """Posterior summary (mean, HDI, R-hat, ESS) + trace plots."""
    summary = _az_summary(idata, PARAM_VARS)
    if out_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        summary.to_csv(out_dir / "posterior_summary.csv")
        try:
            az.plot_trace(idata, var_names=PARAM_VARS, compact=True)
        except (TypeError, ValueError):          # ArviZ >= 1.0
            az.plot_trace(idata, var_names=PARAM_VARS)
        plt.gcf().suptitle("Benchmark MMM — trace plots", y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / "trace_plots.png", dpi=130, bbox_inches="tight")
        plt.close("all")
    return summary


def convergence_flags(summary: pd.DataFrame) -> dict:
    return {"max_rhat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
            "converged": bool((summary["r_hat"] < 1.01).all()
                              and (summary["ess_bulk"] > 400).all())}


# ===========================================================================
# 5. Posterior-based quantities
# ===========================================================================

def _posterior_draws(idata, max_draws=500, seed=0):
    """Stacked posterior draws of all needed parameters, thinned."""
    post = idata.posterior
    if hasattr(post, "dataset"):          # ArviZ >= 1.0 (DataTree)
        post = post.dataset
    post = post.stack(sample=("chain", "draw"))
    n = post.sizes["sample"]
    rng = np.random.default_rng(seed)
    idx = (np.arange(n) if n <= max_draws
           else np.sort(rng.choice(n, max_draws, replace=False)))
    def get(name):
        return np.moveaxis(post[name].values, -1, 0)[idx]  # (S, ...)
    return {name: get(name) for name in
            ["beta_media", "adstock_theta", "hill_alpha", "hill_k",
             "intercept", "beta_control"]}


def _media_contrib_scaled(draws, cube):
    """Per-draw media contributions, shape (S, T, C)."""
    S = draws["beta_media"].shape[0]
    T = cube.shape[1]
    C = cube.shape[0]
    out = np.empty((S, T, C))
    for s in range(S):
        ad = adstock_from_cube(cube, draws["adstock_theta"][s], L_MAX)  # (C,T)
        sat = hill(ad.T, draws["hill_alpha"][s], draws["hill_k"][s])    # (T,C)
        out[s] = sat * draws["beta_media"][s]
    return out


def estimate_interventional_effects(idata, data, factor=1.10, burn_in=5,
                                    max_draws=500) -> pd.DataFrame:
    """
    Model-implied analogue of the ground-truth intervention:
    spend_i -> factor * spend_i (other channels fixed), delta sales summed
    over weeks after burn_in, on the ORIGINAL sales scale.
    Returns posterior mean, sd and HDI of delta-sales and marginal ROAS.
    """
    draws = _posterior_draws(idata, max_draws=max_draws)
    base = _media_contrib_scaled(draws, data["lag_cube"])           # (S,T,C)
    sl = slice(burn_in, None)
    rows = []
    for ci, ch in enumerate(CHANNELS):
        scaled = data["media_scaled"].copy()
        scaled[:, ci] *= factor
        cube_cf = build_lag_cube(scaled, L_MAX)
        cf = _media_contrib_scaled(draws, cube_cf)                  # (S,T,C)
        d_sales = (cf[:, sl, :].sum((1, 2)) - base[:, sl, :].sum((1, 2))) \
            * data["sales_scale"]                                   # (S,)
        d_spend = (factor - 1.0) * data["media_raw"][sl, ci].sum()
        roas = d_sales / d_spend
        hdi_s = _hdi(d_sales)
        hdi_r = _hdi(roas)
        rows.append({"channel": ch, "factor": factor, "burn_in_weeks": burn_in,
                     "delta_sales_total_mean": d_sales.mean(),
                     "delta_sales_total_sd": d_sales.std(),
                     "delta_sales_hdi_low": hdi_s[0],
                     "delta_sales_hdi_high": hdi_s[1],
                     "delta_own_spend": d_spend,
                     "marginal_roas_mean": roas.mean(),
                     "marginal_roas_hdi_low": hdi_r[0],
                     "marginal_roas_hdi_high": hdi_r[1]})
    return pd.DataFrame(rows)


def attribution(idata, data, max_draws=500) -> pd.DataFrame:
    """
    Contribution decomposition (posterior mean, original sales scale):
    media channels + controls + intercept baseline, absolute and % of the
    total modeled sales sum.
    """
    draws = _posterior_draws(idata, max_draws=max_draws)
    media = _media_contrib_scaled(draws, data["lag_cube"])          # (S,T,C)
    media_total = media.sum(1).mean(0) * data["sales_scale"]        # (C,)

    ctrl_contrib = np.einsum("tk,sk->st", data["controls_scaled"],
                             draws["beta_control"])                 # (S,T)
    # per-control decomposition
    ctrl_each = (data["controls_scaled"][None, :, :]
                 * draws["beta_control"][:, None, :])               # (S,T,K)
    ctrl_total = ctrl_each.sum(1).mean(0) * data["sales_scale"]     # (K,)

    base_total = (draws["intercept"].mean() * data["n_obs"]
                  * data["sales_scale"])

    rows = ([{"component": ch, "contribution_abs": media_total[i]}
             for i, ch in enumerate(CHANNELS)]
            + [{"component": c, "contribution_abs": ctrl_total[i]}
               for i, c in enumerate(CONTROLS)]
            + [{"component": "Baseline_Intercept",
                "contribution_abs": base_total}])
    out = pd.DataFrame(rows)
    total = out["contribution_abs"].sum()
    out["contribution_pct"] = out["contribution_abs"] / total * 100
    out["mean_weekly"] = out["contribution_abs"] / data["n_obs"]
    return out


def model_fit(idata, data, max_draws=500) -> dict:
    """
    In-sample fit metrics.

    Bayesian R^2 (Gelman et al. 2019): per posterior draw s,
        R2_s = Var(mu_s) / (Var(mu_s) + Var(y - mu_s)),
    reported as posterior mean with HDI. Additionally the classic R^2 of
    the posterior-mean prediction, plus RMSE / MAE on the original scale.

    Note: with a target noise share of q, the achievable R^2 is bounded
    near 1 - q by construction — and a high R^2 does NOT imply unbiased
    interventional effects (see ground-truth comparison).
    """
    draws = _posterior_draws(idata, max_draws=max_draws)
    media = _media_contrib_scaled(draws, data["lag_cube"])          # (S,T,C)
    mu = (draws["intercept"][:, None]
          + media.sum(-1)
          + np.einsum("tk,sk->st", data["controls_scaled"],
                      draws["beta_control"]))                       # (S,T)
    y = data["sales_scaled"][None, :]
    resid = y - mu
    r2_draws = mu.var(axis=1) / (mu.var(axis=1) + resid.var(axis=1))
    r2_hdi = _hdi(r2_draws)

    mu_mean = mu.mean(0)
    ss_res = ((data["sales_scaled"] - mu_mean) ** 2).sum()
    ss_tot = ((data["sales_scaled"] - data["sales_scaled"].mean()) ** 2).sum()
    r2_classic = 1.0 - ss_res / ss_tot

    resid_raw = (data["sales_scaled"] - mu_mean) * data["sales_scale"]
    return {
        "bayes_r2_mean": float(r2_draws.mean()),
        "bayes_r2_hdi_low": float(r2_hdi[0]),
        "bayes_r2_hdi_high": float(r2_hdi[1]),
        "r2_posterior_mean_pred": float(r2_classic),
        "rmse_sales": float(np.sqrt((resid_raw ** 2).mean())),
        "mae_sales": float(np.abs(resid_raw).mean()),
    }


def average_roas(idata, data, max_draws=500) -> pd.DataFrame:
    """Average ROAS = total modeled contribution / total spend, per channel."""
    draws = _posterior_draws(idata, max_draws=max_draws)
    media = _media_contrib_scaled(draws, data["lag_cube"])          # (S,T,C)
    contrib = media.sum(1) * data["sales_scale"]                    # (S,C)
    spend = data["media_raw"].sum(0)                                # (C,)
    roas = contrib / spend
    hdi = _hdi(roas)
    return pd.DataFrame({
        "channel": CHANNELS,
        "avg_roas_mean": roas.mean(0),
        "avg_roas_sd": roas.std(0),
        "avg_roas_hdi_low": hdi[:, 0],
        "avg_roas_hdi_high": hdi[:, 1]})


# ===========================================================================
# 6. Ground-truth comparison
# ===========================================================================

PRIORS_SPEC = {
    "intercept": "Normal(0.5, 0.5)  [on sales/max scale]",
    "beta_media": "HalfNormal(0.5) per channel",
    "beta_control": "Normal(0, 0.5) per control",
    "adstock_theta": "Beta(2, 2) per channel",
    "hill_alpha": "Gamma(3, 2) per channel",
    "hill_k": "Beta(2, 2) per channel  [on adstocked spend/max scale]",
    "sigma": "HalfNormal(0.5)",
}


def compare_to_ground_truth(effects: pd.DataFrame,
                            cell_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Compare model-implied interventional effects against the exported
    ground truth (true_interventional_effects.csv) using BOTH references:

      * true_marginal_roas_own   = delta Sales / delta OWN spend
      * true_marginal_roas_total = delta Sales / delta TOTAL media spend
                                   (incl. downstream spend induced via the
                                   funnel, which the benchmark cannot model)

    The estimated marginal ROAS holds all other channels fixed, so its
    natural counterpart on the spend side is "own"; the "total" reference
    quantifies the gap to the full business effect. All quantities are on
    the ORIGINAL sales / spend scale.
    """
    truth = pd.read_csv(Path(cell_dir) / "true_interventional_effects.csv")
    merged = effects.merge(
        truth[["channel", "delta_sales_total", "marginal_roas_own",
               "marginal_roas_total"]],
        on="channel", suffixes=("", "_truthfile"))

    out = pd.DataFrame({
        "channel": merged["channel"],
        # --- estimates (original scale) --------------------------------
        "estimated_effect": merged["delta_sales_total_mean"],
        "estimated_effect_hdi_low": merged["delta_sales_hdi_low"],
        "estimated_effect_hdi_high": merged["delta_sales_hdi_high"],
        "estimated_roas": merged["marginal_roas_mean"],
        "estimated_roas_hdi_low": merged["marginal_roas_hdi_low"],
        "estimated_roas_hdi_high": merged["marginal_roas_hdi_high"],
        # --- ground truth ----------------------------------------------
        "true_delta_sales_total": merged["delta_sales_total"],
        "true_marginal_roas_own": merged["marginal_roas_own"],
        "true_marginal_roas_total": merged["marginal_roas_total"],
    })
    # --- effect-level errors (single truth: total sales delta) --------
    out["bias_effect"] = out["estimated_effect"] - out["true_delta_sales_total"]
    out["hdi_covers_effect"] = (
        (out["true_delta_sales_total"] >= out["estimated_effect_hdi_low"])
        & (out["true_delta_sales_total"] <= out["estimated_effect_hdi_high"]))
    # --- ROAS errors vs BOTH references --------------------------------
    for ref in ["own", "total"]:
        t = out[f"true_marginal_roas_{ref}"]
        out[f"bias_vs_{ref}"] = out["estimated_roas"] - t
        out[f"hdi_covers_{ref}"] = ((t >= out["estimated_roas_hdi_low"])
                                    & (t <= out["estimated_roas_hdi_high"]))

    def _mae(x):
        return float(np.abs(x).mean())

    def _rmse(x):
        return float(np.sqrt((x ** 2).mean()))

    agg = {
        "mean_bias_effect": float(out["bias_effect"].mean()),
        "mae_effect": _mae(out["bias_effect"]),
        "rmse_effect": _rmse(out["bias_effect"]),
        "hdi_coverage_effect": float(out["hdi_covers_effect"].mean()),
        "mean_bias_vs_own": float(out["bias_vs_own"].mean()),
        "mae_vs_own": _mae(out["bias_vs_own"]),
        "rmse_vs_own": _rmse(out["bias_vs_own"]),
        "hdi_coverage_vs_own": float(out["hdi_covers_own"].mean()),
        "mean_bias_vs_total": float(out["bias_vs_total"].mean()),
        "mae_vs_total": _mae(out["bias_vs_total"]),
        "rmse_vs_total": _rmse(out["bias_vs_total"]),
        "hdi_coverage_vs_total": float(out["hdi_covers_total"].mean()),
    }
    return out, agg


def _group_dataset(idata, group: str):
    """ArviZ 0.x (Dataset) / 1.x (DataTree) compatible group access."""
    g = getattr(idata, group)
    return g.dataset if hasattr(g, "dataset") else g


def diagnostics_report(idata, model, data, summary: pd.DataFrame,
                       sampler_cfg: dict, idata_saved: bool,
                       out_dir: Path, seed: int = 1) -> dict:
    """
    Per-cell diagnostics_report.json: convergence, sampler settings,
    priors, posterior predictive check summary, idata persistence status.
    """
    # --- sampler statistics -------------------------------------------
    ss = _group_dataset(idata, "sample_stats")
    divergences = (int(np.asarray(ss["diverging"]).sum())
                   if "diverging" in ss else None)
    acc_name = next((n for n in ("acceptance_rate", "mean_tree_accept",
                                 "accept") if n in ss), None)
    acceptance = (float(np.asarray(ss[acc_name]).mean())
                  if acc_name else None)
    post = _group_dataset(idata, "posterior")
    n_chains = int(post.sizes["chain"])
    n_draws = int(post.sizes["draw"])

    # --- posterior predictive check -----------------------------------
    with model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["sales_obs"], progressbar=False,
            random_seed=seed)
    y_rep = np.asarray(_group_dataset(ppc, "posterior_predictive")
                       ["sales_obs"])
    y_rep = y_rep.reshape(-1, y_rep.shape[-1]) * data["sales_scale"]
    y_obs = data["sales_raw"]
    rep_means = y_rep.mean(axis=1)
    rep_sds = y_rep.std(axis=1)
    lo = np.quantile(y_rep, 0.03, axis=0)
    hi = np.quantile(y_rep, 0.97, axis=0)
    ppc_summary = {
        "obs_mean": float(y_obs.mean()),
        "rep_mean": float(rep_means.mean()),
        "bayes_p_mean": float((rep_means >= y_obs.mean()).mean()),
        "obs_sd": float(y_obs.std()),
        "rep_sd": float(rep_sds.mean()),
        "bayes_p_sd": float((rep_sds >= y_obs.std()).mean()),
        "coverage_94_interval": float(((y_obs >= lo) & (y_obs <= hi)).mean()),
    }

    report = {
        "r_hat_max": float(summary["r_hat"].max()),
        "r_hat_median": float(summary["r_hat"].median()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "divergences": divergences,
        "acceptance_rate_mean": acceptance,
        "n_chains": n_chains,
        "n_draws_per_chain": n_draws,
        "sampler_settings": sampler_cfg,
        "priors": PRIORS_SPEC,
        "posterior_predictive_check": ppc_summary,
        "idata_saved": bool(idata_saved),
        "versions": {"pymc": pm.__version__, "arviz": az.__version__},
    }
    with open(out_dir / "diagnostics_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report



# ===========================================================================
# 7. Full per-cell pipeline
# ===========================================================================

def run_benchmark_on_cell(cell_dir: str | Path, draws=1000, tune=1000,
                          chains=4, target_accept=0.99, seed=42,
                          max_posterior_draws=500) -> dict:
    """Fit the benchmark MMM on one exported scenario cell and write all
    outputs into <cell_dir>/benchmark_mmm/."""
    cell_dir = Path(cell_dir)
    out_dir = cell_dir / "benchmark_mmm"
    out_dir.mkdir(exist_ok=True)

    df = load_cell_data(cell_dir)
    data = prepare_data(df)
    model = build_model(data)
    idata = fit(model, draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, seed=seed)

    summary = diagnostics(idata, out_dir)
    conv = convergence_flags(summary)
    try:
        _ss = idata.sample_stats
        conv["divergences"] = int(np.asarray(_ss["diverging"]).sum())
    except Exception:
        conv["divergences"] = None

    effects = estimate_interventional_effects(
        idata, data, max_draws=max_posterior_draws)
    effects.to_csv(out_dir / "estimated_effects.csv", index=False)

    attr = attribution(idata, data, max_draws=max_posterior_draws)
    attr.to_csv(out_dir / "attribution.csv", index=False)

    roas = average_roas(idata, data, max_draws=max_posterior_draws)
    roas.to_csv(out_dir / "roas.csv", index=False)

    fit_metrics = model_fit(idata, data, max_draws=max_posterior_draws)
    with open(out_dir / "model_fit.json", "w") as f:
        json.dump(fit_metrics, f, indent=2)

    comparison, agg = compare_to_ground_truth(effects, cell_dir)
    comparison.to_csv(out_dir / "ground_truth_comparison.csv", index=False)
    with open(out_dir / "comparison_aggregates.json", "w") as f:
        json.dump({**agg, **conv}, f, indent=2)

    idata_saved = False  # disabled on VM to save disk space (idata.nc omitted)

    diagnostics_report(
        idata, model, data, summary,
        sampler_cfg={"draws": draws, "tune": tune, "chains": chains,
                     "target_accept": target_accept, "seed": seed},
        idata_saved=idata_saved, out_dir=out_dir)
    return {"cell": cell_dir.name, **conv, **fit_metrics, **agg}