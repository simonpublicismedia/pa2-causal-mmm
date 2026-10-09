# Causal Discovery and Bayesian Estimation in Marketing Mix Modeling

<p align="center">
  <img src="docs/DHBW-Logo.svg.webp" alt="DHBW Ravensburg" height="55">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/Publicis-Groupe-Logo.svg.webp" alt="Publicis Groupe" height="55">
</p>

> **Note for the reviewer.** This repository is public for the duration of the
> assessment and will be set to private once the assessment is complete. The thesis
> carries a confidentiality notice (Sperrvermerk). The repository itself contains only
> synthetic data and the estimation code, no client or company data.

Code and aggregated results for the Projektarbeit II *"Causal Discovery and Bayesian
Estimation in Marketing Mix Modeling: A Simulation Study on When Structural Information
Pays Off"* (DHBW Ravensburg, 2026).

The study compares four Bayesian Marketing Mix Models, Flat, Discovered, Hybrid and
Oracle, on synthetic data with a known causal ground truth, across a grid of 405
scenario cells spanning interdependence strength, noise level, time-series length and
graph density.

> **Confidentiality.** This repository accompanies a thesis carrying a confidentiality
> notice (Sperrvermerk). After the assessment it is set to private. Access is then
> granted on request.

## Structure

```
src/         estimation pipeline — 20 Python modules, see table below
datasets/    aggregated result files underlying the tables and figures of the thesis
```

Estimation logic and command-line execution are kept in separate modules throughout.
the engines contain no file handling beyond their own exports, and the runners contain
no modelling logic. Every `run_*.py` supports `--help`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: PyMC 6.2.0, ArviZ 1.2.0, tigramite 5.2.10.1, networkx, pandas, numpy.

## Pipeline

The five stages run in order; each consumes the output of the previous one and writes
into the directory of the scenario cell it processes.

| Stage | Modules | Purpose |
|---|---|---|
| 1 — Data generation | `mmm_datagen`, `export_mmm_scenarios` | Data-generating process with known ground truth: adstock, Hill saturation, demand-driven campaign flighting, lagged inter-channel effects. Writes one directory per scenario cell. |
| 2 — Causal discovery | `pcmci_grid_search`, `run_pcmci_grid_search`, `pcmci_plus_pipeline`, `run_pcmci_plus` | Hyperparameter search on seven tuning cells, then PCMCI+ discovery and graph-recovery evaluation across the grid. |
| 3 — Graph construction | `hybrid_dag_builder`, `run_hybrid_dag_builder`, `oracle_hybrid_builder`, `run_oracle_hybrid_builder` | Hybrid graph (channel-to-sales edges imposed as domain knowledge, channel-to-channel edges discovered) and Oracle graph (true interdependencies). |
| 4 — Bayesian models | `benchmark_mmm`, `structural_bayesian_mmm` and their four runners | Flat MMM and a single DAG-aware engine for all three causal variants. The structural engine imports the benchmark's transformations, so the sales equation is structurally identical across models. |
| 5 — Evaluation | `true_direct_indirect`, `direct_indirect_plugin`, `fix_vs_total_metric`, `build_corrected_pivot` | Decomposition into direct and indirect effects on both the ground-truth and the model side, and the merged table on which the evaluation rests. |

The hyperparameter search selects one globally robust configuration rather than
optimising per cell, which would tune on the ground truth of every cell. The selected
configuration (RobustParCorr, τ_max = 2, α_pc = 0.075) is fixed for the production run.

## What is included

`datasets/` holds the aggregated result files from which all reported tables and figures
are derived, together with the scenario manifest and the hyperparameter search protocol.

The synthetic datasets themselves are **not** included. They are fully reproducible from
`mmm_datagen.py` together with the scenario coordinates and seeds encoded in each cell
directory name. Regenerating the grid is deterministic (and takes some time!).

## Separation of ground truth

No component of the estimation pipeline receives the ground-truth graph as an argument.
Discovery reads only the observational files, which `export_mmm_scenarios.py` writes
separately from the ground truth, and the true graph is loaded exclusively in the
evaluation step, after the discovered structure has been written to disk. This
separation is enforced by the file layout and by the control flow of the pipeline, not
merely by convention.

The one exception is the Oracle variant, which is defined by construction as the upper
bound of the comparison and therefore reads the true interdependencies deliberately.

## Validation

Three checks are built into the post-hoc analysis rather than applied afterwards:

- `true_direct_indirect` reconstructs adstock and saturation from raw spend and compares
  them against the stored transformed series; it also verifies that a channel without
  outgoing channel-to-channel edges has a direct effect equal to its stored total.
  A violation of either aborts the computation.
- `direct_indirect_plugin` reproduces the induced downstream spend, which must match
  exactly because the channel equations are linear in the structural shift.
- `fix_vs_total_metric` recomputes the originally stored metric locally before changing
  anything, so that agreement establishes the recomputation reproduces the pipeline.

A denominator inconsistency in an exploratory vs-total metric was found this way. It is
documented in Appendix B of the thesis and the metric is not reported in the study.
