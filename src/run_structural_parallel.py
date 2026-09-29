from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
           "PYTENSOR_FLAGS"):
    os.environ.setdefault(_v, "1" if _v != "PYTENSOR_FLAGS"
                          else "compiledir_format=compiledir_%(short_platform)s")
import argparse
import glob
import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

DEFAULT_GLOB = "exports/full_grid/gamma_*"
DEFAULT_GRAPH_TYPES = ["pcmci_dag", "hybrid", "oracle_hybrid"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
SUMMARY_NAME = "structural_bayesian_mmm_summary.csv"
META_KEYS = ["interdependency", "noise", "n_weeks", "dag_density", "seed"]


# ---------------------------------------------------------------------------
# Worker (top-level function so it is picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker(job: dict) -> dict:
    """Run one (cell, graph_type) fit. Never raises — errors are returned."""
    # per-process compile dir keeps pytensor from serializing on one lock
    os.environ["PYTENSOR_FLAGS"] = (
        f"base_compiledir=/tmp/pytensor_{os.getpid()}")
    import logging as _logging
    _logging.getLogger().addHandler(_logging.NullHandler())
    _logging.getLogger("pymc").setLevel(_logging.ERROR)
    _logging.getLogger("pytensor").setLevel(_logging.ERROR)
    from structural_bayesian_mmm import run_structural_mmm_on_cell

    cell, gt = job["cell"], job["graph_type"]
    t0 = time.time()
    try:
        row = run_structural_mmm_on_cell(
            cell, gt, draws=job["draws"], tune=job["tune"],
            chains=job["chains"], target_accept=job["target_accept"],
            seed=job["seed"], pcmci_setting=job["pcmci_setting"],
            smoke_test=job["smoke_test"])
        row["status"] = "ok"
        return {**job_meta(job), **row}
    except Exception as exc:
        return {**job_meta(job), "cell": Path(cell).name, "graph_type": gt,
                "status": "failed", "error": repr(exc),
                "traceback": traceback.format_exc(),
                "runtime_s": round(time.time() - t0, 1)}


def job_meta(job: dict) -> dict:
    return {k: job["meta"].get(k) for k in META_KEYS}


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def _cell_metadata(cell_dir: Path) -> dict:
    path = cell_dir / "scenario_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_jobs(cell_dirs, graph_types, seeds, args) -> list[dict]:
    """One job per (cell, graph_type), filtered to the requested seeds."""
    jobs = []
    for cell in cell_dirs:
        meta = _cell_metadata(cell)
        if seeds is not None and meta.get("seed") not in seeds:
            continue
        for gt in graph_types:
            out_dir = cell / "structural_bayesian_mmm" / gt
            if out_dir.exists() and not args.overwrite:
                continue
            jobs.append({
                "cell": str(cell), "graph_type": gt, "meta": meta,
                "draws": args.draws, "tune": args.tune, "chains": args.chains,
                "target_accept": args.target_accept, "seed": args.seed,
                "pcmci_setting": args.pcmci_setting,
                "smoke_test": args.smoke_test})
    return jobs


def apply_shard(jobs: list[dict], shard: str) -> list[dict]:
    i, n = (int(x) for x in shard.split("/"))
    if n <= 1:
        return jobs
    return [j for k, j in enumerate(jobs) if k % n == i]


# ---------------------------------------------------------------------------
# Summary merge (crash-safe: rewritten after every completed job)
# ---------------------------------------------------------------------------

def merge_summary(root: Path, rows: list[dict]) -> Path:
    out_path = root / SUMMARY_NAME
    new = pd.DataFrame(rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        if {"cell", "graph_type"}.issubset(old.columns):
            key = new.set_index(["cell", "graph_type"]).index
            old = old[~old.set_index(["cell", "graph_type"]).index.isin(key)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB)
    parser.add_argument("--graph-types", nargs="+", default=DEFAULT_GRAPH_TYPES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="only fit cells with these seeds (default: 1-5); "
                             "pass --all-seeds to disable")
    parser.add_argument("--all-seeds", action="store_true",
                        help="ignore --seeds and fit every cell")
    parser.add_argument("--workers", type=int, default=9,
                        help="parallel fits (default 9 for a 22-vCPU / "
                             "11-core VM, leaving headroom)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed passed to the sampler (not the cell "
                             "selection seed)")
    parser.add_argument("--pcmci-setting", default="controlled",
                        choices=["controlled", "uncontrolled"])
    parser.add_argument("--shard", default="0/1",
                        help="i/n — run every n-th job, offset i (default 0/1)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    # Silence library loggers in the parent process; each worker does the same.
    logging.getLogger().addHandler(logging.NullHandler())
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    cell_dirs = sorted(Path(p) for p in glob.glob(args.cells)
                       if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(f"No cell directories match '{args.cells}'.")
    seeds = None if args.all_seeds else set(args.seeds)

    jobs = build_jobs(cell_dirs, args.graph_types, seeds, args)
    jobs = apply_shard(jobs, args.shard)
    if not jobs:
        raise RuntimeError("No jobs to run (all outputs exist? wrong seeds? "
                           "use --overwrite or --all-seeds).")

    root = cell_dirs[0].parent

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            rows.append(fut.result())
            merge_summary(root, rows)   # crash-safe: persist after every job

    fails = [r for r in rows if r["status"] == "failed"]
    if fails:
        with open(root / "structural_parallel_failures.json", "w") as f:
            json.dump(fails, f, indent=2)
        raise SystemExit(1)


if __name__ == "__main__":
    main()