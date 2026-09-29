from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
SUMMARY_NAME = "benchmark_summary.csv"
META_KEYS = ["interdependency", "noise", "n_weeks", "dag_density", "seed"]


def _worker(job: dict) -> dict:
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir=/tmp/pytensor_bench_{os.getpid()}"
    import logging as _logging
    _logging.getLogger().addHandler(_logging.NullHandler())
    _logging.getLogger("pymc").setLevel(_logging.ERROR)
    _logging.getLogger("pytensor").setLevel(_logging.ERROR)
    from benchmark_mmm import run_benchmark_on_cell

    cell = job["cell"]
    t0 = time.time()
    meta = {k: job["meta"].get(k) for k in META_KEYS}
    try:
        row = run_benchmark_on_cell(
            cell, draws=job["draws"], tune=job["tune"], chains=job["chains"],
            target_accept=job["target_accept"], seed=job["seed"])
        row["graph_type"] = "benchmark"
        row["status"] = "ok"
        row["runtime_s"] = round(time.time() - t0, 1)
        return {**meta, **row}
    except Exception as exc:
        return {**meta, "cell": Path(cell).name, "graph_type": "benchmark",
                "status": "failed", "error": repr(exc),
                "traceback": traceback.format_exc(),
                "runtime_s": round(time.time() - t0, 1)}


def _cell_metadata(cell_dir: Path) -> dict:
    path = cell_dir / "scenario_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.load(open(path))
    except Exception:
        m = {}
        if "seed_" in cell_dir.name:
            try:
                m["seed"] = int(cell_dir.name.rsplit("seed_", 1)[1])
            except ValueError:
                pass
        return m


def build_jobs(cell_dirs, seeds, args):
    jobs = []
    for cell in cell_dirs:
        meta = _cell_metadata(cell)
        if seeds is not None and meta.get("seed") not in seeds:
            continue
        out_dir = cell / "benchmark_mmm"
        if out_dir.exists() and not args.overwrite:
            continue
        jobs.append({"cell": str(cell), "meta": meta,
                     "draws": args.draws, "tune": args.tune,
                     "chains": args.chains, "target_accept": args.target_accept,
                     "seed": args.seed})
    return jobs


def apply_shard(jobs, shard):
    i, n = (int(x) for x in shard.split("/"))
    return jobs if n <= 1 else [j for k, j in enumerate(jobs) if k % n == i]


def merge_summary(root: Path, rows: list[dict]) -> Path:
    out_path = root / SUMMARY_NAME
    new = pd.DataFrame(rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        if "cell" in old.columns:
            old = old[~old["cell"].isin(new["cell"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default=DEFAULT_GLOB)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Silence library loggers in the parent process; each worker does the same.
    logging.getLogger().addHandler(logging.NullHandler())
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    cell_dirs = sorted(Path(p) for p in glob.glob(args.cells) if Path(p).is_dir())
    if not cell_dirs:
        raise FileNotFoundError(f"No cell directories match '{args.cells}'.")
    seeds = None if args.all_seeds else set(args.seeds)

    jobs = apply_shard(build_jobs(cell_dirs, seeds, args), args.shard)
    if not jobs:
        raise RuntimeError("No jobs to run (all done? wrong seeds? use "
                           "--overwrite or --all-seeds).")

    root = cell_dirs[0].parent

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            rows.append(fut.result())
            merge_summary(root, rows)          # crash-safe: persist each result

    fails = [r for r in rows if r["status"] == "failed"]
    if fails:
        with open(root / "benchmark_parallel_failures.json", "w") as f:
            json.dump(fails, f, indent=2)
        raise SystemExit(1)


if __name__ == "__main__":
    main()