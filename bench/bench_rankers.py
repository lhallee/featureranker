"""Benchmark featureranker methods across dataset shapes.

Runs under featureranker v2 and v3 alike, so two environments produce
comparable rows for the same scenarios. Repo-only; not shipped in the wheel.

Example:
    python bench/bench_rankers.py --scenario small-cls --n-jobs -1
    python bench/bench_rankers.py --scenario tall-cls --scale 0.1 --out bench.csv
"""

import argparse
import csv
import time

import numpy as np
import pandas as pd

import featureranker

from pathlib import Path
from sklearn.datasets import make_regression

METHOD_KEYS = ["rf", "xg", "mi", "f_test", "l1"]


def build_scenario(name: str, scale: float) -> tuple[pd.DataFrame, pd.Series, str]:
    if name.startswith("small"):
        n, p, k = 500, 50, 10
    elif name.startswith("tall"):
        n, p, k = int(200_000 * scale), 500, 20
    elif name.startswith("wide"):
        n, p, k = 2_000, int(20_000 * scale), 30
    else:
        raise ValueError(f"Unknown scenario {name!r}.")

    task = "classification" if name.endswith("cls") else "regression"
    rng = np.random.default_rng(42)
    if task == "classification":
        y = rng.integers(0, 2, size=n)  # (n,)
        X = rng.normal(size=(n, p))  # (n, p)
        X[:, :k] += 2.0 * y[:, None]
    else:
        X, y = make_regression(
            n_samples=n, n_features=p, n_informative=k, shuffle=False,
            noise=1.0, random_state=42,
        )
    frame = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(p)])
    return frame, pd.Series(y, name="target"), task


def rank(X: pd.DataFrame, y: pd.Series, task: str, methods: list[str], n_jobs: int):
    if featureranker.__version__.startswith("2."):
        return featureranker.feature_ranking(
            X, y, task=task, choices=methods, n_jobs=n_jobs
        )
    return featureranker.feature_ranking(X, y, task=task, methods=methods, n_jobs=n_jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", required=True,
        choices=["small-cls", "small-reg", "tall-cls", "tall-reg", "wide-cls", "wide-reg"],
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--methods", nargs="*", default=METHOD_KEYS)
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="append rows to a CSV")
    args = parser.parse_args()

    X, y, task = build_scenario(args.scenario, args.scale)
    version = featureranker.__version__
    print(f"featureranker {version} | {args.scenario} | X: {X.shape} | n_jobs={args.n_jobs}")

    rows = []
    for method in args.methods:
        started = time.perf_counter()
        rank(X, y, task, [method], args.n_jobs)
        seconds = time.perf_counter() - started
        rows.append((version, args.scenario, X.shape[0], X.shape[1], method, args.n_jobs, round(seconds, 2)))
        print(f"  {method:8s} {seconds:8.2f} s")

    if not args.skip_ensemble:
        started = time.perf_counter()
        rank(X, y, task, list(args.methods), args.n_jobs)
        seconds = time.perf_counter() - started
        rows.append((version, args.scenario, X.shape[0], X.shape[1], "all", args.n_jobs, round(seconds, 2)))
        print(f"  {'all':8s} {seconds:8.2f} s")

    if args.out is not None:
        new_file = not args.out.exists()
        with open(args.out, "a", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["version", "scenario", "n", "p", "method", "n_jobs", "seconds"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()
