"""Selection-vs-reduction ablation harness over the example registry.

Every example runs one protocol: stratified 80/20 split, one full-ensemble
ranking pass on the training split (which yields all five per-method
rankings plus the vote for free), then probes on the held-out 20% for every
representation at every dimension budget:

- selectors: top-k raw features by each ranking method and by the ensemble vote
- reductions: PCA, FastICA, Gaussian random projection, kernel PCA (RBF),
  Isomap, UMAP, and t-SNE, each fit on the training split at the same k
- probes: standardized logistic regression (ridge for regression), kNN, and
  RBF SVM, capturing linear, local, and kernel separability

Budgets anchor at x, the 99%-variance PCA component count: x, x/2, x/4,
10, 1. Probe fits run in parallel. Each run writes its docs page plus plots
and appends rows to docs/examples/results.csv for write_report.py.

    python examples/selection_vs_reduction.py --name digits_pixels
    python examples/selection_vs_reduction.py --all --artifacts ~/fr_artifacts2
"""

import argparse
import csv
import time

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from _pages import PAGES, image_path, md_table, save_page
from featureranker import feature_ranking, plot_after_vote, plot_rankings, voting
from registry import ExampleSpec, SPECS, load_dataset

RANDOM_STATE = 42
RESULTS = PAGES / "results.csv"
RESULT_COLUMNS = [
    "example", "domain", "task", "n", "p", "n_classes", "pca_99", "k",
    "family", "representation", "probe", "score", "fit_seconds",
]
EIGEN_MAX_K = 100  # UMAP, kernel PCA, Isomap, FastICA
TSNE_MAX_K = 10
PROBES = ("linear", "knn", "svm")


def _k_schedule(E_train: np.ndarray) -> tuple[int, list[int]]:
    """Budgets anchored at the 99%-variance PCA dimensionality x."""
    from sklearn.decomposition import PCA

    p = E_train.shape[1]
    x = int(PCA(n_components=0.99, random_state=RANDOM_STATE).fit(E_train).n_components_)
    candidates = {x, x // 2, x // 4, 10, 1}
    ks = sorted({k for k in candidates if 1 <= k <= p}, reverse=True)
    return x, ks


def _probe_model(task: str, probe: str):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR

    if task == "classification":
        return {
            "linear": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "knn": KNeighborsClassifier(n_neighbors=10),
            "svm": SVC(kernel="rbf", cache_size=500, random_state=RANDOM_STATE),
        }[probe]
    return {
        "linear": Ridge(random_state=RANDOM_STATE),
        "knn": KNeighborsRegressor(n_neighbors=10),
        "svm": SVR(kernel="rbf", cache_size=500),
    }[probe]


def _run_probe(task, probe, X_train, X_test, y_train, y_test) -> float:
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    model = _probe_model(task, probe)
    model.fit(scaler.transform(X_train), y_train)
    predictions = model.predict(scaler.transform(X_test))  # (n_te,)
    metric = accuracy_score if task == "classification" else r2_score
    return float(metric(y_test, predictions))


def _reductions(E, train_idx, test_idx, k: int):
    """Yield (label, train matrix, test matrix, fit seconds) per reduction."""
    from sklearn.decomposition import PCA, FastICA, KernelPCA
    from sklearn.manifold import TSNE, Isomap
    from sklearn.random_projection import GaussianRandomProjection

    E_train, E_test = E[train_idx], E[test_idx]  # (n_tr, p), (n_te, p)

    def timed(label, fit_transform):
        started = time.perf_counter()
        train, test = fit_transform()
        return label, train, test, time.perf_counter() - started

    def fitted(reducer):
        R = reducer.fit(E_train)
        return R.transform(E_train), R.transform(E_test)

    yield timed("PCA", lambda: fitted(PCA(n_components=k, random_state=RANDOM_STATE)))
    if k < min(E_train.shape):
        yield timed("RandProj", lambda: fitted(
            GaussianRandomProjection(n_components=k, random_state=RANDOM_STATE)
        ))
    if k <= EIGEN_MAX_K:
        yield timed("ICA", lambda: fitted(
            FastICA(n_components=k, random_state=RANDOM_STATE, max_iter=500, tol=1e-3)
        ))
        yield timed("KernelPCA", lambda: fitted(
            KernelPCA(n_components=k, kernel="rbf", random_state=RANDOM_STATE, n_jobs=-1)
        ))
        yield timed("Isomap", lambda: fitted(Isomap(n_components=k, n_jobs=-1)))
        try:
            import umap

            yield timed("UMAP", lambda: fitted(
                umap.UMAP(n_components=k, random_state=RANDOM_STATE)
            ))
        except ImportError:
            pass
    if k <= TSNE_MAX_K:
        # no out-of-sample transform: fit on all rows label-blind, split after
        def tsne():
            Z = TSNE(
                n_components=k, method="barnes_hut" if k < 4 else "exact",
                init="random", max_iter=300, random_state=RANDOM_STATE,
            ).fit_transform(E)  # (n, k)
            return Z[train_idx], Z[test_idx]

        yield timed("t-SNE", tsne)


def run_example(spec: ExampleSpec, X, y) -> None:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    named = isinstance(X, pd.DataFrame)
    E = X.to_numpy(dtype=np.float32) if named else np.asarray(X, dtype=np.float32)  # (n, p)
    y = np.asarray(y)  # (n,)
    n, p = E.shape
    n_classes = len(np.unique(y)) if spec.task == "classification" else 0
    stratify = y if spec.task == "classification" else None
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, stratify=stratify, random_state=RANDOM_STATE
    )
    y_train, y_test = y[train_idx], y[test_idx]  # (n_tr,), (n_te,)

    # one standardized space (train-fit) for the budget anchor and every
    # reduction, so scale-dominated raw features neither shrink x nor
    # handicap the distance-based reducers; selection slices raw features
    # and the probes standardize every representation themselves
    E_std = StandardScaler().fit(E[train_idx]).transform(E).astype(np.float32)  # (n, p)

    started = time.perf_counter()
    result = feature_ranking(
        X.iloc[train_idx] if named else E[train_idx], y_train, task=spec.task
    )
    ranking_seconds = time.perf_counter() - started
    vote_table = voting(result)

    plot_rankings(
        result, top_n=20, show=False, save=True,
        save_path=image_path(f"{spec.name}_rankings.png"),
    )
    plot_after_vote(
        vote_table, top_n=20, show=False, save=True,
        save_path=image_path(f"{spec.name}_vote.png"),
    )

    pca_99, ks = _k_schedule(E_std[train_idx])
    name_to_column = {name: i for i, name in enumerate(result.feature_names)}
    selector_tables = {"vote": vote_table} | dict(result.rankings)

    # representation -> (family, k, train, test, fit seconds)
    representations = [("all", "all features", p, E[train_idx], E[test_idx], 0.0)]
    for k in ks:
        for selector, table in selector_tables.items():
            top = [name_to_column[name] for name in table["feature"].head(k)]  # (k,)
            representations.append(
                ("select", f"select:{selector}", k,
                 E[train_idx][:, top], E[test_idx][:, top], ranking_seconds)
            )
        for label, R_train, R_test, seconds in _reductions(E_std, train_idx, test_idx, k):
            representations.append(("dr", f"dr:{label}", k, R_train, R_test, seconds))

    jobs = [
        (family, label, k, probe, train, test, seconds)
        for family, label, k, train, test, seconds in representations
        for probe in PROBES
    ]
    scores = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_run_probe)(spec.task, probe, train, test, y_train, y_test)
        for _, _, _, probe, train, test, _ in jobs
    )
    rows = [
        (family, label, k, probe, score, seconds)
        for (family, label, k, probe, _, _, seconds), score in zip(jobs, scores)
    ]

    _append_results(spec, n, p, n_classes, pca_99, rows)
    _write_page(spec, vote_table, rows, n, p, pca_99, ranking_seconds)
    print(f"{spec.name}: n={n} p={p} pca99={pca_99} ks={ks} "
          f"({len(rows)} probe scores) done")


def _append_results(spec, n, p, n_classes, pca_99, rows) -> None:
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(RESULT_COLUMNS)
        for family, label, k, probe, score, seconds in rows:
            writer.writerow([
                spec.name, spec.domain, spec.task, n, p, n_classes, pca_99, k,
                family, label, probe, round(score, 4), round(seconds, 2),
            ])


def _write_page(spec, vote_table, rows, n, p, pca_99, ranking_seconds) -> None:
    metric = "accuracy" if spec.task == "classification" else "R2"
    table = pd.DataFrame(rows, columns=["family", "representation", "k", "probe", "score", "seconds"])
    pivot = (
        table.pivot_table(index=["k", "representation"], columns="probe", values="score")
        .reindex(columns=list(PROBES))
        .round(4)
        .reset_index()
        .sort_values(["k", "representation"], ascending=[False, True])
    )
    page = f"""# {spec.title}

{spec.blurb}

Data: {spec.source}. {n:,} samples, {p:,} features. One five-method
ensemble ranking of the training split took {ranking_seconds:.0f} s and
provides every selector below; reductions fit on the same split. Three
standardized probes ({metric}: linear, kNN, RBF SVM) score the held-out
20%. Budgets anchor at the 99%-variance PCA count x = {pca_99}, stepping
x, x/2, x/4, 10, 1.

## Consensus ranking

{md_table(vote_table, 10)}

![Aggregated importance](../images/{spec.name}_vote.png)

![Ranks by method](../images/{spec.name}_rankings.png)

## Ablation: selectors vs reductions ({metric})

`select:<method>` keeps that method's top-k raw features
(`select:vote` is the ensemble); `dr:<method>` is a fitted reduction to
the same k.

{md_table(pivot, len(pivot))}

Notes: t-SNE has no out-of-sample transform, so it is fit on all rows
(label-blind) and split afterward, which favors it mildly; UMAP, kernel
PCA, Isomap, and ICA run at budgets up to {EIGEN_MAX_K} and t-SNE up to
{TSNE_MAX_K}, beyond which their cost is impractical, while selection has
no budget ceiling. Cross-dataset findings: [the research report](report.md).

Regenerate with `python examples/selection_vs_reduction.py --name {spec.name}`.
"""
    save_page(spec.name, page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="one example name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"),
                        help="root folder with extracted embedding sets")
    parser.add_argument("--fresh", action="store_true",
                        help="delete results.csv before running")
    args = parser.parse_args()

    if args.fresh and RESULTS.exists():
        RESULTS.unlink()

    chosen = list(SPECS) if args.all else [args.name]
    failures = []
    for name in chosen:
        if name not in SPECS:
            raise SystemExit(f"Unknown example {name!r}. Valid: {sorted(SPECS)}")
        try:
            X, y = load_dataset(name, args.artifacts)
            run_example(SPECS[name], X, y)
        except Exception as error:
            # a batch over 20 downloads and fits must survive one bad example
            failures.append(name)
            print(f"FAILED {name}: {type(error).__name__}: {error}")
    if failures:
        print("failed examples:", ", ".join(failures))


if __name__ == "__main__":
    main()
