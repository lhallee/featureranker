# Contributing and development

## Setup

```bash
git clone https://github.com/lhallee/featureranker.git
cd featureranker
pip install -e ".[dev]"
```

Python >= 3.11.

## Tests

```bash
pytest              # fast suite (slow marker excluded by default)
pytest -m slow      # benchmark-sized tests only
pytest -m ""        # everything
```

The suite enforces the package's behavioral contracts; the strongest ones
live in `tests/test_l1.py` (entry-point correctness, exact-vs-grid path
agreement, wave-vs-dense oracle) and `tests/test_determinism.py` (identical
results across runs and n_jobs settings). Treat those as the specification
when changing ranking code.

## Code conventions

- Typed signatures throughout; user-input errors are `ValueError` or
  `TypeError` with the allowed values in the message; assertions only for
  internal invariants.
- Import blocks: module docstring, then `import x` statements, then
  `from x import y` statements, each block ordered standard library, third
  party, repository-local.
- Numerical code carries shape comments on array-valued lines, for example
  `coefs  # (p, k)`.
- Comments state intent or constraints, never restate the code.
- Prose (docs, docstrings, messages) uses American English, no em dashes.
- One module per responsibility: `ranking.py` orchestrates; each method
  family owns its module and its options dataclass.

## Regenerating example docs

The pages in `docs/examples/` and the images in `docs/images/` are written
by the scripts in `examples/`. After changing ranking or plotting behavior,
rerun them so the documentation shows current output:

```bash
python examples/breast_cancer.py
python examples/diabetes.py
python examples/modernbert_sentiment.py extract --out artifacts
python examples/modernbert_sentiment.py rank --artifacts artifacts
```

The ModernBERT extract phase needs torch, transformers, and datasets (a GPU
helps); its rank phase, and the other two scripts, need only featureranker.
The two phases can run in different interpreters.

## Releasing

1. Update `CHANGELOG.md`.
2. Bump `__version__` in `src/featureranker/__init__.py` (the only version
   site; pyproject reads it through hatchling).
3. Merge to `main` with the tests workflow green.
4. `python -m build` and `twine check dist/*` locally.
5. Create a GitHub release with tag `vX.Y.Z`; publishing the release runs
   `.github/workflows/python-publish.yml`, which builds and uploads to PyPI
   with the `PYPI_API_TOKEN` repository secret.
6. Verify the new version on https://pypi.org/project/featureranker/ and
   `pip install featureranker==X.Y.Z` in a clean environment.
