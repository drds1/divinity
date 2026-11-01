# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Divinity is a Python library for automatic univariate timeseries forecasting, published on PyPI as `divinity-forecast` (imported as `divinity`). The old `divinity` PyPI name belongs to an account that is no longer accessible, so do not publish there. The model is regression with ARIMA errors. The deterministic part is a constant, an optional linear trend and Fourier terms for each seasonal period. The library chooses those terms and the ARIMA order itself, then gives forecasts with prediction intervals.

The main entry point is `divinity.Forecaster`.

```python
import divinity as dv
model = dv.Forecaster().fit(y)       # Series, array or list. NaN allowed
result = model.predict(30)           # DataFrame: forecast, lower, upper
model.summary(); model.components_; model.plot(result, actual=truth)
dv.forecast(y, steps=30)             # one line version
```

## Layout

One module per stage, in the order `Forecaster.fit` uses them.

- [divinity/periods.py](divinity/periods.py) holds candidate periods from the pandas frequency (`default_periods`) and from the periodogram (`detect_periods`).
- [divinity/features.py](divinity/features.py) holds `FeatureGroup` (trend or one Fourier harmonic of one period) and `design_matrix`. Features depend only on the integer time step and the training length, never on the forecast horizon.
- [divinity/selection.py](divinity/selection.py) holds `forward_select`, which is BIC plus a search penalty of `2 * gamma * log(pool size)` per group. The `rho` argument selects on quasi-differenced data (feasible GLS).
- [divinity/residuals.py](divinity/residuals.py) holds the ADF unit root test (gap aware), the ARIMA order ranking by BIC and `is_degenerate`. The unit root decision is made in `Forecaster._unit_root` before feature selection, on residuals from the trend and known periods only. `Forecaster._fit_joint` skips orders whose joint fit is degenerate (AR or MA roots on the unit circle) and falls back to the next ranked order. When a series is differenced, the trend (drift) is admitted at the AIC price, as `auto.arima` does.
- [divinity/forecaster.py](divinity/forecaster.py) holds `Forecaster`, which ties the stages together, does the joint statsmodels ARIMA fit and adds coefficient uncertainty to the forecast variance.
- [divinity/plotting.py](divinity/plotting.py) holds the forecast chart.
- [divinity/synthetic.py](divinity/synthetic.py) holds reproducible test scenarios (`synthetic.make(name, seed)`) that return history, true future and any Forecaster options. They are shared by the tests, the README figures, the benchmark and the notebook.
- [divinity/datasets.py](divinity/datasets.py) holds example series for the applications. Real ones (`co2`, `sunspots`, `elnino`, `us_macro`) wrap data bundled with statsmodels. Simulated ones (`retail_sales`, `web_traffic`, `monthly_revenue`, `sensor_readings`) are seeded and labelled as simulated wherever they appear.
- [divinity/legacy.py](divinity/legacy.py) holds the deprecated version 0.1 `divinity` class (a wrapper over `Forecaster`) and the `gen_*` helpers.
- [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py) refits every scenario over many seeds and reports interval coverage, structure recovery and RMSE against a naive forecast. Its numbers are in the README table.
- [docs/make_figures.py](docs/make_figures.py) regenerates the README charts. Scenario charts (`docs/images/*.png`) use seed 0. Application charts (`docs/images/applications/*.png`) are made by running the README snippets tagged `<!-- example: name -->`, so the charts always match the code shown.
- [notebooks/demo.ipynb](notebooks/demo.ipynb) is the walkthrough of the API and the tricky cases. [notebooks/applications.ipynb](notebooks/applications.ipynb) works through each practical application and checks the forecasts against what happened. `scratch/` is old exploratory work and not part of the package.
- `docs/` pydocmd pages (`pydocmd.yml`, `docs/*.sh`, `docs/endpoints`, `docs/config`) are unmaintained placeholders from 0.1. The README is the real documentation.
- The Docker image ([Docker/Dockerfile](Docker/Dockerfile)) just runs the tests. It has not been built since the 0.2.0 rewrite.

## Commands

```bash
pip install -e ".[dev]"
pytest -m "not slow"                          # fast tests
pytest                                         # plus multi-seed coverage checks (minutes)
python benchmarks/run_benchmark.py --seeds 100      # README coverage table
python docs/make_figures.py                    # README figures
jupyter execute --inplace notebooks/demo.ipynb notebooks/applications.ipynb  # rerun notebooks (outputs are committed)
pre-commit run --all-files                     # black and whitespace checks
```

## CI and releases

- [.github/workflows/tests.yml](.github/workflows/tests.yml) runs the fast tests on pushes to master and on pull requests. The slow tests run only when triggered by hand with tier "full".
- [.github/workflows/publish.yml](.github/workflows/publish.yml) publishes to PyPI via trusted publishing when a GitHub release is created. To release, bump `version` in `pyproject.toml` and `__version__` in `divinity/__init__.py`, then create the release (e.g. `gh release create v0.2.1`).

## Things to keep true

- If you change the fitting algorithm, rerun the benchmark, the figure script and both notebooks, and update the README table. The README claims are only valid if these match.
- Keep README links and images as absolute GitHub URLs. The README is also the PyPI project page, where relative links break.
- `divinity-forecast` and the old `divinity` 0.1 both install a top-level `divinity` package, so they must not be installed together.
- Do not name modules after installed packages (the benchmark was once `coverage.py`, which clashed with the `coverage` package used by pytest-cov).
- `structure_ok` in `benchmarks/run_benchmark.py` defines what "right" means for each scenario. The smoke tests assert it for seed 0.
- README application snippets are executed by [tests/test_readme_examples.py](tests/test_readme_examples.py) and by the figure script. Keep the `<!-- example: name -->` tag above each one, keep each self contained, and end each with a `plot(..., title=...)` call. The prose next to them quotes numbers from the real output, so recheck it after algorithm changes.
- `predict(n)` must equal the first n rows of `predict(m)` for m > n (tested).
- Do not reintroduce plain OLS selection without the GLS step. It invents trends on random walks.
- Keep the code formatted with black, use `:param:` style docstrings and British spelling in prose.

## Security notes

- Never pass SSH keys or other secrets into Docker builds as `--build-arg`. Use BuildKit `--secret` or `--ssh` if a build ever needs credentials.
- Do not commit credentials, `.env` files or PyPI tokens.
