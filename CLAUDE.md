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
- [divinity/residuals.py](divinity/residuals.py) holds the ADF unit root test and the ARIMA order grid search by BIC. The unit root decision is made in `Forecaster._unit_root` before feature selection, on residuals from the trend and known periods only.
- [divinity/forecaster.py](divinity/forecaster.py) holds `Forecaster`, which ties the stages together, does the joint statsmodels ARIMA fit and adds coefficient uncertainty to the forecast variance.
- [divinity/plotting.py](divinity/plotting.py) holds the forecast chart.
- [divinity/synthetic.py](divinity/synthetic.py) holds reproducible test scenarios (`synthetic.make(name, seed)`) that return history, true future and any Forecaster options. They are shared by the tests, the README figures, the benchmark and the notebook.
- [divinity/legacy.py](divinity/legacy.py) holds the deprecated version 0.1 `divinity` class (a wrapper over `Forecaster`) and the `gen_*` helpers.
- [benchmarks/coverage.py](benchmarks/coverage.py) refits every scenario over many seeds and reports interval coverage, structure recovery and RMSE against a naive forecast. Its numbers are in the README table.
- [docs/make_figures.py](docs/make_figures.py) regenerates `docs/images/*.png` for the README (seed 0 for every scenario).
- [notebooks/demo.ipynb](notebooks/demo.ipynb) is the walkthrough. `scratch/` is old exploratory work and not part of the package.

## Commands

```bash
pip install -e ".[dev]"
pytest -m "not slow"                          # fast tests
pytest                                         # plus multi-seed coverage checks (minutes)
python benchmarks/coverage.py --seeds 100      # README coverage table
python docs/make_figures.py                    # README figures
pre-commit run --all-files                     # black and whitespace checks
```

## Things to keep true

- If you change the fitting algorithm, rerun the benchmark and the figure script and update the README table. The README claims are only valid if these match.
- `structure_ok` in `benchmarks/coverage.py` defines what "right" means for each scenario. The smoke tests assert it for seed 0.
- `predict(n)` must equal the first n rows of `predict(m)` for m > n (tested).
- Do not reintroduce plain OLS selection without the GLS step. It invents trends on random walks.
- Keep the code formatted with black, use `:param:` style docstrings and British spelling in prose.

## Security notes

- Never pass SSH keys or other secrets into Docker builds as `--build-arg`. Use BuildKit `--secret` or `--ssh` if a build ever needs credentials.
- Do not commit credentials, `.env` files or PyPI tokens.
