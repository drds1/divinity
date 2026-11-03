"""
Compare Divinity with Prophet, statsforecast and simple baselines on real
series from the M4 competition.

Each model is given the same training data with a regular date index of the
right frequency and is used with its default settings, as a new user would.
The last ``h`` values of each series (the official M4 test period) are held
out. Point accuracy is measured by MASE and sMAPE, interval quality by the
coverage of the 95% interval and the M4 scaled interval score (MSIS).

Needs the optional benchmark dependencies::

    pip install -e ".[benchmark]"
    python benchmarks/compare_models.py --series 100 --jobs 4

M4 data are downloaded to ``--data`` on first use (about 300 MB).
"""

import os

# one thread per worker process, the parallelism comes from the pool
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# horizon, seasonal period (M4 convention, used for MASE scaling and the
# statsforecast models) and pandas frequency for the date index
GROUPS = {
    "Yearly": (6, 1, "YS"),
    "Quarterly": (8, 4, "QS"),
    "Monthly": (18, 12, "MS"),
    "Weekly": (13, 1, "W"),
    "Daily": (14, 1, "D"),
    "Hourly": (48, 24, "h"),
}
MODELS = ["Divinity", "Prophet", "AutoARIMA", "AutoETS", "SeasonalNaive", "Naive"]
LEVEL = 95
ALPHA = 1 - LEVEL / 100


def _quiet():
    warnings.simplefilter("ignore")
    for name in ("cmdstanpy", "prophet", "prophet.plot"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _init_worker():
    _quiet()
    # compile the numba code once per worker so it is not counted in timings
    from statsforecast.models import AutoARIMA, AutoETS

    y = np.sin(np.arange(40.0)) + np.arange(40.0)
    AutoARIMA(season_length=4).forecast(y, 4, level=[LEVEL])
    AutoETS(season_length=4).forecast(y, 4, level=[LEVEL])


def fit_predict(model, y, h, m, freq):
    """
    Fit one model and forecast ``h`` steps ahead.

    :param model: name from ``MODELS``.
    :param y: training values.
    :param h: forecast horizon.
    :param m: seasonal period for the statsforecast models.
    :param freq: pandas frequency of the date index.
    :return: (forecast, lower, upper) arrays of length ``h``.
    """
    index = pd.date_range("2000-01-01", periods=len(y), freq=freq)
    if model == "Divinity":
        import divinity as dv

        fc = dv.Forecaster(interval=LEVEL / 100).fit(pd.Series(y, index=index))
        out = fc.predict(h)
        return (
            out["forecast"].to_numpy(),
            out["lower"].to_numpy(),
            out["upper"].to_numpy(),
        )
    if model == "Prophet":
        from prophet import Prophet

        p = Prophet(interval_width=LEVEL / 100)
        p.fit(pd.DataFrame({"ds": index, "y": y}))
        future = p.make_future_dataframe(h, freq=freq, include_history=False)
        out = p.predict(future)
        return (
            out["yhat"].to_numpy(),
            out["yhat_lower"].to_numpy(),
            out["yhat_upper"].to_numpy(),
        )

    from statsforecast import models as sf

    if model == "AutoARIMA":
        est = sf.AutoARIMA(season_length=m)
    elif model == "AutoETS":
        est = sf.AutoETS(season_length=m)
    elif model == "SeasonalNaive":
        est = sf.SeasonalNaive(season_length=m)
    elif model == "Naive":
        est = sf.Naive()
    else:
        raise ValueError("unknown model {}".format(model))
    out = est.forecast(y, h, level=[LEVEL])
    return out["mean"], out["lo-{}".format(LEVEL)], out["hi-{}".format(LEVEL)]


def _seasonal_naive(y, h, m):
    from statsforecast.models import SeasonalNaive

    out = SeasonalNaive(season_length=m).forecast(y, h, level=[LEVEL])
    return out["mean"], out["lo-{}".format(LEVEL)], out["hi-{}".format(LEVEL)]


def scores(train, test, forecast, lower, upper, m):
    """
    MASE, sMAPE, interval coverage and MSIS for one forecast.
    """
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    if not scale > 0:
        scale = np.nan
    err = np.abs(test - forecast)
    penalty = (2 / ALPHA) * (
        np.clip(lower - test, 0, None) + np.clip(test - upper, 0, None)
    )
    return {
        "mase": np.mean(err) / scale,
        "smape": 200 * np.mean(err / (np.abs(test) + np.abs(forecast))),
        "coverage": np.mean((test >= lower) & (test <= upper)),
        "msis": np.mean(upper - lower + penalty) / scale,
    }


def run_one(task):
    group, uid, model, values = task
    h, m, freq = GROUPS[group]
    train, test = values[:-h], values[-h:]
    start = time.perf_counter()
    try:
        forecast, lower, upper = fit_predict(model, train, h, m, freq)
        forecast, lower, upper = (
            np.asarray(a, dtype=float) for a in (forecast, lower, upper)
        )
        if not np.all(np.isfinite(np.concatenate([forecast, lower, upper]))):
            raise ValueError("non-finite forecast")
        failed = ""
    except Exception as exc:  # a failure falls back to seasonal naive
        forecast, lower, upper = _seasonal_naive(train, h, m)
        failed = "{}: {}".format(type(exc).__name__, exc)[:200]
    seconds = time.perf_counter() - start
    row = {"group": group, "series": uid, "model": model, "n": len(train)}
    row.update(scores(train, test, forecast, lower, upper, m))
    row.update({"seconds": seconds, "failed": failed})
    return row


def load_tasks(groups, n_series, seed, data_dir, models):
    from datasetsforecast.m4 import M4

    tasks = []
    rng = np.random.default_rng(seed)
    for group in groups:
        df, *_ = M4.load(directory=data_dir, group=group)
        ids = df["unique_id"].unique()
        chosen = rng.choice(ids, size=min(n_series, len(ids)), replace=False)
        series = df[df["unique_id"].isin(chosen)].groupby("unique_id")["y"]
        for uid, y in series:
            values = y.to_numpy(dtype=float)
            tasks.extend((group, uid, model, values) for model in models)
    return tasks


def summarise(rows):
    """
    Average each metric by frequency group and model, plus an overall row
    weighted equally per series.
    """
    df = pd.DataFrame(rows)
    df["failed"] = df["failed"].astype(bool)
    agg = {
        "series": ("series", "nunique"),
        "MASE": ("mase", "mean"),
        "sMAPE": ("smape", "mean"),
        "coverage": ("coverage", "mean"),
        "MSIS": ("msis", "mean"),
        "median_s": ("seconds", "median"),
        "failures": ("failed", "sum"),
    }
    by_group = df.groupby(["group", "model"]).agg(**agg).reset_index()
    overall = df.groupby("model").agg(**agg).reset_index()
    overall.insert(0, "group", "All")
    return pd.concat([by_group, overall], ignore_index=True)


def to_markdown(summary, order):
    lines = []
    for group in list(GROUPS) + ["All"]:
        part = summary[summary["group"] == group]
        if part.empty:
            continue
        part = part.set_index("model").reindex(
            [m for m in order if m in part["model"].values]
        )
        h, m, freq = GROUPS.get(group, (None, None, None))
        title = group if group == "All" else "{} (h={}, m={})".format(group, h, m)
        lines += [
            "### {}, {} series".format(title, int(part["series"].max())),
            "",
            "| model | MASE | sMAPE | 95% coverage | MSIS | median seconds | failures |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        best = {c: part[c].min() for c in ("MASE", "sMAPE", "MSIS")}
        for model, r in part.iterrows():
            cell = {
                c: ("**{:.3f}**" if np.isclose(r[c], best[c]) else "{:.3f}").format(
                    r[c]
                )
                for c in ("MASE", "MSIS")
            }
            smape = (
                "**{:.2f}**" if np.isclose(r["sMAPE"], best["sMAPE"]) else "{:.2f}"
            ).format(r["sMAPE"])
            lines.append(
                "| {} | {} | {} | {:.1%} | {} | {:.2f} | {} |".format(
                    model,
                    cell["MASE"],
                    smape,
                    r["coverage"],
                    cell["MSIS"],
                    r["median_s"],
                    int(r["failures"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--series", type=int, default=100, help="series per frequency")
    parser.add_argument(
        "--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS)
    )
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--seed", type=int, default=0, help="seed for sampling series")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--data", default="benchmarks/data")
    parser.add_argument("--out", default="benchmarks/comparison.md")
    parser.add_argument("--csv", default="benchmarks/comparison.csv")
    parser.add_argument(
        "--update",
        action="store_true",
        help="keep rows for other models from the existing --csv, so only "
        "the models given are refitted (same --series and --seed assumed)",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.groups, args.series, args.seed, args.data, args.models)
    # slowest first, so the pool is not left waiting on a long task at the end
    tasks.sort(key=lambda t: -len(t[3]) * (t[2] == "AutoARIMA" and GROUPS[t[0]][1] > 1))
    rows = []
    with ProcessPoolExecutor(args.jobs, initializer=_init_worker) as pool:
        for i, row in enumerate(pool.map(run_one, tasks, chunksize=1), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(tasks):
                print("{}/{} fits done".format(i, len(tasks)), flush=True)

    order = args.models
    if args.update and os.path.exists(args.csv):
        old = pd.read_csv(args.csv)
        old["failed"] = old["failed"].fillna("")
        keep = old[~old["model"].isin(args.models) & old["group"].isin(args.groups)]
        rows = keep.to_dict("records") + rows
        order = [m for m in MODELS if m in set(old["model"]) | set(args.models)]
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    table = to_markdown(summarise(rows), order)
    header = (
        "# Comparison on M4\n\n"
        "{} series per frequency sampled with seed {}. Lower is better for MASE, "
        "sMAPE and MSIS, and coverage should be close to 95%. Best in each "
        "column in bold.\n\n".format(args.series, args.seed)
    )
    with open(args.out, "w") as f:
        f.write(header + table)
    print(table)


if __name__ == "__main__":
    main()
