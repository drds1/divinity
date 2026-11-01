"""
Run every synthetic scenario over many random seeds and report how often the
prediction interval contains the truth, whether the model found the right
structure, and how its error compares with a naive forecast.

Usage::

    python benchmarks/coverage.py --seeds 100 --out benchmarks/results.md
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import divinity as dv
from divinity import synthetic


def structure_ok(name, components):
    """
    Did the model pick up the structure each scenario was built with?
    """
    c = components
    periods = list(c["seasonal_periods"])
    d = c["arima_order"][1] if c["arima_order"] else 0
    if name == "white_noise":
        return not c["trend"] and not periods and d == 0
    if name == "random_walk":
        return not periods and d == 1
    if name == "ar1":
        return not periods and d == 0
    if name == "unknown_period":
        return any(abs(p - 17.3) < 0.5 for p in periods)
    if name in ("trend_seasonal", "missing_data", "outliers"):
        return c["trend"] and any(abs(p - 20) < 0.5 for p in periods)
    if name == "daily_multi_seasonal":
        return 7.0 in periods and 365.25 in periods
    if name == "monthly_multiplicative":
        return c["trend"] and 12.0 in periods
    if name == "short_daily":
        return 7.0 in periods and 365.25 not in periods
    return True


def run_one(args):
    name, seed = args
    s = synthetic.make(name, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = dv.Forecaster(**s.options).fit(s.history)
        fc = model.predict(s.steps)
    truth = s.future.to_numpy()
    inside = (truth >= fc["lower"].to_numpy()) & (truth <= fc["upper"].to_numpy())
    last = s.history.dropna().iloc[-1]
    return {
        "scenario": name,
        "seed": seed,
        "coverage": inside.mean(),
        "rmse": np.sqrt(np.mean((fc["forecast"].to_numpy() - truth) ** 2)),
        "naive_rmse": np.sqrt(np.mean((last - truth) ** 2)),
        "structure": structure_ok(name, model.components_),
    }


def summarise(df):
    g = df.groupby("scenario", sort=False)
    return pd.DataFrame(
        {
            "runs": g.size(),
            "interval coverage": g["coverage"].mean(),
            "structure found": g["structure"].mean(),
            "RMSE vs naive": g["rmse"].mean() / g["naive_rmse"].mean(),
        }
    )


def to_markdown(table):
    lines = [
        "| scenario | runs | 95% interval coverage | structure found | RMSE vs naive |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in table.iterrows():
        lines.append(
            "| `{}` | {} | {:.1%} | {:.0%} | {:.2f} |".format(
                name,
                int(row["runs"]),
                row["interval coverage"],
                row["structure found"],
                row["RMSE vs naive"],
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--scenarios", nargs="*", default=list(synthetic.SCENARIOS))
    parser.add_argument("--out", default=None, help="write a markdown table here")
    args = parser.parse_args()

    jobs = [(name, seed) for name in args.scenarios for seed in range(args.seeds)]
    with ProcessPoolExecutor() as pool:
        df = pd.DataFrame(list(pool.map(run_one, jobs, chunksize=4)))
    table = summarise(df)
    md = to_markdown(table)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
