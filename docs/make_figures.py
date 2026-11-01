"""
Regenerate the README figures.

Scenario figures come from the synthetic scenarios with seed 0, the same
seed as the smoke tests, so nothing is hand picked. Application figures
come from running the code snippets in the README's "Practical
applications" section, so the charts always match the code shown. Run from
the repository root::

    python docs/make_figures.py
"""

import os
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import divinity as dv  # noqa: E402
from divinity import synthetic  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "images")
README = os.path.join(os.path.dirname(__file__), "..", "README.md")
EXAMPLE = re.compile(r"<!-- example: (\w+) -->\s*```python\n(.*?)```", re.S)

# how much history to show, so long daily series stay readable
HISTORY = {"daily_multi_seasonal": 400, "random_walk": 150}

TITLES = {
    "trend_seasonal": "Trend plus seasonality",
    "daily_multi_seasonal": "Daily data, weekly and yearly cycles",
    "monthly_multiplicative": "Monthly growth, multiplicative seasonality (log=True)",
    "white_noise": "Pure noise. Nothing to find",
    "random_walk": "Random walk. Looks trendy, is not",
    "ar1": "Autocorrelated but stationary",
    "missing_data": "15% missing plus a 25 step gap",
    "unknown_period": "Hidden period of 17.3 steps",
    "outliers": "Three large outliers",
    "short_daily": "Only five weeks of daily data",
}


def describe(components):
    """
    One line description of what the model found.
    """
    c = components
    parts = ["trend" if c["trend"] else "no trend"]
    if c["seasonal_periods"]:
        parts.append(
            "periods " + ", ".join("{:g}".format(p) for p in c["seasonal_periods"])
        )
    else:
        parts.append("no seasonality")
    parts.append("ARIMA{}".format(c["arima_order"]))
    return "Found: " + ", ".join(parts)


def readme_examples(path=README):
    """
    The tagged code snippets in the README, as a dict of name to code.
    """
    with open(path, encoding="utf-8") as f:
        return dict(EXAMPLE.findall(f.read()))


def run_example(code):
    """
    Run one README snippet and return the figure it drew.
    """
    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "<README example>", "exec"), {"__name__": "__example__"})
    return plt.gcf()


def application_figures():
    out = os.path.join(OUT, "applications")
    os.makedirs(out, exist_ok=True)
    for name, code in readme_examples().items():
        fig = run_example(code)
        fig.tight_layout()
        path = os.path.join(out, name + ".png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path)


def scenario_figures():
    os.makedirs(OUT, exist_ok=True)
    for name in synthetic.SCENARIOS:
        s = synthetic.make(name, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = dv.Forecaster(**s.options).fit(s.history)
        fc = model.predict(s.steps)
        ax = model.plot(fc, actual=s.future, history=HISTORY.get(name))
        ax.set_title(TITLES[name], loc="left", fontsize=12, color="#0b0b0b", pad=22)
        ax.text(
            0,
            1.02,
            describe(model.components_),
            transform=ax.transAxes,
            fontsize=9,
            color="#52514e",
        )
        ax.figure.tight_layout()
        path = os.path.join(OUT, name + ".png")
        ax.figure.savefig(path, dpi=110)
        plt.close(ax.figure)
        print("wrote", path)


def main():
    scenario_figures()
    application_figures()


if __name__ == "__main__":
    main()
