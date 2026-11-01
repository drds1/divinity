"""
Synthetic test series with known structure, split into history and future.

These are used by the tests, the README figures and the demo notebook, and
are handy for experimenting. Every scenario is reproducible from its seed.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    """
    A synthetic forecasting problem.

    :param name: short identifier.
    :param description: what the series contains and what a good model
        should do with it.
    :param history: training series.
    :param future: true values over the forecast horizon.
    :param options: Forecaster arguments this scenario calls for.
    """

    name: str
    description: str
    history: pd.Series
    future: pd.Series
    options: dict = field(default_factory=dict)

    @property
    def steps(self):
        return len(self.future)


def _split(y, horizon):
    return y.iloc[:-horizon], y.iloc[-horizon:]


def trend_seasonal(seed=0, n=200, horizon=40):
    rng = np.random.default_rng(seed)
    t = np.arange(n + horizon)
    y = 0.05 * t + 2 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.5, len(t))
    history, future = _split(pd.Series(y), horizon)
    return Scenario(
        "trend_seasonal",
        "Linear trend plus a period 20 cycle and noise. The standard case.",
        history,
        future,
    )


def daily_multi_seasonal(seed=0, years=3, horizon=90):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=int(365.25 * years) + horizon, freq="D")
    t = np.arange(len(dates))
    weekly = np.where(dates.day_of_week >= 5, -3.0, 0.6)  # sharp weekend dip
    yearly = 4 * np.sin(2 * np.pi * t / 365.25)
    y = 50 + 0.01 * t + weekly + yearly + rng.normal(0, 1.0, len(t))
    history, future = _split(pd.Series(y, index=dates), horizon)
    return Scenario(
        "daily_multi_seasonal",
        "Daily data with a square weekend dip and a yearly cycle. Weekly and "
        "yearly periods come from the date frequency.",
        history,
        future,
    )


def monthly_multiplicative(seed=0, years=12, horizon=24):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=12 * years + horizon, freq="MS")
    t = np.arange(len(dates))
    log_y = (
        np.log(100)
        + 0.01 * t
        + 0.2 * np.sin(2 * np.pi * t / 12)
        + 0.1 * np.cos(4 * np.pi * t / 12)
        + rng.normal(0, 0.04, len(t))
    )
    history, future = _split(pd.Series(np.exp(log_y), index=dates), horizon)
    return Scenario(
        "monthly_multiplicative",
        "Monthly growth series whose seasonal swing grows with its level. "
        "Fitted on the log scale.",
        history,
        future,
        {"log": True},
    )


def white_noise(seed=0, n=200, horizon=50):
    rng = np.random.default_rng(seed)
    history, future = _split(pd.Series(10 + rng.normal(0, 1, n + horizon)), horizon)
    return Scenario(
        "white_noise",
        "Pure noise around a constant. There is nothing to find, so the "
        "forecast should be flat with no invented seasonality or trend.",
        history,
        future,
    )


def random_walk(seed=0, n=300, horizon=60):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 1, n + horizon))
    history, future = _split(pd.Series(y), horizon)
    return Scenario(
        "random_walk",
        "A random walk. It wanders and looks trendy by chance, so the model "
        "should not extrapolate a confident trend and its interval should "
        "widen with the horizon.",
        history,
        future,
    )


def ar1(seed=0, n=200, horizon=50, phi=0.85):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n + horizon + 100)
    u = np.zeros_like(e)
    for i in range(1, len(e)):
        u[i] = phi * u[i - 1] + e[i]
    history, future = _split(pd.Series(20 + u[100:]), horizon)
    return Scenario(
        "ar1",
        "Strongly autocorrelated but stationary. Forecast should decay back "
        "towards the mean rather than run off as if it were a random walk.",
        history,
        future,
    )


def missing_data(seed=0, n=200, horizon=40):
    base = trend_seasonal(seed, n, horizon)
    rng = np.random.default_rng(seed + 1000)
    history = base.history.copy()
    history[rng.random(n) < 0.15] = np.nan
    history.iloc[90:115] = np.nan
    return Scenario(
        "missing_data",
        "The standard case with 15% of points missing at random and a " "25 step gap.",
        history,
        base.future,
    )


def unknown_period(seed=0, n=240, horizon=60):
    rng = np.random.default_rng(seed)
    t = np.arange(n + horizon)
    y = 3 * np.sin(2 * np.pi * t / 17.3) + rng.normal(0, 1, len(t))
    history, future = _split(pd.Series(y), horizon)
    return Scenario(
        "unknown_period",
        "A period of 17.3 steps that nobody told the model about. It has to "
        "find it in the periodogram.",
        history,
        future,
    )


def outliers(seed=0, n=200, horizon=40):
    base = trend_seasonal(seed, n, horizon)
    history = base.history.copy()
    history.iloc[[40, 120, 170]] += np.array([15.0, -12.0, 18.0])
    return Scenario(
        "outliers",
        "The standard case with three large one-off spikes.",
        history,
        base.future,
    )


def short_daily(seed=0, days=35, horizon=14):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-03-01", periods=days + horizon, freq="D")
    weekly = np.where(dates.day_of_week >= 5, 5.0, 0.0)
    y = 20 + weekly + rng.normal(0, 1, len(dates))
    history, future = _split(pd.Series(y, index=dates), horizon)
    return Scenario(
        "short_daily",
        "Only five weeks of daily data. The yearly default period cannot be "
        "estimated and must be ignored, while the weekly cycle is still used.",
        history,
        future,
    )


SCENARIOS = {
    f.__name__: f
    for f in (
        trend_seasonal,
        daily_multi_seasonal,
        monthly_multiplicative,
        white_noise,
        random_walk,
        ar1,
        missing_data,
        unknown_period,
        outliers,
        short_daily,
    )
}


def make(name, seed=0, **kwargs):
    """
    Build a scenario by name, e.g. ``make("random_walk", seed=3)``.
    """
    try:
        return SCENARIOS[name](seed=seed, **kwargs)
    except KeyError:
        raise ValueError(
            "unknown scenario {!r}, choose from {}".format(name, sorted(SCENARIOS))
        ) from None
