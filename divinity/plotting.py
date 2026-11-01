"""
Forecast plots. Colours follow a colour blind safe categorical palette
(history in neutral grey, forecast in blue, true future values in orange).
"""

import numpy as np
import pandas as pd

HISTORY = "#52514e"
FORECAST = "#2a78d6"
ACTUAL = "#eb6834"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_forecast(history, forecast, actual=None, ax=None, title=None):
    """
    Plot history, forecast with its prediction interval, and optionally the
    true future values.

    :param history: Series of observed values.
    :param forecast: DataFrame with columns forecast, lower and upper.
    :param actual: optional Series or array of true future values.
    :param ax: matplotlib Axes, a new figure is made if None.
    :param title: optional title.
    :return: the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.9))
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)

    pct = int(round(100 * forecast.attrs.get("interval", 0.95)))
    ax.plot(history.index, history.values, color=HISTORY, lw=1.5, label="history")
    ax.fill_between(
        forecast.index,
        forecast["lower"],
        forecast["upper"],
        color=FORECAST,
        alpha=0.15,
        lw=0,
        label="{}% interval".format(pct),
    )
    ax.plot(
        forecast.index, forecast["forecast"], color=FORECAST, lw=2, label="forecast"
    )
    if actual is not None:
        if not isinstance(actual, pd.Series):
            actual = pd.Series(np.asarray(actual), index=forecast.index[: len(actual)])
        ax.plot(actual.index, actual.values, color=ACTUAL, lw=1.5, label="actual")

    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=HISTORY, labelsize=9)
    if isinstance(forecast.index, pd.DatetimeIndex):
        import matplotlib.dates as mdates

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    # below the axes so it never covers the data
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        ncol=4,
    )
    if title:
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    return ax
