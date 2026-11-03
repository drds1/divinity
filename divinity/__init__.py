"""
Divinity, automatic timeseries forecasting.

Quick start::

    import divinity as dv

    model = dv.Forecaster().fit(y)
    result = model.predict(30)
    print(model.summary())
    model.plot(result)

or in one line::

    result = dv.forecast(y, steps=30)
"""

from . import datasets, synthetic
from .forecaster import Forecaster, forecast
from .legacy import divinity, gen_fake, gen_season, gen_trend

__version__ = "0.4.0"

__all__ = [
    "Forecaster",
    "forecast",
    "datasets",
    "synthetic",
    "divinity",
    "gen_fake",
    "gen_season",
    "gen_trend",
]
