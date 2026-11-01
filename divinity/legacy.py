"""
The version 0.1 interface, kept so existing code keeps working.
New code should use :class:`divinity.Forecaster`.
"""

import warnings

import numpy as np
import pandas as pd

from .forecaster import Forecaster


class divinity:
    """
    Deprecated version 0.1 interface, now a thin wrapper around
    :class:`Forecaster`.

    :param forecast_length: number of steps returned by :meth:`predict`.
    :param confidence_interval: interval coverage in percent.
    :param seasonal_periods: candidate seasonal periods. None uses the
        Forecaster defaults.
    :param trend_order: [0] for no trend, anything including 1 for "auto".
    :param kwargs: other version 0.1 arguments are accepted and ignored.
    """

    def __init__(
        self,
        forecast_length,
        confidence_interval=95.0,
        seasonal_periods=None,
        trend_order=(0, 1),
        **kwargs,
    ):
        warnings.warn(
            "divinity.divinity is deprecated, use divinity.Forecaster",
            DeprecationWarning,
            stacklevel=2,
        )
        self.forecast_length = forecast_length
        self.confidence_interval = confidence_interval
        self.model = Forecaster(
            seasonal_periods="auto" if seasonal_periods is None else seasonal_periods,
            trend="auto" if max(trend_order) >= 1 else None,
            interval=confidence_interval / 100.0,
        )

    @property
    def features(self):
        return self.model._features(np.arange(self.model.n_))

    def fit(self, y):
        self.model.fit(y)

    def predict(self):
        """
        :return: numpy array of the forecast. The half width of the
            prediction interval is stored in ``ystd``.
        """
        result = self.model.predict(self.forecast_length)
        self.ypred = result["forecast"].to_numpy()
        self.ystd = ((result["upper"] - result["lower"]) / 2).to_numpy()
        return self.ypred

    def forecast(self, steps):
        self.forecast_length = steps
        return self.predict()


def gen_season(N, periods=(10, 20), sine_amplitudes=(1, 1), cosine_amplitudes=(0, 0)):
    """
    Sinusoidal features and their weighted sum.

    :return: dict with "features" (DataFrame) and "target" (array).
    """
    t = np.arange(N)
    features = {}
    X = np.zeros(N)
    for idx, p in enumerate(periods):
        if p == 0 or p > N / 2:
            warnings.warn("period {} ignored, must be in (0, N/2]".format(p))
            continue
        s = np.sin(2 * np.pi / p * t)
        c = np.cos(2 * np.pi / p * t)
        if sine_amplitudes is not None:
            X += sine_amplitudes[idx] * s
        if cosine_amplitudes is not None:
            X += cosine_amplitudes[idx] * c
        features["sin P=" + str(p)] = s
        features["cos P=" + str(p)] = c
    return {"features": pd.DataFrame(features), "target": X}


def gen_trend(N, coef=(0, 1, 2), amplitudes=(1, 0.5, 0.2)):
    """
    Polynomial trend features and their weighted sum.

    :return: dict with "features" (DataFrame) and "target" (array).
    """
    t = np.arange(N)
    features = {}
    X = np.zeros(N)
    for idx, order in enumerate(coef):
        f = np.array(t**order, dtype=float)
        if amplitudes is not None:
            X += amplitudes[idx] * f
        features["trend order " + str(order)] = f
    return {"features": pd.DataFrame(features), "target": X}


def gen_fake(
    N,
    trend_order,
    trend_amplitudes,
    seasonal_periods,
    seasonal_amplitudes_sine,
    seasonal_amplitudes_cosine,
    normalize="0to1",
):
    """
    Synthetic trend plus seasonal series and its feature matrix. Prefer
    :mod:`divinity.synthetic` for new code.

    :return: dict with "features" (DataFrame) and "target" (array).
    """
    trend = gen_trend(N, coef=list(range(trend_order + 1)), amplitudes=trend_amplitudes)
    season = gen_season(
        N,
        periods=seasonal_periods,
        sine_amplitudes=seasonal_amplitudes_sine,
        cosine_amplitudes=seasonal_amplitudes_cosine,
    )
    features = pd.concat([trend["features"], season["features"]], axis=1)
    if normalize == "0to1":
        spread = features.max() - features.min()
        cols = spread[spread > 0].index
        features[cols] = (features[cols] - features[cols].min()) / spread[cols]
    return {"features": features, "target": trend["target"] + season["target"]}
