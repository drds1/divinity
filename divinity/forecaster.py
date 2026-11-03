"""
The main forecasting model.

The series is modelled as

    y_t = constant + trend + seasonality + u_t

where trend and seasonality are deterministic (a linear term and Fourier
pairs for each seasonal period) and u_t follows an ARIMA process. This is
"regression with ARIMA errors", fitted jointly by maximum likelihood once
the features and the ARIMA order have been chosen.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from .features import FeatureGroup, degenerate_columns, design_matrix, seasonal_groups
from .periods import default_periods, detect_periods, merge_periods
from .residuals import fit_quietly, has_unit_root, is_degenerate, rank_orders
from .selection import forward_select, lag1_autocorrelation

MIN_OBSERVATIONS = 10
MAX_GLS_ITERATIONS = 4
MAX_STATIONARY_RHO = 0.95
# longest period given seasonal ARIMA errors, longer ones are too slow to fit
MAX_SARIMA_PERIOD = 48
# a seasonal MA coefficient at or below this means the pattern is fixed
FIXED_SEASON_THETA = -0.8
# full cycles needed before an evolving pattern can be told from noise
MIN_SEASON_CYCLES = 6


class Forecaster:
    """
    Automatic timeseries forecaster.

    Example::

        model = Forecaster().fit(y)
        result = model.predict(30)   # DataFrame with forecast, lower, upper
        model.plot(result)

    :param seasonal_periods: "auto" to use defaults for the series frequency
        (e.g. 7 and 365.25 for daily data), a list of periods in time steps,
        or None for no predefined periods.
    :param detect_periods: also search the data for periodicities that are
        not in ``seasonal_periods``.
    :param max_detected_periods: how many detected periods to consider.
    :param max_harmonics: maximum Fourier harmonics per period. More
        harmonics allow sharper, less sinusoidal seasonal shapes.
    :param trend: "auto" to include a linear trend only if the data support
        it, "linear" to always include one, or None for no trend.
    :param arima: "auto" to choose the residual ARIMA order, a (p, d, q)
        tuple to fix it, or None to treat residuals as white noise.
    :param log: fit on the log of the series. Use for strictly positive
        data whose seasonal swings grow with its level.
    :param interval: default coverage of the prediction interval.
    :param search_penalty: strength of the penalty for searching over many
        candidate features. Higher values make the model more reluctant to
        add seasonality, 0 gives plain BIC.
    :param evolving_seasonality: test whether the main seasonal pattern
        changes over time and, if it does, model it with seasonal ARIMA
        errors instead of fixed Fourier terms. Only used with
        ``arima="auto"`` and an integer period of at most 48 steps.
    """

    def __init__(
        self,
        seasonal_periods="auto",
        detect_periods=True,
        max_detected_periods=3,
        max_harmonics=3,
        trend="auto",
        arima="auto",
        log=False,
        interval=0.95,
        search_penalty=1.0,
        evolving_seasonality=True,
    ):
        if trend not in ("auto", "linear", None):
            raise ValueError("trend must be 'auto', 'linear' or None")
        if not 0 < interval < 1:
            raise ValueError("interval must be between 0 and 1, e.g. 0.95")
        self.seasonal_periods = seasonal_periods
        self.detect_periods = detect_periods
        self.max_detected_periods = max_detected_periods
        self.max_harmonics = max_harmonics
        self.trend = trend
        self.arima = arima
        self.log = log
        self.interval = interval
        self.search_penalty = search_penalty
        self.evolving_seasonality = evolving_seasonality
        self._result = None

    # ------------------------------------------------------------------ fit

    def fit(self, y, freq=None):
        """
        Fit the model to a series.

        :param y: pandas Series, numpy array or list. A Series with a
            DatetimeIndex gives dated forecasts and frequency based seasonal
            defaults. Missing values (NaN) are allowed.
        :param freq: pandas frequency string (e.g. "D", "h", "MS"). Only
            needed if it cannot be inferred from the index, for example
            when some dates are missing entirely.
        :return: self, so calls can be chained.
        """
        series, self.freq_ = _as_series(y, freq)
        values = series.to_numpy(dtype=float)
        if self.log:
            if np.nanmin(values) <= 0:
                raise ValueError("log=True needs a strictly positive series")
            values = np.log(values)

        n = len(values)
        observed = ~np.isnan(values)
        if observed.sum() < MIN_OBSERVATIONS:
            raise ValueError(
                "need at least {} observed values, got {}".format(
                    MIN_OBSERVATIONS, int(observed.sum())
                )
            )
        self.history_ = series
        self.n_ = n
        self.n_missing_ = int(n - observed.sum())

        groups, forced, pool = self._candidates(values)
        self._drop = degenerate_columns(groups + forced, n)

        # Decide on a unit root before choosing features, using only the
        # trusted terms (trend and known periods). Features fitted to a
        # random walk soak up its wander and make the residuals look
        # stationary, so testing afterwards would confirm its own mistake.
        self.differenced_ = self._unit_root(values, groups, forced)

        # With a unit root, select on differenced data. Otherwise select by
        # OLS, then reselect allowing for AR(1) errors until the choice of
        # features stops changing.
        rho, seen = (1.0 if self.differenced_ else 0.0), []
        for _ in range(MAX_GLS_ITERATIONS):
            selection = forward_select(
                values,
                groups,
                pool,
                drop_columns=self._drop,
                gamma=self.search_penalty,
                forced=forced,
                rho=rho,
                # a drift's effect grows over the whole horizon, so admit it
                # at the AIC price rather than BIC, as auto.arima does
                aic_groups=(
                    [g for g in groups if g.kind == "trend"]
                    if self.differenced_
                    else ()
                ),
            )
            self.groups_ = selection.groups
            X = self._features(np.arange(n))
            resid = _ols_residuals(X, values)
            if self.differenced_ or self.arima is None:
                break
            rho = min(lag1_autocorrelation(resid), MAX_STATIONARY_RHO)
            if set(self.groups_) in seen:
                break
            seen.append(set(self.groups_))
        self.selection_history_ = selection.history
        self.rho_ = rho

        self.seasonal_order_ = None
        if np.nanstd(resid) < 1e-10 * max(1.0, np.nanstd(values)):
            # deterministic series, nothing left for ARIMA to model
            self.order_ = None
            self._beta = _ols_beta(X, values)
            self._result = None
            self._n_obs, self._n_params = int(observed.sum()), 0
            return self

        self.order_ = self._choose_order(resid)
        self._result = self._fit_joint(values, X, self.order_)
        self._try_evolving_season(values)
        self._n_obs = int(observed.sum())
        self._n_params = len(self._result.params)
        return self

    def _evolving_period(self):
        """
        The seasonal period to test for an evolving pattern, or None. Only
        integer periods from the frequency defaults or the user qualify, as
        seasonal ARIMA needs a whole number of steps per cycle. A period
        whose Fourier terms were not selected still qualifies, because a
        pattern that drifts can average out to nothing over the history.
        """
        if not self.evolving_seasonality or self.arima != "auto":
            return None
        n_obs = int(np.sum(~self.history_.isna()))
        for p in self.candidate_periods_["known"]:
            if (
                p == round(p)
                and 2 <= p <= MAX_SARIMA_PERIOD
                and n_obs >= MIN_SEASON_CYCLES * p
            ):
                return int(p)
        return None

    def _try_evolving_season(self, values):
        """
        Refit with seasonal ARIMA(0, 1, 1) errors in place of the Fourier
        terms for one period, and keep that model if the seasonal pattern
        is evolving.

        With seasonal differencing a fixed seasonal pattern shows up as a
        seasonal MA coefficient of -1, so the Fourier model is kept unless
        the coefficient is clearly above that.
        """
        m = self._evolving_period()
        if m is None:
            return
        p, d, q = self.order_
        groups = [
            g
            for g in self.groups_
            # seasonal differencing removes every period m / k as well
            if not (g.kind == "season" and _divides(g.period, m))
            # a linear trend is not identified once differenced twice
            and not (g.kind == "trend" and d > 0)
        ]
        X = design_matrix(np.arange(self.n_), groups, self.n_, self._drop)
        exog = X.to_numpy() if X.shape[1] else None
        for order in dict.fromkeys([(p, d, q), (0, d, 0)]):
            try:
                res = fit_quietly(
                    ARIMA(
                        values,
                        exog=exog,
                        order=order,
                        seasonal_order=(0, 1, 1, m),
                        trend="n",
                    )
                )
            except (np.linalg.LinAlgError, ValueError):
                continue
            theta = res.params[list(res.model.param_names).index("ma.S.L{}".format(m))]
            self.seasonal_theta_ = float(theta)
            if theta <= FIXED_SEASON_THETA:
                return
            if not is_degenerate(res):
                self.groups_ = groups
                self.order_ = order
                self.seasonal_order_ = (0, 1, 1, m)
                self._result = res
                return

    def _candidates(self, values):
        """
        Build candidate feature groups and their search pool sizes.
        """
        n = len(values)
        n_obs = int(np.sum(~np.isnan(values)))
        if self.seasonal_periods == "auto":
            known = default_periods(self.freq_)
        else:
            known = list(self.seasonal_periods or [])
        detected = (
            detect_periods(values, self.max_detected_periods)
            if self.detect_periods
            else []
        )
        known, detected = merge_periods(known, detected, n)
        self.candidate_periods_ = {"known": known, "detected": detected}

        groups, forced, pool = [], [], {}
        trend = FeatureGroup("trend")
        if self.trend == "linear":
            forced.append(trend)
        elif self.trend == "auto":
            groups.append(trend)
            pool[trend] = 1
        for g in seasonal_groups(known, self.max_harmonics):
            pool[g] = len(known) if g.harmonic == 1 else self.max_harmonics
            groups.append(g)
        for g in seasonal_groups(detected, self.max_harmonics, detected=True):
            # a detected period was picked from roughly n / 2 frequencies
            pool[g] = n_obs // 2 if g.harmonic == 1 else self.max_harmonics
            groups.append(g)
        return groups, forced, pool

    def _features(self, t):
        return design_matrix(t, self.groups_, self.n_, self._drop)

    def _fixed_order(self):
        order = tuple(int(v) for v in self.arima)
        if len(order) != 3:
            raise ValueError("arima must be 'auto', None or a (p, d, q) tuple")
        return order

    def _unit_root(self, values, groups, forced):
        """
        Whether the series needs differencing, judged after removing the
        trend and known seasonal periods. Detected periods are excluded
        because they may be spurious.
        """
        if self.arima is None:
            return False
        if self.arima != "auto":
            return self._fixed_order()[1] > 0
        trusted = forced + [g for g in groups if not g.detected]
        X = design_matrix(np.arange(len(values)), trusted, len(values), self._drop)
        return has_unit_root(_ols_residuals(X, values), trend=self.trend is not None)

    def _choose_order(self, resid):
        if self.arima is None:
            return (0, 0, 0)
        if self.arima != "auto":
            return self._fixed_order()
        d = 1 if self.differenced_ else 0
        ranked = [order for order, _ in rank_orders(resid, d)]
        # kept so the joint fit can fall back if the best order degenerates
        self._order_candidates = ranked[1:] + [(0, d, 0)]
        return ranked[0] if ranked else (0, d, 0)

    def _fit_joint(self, values, X, order):
        """
        Fit the features and ARIMA errors together. With an automatic order,
        degenerate fits are rejected in favour of the next best order.
        """
        exog = X.to_numpy() if X.shape[1] else None
        fallbacks = (
            list(getattr(self, "_order_candidates", [])) if self.arima == "auto" else []
        )
        candidates = [order] + [o for o in fallbacks if o != order]
        for i, candidate in enumerate(candidates):
            trend = "c" if candidate[1] == 0 else "n"
            try:
                res = fit_quietly(
                    ARIMA(values, exog=exog, order=candidate, trend=trend)
                )
            except (np.linalg.LinAlgError, ValueError) as err:
                if i == len(candidates) - 1:
                    warnings.warn(
                        "ARIMA{} failed ({}), falling back to white noise "
                        "residuals".format(candidate, err)
                    )
                    self.order_ = (0, 0, 0)
                    return fit_quietly(
                        ARIMA(values, exog=exog, order=(0, 0, 0), trend="c")
                    )
                continue
            if i < len(candidates) - 1 and fallbacks and is_degenerate(res):
                continue
            self.order_ = candidate
            return res

    # -------------------------------------------------------------- predict

    def predict(self, steps, interval=None):
        """
        Forecast ``steps`` time steps beyond the end of the training data.

        :param steps: forecast horizon.
        :param interval: coverage of the prediction interval, defaults to
            the value given to the constructor.
        :return: DataFrame indexed by future time with columns forecast,
            lower and upper.
        """
        self._check_fitted()
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        steps = int(steps)
        interval = self.interval if interval is None else interval
        t = np.arange(self.n_, self.n_ + steps)
        X = self._features(t)

        if self._result is None:
            mean = _with_constant(X) @ self._beta
            se = np.zeros(steps)
        else:
            res = self._result
            fc = res.get_forecast(steps, exog=X.to_numpy() if X.shape[1] else None)
            mean = np.asarray(fc.predicted_mean)
            se = np.sqrt(np.asarray(fc.var_pred_mean) + self._parameter_variance(X))

        # Student t quantiles allow for estimating the noise level, which
        # matters for short series
        dof = max(self._n_obs - self._n_params, 1)
        z = stats.t.ppf(0.5 + interval / 2, dof)
        out = pd.DataFrame(
            {"forecast": mean, "lower": mean - z * se, "upper": mean + z * se},
            index=self._future_index(steps),
        )
        if self.log:
            out = np.exp(out)
        out.attrs["interval"] = interval
        return out

    def _parameter_variance(self, X_future):
        """
        Extra forecast variance from uncertainty in the trend and seasonal
        coefficients, which the ARIMA state space forecast leaves out. This
        matters most for trend extrapolation over long horizons.
        """
        res = self._result
        names = list(res.model.exog_names or [])
        if not names:
            return 0.0
        pos = [list(res.model.param_names).index(name) for name in names]
        cov = np.asarray(res.cov_params())[np.ix_(pos, pos)]
        if not np.all(np.isfinite(cov)):
            return 0.0
        if self.seasonal_order_ is not None:
            x = self._forecast_gradient(X_future, pos)
            return np.einsum("ij,jk,ik->i", x, cov, x)
        has_const = "const" in names
        x = X_future.to_numpy()
        if self.order_[1] > 0:
            # with differencing the level is anchored to the last observation,
            # so only the change in the features since then is uncertain
            x = x - self._features(np.array([self.n_ - 1])).to_numpy()
        if has_const:
            x = np.hstack([np.ones((len(x), 1)), x])
        return np.einsum("ij,jk,ik->i", x, cov, x)

    def _forecast_gradient(self, X_future, pos):
        """
        Change in the forecast per unit change in each regression
        coefficient, holding the ARIMA parameters fixed. The forecast is
        linear in these coefficients, so a unit step is exact.

        :param X_future: features over the forecast horizon.
        :param pos: positions of the regression coefficients in the
            parameter vector.
        :return: array of shape (steps, len(pos)).
        """
        res = self._result
        exog = X_future.to_numpy()
        base = np.asarray(res.get_forecast(len(exog), exog=exog).predicted_mean)
        params = np.asarray(res.params)
        columns = []
        for i in pos:
            shifted = params.copy()
            shifted[i] += 1.0
            moved = res.model.filter(shifted).get_forecast(len(exog), exog=exog)
            columns.append(np.asarray(moved.predicted_mean) - base)
        return np.column_stack(columns)

    def _future_index(self, steps):
        idx = self.history_.index
        if isinstance(idx, pd.DatetimeIndex):
            return pd.date_range(idx[-1], periods=steps + 1, freq=self.freq_)[1:]
        return pd.RangeIndex(idx[-1] + 1, idx[-1] + 1 + steps)

    # ------------------------------------------------------------ reporting

    @property
    def components_(self):
        """
        What the model chose, as a dict.
        """
        self._check_fitted()
        seasonal = {}
        for g in self.groups_:
            if g.kind == "season":
                seasonal[g.period] = max(seasonal.get(g.period, 0), g.harmonic)
        return {
            "trend": any(g.kind == "trend" for g in self.groups_),
            "seasonal_periods": seasonal,
            "detected_periods": sorted({g.period for g in self.groups_ if g.detected}),
            "evolving_period": (
                float(self.seasonal_order_[3]) if self.seasonal_order_ else None
            ),
            "arima_order": self.order_,
            "seasonal_order": self.seasonal_order_,
        }

    @property
    def fitted_(self):
        """
        In-sample one step ahead predictions.
        """
        self._check_fitted()
        if self._result is None:
            values = _with_constant(self._features(np.arange(self.n_))) @ self._beta
        else:
            values = np.asarray(self._result.fittedvalues)
        if self.log:
            values = np.exp(values)
        return pd.Series(values, index=self.history_.index, name="fitted")

    def summary(self):
        """
        A short human readable description of the fitted model.
        """
        c = self.components_
        lines = ["Divinity forecaster"]
        missing = " ({} missing)".format(self.n_missing_) if self.n_missing_ else ""
        lines.append("  observations   {}{}".format(self.n_, missing))
        if self.freq_:
            lines.append("  frequency      {}".format(self.freq_))
        lines.append("  log scale      {}".format("yes" if self.log else "no"))
        lines.append("  trend          {}".format("linear" if c["trend"] else "none"))
        parts = []
        if c["evolving_period"]:
            parts.append("{:g} (evolving)".format(c["evolving_period"]))
        for p, k in c["seasonal_periods"].items():
            tag = ", detected" if p in c["detected_periods"] else ""
            parts.append("{:g} ({} harmonic{}{})".format(p, k, "s" * (k > 1), tag))
        lines.append("  seasonality    " + (", ".join(parts) or "none"))
        if c["arima_order"] is None:
            lines.append("  residuals      none (series is deterministic)")
        elif c["seasonal_order"]:
            P, D, Q, m = c["seasonal_order"]
            lines.append(
                "  residuals      ARIMA{}({}, {}, {})[{}]".format(
                    c["arima_order"], P, D, Q, m
                )
            )
        else:
            lines.append("  residuals      ARIMA{}".format(c["arima_order"]))
        return "\n".join(lines)

    def __repr__(self):
        state = "fitted" if hasattr(self, "groups_") else "not fitted"
        return "Forecaster({})".format(state)

    def plot(
        self, forecast, actual=None, ax=None, history=None, title=None, highlight=None
    ):
        """
        Plot the training data, the forecast and its prediction interval.

        :param forecast: DataFrame returned by :meth:`predict`.
        :param actual: optional Series or array of the true future values,
            drawn for comparison.
        :param ax: matplotlib Axes to draw on. A new figure is made if None.
        :param history: number of training points to show (default all).
        :param title: optional title.
        :param highlight: optional Series of points to mark, e.g. readings
            that fell outside the prediction interval.
        :return: the matplotlib Axes.
        """
        from .plotting import plot_forecast

        hist = self.history_ if history is None else self.history_.iloc[-history:]
        return plot_forecast(
            hist, forecast, actual=actual, ax=ax, title=title, highlight=highlight
        )

    def _check_fitted(self):
        if not hasattr(self, "groups_"):
            raise RuntimeError("call fit() before using the model")


def forecast(y, steps, freq=None, **kwargs):
    """
    One line forecast. Fits a :class:`Forecaster` and returns its prediction.

    :param y: series to forecast (Series, array or list).
    :param steps: forecast horizon.
    :param freq: pandas frequency, only needed if it cannot be inferred.
    :param kwargs: passed to :class:`Forecaster`.
    :return: DataFrame with columns forecast, lower and upper.
    """
    return Forecaster(**kwargs).fit(y, freq=freq).predict(steps)


def _as_series(y, freq):
    """
    Normalise input to a float Series and work out its frequency.
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("pass a single column, not a DataFrame")
        y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        series = y.astype(float)
    else:
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 1:
            raise ValueError("y must be one dimensional")
        series = pd.Series(arr)

    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        if not idx.is_monotonic_increasing:
            series = series.sort_index()
            idx = series.index
        if freq is None:
            freq = idx.freqstr or pd.DatetimeIndex(idx, freq="infer").freqstr
        if freq is None:
            raise ValueError(
                "could not infer the frequency of the DatetimeIndex, probably "
                "because some dates are missing. Pass freq, e.g. fit(y, freq='D'), "
                "and the gaps will be treated as missing values."
            )
        series = series.asfreq(freq)
        return series, series.index.freqstr
    if not pd.api.types.is_integer_dtype(idx) or not (np.diff(idx) == 1).all():
        series = series.reset_index(drop=True)
    return series, freq


def _divides(period, m, tol=0.05):
    """
    True if ``period`` is within ``tol`` (relative) of m / k for a whole k.
    """
    k = max(round(m / period), 1)
    return abs(period - m / k) <= tol * m / k


def _with_constant(X):
    return np.hstack([np.ones((len(X), 1)), X.to_numpy()])


def _ols_beta(X, y):
    observed = ~np.isnan(y)
    beta, *_ = np.linalg.lstsq(_with_constant(X)[observed], y[observed], rcond=None)
    return beta


def _ols_residuals(X, y):
    return y - _with_constant(X) @ _ols_beta(X, y)
