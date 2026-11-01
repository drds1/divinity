"""
Choosing the ARIMA model for whatever the trend and seasonality leave behind.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller


def has_unit_root(resid, alpha=0.05, trend=True):
    """
    Augmented Dickey-Fuller test for a unit root (random walk behaviour).

    Set ``trend`` to True if a trend was removed from the series (or could
    be present). Testing detrended data against the no-trend critical
    values rejects the unit root far too often, so random walks would be
    mistaken for mean reverting series.

    The null hypothesis is a unit root, so the series is treated as a random
    walk unless the test rejects at ``alpha``. When in doubt this errs
    towards differencing, which gives wider but honest intervals, rather
    than towards mean reversion, which gives confident wrong ones.
    """
    r = np.asarray(resid, dtype=float)
    observed = r[~np.isnan(r)]
    if len(observed) < 20 or np.allclose(observed, observed[0]):
        return False
    regression = "ct" if trend else "c"
    if np.isnan(r).any():
        r = fill_short_gaps(r)
    if np.isnan(r).any():
        p_value = _adf_with_gaps(r, regression)
        if p_value is not None:
            return p_value > alpha
        # too few complete rows, fall back to interpolating the gaps
        r = pd.Series(r).interpolate(limit_direction="both").to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_value = adfuller(r, regression=regression, autolag="AIC")[1]
    return p_value > alpha


def fill_short_gaps(r, max_gap=2):
    """
    Linearly interpolate runs of at most ``max_gap`` missing values that
    have data on both sides. Longer gaps are left as NaN.
    """
    r = np.asarray(r, dtype=float).copy()
    missing = np.isnan(r)
    filled = pd.Series(r).interpolate(limit_area="inside").to_numpy()
    i = 0
    while i < len(r):
        if missing[i]:
            j = i
            while j < len(r) and missing[j]:
                j += 1
            if j - i <= max_gap:
                r[i:j] = filled[i:j]
            i = j
        else:
            i += 1
    return r


def _adf_with_gaps(r, regression, max_lag=4, min_rows=30):
    """
    ADF test on a series with missing values, using only the rows where the
    value and all its lags are observed. Interpolating instead would turn
    long gaps into smooth straight lines, which look like a random walk.

    The lag length is chosen by AIC on a common sample, and the p-value
    comes from MacKinnon's tables, as in statsmodels' adfuller.

    :return: p-value, or None if there are too few complete rows.
    """
    n = len(r)
    dy = np.full(n, np.nan)
    dy[1:] = np.diff(r)
    t = np.arange(n, dtype=float)
    rows = np.arange(max_lag + 1, n)
    lags = np.column_stack([dy[rows - k] for k in range(1, max_lag + 1)])
    target = dy[rows]
    level = r[rows - 1]
    base = [level, np.ones(len(rows))]
    if regression == "ct":
        base.append(t[rows] / n)
    base = np.column_stack(base)
    ok = ~(np.isnan(target) | np.isnan(level) | np.isnan(lags).any(axis=1))
    if ok.sum() < min_rows:
        return None
    y, base, lags = target[ok], base[ok], lags[ok]
    m = len(y)

    best = None
    for k in range(max_lag + 1):
        X = np.hstack([base, lags[:, :k]])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ beta) ** 2))
        aic = m * np.log(max(rss, 1e-300) / m) + 2 * X.shape[1]
        if best is None or aic < best[0]:
            best = (aic, X, beta, rss)
    _, X, beta, rss = best
    sigma2 = rss / (m - X.shape[1])
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    t_stat = beta[0] / np.sqrt(cov[0, 0])
    return float(mackinnonp(t_stat, regression=regression, N=1))


def fit_quietly(model):
    """
    Fit a statsmodels model, silencing the routine convergence chatter.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return model.fit()


def rank_orders(resid, d, max_p=2, max_q=2):
    """
    Fit ARIMA(p, d, q) to the residuals over a small grid and rank the
    orders by BIC.

    :param resid: residual series (NaN allowed).
    :param d: differencing order decided beforehand (0 or 1).
    :return: list of (order, fitted results), best first.
    """
    fits = []
    trend = "c" if d == 0 else "n"
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                res = fit_quietly(ARIMA(resid, order=(p, d, q), trend=trend))
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(res.bic):
                fits.append(((p, d, q), res))
    return sorted(fits, key=lambda f: f[1].bic)


def select_order(resid, d, max_p=2, max_q=2):
    """
    The BIC best (p, d, q) for the residuals.

    :return: (order, fitted results), or ((0, d, 0), None) if nothing fits.
    """
    ranked = rank_orders(resid, d, max_p, max_q)
    return ranked[0] if ranked else ((0, d, 0), None)


def is_degenerate(res, tol=1.01):
    """
    True if a fitted ARIMA sits on the edge of its parameter space, with an
    AR or MA root on or next to the unit circle. Such fits usually have an
    AR root nearly cancelled by an MA root, which leaves the regression
    coefficients unidentified and the forecast intervals meaningless.
    """
    roots = [np.atleast_1d(res.arroots), np.atleast_1d(res.maroots)]
    for r in roots:
        if r.size and np.min(np.abs(r)) < tol:
            return True
    return not np.all(np.isfinite(np.diag(np.asarray(res.cov_params()))))
