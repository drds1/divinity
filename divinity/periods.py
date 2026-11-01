"""
Candidate seasonal periods, from the series frequency and from the data.
"""

import numpy as np
import pandas as pd
from scipy import signal


def default_periods(freq):
    """
    Sensible seasonal periods (in time steps) for a pandas frequency.
    Returns an empty list if the frequency is unknown.
    """
    if freq is None:
        return []
    offset = pd.tseries.frequencies.to_offset(freq)
    if offset.n != 1:
        return []
    o = pd.offsets
    if isinstance(offset, o.BusinessDay):
        return [5.0, 261.0]
    if isinstance(offset, o.Day):
        return [7.0, 365.25]
    if isinstance(offset, o.Hour):
        return [24.0, 168.0]
    if isinstance(offset, o.Minute):
        return [60.0, 1440.0]
    if isinstance(offset, o.Week):
        return [365.25 / 7]
    if isinstance(offset, (o.MonthBegin, o.MonthEnd)):
        return [12.0]
    if isinstance(offset, (o.QuarterBegin, o.QuarterEnd)):
        return [4.0]
    return []


def detect_periods(y, max_periods=3, min_period=2.0, max_period=None, oversample=8):
    """
    Find the strongest periodicities in a series from its periodogram.

    The series is linearly detrended first (gaps are interpolated for this
    step only). The periodogram is zero padded so that periods falling
    between Fourier frequencies are located accurately.

    :param y: 1D array, may contain NaN.
    :param max_periods: maximum number of periods to return.
    :param min_period: shortest period considered, in time steps.
    :param max_period: longest period considered. Defaults to half the
        series length, so at least two full cycles are observed.
    :return: list of periods, strongest first.
    """
    y = pd.Series(np.asarray(y, dtype=float)).interpolate(limit_direction="both")
    y = y.to_numpy()
    n = len(y)
    if n < 8 or np.allclose(y, y[0]):
        return []
    if max_period is None:
        max_period = n / 2
    y = signal.detrend(y)
    freqs, power = signal.periodogram(y, nfft=oversample * n, detrend=False)
    with np.errstate(divide="ignore"):
        periods = 1.0 / freqs
    valid = (periods >= min_period) & (periods <= max_period)
    if not np.any(valid):
        return []
    peaks, _ = signal.find_peaks(np.where(valid, power, 0.0))
    peaks = peaks[valid[peaks]]
    order = peaks[np.argsort(power[peaks])[::-1]]

    found = []
    for idx in order:
        p = periods[idx]
        # skip harmonics and near duplicates of periods already found
        if any(_is_related(p, q) for q in found):
            continue
        found.append(float(p))
        if len(found) == max_periods:
            break
    return found


def _is_related(p, q, tol=0.05):
    """
    True if period p is within ``tol`` of q or of one of its harmonics q/k.
    """
    for k in range(1, 6):
        if abs(p - q / k) <= tol * q / k:
            return True
    return False


def merge_periods(known, detected, n, tol=0.05):
    """
    Combine known and detected periods, dropping detected periods that match
    a known one and any period that cannot be seen twice in ``n`` steps.

    :return: (known, detected) with invalid entries removed.
    """
    known = [float(p) for p in known if 2 <= p <= n / 2]
    detected = [
        float(p)
        for p in detected
        if 2 <= p <= n / 2 and not any(_is_related(p, q, tol) for q in known)
    ]
    return known, detected
