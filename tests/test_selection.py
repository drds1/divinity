import numpy as np

from divinity.features import FeatureGroup, seasonal_groups
from divinity.selection import forward_select, lag1_autocorrelation


def _candidates(periods):
    groups = [FeatureGroup("trend")] + seasonal_groups(periods, 2)
    pool = {g: len(periods) if g.harmonic == 1 else 2 for g in groups}
    pool[groups[0]] = 1
    return groups, pool


def test_finds_true_period_and_trend():
    rng = np.random.default_rng(0)
    t = np.arange(200)
    y = 0.02 * t + np.sin(2 * np.pi * t / 10) + rng.normal(0, 0.3, 200)
    groups, pool = _candidates([10, 13, 31])
    chosen = {g.name for g in forward_select(y, groups, pool).groups}
    assert chosen == {"trend", "P=10,k=1"}


def test_white_noise_selects_nothing_most_of_the_time():
    groups, pool = _candidates([5, 7, 11, 13, 17, 23, 29])
    hits = 0
    for seed in range(30):
        y = np.random.default_rng(seed).normal(size=150)
        hits += bool(forward_select(y, groups, pool).groups)
    assert hits <= 3


def test_gls_rejects_spurious_trend_on_random_walk():
    groups, pool = _candidates([12, 30])
    ols_hits = gls_hits = 0
    for seed in range(20):
        y = np.cumsum(np.random.default_rng(seed).normal(size=200))
        ols_hits += bool(forward_select(y, groups, pool).groups)
        gls_hits += bool(forward_select(y, groups, pool, rho=1.0).groups)
    assert ols_hits >= 15
    assert gls_hits <= 3


def test_forced_groups_always_present():
    y = np.random.default_rng(0).normal(size=100)
    trend = FeatureGroup("trend")
    result = forward_select(y, [], {}, forced=[trend])
    assert result.groups == [trend]


def test_lag1_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=5000)
    u = np.zeros_like(e)
    for i in range(1, len(e)):
        u[i] = 0.7 * u[i - 1] + e[i]
    assert abs(lag1_autocorrelation(u) - 0.7) < 0.03
    u[::7] = np.nan
    assert abs(lag1_autocorrelation(u) - 0.7) < 0.03
