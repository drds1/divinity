import numpy as np

from divinity.residuals import has_unit_root, select_order


def test_unit_root_found_in_random_walks():
    hits = sum(
        has_unit_root(np.cumsum(np.random.default_rng(s).normal(size=300)))
        for s in range(20)
    )
    assert hits >= 17


def test_no_unit_root_in_stationary_ar1():
    hits = 0
    for s in range(20):
        e = np.random.default_rng(s).normal(size=300)
        u = np.zeros_like(e)
        for i in range(1, len(e)):
            u[i] = 0.7 * u[i - 1] + e[i]
        hits += has_unit_root(u)
    assert hits <= 2


def test_unit_root_handles_gaps_and_constants():
    y = np.cumsum(np.random.default_rng(0).normal(size=200))
    y[50:60] = np.nan
    assert has_unit_root(y)
    assert not has_unit_root(np.ones(50))


def test_select_order_recovers_ar1():
    e = np.random.default_rng(1).normal(size=500)
    u = np.zeros_like(e)
    for i in range(1, len(e)):
        u[i] = 0.6 * u[i - 1] + e[i]
    order, result = select_order(u, d=0)
    assert order == (1, 0, 0)
    assert abs(result.params[1] - 0.6) < 0.1


def test_fill_short_gaps_leaves_long_gaps():
    from divinity.residuals import fill_short_gaps

    r = np.arange(12, dtype=float)
    r[[2, 5, 6]] = np.nan
    r[8:11] = np.nan
    out = fill_short_gaps(r)
    np.testing.assert_allclose(out[:8], np.arange(8))
    assert np.isnan(out[8:11]).all()
