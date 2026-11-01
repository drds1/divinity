import numpy as np

from divinity.features import (
    FeatureGroup,
    degenerate_columns,
    design_matrix,
    seasonal_groups,
)


def test_trend_scaled_by_training_length_not_horizon():
    g = FeatureGroup("trend")
    a = design_matrix(np.arange(10), [g], n_train=10)
    b = design_matrix(np.arange(10, 15), [g], n_train=10)
    assert a["trend"].iloc[-1] == 0.9
    assert b["trend"].iloc[0] == 1.0


def test_fourier_pair_values():
    g = FeatureGroup("season", 4.0, 1)
    X = design_matrix(np.arange(4), [g], n_train=4)
    np.testing.assert_allclose(X["sin(P=4,k=1)"], [0, 1, 0, -1], atol=1e-12)
    np.testing.assert_allclose(X["cos(P=4,k=1)"], [1, 0, -1, 0], atol=1e-12)


def test_harmonics_capped_by_period():
    groups = seasonal_groups([7, 3, 2], max_harmonics=3)
    counts = {p: sum(g.period == p for g in groups) for p in (7, 3, 2)}
    assert counts == {7: 3, 3: 1, 2: 1}


def test_period_two_sine_is_degenerate():
    groups = seasonal_groups([2], max_harmonics=1)
    assert degenerate_columns(groups, 20) == {"sin(P=2,k=1)"}
