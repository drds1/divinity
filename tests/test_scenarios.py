"""
Smoke tests on every synthetic scenario, plus slower checks that the
prediction intervals are honest across many random seeds.
"""

import os
import sys
import warnings

import numpy as np
import pytest

import divinity as dv
from divinity import synthetic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from coverage import run_one, structure_ok  # noqa: E402


@pytest.mark.parametrize("name", list(synthetic.SCENARIOS))
def test_scenario_smoke(name):
    s = synthetic.make(name, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = dv.Forecaster(**s.options).fit(s.history)
    fc = model.predict(s.steps)
    assert np.isfinite(fc.to_numpy()).all()
    assert structure_ok(name, model.components_), model.summary()


@pytest.mark.slow
@pytest.mark.parametrize("name", list(synthetic.SCENARIOS))
def test_interval_coverage(name):
    runs = [run_one((name, seed)) for seed in range(100, 120)]
    coverage = np.mean([r["coverage"] for r in runs])
    structure = np.mean([r["structure"] for r in runs])
    assert coverage >= 0.85, coverage
    # outliers inflate the noise estimate, so over-coverage is expected there
    if name != "outliers":
        assert coverage <= 0.995, coverage
    assert structure >= 0.8, structure
