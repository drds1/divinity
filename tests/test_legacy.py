import numpy as np
import pytest

import divinity as dv


def test_version_01_interface_still_works():
    rng = np.random.default_rng(123456)
    n, n_train = 120, 100
    t = np.arange(n)
    y = 0.1 * t + np.sin(2 * np.pi / 20 * t) + rng.normal(0, 0.5, n)

    with pytest.warns(DeprecationWarning):
        model = dv.divinity(forecast_length=n - n_train, confidence_interval=70.0)
    model.fit(y[:n_train])
    y_forecast = model.predict()
    assert y_forecast.shape == (20,)
    assert model.ystd.shape == (20,)
    assert np.sqrt(np.mean((y_forecast - y[n_train:]) ** 2)) < 1.5
    assert model.forecast(50).shape == (50,)


def test_gen_fake_normalises_to_unit_range():
    out = dv.gen_fake(100, 1, [1, 0.5], [10, 20], [1, 1], [0, 0])
    features = out["features"]
    varying = features.loc[:, features.std() > 0]
    np.testing.assert_allclose(varying.min(), 0)
    np.testing.assert_allclose(varying.max(), 1)
    assert out["target"].shape == (100,)
