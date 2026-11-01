import numpy as np
import pandas as pd
import pytest

import divinity as dv
from divinity import synthetic


@pytest.fixture(scope="module")
def seasonal():
    return synthetic.make("trend_seasonal", seed=3)


@pytest.fixture(scope="module")
def fitted(seasonal):
    return dv.Forecaster().fit(seasonal.history)


def test_predict_returns_tidy_frame(fitted, seasonal):
    fc = fitted.predict(10)
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert len(fc) == 10
    assert list(fc.index) == list(range(200, 210))
    assert (fc["lower"] < fc["forecast"]).all() and (fc["forecast"] < fc["upper"]).all()
    assert fc.attrs["interval"] == 0.95


@pytest.mark.parametrize("kind", [list, np.array, pd.Series])
def test_accepts_lists_arrays_and_series(kind, seasonal):
    y = kind(seasonal.history.to_numpy().tolist())
    fc = dv.Forecaster().fit(y).predict(5)
    assert fc.index[0] == 200


def test_forecast_does_not_depend_on_horizon(fitted):
    short = fitted.predict(10)
    long = fitted.predict(60)
    pd.testing.assert_frame_equal(short, long.iloc[:10], check_freq=False)


def test_narrower_interval_is_narrower(fitted):
    wide = fitted.predict(10, interval=0.95)
    narrow = fitted.predict(10, interval=0.5)
    assert ((narrow["upper"] - narrow["lower"]) < (wide["upper"] - wide["lower"])).all()


def test_one_line_forecast(seasonal):
    fc = dv.forecast(seasonal.history, steps=7)
    assert len(fc) == 7


def test_dates_carry_through():
    s = synthetic.make("short_daily")
    fc = dv.Forecaster().fit(s.history).predict(s.steps)
    pd.testing.assert_index_equal(fc.index, s.future.index, check_names=False)


def test_missing_dates_need_freq_then_become_gaps():
    s = synthetic.make("short_daily")
    y = s.history.drop(s.history.index[[5, 6, 20]])
    with pytest.raises(ValueError, match="freq"):
        dv.Forecaster().fit(y)
    model = dv.Forecaster().fit(y, freq="D")
    assert model.n_missing_ == 3
    assert model.predict(3).index[0] == s.future.index[0]


def test_unsorted_dates_are_sorted():
    s = synthetic.make("short_daily")
    y = s.history.sample(frac=1, random_state=0)
    fc = dv.Forecaster().fit(y).predict(2)
    assert fc.index[0] == s.future.index[0]


def test_random_walk_interval_widens(seed=0):
    s = synthetic.make("random_walk", seed=seed)
    fc = dv.Forecaster().fit(s.history).predict(50)
    width = fc["upper"] - fc["lower"]
    assert width.iloc[-1] > 3 * width.iloc[0]


def test_constant_series_gives_exact_forecast():
    model = dv.Forecaster().fit(np.full(30, 4.2))
    fc = model.predict(5)
    np.testing.assert_allclose(fc.to_numpy(), 4.2)
    assert model.components_["arima_order"] is None


def test_perfect_line_extrapolates():
    fc = dv.Forecaster().fit(np.arange(50.0)).predict(3)
    np.testing.assert_allclose(fc["forecast"], [50, 51, 52], atol=1e-6)


def test_log_scale_stays_positive():
    s = synthetic.make("monthly_multiplicative")
    model = dv.Forecaster(log=True).fit(s.history)
    fc = model.predict(s.steps)
    assert (fc > 0).all().all()
    assert "log scale      yes" in model.summary()


def test_fixed_options_are_respected(seasonal):
    model = dv.Forecaster(
        seasonal_periods=None, detect_periods=False, trend="linear", arima=(1, 0, 0)
    ).fit(seasonal.history)
    c = model.components_
    assert c["trend"] and not c["seasonal_periods"]
    assert c["arima_order"] == (1, 0, 0)


def test_no_trend_option(seasonal):
    model = dv.Forecaster(trend=None).fit(seasonal.history)
    assert not model.components_["trend"]


def test_white_noise_residuals_option(seasonal):
    model = dv.Forecaster(arima=None).fit(seasonal.history)
    assert model.components_["arima_order"] == (0, 0, 0)


@pytest.mark.parametrize(
    "y, kwargs, match",
    [
        (np.arange(5.0), {}, "at least"),
        (np.ones((10, 2)), {}, "one dimensional"),
        (np.r_[np.ones(20), -1.0], {"log": True}, "positive"),
    ],
)
def test_bad_input_gives_clear_errors(y, kwargs, match):
    with pytest.raises(ValueError, match=match):
        dv.Forecaster(**kwargs).fit(y)


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        dv.Forecaster(trend="cubic")
    with pytest.raises(ValueError):
        dv.Forecaster(interval=95)


def test_predict_before_fit():
    with pytest.raises(RuntimeError, match="fit"):
        dv.Forecaster().predict(3)


def test_summary_fitted_and_plot(fitted, seasonal):
    assert "trend          linear" in fitted.summary()
    assert "Forecaster(fitted)" == repr(fitted)
    assert len(fitted.fitted_) == len(seasonal.history)
    ax = fitted.plot(fitted.predict(40), actual=seasonal.future, history=100)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["history", "95% interval", "forecast", "actual"]
