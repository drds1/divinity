import numpy as np
import pandas as pd
import pytest

from divinity import datasets


@pytest.mark.parametrize(
    "loader, freq, length",
    [
        (datasets.co2, "W-SAT", 2284),
        (datasets.sunspots, "YS-JAN", 309),
        (datasets.elnino, "MS", 732),
        (datasets.us_macro, "QS-JAN", 203),
        (datasets.retail_sales, "D", 3 * 365 + 90),
        (datasets.web_traffic, "h", 10 * 7 * 24),
        (datasets.monthly_revenue, "MS", 120),
        (datasets.sensor_readings, "h", 31 * 24),
    ],
)
def test_datasets_are_dated_series(loader, freq, length):
    y = loader()
    assert isinstance(y, pd.Series)
    assert isinstance(y.index, pd.DatetimeIndex)
    assert y.index.freqstr == freq
    assert len(y) == length


def test_simulated_datasets_are_reproducible():
    pd.testing.assert_series_equal(datasets.retail_sales(3), datasets.retail_sales(3))
    assert not datasets.retail_sales(3).equals(datasets.retail_sales(4))


def test_co2_keeps_its_gaps():
    assert datasets.co2().isna().sum() == 59


def test_us_macro_column_check():
    assert datasets.us_macro("unemp").name == "unemp"
    with pytest.raises(ValueError, match="unknown column"):
        datasets.us_macro("year")


def test_sensor_fault_is_recorded():
    y = datasets.sensor_readings()
    fault = y.attrs["fault"]
    assert len(fault) == 6
    assert (y.loc[fault] > y.drop(fault).quantile(0.99)).sum() >= 3
    assert np.isnan(y.to_numpy()).sum() == 12
