"""
Example series for trying divinity on practical problems.

Real data (bundled with statsmodels, no download needed)

- :func:`co2` weekly atmospheric CO2 at Mauna Loa, 1958 to 2001
- :func:`sunspots` yearly sunspot activity, 1700 to 2008
- :func:`elnino` monthly Pacific sea surface temperature, 1950 to 2010
- :func:`us_macro` quarterly US macroeconomic series, 1959 to 2009

Simulated data (realistic shapes, generated from a seed)

- :func:`retail_sales` daily shop sales
- :func:`web_traffic` hourly website visits
- :func:`monthly_revenue` monthly revenue of a growing business
- :func:`sensor_readings` hourly temperature sensor with a fault

Every function returns a pandas Series with a DatetimeIndex.
"""

import numpy as np
import pandas as pd


def co2():
    """
    Weekly atmospheric CO2 concentration (ppm) at Mauna Loa, Hawaii,
    March 1958 to December 2001. Contains 59 missing weeks.
    """
    from statsmodels.datasets import co2 as source

    y = source.load_pandas().data["co2"]
    y.index.freq = "W-SAT"
    return y.rename("co2")


def sunspots():
    """
    Yearly mean sunspot activity, 1700 to 2008.
    """
    from statsmodels.datasets import sunspots as source

    data = source.load_pandas().data
    index = pd.date_range("1700", periods=len(data), freq="YS")
    return pd.Series(data["SUNACTIVITY"].to_numpy(), index=index, name="sunspots")


def elnino():
    """
    Monthly sea surface temperature (degrees C) in the El Nino region of the
    Pacific, January 1950 to December 2010.
    """
    from statsmodels.datasets import elnino as source

    data = source.load_pandas().data.set_index("YEAR")
    values = data.to_numpy().ravel()
    index = pd.date_range("1950-01-01", periods=len(values), freq="MS")
    return pd.Series(values, index=index, name="sea_temperature")


def us_macro(column="realgdp"):
    """
    Quarterly US macroeconomic data, 1959 Q1 to 2009 Q3.

    :param column: one of the statsmodels macrodata columns, e.g. "realgdp"
        (real GDP, billions of 2005 dollars), "unemp" (unemployment rate),
        "cpi", "tbilrate" (3 month Treasury bill rate) or "infl".
    """
    from statsmodels.datasets import macrodata as source

    data = source.load_pandas().data
    if column not in data.columns or column in ("year", "quarter"):
        raise ValueError("unknown column {!r}".format(column))
    index = pd.date_range("1959-01-01", periods=len(data), freq="QS")
    return pd.Series(data[column].to_numpy(), index=index, name=column)


def retail_sales(seed=0, start="2022-01-01", days=3 * 365 + 90):
    """
    Simulated daily sales (units) for a shop. Busy Fridays and Saturdays,
    quiet Sundays, a summer lull, a run up to Christmas, steady growth and
    day to day noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    weekday = np.array([0.0, 2.0, 4.0, 6.0, 18.0, 24.0, -14.0])[dates.day_of_week]
    doy = dates.day_of_year.to_numpy()
    yearly = 12 * np.cos(2 * np.pi * (doy - 350) / 365.25)
    yearly += 10 * np.cos(4 * np.pi * (doy - 350) / 365.25)
    y = 120 + 0.03 * t + weekday + yearly + rng.normal(0, 6, days)
    return pd.Series(np.round(y), index=dates, name="units_sold")


def web_traffic(seed=0, start="2024-09-02", weeks=10):
    """
    Simulated hourly visits to a website. Traffic peaks mid morning and
    early evening, is low overnight and lighter at weekends, and grows
    slowly.
    """
    rng = np.random.default_rng(seed)
    hours = weeks * 7 * 24
    dates = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    h = dates.hour.to_numpy()
    daily = 900 * np.exp(-((h - 10) ** 2) / 8) + 700 * np.exp(-((h - 19) ** 2) / 6)
    weekend = np.where(dates.day_of_week >= 5, 0.6, 1.0)
    level = 300 + 0.05 * t
    noise = np.zeros(hours)
    for i in range(1, hours):
        noise[i] = 0.6 * noise[i - 1] + rng.normal(0, 40)
    y = (level + daily) * weekend + noise
    return pd.Series(np.round(np.maximum(y, 0)), index=dates, name="visits")


def monthly_revenue(seed=0, start="2015-01-01", years=10):
    """
    Simulated monthly revenue (thousands of pounds) for a business growing
    about 12% a year, with a seasonal swing that grows with it.
    """
    rng = np.random.default_rng(seed)
    months = 12 * years
    dates = pd.date_range(start, periods=months, freq="MS")
    t = np.arange(months)
    log_y = (
        np.log(200)
        + np.log(1.12) / 12 * t
        + 0.15 * np.sin(2 * np.pi * (t - 2) / 12)
        + 0.06 * np.cos(4 * np.pi * t / 12)
        + rng.normal(0, 0.03, months)
    )
    return pd.Series(np.round(np.exp(log_y), 1), index=dates, name="revenue")


def sensor_readings(seed=0, start="2024-06-01", days=31, fault_day=29):
    """
    Simulated hourly temperature readings (degrees C) from a sensor with a
    daily cycle, slowly varying weather and a few dropped readings. On
    ``fault_day`` (counted from zero) the sensor develops a fault and its
    readings drift upwards for six hours.

    :return: Series of readings. ``attrs["fault"]`` holds the timestamps of
        the faulty readings.
    """
    rng = np.random.default_rng(seed)
    hours = days * 24
    dates = pd.date_range(start, periods=hours, freq="h")
    h = dates.hour.to_numpy()
    daily = 4 * np.cos(2 * np.pi * (h - 15) / 24)
    weather = np.zeros(hours)
    for i in range(1, hours):
        weather[i] = 0.95 * weather[i - 1] + rng.normal(0, 0.25)
    y = 18 + daily + weather + rng.normal(0, 0.3, hours)
    fault = np.arange(fault_day * 24 + 9, fault_day * 24 + 15)
    y[fault] += np.linspace(2.5, 6.0, len(fault))
    dropped = rng.choice(fault_day * 24, size=12, replace=False)
    y[dropped] = np.nan
    out = pd.Series(np.round(y, 2), index=dates, name="temperature")
    out.attrs["fault"] = list(dates[fault])
    return out
