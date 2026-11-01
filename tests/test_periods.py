import numpy as np
import pytest

from divinity.periods import default_periods, detect_periods, merge_periods


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("D", [7.0, 365.25]),
        ("h", [24.0, 168.0]),
        ("MS", [12.0]),
        ("ME", [12.0]),
        ("QS", [4.0]),
        ("B", [5.0, 261.0]),
        ("2D", []),
        (None, []),
    ],
)
def test_default_periods(freq, expected):
    assert default_periods(freq) == expected


def test_detects_non_integer_period():
    rng = np.random.default_rng(1)
    t = np.arange(300)
    y = 2 * np.sin(2 * np.pi * t / 17.3) + rng.normal(0, 1, 300)
    found = detect_periods(y, max_periods=1)
    assert found and abs(found[0] - 17.3) < 0.3


def test_detection_tolerates_missing_values():
    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 25)
    y[50:70] = np.nan
    assert abs(detect_periods(y, max_periods=1)[0] - 25) < 0.5


def test_constant_series_has_no_periods():
    assert detect_periods(np.ones(100)) == []


def test_merge_drops_duplicates_harmonics_and_long_periods():
    known, detected = merge_periods([7, 365.25], [7.1, 3.5, 30], n=200)
    assert known == [7.0]
    assert detected == [30.0]
