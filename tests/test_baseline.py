import math

import numpy as np

from marathon.baseline import (
    HALF_KM,
    MARATHON_KM,
    calibrated_predict,
    error_summary,
    fit_fade_factor,
    riegel,
    riegel_from_half,
)


def test_riegel_same_distance_returns_same_time():
    assert riegel(3600, 10, 10) == 3600


def test_riegel_from_half_matches_formula():
    expected = 3600 * (MARATHON_KM / HALF_KM) ** 1.06
    assert math.isclose(float(riegel_from_half(3600)), expected)


def test_riegel_predicts_slower_than_simple_doubling():
    # An exponent above 1 means runners are assumed to slow down.
    assert riegel_from_half(3600) > 2 * 3600


def test_fade_factor_is_median_ratio():
    first = [3600, 3600, 3600]
    finish = [7200, 7560, 9000]
    assert math.isclose(fit_fade_factor(first, finish), 7560 / 3600)


def test_calibrated_predict_scales_first_half():
    assert np.allclose(calibrated_predict([3600, 4000], 2.1), [7560, 8400])


def test_error_summary_known_values():
    actual = [3600, 7200]
    predicted = [3660, 7080]  # +1 minute and -2 minutes
    s = error_summary(actual, predicted)
    assert s["n"] == 2
    assert math.isclose(s["mae_min"], 1.5)
    assert math.isclose(s["bias_min"], -0.5)
    assert math.isclose(s["rmse_min"], math.sqrt(2.5))
    assert math.isclose(s["median_abs_error_min"], 1.5)


def test_perfect_predictions_have_zero_error():
    s = error_summary([10000, 12000], [10000, 12000])
    assert s["mae_min"] == 0 and s["rmse_min"] == 0 and s["bias_min"] == 0
