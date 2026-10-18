"""Baseline predictors. These are the numbers the machine learning model has to beat.

Riegel's formula scales a time from one distance to another:

    T2 = T1 * (D2 / D1) ** 1.06

Here it is applied to the first half of the race to predict the full marathon.
Because 1.06 is greater than 1, it already assumes runners slow down a little.
"""
from __future__ import annotations

import numpy as np

MARATHON_KM = 42.195
HALF_KM = MARATHON_KM / 2
RIEGEL_EXPONENT = 1.06


def riegel(time_s, from_km: float, to_km: float, exponent: float = RIEGEL_EXPONENT):
    """Predict the time (seconds) for distance to_km from a time at distance from_km."""
    return np.asarray(time_s, dtype=float) * (to_km / from_km) ** exponent


def riegel_from_half(first_half_s, exponent: float = RIEGEL_EXPONENT):
    """Classic baseline: predict the finish time from the first half split."""
    return riegel(first_half_s, HALF_KM, MARATHON_KM, exponent)


def fit_fade_factor(first_half_s, finish_s) -> float:
    """Calibrated baseline: the typical finish time divided by first half time,
    measured on training data only. A stronger yardstick than plain Riegel,
    because it is tuned to this dataset."""
    first_half_s = np.asarray(first_half_s, dtype=float)
    finish_s = np.asarray(finish_s, dtype=float)
    return float(np.median(finish_s / first_half_s))


def calibrated_predict(first_half_s, factor: float):
    return np.asarray(first_half_s, dtype=float) * factor


def error_summary(actual_s, predicted_s) -> dict:
    """Errors in minutes. Positive bias means the predictions run slower than reality."""
    actual_s = np.asarray(actual_s, dtype=float)
    predicted_s = np.asarray(predicted_s, dtype=float)
    err_min = (predicted_s - actual_s) / 60
    return {
        "n": int(len(err_min)),
        "mae_min": float(np.mean(np.abs(err_min))),
        "rmse_min": float(np.sqrt(np.mean(err_min**2))),
        "median_abs_error_min": float(np.median(np.abs(err_min))),
        "bias_min": float(np.mean(err_min)),
    }
