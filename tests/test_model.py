import numpy as np
import pandas as pd
import pytest

from marathon.data import clean_results, split_train_test
from marathon.model import (
    FEATURES,
    build_features,
    fade_target,
    fit_and_predict,
    fit_interval,
    interval_coverage,
    make_boosted,
    make_ridge,
    paired_bootstrap_gain,
    to_finish_seconds,
)
from marathon.synthetic import make_synthetic_results


@pytest.fixture(scope="module")
def split():
    data, _ = clean_results(make_synthetic_results(n=3000, seed=3))
    return split_train_test(data, 0.2, seed=42)


def test_features_use_only_allowed_inputs(split):
    train, _ = split
    X = build_features(train)
    assert list(X.columns) == FEATURES
    # Nothing from the second half or the finish time may leak into the inputs.
    assert not any("finish" in c or "second" in c for c in X.columns)


def test_target_round_trip_recovers_finish(split):
    train, _ = split
    back = to_finish_seconds(train, fade_target(train))
    assert np.allclose(back, train["finish"].to_numpy(float))


def test_models_return_one_positive_prediction_per_test_row(split):
    train, test = split
    for maker in (lambda: make_ridge(1.0), lambda: make_boosted(0.1, 2, 50)):
        pred = fit_and_predict(maker, train, test)
        assert len(pred) == len(test)
        assert (pred > 0).all()


def test_model_beats_a_constant_guess_on_synthetic_data(split):
    train, test = split
    pred = fit_and_predict(lambda: make_ridge(1.0), train, test)
    model_miss = np.abs(pred - test["finish"]).mean()
    constant_miss = np.abs(train["finish"].median() - test["finish"]).mean()
    assert model_miss < constant_miss


def test_interval_is_ordered_and_covers_roughly_eighty_percent(split):
    train, test = split
    low, high = fit_interval(train, test)
    assert (low <= high).all()
    cov = interval_coverage(test["finish"], low, high)["coverage"]
    assert 0.6 < cov < 0.95


def test_interval_coverage_known_values():
    out = interval_coverage([10, 20, 30], [5, 25, 25], [15, 35, 35])
    assert out["coverage"] == pytest.approx(2 / 3)


def test_bootstrap_gain_is_positive_when_model_is_clearly_better():
    actual = np.full(200, 12000.0)
    base = actual + 600
    model = actual + 60
    gain = paired_bootstrap_gain(actual, base, model, n_boot=200)
    assert gain["gain_min"] == pytest.approx(9.0)
    assert gain["ci95_low_min"] > 0


def test_bootstrap_gain_ci_spans_zero_when_models_are_equal():
    rng = np.random.default_rng(0)
    actual = rng.normal(12000, 1500, 500)
    a = actual + rng.normal(0, 600, 500)
    b = actual + rng.normal(0, 600, 500)
    gain = paired_bootstrap_gain(actual, a, b, n_boot=300)
    assert gain["ci95_low_min"] < 0 < gain["ci95_high_min"]
