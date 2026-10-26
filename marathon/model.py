"""Week 2: machine learning models that try to beat the baselines.

Every model predicts the finish time from the same few inputs the baselines can see:
first half time, age, and gender. Nothing from the second half is ever used, because
that would leak the answer.

Instead of predicting the finish time directly, the models predict the "fade ratio"
(finish divided by first half, on a log scale) and multiply it back by the first half
time. That keeps the model focused on how much a runner slows down, which is the thing
we actually do not know.

Rules that keep the score honest:
1. Settings are chosen with cross validation on the TRAIN rows only.
2. The test rows are touched once, at the end, to report the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

FEATURES = ["log_first_half", "age", "age_sq", "is_female", "is_x"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned rows into model inputs. Uses first half, age and gender only."""
    out = pd.DataFrame(index=df.index)
    out["log_first_half"] = np.log(df["first_half"].astype(float))
    out["age"] = df["age"].astype(float)
    out["age_sq"] = out["age"] ** 2 / 100  # scaled so the ridge penalty treats it fairly
    out["is_female"] = (df["gender"] == "F").astype(float)
    out["is_x"] = (df["gender"] == "X").astype(float)
    return out[FEATURES]


def fade_target(df: pd.DataFrame) -> np.ndarray:
    """Log of finish divided by first half. This is what the models learn."""
    return np.log(df["finish"].to_numpy(float) / df["first_half"].to_numpy(float))


def to_finish_seconds(df: pd.DataFrame, log_ratio) -> np.ndarray:
    return df["first_half"].to_numpy(float) * np.exp(np.asarray(log_ratio, dtype=float))


def make_ridge(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)


def make_boosted(learning_rate: float = 0.05, max_depth: int = 3, max_iter: int = 300, seed: int = 42):
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def make_quantile(q: float, seed: int = 42):
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        learning_rate=0.05,
        max_depth=3,
        max_iter=200,
        min_samples_leaf=40,
        random_state=seed,
    )


def cv_mae_minutes(make_model, train: pd.DataFrame, folds: int = 5, seed: int = 42) -> float:
    """Average miss in minutes, measured only inside the training rows."""
    X, y = build_features(train), fade_target(train)
    misses = []
    for fit_idx, val_idx in KFold(folds, shuffle=True, random_state=seed).split(X):
        model = make_model().fit(X.iloc[fit_idx], y[fit_idx])
        val = train.iloc[val_idx]
        pred = to_finish_seconds(val, model.predict(X.iloc[val_idx]))
        misses.append(np.abs(pred - val["finish"].to_numpy(float)) / 60)
    return float(np.mean(np.concatenate(misses)))


def pick_best(candidates: dict, train: pd.DataFrame, folds: int = 5, seed: int = 42):
    """candidates maps a label to a function that builds a fresh model.
    Returns (best label, table of cross validation scores)."""
    scores = {label: cv_mae_minutes(maker, train, folds, seed) for label, maker in candidates.items()}
    best = min(scores, key=scores.get)
    return best, scores


def fit_and_predict(make_model, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit on train, return predicted finish times (seconds) for the test rows."""
    model = make_model().fit(build_features(train), fade_target(train))
    return to_finish_seconds(test, model.predict(build_features(test)))


def fit_interval(train: pd.DataFrame, test: pd.DataFrame, low_q: float = 0.1, high_q: float = 0.9):
    """Predict a range (seconds) meant to hold the real finish about 80 percent of the time."""
    X, y = build_features(train), fade_target(train)
    low = make_quantile(low_q).fit(X, y).predict(build_features(test))
    high = make_quantile(high_q).fit(X, y).predict(build_features(test))
    low_s, high_s = to_finish_seconds(test, low), to_finish_seconds(test, high)
    return np.minimum(low_s, high_s), np.maximum(low_s, high_s)


def interval_coverage(actual_s, low_s, high_s) -> dict:
    actual_s, low_s, high_s = (np.asarray(a, dtype=float) for a in (actual_s, low_s, high_s))
    inside = (actual_s >= low_s) & (actual_s <= high_s)
    return {
        "coverage": float(inside.mean()),
        "mean_width_min": float(np.mean(high_s - low_s) / 60),
    }


def paired_bootstrap_gain(actual_s, base_pred_s, model_pred_s, n_boot: int = 2000, seed: int = 42) -> dict:
    """How much smaller is the model's average miss than the baseline's, and how sure are we?
    Resamples the test runners and recomputes the gap each time. Positive means the model wins."""
    actual_s, base_pred_s, model_pred_s = (np.asarray(a, dtype=float) for a in (actual_s, base_pred_s, model_pred_s))
    base_err = np.abs(base_pred_s - actual_s) / 60
    model_err = np.abs(model_pred_s - actual_s) / 60
    diff = base_err - model_err
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return {
        "gain_min": float(diff.mean()),
        "ci95_low_min": float(np.percentile(boots, 2.5)),
        "ci95_high_min": float(np.percentile(boots, 97.5)),
    }
