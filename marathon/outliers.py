"""Look closely at the runners the Week 1 cleaner dropped.

Week 1 dropped 50 runners whose first half was under 35 percent of their finish time. That
means the second half took almost twice as long as the first. Two stories fit:

1. A timing error, such as a missed chip read at the halfway mat (the first half looks too fast).
2. A real blowup, such as an injury or a collapse late in the race.

This module does not decide which story is true for any one runner. It measures what it can:
how impossible the first half times look, and how much these runners would change the results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marathon.data import (
    MAX_AGE, MAX_FINISH_S, MAX_SPLIT_RATIO, MIN_AGE, MIN_FINISH_S, MIN_SPLIT_RATIO,
    REQUIRED, _normalize_gender, standardize_columns, to_seconds,
)

# The world record for the half marathon is about 57 minutes. A first half under 1 hour is
# not realistic for anyone in this field, so it points to a timing error.
IMPOSSIBLE_FIRST_HALF_S = 3600


def dropped_by_split_rule(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass every other cleaning rule but fail the split ratio rule."""
    df = standardize_columns(raw)[REQUIRED].copy()
    df["first_half"] = to_seconds(df["first_half"])
    df["finish"] = to_seconds(df["finish"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = _normalize_gender(df["gender"])
    df = df[df.notna().all(axis=1)]
    df = df[(df["finish"] > 0) & (df["first_half"] > 0)]
    df = df[df["finish"].between(MIN_FINISH_S, MAX_FINISH_S) & df["age"].between(MIN_AGE, MAX_AGE)]
    df = df.assign(split_ratio=df["first_half"] / df["finish"])
    out = df[~df["split_ratio"].between(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO)].copy()
    out["finish_min"] = out["finish"] / 60
    out["first_half_min"] = out["first_half"] / 60
    return out.reset_index(drop=True)


def describe(dropped: pd.DataFrame) -> dict:
    """Plain numbers about the dropped runners."""
    if dropped.empty:
        return {"n": 0}
    impossible = dropped["first_half"] < IMPOSSIBLE_FIRST_HALF_S
    return {
        "n": int(len(dropped)),
        "below_low_limit": int((dropped["split_ratio"] < MIN_SPLIT_RATIO).sum()),
        "above_high_limit": int((dropped["split_ratio"] > MAX_SPLIT_RATIO).sum()),
        "impossible_first_half": int(impossible.sum()),
        "median_split_ratio": float(dropped["split_ratio"].median()),
        "median_finish_min": float(dropped["finish_min"].median()),
        "median_first_half_min": float(dropped["first_half_min"].median()),
        "share_female": float((dropped["gender"] == "F").mean()),
    }


def effect_on_scores(dropped: pd.DataFrame, test: pd.DataFrame, test_pred_s: np.ndarray,
                     dropped_pred_s: np.ndarray) -> dict:
    """Average miss on the normal test runners, then with the dropped runners added.

    The predictions come from a model fit on the same training rows as before. The dropped
    runners were never in the training or test rows, so this only shows how the headline
    score would change if the rule had kept them in the test set.
    """
    base_err = np.abs(test_pred_s - test["finish"].to_numpy(float)) / 60
    drop_err = np.abs(dropped_pred_s - dropped["finish"].to_numpy(float)) / 60
    combined = np.concatenate([base_err, drop_err])
    return {
        "test_mae_min": float(base_err.mean()),
        "dropped_mae_min": float(drop_err.mean()),
        "combined_mae_min": float(combined.mean()),
        "change_min": float(combined.mean() - base_err.mean()),
    }
