"""Load and clean marathon results.

The cleaner needs four fields: age, gender, first half time, and finish time.
Column names are matched loosely, so "First Half", "first_half" and "FIRST HALF"
all work. Times can be seconds (numbers) or clock strings like 1:45:30.

Every row that gets dropped is counted by reason, so the cleaning step can be
explained and defended later.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REQUIRED = ["age", "gender", "first_half", "finish"]

# Other names the same field might go by in different datasets.
ALIASES = {
    "age": ["age"],
    "gender": ["gender", "sex"],
    "first_half": ["first_half", "firsthalf", "half1", "first_half_time", "split_half"],
    "finish": ["finish", "finish_time", "official_time", "chip_time", "net_time"],
}

# Plausibility limits. Anything outside these is treated as a data error.
MIN_FINISH_S = 2 * 3600       # faster than 2:00:00 is not a normal marathon result
MAX_FINISH_S = 8 * 3600       # slower than 8:00:00 is far past typical course limits
MIN_AGE, MAX_AGE = 18, 90
MIN_SPLIT_RATIO, MAX_SPLIT_RATIO = 0.35, 0.65  # first half time divided by finish time


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns we need to a standard set of names."""
    df = raw.copy()
    df.columns = [_snake(c) for c in df.columns]

    rename = {}
    missing = []
    for target, options in ALIASES.items():
        found = next((c for c in options if c in df.columns), None)
        if found is None:
            missing.append(target)
        else:
            rename[found] = target
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}. Columns in the file: {list(df.columns)}"
        )
    return df.rename(columns=rename)


def _clock_to_seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        nums = [float(p) for p in str(value).strip().split(":")]
    except ValueError:
        return np.nan
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, (m, s) = 0.0, nums
    else:
        return np.nan
    return h * 3600 + m * 60 + s


def to_seconds(series: pd.Series) -> pd.Series:
    """Turn a column of times into float seconds. Accepts seconds or H:MM:SS text."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return numeric.astype(float)
    return series.map(_clock_to_seconds).astype(float)


def _normalize_gender(series: pd.Series) -> pd.Series:
    first = series.astype("string").str.strip().str.upper().str[0]
    return first.where(first.isin(["M", "F", "X"]))


def clean_results(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean table, report of what was dropped and why)."""
    df = standardize_columns(raw)[REQUIRED].copy()
    df["first_half"] = to_seconds(df["first_half"])
    df["finish"] = to_seconds(df["finish"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = _normalize_gender(df["gender"])

    report: dict = {"rows_in": int(len(df)), "dropped": {}}

    def keep(reason: str, mask: pd.Series) -> None:
        nonlocal df
        report["dropped"][reason] = int((~mask).sum())
        df = df[mask]

    keep("missing_values", df.notna().all(axis=1))
    keep("did_not_finish_or_zero_time", (df["finish"] > 0) & (df["first_half"] > 0))
    keep("finish_time_out_of_range", df["finish"].between(MIN_FINISH_S, MAX_FINISH_S))
    keep("age_out_of_range", df["age"].between(MIN_AGE, MAX_AGE))
    keep(
        "split_ratio_out_of_range",
        (df["first_half"] / df["finish"]).between(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO),
    )

    df = df.reset_index(drop=True)
    df["age"] = df["age"].astype(int)
    df["second_half"] = df["finish"] - df["first_half"]
    df["split_ratio"] = df["first_half"] / df["finish"]
    df["finish_min"] = df["finish"] / 60
    df["first_half_min"] = df["first_half"] / 60

    report["rows_out"] = int(len(df))
    return df, report


def load_results(path) -> tuple[pd.DataFrame, dict]:
    """Read a results CSV from disk and clean it."""
    return clean_results(pd.read_csv(path))


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle with a fixed seed, then split. The same seed always gives the same split,
    so every model in the project is scored on the exact same test rows."""
    order = np.random.default_rng(seed).permutation(len(df))
    n_test = int(round(len(df) * test_size))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
