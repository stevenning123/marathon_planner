"""Read a Strava bulk export (activities.csv) and summarize recent running.

This is DEMO INPUT ONLY. Strava's terms do not allow training machine learning
models on its data, so nothing in this file is ever used to fit a model. It only
reads your own runs so the bridge and the planner have something to work from.

Strava's export has several columns with the same name, such as two "Distance"
columns, so the loader checks units instead of trusting the column name.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_RUN_KM, MAX_RUN_KM = 1.0, 100.0
MIN_PACE_S_PER_KM, MAX_PACE_S_PER_KM = 150.0, 720.0  # 2:30 to 12:00 per km
KM_PER_MILE = 1.609344


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def _first_present(df: pd.DataFrame, names: list[str]):
    return next((n for n in names if n in df.columns), None)


def _distance_km(df: pd.DataFrame, unit: str) -> pd.Series:
    """Pick the distance column and convert it to kilometers.

    Strava exports a second distance column in meters ("Distance.1"), which is
    the safest one. If only one distance column exists, `unit` ("km" or "miles")
    says how to read it.
    """
    meters_col = "distance_1" if "distance_1" in df.columns else None
    if meters_col is not None:
        values = pd.to_numeric(df[meters_col], errors="coerce")
        if values.dropna().median() > 200:  # looks like meters
            return values / 1000.0
    base = pd.to_numeric(df["distance"], errors="coerce")
    return base * (KM_PER_MILE if unit == "miles" else 1.0)


def load_activities(source, unit: str = "km") -> tuple[pd.DataFrame, dict]:
    """Return (runs, report). `source` is a path or an uploaded file object.

    Columns of `runs`: date, distance_km, moving_s, pace_s_per_km.
    The report counts every row dropped and why, like the Boston cleaner does.
    """
    raw = pd.read_csv(source)
    df = raw.copy()
    df.columns = [_snake(c) for c in df.columns]
    # pandas renames the second "Distance" column to "Distance.1", which becomes distance_1.

    date_col = _first_present(df, ["activity_date", "date", "start_date", "start_date_local"])
    type_col = _first_present(df, ["activity_type", "type", "sport_type"])
    time_col = _first_present(df, ["moving_time", "elapsed_time", "elapsed_time_1"])
    if date_col is None or "distance" not in df.columns or time_col is None:
        raise ValueError(
            "This does not look like a Strava activities.csv. Needed a date, a distance, and a "
            f"time column. Columns found: {list(raw.columns)}"
        )

    report: dict = {"rows_in": int(len(df)), "dropped": {}}

    def keep(reason: str, mask: pd.Series) -> None:
        nonlocal df
        report["dropped"][reason] = int((~mask).sum())
        df = df[mask]

    if type_col is not None:
        is_run = df[type_col].astype("string").str.contains("run", case=False, na=False)
        is_virtual = df[type_col].astype("string").str.contains("virtual", case=False, na=False)
        keep("not_a_run", is_run & ~is_virtual)

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    out["distance_km"] = _distance_km(df, unit)
    out["moving_s"] = pd.to_numeric(df[time_col], errors="coerce")
    df = out
    keep("missing_values", df.notna().all(axis=1))
    keep("distance_out_of_range", df["distance_km"].between(MIN_RUN_KM, MAX_RUN_KM))
    pace = df["moving_s"] / df["distance_km"]
    keep("pace_out_of_range", pace.between(MIN_PACE_S_PER_KM, MAX_PACE_S_PER_KM))

    df = df.sort_values("date").reset_index(drop=True)
    df["pace_s_per_km"] = df["moving_s"] / df["distance_km"]
    report["rows_out"] = int(len(df))
    return df, report


def week_start(day) -> pd.Timestamp:
    """Monday of the week that contains `day`."""
    day = pd.Timestamp(day).normalize()
    return day - pd.Timedelta(days=int(day.dayofweek))


def weekly_summary(runs: pd.DataFrame, as_of=None, weeks: int = 16) -> pd.DataFrame:
    """One row per week (Monday start) for the last `weeks` weeks, zeros for empty weeks."""
    as_of = pd.Timestamp(as_of) if as_of is not None else runs["date"].max()
    last = week_start(as_of)
    starts = [last - pd.Timedelta(weeks=i) for i in range(weeks - 1, -1, -1)]
    runs = runs.assign(week=runs["date"].dt.normalize().map(week_start))
    grouped = runs.groupby("week").agg(
        km=("distance_km", "sum"), runs=("distance_km", "size"), longest_km=("distance_km", "max")
    )
    table = grouped.reindex(starts).fillna(0.0)
    table.index.name = "week_start"
    return table.reset_index()


def recent_stats(runs: pd.DataFrame, as_of=None, weeks: int = 4, long_weeks: int = 8) -> dict:
    """Average weekly kilometers over the last `weeks` full weeks and the longest recent run."""
    as_of = pd.Timestamp(as_of) if as_of is not None else runs["date"].max()
    summary = weekly_summary(runs, as_of, weeks=weeks + 1)
    # Use the weeks before the current partial week so a half finished week does not drag the average down.
    full_weeks = summary.iloc[:-1]
    since = as_of - pd.Timedelta(weeks=long_weeks)
    recent = runs[(runs["date"] > since) & (runs["date"] <= as_of + pd.Timedelta(days=1))]
    return {
        "avg_weekly_km": float(full_weeks["km"].mean()),
        "longest_run_km": float(recent["distance_km"].max()) if len(recent) else 0.0,
        "runs_per_week": float(full_weeks["runs"].mean()),
        "as_of": str(as_of.date()),
    }


def days_since_last_run(runs: pd.DataFrame, as_of=None) -> int:
    """How many days before `as_of` (default today) the newest run in the file happened."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    return int((as_of.normalize() - runs["date"].max().normalize()).days)
