"""FAKE data for checking that the code runs. Never report results from this.

It imitates the shape of the Boston results file (seconds for times, a few
dirty rows) so the loader and cleaner can be exercised before real data is
downloaded. The numbers mean nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_results(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    finish = np.clip(rng.lognormal(np.log(4.1 * 3600), 0.2, n), 2.2 * 3600, 7.5 * 3600)
    hours = finish / 3600
    # Runners usually slow down, so the first half is a bit under half the finish time.
    ratio = np.clip(0.485 - 0.004 * (hours - 3) + rng.normal(0, 0.02, n), 0.40, 0.52)

    df = pd.DataFrame(
        {
            "Bib": np.arange(1, n + 1),
            "Age": np.clip(rng.normal(43, 11, n), 18, 80).round(),
            "Gender": rng.choice(["M", "F"], size=n, p=[0.55, 0.45]),
            "First Half": finish * ratio,
            "Second Half": finish * (1 - ratio),
            "Finish": finish,
        }
    )

    # Add the kind of dirty rows real files have.
    dnf = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[dnf, ["Finish", "First Half"]] = 0
    missing = rng.choice(n, size=int(n * 0.005), replace=False)
    df.loc[missing, "Age"] = np.nan
    return df


def make_synthetic_strava(weeks: int = 20, seed: int = 11, end: str = "2026-10-26") -> pd.DataFrame:
    """FAKE Strava style export, only for testing the loader, bridge, planner and app.

    Mimics the odd parts of the real file: two columns both named Distance (kilometers, then
    meters), text dates like "Oct 5, 2026, 6:12:45 AM", and some activities that are not runs.
    """
    rng = np.random.default_rng(seed)
    end_day = pd.Timestamp(end)
    rows = []
    for w in range(weeks):
        monday = end_day - pd.Timedelta(weeks=weeks - 1 - w) - pd.Timedelta(days=int(end_day.dayofweek))
        base = 25 + w * 1.2
        for day, share in ((1, 0.2), (3, 0.2), (5, 0.2), (6, 0.4)):
            km = max(base * share * rng.normal(1, 0.08), 2.0)
            pace = rng.normal(335, 12) if day != 6 else rng.normal(350, 10)  # seconds per km
            moving = km * pace
            when = monday + pd.Timedelta(days=day, hours=7, minutes=int(rng.integers(0, 50)))
            rows.append(("Run", when, km, moving))
        if w % 3 == 0:
            rows.append(("Ride", monday + pd.Timedelta(days=2, hours=18), 20.0, 3000.0))
    df = pd.DataFrame(rows, columns=["type", "when", "km", "moving"])
    out = pd.DataFrame({
        "Activity ID": np.arange(len(df)) + 1000,
        "Activity Date": [t.strftime("%b %-d, %Y, %-I:%M:%S %p") for t in df["when"]],
        "Activity Name": ["Morning Run" if t == "Run" else "Ride" for t in df["type"]],
        "Activity Type": df["type"],
        "Elapsed Time": (df["moving"] * 1.03).round(),
        "Distance": df["km"].round(2),
        "Moving Time": df["moving"].round(),
        "Distance.1": (df["km"] * 1000).round(1),
    })
    # Write the second distance column under the same name, like the real export.
    out.columns = [c if c != "Distance.1" else "Distance" for c in out.columns]
    return out
