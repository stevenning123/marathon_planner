"""The bridge: turn a recent run from your Strava export into a marathon estimate.

The Week 2 model needs a halfway time from inside a marathon. A runner planning a
race does not have one, so this module scales a recent hard effort up to the
marathon distance with Riegel's formula, the same formula used as the Week 1
baseline.

HONEST LIMITS (these go in the notes and the video):
1. This is an ASSUMPTION, not something measured on Boston data. The Boston file
   has halfway splits, not training runs, so nothing here has been tested against
   real finishes except the one check in `backtest_bridge`.
2. Riegel assumes an all out race effort. A training run is usually easier than
   that, so the estimate can be too slow, and a short hard run can make it too fast.
   On Boston, Riegel ran about 8 minutes fast, so lean toward the slower end.
3. The range reuses the width of the Week 2 range (about plus or minus 6 percent).
   That range was measured with a halfway time as the input. A weaker input is
   probably less accurate, so treat the range as a minimum.
"""
from __future__ import annotations

import pandas as pd

from marathon.baseline import MARATHON_KM, riegel

DEFAULT_REL_HALF_WIDTH = 0.063  # about 28.2 min / 2 / 224.6 min, from the Week 2 80 percent range
LOW_ENDURANCE_KM = 24.0         # same line the planner calls "borderline" for a longest run


def fmt_time(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def marathon_equivalent(moving_s: float, distance_km: float, exponent: float = 1.06) -> float:
    """Riegel: marathon time (seconds) implied by one effort."""
    return float(riegel(moving_s, distance_km, MARATHON_KM, exponent))


def best_recent_effort(runs: pd.DataFrame, as_of=None, lookback_weeks: int = 8,
                       min_km: float = 10.0, exponent: float = 1.06):
    """The run in the last `lookback_weeks` weeks that implies the fastest marathon.

    Runs under `min_km` are skipped, because scaling a short run up to 42 km is where
    Riegel is least reliable. Returns None when no run qualifies.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else runs["date"].max()
    since = as_of - pd.Timedelta(weeks=lookback_weeks)
    pool = runs[(runs["date"] > since) & (runs["date"] <= as_of + pd.Timedelta(days=1))
                & (runs["distance_km"] >= min_km)].copy()
    if pool.empty:
        return None
    pool["marathon_s"] = [marathon_equivalent(t, d, exponent) for t, d in zip(pool["moving_s"], pool["distance_km"])]
    return pool.loc[pool["marathon_s"].idxmin()]


def estimate_warnings(effort_km: float, longest_recent_km: float | None) -> list[str]:
    """Plain language reasons the estimate may be too fast."""
    notes = []
    if effort_km < 10:
        notes.append(
            f"The estimate comes from a {effort_km:.1f} km run. Scaling a short run up to 42 km is where "
            "Riegel is least reliable, so treat the time as a best case."
        )
    if longest_recent_km is not None and longest_recent_km < LOW_ENDURANCE_KM:
        notes.append(
            f"Your longest run in the last 8 weeks was {longest_recent_km:.1f} km. Riegel assumes marathon level "
            "endurance training, which this does not show yet, so the real time today is likely slower."
        )
    return notes


def bridge_estimate(runs: pd.DataFrame, as_of=None, lookback_weeks: int = 8, min_km: float = 10.0,
                    fallback_min_km: float = 5.0, exponent: float = 1.06,
                    rel_half_width: float = DEFAULT_REL_HALF_WIDTH):
    """Point estimate and range for the marathon, or None if no run qualifies.

    Tries runs of `min_km` or more first. If there are none, falls back to runs of
    `fallback_min_km` or more and says so in `warnings`.
    """
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else runs["date"].max()
    effort = best_recent_effort(runs, as_of_ts, lookback_weeks, min_km, exponent)
    if effort is None and fallback_min_km < min_km:
        effort = best_recent_effort(runs, as_of_ts, lookback_weeks, fallback_min_km, exponent)
    if effort is None:
        return None
    since = as_of_ts - pd.Timedelta(weeks=lookback_weeks)
    recent = runs[(runs["date"] > since) & (runs["date"] <= as_of_ts + pd.Timedelta(days=1))]
    longest = float(recent["distance_km"].max())
    point = float(effort["marathon_s"])
    return {
        "point_s": point,
        "low_s": point * (1 - rel_half_width),
        "high_s": point * (1 + rel_half_width),
        "effort_date": str(effort["date"].date()),
        "effort_km": float(effort["distance_km"]),
        "effort_s": float(effort["moving_s"]),
        "effort_pace_s_per_km": float(effort["pace_s_per_km"]),
        "longest_recent_km": longest,
        "exponent": exponent,
        "rel_half_width": rel_half_width,
        "warnings": estimate_warnings(float(effort["distance_km"]), longest),
    }


def estimate_from_effort(distance_km: float, time_s: float, exponent: float = 1.06,
                         rel_half_width: float = DEFAULT_REL_HALF_WIDTH) -> dict:
    """Same estimate from a distance and time typed in by hand."""
    point = marathon_equivalent(time_s, distance_km, exponent)
    return {"point_s": point, "low_s": point * (1 - rel_half_width), "high_s": point * (1 + rel_half_width),
            "effort_km": distance_km, "effort_s": time_s, "exponent": exponent, "rel_half_width": rel_half_width,
            "warnings": estimate_warnings(distance_km, None)}


def backtest_bridge(runs: pd.DataFrame, race_date, race_time_s: float, gap_days: int = 21, **kwargs) -> dict:
    """One sanity check: estimate a past marathon from the runs before it, then compare.

    Uses only runs from at least `gap_days` before the race. One race proves very little,
    so report it as an example, never as an accuracy number.
    """
    cutoff = pd.Timestamp(race_date) - pd.Timedelta(days=gap_days)
    est = bridge_estimate(runs, as_of=cutoff, **kwargs)
    if est is None:
        return {"ok": False, "reason": "no qualifying run in the weeks before that race"}
    miss_min = (est["point_s"] - race_time_s) / 60
    return {"ok": True, "predicted_s": est["point_s"], "actual_s": race_time_s, "miss_min": miss_min,
            "inside_range": est["low_s"] <= race_time_s <= est["high_s"], "effort_date": est["effort_date"],
            "effort_km": est["effort_km"], "longest_recent_km": est["longest_recent_km"], "warnings": est["warnings"]}
