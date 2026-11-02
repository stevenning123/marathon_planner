"""Adaptive weekly training plan.

The plan follows common coaching rules of thumb. These are NOT learned from data and
they are not medical advice. They are listed here so each one can be explained:

1. Weekly distance rises at most 10 percent per week.
2. Every 4th build week is a cutback week at 80 percent, so the body can absorb the work.
3. The long run is at most 45 percent of the week (the rest is spread over easy runs, which suits a
   4 run week) and never over 32 km (20 miles).
4. The last 3 weeks are a taper at 80, 60, then 40 percent of peak distance.
5. The plan aims for a peak week of about 65 km. It only climbs as far as rule 1 allows in the weeks
   available, so a short runway ends below the target and is rated honestly.
6. A runner needs about 29 km or more for the longest run to be "ready"; 24 to 29 km is borderline.

Adapting: when a week is logged, every later week is rebuilt from what was really done,
using the same rules. A missed week restarts the climb from the lower number, so the plan
never asks for a jump the runner has not earned.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

MAX_WEEKLY_INCREASE = 0.10
CUTBACK_EVERY = 4
CUTBACK_FACTOR = 0.80
LONG_RUN_SHARE = 0.45
LONG_RUN_CAP_KM = 32.0
LONG_RUN_STEP_KM = 2.0
TAPER_FACTORS = (0.80, 0.60, 0.40)
TARGET_PEAK_KM = 65.0   # about enough for a 29 km long run at 45 percent of the week
PEAK_CAP_KM = 100.0
READY_LONG_KM, BORDERLINE_LONG_KM = 29.0, 24.0


@dataclass
class PlanSettings:
    race_date: date
    start_date: date              # any day in the first week of the plan
    current_weekly_km: float
    longest_run_km: float
    runs_per_week: int = 4
    peak_km: float | None = None  # default aims for TARGET_PEAK_KM

    def resolved_peak(self) -> float:
        if self.peak_km is not None:
            return self.peak_km
        return min(PEAK_CAP_KM, max(TARGET_PEAK_KM, self.current_weekly_km * 1.2))


def monday_of(day: date) -> date:
    return day - timedelta(days=day.weekday())


def default_start_date(today: date) -> date:
    """The Monday a new plan should start: this week early on, next week from Saturday on."""
    return monday_of(today) + timedelta(weeks=1) if today.weekday() >= 5 else monday_of(today)


def weeks_until_race(start_date: date, race_date: date) -> int:
    """Number of Monday starting weeks from the start week through the race week."""
    return (monday_of(race_date) - monday_of(start_date)).days // 7 + 1


def _row(index: int, start: date, phase: str, km: float, long_km: float, runs: int) -> dict:
    easy_runs = max(runs - 1, 1)
    return {
        "week": index,
        "week_start": start,
        "phase": phase,
        "planned_km": round(km, 1),
        "long_run_km": round(long_km, 1),
        "easy_runs": easy_runs,
        "easy_run_km": round(max(km - long_km, 0) / easy_runs, 1),
    }


def build_plan(settings: PlanSettings, first_week_index: int = 1, start_km: float | None = None,
               start_long_km: float | None = None) -> list[dict]:
    """Build every week from the start week to race week. Returns a list of week dictionaries."""
    first_monday = monday_of(settings.start_date)
    total = weeks_until_race(settings.start_date, settings.race_date)
    if total < 1:
        raise ValueError("The race date is before the start date.")
    peak = settings.resolved_peak()
    prev_km = settings.current_weekly_km if start_km is None else start_km
    prev_long = settings.longest_run_km if start_long_km is None else start_long_km
    taper_weeks = min(len(TAPER_FACTORS), max(total - 1, 0))
    build_weeks = total - taper_weeks

    rows: list[dict] = []
    for i in range(total):
        start = first_monday + timedelta(weeks=i)
        index = first_week_index + i
        if total == 1:
            km, long_km, phase = prev_km * TAPER_FACTORS[-1], min(LONG_RUN_SHARE * prev_km * TAPER_FACTORS[-1], prev_long), "Race week"
        elif i >= build_weeks:
            factor = TAPER_FACTORS[len(TAPER_FACTORS) - taper_weeks + (i - build_weeks)]
            km = prev_km * factor  # taper from the highest week actually reached, never from an unreached peak
            phase = "Race week" if i == total - 1 else "Taper"
            long_km = min(LONG_RUN_SHARE * km, prev_long)
        else:
            km = min(prev_km * (1 + MAX_WEEKLY_INCREASE), peak)
            cutback = (i + 1) % CUTBACK_EVERY == 0 and i < build_weeks - 1
            long_km = min(prev_long + LONG_RUN_STEP_KM, LONG_RUN_SHARE * km, LONG_RUN_CAP_KM)
            if cutback:
                km *= CUTBACK_FACTOR
                long_km *= 0.75
                phase = "Cutback"
            else:
                phase = "Build"
            prev_km = km if not cutback else prev_km  # the climb continues from the pre cutback week
            prev_long = long_km if not cutback else prev_long
        rows.append(_row(index, start, phase, km, long_km, settings.runs_per_week))
    return rows


def replan(settings: PlanSettings, original: list[dict], actual_km: dict[int, float]) -> list[dict]:
    """Rebuild the plan after some weeks have been logged.

    `actual_km` maps week number to kilometers really run. Weeks up to the last logged one
    keep their planned values (with the actual number added); later weeks are rebuilt from
    what was actually done. A big overshoot is capped at 10 percent above plan.
    """
    if not actual_km:
        return [dict(r, actual_km=None) for r in original]
    last = max(actual_km)
    done = []
    for row in original:
        if row["week"] <= last:
            done.append(dict(row, actual_km=actual_km.get(row["week"])))
    last_row = next(r for r in original if r["week"] == last)
    base_km = actual_km[last]
    base_km = min(base_km, last_row["planned_km"] * (1 + MAX_WEEKLY_INCREASE))
    # If the logged week was far below plan, long run progress is assumed to scale down with it.
    ratio = min(1.0, base_km / last_row["planned_km"]) if last_row["planned_km"] else 1.0
    base_long = last_row["long_run_km"] * ratio
    remaining_start = last_row["week_start"] + timedelta(weeks=1)
    if monday_of(remaining_start) > monday_of(settings.race_date):
        return done
    rest_settings = PlanSettings(
        race_date=settings.race_date, start_date=remaining_start, current_weekly_km=base_km,
        longest_run_km=base_long, runs_per_week=settings.runs_per_week, peak_km=settings.resolved_peak(),
    )
    future = build_plan(rest_settings, first_week_index=last + 1, start_km=base_km, start_long_km=base_long)
    return done + [dict(r, actual_km=None) for r in future]


def readiness(plan: list[dict]) -> dict:
    """Summarize how well prepared the plan leaves the runner. Uses the rules at the top of this file."""
    longest = max(r["long_run_km"] for r in plan)
    peak = max(r["planned_km"] for r in plan)
    if longest >= READY_LONG_KM:
        status = "ready"
    elif longest >= BORDERLINE_LONG_KM:
        status = "borderline"
    else:
        status = "under prepared"
    return {"status": status, "peak_long_run_km": round(longest, 1), "peak_week_km": round(peak, 1)}


def missed_weeks(plan: list[dict], threshold: float = 0.80) -> list[int]:
    """Weeks where the logged distance was under `threshold` of the plan."""
    return [r["week"] for r in plan
            if r.get("actual_km") is not None and r["actual_km"] < threshold * r["planned_km"]]


def total_planned_km(plan: list[dict]) -> float:
    return float(sum(r["planned_km"] for r in plan))
