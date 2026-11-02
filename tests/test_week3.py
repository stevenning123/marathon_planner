from datetime import date

import numpy as np
import pandas as pd
import pytest

from marathon.bridge import backtest_bridge, bridge_estimate, estimate_from_effort, marathon_equivalent
from marathon.outliers import dropped_by_split_rule, describe
from marathon.planner import (
    CUTBACK_EVERY, LONG_RUN_CAP_KM, MAX_WEEKLY_INCREASE, PlanSettings, build_plan, missed_weeks,
    readiness, replan, weeks_until_race,
)
from marathon.strava import load_activities, recent_stats, weekly_summary
from marathon.synthetic import make_synthetic_results, make_synthetic_strava


@pytest.fixture(scope="module")
def strava_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("strava") / "activities.csv"
    make_synthetic_strava().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def runs(strava_file):
    return load_activities(strava_file)[0]


# ---- Strava loader ----

def test_loader_keeps_only_runs_and_reports_drops(strava_file):
    runs, report = load_activities(strava_file)
    assert report["dropped"]["not_a_run"] > 0
    assert report["rows_in"] == report["rows_out"] + sum(report["dropped"].values())
    assert runs["distance_km"].between(1, 100).all()


def test_loader_uses_the_meters_column_when_two_distance_columns_exist(strava_file, runs):
    # the synthetic long runs are about 10 to 25 km, so a meters mixup would give values near 10000
    assert runs["distance_km"].max() < 60


def test_loader_reads_miles_when_told(tmp_path):
    df = pd.DataFrame({"Activity Date": ["Oct 5, 2026, 6:00:00 AM"], "Activity Type": ["Run"],
                       "Distance": [10.0], "Moving Time": [4000]})
    path = tmp_path / "a.csv"
    df.to_csv(path, index=False)
    km = load_activities(path, unit="miles")[0]["distance_km"].iloc[0]
    assert km == pytest.approx(16.09344)


def test_loader_rejects_a_file_that_is_not_strava(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Strava"):
        load_activities(path)


def test_weekly_summary_has_one_row_per_week_with_zeros(runs):
    table = weekly_summary(runs, weeks=6)
    assert len(table) == 6
    assert (table["week_start"].dt.dayofweek == 0).all()


def test_recent_stats_are_plausible(runs):
    stats = recent_stats(runs, as_of=runs["date"].max())
    assert 20 < stats["avg_weekly_km"] < 80
    assert stats["longest_run_km"] > 10


# ---- Bridge ----

def test_marathon_equivalent_matches_riegel_formula():
    assert marathon_equivalent(3600, 10) == pytest.approx(3600 * (42.195 / 10) ** 1.06)


def test_bridge_ignores_runs_shorter_than_the_minimum(runs):
    assert bridge_estimate(runs, min_km=30, fallback_min_km=30) is None


def test_bridge_range_surrounds_the_point(runs):
    est = bridge_estimate(runs)
    assert est["low_s"] < est["point_s"] < est["high_s"]


def test_manual_effort_estimate():
    est = estimate_from_effort(21.0975, 2 * 3600)
    assert est["point_s"] == pytest.approx(7200 * 2 ** 1.06)


def test_backtest_only_uses_runs_before_the_cutoff(runs):
    result = backtest_bridge(runs, "2026-08-30", 4 * 3600)
    assert result["ok"]
    assert pd.Timestamp(result["effort_date"]) <= pd.Timestamp("2026-08-30") - pd.Timedelta(days=21)


def test_bridge_falls_back_to_shorter_runs_and_warns(runs):
    est = bridge_estimate(runs, min_km=60, fallback_min_km=5)
    assert est is not None and est["effort_km"] >= 5 and est["effort_km"] < 60


def test_low_endurance_warning_when_longest_run_is_short():
    from marathon.bridge import estimate_warnings
    assert estimate_warnings(8.0, 8.0) and len(estimate_warnings(8.0, 8.0)) == 2
    assert estimate_warnings(21.0, 30.0) == []


def test_stale_file_is_detected(runs):
    from marathon.strava import days_since_last_run
    assert days_since_last_run(runs, runs["date"].max() + pd.Timedelta(days=20)) == 20


def test_default_start_is_next_monday_on_weekends():
    from marathon.planner import default_start_date
    assert default_start_date(date(2026, 11, 1)) == date(2026, 11, 2)   # Sunday
    assert default_start_date(date(2026, 11, 4)) == date(2026, 11, 2)   # Wednesday


# ---- Planner ----

def settings(**kw):
    base = dict(race_date=date(2027, 3, 14), start_date=date(2026, 11, 2), current_weekly_km=40, longest_run_km=16)
    base.update(kw)
    return PlanSettings(**base)


def test_week_count_runs_from_start_week_through_race_week():
    assert weeks_until_race(date(2026, 11, 2), date(2027, 3, 14)) == 19


def test_plan_never_climbs_more_than_ten_percent_a_week():
    plan = build_plan(settings())
    build = [r for r in plan if r["phase"] == "Build"]
    for a, b in zip(build, build[1:]):
        assert b["planned_km"] <= a["planned_km"] * (1 + MAX_WEEKLY_INCREASE) + 0.11


def test_plan_has_cutbacks_taper_and_a_race_week():
    plan = build_plan(settings())
    phases = [r["phase"] for r in plan]
    assert "Cutback" in phases and phases[-1] == "Race week" and phases.count("Taper") == 2
    assert phases[CUTBACK_EVERY - 1] == "Cutback"


def test_long_run_never_exceeds_the_cap():
    assert max(r["long_run_km"] for r in build_plan(settings(current_weekly_km=90, longest_run_km=30))) <= LONG_RUN_CAP_KM


def test_race_before_start_is_an_error():
    with pytest.raises(ValueError):
        build_plan(settings(race_date=date(2026, 10, 1)))


def test_short_plan_still_builds():
    plan = build_plan(settings(race_date=date(2026, 11, 20)))
    assert plan[-1]["phase"] == "Race week"


def test_replan_after_a_missed_week_lowers_later_weeks_and_keeps_the_calendar():
    s = settings()
    plan = build_plan(s)
    new = replan(s, plan, {3: 20.0, 4: 25.0})
    assert [r["week_start"] for r in new] == [r["week_start"] for r in plan]
    assert new[4]["planned_km"] < plan[4]["planned_km"]
    assert new[4]["planned_km"] <= 25.0 * (1 + MAX_WEEKLY_INCREASE) + 0.11
    assert missed_weeks(new) == [3, 4]


def test_replan_taper_never_jumps_above_the_last_build_week():
    s = settings()
    new = replan(s, build_plan(s), {3: 15.0})
    last_build = max(r["planned_km"] for r in new if r["phase"] in ("Build", "Cutback") and r["week"] > 3)
    first_taper = next(r for r in new if r["phase"] == "Taper")
    assert first_taper["planned_km"] < last_build


def test_replan_with_nothing_logged_changes_nothing():
    s = settings()
    plan = build_plan(s)
    same = replan(s, plan, {})
    assert [r["planned_km"] for r in same] == [r["planned_km"] for r in plan]


def test_overshoot_is_capped_at_ten_percent_above_plan():
    s = settings()
    plan = build_plan(s)
    new = replan(s, plan, {1: 200.0})
    assert new[1]["planned_km"] <= plan[0]["planned_km"] * 1.1 * 1.1 + 0.2


def test_readiness_reflects_the_longest_run():
    short = build_plan(settings(race_date=date(2026, 12, 6)))
    assert readiness(short)["status"] != "ready"
    assert readiness(build_plan(settings(current_weekly_km=60, longest_run_km=26)))["status"] == "ready"


# ---- Dropped runners ----

def test_dropped_runners_are_only_the_split_rule_failures():
    raw = make_synthetic_results(n=500, seed=2)
    bad = raw.index[:3]
    raw.loc[bad, "First Half"] = raw.loc[bad, "Finish"] * 0.2
    dropped = dropped_by_split_rule(raw)
    assert len(dropped) >= 3
    assert not dropped["split_ratio"].between(0.35, 0.65).any()
    assert dropped["split_ratio"].lt(0.35).sum() >= 3
    assert describe(dropped)["n"] == len(dropped)
