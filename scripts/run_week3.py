"""Week 3: revisit the dropped Boston runners, read a Strava export, estimate a marathon,
and build an adaptive plan.

Usage:
    python scripts/run_week3.py data/raw/Athletes.csv
    python scripts/run_week3.py data/raw/Athletes.csv --strava data/strava/activities.csv --race-date 2027-03-14
    python scripts/run_week3.py data/raw/Athletes.csv --strava data/strava/activities.csv --race-date 2027-03-14 --missed 3:20
    python scripts/run_week3.py data/raw/Athletes.csv --weekly-km 35 --long-km 16 --race-date 2027-03-14 --missed 3:15

The Strava file is demo input only. It is never used to train anything.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from marathon.bridge import DEFAULT_REL_HALF_WIDTH, backtest_bridge, bridge_estimate, fmt_time  # noqa: E402
from marathon.data import load_results, split_train_test  # noqa: E402
from marathon.model import build_features, fade_target, fit_and_predict, fit_interval, make_boosted, to_finish_seconds  # noqa: E402
from marathon.outliers import describe, dropped_by_split_rule, effect_on_scores  # noqa: E402
from marathon.planner import PlanSettings, build_plan, default_start_date, missed_weeks, readiness, replan  # noqa: E402
from marathon.strava import days_since_last_run, load_activities, recent_stats  # noqa: E402

ACCENT, HIGHLIGHT, INK, MUTED, GRID = "#2F5DA8", "#E07A1F", "#1F2937", "#6B7280", "#E5E7EB"


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def parse_clock(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def part_a_dropped(raw_path: Path, out: dict) -> float:
    """Revisit the 50 dropped runners. Returns the relative half width of the 80 percent range."""
    raw = pd.read_csv(raw_path)
    dropped = dropped_by_split_rule(raw)
    data, _ = load_results(raw_path)
    train, test = split_train_test(data, test_size=0.2, seed=42)

    facts = describe(dropped)
    print("\n=== The runners the cleaner dropped ===")
    print(f"Runners dropped by the split rule: {facts['n']}")
    print(f"  Below the 35 percent limit: {facts['below_low_limit']}   Above the 65 percent limit: {facts['above_high_limit']}")
    print(f"  First half under 1 hour (impossible, would point to a timing error): {facts['impossible_first_half']}")
    print(f"  Median first half: {fmt_time(facts['median_first_half_min'] * 60)}   Median finish: {fmt_time(facts['median_finish_min'] * 60)}")
    print(f"  Median finish of the normal test runners: {fmt_time(test['finish'].median())}")

    maker = lambda: make_boosted(0.10, 3)  # same settings the Week 2 cross validation chose
    test_pred = fit_and_predict(maker, train, test)
    model = maker().fit(build_features(train), fade_target(train))
    drop_pred = to_finish_seconds(dropped, model.predict(build_features(dropped)))
    effect = effect_on_scores(dropped, test, test_pred, drop_pred)
    print(f"Average miss on the normal test runners: {effect['test_mae_min']:.2f} min")
    print(f"Average miss on the dropped runners:     {effect['dropped_mae_min']:.2f} min")
    print(f"Average miss if they were added to the test set: {effect['combined_mae_min']:.2f} min "
          f"({effect['change_min']:+.2f} min)")

    low, high = fit_interval(train, test)
    rel_half = float(np.mean((high - low) / 2 / test_pred))
    print(f"80 percent range as a share of the prediction: plus or minus {rel_half * 100:.1f} percent")

    out["dropped_runners"] = {"facts": facts, "effect_on_scores": effect, "rel_half_width": rel_half}

    fig, ax = plt.subplots(figsize=(7, 4.6))
    sample = test.sample(min(len(test), 2500), random_state=1)
    ax.scatter(sample["first_half"] / 60, sample["finish"] / 60, s=7, color=ACCENT, alpha=0.22, linewidths=0, label="Kept (test sample)")
    ax.scatter(dropped["first_half"] / 60, dropped["finish"] / 60, s=26, color=HIGHLIGHT, label="Dropped by split rule")
    ax.set_xlabel("First half (minutes)", color=MUTED)
    ax.set_ylabel("Finish (minutes)", color=MUTED)
    ax.set_title("The 50 dropped runners started normally and finished far slower", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "outputs" / "week3_dropped_runners.png", dpi=160, facecolor="white")
    plt.close(fig)
    return rel_half


def part_b_strava(args, rel_half: float, out: dict) -> dict | None:
    """Read the Strava export (demo only) and print the bridge estimate. Returns recent stats."""
    runs, report = load_activities(args.strava, unit=args.unit)
    print("\n=== Strava export (demo input only, never used to train) ===")
    print(f"Rows in: {report['rows_in']:,}   Runs kept: {report['rows_out']:,}")
    for reason, count in report["dropped"].items():
        print(f"  dropped {count:>6,}  {reason}")
    today = pd.Timestamp.today().normalize()
    age = days_since_last_run(runs, today)
    print(f"Newest run in the file: {runs['date'].max().date()} ({age} days ago)")
    if age > 14:
        print(f"  WARNING: your file is {age} days old. Weeks after the newest run count as zero. "
              "If you have run since then, export from Strava again.")
    stats = recent_stats(runs, as_of=today)
    print(f"Last 4 full weeks: {stats['avg_weekly_km']:.1f} km per week, {stats['runs_per_week']:.1f} runs per week")
    print(f"Longest run in the last 8 weeks: {stats['longest_run_km']:.1f} km (check this looks right; "
          f"if it is wrong, rerun with --unit miles)")
    est = bridge_estimate(runs, as_of=today, rel_half_width=rel_half)
    out["strava"] = {"cleaning": report, "recent": stats, "bridge": est, "days_since_last_run": age}
    if est is None:
        print("No run of 5 km or more in the last 8 weeks, so there is no bridge estimate.")
    else:
        print(f"\nBridge estimate (Riegel from your run on {est['effort_date']}: {est['effort_km']:.1f} km in {fmt_time(est['effort_s'])})")
        print(f"  Marathon estimate: {fmt_time(est['point_s'])}   Range: {fmt_time(est['low_s'])} to {fmt_time(est['high_s'])}")
        for note in est["warnings"]:
            print(f"  WARNING: {note}")
        print("  This is an assumption, not a measured result. See marathon/bridge.py for the limits.")
    if args.check_race:
        race_day, race_time = args.check_race
        result = backtest_bridge(runs, race_day, parse_clock(race_time), rel_half_width=rel_half)
        out["strava"]["backtest"] = result
        if result["ok"]:
            side = "faster" if result["miss_min"] < 0 else "slower"
            print(f"\nOne race check ({race_day}): the bridge said {fmt_time(result['predicted_s'])}, "
                  f"you ran {fmt_time(result['actual_s'])}. The estimate was {abs(result['miss_min']):.1f} min {side} than you ran. "
                  f"Inside the range: {result['inside_range']}. It used a {result['effort_km']:.1f} km run on {result['effort_date']}; "
                  f"your longest run in the 8 weeks before was {result['longest_recent_km']:.1f} km. "
                  f"One race is an example, not an accuracy number.")
        else:
            print(f"\nRace check skipped: {result['reason']}")
    return stats


def part_c_plan(args, stats: dict | None, out: dict) -> None:
    weekly = args.weekly_km if args.weekly_km else (stats or {}).get("avg_weekly_km")
    long_km = args.long_km if args.long_km else (stats or {}).get("longest_run_km")
    if not args.race_date or not weekly or not long_km:
        print("\nSkipping the plan. Give --race-date plus either --strava or both --weekly-km and --long-km.")
        return
    race = date.fromisoformat(args.race_date)
    start = date.fromisoformat(args.start_date) if args.start_date else default_start_date(date.today())
    settings = PlanSettings(race_date=race, start_date=start, current_weekly_km=weekly, longest_run_km=long_km)
    plan = build_plan(settings)
    print(f"\n=== Plan: {len(plan)} weeks to the race on {race} (starting from {weekly:.0f} km per week) ===")
    print(f"{'Wk':>3} {'Week of':<11}{'Phase':<10}{'Planned km':>11}{'Long run':>10}")
    for r in plan:
        print(f"{r['week']:>3} {str(r['week_start']):<11}{r['phase']:<10}{r['planned_km']:>11.1f}{r['long_run_km']:>10.1f}")
    ready = readiness(plan)
    print(f"Peak week {ready['peak_week_km']} km, longest run {ready['peak_long_run_km']} km, readiness: {ready['status']}")
    out["plan"] = {"settings": {"race_date": str(race), "start_date": str(start), "weekly_km": weekly, "long_km": long_km},
                   "weeks": [{**r, "week_start": str(r["week_start"])} for r in plan], "readiness": ready}

    adjusted = None
    if args.missed:
        actual = {int(w): float(k) for w, k in (item.split(":") for item in args.missed.split(","))}
        adjusted = replan(settings, plan, actual)
        print(f"\nAfter logging {actual}:")
        print(f"{'Wk':>3} {'Phase':<10}{'Original':>10}{'Adjusted':>10}{'Actual':>8}")
        for old, new in zip(plan, adjusted):
            act = "" if new.get("actual_km") is None else f"{new['actual_km']:.0f}"
            print(f"{old['week']:>3} {new['phase']:<10}{old['planned_km']:>10.1f}{new['planned_km']:>10.1f}{act:>8}")
        after = readiness(adjusted)
        print(f"Missed weeks: {missed_weeks(adjusted)}. Readiness now: {after['status']} "
              f"(longest run {after['peak_long_run_km']} km, peak week {after['peak_week_km']} km)")
        out["replan"] = {"actual": actual, "readiness": after, "missed": missed_weeks(adjusted),
                         "weeks": [{**r, "week_start": str(r["week_start"])} for r in adjusted]}

    fig, ax = plt.subplots(figsize=(8, 4.6))
    weeks = [r["week"] for r in plan]
    ax.plot(weeks, [r["planned_km"] for r in plan], color=MUTED, linestyle="--", marker="o", ms=4, label="Original plan")
    if adjusted:
        ax.plot([r["week"] for r in adjusted], [r["planned_km"] for r in adjusted], color=ACCENT, marker="o", ms=5, label="Adjusted plan")
        pts = [(r["week"], r["actual_km"]) for r in adjusted if r.get("actual_km") is not None]
        ax.scatter(*zip(*pts), color=HIGHLIGHT, s=70, zorder=5, label="What I actually ran")
        ax.set_title("The plan rebuilds itself after a missed week", loc="left", color=INK, fontsize=12)
    else:
        ax.set_title("Weekly plan to race day", loc="left", color=INK, fontsize=12)
    ax.set_xlabel("Week", color=MUTED)
    ax.set_ylabel("Kilometers per week", color=MUTED)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "outputs" / "week3_plan.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("boston_csv")
    p.add_argument("--strava", help="path to your Strava activities.csv (demo input only)")
    p.add_argument("--unit", default="km", choices=["km", "miles"], help="distance unit if the file has only one distance column")
    p.add_argument("--race-date", help="marathon date, like 2027-03-14")
    p.add_argument("--start-date", help="first day of the plan, default this Monday (next Monday on weekends)")
    p.add_argument("--weekly-km", type=float, help="current weekly kilometers, if not using --strava")
    p.add_argument("--long-km", type=float, help="longest recent run in km, if not using --strava")
    p.add_argument("--missed", help="log weeks as week:km pairs, like 3:20,4:25")
    p.add_argument("--check-race", nargs=2, metavar=("DATE", "TIME"), help="a past marathon to test the bridge on, like 2026-06-07 3:58:00")
    args = p.parse_args()

    out: dict = {}
    rel_half = part_a_dropped(Path(args.boston_csv), out)
    stats = part_b_strava(args, rel_half, out) if args.strava else None
    part_c_plan(args, stats, out)

    Path(ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "week3_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved charts and week3_metrics.json in outputs/")


if __name__ == "__main__":
    main()
