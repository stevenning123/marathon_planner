"""Rough Week 3 version of the app.

Run it with:  streamlit run app.py

Upload a Strava activities.csv (or type in one recent effort), pick a race date, and see
a marathon estimate plus a weekly plan. Type what you really ran into the "Actual km"
column and the later weeks rebuild themselves.

The Strava file is demo input only. It stays in this session and is never used to train a model.
"""
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from marathon.bridge import DEFAULT_REL_HALF_WIDTH, bridge_estimate, estimate_from_effort, fmt_time
from marathon.planner import PlanSettings, build_plan, default_start_date, missed_weeks, readiness, replan
from marathon.strava import days_since_last_run, load_activities, recent_stats, weekly_summary

APP_NAME = "Marathon Planner"
BOSTON_FILE = Path(__file__).parent / "data" / "raw" / "Athletes.csv"

st.set_page_config(page_title=APP_NAME, layout="wide")


@st.cache_data(show_spinner=False)
def boston_half_width() -> tuple[float, bool]:
    """Relative width of the 80 percent range measured on Boston. Falls back to the saved number."""
    if not BOSTON_FILE.exists():
        return DEFAULT_REL_HALF_WIDTH, False
    from marathon.data import load_results, split_train_test
    from marathon.model import fit_and_predict, fit_interval, make_boosted
    import numpy as np

    data, _ = load_results(BOSTON_FILE)
    train, test = split_train_test(data, 0.2, seed=42)
    pred = fit_and_predict(lambda: make_boosted(0.10, 3), train, test)
    low, high = fit_interval(train, test)
    return float(np.mean((high - low) / 2 / pred)), True


def parse_clock(text: str):
    try:
        parts = [float(p) for p in text.strip().split(":")]
    except ValueError:
        return None
    if not 1 <= len(parts) <= 3:
        return None
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# ---------- Sidebar ----------
st.sidebar.header("Your inputs")
mode = st.sidebar.radio("Where should the estimate come from?", ["Upload Strava export", "Type in a recent effort"])
rel_half, measured = boston_half_width()

runs = None
estimate = None
stats = None
unit = "km"

TODAY = pd.Timestamp.today().normalize()
age_days = 0

if mode == "Upload Strava export":
    unit = st.sidebar.selectbox("Distance unit in your file", ["km", "miles"],
                                help="Only matters if the file has a single Distance column.")
    upload = st.sidebar.file_uploader("activities.csv from your Strava export", type="csv")
    if upload is not None:
        try:
            runs, report = load_activities(upload, unit=unit)
        except ValueError as err:
            st.sidebar.error(str(err))
            runs = None
        if runs is not None and len(runs):
            age_days = days_since_last_run(runs, TODAY)
            file_stats = recent_stats(runs, as_of=TODAY)
            estimate = bridge_estimate(runs, as_of=TODAY, rel_half_width=rel_half)
            st.sidebar.caption("Read from your file. Change them if they look wrong, for example if you ran since the export.")
            key = f"{upload.name}_{upload.size}"
            weekly_in = st.sidebar.number_input("Current kilometers per week", 0.0, 200.0,
                                                round(file_stats["avg_weekly_km"], 1), step=1.0, key=f"wk_{key}")
            long_in = st.sidebar.number_input("Longest recent run (km)", 0.0, 60.0,
                                              round(file_stats["longest_run_km"], 1), step=0.5, key=f"lr_{key}")
            stats = {"avg_weekly_km": max(weekly_in, 1.0), "longest_run_km": max(long_in, 1.0)}
else:
    dist = st.sidebar.number_input("Distance of the effort (km)", 5.0, 42.2, 21.1, step=0.1)
    clock = st.sidebar.text_input("Time (H:MM:SS)", "1:55:00")
    secs = parse_clock(clock)
    if secs:
        estimate = estimate_from_effort(dist, secs, rel_half_width=rel_half)
    else:
        st.sidebar.error("Time should look like 1:55:00")
    weekly_in = st.sidebar.number_input("Current kilometers per week", 10.0, 160.0, 40.0, step=1.0)
    long_in = st.sidebar.number_input("Longest run in the last 8 weeks (km)", 3.0, 60.0, 16.0, step=0.5)
    stats = {"avg_weekly_km": weekly_in, "longest_run_km": long_in}

race_day = st.sidebar.date_input("Race date", date.today() + timedelta(weeks=18))
runs_per_week = st.sidebar.slider("Runs per week", 3, 6, 4)

# ---------- Main page ----------
st.title(APP_NAME)
st.caption("Rough Week 3 version. Predicts a marathon time from a recent effort and builds a weekly plan that adjusts.")

if stats is None or estimate is None:
    st.info("Upload your Strava activities.csv, or switch to typing in a recent effort, to get started.")
    st.stop()

left, mid = st.columns(2)
left.metric("Marathon estimate (best case)", fmt_time(estimate["point_s"]))
mid.metric("Likely range (80 percent)", f"{fmt_time(estimate['low_s'])} to {fmt_time(estimate['high_s'])}")
if "effort_date" in estimate:
    basis = f"{estimate['effort_km']:.1f} km in {fmt_time(estimate['effort_s'])} on {estimate['effort_date']}"
else:
    basis = f"{estimate['effort_km']:.1f} km in {fmt_time(estimate['effort_s'])}"
st.caption(f"Based on your run of {basis}.")
if age_days > 14:
    st.warning(f"Your file's newest run was {age_days} days ago, so weeks since then count as zero. "
               "If you have run since, export from Strava again.")
for note in estimate.get("warnings", []):
    st.warning(note)

with st.expander("How much should I trust this?", expanded=False):
    width_note = "measured on Boston" if measured else "a saved number from Boston, since the Boston file was not found"
    st.write(
        f"The estimate scales your recent effort up to 42.2 km with Riegel's formula. That is an assumption, "
        f"not something measured on real finishes. A training run is usually easier than a race, so the estimate "
        f"may be too slow, and a short hard run can make it too fast. The range is plus or minus "
        f"{rel_half * 100:.1f} percent ({width_note}) and was measured when the input was a halfway time "
        f"inside a marathon, so with this input the real range is probably wider."
    )

if runs is not None:
    st.subheader("Your last 12 weeks")
    weekly = weekly_summary(runs, as_of=TODAY, weeks=12)
    weekly["Week of"] = weekly["week_start"].dt.strftime("%b %d")
    chart = alt.Chart(weekly).mark_bar(color="#2F5DA8").encode(
        x=alt.X("Week of:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("km:Q", title="Kilometers"), tooltip=["Week of", alt.Tooltip("km:Q", format=".1f")])
    st.altair_chart(chart, width="stretch")
    st.caption(f"Longest run in the last 8 weeks reads as {file_stats['longest_run_km']:.1f} km. "
               "If that looks wrong, change the distance unit in the sidebar.")

# ---------- Plan ----------
st.subheader("Weekly plan")
if race_day <= date.today():
    st.error("Pick a race date in the future.")
    st.stop()

settings = PlanSettings(race_date=race_day, start_date=default_start_date(date.today()), current_weekly_km=stats["avg_weekly_km"],
                        longest_run_km=stats["longest_run_km"], runs_per_week=runs_per_week)
plan = build_plan(settings)
original = pd.DataFrame(plan)

st.write("Type what you really ran in the **Actual km** column. Later weeks rebuild from it.")
table = original[["week", "week_start", "phase", "planned_km", "long_run_km"]].copy()
table["week_start"] = table["week_start"].astype(str)
table["actual_km"] = pd.Series([None] * len(table), dtype="float64")
edited = st.data_editor(
    table, hide_index=True, width="stretch", key="plan_editor",
    disabled=["week", "week_start", "phase", "planned_km", "long_run_km"],
    column_config={"week": "Week", "week_start": "Week of", "phase": "Phase", "planned_km": "Planned km",
                   "long_run_km": "Long run km", "actual_km": st.column_config.NumberColumn("Actual km", min_value=0.0, max_value=250.0)},
)

actual = {int(r.week): float(r.actual_km) for r in edited.itertuples() if pd.notna(r.actual_km)}
adjusted = replan(settings, plan, actual) if actual else plan
ready_now = readiness(adjusted)

if actual:
    st.subheader("Adjusted plan")
    adj = pd.DataFrame(adjusted)
    compare = pd.DataFrame({"Original plan": original["planned_km"].values, "Adjusted plan": adj["planned_km"].values},
                           index=original["week"])
    st.line_chart(compare, height=260)
    st.dataframe(adj[["week", "week_start", "phase", "planned_km", "long_run_km", "actual_km"]]
                 .rename(columns={"week": "Week", "week_start": "Week of", "phase": "Phase", "planned_km": "Planned km",
                                  "long_run_km": "Long run km", "actual_km": "Actual km"}),
                 hide_index=True, width="stretch")
    missed = missed_weeks(adjusted)
    if missed:
        st.warning(f"Weeks under 80 percent of plan: {missed}. Later weeks now climb from what you actually ran.")

icon = {"ready": "Ready", "borderline": "Borderline", "under prepared": "Under prepared"}[ready_now["status"]]
st.metric("Readiness for race day", icon,
          help="Ready means the plan reaches a long run of 29 km or more. Borderline is 24 to 29 km. These are rules of thumb, not predictions.")
st.caption(f"Plan peaks at {ready_now['peak_week_km']} km in a week and a {ready_now['peak_long_run_km']} km long run. "
           "The rules behind the plan are listed at the top of marathon/planner.py. They are common coaching rules of thumb, "
           "not learned from data, and not medical advice.")
