"""Mile20: predict your marathon finish, then train so mile 20 feels like mile 10.

Run it with:  streamlit run app.py

Type in one recent effort (or upload a Strava activities.csv), pick a race date, and see
a marathon estimate plus a weekly plan. Type what you really ran into the "Actual km"
column and the later weeks rebuild themselves.

The Strava file is demo input only. It stays in this session and is never used to train a model.

This file is presentation only. The math lives in marathon/bridge.py and marathon/planner.py
and is not changed here.
"""
from datetime import date, timedelta
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from marathon.baseline import MARATHON_KM
from marathon.bridge import DEFAULT_REL_HALF_WIDTH, LOW_ENDURANCE_KM, bridge_estimate, estimate_from_effort, fmt_time
from marathon.planner import (
    BORDERLINE_LONG_KM, CUTBACK_EVERY, CUTBACK_FACTOR, LONG_RUN_CAP_KM, LONG_RUN_SHARE, LONG_RUN_STEP_KM,
    MAX_WEEKLY_INCREASE, READY_LONG_KM, TAPER_FACTORS, TARGET_PEAK_KM, PlanSettings, build_plan,
    default_start_date, missed_weeks, readiness, replan,
)
from marathon.strava import (
    MAX_PACE_S_PER_KM, MIN_PACE_S_PER_KM, days_since_last_run, load_activities, recent_stats, weekly_summary,
)

APP_NAME = "Mile20"
TAGLINE = "Predict your marathon finish, then train so mile 20 feels like mile 10."
BOSTON_FILE = Path(__file__).parent / "data" / "raw" / "Athletes.csv"
ICON_FILE = Path(__file__).parent / "assets" / "mile20_icon.png"
REPO_URL = "https://github.com/stevenning123/marathon_planner"
SUBLINE = "A personal project. Estimates are best case and plans follow rules of thumb."
FONT = "Inter"
CARD_RADIUS, INNER_RADIUS = "14px", "10px"

# Palette. Navy for text, one blue for data, one orange accent, light blue grey for cards.
NAVY, BLUE, ORANGE, CARD = "#1B2A41", "#2F5DA8", "#E4572E", "#F2F5FA"
# Phase colors, used only on the plan chart and its legend.
PHASES = ["Build", "Cutback", "Taper", "Race week"]
ORDINAL = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
PHASE_COLORS = {"Build": BLUE, "Cutback": "#9DB4DD", "Taper": "#7C8AA0", "Race week": NAVY}

st.set_page_config(page_title=f"{APP_NAME} | Marathon finish predictor", page_icon=str(ICON_FILE) if ICON_FILE.exists() else "🏃", layout="wide")

CSS = f"""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
  display: none !important;
}}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ max-width: 1080px; padding-top: 2.2rem; padding-bottom: 4rem; }}
html, body, [class*="css"] {{ color: {NAVY}; }}
/* Tabular numbers everywhere so times and distances line up */
body, .stApp, .stApp * {{ font-variant-numeric: tabular-nums; }}

/* Header */
.m20-header {{ margin: 0 0 1.4rem 0; padding: 1.7rem 1.9rem 1.5rem 1.9rem; border-radius: {CARD_RADIUS};
  background: linear-gradient(120deg, #DCE6F6 0%, #EAF0FA 45%, #F7F9FD 100%); border: 1px solid #E1E7F0; }}
.m20-sub {{ font-size: 0.82rem; color: {NAVY}; opacity: 0.6; margin-top: 0.7rem; padding-top: 0.6rem;
  border-top: 1px solid rgba(27, 42, 65, 0.1); }}
.m20-logo {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; color: {NAVY}; }}
.m20-logo span {{ color: {ORANGE}; }}
.m20-tagline {{ font-size: 1.05rem; color: {NAVY}; opacity: 0.72; margin-top: 0.45rem; }}

/* Tabs */
.stTabs [role="tablist"] {{ gap: 1.6rem; border-bottom: 1px solid #DDE3EE; }}
.stTabs [role="tab"] {{ padding: 0.6rem 0.1rem; transition: color 0.15s ease; }}
.stTabs [role="tab"]:hover, .stTabs [role="tab"]:hover p {{ color: {BLUE} !important; }}
.stTabs [role="tab"][aria-selected="true"]:hover,
.stTabs [role="tab"][aria-selected="true"]:hover p {{ color: {ORANGE} !important; }}
.stTabs [role="tab"] p {{ font-size: 1rem; font-weight: 600; }}
.stTabs [role="tabpanel"] {{ padding-top: 1.4rem; }}

/* Cards: bordered containers get the light card color */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
div[class*="st-key-card"] {{
  background: {CARD};
  border-color: #E1E7F0 !important;
  border-radius: {CARD_RADIUS} !important;
}}
[data-testid="stExpander"] details {{ border-radius: {CARD_RADIUS} !important; border-color: #E1E7F0 !important; }}
/* Consistent space between stacked cards */
[data-testid="stVerticalBlock"] {{ row-gap: 1rem; }}
div[class*="st-key-card"] {{ padding: 1.25rem 1.35rem 1.7rem 1.35rem !important; }}

/* Type scale */
.m20-eyebrow {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: {NAVY}; opacity: 0.6; margin-bottom: 0.35rem; }}
.m20-h2 {{ font-size: 1.25rem; font-weight: 700; color: {NAVY}; margin: 0 0 0.4rem 0; }}
.m20-body {{ font-size: 0.98rem; line-height: 1.6; color: {NAVY}; }}
.m20-muted {{ font-size: 0.9rem; line-height: 1.5; color: {NAVY}; opacity: 0.68; }}

/* Big estimate */
.m20-big {{ font-size: clamp(3.6rem, 16vw, 7.2rem); font-weight: 800; letter-spacing: -0.04em; line-height: 0.95;
  color: {NAVY}; font-variant-numeric: tabular-nums; white-space: nowrap; margin: 0.3rem 0 0.6rem 0; }}
.m20-pill {{ display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 999px;
  background: #FFFFFF; border: 1px solid #D5DDEA; color: {NAVY}; vertical-align: middle; }}
.m20-pill.accent {{ background: {ORANGE}; border-color: {ORANGE}; color: #FFFFFF; }}
.m20-pill.blue {{ background: {BLUE}; border-color: {BLUE}; color: #FFFFFF; }}

/* Range band */
.m20-band {{ margin: 1.4rem 0 0.4rem 0; }}
.m20-track {{ position: relative; height: 14px; border-radius: 999px; background: #DDE4EF; }}
.m20-fill {{ position: absolute; top: 0; bottom: 0; border-radius: 999px; background: {BLUE}; opacity: 0.85; }}
.m20-marker {{ position: absolute; top: -6px; width: 4px; height: 26px; margin-left: -2px; border-radius: 2px;
  background: {ORANGE}; box-shadow: 0 0 0 3px {CARD}; }}
.m20-ticks {{ position: relative; height: 3rem; margin-top: 0.45rem; }}
.m20-tick {{ position: absolute; transform: translateX(-50%); text-align: center; white-space: nowrap; }}
.m20-tick b {{ display: block; font-size: 0.98rem; font-variant-numeric: tabular-nums; color: {NAVY}; }}
.m20-tick small {{ font-size: 0.74rem; color: {NAVY}; opacity: 0.6; }}

/* Calm notes */
.m20-note {{ display: flex; gap: 0.7rem; background: #FFFFFF; border: 1px solid #E1E7F0; border-left: 3px solid #9DB4DD;
  border-radius: {INNER_RADIUS}; padding: 0.75rem 0.95rem; margin: 0.6rem 0; font-size: 0.92rem; line-height: 1.5; color: {NAVY}; }}
.m20-note .i {{ flex: 0 0 auto; width: 1.25rem; height: 1.25rem; border-radius: 50%; background: {CARD};
  border: 1px solid #C9D4E5; font-size: 0.75rem; font-weight: 700; text-align: center; line-height: 1.15rem; }}

/* Stat row */
.m20-stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin-top: 1rem; }}
.m20-stat {{ background: #FFFFFF; border: 1px solid #E1E7F0; border-radius: {INNER_RADIUS}; padding: 0.7rem 0.85rem; }}
.m20-stat .v {{ font-size: 1.3rem; font-weight: 700; white-space: nowrap; font-variant-numeric: tabular-nums; color: {NAVY}; }}
.m20-stat .l {{ font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: {NAVY}; opacity: 0.62; }}

/* Readiness meter */
.m20-status {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: {NAVY}; margin: 0.1rem 0 0.2rem 0; }}
.m20-meter {{ position: relative; height: 10px; border-radius: 999px; background: #DDE4EF; margin: 1rem 0 0.3rem 0; }}
.m20-meter .f {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: {BLUE}; }}
.m20-meter .t {{ position: absolute; top: -5px; width: 2px; height: 20px; background: {NAVY}; opacity: 0.55; }}
.m20-meter-labels {{ position: relative; height: 1.2rem; font-size: 0.74rem; color: {NAVY}; opacity: 0.65; }}
.m20-meter-labels span {{ position: absolute; transform: translateX(-50%); white-space: nowrap; }}

/* Legend */
.m20-legend {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.1rem; font-size: 0.85rem; color: {NAVY}; margin: 0.2rem 0 0.4rem 0; }}
.m20-legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; }}
.m20-legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; }}
.m20-legend i.line {{ height: 3px; width: 18px; border-radius: 2px; }}
.m20-legend i.dash {{ height: 0; width: 18px; border-top: 2px dashed {NAVY}; opacity: 0.55; border-radius: 0; }}

/* How it works */
.m20-rule {{ display: flex; gap: 0.9rem; align-items: flex-start; margin: 0.9rem 0; }}
.m20-rule .n {{ flex: 0 0 auto; width: 2rem; height: 2rem; border-radius: 50%; background: {BLUE}; color: #FFFFFF;
  font-weight: 700; text-align: center; line-height: 2rem; }}
.m20-rule b {{ display: block; font-size: 1rem; color: {NAVY}; }}

/* Footer */
.m20-footer {{ margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid #E1E7F0; display: flex;
  flex-wrap: wrap; justify-content: space-between; gap: 0.4rem 1rem; font-size: 0.85rem; color: {NAVY}; opacity: 0.75; }}
.m20-footer a {{ color: {BLUE}; text-decoration: none; font-weight: 600; }}
.m20-footer a:hover {{ color: {ORANGE}; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem; }}
  .m20-logo {{ font-size: 2.1rem; }}
  .m20-header {{ padding: 1.2rem 1.1rem 1.1rem 1.1rem; }}
  .stTabs [role="tab"] p {{ font-size: 0.95rem; }}
  .m20-tagline {{ font-size: 0.95rem; }}
  .stTabs [role="tablist"] {{ gap: 1rem; }}
  .m20-stats {{ gap: 0.5rem; }}
  .m20-stat {{ padding: 0.55rem 0.6rem; }}
  .m20-stat .v {{ font-size: 1.1rem; }}
  .st-key-card_ready .m20-stats {{ margin-bottom: 1.2rem; }}
  div[class*="st-key-card"] {{ padding: 1rem 1rem 1.4rem 1rem !important; }}
  /* Show the estimate above the inputs, and keep stacked columns close together */
  .st-key-est_row > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{ flex-direction: column-reverse; }}
  .st-key-est_row > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }}
  [data-testid="stHorizontalBlock"] {{ row-gap: 1rem !important; gap: 1rem !important; }}
}}
</style>
"""


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
    if len(parts) == 1:  # a lone number means minutes, so "25" is 25 minutes
        return parts[0] * 60
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


# ---------- Small view helpers (presentation only) ----------

def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def nice_date(value) -> str:
    """Dates shown as 'Oct 25, 2026' so no hyphens reach the page."""
    ts = pd.Timestamp(value)
    return f"{ts.strftime('%b')} {ts.day}, {ts.year}"


def short_date(value) -> str:
    ts = pd.Timestamp(value)
    return f"{ts.strftime('%b')} {ts.day}"


def fmt_pace(seconds_per_km: float) -> str:
    total = int(round(seconds_per_km))
    return f"{total // 60}:{total % 60:02d}"


def note(text: str) -> None:
    html(f'<div class="m20-note"><div class="i">i</div><div>{escape(text)}</div></div>')


def range_band(estimate: dict) -> str:
    low, point, high = estimate["low_s"], estimate["point_s"], estimate["high_s"]
    pad = (high - low) * 0.45
    lo_edge, hi_edge = low - pad, high + pad
    span = hi_edge - lo_edge

    def pct(x: float) -> float:
        return (x - lo_edge) / span * 100

    left, right, mid = pct(low), pct(high), pct(point)
    return (
        '<div class="m20-band">'
        '<div class="m20-track">'
        f'<div class="m20-fill" style="left:{left:.2f}%;width:{right - left:.2f}%"></div>'
        f'<div class="m20-marker" style="left:{mid:.2f}%"></div>'
        '</div>'
        '<div class="m20-ticks">'
        f'<div class="m20-tick" style="left:{left:.2f}%"><b>{fmt_time(low)}</b><small>faster end</small></div>'
        f'<div class="m20-tick" style="left:{right:.2f}%"><b>{fmt_time(high)}</b><small>slower end</small></div>'
        '</div></div>'
    )


def stat_row(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div class="m20-stat"><div class="v">{escape(v)}</div><div class="l">{escape(l)}</div></div>'
                    for v, l in items)
    return f'<div class="m20-stats">{cells}</div>'


def legend(with_actual: bool, with_original: bool) -> str:
    parts = [f'<span><i style="background:{PHASE_COLORS[p]}"></i>{p}</span>' for p in PHASES]
    if with_actual:
        parts.append(f'<span><i class="line" style="background:{ORANGE}"></i>Your actual km</span>')
    if with_original:
        parts.append('<span><i class="dash"></i>Original plan</span>')
    return f'<div class="m20-legend">{"".join(parts)}</div>'


def plan_chart(adjusted: pd.DataFrame, original: pd.DataFrame, show_original: bool) -> alt.LayerChart:
    data = adjusted.copy()
    data["Week of"] = data["week_start"].map(short_date)
    race = data.loc[data["phase"] == "Race week", "week"]
    race_week = int(race.iloc[0]) if len(race) else -1
    is_race = f"datum.value == {race_week}"
    axis_x = alt.X("week:O", title="Week", axis=alt.Axis(
        labelAngle=0, labelColor=NAVY, titleColor=NAVY, domainColor="#C9D4E5", tickColor="#C9D4E5", labelFontSize=11,
        labelExpr=f"{is_race} ? 'Race' : datum.label",
        labelFontWeight=alt.ExprRef(f"{is_race} ? 'bold' : 'normal'")))
    bars = alt.Chart(data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=axis_x,
        y=alt.Y("planned_km:Q", title="Kilometers",
                axis=alt.Axis(labelColor=NAVY, titleColor=NAVY, gridColor="#D9E1EC", gridDash=[3, 3],
                              gridOpacity=0.9, domain=False, ticks=False, tickCount=5)),
        color=alt.Color("phase:N", scale=alt.Scale(domain=PHASES, range=[PHASE_COLORS[p] for p in PHASES]), legend=None),
        tooltip=[alt.Tooltip("week:O", title="Week"), alt.Tooltip("Week of:N"), alt.Tooltip("phase:N", title="Phase"),
                 alt.Tooltip("planned_km:Q", title="Planned km", format=".1f"),
                 alt.Tooltip("long_run_km:Q", title="Long run km", format=".1f")],
    )
    layers = [bars]
    if show_original:
        orig = original[["week", "planned_km"]].rename(columns={"planned_km": "original_km"})
        layers.append(alt.Chart(orig).mark_line(color=NAVY, opacity=0.5, strokeDash=[5, 4], strokeWidth=1.6).encode(
            x=axis_x, y="original_km:Q",
            tooltip=[alt.Tooltip("week:O", title="Week"), alt.Tooltip("original_km:Q", title="Original plan km", format=".1f")]))
    if "actual_km" in data and data["actual_km"].notna().any():
        logged = data[data["actual_km"].notna()]
        actual = alt.Chart(logged).encode(
            x=axis_x, y="actual_km:Q",
            tooltip=[alt.Tooltip("week:O", title="Week"), alt.Tooltip("actual_km:Q", title="Actual km", format=".1f")])
        layers += [actual.mark_line(color=ORANGE, strokeWidth=2.5),
                   actual.mark_point(color=ORANGE, filled=True, size=70, opacity=1)]
    return (alt.layer(*layers).properties(height=300)
            .configure_view(strokeWidth=0).configure(background="transparent", font=FONT))


def readiness_copy(ready: dict, weeks: int, missed: list[int]) -> tuple[str, str]:
    longest = ready["peak_long_run_km"]
    status = ready["status"]
    why = (f"The longest run in this plan reaches {longest:g} km. The planner calls a plan ready when the longest run "
           f"reaches {READY_LONG_KM:g} km or more, borderline from {BORDERLINE_LONG_KM:g} to {READY_LONG_KM:g} km, "
           f"and under prepared below that.")
    if status == "ready":
        change = (f"Staying close to the plan keeps it here. Logging weeks well under plan rebuilds the later weeks "
                  f"lower and could pull the longest run under {READY_LONG_KM:g} km.")
    else:
        change = (f"With {weeks} weeks to race day, the long run can only climb about {LONG_RUN_STEP_KM:g} km a week, "
                  f"and the week can only grow {MAX_WEEKLY_INCREASE * 100:.0f} percent at a time. A later race date, "
                  f"more kilometers per week to start from, or a longer current long run would raise it.")
    if missed:
        weeks_txt = ", ".join(str(w) for w in missed)
        label = "Week" if len(missed) == 1 else "Weeks"
        change += f" {label} {weeks_txt} came in under 80 percent of plan, so later weeks now climb from what you ran."
    return why, change


# ---------- Page ----------

html(CSS)
html(f'<div class="m20-header"><div class="m20-logo">Mile<span>20</span></div>'
     f'<div class="m20-tagline">{escape(TAGLINE)}</div>'
     f'<div class="m20-sub">{escape(SUBLINE)}</div></div>')

rel_half, measured = boston_half_width()
TODAY = pd.Timestamp.today().normalize()

tab_est, tab_plan, tab_how = st.tabs(["Estimate", "Plan", "How it works"])

runs = None
estimate = None
stats = None
file_stats = None
age_days = 0

# ---------- Estimate tab ----------
with tab_est:
    with st.container(key="est_row"):
        col_in, col_out = st.columns([1, 1.65], gap="large")

        with col_in:
            with st.container(border=True, key="card_inputs"):
                html('<div class="m20-eyebrow">Your running</div>')
                mode = st.radio("Where should the estimate come from?",
                                ["Type in a recent effort", "Upload Strava export"], horizontal=False)
                if mode == "Upload Strava export":
                    unit = st.selectbox("Distance unit in your file", ["km", "miles"],
                                        help="Only matters if the file has a single Distance column.")
                    upload = st.file_uploader("activities.csv from your Strava export", type="csv",
                                              help="Demo input only. It stays in this session and never trains anything.")
                    if upload is not None:
                        try:
                            runs, report = load_activities(upload, unit=unit)
                        except ValueError as err:
                            st.error(str(err))
                            runs = None
                        if runs is not None and len(runs):
                            age_days = days_since_last_run(runs, TODAY)
                            file_stats = recent_stats(runs, as_of=TODAY)
                            estimate = bridge_estimate(runs, as_of=TODAY, rel_half_width=rel_half)
                            html('<div class="m20-muted">Read from your file. Change them if they look wrong, '
                                 'for example if you ran since the export.</div><div style="height:0.6rem"></div>')
                            key = f"{upload.name}_{upload.size}"
                            weekly_in = st.number_input("Current kilometers per week", 0.0, 200.0,
                                                        round(file_stats["avg_weekly_km"], 1), step=1.0, key=f"wk_{key}")
                            long_in = st.number_input("Longest recent run (km)", 0.0, 60.0,
                                                      round(file_stats["longest_run_km"], 1), step=0.5, key=f"lr_{key}")
                            stats = {"avg_weekly_km": max(weekly_in, 1.0), "longest_run_km": max(long_in, 1.0)}
                else:
                    c1, c2 = st.columns(2)
                    dist = c1.number_input("Distance (km)", 5.0, 42.2, 21.1, step=0.1)
                    clock = c2.text_input("Time (H:MM:SS)", "1:55:00")
                    secs = parse_clock(clock)
                    if not secs:
                        st.error("Time should look like 1:55:00, or just minutes like 25")
                    elif not MIN_PACE_S_PER_KM <= secs / dist <= MAX_PACE_S_PER_KM:
                        st.error("That pace is not realistic. It should be between 2:30 and 12:00 per km, "
                                 "so check the distance and the time.")
                    else:
                        estimate = estimate_from_effort(dist, secs, rel_half_width=rel_half)
                    weekly_in = st.number_input("Current kilometers per week", 10.0, 160.0, 40.0, step=1.0)
                    long_in = st.number_input("Longest run in the last 8 weeks (km)", 3.0, 60.0, 16.0, step=0.5)
                    stats = {"avg_weekly_km": weekly_in, "longest_run_km": long_in}

        with col_out:
            if estimate is None:
                with st.container(border=True, key="card_empty"):
                    html('<div class="m20-eyebrow">Marathon estimate</div>')
                    html('<div class="m20-h2">Add a run to see your estimate</div>')
                    if mode == "Upload Strava export" and runs is not None and len(runs):
                        msg = "No run of 5 km or more in the last 8 weeks of this file, so there is nothing to scale up yet."
                    else:
                        msg = ("Upload your Strava activities.csv, or switch to typing in a recent effort. "
                               "One good run is enough.")
                    html(f'<div class="m20-body">{escape(msg)}</div>')
            else:
                with st.container(border=True, key="card_estimate"):
                    html('<div class="m20-eyebrow">Marathon estimate</div>')
                    html(f'<div class="m20-big">{fmt_time(estimate["point_s"])}</div>')
                    pace = fmt_pace(estimate["point_s"] / MARATHON_KM)
                    html(f'<span class="m20-pill accent">Best case</span>&nbsp; '
                         f'<span class="m20-muted">about {pace} per km</span>')
                    html(range_band(estimate))
                    html(f'<div class="m20-muted">The blue band is the likely range (80 percent): '
                         f'{fmt_time(estimate["low_s"])} to {fmt_time(estimate["high_s"])}. '
                         f'The orange mark is the estimate.</div>')
                    if "effort_date" in estimate:
                        basis = (f"{estimate['effort_km']:.1f} km in {fmt_time(estimate['effort_s'])} "
                                 f"on {nice_date(estimate['effort_date'])}")
                    else:
                        basis = f"{estimate['effort_km']:.1f} km in {fmt_time(estimate['effort_s'])}"
                    html(stat_row([
                        (f"{estimate['effort_km']:.1f} km", "your run"),
                        (fmt_time(estimate["effort_s"]), "its time"),
                        (f"±{rel_half * 100:.1f}%", "range"),
                    ]))
                    html(f'<div class="m20-muted" style="margin-top:0.8rem">Based on your run of {escape(basis)}.</div>')

                if age_days > 14:
                    note(f"Your file's newest run was {age_days} days ago, so weeks since then count as zero. "
                         "If you have run since, export from Strava again.")
                for text in estimate.get("warnings", []):
                    note(text)

                with st.expander("How much should I trust this?", expanded=False):
                    width_note = ("measured on Boston" if measured
                                  else "a saved number from Boston, since the Boston file was not found")
                    st.write(
                        f"The estimate scales your recent effort up to 42.2 km with Riegel's formula. That is an assumption, "
                        f"not something measured on real finishes. A training run is usually easier than a race, so the "
                        f"estimate may be too slow, and a short hard run can make it too fast. The range is plus or minus "
                        f"{rel_half * 100:.1f} percent ({width_note}) and was measured when the input was a halfway time "
                        f"inside a marathon, so with this input the real range is probably wider."
                    )

    if runs is not None and len(runs):
        with st.container(border=True, key="card_history"):
            html('<div class="m20-eyebrow">From your file</div><div class="m20-h2">Your last 12 weeks</div>')
            weekly = weekly_summary(runs, as_of=TODAY, weeks=12)
            weekly["Week of"] = weekly["week_start"].map(short_date)
            hist = alt.Chart(weekly).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Week of:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelColor=NAVY,
                                                                          labelOverlap=True, domainColor="#C9D4E5")),
                y=alt.Y("km:Q", title="Kilometers", axis=alt.Axis(labelColor=NAVY, titleColor=NAVY,
                                                                   gridColor="#E6EBF3", domain=False, tickCount=5)),
                tooltip=["Week of", alt.Tooltip("km:Q", format=".1f")],
            ).properties(height=220).configure_view(strokeWidth=0).configure(background="transparent", font=FONT)
            st.altair_chart(hist, width="stretch")
            html(f'<div class="m20-muted">Longest run in the last 8 weeks reads as {file_stats["longest_run_km"]:.1f} km. '
                 'If that looks wrong, change the distance unit above.</div>')

# ---------- Plan tab ----------
with tab_plan:
    with st.container(border=True, key="card_plan_inputs"):
        c1, c2 = st.columns(2)
        race_day = c1.date_input("Race date", date.today() + timedelta(weeks=18), format="MM/DD/YYYY")
        runs_per_week = c2.slider("Runs per week", 3, 6, 4)
        if stats is not None:
            html(f'<div class="m20-muted">Starting from {stats["avg_weekly_km"]:.1f} km per week and a '
                 f'{stats["longest_run_km"]:.1f} km longest run, from the Estimate tab.</div>')

    if stats is None or estimate is None:
        with st.container(border=True, key="card_plan_empty"):
            html('<div class="m20-h2">Your plan appears here</div>'
                 '<div class="m20-body">Add a recent run on the Estimate tab first. The plan starts from how much '
                 'you run now.</div>')
    elif race_day <= date.today():
        with st.container(border=True, key="card_plan_past"):
            html('<div class="m20-h2">Pick a race date in the future</div>'
                 '<div class="m20-body">The plan counts the weeks from now until race day.</div>')
    else:
        settings = PlanSettings(race_date=race_day, start_date=default_start_date(date.today()),
                                current_weekly_km=stats["avg_weekly_km"], longest_run_km=stats["longest_run_km"],
                                runs_per_week=runs_per_week)
        plan = build_plan(settings)
        original = pd.DataFrame(plan)

        # Reserve space above the table, then fill it once the table edits are known.
        top = st.container()
        chart_slot = st.container()

        with st.container(border=True, key="card_table"):
            html('<div class="m20-eyebrow">Log your weeks</div>'
                 '<div class="m20-h2">Weekly plan</div>'
                 '<div class="m20-muted" style="margin-bottom:0.6rem">Type what you really ran in the '
                 '<b>Actual km</b> column. Later weeks rebuild from it.</div>')
            table = original[["week", "week_start", "phase", "planned_km", "long_run_km"]].copy()
            table["week_start"] = table["week_start"].map(short_date)
            table["actual_km"] = pd.Series([None] * len(table), dtype="float64")
            edited = st.data_editor(
                table, hide_index=True, width="stretch", key="plan_editor",
                column_order=["week", "week_start", "planned_km", "actual_km", "phase", "long_run_km"],
                disabled=["week", "week_start", "phase", "planned_km", "long_run_km"],
                column_config={
                    "week": st.column_config.NumberColumn("Week", width="small"),
                    "week_start": "Week of", "phase": "Phase",
                    "planned_km": st.column_config.NumberColumn("Planned km", format="%.1f"),
                    "long_run_km": st.column_config.NumberColumn("Long run km", format="%.1f"),
                    "actual_km": st.column_config.NumberColumn("Actual km", min_value=0.0, max_value=250.0, format="%.1f"),
                },
            )

        actual = {int(r.week): float(r.actual_km) for r in edited.itertuples() if pd.notna(r.actual_km)}
        adjusted = replan(settings, plan, actual) if actual else plan
        ready_now = readiness(adjusted)
        adj = pd.DataFrame(adjusted)
        missed = missed_weeks(adjusted)

        if actual:
            with st.container(border=True, key="card_adjusted"):
                html('<div class="m20-eyebrow">Rebuilt from your log</div><div class="m20-h2">Adjusted plan</div>')
                shown = adj[["week", "week_start", "planned_km", "actual_km", "phase", "long_run_km"]].copy()
                shown["week_start"] = shown["week_start"].map(short_date)
                st.dataframe(
                    shown.rename(columns={"week": "Week", "week_start": "Week of", "phase": "Phase",
                                          "planned_km": "Planned km", "long_run_km": "Long run km",
                                          "actual_km": "Actual km"}),
                    hide_index=True, width="stretch",
                    column_config={"Planned km": st.column_config.NumberColumn(format="%.1f"),
                                   "Long run km": st.column_config.NumberColumn(format="%.1f"),
                                   "Actual km": st.column_config.NumberColumn(format="%.1f")})

        with top:
            label = {"ready": "Ready", "borderline": "Borderline", "under prepared": "Under prepared"}[ready_now["status"]]
            pill = {"ready": "blue", "borderline": "", "under prepared": "accent"}[ready_now["status"]]
            why, change = readiness_copy(ready_now, len(adj), missed)
            scale_max = max(LONG_RUN_CAP_KM, ready_now["peak_long_run_km"])
            fill = min(ready_now["peak_long_run_km"] / scale_max * 100, 100)
            b_at, r_at = BORDERLINE_LONG_KM / scale_max * 100, READY_LONG_KM / scale_max * 100
            with st.container(border=True, key="card_ready"):
                left, right = st.columns([1.15, 1], gap="large")
                with left:
                    html('<div class="m20-eyebrow">Readiness for race day</div>'
                         f'<div class="m20-status">{label}</div>'
                         f'<span class="m20-pill {pill}">Longest run {ready_now["peak_long_run_km"]:g} km</span>'
                         f'<div class="m20-meter"><div class="f" style="width:{fill:.1f}%"></div>'
                         f'<div class="t" style="left:{b_at:.1f}%"></div><div class="t" style="left:{r_at:.1f}%"></div></div>'
                         f'<div class="m20-meter-labels"><span style="left:{b_at:.1f}%">{BORDERLINE_LONG_KM:g} km</span>'
                         f'<span style="left:{r_at:.1f}%">{READY_LONG_KM:g} km</span></div>')
                    html(stat_row([
                        (f"{len(adj)}", "weeks"),
                        (f"{ready_now['peak_week_km']:g} km", "peak week"),
                        (f"{ready_now['peak_long_run_km']:g} km", "longest run"),
                    ]))
                with right:
                    html(f'<div class="m20-h2">Why it landed here</div><div class="m20-body">{escape(why)}</div>'
                         f'<div class="m20-h2" style="margin-top:0.9rem">What would change it</div>'
                         f'<div class="m20-body">{escape(change)}</div>'
                         '<div class="m20-muted" style="margin-top:0.8rem">These are rules of thumb, '
                         'not predictions, and not medical advice.</div>')

        with chart_slot:
            with st.container(border=True, key="card_chart"):
                html('<div class="m20-eyebrow">Week by week</div><div class="m20-h2">Kilometers per week</div>')
                html(legend(with_actual=bool(actual), with_original=bool(actual)))
                st.altair_chart(plan_chart(adj, original, show_original=bool(actual)), width="stretch")

# ---------- How it works tab ----------
with tab_how:
    taper_txt = ", ".join(f"{f * 100:.0f}" for f in TAPER_FACTORS)
    with st.container(border=True, key="card_how_estimate"):
        html('<div class="m20-eyebrow">The estimate</div><div class="m20-h2">How the finish time is made</div>'
             '<div class="m20-body">Mile20 takes your fastest recent run of 10 km or more from the last 8 weeks '
             '(or the run you type in) and scales it up to 42.2 km with Riegel\'s formula, the same formula runners '
             'have used for decades. If your file has no run that long, it falls back to runs of 5 km or more and '
             'says so.</div>'
             '<div class="m20-body" style="margin-top:0.6rem"><b>The machine learning model is not what makes this '
             'estimate.</b> The model from Weeks 1 and 2 predicts a finish from a halfway time inside a race. A runner '
             'planning a race does not have one yet, so this app uses Riegel scaling instead.</div>'
             f'<div class="m20-body" style="margin-top:0.6rem">The range around the estimate is plus or minus '
             f'{rel_half * 100:.1f} percent. That width comes from the model\'s 80 percent range on Boston Marathon '
             'results, where it held the real finish 80 percent of the time. It was measured with a halfway time as '
             'the input, so with a single training run the real range is probably wider.</div>')

    with st.container(border=True, key="card_how_rules"):
        html('<div class="m20-eyebrow">The plan</div><div class="m20-h2">Three rules the planner follows</div>'
             '<div class="m20-rule"><div class="n">1</div><div class="m20-body">'
             f'<b>Grow at most {MAX_WEEKLY_INCREASE * 100:.0f} percent a week</b>'
             'Weekly distance never jumps more than that, so the plan only asks for what you have earned. '
             f'It aims for a peak week of about {TARGET_PEAK_KM:g} km.</div></div>'
             '<div class="m20-rule"><div class="n">2</div><div class="m20-body">'
             f'<b>Cut back every {ORDINAL.get(CUTBACK_EVERY, f"{CUTBACK_EVERY}th")} week</b>'
             f'Every {ORDINAL.get(CUTBACK_EVERY, f"{CUTBACK_EVERY}th")} build week drops to {CUTBACK_FACTOR * 100:.0f} percent so your body can absorb the work.'
             '</div></div>'
             '<div class="m20-rule"><div class="n">3</div><div class="m20-body">'
             f'<b>Taper for {len(TAPER_FACTORS)} weeks</b>'
             f'The last weeks drop to {taper_txt} percent of your highest week, so you start fresh.</div></div>'
             f'<div class="m20-muted">The long run stays at most {LONG_RUN_SHARE * 100:.0f} percent of the week and never '
             f'over {LONG_RUN_CAP_KM:g} km. When you log a week, every later week is rebuilt from what you really ran, '
             'so one missed week never forces a big jump.</div>')

    with st.container(border=True, key="card_how_limits"):
        html('<div class="m20-eyebrow">Honest limits</div><div class="m20-h2">What this cannot tell you</div>'
             '<div class="m20-rule"><div class="n">1</div><div class="m20-body"><b>The estimate is an assumption</b>'
             'Scaling one run up to a marathon has not been measured on real finishes. It sees how fast you run, '
             'not how much long distance training is behind you, which is why it is labeled best case.</div></div>'
             '<div class="m20-rule"><div class="n">2</div><div class="m20-body"><b>The rules are rules of thumb</b>'
             'They are common coaching guidelines, not learned from data, and not medical advice. Readiness reports '
             'whether the plan reaches a long enough long run. It does not predict minutes gained or lost.</div></div>'
             '<div class="m20-rule"><div class="n">3</div><div class="m20-body"><b>Nothing is trained on Strava data</b>'
             'Your Strava file is demo input only. It stays in this session and is never used to train a model.'
             '</div></div>')

# ---------- Footer ----------
html(f'<div class="m20-footer"><span>Built by Steven Ning</span>'
     f'<a href="{REPO_URL}" target="_blank" rel="noopener">Source on GitHub</a></div>')
