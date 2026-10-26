"""Week 2: train machine learning models and score them against the Week 1 baselines.

Usage:
    python scripts/run_week2.py data/raw/Athletes.csv
    python scripts/run_week2.py synthetic

Uses the exact same cleaning and the exact same train and test split as Week 1,
so the numbers are directly comparable. "synthetic" is FAKE data, only for checking
that the code runs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from marathon.baseline import (  # noqa: E402
    calibrated_predict,
    error_summary,
    fit_fade_factor,
    riegel_from_half,
)
from marathon.data import clean_results, load_results, split_train_test  # noqa: E402
from marathon.model import (  # noqa: E402
    fit_and_predict,
    fit_interval,
    interval_coverage,
    make_boosted,
    make_ridge,
    paired_bootstrap_gain,
    pick_best,
)
from marathon.synthetic import make_synthetic_results  # noqa: E402

ACCENT = "#2F5DA8"
HIGHLIGHT = "#E07A1F"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def group_table(test: pd.DataFrame, train: pd.DataFrame, preds: dict) -> dict:
    """Average miss in minutes for slices of the test set."""
    cuts = train["first_half"].quantile([1 / 3, 2 / 3]).to_numpy()  # slices decided on train only
    pace = pd.cut(test["first_half"], [-np.inf, cuts[0], cuts[1], np.inf], labels=["faster third", "middle third", "slower third"])
    age = pd.cut(test["age"], [17, 34, 49, 90], labels=["18 to 34", "35 to 49", "50 and over"])
    groups = {"pace": pace, "age": age, "gender": test["gender"]}
    out: dict = {}
    for gname, labels in groups.items():
        out[gname] = {}
        order = list(labels.cat.categories) if hasattr(labels, "cat") else sorted(labels.dropna().unique())
        for label in order:
            mask = (labels == label).to_numpy()
            if mask.sum() < 30:
                continue
            out[gname][str(label)] = {"n": int(mask.sum())}
            for model_name, pred in preds.items():
                out[gname][str(label)][model_name] = error_summary(test["finish"][mask], pred[mask])["mae_min"]
    return out


def make_chart(rows, groups, synthetic: bool, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")

    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [HIGHLIGHT if r[2] else ACCENT for r in rows]
    bars = ax1.bar(range(len(names)), values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", color=INK)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, max(values) * 1.2)
    ax1.set_ylabel("Average miss (minutes)", color=MUTED)
    ax1.set_title("Test set error, lower is better", loc="left", color=INK, fontsize=12)
    style(ax1)

    labels, base_vals, model_vals = [], [], []
    for label, row in groups["pace"].items():
        labels.append(label)
        base_vals.append(row["calibrated"])
        model_vals.append(row["best_model"])
    x = np.arange(len(labels))
    ax2.bar(x - 0.18, base_vals, width=0.36, color=ACCENT, label="Calibrated baseline")
    ax2.bar(x + 0.18, model_vals, width=0.36, color=HIGHLIGHT, label="Boosted trees")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Where the error lives (by pace)", loc="left", color=INK, fontsize=12)
    ax2.legend(frameon=False, fontsize=9)
    style(ax2)

    headline = "Week 2: does machine learning beat the baseline?"
    if synthetic:
        headline = "SYNTHETIC DATA, NOT REAL RESULTS"
    fig.suptitle(headline, x=0.01, ha="left", color="#B91C1C" if synthetic else INK, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    synthetic = source == "synthetic"
    if synthetic:
        print("\n*** SYNTHETIC DATA: fake numbers, only checking that the code runs ***\n")
        data, report = clean_results(make_synthetic_results())
    else:
        data, report = load_results(source)

    train, test = split_train_test(data, test_size=0.2, seed=42)
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}")

    # Baselines, same as Week 1.
    factor = fit_fade_factor(train["first_half"], train["finish"])
    preds = {
        "riegel": np.asarray(riegel_from_half(test["first_half"])),
        "calibrated": np.asarray(calibrated_predict(test["first_half"], factor)),
    }

    # Choose settings using cross validation on TRAIN only. The test set is not touched here.
    candidates = {
        "ridge a=1": lambda: make_ridge(1.0),
        "boosted lr=0.05 depth=2": lambda: make_boosted(0.05, 2),
        "boosted lr=0.05 depth=3": lambda: make_boosted(0.05, 3),
        "boosted lr=0.10 depth=3": lambda: make_boosted(0.10, 3),
    }
    print("\nPicking settings with 5 fold cross validation on the train rows only...")
    best_label, cv_scores = pick_best(candidates, train)
    for label, score in cv_scores.items():
        mark = "  <- chosen" if label == best_label else ""
        print(f"  {label:<26}{score:>8.2f} min{mark}")

    preds["ridge"] = fit_and_predict(candidates["ridge a=1"], train, test)
    preds["best_model"] = fit_and_predict(candidates[best_label], train, test)

    metrics = {name: error_summary(test["finish"], pred) for name, pred in preds.items()}
    gain = paired_bootstrap_gain(test["finish"], preds["calibrated"], preds["best_model"])
    low, high = fit_interval(train, test)
    interval = interval_coverage(test["finish"], low, high)

    print(f"\n{'Method':<28}{'MAE':>8}{'RMSE':>8}{'Median':>8}{'Bias':>8}   (minutes)")
    labels = {
        "riegel": "Classic Riegel",
        "calibrated": "Calibrated baseline",
        "ridge": "Ridge regression",
        "best_model": f"Boosted trees ({best_label})" if best_label.startswith("boosted") else f"Best: {best_label}",
    }
    for key in ("riegel", "calibrated", "ridge", "best_model"):
        m = metrics[key]
        print(f"{labels[key]:<28}{m['mae_min']:>8.2f}{m['rmse_min']:>8.2f}{m['median_abs_error_min']:>8.2f}{m['bias_min']:>+8.2f}")

    print(
        f"\nBest model vs calibrated baseline: {gain['gain_min']:+.2f} min better on average "
        f"(95% range {gain['ci95_low_min']:+.2f} to {gain['ci95_high_min']:+.2f})"
    )
    print(
        f"80% prediction range: real finish landed inside {interval['coverage'] * 100:.1f}% of the time, "
        f"average width {interval['mean_width_min']:.1f} min"
    )

    groups = group_table(test, train, preds)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    tag = "synthetic" if synthetic else "real"
    result = {
        "synthetic": synthetic,
        "chosen_model": best_label,
        "cv_mae_min": cv_scores,
        "test_metrics": metrics,
        "gain_vs_calibrated": gain,
        "interval_80": interval,
        "by_group_mae_min": groups,
    }
    (out_dir / f"week2_metrics_{tag}.json").write_text(json.dumps(result, indent=2))

    rows = [
        ("Classic\nRiegel", metrics["riegel"]["mae_min"], False),
        ("Calibrated\nbaseline", metrics["calibrated"]["mae_min"], False),
        ("Ridge\nregression", metrics["ridge"]["mae_min"], False),
        ("Boosted\ntrees", metrics["best_model"]["mae_min"], True),
    ]
    chart = out_dir / f"week2_model_{tag}.png"
    make_chart(rows, groups, synthetic, chart)
    print(f"\nSaved {chart.relative_to(ROOT)} and week2_metrics_{tag}.json in outputs/")


if __name__ == "__main__":
    main()
