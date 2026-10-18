"""Week 1: clean the Boston results and score two baselines on a held out test set.

Usage:
    python scripts/run_week1.py data/raw/your_boston_file.csv
    python scripts/run_week1.py synthetic

"synthetic" uses FAKE data. It only checks that the code runs. Never show those numbers.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from marathon.baseline import (  # noqa: E402
    calibrated_predict,
    error_summary,
    fit_fade_factor,
    riegel_from_half,
)
from marathon.data import clean_results, load_results, split_train_test  # noqa: E402
from marathon.synthetic import make_synthetic_results  # noqa: E402

ACCENT = "#2F5DA8"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"


def hms(minutes: float) -> str:
    total = int(round(minutes * 60))
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def make_chart(test, riegel_pred_s, metrics, synthetic: bool, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("white")

    sample = test.sample(min(len(test), 3000), random_state=1)
    actual_h = sample["finish"] / 3600
    pred_h = riegel_from_half(sample["first_half"]) / 3600
    ax1.scatter(pred_h, actual_h, s=8, color=ACCENT, alpha=0.25, linewidths=0)
    lo, hi = min(actual_h.min(), pred_h.min()), max(actual_h.max(), pred_h.max())
    ax1.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2)
    ax1.set_xlabel("Predicted finish time (hours)", color=MUTED)
    ax1.set_ylabel("Actual finish time (hours)", color=MUTED)
    ax1.set_title("Classic Riegel from the half split", loc="left", color=INK, fontsize=12)

    names = ["Classic Riegel", "Calibrated fade factor"]
    values = [metrics["riegel"]["mae_min"], metrics["calibrated"]["mae_min"]]
    bars = ax2.bar(names, values, color=ACCENT, width=0.5)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f} min", ha="center", va="bottom", color=INK)
    ax2.set_ylim(0, max(values) * 1.25)
    ax2.set_title("Average miss on the test set (lower is better)", loc="left", color=INK, fontsize=12)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)

    headline = "Week 1 baselines: the numbers my model has to beat"
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

    print(f"Rows in: {report['rows_in']:,}   Rows kept: {report['rows_out']:,}")
    for reason, count in report["dropped"].items():
        print(f"  dropped {count:>7,}  {reason}")

    train, test = split_train_test(data, test_size=0.2, seed=42)
    print(f"\nTrain rows: {len(train):,}   Test rows: {len(test):,}")

    factor = fit_fade_factor(train["first_half"], train["finish"])
    metrics = {
        "riegel": error_summary(test["finish"], riegel_from_half(test["first_half"])),
        "calibrated": error_summary(test["finish"], calibrated_predict(test["first_half"], factor)),
        "fade_factor": factor,
        "synthetic": synthetic,
        "cleaning": report,
    }

    print(f"\nMedian finish time on the test set: {hms(test['finish_min'].median())}")
    print(f"Fade factor fit on train (finish divided by first half): {factor:.4f}\n")
    print(f"{'Baseline':<26}{'MAE':>10}{'RMSE':>10}{'Median':>10}{'Bias':>10}   (minutes)")
    for key, label in (("riegel", "Classic Riegel (1.06)"), ("calibrated", "Calibrated fade factor")):
        m = metrics[key]
        print(
            f"{label:<26}{m['mae_min']:>10.2f}{m['rmse_min']:>10.2f}"
            f"{m['median_abs_error_min']:>10.2f}{m['bias_min']:>+10.2f}"
        )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    tag = "synthetic" if synthetic else "real"
    (out_dir / f"week1_metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    chart_path = out_dir / f"week1_baseline_{tag}.png"
    make_chart(test, riegel_from_half(test["first_half"]), metrics, synthetic, chart_path)
    print(f"\nSaved {chart_path.relative_to(ROOT)} and the metrics file in outputs/")


if __name__ == "__main__":
    main()
