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
