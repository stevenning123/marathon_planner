import numpy as np
import pandas as pd
import pytest

from marathon.data import clean_results, split_train_test, standardize_columns, to_seconds
from marathon.synthetic import make_synthetic_results


def good_row(**overrides):
    row = {"Age": 40, "Gender": "M", "First Half": 7200, "Finish": 14700}
    row.update(overrides)
    return row


def test_to_seconds_accepts_numbers_and_clock_strings():
    assert to_seconds(pd.Series([3600, 7200])).tolist() == [3600.0, 7200.0]
    assert to_seconds(pd.Series(["1:00:00", "2:01:30", "45:10"])).tolist() == [3600.0, 7290.0, 2710.0]


def test_to_seconds_bad_text_becomes_nan():
    assert np.isnan(to_seconds(pd.Series(["1:00:00", "n/a", "2:00:00", "3:00:00"])).iloc[1])


def test_standardize_columns_matches_loose_names():
    raw = pd.DataFrame(columns=["Age", "Gender", "First Half", "Finish", "Bib"])
    out = standardize_columns(raw)
    assert {"age", "gender", "first_half", "finish"} <= set(out.columns)


def test_standardize_columns_raises_when_a_field_is_missing():
    raw = pd.DataFrame(columns=["Age", "Gender", "Finish"])
    with pytest.raises(ValueError, match="first_half"):
        standardize_columns(raw)


def test_clean_keeps_a_good_row_and_adds_columns():
    df, report = clean_results(pd.DataFrame([good_row()]))
    assert report["rows_in"] == 1 and report["rows_out"] == 1
    row = df.iloc[0]
    assert row["gender"] == "M"
    assert row["second_half"] == 14700 - 7200
    assert row["finish_min"] == pytest.approx(245.0)
    assert row["split_ratio"] == pytest.approx(7200 / 14700)


def test_clean_drops_each_kind_of_bad_row_once():
    raw = pd.DataFrame(
        [
            good_row(),
            good_row(Finish=0, **{"First Half": 0}),    # did not finish
            good_row(Age=None),                         # missing age
            good_row(Finish=5000, **{"First Half": 2500}),  # finish under 2 hours
            good_row(Age=12),                           # age out of range
            good_row(**{"First Half": 12000}),          # first half is 82 percent of finish
            good_row(Gender="?"),                       # unknown gender
        ]
    )
    df, report = clean_results(raw)
    assert len(df) == 1
    assert report["rows_out"] == 1
    assert sum(report["dropped"].values()) == 6
    assert report["rows_in"] == report["rows_out"] + sum(report["dropped"].values())


def test_clean_normalizes_gender_words():
    raw = pd.DataFrame([good_row(Gender="Male"), good_row(Gender="female")])
    df, _ = clean_results(raw)
    assert df["gender"].tolist() == ["M", "F"]


def test_split_is_deterministic_disjoint_and_complete():
    df = pd.DataFrame({"x": range(100)})
    train1, test1 = split_train_test(df, 0.2, seed=42)
    train2, test2 = split_train_test(df, 0.2, seed=42)
    assert test1["x"].tolist() == test2["x"].tolist()
    assert len(test1) == 20 and len(train1) == 80
    assert set(train1["x"]).isdisjoint(set(test1["x"]))
    assert set(train1["x"]) | set(test1["x"]) == set(range(100))


def test_synthetic_data_runs_through_the_cleaner():
    df, report = clean_results(make_synthetic_results(n=2000, seed=1))
    assert report["dropped"]["did_not_finish_or_zero_time"] > 0
    assert report["rows_out"] > 1800
    assert df["split_ratio"].between(0.35, 0.65).all()
