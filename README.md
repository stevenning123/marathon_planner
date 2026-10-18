# Marathon Finish Time Predictor and Adaptive Training Planner

A personal project for the Alpha Kappa Psi pledgeship. The goal is to predict a runner's marathon finish time from real race data, beat a classic baseline formula with a machine learning model, and build a training plan that adjusts to how each week actually goes.

## Where things stand

Week 1 is done when this works on real data:

1. Load and clean the Boston Marathon 2024 results.
2. Split the data into train and test sets with a fixed seed.
3. Score two baselines on the test set: classic Riegel, and a calibrated version.

Week 2 adds the machine learning model, which has to beat both baselines.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

## Get the data

1. Make a free Kaggle account.
2. Search Kaggle for "2024 Boston Marathon weather and splits" by runningwithrock.
3. Download it and put the results CSV in `data/raw/`.
4. Check the license on the dataset page and cite it in your report.

The cleaner needs age, gender, first half time, and finish time. Column names are matched loosely. If the file uses different names, the error message lists the columns it found, and you can add them to `ALIASES` in `marathon/data.py`.

## Run it

```
python scripts/run_week1.py data/raw/YOUR_FILE.csv
```

This prints how many rows were dropped and why, the baseline errors in minutes, and saves a chart and a metrics file in `outputs/`.

To check that the code runs before you have the data:

```
python scripts/run_week1.py synthetic
```

That uses fake data and stamps the chart "SYNTHETIC DATA, NOT REAL RESULTS". Never show those numbers.

## Run the tests

```
pytest
```

## What is in here

| File | What it does |
| --- | --- |
| `marathon/data.py` | Loads and cleans results, reports every dropped row, makes the train and test split |
| `marathon/baseline.py` | Riegel baseline, calibrated baseline, and error metrics in minutes |
| `marathon/synthetic.py` | Fake data for testing the code only |
| `scripts/run_week1.py` | Runs the whole Week 1 pipeline and saves the chart |
| `tests/` | Tests for the cleaning, the split, and the baseline math |
| `NOTES.md` | Decision log, written in your own words |

## Known limits

1. The first half split is only known partway through a race, so this model predicts the finish from a split, not from training history. Training history comes in through the second dataset and your own Strava export.
2. Riegel was designed for different race distances, not for the two halves of one race, so it is a rough yardstick. That is why a calibrated baseline is included too.
3. No large public dataset links training logs to finish times. This is a stated limitation of the project.
4. The model is trained only on public data. Strava API data is not used for training, because Strava's terms prohibit it.
