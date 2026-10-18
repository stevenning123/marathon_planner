# Decision Log

Rewrite each answer in your own words. This file becomes your weekly update script and your final report, and it is what you will use to answer questions from the active body.

## Week 1

**Why does the project start with a baseline before any machine learning?**
Without a number to beat, a model's error means nothing. Riegel is the number runners already use, so beating it is a result anyone can understand.

**Why two baselines?**
Classic Riegel uses a fixed exponent of 1.06. The calibrated baseline learns the typical finish to first half ratio from the training data. The model should beat the stronger one, because beating only the weak one proves less.

**What did the cleaning step remove, and why?**
Fill this in after running on the real file. The script prints the counts by reason: missing values, zero or DNF times, finish times outside 2 to 8 hours, ages outside 18 to 90, and first halves that are not 35 to 65 percent of the finish time.

**Why a fixed seed on the train and test split?**
Every model gets scored on the exact same test rows, so comparisons are fair and results can be reproduced.

**Why MAE as the main metric?**
It is the average miss in minutes, which a runner understands immediately. RMSE and bias are reported too: RMSE punishes big misses, and bias shows whether predictions run fast or slow.

**What surprised me?**
Fill this in after you look at the real numbers.

## Open questions for Week 2

1. Which features beyond first half time, age, and gender could help?
2. How much better than the calibrated baseline is a realistic result?
3. How should uncertainty be shown to a runner?
