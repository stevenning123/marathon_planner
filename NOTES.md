# Decision Log

Rewrite each answer in your own words. This file becomes your weekly update script and your final report, and it is what you will use to answer questions from the active body.

## Week 1

**Why does the project start with a baseline before any machine learning?**
Without a number to beat, a model's error means nothing. Riegel is the number runners already use, so beating it is a result anyone can understand.

**Why two baselines?**
Classic Riegel uses a fixed exponent of 1.06. The calibrated baseline learns the typical finish to first half ratio from the training data. The model should beat the stronger one, because beating only the weak one proves less.

**What did the cleaning step remove, and why?**
I started with 17,220 runners and kept 17,170. The only rule that removed anyone was the split check. It dropped 50 runners, about 0.3 percent, whose first half was under 35 percent of their finish time, which means their second half took almost twice as long as their first. Some of those could be timing errors and some could be real blow ups, so I can't be sure which. I kept the rule because rows that extreme would pull the model around, but I want to revisit it in Week 2. Nothing else was dropped. There were no missing values, no zero times, and no ages or finish times out of range.

**Why a fixed seed on the train and test split?**
Every model gets scored on the exact same test rows, so comparisons are fair and results can be reproduced.

**Why MAE as the main metric?**
It is the average miss in minutes, which a runner understands immediately. RMSE and bias are reported too: RMSE punishes big misses, and bias shows whether predictions run fast or slow.

**What surprised me?**
I expected Riegel to be close, since runners have used it forever. On the Boston data, it misses by 11 minutes on average and runs about 8 minutes too fast. Boston runners slow down more in the second half than the formula expects, and I think the late-course hills are part of why, though I haven't tested that. A simple calibrated factor already cuts the average miss to 9.8 minutes, so my model has to beat 9.8, not 11. That changed how I think about the project, because beating a weak baseline would not have meant much.

## Open questions for Week 2

1. Which features beyond first half time, age, and gender could help?
2. How much better than the calibrated baseline is a realistic result?
3. How should uncertainty be shown to a runner?
