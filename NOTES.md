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

**What does the model get to see, and what is off limits?**
Only first half time, age, and gender, the same things the baselines see. The raw file also has second half time, positive split, and percent change, but those contain the answer, so I never load them. If the model could see them, the score would be fake.

**Why predict the fade ratio instead of the finish time?**
A runner's first half time already tells me most of the story. What I don't know is how much they slow down, so the model predicts finish divided by first half and the code multiplies it back.

**How did I choose the model settings without peeking at the test set?**
I used five fold cross validation inside the training runners only. The 3,434 test runners were used once, at the very end.

**Did machine learning beat the baseline?**
Yes, but only a little. The average miss went from 9.79 to 9.25 minutes. When I resampled the test runners, the gain stayed between 0.39 and 0.70 minutes, so it's real but small. Ridge regression got most of that gain at 9.44, so the fancier boosted trees only added a bit more. The median miss barely changed, 7.30 to 7.24. The clearest win is bias: the baseline runs about 3 minutes too fast, and the model is close to zero.

**Where is the error?**
Slower runners are the hardest to predict, about 11 to 12 minutes off, and faster runners are the easiest, about 7.5. Runners 50 and over have a bigger miss than younger runners.

**What does the prediction range show?**
An 80 percent range held the real finish 80 percent of the time on the test set, and it's about 28 minutes wide. So the range is honest, but a halfway time can't narrow the finish much more than that.

**What surprised me?**
I expected machine learning to beat the formula by a lot, and it only won by about half a minute. The first half time already carries most of the information, so there isn't much left for a model to learn. It made me realize that a better model won't fix this. Better inputs will, which is why Strava training data is the next step.

**What I did not get to**
In Week 1 I said I wanted to revisit the 50 runners I dropped. I did not do that this week, so it moves to Week 3.

## Open questions for Week 3

1. How do I feed a Strava training effort into a model trained on race splits?
2. Is a 28-minute range useful to a runner, or does it need a better input than the half split?
3. How should the weekly plan adjust when a week of training is missed?
4. Were the 50 dropped runners timing errors or real blow-ups?
