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

## Week 2

**What does the model get to see, and what is off limits?**
Only first half time, age, and gender, the same things the baselines see. The raw file also has second half time, positive split, and percent change, but those contain the answer, so I never load them. If the model could see them, the score would be fake.

**Why predict the fade ratio instead of the finish time?**
A runner's first half time already tells me most of the story. What I don't know is how much they slow down, so the model predicts finish divided by first half and the code multiplies it back.

**How did I choose the model settings without peeking at the test set?**
I used five fold cross validation inside the training runners only. The 3,434 test runners were used once, at the very end.

**Did machine learning beat the baseline?**
Yes, but only a little. The average miss went from 9.79 to 9.25 minutes. When I resampled the test runners, the gain stayed between 0.39 and 0.70 minutes, so it is real but small. Ridge regression got most of that gain at 9.44, so the fancier boosted trees only added a bit more. The median miss barely changed, 7.30 to 7.24. The clearest win is bias: the baseline runs about 3 minutes too fast, and the model is close to zero.

**Where is the error?**
Slower runners are the hardest to predict, about 11 to 12 minutes off, and faster runners are the easiest, about 7.5. Runners 50 and over have a bigger miss than younger runners.

**What does the prediction range show?**
An 80 percent range held the real finish 80 percent of the time on the test set, and it is about 28 minutes wide. So the range is honest, but a halfway time cannot narrow the finish much more than that.

**What surprised me?**
I expected machine learning to beat the formula by a lot, and it only won by about half a minute. The first half time already carries most of the information, so there is not much left for a model to learn. It made me realize that a better model will not fix this. Better inputs will, which is why Strava training data is the next step.

## Week 3

**What did I find out about the 50 runners I dropped in Week 1?**
Not one of them has an impossible first half. Their median first half was 1:39:32, which is a 3:19 marathon pace and quicker than the typical runner in this field, but their median finish was 5:07:21, about 1 hour 23 minutes slower than the typical test runner (3:44:39). So they started fast and then slowed to almost half speed. That fits a real blowup better than a timing error, but I cannot prove it with this file, because a missed chip read at the halfway mat would look the same.

**Did dropping them change my results?**
Yes. My model misses by about 88 minutes on those 50 runners. If I had kept them in the test set, the average miss would go from 9.25 to 10.37 minutes, a 1.13 minute difference from only 50 runners. So my score is really the score for runners who pace a race in a normal way, and I should say that out loud when I present it.

**How does a Strava run become a marathon estimate?**
The bridge takes the fastest run of 10 km or more from the last 8 weeks and scales it up to 42.2 km with Riegel's formula. If there is no run that long, it falls back to runs of 5 km or more and shows a warning. Riegel is least reliable when it scales a short run a long way up. The range uses the width of my Week 2 range, about plus or minus 6 percent.

**Did the bridge work on my own marathon?**
No. I ran a marathon on July 26, 2026. Strava shows 42.6 km in 4:05:05 of moving time (4:16:21 on the clock; I will use my official time here if I have it). Using only runs from before that race, the bridge said 3:33:46, which is 31 minutes faster than I ran. The runs it used were short: my longest run in the 8 weeks before was 8.9 km. Riegel assumes a runner has trained for the marathon distance, and my Strava file does not show that. One race is an example, not an accuracy number, but it points at the real gap: the bridge sees speed and misses endurance.

**How much do I trust the bridge?**
Not as much as the Week 2 model. It is an assumption, not something I measured on real finishes. Riegel ran about 8 minutes too fast on Boston, a training run is usually easier than an all out race, and my Week 2 range was measured with a halfway time as the input, so the real range with a training run is probably wider. The app labels the estimate "best case" and warns when my longest recent run is under 24 km.

**How does the planner decide what to do?**
It uses common coaching rules of thumb, not rules learned from data: no more than 10 percent more distance per week, a cutback week every 4th week, a long run of at most 45 percent of the week and never over 32 km, and a 3 week taper. It aims for a peak week of about 65 km and only climbs as far as the 10 percent rule allows in the weeks I have, so a short runway shows up as a lower readiness rating. When I log a week, later weeks are rebuilt from what I really ran, so one missed week never forces a big jump.

**What can the planner not do?**
It does not predict how many minutes a missed week costs me. I do not have data that links training weeks to finish time, so it only reports readiness, based on whether the plan still reaches a long run of 29 km or more.

**What surprised me?**
I expected my marathon estimate to be close, and it was 31 minutes too fast on a race I actually ran. The formula only sees how fast I run, not how little long distance training I had behind me. I also did not expect fifty removed runners to move my score by more than a minute. Both of those told me that the data I leave out, and the training the formula cannot see, matter as much as the model.

## Open questions for Week 4

1. Can a small public training dataset like the Kaggle Marathon Time Predictions file (weekly distance and long run linked to finish time) fix the endurance gap that showed up in my own marathon check?
2. How should the app explain the readiness rating so a runner trusts it?
3. What should the 4 to 5 minute Week 4 video and the final demo show?
