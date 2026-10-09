# Is the model's lead just a wing-extrapolation effect?

Script `exp_wing_benchmark.py`; tables `summary.csv` and `per_day.csv` in this folder. Test period 2022-01-03 onward, 1,173 days, mean of
daily implied-volatility RMSE in vol points, paired block bootstrap (20-day blocks) for the intervals.

**Why this exists.** The independent verification (`results/VERIFICATION.md`, finding L1) showed that the "time-scaled smile" benchmark
used for the headline repeats the edge value for any strike outside the range yesterday quoted, and that inside that range it is slightly
better than the jump model + misfit. This experiment adds a second model-free benchmark that continues the wings linearly (the line through
yesterday's 3 outermost quotes, floored at 2 vol points). The setting "3 quotes" was fixed before any score was computed; nothing was tuned.
It was added after the verification, not before the headline, and only this one variant was run.

## Headline (source: summary.csv)

| Quotes | Edge-value benchmark | Linear-wing benchmark | Jump model + misfit | Model minus linear-wing benchmark [95% interval] | Days model is ahead |
|---|---|---|---|---|---|
| All (524,959 quotes) | 1.236 | 1.078 | 1.034 | **-0.045** [-0.061, -0.032] | 71% |
| Inside yesterday's quoted range (93.5%) | 1.006 | 1.006 | 1.018 | **+0.012** [+0.009, +0.016] | 52% |
| Outside it (6.5%) | 2.288 | 1.511 | 1.100 | **-0.411** [-0.503, -0.338] | 81% |

- The earlier headline lead (-0.202 vol points) was against the weaker benchmark. A simple wing extrapolation closes **78%** of it
  (benchmark 1.236 to 1.078); the model's remaining lead is **-0.045** vol points, still statistically clear (interval excludes zero).
- Where yesterday's smile covers the strike (93.5% of quotes) the benchmark is **slightly better** than the model (+0.012, the model ahead
  on 52% of days). The model's value is in strikes yesterday did not quote.
- Not tested: a stronger wing benchmark (an SVI continuation, or tuning the number of edge quotes). The lead over such a benchmark could be
  smaller still. The model was not tuned against these benchmarks either.

Inside the range the two benchmarks are identical by construction (they differ only outside it).
