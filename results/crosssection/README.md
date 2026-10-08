# Cross-sectional out-of-sample test

Sample: every 10th trading day of the test period (2022-01-01 onward), 118 days. Independent per-day fits with exactly `dh2.calib.fit_day` (4 starts, soft-L1 vega-weighted IV objective, 14 quotes per expiry), on a training subset only; scored on held-out quotes, implied-vol RMSE in vol points. Starting vectors are built from the training quotes only. Numbers come from `exp_crosssection.py`; nothing is hand-edited.

## Headline (tables: `results/crosssection/summary.csv`, `pairs.csv`; per-day scores `per_day.csv`; figure `crosssection.png`)

Held-out implied-vol RMSE in vol points, mean over days [95% CI]; all paired Wilcoxon tests are two-sided on the per-day scores.

- **Random 20% of quotes held out (interpolation):** bs 5.79 [5.47, 6.12]; heston 3.24 [2.98, 3.52]; dh 2.49 [2.24, 2.74]; dhj 0.71 [0.64, 0.78]. Every model beats the one before it on at least 115 of 118 days (largest p-value among the three adjacent comparisons 6.0e-21); jumps are the largest step (dh -> dhj -1.78). Held-out error is within about 0.1 of the training-quote error for every model, so there is no sign of overfitting to the fitted quotes.
- **Fit on expiries <= 60 days, score on longer expiries (extrapolation in maturity):** bs 2.67 [2.53, 2.84]; heston 1.37 [1.21, 1.57]; dh 2.02 [1.80, 2.27]; dhj 1.41 [1.24, 1.62]. Single Heston and the jump model are level (+0.04 [-0.08, +0.16], p = 0.73); the diffusion Double Heston is WORSE than single Heston here (+0.65, better on only 33 of 118 days, p = 4.1e-08): a plausible reason (not tested separately) is that ten parameters fitted on short expiries alone extrapolate worse than five, which the jump model then recovers. All three smile models beat flat volatility.

## (a) Random 80/20 split, stratified by expiry, 3 seeds (seeds averaged per day)

| model | days | mean [95% CI] | median | 90th pct | seed 0 | seed 1 | seed 2 | training-quote RMSE |
|---|---|---|---|---|---|---|---|---|
| bs | 118 | 5.787 [5.469, 6.123] | 5.714 | 8.100 | 5.787 | 5.876 | 5.698 | 5.845 |
| heston | 118 | 3.245 [2.980, 3.523] | 2.840 | 5.228 | 3.231 | 3.303 | 3.200 | 3.339 |
| dh | 118 | 2.486 [2.242, 2.743] | 2.075 | 4.230 | 2.477 | 2.537 | 2.444 | 2.557 |
| dhj | 118 | 0.705 [0.639, 0.780] | 0.597 | 1.225 | 0.700 | 0.690 | 0.725 | 0.683 |

Paired differences (b minus a, negative = b better), two-sided Wilcoxon signed-rank on the per-day scores:

| a | b | mean a | mean b | b - a [95% CI] | days b better | Wilcoxon p |
|---|---|---|---|---|---|---|
| bs | heston | 5.787 | 3.245 | -2.542 [-2.725, -2.367] | 118/118 | 4.21e-21 |
| bs | dh | 5.787 | 2.486 | -3.301 [-3.471, -3.138] | 118/118 | 4.21e-21 |
| bs | dhj | 5.787 | 0.705 | -5.082 [-5.403, -4.785] | 118/118 | 4.21e-21 |
| heston | dh | 3.245 | 2.486 | -0.759 [-0.825, -0.693] | 115/118 | 6.02e-21 |
| heston | dhj | 3.245 | 0.705 | -2.539 [-2.798, -2.304] | 118/118 | 4.21e-21 |
| dh | dhj | 2.486 | 0.705 | -1.781 [-2.016, -1.565] | 118/118 | 4.21e-21 |

## (b) Fit on expiries with <= 60 calendar days, score on the longer expiries

| model | days | mean [95% CI] | median | 90th pct |
|---|---|---|---|---|
| bs | 118 | 2.674 [2.526, 2.843] | 2.480 | 3.574 |
| heston | 118 | 1.368 [1.210, 1.572] | 1.139 | 2.228 |
| dh | 118 | 2.022 [1.804, 2.268] | 1.699 | 3.597 |
| dhj | 118 | 1.410 [1.239, 1.619] | 1.106 | 2.410 |

| a | b | mean a | mean b | b - a [95% CI] | days b better | Wilcoxon p |
|---|---|---|---|---|---|---|
| bs | heston | 2.674 | 1.368 | -1.306 [-1.514, -1.063] | 108/118 | 2.28e-18 |
| bs | dh | 2.674 | 2.022 | -0.652 [-0.902, -0.372] | 93/118 | 2.6e-09 |
| bs | dhj | 2.674 | 1.410 | -1.264 [-1.472, -1.022] | 109/118 | 9.64e-19 |
| heston | dh | 1.368 | 2.022 | +0.653 [+0.440, +0.871] | 33/118 | 4.07e-08 |
| heston | dhj | 1.368 | 1.410 | +0.042 [-0.079, +0.164] | 65/118 | 0.728 |
| dh | dhj | 2.022 | 1.410 | -0.611 [-0.820, -0.410] | 83/118 | 9.02e-08 |

The maturity split needs at least two training expiries and ten held-out quotes; 0 of 118 days were skipped for that reason. Held-out quotes per day (median): random 89, maturity 74.

## Reading it, and limits

- The days are 10 trading days apart, so they are close to independent; the random-split seeds for one day are not, which is why the seeds are averaged before any test.
- Random held-out quotes sit between fitted quotes of the same expiry (interpolation). The maturity split asks the model to extrapolate in maturity, which only the term-structure part of the model can do; a model with a flat or short-memory term structure is expected to lose there.
- Fits use 14 quotes per expiry (the `fit_day` thinning) taken from the training quotes, so a held-out quote can lie between two fitted ones but never coincides with one.
- `training-quote RMSE` is the fit error on the training quotes (all of them, not only the 14 per expiry that enter the objective).

## What did not work, and what was not done

- No fit failed (no error column in `per_split.csv`). Results are cached in `results/_cache/xsec_v1.pkl`, so the README can be regenerated without refitting.
- The reverse maturity split (fit long, score short) was not required and not run.
- Each model is fitted with `fit_day`'s 4 starts, not a larger search; a model could do better with more starts, and the ten- and thirteen-parameter models are the ones most exposed to a poor local optimum.
- The two settings score different quote sets (random: all expiries, mostly near-dated; maturity: only expiries beyond 60 days), so errors are comparable across models within a setting, not across the two settings.

## Files

`results/crosssection/summary.csv`, `pairs.csv`, `per_day.csv`, `per_split.csv` (every day, split and model), `crosssection.png` (300 dpi); code `exp_crosssection.py` with `c2_common.py`.
