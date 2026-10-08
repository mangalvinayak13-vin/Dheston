# DH1 -> DH2.0 attribution waterfall

Sample: every 10th trading day of the test period (2022-01-01 onward), 118 days (2022-01-03 to 2026-09-30); all numbers below use the 118 days (same-day) and 118 days (next-day) on which every step has a value. The sample is fixed by that rule, identical for every step and model, and no design choice was tuned on it. Numbers are produced by `exp_attribution.py`; nothing here is hand-edited.

Error = implied-volatility RMSE in volatility points over all of a day's clean-surface quotes (the same scoring set for every step). Same-day: the day-t fit scored on day t. Next-day: the same parameters carried to day t+1 (its forwards and expiries given), scored on day t+1. Intervals are 95% bootstrap intervals over days; the p-values are paired two-sided Wilcoxon tests against the previous step.

## The waterfall

| step | what changes | same-day mean [95%] | same-day median | step change (paired) | next-day mean [95%] | next-day median | step change (paired) | next-day p |
|---|---|---|---|---|---|---|---|---|
| 0 DH1 (1.0 recipe) | 1.0 data + price objective + latent 5-start optimiser | 4.839 [4.551, 5.144] | 4.539 |  | 5.027 [4.680, 5.393] | 4.707 |  |  |
| 1 + clean data | traded OTM quotes, parity forwards, price >= Rs 1, >= 3 days | 4.705 [4.404, 5.027] | 4.443 | -0.134 [-0.292, +0.009] | 4.929 [4.565, 5.317] | 4.552 | -0.098 [-0.255, +0.047] | 0.265 |
| 2 + IV-space vega-weighted objective | vega-weighted implied-vol residual, soft-L1 | 3.589 [3.333, 3.868] | 3.310 | -1.116 [-1.200, -1.034] | 3.881 [3.550, 4.244] | 3.334 | -1.048 [-1.137, -0.964] | 4.32e-21 |
| 3 + global multi-start, kappa2 > kappa1 | dh2.calib.fit_day: flat-corner + random starts, box bounds, ordering | 2.555 [2.312, 2.812] | 2.219 | -1.034 [-1.149, -0.920] | 2.838 [2.531, 3.181] | 2.469 | -1.043 [-1.151, -0.936] | 7.38e-21 |
| 4 + structure/state split, walk-forward | slow parameters over 40-day window, refit every 10 days; daily state | 2.683 [2.430, 2.952] | 2.338 | +0.128 [+0.081, +0.179] | 2.917 [2.596, 3.277] | 2.561 | +0.078 [+0.029, +0.133] | 0.0026 |
| 5 + Merton jumps | jump intensity, mean and size in the structure | 0.994 [0.893, 1.102] | 0.842 | -1.689 [-1.910, -1.480] | 1.254 [1.106, 1.415] | 1.000 | -1.662 [-1.943, -1.412] | 8.17e-21 |
| 6 + yesterday's misfit carried | next-day forecast = model + yesterday's remaining misfit | 0.994 [0.893, 1.102] | 0.842 | +0.000 [+0.000, +0.000] (carried misfit: same fit) | 1.100 [0.965, 1.246] | 0.889 | -0.155 [-0.209, -0.104] | 3.89e-08 |

Context rows, same days:

| reference | same-day mean | next-day mean [95%] | next-day median |
|---|---|---|---|
| yesterday's smile (model-free) | n/a | 1.585 [1.348, 1.843] | 1.151 |
| time-scaled smile (model-free) | n/a | 1.333 [1.156, 1.529] | 1.002 |
| flat Black-Scholes, daily refit, carried | 5.839 | 5.970 [5.591, 6.364] | 5.806 |
| single Heston, independent daily fits | 3.346 | 3.585 [3.246, 3.949] | 3.271 |
| DH + jumps, independent daily fits | 0.694 | 1.142 [1.010, 1.284] | 0.926 |

Overall, from DH1 to the full DH2.0 forecast: same-day 4.839 -> 0.994 vol points, next-day 5.027 -> 1.100 vol points. Figure: `waterfall.png` (300 dpi); per-day numbers: `per_day.csv`; table: `waterfall.csv`.

## How DH1 was rebuilt (and what could not be recovered)

DH1 is the 1.0 research project's Double Heston calibration of NIFTY. The script that produced `outputs/real_markets/nifty_comparison/nifty_comparison_summary.json` was lost with a session scratchpad (the header of `src/mentor_dh_pinn/baselines.py` in the old repository says so). The recipe was rebuilt, read-only, from the surviving modules (`src/mentor_dh_pinn/nifty_panel.py`, `scripts/mentor_dh_pinn/evaluate_real_markets.py`, `src/mentor_dh_pinn/baselines.py`, `params_v2.py`; `legacy_streamlit_site/models.py` holds only a single-Heston vega-weighted calibrator, not the Double Heston one):

- **Data**: NSE settlement prices (not last-trade closes), expiries kept when they have traded volume and at least 10,000 open interest in total, any days to expiry, no minimum price, forward and discount per expiry from a Huber-reweighted put-call-parity regression, out-of-the-money side, 0.78 to 1.22 of the forward.
- **Quotes**: every third strike of each expiry left out of the fit (the fit uses the other two thirds).
- **Objective**: price space, (model forward call - market forward call) / forward, no vega weights, plain least squares.
- **Optimiser**: scipy `least_squares` (trf), diff_step 1e-6, max_nfev 220, in unbounded latent coordinates that build in kappa_fast > kappa_slow, the Feller condition and the correlation disk; five fixed start vectors, keep the lowest cost. The first start violates Feller and is skipped silently by 1.0's own loop, so four starts run.
- **Pricer**: `pricer/models.py` (as instructed), not 1.0's own quadrature pricer.

What does not match: (1) the old module requires 10,000 open interest per contract, but only a per-expiry floor reproduces 1.0's own counts on 2026-04-08 (833 quotes, 10 expiries); that is what is used. (2) Pre-June-2024 NSE files carry no index level, so the level used only for the parity-validity gates is the strike where call = put. (3) The held-out set and the Black-Scholes fit could not be matched exactly (below).

### Replication check on 1.0's own ten dates (8-22 April 2026): how close is it?

| quantity (10 dates) | rebuilt | published by 1.0 | verdict |
|---|---|---|---|
| quotes per date | 795-936 | 795-936 | identical on every date |
| expiries per date | 10-11 | 10-11 | identical on every date |
| held-out quotes per date | 170-193 | 133-142 | NOT matched (rebuilt set is larger) |
| Black-Scholes one-sigma fit | 0.1688 (mean) | 0.1537 (mean) | NOT matched: 10% higher |
| median held-out IV RMSE, Black-Scholes (vol points) | 2.98 | 3.04 | close in aggregate; per-date correlation 0.25 |
| median held-out IV RMSE, cold Double Heston (vol points) | 2.53 | 2.37 | +7% in aggregate; per-date correlation 0.74 |

**Plain reading.** The data universe of 1.0 is reproduced exactly (same quote and expiry counts on all ten dates, same index level), which fixes the quote set, but the 1.0 held-out third and its Black-Scholes sigma are not: the script that produced the published table was lost, and I could not recover which quotes it fitted and held out. The rebuilt Double Heston optimiser lands within 7% of 1.0's published median (the pricer and the exact held-out set differ), so it behaves like the 1.0 recipe in aggregate, but this is a close rebuild, not a verified reproduction; date by date it agrees only moderately. On the 118 sample days the same rebuilt recipe scores 2.27 vol points mean on 1.0's own metric (held-out third, 20-800 days), in line with the 2.37 that 1.0 published for April 2026, whereas step 0 above scores 4.84 same-day: the difference is the scoring set, not a different fit. The waterfall scores every quote of the 2.0 clean surface, including 3-19 day expiries and the far wings (|ln K/F| up to 0.22), which a price-space fit with no jump component fits worst; 1.0's metric excluded expiries under 20 days. Table: `replication_check.csv`.

## What each data step changes

On the sampled days DH1's data has on average 614 quotes over 10.9 expiries; the 2.0 clean surface has 450 quotes over 8.9 expiries. 68% of DH1's quotes are also in the clean surface and 93% of the clean quotes are in DH1's. Of DH1's quotes, 27% did not trade that day (zero volume), 20% have zero open interest, 5.1% are priced below Rs 1 and 2.1% expire in under 3 days: these are what the clean data removes. For the same expiry the two forwards differ by a median of 8.7 basis points. For quotes in both sets the settlement price and the close are identical in the median (0.00% log difference) but not always: 8.9% of the common quotes differ by more than 1% (mean absolute log difference 6.74%). The change at step 1 is therefore mainly which quotes and expiries enter (untraded, near-zero-priced and very-short-dated ones leave), plus a minority of different prices. (`data_differences.csv`)

### By year (mean IV RMSE, vol points)

| year | days | same-day: DH1 | step 3 | step 5 | next-day: DH1 | step 3 | step 4 | step 5 | step 6 | time-scaled smile |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 25 | 5.15 | 3.21 | 1.23 | 5.30 | 3.41 | 3.52 | 1.49 | 1.39 | 1.50 |
| 2023 | 25 | 5.31 | 2.62 | 0.80 | 5.53 | 2.95 | 2.96 | 0.98 | 0.91 | 1.29 |
| 2024 | 24 | 4.86 | 2.65 | 1.15 | 5.41 | 3.26 | 3.35 | 1.44 | 1.17 | 1.51 |
| 2025 | 25 | 4.59 | 2.30 | 0.92 | 4.17 | 1.90 | 1.97 | 0.83 | 0.68 | 0.85 |
| 2026 | 19 | 4.11 | 1.82 | 0.84 | 4.67 | 2.63 | 2.77 | 1.63 | 1.42 | 1.57 |

## Caveats

- Step 4 changes two things at once that cannot be separated with the saved results: the structural/state split and the walk-forward refits (shared parameters over a window). Same-day fit gets worse there by construction (a day's fit has two free numbers instead of ten); the gain is in the next-day column.
- Steps 0-2 fit a two-thirds subset of strikes (1.0's protocol); step 3 uses `fit_day`'s own subset (14 quotes per expiry). Scoring always uses every quote of the clean surface, including the ones a step did not fit.
- Step 2 changes the loss space, the weights and the robust loss together (they are one objective in `dh2/calib.py`).
- The sample is 118 days spaced 10 trading days apart, so a per-step difference can be small compared with day-to-day noise; use the paired intervals, not the means alone.
- Next-day for steps 0-3 carries all ten parameters; for steps 4-6 the structure is held and only the state is carried (that is the DH2.0 forecast).

## What did not work

- The exact 1.0 script behind `nifty_comparison_summary.json` could not be recovered (lost with a scratchpad); DH1 is a documented rebuild. The held-out set and the Black-Scholes sigma of 1.0 were not matched (table above); data universe and quote counts were.
- The per-contract open-interest floor in 1.0's surviving code (10,000) does not reproduce 1.0's own quote counts; zero is used for contracts and 10,000 for the expiry total.
- A first run of this script crashed at its last step (a numpy string passed to pandas Timestamp, in the data-differences table, after all fits were done); fixed, and the outputs here were regenerated from the cached fits without refitting.
- The waterfall cannot separate the structure/state split from the walk-forward refits (step 4), nor the loss space from the robust loss (step 2); no saved result isolates them.

## Files

- `results/attribution/waterfall.csv` the table above (means, medians, bootstrap intervals, paired step changes, Wilcoxon p, days improved); `results/attribution/per_day.csv` every day, every step; `results/attribution/waterfall.png` (300 dpi); `results/attribution/replication_check.csv`; `results/attribution/data_differences.csv`.
- Code: `exp_attribution.py` (this table), `c2_dh1.py` (the DH1 rebuild), `c2_common.py` (sample, fits, statistics). Cached fits: `results/_cache/attr_steps_v1.pkl`, `daily_fits_v1.pkl`, `repl_v1.pkl`.
