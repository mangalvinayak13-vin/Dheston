# Per-bucket in-sample implied-vol RMSE

118 sample days (every 10th test-period trading day, 2022-01-03 to 2026-09-30); all quotes of each day. Bucket value = RMSE (vol points) over the quotes of the bucket pooled over days; cells with fewer than 40 quotes are greyed. Numbers come from `exp_buckets.py`.

Same-day independent fits (bs, heston, dh, dhj) use the `dh2.calib.fit_day` objective with 4 starts. 'DH2.0' rows are the saved walk-forward state fits (`walk_dh_main.pkl`, `walk_dhj_main.pkl`): structural parameters shared over a 40-day window, only the two variance levels fitted on the day; their quote-level errors are recomputed through `dh2.models` and agree with the stored per-day RMSE.

## Pooled RMSE

| model | pooled IV RMSE (vol points) | quotes |
|---|---|---|
| Black-Scholes, one vol (same-day fit) | 6.131 | 53046 |
| Heston (same-day fit) | 3.669 | 53046 |
| Double Heston (same-day fit) | 2.912 | 53046 |
| Double Heston + jumps (same-day fit) | 0.774 | 53046 |
| DH2.0 diffusion (walk-forward, state fit) | 3.068 | 53046 |
| DH2.0 + jumps (walk-forward, state fit) | 1.154 | 53046 |

## Highlights (all from the tables below and `bucket_rmse.csv`)

- Pooled over quotes, the same-day error falls from 6.13 (one vol) to 3.67 (Heston), 2.91 (Double Heston) and 0.77 (with jumps). The DH2.0 walk-forward state fits score 3.07 (diffusion) and 1.15 (jumps): sharing the structure over a window costs fit quality relative to a fresh ten- or thirteen-parameter fit (diffusion +0.16, jumps +0.38). These are pooled-quote numbers; the per-day means in `results/attribution/` weight days differently, so the two do not coincide.
- The error lives in the short-dated wings. In the 3-7 day, deep-put bucket (ln(K/F) from -22% to -12%) the errors are: one vol 29.2, Heston 22.1, Double Heston 19.1, with jumps 1.48; the 8-14 day deep-put bucket: 9.2 without and 0.96 with jumps. Without jumps no diffusion model gets the wings of the shortest expiries (the 3-7 day call wing at +6 to +12% is 12.1 for Heston and 10.1 for Double Heston).
- Far from the wings every smile model is close: for expiries beyond 120 days all bucket errors are between 0.43 and 1.00 vol points (one vol: 1.58 to 4.32).
- The walk-forward jump model is worse than a fresh daily jump fit mostly at the shortest expiries: 3-7 days deep puts 3.65 vs 1.48; for expiries beyond 30 days the gap per bucket is 0.04 to 0.35 vol points. The diffusion walk-forward fit is close to a fresh diffusion fit in the wings (neither reaches them).
- The jump model's highest bucket errors are in the 3-7 day row (up to 1.48 vol points), against at most 0.96 in every longer-dated row.

## RMSE by bucket (rows: days to expiry, columns: ln(K/F))

**Black-Scholes, one vol (same-day fit)**

| | -22 to -12% | -12 to -6% | -6 to -2% | -2 to +2% | +2 to +6% | +6 to +12% | +12 to +22% |
|---|---|---|---|---|---|---|---|
| 3-7 d | 29.23 | 17.00 | 7.07 | 2.46 | 4.51 | 14.22 | n<40 |
| 8-14 d | 21.72 | 10.50 | 4.00 | 2.07 | 2.04 | 6.70 | 16.66 |
| 15-30 d | 14.14 | 6.49 | 2.53 | 1.49 | 1.44 | 2.90 | 8.46 |
| 31-60 d | 8.73 | 3.94 | 1.55 | 1.12 | 1.61 | 1.66 | 4.88 |
| 61-120 d | 6.45 | 2.89 | 1.18 | 0.96 | 1.58 | 1.96 | 3.14 |
| > 120 d | 4.32 | 2.62 | 1.88 | 1.58 | 1.72 | 2.09 | 2.29 |

**Heston (same-day fit)**

| | -22 to -12% | -12 to -6% | -6 to -2% | -2 to +2% | +2 to +6% | +6 to +12% | +12 to +22% |
|---|---|---|---|---|---|---|---|
| 3-7 d | 22.06 | 10.91 | 3.70 | 1.61 | 3.97 | 12.13 | n<40 |
| 8-14 d | 12.30 | 4.26 | 1.16 | 0.91 | 1.35 | 5.01 | 12.77 |
| 15-30 d | 5.06 | 1.43 | 0.82 | 1.02 | 0.82 | 1.85 | 5.61 |
| 31-60 d | 1.66 | 0.94 | 0.69 | 0.90 | 0.72 | 0.89 | 2.60 |
| 61-120 d | 0.92 | 0.85 | 0.63 | 0.79 | 0.83 | 1.00 | 1.32 |
| > 120 d | 0.90 | 1.00 | 0.96 | 0.81 | 0.66 | 0.83 | 0.76 |

**Double Heston (same-day fit)**

| | -22 to -12% | -12 to -6% | -6 to -2% | -2 to +2% | +2 to +6% | +6 to +12% | +12 to +22% |
|---|---|---|---|---|---|---|---|
| 3-7 d | 19.10 | 8.96 | 2.90 | 1.41 | 3.27 | 10.10 | n<40 |
| 8-14 d | 9.16 | 2.75 | 0.91 | 0.79 | 0.92 | 3.28 | 10.17 |
| 15-30 d | 3.14 | 0.92 | 0.70 | 0.96 | 0.77 | 0.98 | 3.94 |
| 31-60 d | 0.91 | 0.65 | 0.59 | 0.80 | 0.59 | 0.75 | 1.46 |
| 61-120 d | 0.70 | 0.55 | 0.52 | 0.67 | 0.78 | 0.86 | 0.79 |
| > 120 d | 0.47 | 0.46 | 0.53 | 0.51 | 0.57 | 0.54 | 0.56 |

**Double Heston + jumps (same-day fit)**

| | -22 to -12% | -12 to -6% | -6 to -2% | -2 to +2% | +2 to +6% | +6 to +12% | +12 to +22% |
|---|---|---|---|---|---|---|---|
| 3-7 d | 1.48 | 0.88 | 1.39 | 1.30 | 1.10 | 0.88 | n<40 |
| 8-14 d | 0.96 | 0.76 | 0.80 | 0.68 | 0.67 | 0.48 | 0.61 |
| 15-30 d | 0.84 | 0.76 | 0.67 | 0.94 | 0.72 | 0.61 | 0.69 |
| 31-60 d | 0.85 | 0.63 | 0.57 | 0.77 | 0.59 | 0.72 | 0.67 |
| 61-120 d | 0.74 | 0.57 | 0.50 | 0.63 | 0.76 | 0.85 | 0.58 |
| > 120 d | 0.43 | 0.43 | 0.51 | 0.51 | 0.51 | 0.47 | 0.46 |

**DH2.0 diffusion (walk-forward, state fit)**

| | -22 to -12% | -12 to -6% | -6 to -2% | -2 to +2% | +2 to +6% | +6 to +12% | +12 to +22% |
|---|---|---|---|---|---|---|---|
| 3-7 d | 19.42 | 8.91 | 3.05 | 1.77 | 3.65 | 10.97 | n<40 |
| 8-14 d | 9.02 | 2.82 | 1.18 | 1.00 | 1.22 | 3.85 | 11.01 |
| 15-30 d | 3.22 | 1.07 | 0.87 | 1.03 | 0.95 | 1.42 | 4.40 |
| 31-60 d | 1.14 | 0.83 | 0.69 | 0.87 | 0.73 | 0.91 | 1.51 |
| 61-120 d | 0.92 | 0.76 | 0.67 | 0.75 | 0.84 | 0.99 | 0.79 |
| > 120 d | 0.79 | 0.70 | 0.65 | 0.67 | 0.75 | 0.81 | 0.76 |

**DH2.0 + jumps (walk-forward, state fit)**

| | -22 to -12% | -12 to -6% | -6 to -2% | -2 to +2% | +2 to +6% | +6 to +12% | +12 to +22% |
|---|---|---|---|---|---|---|---|
| 3-7 d | 3.65 | 2.29 | 2.03 | 1.67 | 1.65 | 1.93 | n<40 |
| 8-14 d | 1.63 | 1.52 | 1.24 | 0.94 | 1.04 | 1.09 | 1.31 |
| 15-30 d | 1.27 | 1.05 | 0.91 | 1.02 | 0.94 | 1.02 | 1.28 |
| 31-60 d | 1.10 | 0.84 | 0.70 | 0.84 | 0.71 | 0.86 | 0.85 |
| 61-120 d | 0.99 | 0.80 | 0.64 | 0.72 | 0.82 | 0.97 | 0.62 |
| > 120 d | 0.75 | 0.69 | 0.66 | 0.71 | 0.75 | 0.81 | 0.70 |

## Limits, and what did not work

- Buckets pool 53046 quotes over 118 days that are not independent within a day; no confidence intervals are given (a shown bucket rests on 213 to 4128 quotes). Cells under 40 quotes are greyed, notably the 3-7 day, +12 to +22% cell.
- Same-day fits are fitted to the whole day, so these are in-sample errors; they show where a model CAN fit, not forecast accuracy (that is `results/attribution/`).
- The independent same-day fits use `fit_day`'s 4 starts; a few days may sit in a local optimum, which would inflate the corresponding rows (the identifiability study uses 16 starts).
- The heatmap colour scale is square-root (labelled on the colourbar) because the errors span 0.4 to 29 vol points; the printed numbers are exact.
- Nothing failed; the walk-forward quote-level errors reproduce the stored per-day RMSE exactly (max difference 0 over 236 day-models, `logs/c2_buckets.log`).

## Files

`results/buckets/bucket_rmse.csv` (every cell: n, RMSE, bias, mean absolute error), `bucket_summary.csv` (pooled RMSE), `heatmaps_rmse.png` and `heatmaps_counts.png` (300 dpi); code `exp_buckets.py` with `c2_common.py`; fits from `results/_cache/daily_fits_v1.pkl` and `data/results/walk_dh_main.pkl`, `walk_dhj_main.pkl`.

