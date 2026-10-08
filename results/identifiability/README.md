# Identifiability of the Double Heston parameters (one day at a time)

118 sample days (every 10th test-period trading day); `dh` (10 parameters) and `dhj` (with Merton jumps, 13 parameters). Each day: 16 starts of the `dh2.calib.fit_day` fit (same objective, bounds, thinning and soft-L1 vega-weighted residual), every solution kept. Failed day-model jobs: 0. Numbers come from `exp_identifiability.py`; nothing is hand-edited.

Definitions: the residual is price error / vega, i.e. vol points. J is its Jacobian at the best solution in the internal parameter vector (kappa2 = kappa1 + dk). Primary J = the `jac` that scipy's `least_squares` returns (loss-modified for the robust loss); `fd` columns of the CSVs use a central finite-difference Jacobian of the raw residuals as a check. Condition numbers: *column-normalised* (each column scaled to unit length) and *range-scaled* (column j times the width of parameter j's bound interval); the raw one in internal units mixes variances (~0.01) with speeds (~1-15) and is reported only in `per_day.csv`. Standard errors: sqrt(diag((J'J)^-1) s^2) with the pseudo-inverse, s^2 = residual sum of squares / (m - p), shown as a share of the bound interval's width. A parameter 'at a bound' is within 2% of the interval width of either end. Two solutions are 'the same parameter set' when every parameter differs by less than 5% of its interval width (2% and 10% in the CSV).

## Headline

| | Double Heston | Double Heston + jumps |
|---|---|---|
| days | 118 | 118 |
| parameters | 10 | 13 |
| median condition number, columns normalised | 76.2 | 529 |
| median condition number, columns scaled by bound width | 1.29e+03 | 1.94e+04 |
| share of days with column-normalised condition number > 1e6 | 0.0% | 0.0% |
| median effective rank (singular values > 1e-3 of the largest, range-scaled) | 9 | 11 |
| share of parameters within 2% of a bound (all days pooled) | 41.0% | 38.7% |
| share of days with at least one parameter within 2% of a bound | 100.0% | 100.0% |
| share of parameters ON a bound (within 0.3% of the interval width) | 31.1% | 26.7% |
| share of days with at least one parameter on a bound (0.3%) | 100.0% | 100.0% |
| largest column-normalised condition number on any day | 1338 | 6681 |
| median column-normalised condition number, finite-difference Jacobian (check) | 66.6 | 533 |
| median (over parameters and days) standard error / bound width | 0.213 | 0.124 |
| mean number of the 16 starts within 1% of the best cost | 12.2 | 11.4 |
| mean number of distinct parameter sets within 1% of the best cost (5% tolerance) | 1.27 | 1.50 |
| same, 2% tolerance | 1.47 | 1.83 |
| same, 10% tolerance | 1.19 | 1.36 |
| share of days with two or more distinct sets within 1% of the best | 16.1% | 29.7% |
| median worst-start cost / best cost | 1.14 | 1.07 |

## Plain reading

- **Locally the problem is not singular.** The column-normalised Jacobian condition number is 76 (diffusion) and 529 (jumps) at the median, never above 1338 / 6681; the finite-difference check agrees (67 / 533). Every direction has some curvature at the optimum, and the jump model is the worse conditioned of the two.
- **But the data pin most parameters only loosely compared with what the model allows.** The median standard error is 21% (diffusion) and 12% (jumps) of a parameter's whole bound interval. Parameters whose standard error exceeds the entire interval on at least a quarter of days: diffusion: kappa1 (29%), theta1 (30%), rho1 (31%), dk=k2-k1 (27%); jumps: kappa1 (39%), theta1 (58%), dk=k2-k1 (46%). Tightly determined (median SE below 5% of the interval): diffusion: v0_1 (0.014), v0_2 (0.014), rho2 (0.045); jumps: v0_1 (0.028), v0_2 (0.030), lam (0.005).
- **Many optima sit on a bound.** 31% (diffusion) and 27% (jumps) of fitted parameters are within 0.3% of an interval end, and 100% / 100% of days have at least one. Most often on a bound (share of days): diffusion: rho1 (64%), dk=k2-k1 (55%), xi2 (91%); jumps: dk=k2-k1 (52%).
- **The cost surface is not a many-minima problem here.** Of 16 starts, on average 12.2 (diffusion) and 11.4 (jumps) reach a cost within 1% of the best; at the 5% tolerance the mean number of distinct parameter sets that fit equally well is 1.27 / 1.50, and 16% / 30% of days have two or more. This is far fewer than the 99.1% of surfaces with several equally good parameter sets that the 1.0 study reported; the definitions differ (1.0: a cost tolerance on price errors, no ordering constraint), and here kappa2 > kappa1 is imposed (removing the factor-swap duplicates) and the optimum is often pinned by a bound rather than by a second valley. What remains is non-identification by ridges and bounds, not by competing minima.
- **Fit quality and identification are separate questions.** Same-day IV RMSE is 2.56 (diffusion) and 0.69 (jumps) vol points; even for the jump model, which fits closely, most structural parameters are loosely determined by one day of prices, so the fitted surfaces, not the individual parameters, are what can be validated (the same conclusion as the project's earlier note).

## By parameter

| model | parameter | days within 2% of a bound | optimum position in its bound interval (quartiles) | median SE / width | days SE > width | mean spread of near-optimal sets / width |
|---|---|---|---|---|---|---|
| dh | v0_1 | 45% | 0.01 / 0.02 / 0.04 | 0.0141 | 0% | 0.00 |
| dh | kappa1 | 39% | 0.04 / 0.30 / 0.76 | 0.683 | 29% | 0.04 |
| dh | theta1 | 21% | 0.06 / 0.10 / 0.34 | 0.158 | 30% | 0.04 |
| dh | xi1 | 9% | 0.08 / 0.12 / 0.18 | 0.233 | 8% | 0.03 |
| dh | rho1 | 67% | 0.00 / 0.00 / 0.16 | 0.668 | 31% | 0.03 |
| dh | v0_2 | 42% | 0.01 / 0.02 / 0.05 | 0.0144 | 0% | 0.00 |
| dh | dk=k2-k1 | 58% | 0.00 / 0.23 / 0.70 | 0.68 | 27% | 0.05 |
| dh | theta2 | 31% | 0.01 / 0.03 / 0.06 | 0.0693 | 7% | 0.00 |
| dh | xi2 | 91% | 1.00 / 1.00 / 1.00 | 0.353 | 4% | 0.01 |
| dh | rho2 | 8% | 0.29 / 0.33 / 0.36 | 0.0455 | 6% | 0.02 |
| dhj | v0_1 | 53% | 0.00 / 0.02 / 0.03 | 0.0277 | 1% | 0.01 |
| dhj | kappa1 | 60% | 0.00 / 0.02 / 0.44 | 0.693 | 39% | 0.05 |
| dhj | theta1 | 31% | 0.06 / 0.36 / 0.87 | 5.12 | 58% | 0.08 |
| dhj | xi1 | 2% | 0.09 / 0.15 / 0.21 | 0.161 | 8% | 0.07 |
| dhj | rho1 | 39% | 0.00 / 0.08 / 0.19 | 0.304 | 19% | 0.04 |
| dhj | v0_2 | 47% | 0.00 / 0.02 / 0.06 | 0.0295 | 1% | 0.01 |
| dhj | dk=k2-k1 | 53% | 0.26 / 0.78 / 1.00 | 0.9 | 46% | 0.05 |
| dhj | theta2 | 63% | 0.00 / 0.01 / 0.03 | 0.06 | 3% | 0.02 |
| dhj | xi2 | 23% | 0.17 / 0.33 / 0.84 | 0.274 | 9% | 0.08 |
| dhj | rho2 | 26% | 0.01 / 0.23 / 0.32 | 0.17 | 20% | 0.05 |
| dhj | lam | 68% | 0.01 / 0.01 / 0.02 | 0.00539 | 0% | 0.00 |
| dhj | mu_j | 2% | 0.49 / 0.54 / 0.60 | 0.0648 | 2% | 0.00 |
| dhj | delta_j | 37% | 0.65 / 0.85 / 1.00 | 0.185 | 3% | 0.01 |

Figure: `identifiability.png` (300 dpi). Tables: `summary.csv`, `per_parameter.csv`, `per_day.csv`.


## Limits, and what did not work

- 'Within 2% of a bound' depends on the interval width: v0 (0.002 to 0.30), theta2 (0.002 to 0.25) and lam (0 to 8) have optima at small absolute values (median variance level about 0.009 per factor, because two factors share a total of about 0.018) that are only 1-3% into a wide interval, so the 2% share overstates real bound hits for them; the 0.3% 'on a bound' share is the stricter measure and is the one to quote.
- No job failed (0 failed day-model jobs). Nothing was left out of the requested analysis; the exact number of 'distinct' sets depends on the tolerance (reported at 2%, 5% and 10%).
- Standard errors from (J'J)^-1 are local and linear; for a parameter sitting on a bound they are not meaningful, and with a robust (soft-L1) loss they describe the loss-weighted problem. Treat them as an indicator of which directions are weakly pinned, not as confidence intervals.
- 16 starts is a finite search: more distinct near-optimal sets may exist. The count is a lower bound on non-uniqueness.
- The bounds are those of `dh2.calib.BOUNDS`; shares 'near a bound' and 'SE / width' depend on how wide they are.

## Files

`results/identifiability/summary.csv` (headline table), `per_parameter.csv`, `per_day.csv` (condition numbers from both Jacobians, standard errors summary, bound and multi-start statistics for every day and model), `identifiability.png` (300 dpi); code `exp_identifiability.py` with `c2_common.py` (`fit_day_full`); every start's solution and Jacobian is cached in `results/_cache/ident_v1.pkl`.
