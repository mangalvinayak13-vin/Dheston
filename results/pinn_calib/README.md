# PINN-accelerated calibration of the diffusion Double Heston (work package D, brief Phase 5)

Question: can the neural surrogate of work package B do the **global** search of a daily Double Heston calibration, with the **exact** Fourier pricer
only polishing, and still land where the pure exact calibration (the Phase-3 recipe, `dh2.calib.fit_day`: 4 starts, `max_nfev` 120) lands, faster?

**Short answer.** Yes on the quantity that is reported (same-day implied-vol RMSE, exact pricer): no detectable difference from the exact calibration, in 8.0x
less wall time and 11.6x fewer exact-pricer calls. The surrogate alone is *not* a usable fit (its objective cost is 18% above the exact one and its parameters are
far from the exact ones), so the polish matters. The accelerated fit reaches a slightly higher objective cost than the exact one (+0.8% on average, up to +10.9% on
one day), it does not converge inside its small polish budget on 115 of 118 days, and it is worse on the days whose exact optimum lies outside the surrogate's box.
All numbers below are copied from the CSV tables named next to them (nothing is edited by hand); every fit metric is computed with the **exact** pricer.

## 1. What was built (all new files; nothing existing was edited, `results/_cache` untouched)

| Path | What |
|---|---|
| `dh2/calib_pinn.py` | the method: box, Sobol starts, batched bounded Levenberg-Marquardt on the surrogate with autograd Jacobians, optional exact re-ranking / defect correction, exact `scipy` polish, instrumented exact baseline (`fit_day_exact`, same loop as `calib.fit_day`, counts pricer calls) |
| `exp_pinn_calib.py` | `design` (development days only), `test` (the 118 paired days, once), `report` (tables and figures). Progress file `logs/progress/pinncal_00.json` |
| `tests/test_calib_pinn.py` | 12 fast tests, all pass (about 5-9 s): residual agreement with the exact pricer, autograd Jacobian, scipy's soft-L1 cost, starts and candidates inside the surrogate box, final solution inside the calibrator's bounds, quotes beyond tau 0.6 y left out of the global stage only, **recovery of a known surface priced by the exact pricer in prices (IV RMSE < 0.15, max < 0.6 vol points; parameters not asserted)**, **polish never worsens the exact cost**, defect correction never worsens its start, instrumented baseline equals `calib.fit_day` |
| `results/pinn_calib/` | tables (CSV), figures (300 dpi PNG), `frozen_config.json`, `headline.json`, this file |

## 2. Method (and where it deviates from the brief's wording)

Objective, thinning (`M.thin(S, 14)`), residual `(model price - market price) / max(vega, 0.05)`, soft-L1 loss with `f_scale` 0.8, calibrator bounds with `kappa2 = kappa1 + dk, dk >= 0.2`: identical to `dh2.calib.fit_day`.

1. **Global stage (surrogate, CPU, 2 threads).** Scrambled-Sobol starts in the *intersection* of the calibrator's bounds and the surrogate's training box (`results/pinn/README.md` section 6; the box is much narrower than the calibrator's bounds for sigma1, rho1, theta2, sigma2, rho2). Quotes the surrogate cannot price (tau > 0.6 y) are excluded from this stage only: 9.6% of the fit quotes on average, present on 92% of the days. **Deviation:** the starts are not optimised one at a time by `scipy.least_squares`; all of them are optimised at once by my own batched, bounded Levenberg-Marquardt (same soft-L1 cost, scipy's robust-loss Gauss-Newton weights, same variable scaling), with the Jacobian from one autograd backward pass over (starts x quotes), and successive halving. Reason: a loop of scipy runs costs more than the exact calibration it is meant to replace.
2. **Re-rank and correct (my addition, chosen on development days).** The 8 survivors are ranked by their exact cost (8 exact residual evaluations) and refined by 2 rounds of additive defect correction (the surrogate residual is shifted to equal the exact residual at the current point, minimised in a trust region, accepted only if the exact cost falls). Both use exact residual values but never an exact Jacobian. See section 6: this addition did not demonstrably help.
3. **Polish (exact).** `scipy.least_squares` on the exact pricer from the best distinct candidate, identical objective, bounds and scaling as `calib.fit_day`, budget `max_nfev` = 10 (each Jacobian costs 10 more pricer calls; `n_exact_calls` counts every pricer call, finite-difference columns included).

**Frozen design** (`frozen_config.json`, chosen on 52 *development* days = every 25th day of 2016-2021, `design_*.csv`; asserted in code to be before 2022): 32 Sobol starts, batched LM 8 iterations for all, 20 more for the best 8; re-rank + 2 correction rounds; polish 1 candidate with 10 function evaluations. Rules fixed before the results were seen: the cheapest wall time whose mean log(cost / exact cost) is within 0.01 of the best in the grid (`design_global_configs.csv`: 6 start/iteration budgets; `design_variant_polish_grid.csv`: 3 variants x 1-3 candidates x polish budgets 5/10/20/40). On those development days the chosen setting had mean log cost ratio +0.007, mean IV RMSE 2.928 vs 2.948 for the exact calibration, 122 exact-pricer calls vs 1,586, 0.78 s vs 6.64 s. It was then run **once** on the test sample. (`fig6_design_development_days.png`.)

**Test sample.** The 118 days of `results/svi/sample_days.csv` (every 10th test-period day from 2022-01-03; checked equal in code). The exact baseline was **re-run for all 118 days** in this experiment (seed = day index, as in C2) and reproduces the C2 cache bit for bit (118 of 118 identical costs, `headline.json`: max difference 0.0).

## 3. Same-day implied-vol RMSE (vol points, exact pricer, all quotes of the day)

`accuracy_by_method.csv`, `accuracy_paired.csv`, `per_day.csv`, `fig1_rmse_paired.png`, `fig5_rmse_difference_over_time.png`. 95% intervals resample days; p = paired Wilcoxon signed-rank (`dh2.stats`).

| method | mean RMSE [95% CI] | median | paired difference to exact: mean [95% CI], Wilcoxon p |
|---|---|---|---|
| exact calibration (4 starts, 120 nfev) | 2.555 [2.312, 2.812] | 2.219 | - |
| **accelerated** (surrogate global + exact polish) | **2.560** [2.318, 2.819] | 2.229 | +0.0054 [-0.0091, +0.0205], p = 0.17 |
| surrogate-only (no polish) | 2.554 [2.311, 2.813] | 2.132 | -0.0006 [-0.0250, +0.0221], p = 0.70 |
| control: exact polish from the Phase-3 start(s), same polish budget, no surrogate | 2.837 [2.567, 3.132] | 2.452 | +0.282 [+0.211, +0.361], p = 5e-19 |
| accelerated without re-rank/correction (ablation) | 2.545 [2.302, 2.804] | 2.177 | accelerated minus this: +0.0154 [+0.0031, +0.0302], p = 0.18 |

Reading. The accelerated fit and the exact fit are indistinguishable in RMSE (median absolute day-by-day difference 0.003, 90th percentile 0.11; 52 days better, 66 worse, 7 days worse by more than 0.1, 6 days better by more than 0.1, worst +0.52 on 2023-03-17). The control shows what the surrogate stage buys: the same exact budget spent from one exact-pricer start is 0.28 vol points worse.

**The surrogate-only RMSE equals the exact one by coincidence of the metric, not because the surrogate fits as well.** The RMSE is not what is optimised. On the quotes that are fitted (vega-weighted residuals, plain RMS, `fit_quote_vs_all_quote_error.csv`) exact is 1.190, accelerated 1.193, surrogate-only 1.315 (+10%); on the objective itself (soft-L1 cost, `accuracy_by_method.csv`, `fig2_cost_ratio.png`):

| method | mean log(cost / exact cost) [95% CI] | days with cost within 1% of exact or better | days more than 5% worse |
|---|---|---|---|
| accelerated | +0.0079 [+0.0045, +0.0115] (= +0.8%), p = 7e-18 | 82.2% | 5.9% (7 days; worst +10.9% on 2024-02-09) |
| surrogate-only | +0.1664 [+0.1338, +0.2023] (= +18%) | 6.8% | 71.2% |
| control (exact polish, no surrogate) | +0.0851 | 25.4% | 46.6% |
| accelerated without re-rank/correction | +0.0095 | 77.1% | 6.8% |

So the surrogate's 0.05 vol-point error is not negligible *for an optimiser*: the minimiser of the surrogate sits in a flat valley of the objective where that error moves the optimum far. How much of the benefit survives the polish: all of the speed (next section) and almost all of the cost (+0.8% of the exact calibration's cost); the accelerated solution is not exactly the exact optimum (its cost is statistically higher, Wilcoxon p = 7e-18, but by a median of 0.08%). It is lower than the exact calibration's by more than 1% on 2 days (2024-11-04: -6.7%, the exact 4-start run is not a global optimiser either).

## 4. Time and exact-pricer evaluations

`timing_by_method.csv`, `speedup.csv`, `timing_by_load.csv`, `retimed_20_days.csv`, `fig3_time.png`. Single process, one day at a time, exact and accelerated fits run back to back with the order alternated by day; torch 2 threads.

| | exact | accelerated | control (exact polish only) |
|---|---|---|---|
| mean wall time per day [95% CI] | 5.85 s [5.39, 6.33] | 0.729 s [0.695, 0.764] | 0.397 s |
| median | 5.28 s | 0.685 s | 0.383 s |
| exact-pricer calls per day (mean) | 1,380 | 118.6 | 101.2 |

* **Speedup, total wall time 8.0x [7.4, 8.7]**; total CPU time 7.3x [6.8, 7.9]; median of the per-day speedups 6.7x [6.3, 7.6], mean 8.2x, smallest 4.0x, largest 23.5x; the accelerated fit was faster on every one of the 118 days. **Exact-pricer calls: 11.6x fewer [10.7, 12.6].**
* Where the accelerated time goes (mean per day): surrogate search 0.12 s, exact re-rank / correction 0.18 s, exact polish 0.43 s. The exact pricer is already fast (3.4 ms per surface evaluation, 38 microseconds per quote), which caps the speedup of anything that still calls it.
* **Load (what the machine was doing).** Re-timed exact fits: the C2 package recorded a median of 10.6 s (mean 33.7 s, two days of 1,104 and 1,283 s: measured with its own workers in parallel / under load), my re-timing gives a median of 5.28 s, so the recorded times are not used. During the first 6 test days (2022-01-03 .. 2022-03-16) another job's three worker processes were running (`ps`: 99-300% CPU, 1-minute load average 4.0-4.4); after that no other python process used more than 5% CPU for the remaining 112 days (load average 1.1-3.7, the decay of the earlier load plus my own process; mean over days 1.7). So most of the timings were taken on a nearly idle machine. Speedup by load bin (`timing_by_load.csv`): 8.3x on 94 days with load < 2, 6.6x on 16 days with load 2-3.5, 7.8x on 8 days with load >= 3.5: no systematic dependence. Re-timed subset of 20 days (every 6th test day, `retimed_20_days.csv`): C2 recorded median 7.96 s, re-timed exact median 4.55 s, accelerated 0.67 s, speedup 7.3x total (median 6.1x). Apple M5, 16 GB; the surrogate runs on the CPU.

## 5. Parameters: agreement and weak identifiability

`parameter_agreement.csv`, `parameter_distance.csv`, `weak_identifiability.csv`, `params_*.csv`, `fig4_parameters.png`.

* The accelerated and exact solutions are **the same parameter vector on only 37% of the days** (every parameter within 5% of its bound interval, C2's definition; 34% within 2%, 41% within 10%); median over days of the largest gap 20% of a bound interval; rms distance in the prior's units (log/0.7, rho/0.3) median 0.43, mean 0.72, 90th percentile 1.95. The surrogate-only solution is the same vector as the exact one on 2.5% of the days (median rms distance 1.31).
* Parameters that agree (median |difference|, log units for positive parameters): v0_1 0.005, v0_2 0.006, rho1 0.004, rho2 0.003, xi2 0 (xi2 sits at its upper bound 2.0 in 91.5% of the exact fits, 97.5% of the accelerated ones), xi1 0.03, theta2 0.03, dk 0.03. Parameters that do not: **kappa1 0.40 (a factor 1.5) and theta1 0.22** (90th percentile 1.9 and 1.3 log units). The two states and the correlations are pinned by the day's prices, the slow factor's speed and level are not.
* **Equal cost, different parameters (the weak-identifiability point).** Cost within 1% and the largest parameter gap above 5% of its bound interval: **51 of 118 days (43%)**; the two fitted implied-vol surfaces then differ by a median of 0.04 vol points (90th percentile 0.14). With cost within 0.1%: 17 days (14%), median surface gap 0.01 vol points. Full grid of tolerances in `weak_identifiability.csv`. Different parameters, same prices: the calibration cannot tell the solutions apart, which is also why the RMSE comparison above cannot.
* Exact fits with parameters at a bound: rho1 62% of days, dk 55%, kappa1 31%, theta2 23%, theta1 19%, xi2 92% (`parameter_agreement.csv`).

## 6. Where the accelerated path is worse, and what did not work

`days_where_accelerated_is_worse.csv` (30 days with RMSE worse by more than 0.02 or cost more than 1% higher; 21 days cost > 1% higher, 17 > 2%, 7 > 5%), `segments.csv`.

* **The exact optimum outside the surrogate's box.** 19 of 118 days (16%): rho1 = 0.98 (6 days; the box stops at 0.457), xi2 between 0.19 and 1.33 (8 days; box from 1.49), rho2 above -0.154 (8 days). The global stage cannot find these; the polish has 10 evaluations to travel. On those days the accelerated cost is more than 1% above the exact one on 47% of days (12% when the exact optimum is inside the box). The worst RMSE day (2023-03-17, +0.52 vol points) is one of them. In RMSE terms these days are not worse on average (-0.004), because RMSE is a different criterion.
* **Quotes beyond tau 0.6 y** are not seen by the global stage (a mean of 10.9 of 111 fit quotes, 9.6%, on 92% of the days); they enter at the correction/polish stage only. Days without such quotes are too few (9) to say whether this costs anything (accelerated cost on those 9 days: median ratio 1.000).
* **The polish does not converge.** The budget (10 evaluations) is exhausted on 115 of 118 days; the result is a point near, not at, the optimum (cost +0.8%). More budget buys accuracy at a measured price (`design_variant_polish_grid.csv`: 3 candidates x 40 evaluations reaches the exact cost on average but takes 3.3 s on development days).
* **More surrogate starts do not help** (`design_global_configs.csv`): 32, 64, 128, 256 starts and 60 instead of 20 iterations give the same surrogate-only cost to three decimals (mean log ratio 0.342-0.344) and polished costs within 0.006; the limit is the surrogate's error in the flat valleys, not the search. Starting the surrogate from the whole box instead of the day's variance range gave the same result within noise (polished mean log cost ratio 0.0089 vs 0.0113).
* **Re-ranking and defect correction did not demonstrably help.** On development days they cut the polish budget needed (cost +0.0070 after 10 evaluations vs +0.036 for plain polish), so the pre-registered rule chose them; on the test days the plain variant is as good or marginally better (mean RMSE 0.015 lower, 95% CI 0.003 to 0.030, but Wilcoxon p = 0.18; cost +0.95% vs +0.79%). The defect correction itself changes little on top of the re-ranking (`design_variant_polish_grid.csv`: it lowers the mean log cost by 0.004-0.009 when one candidate is polished and by 0.001 or less with three candidates and 10 or more evaluations). I kept the frozen design as it was chosen rather than revising it after seeing test results.
* **The surrogate alone is not a calibration** (section 3). Its 0.05 vol-point error looks harmless as a price error and is not harmless to an optimiser.
* The exact baseline itself is a 4-start local search; on 2 days the accelerated method found a lower cost. "Equal to the exact calibration" is therefore not "equal to the global optimum".

## 7. Limitations

* **Diffusion Double Heston only** (the surrogate has no jumps): this says nothing about the `dhj` model of the main study, whose same-day misfit is about 0.35-0.6 vol points against about 1.7 for the diffusion model (2025-26, `AUDIT.md` section 4; 2.55 on average over the 118 test days here).
* **Box.** The method works inside the surrogate's training box, which excludes the region where 16% of the exact daily solutions lie; extending it needs a retrained surrogate.
* **No-jump, independent daily fits are poorly identified** (section 5): the comparison is between two sets of equally good surfaces, not between a right and a wrong answer. Parameter agreement should not be read as accuracy.
* The timing is that of one machine mostly idle after the first 6 days; the speedup of 8x is for this exact pricer (about 3.4 ms per surface), a slower exact pricer would raise it, a faster one lower it. The surrogate global stage used the CPU; the GPU (79x for batches in the surrogate's own speed table) was not used.
* The design used 52 development days and rules fixed in advance; differences between configurations inside the 1% tolerance are noise. The surrogate was trained on development-period parameters (no test-period information), so development-day accuracy may be slightly optimistic compared with test days (test-day surrogate error 0.0485 vol points, `results/pinn/README.md`).
* 118 days, one seed for the Sobol starts (seed 0) and the exact baseline's seeds; the day-resampling intervals ignore day-to-day dependence (volatility regimes persist) and are probably a little narrow.

## 8. Files

Tables: `accuracy_by_method.csv`, `accuracy_paired.csv`, `timing_by_method.csv`, `speedup.csv`, `timing_by_load.csv`, `retimed_20_days.csv`, `parameter_agreement.csv`, `parameter_distance.csv`, `weak_identifiability.csv`, `segments.csv`, `days_where_accelerated_is_worse.csv`, `fit_quote_vs_all_quote_error.csv`, `per_day.csv` (118 rows, every metric per day and method), `params_exact.csv`, `params_accelerated.csv`, `params_surrogate_only.csv`, `params_exact_polish_from_phase3_starts.csv`, `design_global_configs.csv`, `design_variant_polish_grid.csv`, `design_per_day.csv`, `frozen_config.json`, `headline.json`. Cached per-day results: `_cache/` (resume; the report is computed from it).
Figures (300 dpi): `fig1_rmse_paired.png`, `fig2_cost_ratio.png`, `fig3_time.png`, `fig4_parameters.png`, `fig5_rmse_difference_over_time.png`, `fig6_design_development_days.png`.

Reproduce: `python3 exp_pinn_calib.py design && python3 exp_pinn_calib.py test && python3 exp_pinn_calib.py report` (about 30, 15 and 1 minutes; one process); `python3 -m pytest tests/test_calib_pinn.py -q`.
