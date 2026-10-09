# RESULTS: Double Heston 2.0 on NIFTY 50 options (technical report)

Branch `dh2-upgrade`. Written in plain English for a reader who is learning the subject; every term is defined where it first appears, and `docs/EXPLAINER.md` explains
the ideas at more length. Every number below is copied from `results/metrics.json`, a file in `results/tables/`, or a README under `results/`; the source path is in
square brackets after it (paths are relative to the repository root). The few numbers that come from configuration or code constants are listed in the notes at the end.
Not investment advice.

**Units.** Errors are in **volatility points** (vol points): one percentage point of implied volatility. "Test period" is 2022-2026; "development period" is 2016-2021.
Lower error is better. "Misfit" is yesterday's remaining market-minus-model error, carried to tomorrow as a correction.

---

## 1. Summary

The project fits the Double Heston option model to ten years of NIFTY 50 option prices, forecasts the next day's implied-volatility surface, and compares the result with simple
no-model forecasts. It also checks the pricing formula, builds a neural-network (PINN) copy of the pricer, and uses it to speed up calibration.

**Headline numbers**

1. Next-day forecast, test period (1,173 days): jump model + yesterday's misfit **1.034** vol points against **1.236** for the best no-model benchmark (the time-scaled smile); paired difference
   **-0.202 [-0.227, -0.181]**; better in 11 of 11 calendar years [results/tables/forecast_scores.csv, results/tables/forecast_by_year.csv; metrics.json `forecast.dhj_main.carry_res.*`].
2. From the 1.0 recipe to 2.0 (118 sampled test days): next-day error **5.03 to 1.10** vol points, same-day error 4.84 to 0.99 [results/attribution/waterfall.csv].
3. The project pricer agrees with an independent COS pricer to at most **1.5e-6 vol points** on 18,071 real-chain points [results/pricer_validation/errors.csv; results/pricer_validation/README.md].
4. PINN surrogate error against the exact pricer on real held-out chains: data-only network **0.049**, hybrid (data + physics) **0.077** vol points [results/pinn/tables/ablation_summary.csv].
5. PINN-accelerated calibration: **8.0x** [7.4, 8.7] less wall time, same-day error 2.560 against 2.555 for the exact calibration [results/pinn_calib/headline.json, results/pinn_calib/accuracy_by_method.csv].
6. Hidden-quote (cross-sectional) test, 118 days: jump model **0.71** versus diffusion Double Heston **2.49** vol points [results/crosssection/summary.csv].
7. Link to India VIX: correlation of levels **0.989** over 2,430 days; of daily changes 0.879 (all years) and 0.948 (test years) for the jump model [results/vix/vix_vs_model_stats.csv].
8. Identifiability: on **every one** of the 118 sampled days at least one parameter sits on a bound [results/identifiability/summary.csv].

**What did not work or is not shown (stated plainly)**

- (a) **The shared-slow-parameter ("structure/state") design is not shown to help.** The waterfall shows the gain comes from the IV-space objective, the global multi-start with the kappa ordering, and Merton jumps. Adding the
  structure/state walk-forward made the diffusion model slightly *worse* (+0.08 vol points next day, p = 0.003). Independent daily jump fits forecast better (1.142) than the walk-forward jump model (1.254) on the same days.
  Yesterday's misfit adds a further small gain (-0.155) [results/attribution/waterfall.csv; DECISIONS.md decision 11].
- (b) **DH1 is a documented rebuild** of a lost script: close in aggregate (median held-out error 2.53 against 1.0's published 2.37) but not an exact reproduction [results/attribution/README.md].
- (c) **The PINN:** the plain data-trained network beats the physics-informed network on price accuracy; the hybrid is better on gamma, PDE residual and strike convexity; physics-only does not work as a pricer;
  the surrogate is valid only inside its training box and covers the diffusion model only [results/pinn/README.md; DECISIONS.md decisions 8-9].
- (d) **PINN-accelerated calibration** is about 8x faster with the same implied-vol error, but its objective cost is about 0.8% higher (worst day 10.9%); the surrogate alone is not enough and the exact polish is essential;
  it is weaker on days whose exact optimum lies outside the box [results/pinn_calib/README.md].
- (e) **Near the money the time-scaled smile beats the jump model + misfit** (1.071 against 1.088, a significant difference); the model's advantage is in the wings and at short expiries [results/stats/tests_test_period.md].
- (f) **The parameters are weakly identified:** interpret fitted surfaces, not individual parameters [results/identifiability/README.md].
- (g) **Trading test:** 41 to 45 of 100 trades profit; wins are bigger than losses; the edge is **not shown to survive a one-day delay**. Not investment advice [results/tables/trading_summary.md].
- (h) The project pricer is accurate (1.5e-6 vol points on real chains) with **one documented corner case** (1-year options, extreme parameters, up to 1.0e-4 vol points) [results/pricer_validation/README.md].
- (i) **The diffusion Double Heston loses to the time-scaled smile as a forecast** (1.377 against 1.236, with misfit; 2.792 without), and, in the maturity-extrapolation test, to plain Heston (2.02 against 1.37). The jump model is the strong one
  [results/tables/forecast_scores.csv; results/crosssection/summary.csv].

Figure for this section: `results/figures/01_summary_panel.png`.

---

## 2. Data

**Source and period.** NSE India's daily F&O files ("bhavcopy"), 2,470 files from 2016-10-03 to 2026-10-06; an old format through May 2024 and the UDiFF format after [results/dq/README.md; AUDIT.md section 5].
Only NIFTY index options are used; the files also hold BANKNIFTY from 2024 [results/dq/README.md, "Things to know"]. NSE's closing price is reported even for strikes that did not trade, so untraded rows carry stale prices.

**Per-day surface.** For each day the code builds a *surface*: all out-of-the-money quotes with strike, expiry, call/put, close, implied volatility (by bisection), and vega. Each expiry gets a forward price from **put-call parity**
(the six strikes nearest the money), and a financing rate is implied from the options and smoothed with a trailing 10-day median so that it only uses past days [dh2/data.py docstring].
A forward-based design means spot, rate and dividend enter only through the forward and the discount factor [AUDIT.md section 3].

**Quote funnel, 2016-2026** [results/dq/README.md; results/tables/data_quality_funnel.md]:

| Filter | Quotes | Share of all NIFTY rows |
|---|---|---|
| NIFTY option rows in the files | 5,254,005 | 100.0% |
| traded (volume > 0) | 1,517,128 | 28.9% |
| expiry at least 3 days away | 1,342,577 | 25.6% |
| put-call parity gives a forward | 1,299,272 | 24.7% |
| out of the money | 794,565 | 15.1% |
| price at least Rs 1 | 766,943 | 14.6% |
| absolute ln(K/F) at most 0.22 | 736,313 | 14.0% |
| implied vol solves = final quotes | 736,313 | 14.0% |

All 2,470 days give a usable surface (100%), with 298 quotes a day on average (median 311) in 6.9 expiries (2 to 11). The quotes per day grow from 94.5 in 2016 to 550.9 in 2026, so the early years are thin
[results/dq/README.md].

**No-arbitrage diagnostics, not filters.** With a tolerance of one tick (Rs 0.05), 64.1% of quotes take part in at least one violation (monotonicity 14.81%, butterfly convexity 62.68%, calendar 1.64%); with an extra 0.25 vol point of slack for
closing-print noise, 12.0% do [results/dq/NOARB.md]. Removing them would change every surface and invalidate about 22 hours of finished runs, so they are measured and the forecasts are rescored without them [DECISIONS.md decision 4].
Across all of these exclusion variants the jump model's advantage over the time-scaled smile stays between -0.166 and -0.224 vol points (headline -0.202) [results/dq/NOARB.md; results/tables/noarb_sensitivity.md].

**Split: development 2016-2021, test 2022-2026.** 1,256 development days and 1,173 test days are scored [results/tables/forecast_scores.csv]. The split was fixed before any result existed; all settings were chosen on 2016-2021 and 2022-2026 is
reported [DECISIONS.md decision 1; README.md]. This is more conservative than a 70/30 split of a short history: the test period spans several volatility regimes and is long, and moving the line after seeing results would be a tuning choice.
The walk-forward needs 40 burn-in days, so 2,430 days carry a calibration [results/vix/README.md, 2,430 days] and 2,429 are scored as forecasts [COMPARISON_1.0_vs_2.0.md section 2].
Test years are calmer than the development years for every method (time-scaled smile: 1.236 test, 1.624 development) [results/tables/forecast_scores.csv], so levels of error are not comparable across the two periods; the paired differences are what matter.

Figures: `results/figures/07_quote_counts_by_bucket.png`. Tables: `results/tables/data_quality_funnel.md`.

---

## 3. Model specification

**Double Heston** (the brief's model): two independent variance factors, each correlated with the index [AUDIT.md section 2]:

```
dS/S   = (r - q) dt + sqrt(v1) dW1 + sqrt(v2) dW2
dv_j   = kappa_j (theta_j - v_j) dt + sigma_j sqrt(v_j) dZ_j        j = 1, 2,   corr(dW_j, dZ_j) = rho_j
```

Ten parameters in the pricer's order `(v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2)`; `xi` is the brief's `sigma` [dh2/models.py]. Factor 1 is slow, factor 2 fast. In the fit, kappa2 = kappa1 + dk with dk >= 0.2, so the two
factors cannot swap labels [dh2/calib.py]. The log characteristic function (the Fourier transform of the return distribution, the "fingerprint" that a Fourier pricer uses) is the sum of the two factors' logs, in the "little trap" form of
Albrecher et al. (2007), which avoids a complex-logarithm branch problem [AUDIT.md section 3]. The Feller condition (2 kappa theta > xi squared) is **not imposed** in 2.0; the pricer does not need it, and the calibrated fast factor is
non-Feller [dh2/calib.py; results/pricer_validation/README.md, "Two things found in my own COS"].

**Jumps (the extension, the headline model).** A Merton jump component adds `lam` jumps a year with normal log-size (mean `mu_j`, spread `delta_j`), compensated so the forward is unchanged; it is an independent factor, so its log
characteristic function simply adds to the diffusion's [dh2/models.py]. Thirteen parameters in all. The brief's PDE has no jump term, so the PINN covers the diffusion model only [DECISIONS.md decision 3]; the diffusion model is reported
next to the jump model throughout. Parameter bounds are in `configs/nifty.yaml`.

---

## 4. Exact pricer and its validation (COS, Lewis, Monte Carlo)

Everything (calibration, PINN training data) rests on `pricer/models.py`, a midpoint-rule integral of the characteristic function whose accuracy had never been checked independently. Three independent pricers were built
[results/pricer_validation/README.md]:

- **COS** (Fang and Oosterlee 2008): a cosine-series expansion of the density with analytic payoff coefficients, own characteristic function from the Riccati equations (never imports the project's), adaptive number of terms and adaptive truncation width.
- **Lewis** (2001): a different single integral, with a Black-Scholes control variate.
- **Monte Carlo**: Andersen's (2008) QE scheme for both variance factors, antithetic paths, no characteristic function at all.

**Results** [results/pricer_validation/errors.csv; results/tables/pricer_validation.md]:

| Points | Comparison | n | Max IV error (vol points) |
|---|---|---|---|
| real chains (calibrated parameters of 62 sampled days), diffusion | project vs COS | 18,071 | 1.45e-06 |
| real chains, jumps | project vs COS | 18,071 | 6.03e-07 |
| synthetic grid, diffusion | project vs COS | 8,736 | 1.04e-04 |
| synthetic grid, jumps | project vs COS | 8,736 | 1.01e-04 |
| real chains / synthetic grid | Lewis vs COS | 18,071 / 8,736 | 7.57e-10 / 1.51e-09 (diffusion) |
| Monte Carlo (10 parameter sets x 7 strikes, 400,000 paths each) | Monte Carlo vs COS | 70 | max abs z = 1.98, mean abs z = 0.60 standard errors |

The characteristic function agrees with a 30-digit mpmath solution of the ODE at 540 points to at most 5e-14 in the log. COS and Lewis, two completely different inversions, agree to 1.3e-9 relative [results/pricer_validation/README.md].

**The one disagreement (documented corner case).** For 1-year options with extreme parameters (the March-April 2020 fits) the project pricer is off by up to Rs 0.0076, 1.0e-4 vol points; all 40 worst points are two 2020 parameter sets at 365 days.
Cause (shown, not guessed): the fixed integration spacing 0.5 aliases the density when the return distribution is very wide; halving the spacing removes it. It is negligible against calibration residuals of 0.3-1 vol point and is encoded as two
strict expected-failure tests; the pricer is left unchanged by design [results/pricer_validation/README.md].

**Speed.** On one real chain (2026-10-01, 659 quotes, 9 expiries) the project pricer costs about 26 microseconds per option (diffusion), COS with a checked width 156, COS with a fixed width 52, and the Lewis integral about 6,000
[results/pricer_validation/timing.json; metrics.json `pricer.timing.*`]. COS is valued for independence, not speed.

Figures: `results/figures/17_pricer_validation.png`.

---

## 5. Calibration method, and what fixed DH1

**Method** [dh2/calib.py; dh2/struct.py; configs/nifty.yaml]:

- **Residual** for each quote: (model price - market price) / vega, which is the implied-volatility error in vol points to first order. Vega is floored at 0.05 rupees per vol point so near-worthless options carry little weight [DECISIONS.md decision 10].
- **Robust loss:** soft-L1 with scale 0.8 vol points, because some closing prints are stale.
- **Bounded multi-start** `scipy.optimize.least_squares`: a flat-volatility corner plus random starts (4 in the daily fit), with the kappa2 = kappa1 + dk ordering, warm-started where possible.
- **Walk-forward:** structural parameters over a trailing 40-day window refitted every 10 days; today's two variance levels fitted each day (the "structure/state split"; see (a) above for what the evidence says about it).
- **Forecast variants** [hybrids.py]: "carry" uses today's calibrated model on tomorrow's surface; "carry_res" ("+ misfit") adds yesterday's market-minus-model error, carried in log-moneyness.

**What fixed DH1: the attribution waterfall.** DH1 is the 1.0 research project's calibration style (price-space objective, independent daily fits from a latent coordinate system, five fixed starts), rebuilt on the same clean data. Each step adds one change; all 118 sampled test days
(every 10th test trading day) are paired across steps; intervals are 95% bootstrap over days; p-values are paired Wilcoxon against the previous step [results/attribution/README.md; results/attribution/waterfall.csv].

| Step | What changes | Same-day mean | Next-day mean [95%] | Next-day change (p) |
|---|---|---|---|---|
| 0 DH1 (1.0 recipe) | 1.0 data, price objective, 5-start optimiser | 4.839 | 5.027 [4.680, 5.393] | |
| 1 + clean data | traded OTM quotes, parity forwards, price >= Rs 1, >= 3 days | 4.705 | 4.929 [4.565, 5.317] | -0.098 (p = 0.26) |
| 2 + IV-space vega-weighted objective | soft-L1 on IV residuals | 3.589 | 3.881 [3.550, 4.244] | -1.048 (p = 4e-21) |
| 3 + global multi-start, kappa2 > kappa1 | `dh2.calib.fit_day` | 2.555 | 2.838 [2.531, 3.181] | -1.043 (p = 7e-21) |
| 4 + structure/state split, walk-forward | 40-day window, refit every 10 days | 2.683 | 2.917 [2.596, 3.277] | **+0.078 (p = 0.003)** |
| 5 + Merton jumps | jump parameters in the structure | 0.994 | 1.254 [1.106, 1.415] | -1.662 (p = 8e-21) |
| 6 + yesterday's misfit carried | next-day only | 0.994 | 1.100 [0.965, 1.246] | -0.155 (p = 4e-08) |

Reference rows on the same days: yesterday's smile 1.585, time-scaled smile 1.333, flat Black-Scholes refitted daily 5.970, single Heston with independent daily fits 3.585, jump model with independent daily fits 1.142 [results/attribution/waterfall.csv].

**Reading it honestly.**

- The gains are the IV-space objective (-1.05), the global multi-start with the kappa ordering (-1.04) and the jumps (-1.66), then the misfit (-0.155). The data cleaning is not significant on its own (p = 0.26).
- Step 4 (the structure/state walk-forward) makes the diffusion model slightly **worse**, and independent jump fits (1.142) forecast better than the walk-forward jump model (1.254). The shared-slow-parameter design is therefore **not** shown to help; the earlier claim is withdrawn
  [DECISIONS.md decision 11]. It is kept only because all the finished walk-forward results were produced with it. Step 4 also bundles the split and the walk-forward refits, which no saved result can separate; step 2 bundles the loss space, the weights and the robust loss; step 3 bundles
  multi-start, bounds and the ordering [results/attribution/README.md, "Caveats"]. (The 1.0 latent coordinates also imposed the Feller condition, which `fit_day` does not [results/attribution/README.md, DH1 recipe; dh2/calib.py]; no saved result separates that effect either.)
- **DH1 is a rebuild, not an exact reproduction.** The script behind 1.0's published NIFTY table was lost. The rebuilt recipe reproduces 1.0's data universe exactly (same quote and expiry counts on all ten dates of April 2026) but not its held-out set (170-193 quotes per date against 133-142) or its Black-Scholes
  sigma (10% higher). Its median held-out error is 2.53 vol points against 1.0's published 2.37 (+7%; per-date correlation 0.74) [results/attribution/README.md; results/attribution/replication_check.csv]. On 1.0's own metric (held-out third, expiries of 20 to 800 days) the same rebuilt recipe scores 2.27 on the 118 days;
  step 0's 4.84 is higher because the waterfall scores every quote of the clean surface, including 3 to 19 day expiries and the far wings that a price-space fit without jumps fits worst [results/attribution/README.md].

Figures: `results/figures/04_waterfall_dh1_to_dh2.png`. Tables: `results/tables/waterfall.md`.

---

## 6. Results on the test period

### 6.1 Model comparison

Mean next-day IV error over the 1,173 test days (vol points), 95% moving-block bootstrap intervals (blocks of 20 days, 2,000 resamples), difference against the best benchmark (time-scaled smile) [results/tables/forecast_scores.csv; results/report.md section 1]:

| Forecast | Development | Test | 95% interval of the test mean | Difference vs time-scaled smile |
|---|---|---|---|---|
| yesterday's smile (no model) | 1.876 | 1.468 | [1.335, 1.614] | |
| time-scaled smile (no model) | 1.624 | 1.236 | [1.112, 1.380] | |
| Heston | 4.028 | 3.532 | [3.306, 3.762] | +2.296 [+2.127, +2.480] |
| Heston + misfit | 1.776 | 1.416 | [1.286, 1.560] | +0.181 [+0.160, +0.203] |
| diffusion Double Heston | 3.409 | 2.792 | [2.574, 3.018] | +1.557 [+1.419, +1.694] |
| diffusion Double Heston + misfit | 1.736 | 1.377 | [1.245, 1.524] | +0.141 [+0.119, +0.164] |
| jump model | 1.734 | 1.201 | [1.053, 1.372] | -0.035 [-0.095, +0.028] |
| **jump model + misfit** | 1.317 | **1.034** | [0.915, 1.171] | **-0.202 [-0.227, -0.181]** |

The jump model alone ties the benchmark (its interval includes zero); only the jump model plus misfit is clearly better. The diffusion models, with or without the misfit, are worse than the time-scaled smile. The jump model + misfit beats the time-scaled smile in 11 of
11 calendar years (2016 and 2026 are partial years) [results/tables/forecast_by_year.csv]:

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| time-scaled smile | 0.802 | 0.855 | 1.049 | 1.634 | 3.015 | 1.620 | 1.600 | 1.073 | 1.324 | 0.998 | 1.163 |
| jump model + misfit | 0.707 | 0.609 | 0.850 | 1.198 | 2.695 | 1.266 | 1.372 | 0.808 | 1.125 | 0.815 | 1.050 |

**Robustness.** Fifteen reruns of the jump model with different windows (20 to 120 days), refit schedules, prior strengths, a pattern-memory prior, wide bounds and a daily jump rate score between 1.033 and 1.036 for the final forecast on the test period, while the pure model forecast ranges from
1.185 to 1.436 [results/tables/robustness_runs.csv; results/tables/robustness_memory.md]. Pattern memory as a forecast correction changes the final forecast by -0.004 [-0.012, +0.003] (not significant) and the time-scaled smile by -0.029 [-0.043, -0.018]; as a prior it changes nothing
[results/tables/robustness_memory.md; COMPARISON_1.0_vs_2.0.md]. Some memory and jump settings landed at the edge of the grid searched, so a wider search could move them [COMPARISON_1.0_vs_2.0.md section 5].

### 6.2 Paired tests (Wilcoxon and Diebold-Mariano)

Test period, 1,173 days, the daily next-day IV RMSE as the loss; Holm adjustment over the 10 pairs; Diebold-Mariano with Newey-West variance (6 lags) and the Harvey-Leybourne-Newbold correction [results/stats/tests_test_period.md; results/tables/significance_tests.md]:

| A | B | A - B (vol points) | Days A better | Diebold-Mariano stat | Verdict (Holm 5%) |
|---|---|---|---|---|---|
| jump model + misfit | time-scaled smile | -0.202 | 90% | -19.03 | A better; p < 1e-10 (both tests) |
| jump model + misfit | yesterday's smile | -0.434 | 78% | -25.59 | A better |
| jump model + misfit | diffusion DH + misfit | -0.343 | 77% | -21.96 | A better |
| jump model + misfit | Heston + misfit | -0.383 | 78% | -24.86 | A better |
| diffusion DH + misfit | time-scaled smile | +0.141 | 41% | +14.40 | B (the benchmark) better |
| Heston + misfit | time-scaled smile | +0.181 | 34% | +18.87 | B better |

**Near the money (absolute ln(K/F) at most 0.04):** jump model + misfit 1.088 against the time-scaled smile 1.071, difference +0.017, model better on only 47% of days, Wilcoxon (Holm) p = 2.85e-07, Diebold-Mariano stat +9.41: **the benchmark is better**.
The model's advantage is therefore in the wings and at short expiries, not at the money [results/stats/tests_test_period.md; COMPARISON_1.0_vs_2.0.md section 3].

### 6.3 Where the models miss (per bucket)

Same-day implied-vol RMSE pooled over the 118 sampled days (53,046 quotes): one volatility 6.13, Heston 3.67, Double Heston 2.91, with jumps 0.77; the walk-forward versions score 3.07 (diffusion) and 1.15 (jumps) [results/buckets/bucket_summary.csv; results/buckets/README.md].
The error lives in the short-dated wings: in the 3-7 day, deep-put bucket (ln(K/F) from -22% to -12%) the errors are 29.2 (one vol), 22.1 (Heston), 19.1 (Double Heston) and 1.48 (with jumps); the walk-forward jump model scores 3.65 there. For expiries beyond 120 days all smile-model
buckets are between 0.43 and 1.00 [results/buckets/README.md]. Quote counts per bucket are in figure 07 (the +12 to +22%, 3-7 day cell has 29 quotes and is greyed out in figure 06).

### 6.4 Cross-sectional out-of-sample tests

Independent per-day fits (four starts), 118 days [results/crosssection/summary.csv; results/crosssection/README.md]:

| Test | Black-Scholes | Heston | Double Heston | Double Heston + jumps |
|---|---|---|---|---|
| random 20% of quotes hidden (interpolation) | 5.787 [5.469, 6.123] | 3.245 [2.980, 3.523] | 2.486 [2.242, 2.743] | **0.705** [0.639, 0.780] |
| fit expiries <= 60 days, score longer ones (extrapolation in maturity) | 2.674 [2.526, 2.843] | **1.368** [1.210, 1.572] | 2.022 [1.804, 2.268] | 1.410 [1.239, 1.619] |

Held-out error is within about 0.1 of the training-quote error for every model, so there is no sign of overfitting to the fitted quotes. In the maturity test the diffusion Double Heston is **worse** than single Heston (+0.65, better on only 33 of 118 days, p = 4.1e-08); a plausible reason, not tested
separately, is that ten parameters fitted to short expiries extrapolate worse than five. Single Heston and the jump model are level (+0.04 [-0.08, +0.16], p = 0.73). The two settings score different quotes, so errors are comparable across models within a setting, not across settings.

Figures: `results/figures/03_model_comparison.png`, `results/figures/02_error_by_year.png`, `results/figures/05_market_vs_model_smiles.png`, `results/figures/06_where_models_miss.png`, `results/figures/08_calibrated_volatility_and_jumps.png`, `results/figures/18_crosssection.png`.

---

## 7. Identifiability

118 sampled days, 16 starts per day, diffusion (10 parameters) and jump (13 parameters) models [results/identifiability/README.md; results/tables/identifiability.md]:

| | Double Heston | Double Heston + jumps |
|---|---|---|
| median condition number (columns normalised) | 76 | 529 |
| median effective rank | 9 | 11 |
| share of parameters within 2% of a bound | 41.0% | 38.7% |
| share of days with at least one parameter on a bound (within 0.3%) | 100% | 100% |
| median standard error / bound width | 0.213 | 0.124 |
| mean distinct near-optimal parameter sets (16 starts) | 1.27 | 1.50 |
| days with 2 or more distinct near-optimal sets | 16.1% | 29.7% |

**Plain reading.** Locally the problem is not singular (no day above a column-normalised condition number of 1,338 for the diffusion model and 6,681 with jumps). But the data pin most parameters only loosely: the standard error exceeds the entire allowed interval on at least a quarter of days for kappa1, theta1, rho1 and
dk (diffusion). Today's variance levels (v0) are tight (standard error about 1.4% of the interval). The cost surface is not a many-minima problem here (on average 12.2 of 16 starts end within 1% of the best cost); what remains is ridges and bounds. This is far fewer than the 99.1% of surfaces with several equally good
parameter sets that the 1.0 study reported; the definitions differ (1.0: a price-cost tolerance and no ordering constraint). **Fit quality and identification are separate questions: even the jump model, which fits closely, has loosely determined structural parameters. Interpret fitted surfaces, not individual parameters.**

The PINN-accelerated calibration study shows the same thing from another side: on 51 of 118 days the accelerated and exact fits reach a cost within 1% of each other with parameters that differ by more than 5% of their range; the two fitted surfaces then differ by a median of 0.04 vol points (90th percentile 0.14)
[results/pinn_calib/README.md].

Figures: `results/figures/19_identifiability.png`.

---

## 8. The PINN surrogate

A **physics-informed neural network** (PINN; Raissi, Perdikaris and Karniadakis 2019) is a network trained to match examples and to obey a pricing equation. Here the target is the diffusion Double Heston price in normalised coordinates, validated against the exact Fourier pricer
[results/pinn/README.md]. It *approximates* the exact pricer; its value is speed when many prices are needed, smooth automatic-differentiation Greeks, and a measurable physics (PDE) consistency.

**Design.** u = C / (K x discount factor) is written u_BS(x, w) + g x NN(x, v1, v2, tau, Theta), where x = ln(F/K), tau is time to expiry, the gate g vanishes at tau = 0 so the starting condition (the payoff) is exact by construction. Network: width 256, depth 5, SiLU, 267,521 weights; Adam for
30,000 steps, batch 2,048; 1,000,000 exact labels (125,000 parameter/state/maturity combinations x 8 strikes) from the project pricer. Deliberate deviation from the brief: the gate is the Black vega plus a fat tail instead of sqrt(tau): 0.053 against 0.242 vol points on the monitoring set, same steps. The domain
(a box: tau from 1 day to 0.6 years, absolute ln(F/K) at most 0.30, variance levels 0.0016 to 0.36, and restricted parameter ranges) was built from development-period fits only; the last 20% of development days (251 days) never enter training and serve as real held-out chains [results/pinn/README.md, section 3].

**PDE verification.** The brief's PDE was verified before any training: seven-point finite differences of exact prices from the project pricer satisfy it to a relative residual that falls about 64 times per halving of the step (median 5.7e-5, 1e-6, below 1e-6 at step fractions 0.25, 0.125, 0.0625); with the cross term's sign flipped the same check is off by 0.199 (median), so a sign
error would have been caught. No change to the equation was needed [results/pinn/README.md, section 2].

**Validation and ablation** (implied-vol error against the exact pricer, vol points, held-out) [results/pinn/tables/ablation_summary.csv; results/tables/pinn_accuracy.md]:

| Network | Real held-out mean | median | 95th pct | max | Box-set mean | PDE residual median (real) |
|---|---|---|---|---|---|---|
| data-only (shipped for calibration) | **0.049** | 0.039 | 0.130 | 0.444 | 0.082 | 0.733 |
| hybrid data + PDE (shipped for Greeks) | 0.077 | 0.064 | 0.196 | 0.651 | 0.141 | 0.362 |
| data-only, brief's gate sqrt(tau) | 0.172 | 0.129 | 0.478 | 2.163 | 0.324 | 1.801 |
| physics-only (no labels) | 1.307 | 0.969 | 3.207 | 23.786 | 4.511 | 0.260 |

(Real held-out chains: 65,465 quotes of the 251 held-out development days; box set: unseen parameter sets from the training box; three seeds averaged for the first two rows.)

- **The plain data-trained network beats the physics-informed network on price accuracy.** The PDE term did not improve data efficiency in this budget: with 1,000 exact prices both are about 0.38-0.40; at 1,000,000 data-only reaches 0.049 and the hybrid 0.077 [results/pinn/tables/data_efficiency.csv].
- **The hybrid is better where the physics matters:** median gamma error 1.8-2.4% of the Black gamma against 2.5-6.0% for data-only; PDE residual median 0.36 against 0.73; butterfly-convexity violations at tolerance 1e-6 in 0.027% / 0.008% / 0.009% of triples (hybrid, three seeds) against 0.24% / 1.04% / 0.24% (data-only). The exact pricer has zero violations. The surrogate is therefore not arbitrage-free by construction
  [results/pinn/README.md, section 4].
- **Physics-only does not work as a pricer:** the lowest PDE residual of all (0.26) and a 1.3 vol point price error. A small normalised PDE residual does not pin down the price [results/pinn/README.md, section 5].
- Decision taken as measured, not chosen in advance: the data-only network (`final.pt`) is used for calibration speed and the hybrid (`final_greeks.pt`) for Greeks [DECISIONS.md decision 8].

**Speed** (one real chain of 283 quotes in 8 expiries; the exact pricer is already 38 microseconds per quote) [results/pinn/tables/speed.csv; results/tables/pinn_speed.md]:

| Task | Exact pricer | Surrogate CPU | Surrogate GPU (MPS) |
|---|---|---|---|
| one chain, prices | 10.9 ms | 1.1 ms (10.0x) | 3.3 ms |
| 128 parameter sets x chain | 1,555.5 ms | 75.1 ms (21x) | 19.6 ms (79x) |
| Greeks of one chain | 119.6 ms (finite differences) | 21.8 ms (5.5x, automatic differentiation) | |

**Greeks.** Median delta error 0.0008 to 0.0012 (data-only) and 0.0011 to 0.0015 (hybrid) over three seeds each; the second derivative (gamma) is clearly better and more reproducible in the hybrid [results/pinn/README.md, section 4].

**Limits.** The surrogate is valid only inside its box (2.3% of real quotes have tau above 0.6 years and need the exact pricer); on 1.9% of the 1,174 calibrated test days rho2 is slightly above the box; it covers the diffusion model only and cannot price the jump model; its tails reach 1.6 to 3
vol points at box corners; the second derivatives of the data-only network are noisy; the 30,000-step training budget was stopped early, so all errors were still falling and longer training might change the comparison (not tested); one seed only for physics-only, the spec-gate variant and the data-efficiency points
below 1e6 [results/pinn/README.md, section 6].

Figures: `results/figures/20_pinn_ablation.png`, `21_pinn_data_efficiency.png`, `22_pinn_error_maps.png`, `23_pinn_pde_residual_maps.png`, `24_pinn_speed.png`, `25_pinn_greeks.png`, `26_pinn_smile_example.png`, `27_pinn_training_curves.png` (all in `results/figures/`).

---

## 9. PINN-accelerated calibration

**Method** [results/pinn_calib/README.md]: the surrogate does the global search (scrambled-Sobol starts inside the intersection of the calibrator's bounds and the surrogate box, optimised in one batch by a bounded Levenberg-Marquardt with automatic-differentiation Jacobians); the best candidates are re-ranked
with a few exact residual evaluations; then the exact pricer polishes with a small budget (10 evaluations). The objective, bounds and scaling are identical to `dh2.calib.fit_day`. The design was chosen on 52 development days (every 25th day of 2016-2021) with rules fixed in advance, frozen, and run once on the 118 paired test days;
the exact baseline was re-run on all 118 days and reproduces the earlier cache bit for bit.

**Same-day IV error, scored with the exact pricer** [results/pinn_calib/accuracy_by_method.csv; results/tables/pinn_calibration.md]:

| Method | Mean RMSE [95%] | Mean log(cost / exact cost) | Days within 1% of exact cost |
|---|---|---|---|
| exact (4 starts) | 2.555 [2.312, 2.812] | 0.000 | 100% |
| **accelerated** (surrogate global + exact polish) | 2.560 [2.318, 2.819] | +0.008 (about +0.8%) | 82% |
| surrogate-only (no polish) | 2.554 [2.311, 2.813] | +0.166 (about +18%) | 7% |
| control: exact polish from the Phase-3 starts, no surrogate | 2.837 [2.567, 3.132] | +0.085 | 25% |

Paired RMSE difference, accelerated minus exact: +0.0054 [-0.0091, +0.0205] (Wilcoxon p = 0.17). **Speed:** wall time per day 5.85 s against 0.73 s, a total speedup of **8.0x [7.4, 8.7]**, and 11.6x [10.7, 12.6] fewer exact-pricer calls (1,380 against 119 per day); the accelerated fit was faster on every one of the 118 days
(smallest 4.0x, largest 23.5x) [results/pinn_calib/headline.json; results/pinn_calib/README.md].

**Honest reading (d).**

- The same RMSE hides a slightly worse objective: the accelerated cost is 0.8% above the exact one on average (statistically higher, Wilcoxon p = 7e-18, by a median of 0.08%), and on one day 10.9% higher (2024-02-09). The polish budget is exhausted on 115 of 118 days, so the result is near, not at, the optimum.
- **The surrogate alone is not enough:** its RMSE equals the exact one only by coincidence of the metric (its cost is 18% above, only 7% of days within 1%, and its parameters match the exact ones on only 2.5% of days). The surrogate's 0.05 vol point error is harmless as a price error and not harmless to an optimiser in flat valleys. The exact polish is essential.
- **It is weaker on days whose exact optimum lies outside the surrogate's box:** 19 of 118 days (16%); on these days the accelerated cost is more than 1% above the exact one on 47% of days (12% when the optimum is inside the box). The worst RMSE day, 2023-03-17 (+0.52 vol points), is one of them.
- The accelerated and exact solutions are the same parameter vector on only 37% of days (kappa1 and theta1 differ most), because the problem is weakly identified (Section 7); parameter agreement should not be read as accuracy.
- Re-ranking and defect correction did not demonstrably help on the test days (the plain variant was as good), but the frozen design was kept as chosen.
- Limits: diffusion model only; 118 days, one seed; the exact baseline is itself a four-start local search (on 2 days the accelerated method found a lower cost); the 8x applies to this pricer (3.4 ms per surface evaluation), and a slower exact pricer would raise it [results/pinn_calib/README.md, section 7].

Figures: `results/figures/28_pinn_calibration_rmse.png`, `29_pinn_calibration_cost_ratio.png`, `30_pinn_calibration_time.png`, `31_pinn_calibration_parameters.png`, `32_pinn_calibration_rmse_over_time.png`, `33_pinn_calibration_design.png`.

---

## 10. Link to India VIX

India VIX is the exchange's model-free measure of 30-day expected volatility (taken from NSE's own index-close archive, 2,470 days). The model-implied 30-day volatility is the square root of the model's risk-neutral expected average variance over the next 30/365 years, from each day's calibrated parameters
(jump model: plus the jumps' variance) [results/vix/README.md].

| Series | Period | Days | Correlation of levels | Correlation of daily changes | Mean gap, model minus VIX (pts) | RMSE of gap (pts) |
|---|---|---|---|---|---|---|
| jump model | 2016-2026 | 2,430 | 0.989 | 0.879 | +0.44 | 1.07 |
| jump model | 2022-2026 test | 1,174 | 0.984 | 0.948 | +0.63 | 0.93 |
| diffusion Double Heston | 2016-2026 | 2,430 | 0.989 | 0.874 | -0.00 | 0.97 |
| diffusion Double Heston | 2022-2026 test | 1,174 | 0.984 | 0.949 | +0.13 | 0.72 |
| surface at-the-money 30-day IV | 2016-2026 | 2,415 | 0.985 | 0.831 | -1.24 | 1.63 |

[results/vix/vix_vs_model_stats.csv; results/tables/vix_link.md]. The regression of VIX on the jump model has slope 1.01 and R-squared 0.98 in levels (slope 0.86, R-squared 0.77 in daily changes). The lead-lag correlation peaks at lag 0 (about 0.88) and is at most about 0.11 elsewhere: the model does not lead or lag VIX
[results/vix/README.md]. On 16 days (2020-03-13 to 2020-04-08) at least one variance level sits on its upper bound, so the model-implied volatility is capped (52.7-68.6% for the jump model while VIX ranged 51.5 to 83.6) and stops responding. **Reading:** both numbers are computed from the same day's option prices,
so a high co-movement says the fit tracks the market's volatility level; it does not say the model forecasts VIX. A level gap of a few vol points is built in because VIX's formula weights the whole strip, including wings the filtered quotes do not cover.

Figures: `results/figures/12_vix_timeseries.png`, `13_vix_scatter_levels.png`, `14_vix_scatter_changes.png`, `15_vix_lead_lag.png`, `16_vix_gap_and_rolling_correlation.png`.

---

## 11. SVI reference

Raw SVI (Gatheral and Jacquier 2014 for the arbitrage-free background) is a 5-parameter curve per expiry with no dynamics: the "best achievable smile fit" yardstick. On the 118 sampled test days, 1,031 expiries fitted, same-day implied-vol RMSE (vol points, pooled over quotes of SVI-fitted expiries)
[results/svi/README.md; results/svi/svi_vs_dh_by_bucket.csv; results/tables/svi_reference.md]:

| Quotes | n | SVI | Double Heston + jumps | Double Heston |
|---|---|---|---|---|
| all quotes | 52,980 | 0.502 | 1.155 | 3.069 |
| near the money (absolute k at most 0.04) | 22,168 | 0.533 | 1.042 | 1.107 |
| wings (absolute k above 0.10) | 7,608 | 0.513 | 1.421 | 6.235 |

SVI fits the same-day smile better than the model, as it should: it has five free parameters for every expiry and nothing connects expiries or days. 48.8% of its fitted expiries violate butterfly no-arbitrage somewhere on the grid (3.8% inside their own quote range), so SVI's error is a slightly optimistic reference
(an arbitrage-free SVI could only fit worse) [results/svi/README.md]. The walk-forward jump model's gap to SVI is largest at 7 days or less (1.65 to 3.24 against 0.35 to 0.54).

Figures: `results/figures/09_svi_vs_dh_by_day.png`, `10_svi_vs_dh_by_bucket.png`, `11_svi_example_smile.png`.

---

## 12. Trading test

One trade is one NIFTY option (the 5 most-traded contracts a day within 5% of the forward, 5 to 45 days to expiry), bought if the model says it is cheap and sold if rich (or if its volatility forecast is above or below today's market volatility), at the day's close, closed at the next close. A win is a profit above zero after cost; "hedged"
removes the index move with the option's delta on the forward. Thresholds were chosen on 2016-2021; the rows are the 2022-2026 test years; cost is a share of the option price on each side [trade_test.py docstring; COMPARISON_1.0_vs_2.0.md]. Ranges resample whole days.

**Entered at the signal day's close** [results/tables/trading_summary.md]:

| Signal / holding | Cost per side | Trades | Wins per 100 | Blind trades, same mix | Rs per trade [95% range] |
|---|---|---|---|---|---|
| forecast_edge / plain | 0% | 1,370 | 43.9 | 42.1 | +6.38 [+1.35, +12.95] |
| forecast_edge / hedged | 0% | 1,370 | 43.5 | 41.4 | +5.78 [+1.83, +10.03] |
| rich_cheap_edge / plain | 0% | 4,168 | 44.9 | 43.4 | +2.69 [+0.22, +5.45] |
| rich_cheap_edge / hedged | 0% | 1,313 | 44.5 | 41.9 | +6.48 [+2.33, +11.21] |
| rich_cheap_edge / hedged | 1% | 1,313 | 42.0 | 39.0 | +4.34 [+0.19, +8.98] |

**The same trades entered one day later** (matched on signal day and contract; no cost) [COMPARISON_1.0_vs_2.0.md section 2; metrics.json `trading.matched.*`]:

| Signal / holding | Matched trades | Wins per 100: signal close / one day later | Rs per trade: signal close / one day later | Profit lost by waiting [95% range] |
|---|---|---|---|---|
| forecast_edge / plain | 845 | 45.8 / 44.6 | +5.84 / +6.04 | -0.20 [-10.25, +9.42] |
| forecast_edge / hedged | 845 | 47.8 / 40.9 | +6.54 / +1.55 | +4.99 [-2.14, +11.92] |
| rich_cheap_edge / plain | 3,159 | 45.8 / 43.3 | +2.59 / +0.95 | +1.64 [-2.87, +6.07] |
| rich_cheap_edge / hedged | 838 | 48.2 / 40.6 | +7.23 / +1.21 | +6.03 [-0.99, +13.45] |

**Reading it (g).** 41 to 45 of 100 trades profit in every row of `results/tables/trading_summary.md` (a coin flip would give 50). The model mostly says buy, and a bought option loses a day of time value, so the typical trade loses; the average is positive because wins are larger than losses (the median trade is negative in
every row). At the signal day's closing price three of the four versions have a positive average whose range stays above zero with no cost, and the index-hedged rich/cheap version stays above zero at a 1% cost. Entered one day later, the two hedged versions lose most of their profit and turn negative at 1% cost, but the size of
that loss is itself not statistically certain (its range includes zero), and none of the one-day-later averages is distinguishable from zero. **The edge is not shown to survive a one-day delay.** Closing prints are last trades, not quotes you can deal at. An independent audit found no look-ahead [COMPARISON_1.0_vs_2.0.md].
Note: the "one day later" rows of the first table (`trading_summary`) use different signal thresholds for some versions (the 4,525-trade hedged rows use no threshold), so the matched table above is the fair comparison. This is a backtest, **not investment advice**.

---

## 13. Limitations

- **Data.** NSE end-of-day files hold closing prints, not simultaneous quotes; illiquid options can be stale, which flatters persistence in the thin early years (2016-2018). The data are not filtered for no-arbitrage violations (64.1% of quotes take part in one at a tolerance of one tick); the headline difference is robust to excluding them (Section 2).
- **Timing.** Time to expiry is date-level (calendar days / 365); the expiry's time of day (15:30) is not modelled, which matters only for 1 to 3 day options [AUDIT.md section 4]. Options within 3 days of expiry are dropped.
- **Shortest expiries.** The model's errors are largest at 3 to 7 days (Section 6.3); the walk-forward jump model is worse than a fresh daily jump fit there.
- **Model range.** Variance levels are bounded above (0.30 per factor), so the model is capped near the March 2020 peak (16 days) [results/vix/README.md]; a wide-bounds run helps the pure model in 2020 but not the final forecast [COMPARISON_1.0_vs_2.0.md section 3].
- **Identifiability.** Individual parameters are not identified; do not interpret them one by one (Section 7).
- **Structure/state design** is not shown to help (Section 5); the headline walk-forward numbers come from a configuration that independent jump fits beat on the 118 sampled days (1.142 against 1.254, no misfit in either).
- **Near the money** the time-scaled smile is better than the model (Section 6.2). **The diffusion model** loses to the benchmark and, in one test, to plain Heston.
- **PINN.** Valid only inside its box; diffusion only; 30,000-step budget stopped early; three seeds; the surrogate is not arbitrage-free by construction.
- **PINN-accelerated calibration.** Higher objective cost (about 0.8%), non-converged polish on 115 of 118 days, weaker when the optimum is outside the box; 118 days and one seed; applies to the diffusion model only.
- **Samples.** The expensive experiments use 118 days spaced 10 trading days apart; the intervals resample days and may be slightly narrow because volatility regimes persist [results/pinn_calib/README.md].
- **Comparison with 1.0.** DH1 is a rebuild (Section 5). 1.0 and 2.0 differ in data, instrument and metric; rows marked "not identical" in `COMPARISON_1.0_vs_2.0.md` are not like-for-like.
- **Trading.** Backtest on closing prints; the edge is not shown to survive a one-day delay; not advice.
- **Pricer.** One documented corner case (Section 4); QuantLib was not available and was not used for a cross-check.

---

## 14. Future work

- Retrain the surrogate with a wider box (the exact optimum lies outside the box on 16% of days) and for longer; test whether the ranking of data-only against hybrid changes with a bigger budget (not tested).
- A PINN with a jump (integro-differential) term, so that it can price the headline model [DECISIONS.md decision 3].
- Separate the structure/state split from the walk-forward refits and the three bundled parts of step 3 in the waterfall (no saved result can do this now), and test independent jump fits plus misfit on the full test period.
- Fix the pricer's integration spacing so that it shrinks with the spread of the distribution (a documented, tested remedy) [results/pricer_validation/README.md].
- Extend the heavy tests from 118 sampled days to all test days; search wider than the edges of the memory and jump grids; run BANKNIFTY.
- A real execution study with bid/ask quotes before any claim about trading.
- Replace the date-level expiry with the exact expiry time for 1 to 3 day options.

---

## 15. References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options.
- Christoffersen, P., Heston, S. and Jacobs, K. (2009). The shape and term structure of the index option smirk: why multifactor stochastic volatility models work so well.
- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap.
- Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions.
- Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other exponential Levy processes.
- Carr, P. and Madan, D. (1999). Option valuation using the fast Fourier transform.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model.
- Jaeckel, P. (2015). Let's be rational. (Not used: implied volatilities here come from a vectorised bisection, accurate to float64 round-off [AUDIT.md section 5].)
- Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
- Raissi, M., Perdikaris, P. and Karniadakis, G. E. (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
- Horvath, B., Muguruza, A. and Tomas, M. (2021). Deep learning volatility.
- Wang, S., Teng, Y. and Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. (Its adaptive loss weights were tried and were worse in this project [results/pinn/README.md, section 5].)

References are cited by author, year and title only.

---

## Notes on sources

- Numbers that come from configuration or code constants rather than a results file: the walk-forward window (40 days), refit interval (10 days), burn-in (40 days), bootstrap block and resamples (20 and 2,000) and parameter bounds [configs/nifty.yaml; also results/PIPELINE.md]; the soft-L1 scale (0.8) and vega floor (0.05) [dh2/calib.py; DECISIONS.md decision 10]; the parity strike count (6) [dh2/data.py; results/dq/README.md].
- The Section 12 "Rs per trade" and "wins" for matched trades come from `COMPARISON_1.0_vs_2.0.md` and `results/metrics.json` (source `data/trade_matched_dhj_main.json`), not from a file in `results/tables/`.
- "The median trade is negative in every row" is read from the median column of the trading table in `COMPARISON_1.0_vs_2.0.md`.
- The development-period test of the forecast (dev columns) is from `results/tables/forecast_scores.csv`.
- No new number was computed for this report.
