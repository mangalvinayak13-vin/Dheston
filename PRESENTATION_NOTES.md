# PRESENTATION NOTES: Double Heston 2.0 on NIFTY 50 options

A slide-by-slide plan for a talk of about 12 to 15 minutes (14 main slides) plus backup slides for questions. Every slide is tied to one figure in `results/figures/` (by file name) or one table in `results/tables/`.
Every number has its source in square brackets (paths relative to the repository root). Terms are explained in `docs/EXPLAINER.md`; the full report is `RESULTS.md`.
Units: errors are in **volatility points** (one percentage point of implied volatility); lower is better. "Test period" means 2022-2026. Not investment advice.

**Suggested order:** the answer first (slide 1), then why you can trust the machinery (data, pricer), then what the model does (smiles, where it misses), then the main test (forecast, by year), then the honest negatives
(near the money, waterfall, identifiability), then the extras (VIX, PINN, PINN calibration), then trading and limits. Say the negatives yourself before you are asked.

---

## Main slides

### Slide 1. The question and the answer in one picture
**Figure:** `results/figures/01_summary_panel.png`
- The question: can the Double Heston option model fit NIFTY 50 option prices and forecast tomorrow's implied-volatility surface better than "tomorrow equals yesterday"?
- The answer: with jumps and yesterday's leftover error, 1.034 vol points against 1.236 for the best no-model benchmark; paired difference -0.202 [-0.227, -0.181] over 1,173 test days [results/tables/forecast_scores.csv].
- The map of the talk: data, pricer check, forecast test, what fixed the old recipe, a neural-network speed-up, and what did not work.

### Slide 2. The data
**Table:** `results/tables/data_quality_funnel.md`
- 2,470 NSE daily files (Oct 2016 to Oct 2026); 5,254,005 NIFTY option rows, of which 1,517,128 traded and 736,313 survive the filters (14.0% of all rows); every day gives a usable surface, with 298 quotes a day on average [results/dq/README.md].
- Filters in plain words: traded, at least 3 days to expiry, forward from put-call parity, out of the money, price at least Rs 1, within 22% of the forward.
- Split fixed before any result: development 2016-2021 (1,256 days), test 2022-2026 (1,173 days) [results/tables/forecast_scores.csv; DECISIONS.md decision 1]. All settings were chosen on development only.

### Slide 3. Is the pricing formula right? (the exact pricer)
**Figure:** `results/figures/17_pricer_validation.png`
- Checked against three independent pricers (COS, Lewis, Monte Carlo): the project pricer is within 1.5e-6 vol points on 18,071 real-chain points; Lewis and COS agree to 1.3e-9 relative; Monte Carlo agrees at all 70 points (largest |z| 1.98) [results/pricer_validation/README.md; results/tables/pricer_validation.md].
- One documented corner case: 1-year options with extreme March-April 2020 parameters, up to 1.0e-4 vol points; cause found (integration spacing 0.5 aliases a very wide distribution).
- Why it matters: everything else (calibration, neural-network training data) rests on this formula.

### Slide 4. What a fit looks like
**Figure:** `results/figures/05_market_vs_model_smiles.png`
- Three real test days, three expiries each: at 9-10 days the diffusion curve (green) is too flat in the wings while the jump model (orange) follows the market; at 86-121 days all curves are close.
- Not perfect: on 2022-02-01 the 9-day jump curve sags below the quotes near the money (error 2.16 vol points) [results/tables/smile_figure_days.csv]. Days chosen by a fixed rule, not by fit quality.
- Jumps are the extension to the brief's diffusion model; the project reports both.

### Slide 5. Where the models miss
**Figure:** `results/figures/06_where_models_miss.png`
- Same-day error pooled over 118 sampled days: one volatility 6.13, Heston 3.67, Double Heston 2.91, with jumps 0.77 [results/buckets/README.md].
- The error lives in the short-dated wings: 3-7 days, deep puts: 19.1 (Double Heston) against 1.48 (with jumps). For expiries beyond 120 days all smile models are between 0.43 and 1.00.
- Honest point: the walk-forward jump model is worse than a fresh daily jump fit at the shortest expiries (3.65 against 1.48).

### Slide 6. The main test: forecast against benchmarks
**Figure:** `results/figures/03_model_comparison.png`
- Walk-forward: fit on days up to t, predict day t+1, score. Benchmarks: yesterday's smile (1.468) and the time-scaled smile (1.236) [results/tables/forecast_scores.csv].
- Jump model + misfit 1.034 [95% 0.915, 1.171]; the jump model alone 1.201 is only a tie (difference -0.035 [-0.095, +0.028]).
- Wilcoxon and Diebold-Mariano, Holm-adjusted: the jump model + misfit beats the time-scaled smile on 90% of days, p below 1e-10 [results/stats/tests_test_period.md].

### Slide 7. Is it consistent through time?
**Figure:** `results/figures/02_error_by_year.png`
- The jump model + misfit beats the time-scaled smile in all 11 calendar years (2016 and 2026 are partial) [results/tables/forecast_by_year.csv].
- Every method's error peaks in 2020 (time-scaled smile 3.015, jump + misfit 2.695).
- Robustness: 15 reruns with different windows, refit schedules, priors and bounds give 1.033 to 1.036 for the final forecast [results/tables/robustness_runs.csv].

### Slide 8. Two honest negatives about the forecast
**Table:** `results/tables/significance_tests.md`
- The diffusion Double Heston loses to the time-scaled smile: 1.377 against 1.236 with the misfit (+0.141), 2.792 without; the Wilcoxon and Diebold-Mariano tests favour the benchmark [results/tables/forecast_scores.csv; results/stats/tests_test_period.md].
- Near the money (absolute ln(K/F) at most 0.04) the benchmark is slightly better than the jump model: 1.071 against 1.088, significant (Diebold-Mariano +9.41); the model's advantage is in the wings and at short expiries [results/stats/tests_test_period.md].
- Cross-sectional test: in the maturity-extrapolation test the diffusion Double Heston (2.02) is worse than plain Heston (1.37) [results/crosssection/summary.csv].

### Slide 9. What fixed the old recipe (the waterfall)
**Figure:** `results/figures/04_waterfall_dh1_to_dh2.png`
- Next-day error 5.03 to 1.10 on 118 sampled test days, one change at a time [results/attribution/waterfall.csv].
- What helped: IV-space objective -1.05, global multi-start with the kappa ordering -1.04, Merton jumps -1.66, yesterday's misfit -0.155. What did not: the structure/state walk-forward step made the diffusion model slightly worse (+0.08, p = 0.003); independent jump fits (1.142) beat the walk-forward jump model (1.254). The shared-slow-parameter claim is withdrawn [DECISIONS.md decision 11].
- DH1 is a documented rebuild of a lost script: close in aggregate (2.53 against 2.37) but not an exact reproduction [results/attribution/README.md].

### Slide 10. The parameters are not identified
**Figure:** `results/figures/19_identifiability.png`
- On every one of 118 days at least one parameter sits on a bound; the median standard error is 21% of the allowed range (diffusion model) [results/identifiability/summary.csv].
- Different parameters give the same prices: on 51 of 118 days the exact and accelerated fits have the same cost (within 1%) with parameters differing by more than 5% of their range [results/pinn_calib/README.md].
- Take-away for the audience: interpret fitted surfaces, not individual parameters.

### Slide 11. Link to India VIX
**Figure:** `results/figures/12_vix_timeseries.png`
- The model-implied 30-day volatility tracks India VIX: correlation of levels 0.989 over 2,430 days; of daily changes 0.879 (all years), 0.948 (test years), for the jump model [results/vix/vix_vs_model_stats.csv].
- No lead or lag: the correlation peaks at lag 0 (about 0.88) [results/vix/README.md].
- Caveat: both come from the same day's prices, so this is a sanity check, not a forecast; in the March-April 2020 crash the model is capped by its bounds on 16 days.

### Slide 12. The neural-network copy of the pricer (PINN ablation)
**Figure:** `results/figures/20_pinn_ablation.png`
- Compared with the exact pricer on 65,465 real held-out quotes: data-only network 0.049 vol points, hybrid (data + physics) 0.077, physics-only 1.307 [results/pinn/tables/ablation_summary.csv].
- The plain network beats the PINN on price; the hybrid is better on gamma (median error 1.8-2.4% against 2.5-6.0%), PDE residual and butterfly convexity [results/pinn/README.md]. The PDE was verified against exact prices before training.
- Valid only inside its training box and for the diffusion model only.

### Slide 13. PINN-accelerated calibration
**Figure:** `results/figures/30_pinn_calibration_time.png`
- 8.0x [7.4, 8.7] less wall time (5.85 s to 0.73 s a day) and 11.6x fewer exact-pricer calls, with the same same-day error (2.560 against 2.555) [results/pinn_calib/headline.json; results/pinn_calib/accuracy_by_method.csv].
- The cost: objective 0.8% higher on average (worst day 10.9%); the surrogate alone ends 18% above and only 7% of days within 1% of exact cost, so the exact polish is essential; weaker on the 16% of days whose exact optimum is outside the box.
- Measured fact to keep: the exact pricer is already 38 microseconds a quote, so the gain is real but modest; for 128 parameter sets at once the surrogate is 21x (CPU) and 79x (GPU) faster [results/pinn/tables/speed.csv].

### Slide 14. Trading test and limits
**Table:** `results/tables/trading_summary.md`
- 41 to 45 of 100 trades profit; wins are bigger than losses; entered at the signal day's close the average is positive (for example +6.48 rupees per trade [+2.33, +11.21], hedged rich/cheap, no cost) [results/tables/trading_summary.md].
- Entered one day later, the two hedged versions lose most of their profit and none of the averages is distinguishable from zero: the edge is not shown to survive a one-day delay [COMPARISON_1.0_vs_2.0.md]. Backtest on closing prints, not advice.
- Close with the limits: closing prints not quotes; individual parameters not identified; shortest expiries weak; PINN covers the diffusion model only; future work: wider and longer-trained surrogate, a jump term for the PINN, untangle the structure/state design, real bid/ask for trading [RESULTS.md sections 13-14].

---

## The story in 1 minute

"I fitted a two-factor stochastic-volatility model, Double Heston, with jumps, to ten years of NIFTY option prices and asked whether it can forecast tomorrow's volatility surface better than 'tomorrow looks like today'. Over 1,173 test days it does: 1.034 volatility points against 1.236
for the best simple benchmark, and it wins in every calendar year. The gain is concentrated in the wings and at short expiries; near the money the simple benchmark is slightly better, and the version without jumps loses to the benchmark. I checked the pricing formula against three independent methods,
and I found what fixed the old research recipe: a volatility-based objective, a global search and jumps; the 'shared slow parameters' idea I started with did not help, so I dropped that claim. I also built a neural-network copy of the pricer that makes calibration eight times faster with the same error, but it is only valid
for the diffusion model, and the plain network turned out better than the physics-informed one. The individual parameters are not identified, so I interpret fitted surfaces, not parameters, and the trading test is not shown to be an edge."

## The story in 5 minutes

1. **Question and data (45 s).** NIFTY 50 options, 2,470 trading days, 736,313 clean quotes; implied volatility; a development period (2016-2021) where all settings were chosen and a test period (2022-2026) where I only report [results/dq/README.md; DECISIONS.md decision 1].
2. **The pricer (30 s).** A Fourier pricer, checked against COS, Lewis and Monte Carlo: at most 1.5e-6 vol points on real chains, one documented corner case [results/pricer_validation/README.md].
3. **The test (60 s).** Walk-forward: calibrate on days up to t, forecast day t+1. Benchmarks: yesterday's smile 1.468, time-scaled smile 1.236. Jump model + yesterday's misfit 1.034, difference -0.202 [-0.227, -0.181], better in 11 of 11 years, significant on Wilcoxon and Diebold-Mariano [results/tables/forecast_scores.csv; results/stats/tests_test_period.md].
4. **The honest parts of the result (45 s).** The diffusion model alone loses to the benchmark; near the money the benchmark wins slightly (1.071 against 1.088); in the maturity-extrapolation test plain Heston beats the diffusion Double Heston (1.37 against 2.02) [results/stats/tests_test_period.md; results/crosssection/summary.csv].
5. **What fixed the old recipe (45 s).** Waterfall 5.03 to 1.10: objective in volatility space, global multi-start with ordering, jumps, misfit. The structure/state design made the diffusion model slightly worse; claim withdrawn. The old recipe (DH1) is a rebuild [results/attribution/waterfall.csv; results/attribution/README.md].
6. **The parameters (20 s).** Not identified: every day has a parameter on a bound; interpret surfaces [results/identifiability/summary.csv].
7. **The neural copy (60 s).** A PINN surrogate of the diffusion pricer: the plain network is more accurate on price (0.049 against 0.077), the hybrid is better on gamma, residual and convexity; physics-only fails (1.3). Accelerated calibration: 8.0x faster, same error, 0.8% higher cost, exact polish essential, weaker outside the box [results/pinn/tables/ablation_summary.csv; results/pinn_calib/headline.json].
8. **Sanity checks and the end (15 s).** India VIX correlation 0.989 in levels; the trading test is not shown to be an edge; not investment advice.

---

## "If asked" backup slides

| Backup slide | Figure or table | Use when asked about |
|---|---|---|
| B1. Hidden-quote tests | `results/figures/18_crosssection.png` | overfitting; maturity extrapolation (held-out 0.71 against 2.49; 1.37 / 1.41 / 2.02 / 2.67) [results/crosssection/summary.csv] |
| B2. SVI reference | `results/figures/10_svi_vs_dh_by_bucket.png` (and `09_svi_vs_dh_by_day.png`, `11_svi_example_smile.png`) | "how does this compare with SVI?" (0.502 against 1.155 against 3.069 same-day) [results/svi/README.md] |
| B3. Quote counts | `results/figures/07_quote_counts_by_bucket.png` | how reliable is a bucket error |
| B4. Calibrated volatility and jump rate | `results/figures/08_calibrated_volatility_and_jumps.png` | what jumps look like through time; the March 2020 cap |
| B5. Data efficiency | `results/figures/21_pinn_data_efficiency.png` | "why does the PINN not help?" (0.38 / 0.40 at 1,000 prices; 0.049 / 0.077 at a million) [results/pinn/tables/data_efficiency.csv] |
| B6. PINN error and residual maps | `results/figures/22_pinn_error_maps.png`, `23_pinn_pde_residual_maps.png` | where the surrogate fails; small residual is not accuracy |
| B7. PINN Greeks | `results/figures/25_pinn_greeks.png` | Greeks and gamma |
| B8. PINN speed | `results/figures/24_pinn_speed.png` (table `results/tables/pinn_speed.md`) | "is 8x meaningful?" |
| B9. Smile example and training curves | `results/figures/26_pinn_smile_example.png`, `27_pinn_training_curves.png` | typical surrogate error; training budget |
| B10. Cost ratio and RMSE scatter | `results/figures/29_pinn_calibration_cost_ratio.png`, `28_pinn_calibration_rmse.png` | "why use the exact pricer to polish?" |
| B11. Parameters and time behaviour | `results/figures/31_pinn_calibration_parameters.png`, `32_pinn_calibration_rmse_over_time.png`, `33_pinn_calibration_design.png` | weak identifiability from a second angle; the worst days; the design choice |
| B12. VIX details | `results/figures/13_vix_scatter_levels.png`, `14_vix_scatter_changes.png`, `15_vix_lead_lag.png`, `16_vix_gap_and_rolling_correlation.png` | how tight and how stable the VIX link is |
| B13. No-arbitrage sensitivity | table `results/tables/noarb_sensitivity.md` | "your data has arbitrage violations": 64.1% of quotes at one tick, 12.0% with slack; the advantage stays between -0.166 and -0.224 [results/dq/NOARB.md] |
| B14. Robustness | tables `results/tables/robustness_runs.csv`, `results/tables/robustness_memory.md` | tuning sensitivity: 1.033 to 1.036 across 15 reruns; pattern memory -0.004 (not significant) |
| B15. Identifiability table | `results/tables/identifiability.md` | condition numbers and distinct parameter sets |
| B16. Pricer table | `results/tables/pricer_validation.md` | the full error table |
| B17. Matched one-day-later trades | `results/tables/trading_summary.md` plus the matched table in `COMPARISON_1.0_vs_2.0.md` | the trading delay test (hedged versions: +6.54 to +1.55 and +7.23 to +1.21 rupees per trade) |
| B18. VIX table | `results/tables/vix_link.md` | exact statistics |

## Tips for delivery

- Put the numbers in the slides from this file only; every one has a source.
- Say "volatility points" once and define it; after that say "points".
- Say the negatives yourself: diffusion model loses, near the money the benchmark wins, structure/state split not shown to help, the PINN is not more accurate than the plain network, the accelerated calibration costs 0.8% more, parameters not identified, trading edge not shown.
- If asked something you cannot trace to a file, say "I did not test that" (for example whether longer PINN training would change the data-only against hybrid ranking, or whether imposing the Feller condition would improve forecasts).
