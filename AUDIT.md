# AUDIT (Phase 0): what exists, what the brief adds

Written 2026-10-08 before any upgrade work. Every statement below was checked in code or in a saved result; where something was not
checked it says so. The upgrade brief is integrated with the finished DH2.0 work: nothing finished is redone.

## 1. The three things being compared

| Name | What it is here | Where |
|---|---|---|
| **DH1** | The original research project's Double Heston (forked from Viraj Choudhary's repository). Two parts matter: its pricer, copied unchanged as `pricer/models.py` (the website's pricer), and its calibration style: every day fitted independently from scratch, with fixed default parameters on the website. | `pricer/models.py`; 1.0 numbers in `v1_results/` |
| **DH2.0 (pre-upgrade)** | The recalibration finished on 2026-10-08: slow parameters shared over a 40-day window and refit every 10 days, only two daily variance levels refit per day, optional Merton jumps (`dhj`), pattern memory. Tagged `dh2.0-pre-upgrade` (commit 3d72c53). | `dh2/`, `run_walk.py`, `COMPARISON_1.0_vs_2.0.md` |
| **DH2.0 upgrade** | This branch (`dh2-upgrade`): adds an independent exact pricer (COS), a PINN surrogate, PINN-accelerated calibration, and the evaluation the brief asks for. | to be built |

## 2. The model specification: matches the brief

`pricer/models.py` prices Double Heston as two independent variance factors, each with its own correlation to the index:
`dS/S = (r-q)dt + sqrt(v1) dW1 + sqrt(v2) dW2`, `dv_j = kappa_j (theta_j - v_j) dt + sigma_j sqrt(v_j) dZ_j`, `corr(dW_j, dZ_j) = rho_j`.
Its log characteristic function is the sum of the two factors' logs (`double_heston_price`, uses `_log_cf_factor` twice). Parameter order
`(v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2)`; `xi` is the brief's `sigma`. No difference to document.
Jumps are an extension (DH2.0 `dhj`): a Merton jump log-characteristic function added to the same pricer.

## 3. The pricer (exact ground truth): what is verified and what is not

- Characteristic function: the "little trap" form (Albrecher et al. 2007), `_log_cf_factor`. No branch-cut problem.
- Integration: Gil-Pelaez with an evenly spaced midpoint rule, spacing 0.5, upper limit found by probing where the characteristic function falls below 1e-8. The author's comment claims ~1e-7 error against a finer grid. **Not yet independently verified** (brief Phase 2: COS, Lewis, Monte Carlo cross-checks; missing, assigned).
- The pricer works from the forward (`ln(S_T/F)`), so spot, rate and dividend enter only through F and the discount factor.
- Existing tests (12): the pricer's reference price, Black-Scholes and no-jump limits, jump behaviour, parity recovery, causal rate, pattern matcher. No COS/Monte Carlo/convexity tests yet.

## 4. Why DH1 calibrated poorly: evidence already in hand

Checked item by item against the brief's list. "Evidence" points to a saved result.

| Brief item | Finding | Evidence |
|---|---|---|
| Characteristic function trap | Not the problem: little-trap form in use | `pricer/models.py` `_log_cf_factor` |
| Integration accuracy | Unverified independently (see section 3) | gap |
| Time to expiry | Date-level ACT/365 (expiry date minus quote date). Expiry time of day (15:30) is not modelled; matters only for 1-3 day options | `dh2/data.py`; documented limitation |
| Spot vs forward | 2.0 uses a per-expiry forward from put-call parity; the 1.0 website example uses the quoted spot with a fixed 5.32% rate (how 1.0's research code set forwards was not checked) | `dh2/data.py`, `COMPARISON` row 6 |
| Rates and dividends | 2.0 infers the rate from parity (7-12% "effective financing rate" incl. funding spread; only F and DF matter) | `data/` build, `dh2/data.py` |
| Price vs IV space | 2.0 fits vega-weighted price residuals = IV-space; 1.0 measured price errors on <= 20 quotes per surface | `dh2/calib.py` `residual`; `COMPARISON` header |
| ITM / illiquid quotes | 2.0 keeps only traded OTM quotes, price >= Rs 1, >= 3 days, |ln K/F| <= 0.22 | `dh2/data.py` constants |
| Optimizer | 2.0: multi-start + bounded `least_squares` (soft-L1), warm start | `dh2/calib.py` |
| Factor swap | 2.0: kappa2 = kappa1 + dk, dk >= 0.2, so the factors cannot swap | `dh2/calib.py` |
| Units | variance decimals, vol in vol points; consistent | tests |
| **No jumps** | **Largest single cause.** Same-day misfit ~1.7 vol points without jumps, ~0.35-0.6 with (2025-26) | `COMPARISON` section 3 |
| **Independent daily refits** | 1.0: 99.1% of surfaces had several equally good parameter sets, 91.6% a parameter near a bound; carried forward, it loses to yesterday's smile (2.06 vs 1.20 vol points, 2025-26) | `v1_results/`, `diagnose.py` |
| **No benchmark** | 1.0 never had to beat "yesterday's smile"; the website's fixed defaults price a NIFTY call at Rs 600.40 against a market Rs 370.15 | `COMPARISON` row 6 |

## 5. Data pipeline: already built (Phase 1 of the brief)

2,470 NSE daily F&O files (Oct 2016 to Oct 2026; old format through May 2024, UDiFF after), resumable downloaders, per-day surfaces
(OTM quotes, parity forwards, causal implied rate, implied vols by bisection), cached. Differences from the brief, all intentional:

- Period is ten years, not 6-12 months. Split is **development 2016-2021 (1,256 days) / test 2022-2026 (1,173 days)**, not 70/30; chosen before looking at results.
- Expiry dates come from the data (no hardcoded weekday).
- No-arbitrage monotonicity/convexity filtering was **not** applied. Applying it now would change the dataset and invalidate ~22 hours of runs, so it is reported as a diagnostic instead (assigned).
- India VIX: not downloaded yet (assigned). Data-quality report with quote counts per filter: not written yet (assigned).
- Market IVs use a vectorised bisection, not Jaeckel's algorithm; accurate to float64 round-off.

## 6. Where the finished work already answers the brief

| Brief item | Status |
|---|---|
| P1 data pipeline, parity forwards, expiries, OTM, split | done (differences in section 5) |
| P3 symmetry constraint, IV-space objective, multi-start + polish, warm start, bounds | done |
| P3 temporal regularisation toward yesterday | done and ablated (prior strength 5/10/20, memory prior): changes nothing for the final forecast |
| P3 benchmarks BS, single Heston, DH1-style, pre-upgrade | BS (flat) and Heston done; DH1-style daily fits only for 2025-26 (assigned: test-period sample); SVI reference missing |
| P3 identifiability | parameters-at-bounds shares done; Jacobian condition number / standard errors missing |
| P6 next-day out-of-sample (keep structure, refit state) | done (the walk-forward IS this test) |
| P6 paired tests | block bootstrap done; Wilcoxon / Diebold-Mariano missing |
| P6 cross-sectional (random 80% of quotes; short -> long maturities) | missing |
| P6 model-implied 30-day vol vs India VIX | missing |
| P6 Monte Carlo check of one surface | pricer has MC paths; no comparison on a calibrated surface yet |
| P6 stretch: jumps for short expiries | **done and is the main result** (jumps are what fix the fit) |
| P4 PINN, P5 PINN-accelerated calibration | missing here. Prior art in the 1.0 research (Dhruva Ambhaikar's PINN reproduces the exact pricer to 1.1e-5); not reused, rebuilt to the brief's normalised design |
| P7 deliverables, one command, docs | partly (README, comparison note, figures); one-command pipeline and docs missing |

## 7. UI interface (the UI is finished and is not touched)

The website calls `POST /api/price` with `{strike, kind, params: {ten settings}, market: {spot, t, r, q, strikes}, mc}` and gets back
price, implied vol, Greeks, smile, model chain, Feller check and a Monte Carlo check. It is a pure function of the parameters, so DH2.0
connects through an adapter that supplies calibrated parameters (a small `today.json`); the request/response schema does not change.

## 8. Environment

Python 3.13, torch 2.14 with Apple GPU (MPS) available, numpy, scipy, pandas, sklearn, matplotlib, joblib, pyyaml, mpmath. Not installed:
QuantLib, py_vollib, statsmodels, pyarrow (the plan avoids them; Wilcoxon/DM come from scipy). Apple M5, 4 performance + 6 efficiency
cores, 16 GB. Heavy CPU jobs run one at a time; the PINN trains on the GPU. Yahoo's `^INDIAVIX` history is reachable.
