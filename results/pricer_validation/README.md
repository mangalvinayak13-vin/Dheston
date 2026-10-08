# Pricer validation: is `pricer/models.py` accurate? (work package A)

Question: everything (calibration, the PINN training data) rests on `pricer/models.py` (midpoint-rule Gil-Pelaez, "little trap" characteristic function,
integration accuracy never checked). Answer, measured against three independent pricers: **yes, to better than 2e-6 vol points on every real NIFTY
quote and to better than 1e-4 vol points anywhere in the synthetic grid; one real, understood, practically irrelevant error exists (aliasing at long
maturities with extreme parameters).**

## What was built (all new files, nothing existing edited)

| File | What |
|---|---|
| `dh2/cos.py` | COS pricer (Fang-Oosterlee), vectorised float64, own characteristic function, analytic cumulants, adaptive terms AND adaptive truncation width. `cos_price`, `cos_price_many`, `normalized_price(x, tau, params)` (u = C/(K DF), x = ln(F/K)), optional Merton jumps |
| `dh2/lewis.py` | Lewis (2001) single integral, `scipy.integrate.quad`, Black-Scholes control variate (keeps relative accuracy far out of the money) |
| `dh2/mc.py` | Monte Carlo, Andersen QE for both variance factors, martingale-corrected log-price, antithetic pairs, optional control variate, fixed seed, price + standard error, optional jumps |
| `tests/test_pricer_cross.py` | 101 tests: 99 pass, 2 strict xfail (the finding below). Fast ones ~28 s, with the 11 slow Monte Carlo ones ~39 s |
| `results/pricer_validation/errors.csv` | errors of project / Lewis / MC vs COS by maturity bucket x moneyness bucket x model x domain, three sources of points |
| `results/pricer_validation/worst_cases.csv` | the 40 largest project-vs-COS implied-vol errors |
| `results/pricer_validation/timing.json` | one real day's chain (2026-10-01, 659 quotes, 9 expiries, 5-453 days), bulk throughput |
| `results/pricer_validation/make_validation.py`, `cf_scan_mpmath.py` | regenerate the above (`make_validation.py` ~10 min, one process; `cf_scan_mpmath.py` ~6 min) |

How the independence is real: the characteristic function in `dh2/cos.py` is derived from the Riccati ODE and coded from scratch (it never imports
`_log_cf_factor`); it is checked against mpmath (30 digits) solving the ODE itself, and the three inversions (cosine series, Lewis integral, simulation)
share nothing but the model. Lewis and COS share my characteristic function, so they cannot expose an error in it; mpmath and the Monte Carlo (no
characteristic function at all) do.

## Headline numbers

All errors are project pricer minus COS unless stated; price errors in rupees (NIFTY ~ 25,000, so 1 rupee ~ 4e-5 of the index); implied-vol errors in
vol POINTS and only for options worth at least one NSE tick (Rs 0.05) (below that the inversion is ill-conditioned, see caveats).

| Comparison | Points | Max price error | Max relative | Max IV error | Mean IV error |
|---|---|---|---|---|---|
| Project vs COS, **real chains** (62 days, calibrated parameters, the quotes the calibration uses, diffusion) | 18,071 | 4.1e-5 Rs | 9.6e-7 | **1.5e-6 pts** | 3.9e-9 pts |
| Project vs COS, real chains, with jumps (dhj, the model of record) | 18,071 | 4.2e-5 Rs | 6.8e-8 | **6.0e-7 pts** | 1.4e-9 pts |
| Project vs COS, synthetic grid (26 sets x 16 maturities 1d..1y x 21 strikes to +-25%), diffusion | 8,736 | 7.6e-3 Rs | 2.0e-5 | **1.0e-4 pts** (1 year, extreme set) | 3.3e-7 pts |
| same, with jumps | 8,736 | 7.6e-3 Rs | 1.0e-5 | 1.0e-4 pts | 2.2e-7 pts |
| same, synthetic, expiries up to 120 days only | | 3.8e-7 Rs | 1.7e-6 | 6.1e-6 pts | |
| **Lewis vs COS** (all of the above points) | 53,600 | 2.2e-8 Rs | **1.3e-9** | 8.9e-9 pts | 1e-11 pts |
| **Monte Carlo vs COS** (10 parameter sets x 7 strikes, 200k antithetic pairs = 400k paths each) | 70 | | | max abs z = **1.98**, mean abs z = 0.60 | |

Reading: COS and Lewis, two completely different inversions, agree to 1.3e-9 relative (the brief asked for ~1e-6). Monte Carlo agrees with COS within
statistical error at every one of 70 points (all inside the 99% interval, 2.576 standard errors) and the project pricer is, point for point, the same as
COS at those points (project vs COS on the Monte Carlo points: max 8.6e-10 Rs). The characteristic function (mine and the project's `_log_cf_factor`)
matches the mpmath ODE solution to <= 5e-14 in the log (<= 6e-16 in the CF value itself) at 540 points (5 extreme parameter sets x 4 maturities x Im(u) in {0, -0.5, -1} x 9 values of u from 0 to 1000, 1 day..1 year,
`cf_scan_mpmath.py`; also 9 points in the test suite), so there is no complex-logarithm branch problem for these parameters in either implementation.

## The one disagreement found, and why

**The project pricer is off by up to Rs 0.0076 (2e-5 relative, 1.0e-4 vol points) for 1-year options when the return distribution is very wide and
heavy-tailed** (calibrated days of March-April 2020: v0 = 0.3, xi = 1.5, rho = -0.98 and similar). Largest case: 2020-03-18 parameters, T = 1 year,
|ln K/F| = 0.25, COS = Lewis = 713.24617, project = 713.23861. In `worst_cases.csv` all 40 worst points are the two 2020 parameter sets at T = 365 days.
Nothing above 182 days is affected for ordinary parameter sets; the real NIFTY chains (including the 453-day expiry) show at most 4e-5 Rs.

Cause (shown, not guessed): the fixed integration spacing `_DU = 0.5` makes the midpoint rule alias the density at +-2 pi / 0.5 = +-12.57 in ln(K/F). For the
2020 set at one year the log-return has variance c2 = 0.41 and fourth cumulant c4 = 2.8, i.e. real probability mass beyond 12. The truncation floor
(`_CF_FLOOR = 1e-8`) is NOT the cause (the CF is 2e-21 at the cut-off, and lowering the floor changes nothing). Halving the spacing removes it: at T = 1, for the strikes
ln(F/K) = +0.22 / 0 / -0.22 the error is 7.1e-3 / 4.4e-3 / 2.8e-3 Rs at spacing 0.5 (as shipped), 3.6e-9 / 2.2e-9 / 1.4e-9 Rs at 0.25 and <= 2e-11 Rs at 0.125
(round-off); at T = 0.75 it goes from 1.0e-4 Rs to 1.6e-11 Rs, at T = 0.5 from 2.1e-8 Rs to 3e-11 Rs. (Experiment run on a private copy of the module with `_DU`
patched; the file itself was not touched.)

Practical weight: 1e-4 vol points against calibration residuals of 0.3-1 vol points, in a corner (one year, extreme parameters, 25% out of the money) that the
data filter (|ln K/F| <= 0.22, traded OTM quotes) barely reaches. The fix, if ever wanted, is a spacing that shrinks with the spread of the distribution
(spacing <= pi / (a few times the 99.99% range of z)); not done, the pricer is "imported unchanged" by design.

Encoded in the tests as two **strict xfails** (`test_project_pricer_vs_cos_relative_price_in_the_calibration_domain[covid2020]` and the jump twin): the
tolerance (1e-6 relative price, fixed before the comparison) was not loosened; they turn red the day the pricer is fixed.

Nothing else disagreed. Smaller observations: (i) the project pricer clamps xi to >= 1e-3, so its Black-Scholes limit is a 1e-3-vol-of-vol smile (tested, price
differences vs xi = 0 are O(xi) ~ Rs 0.28); (ii) below about Rs 1e-6 its absolute noise (1e-10 to 1e-8 Rs) is larger than the option's value, irrelevant
for traded quotes (the data starts at Rs 1).

## What passed (tests/test_pricer_cross.py)

- (a) characteristic function vs mpmath ODE (9 points incl. complex shifts, xi = 1e-4, u up to 1000), exact deterministic-variance limit, martingale
  property (phi(-i) = 1) for diffusion and jumps, same jump convention as `dh2/models.py`, all four cumulants vs mpmath differentiation (diffusion and jumps).
- (b) COS error falls exponentially with the number of terms (16 ... 1024 vs 8192) to round-off (< 1e-11), adaptive stopping reports convergence.
- (c) COS vs Lewis on 9 parameter sets x 8 maturities (1 day..1 year) x 9 strikes (+-25%) x call/put, 1e-6 relative with 1e-11 absolute floor (observed
  1.3e-9), and with jumps for 5 sets.
- (d) project pricer vs COS: implied vol within 0.01 vol point (observed <= 1.0e-4) for all 9 sets and 5 jump sets in the calibration domain (>= 3 days,
  |ln K/F| <= 0.22, price >= Rs 1); vs Lewis too; 1-2 day options and +-25% (tradeable prices) within 0.1 vol point. Relative price 1e-6: all pass except the two xfails.
- (e) single-Heston limit (factor 2 switched off) vs the project's `heston_price` (1e-8 relative); Black-Scholes limit at xi = 0 exact (1e-9), small xi bounded
  by 0.1 xi F, COS = Lewis = project at xi = 1e-3.
- (f) put-call parity with both legs integrated directly (not by parity) to 1e-10 F, bounds, monotone and convex in strike, at 1, 2, 14 and 365 days, five
  parameter sets; stability at 1-2 days (and a 1-hour option) against Lewis.
- (g) Monte Carlo (slow): 10 parameter sets, ATM call inside the 99% interval, all seven strikes within 4 standard errors, E[exp z] = 1; puts, jumps and the
  control variate in a further test; reproducible with the seed.
- **Not done: QuantLib comparison. QuantLib is not installed (and was not installed).**

## Two things found in my own COS while building it (fixed, kept as tests)

1. **Textbook truncation (L = 10) is not enough.** With the fast variance factor at vol-of-vol ~ 2 and non-Feller (as calibrated), or with jumps, the tails are
   heavier than the cumulants say: at L = 10 the put leg lost ~3e-9 of probability mass (5e-5 Rs on a Rs 1 option; up to 2e-4 relative on far out-of-the-money
   options, and L = 14 is still 1e-6 relative off for the worst set). So the width is checked, not assumed: price at L = 14 and 21, keep widening x1.5 until the price stops changing (2e-13 + 1e-9 relative). Over 122
   calibrated parameter sets x 9 maturities the verified width was 14 (confirmed by 21) in 89% of cases, 31.5 in 10%, 47 in 0.7%; every row reports `L_used`.
2. **The call payoff e^z amplifies round-off by e^(x+b)** when the interval reaches far right (1-year, extreme sets: b ~ 20). Whenever e^(x+b) > e^6 the
   out-of-the-money call is obtained from the put leg by parity instead (absolute accuracy ~1e-15 in u). With this, direct and parity-based prices agree to 1e-10 F.

## Timing (one process, one BLAS thread; other jobs were running, so only the order of magnitude is reliable; two runs shown)

Day 2026-10-01: 659 quotes, 9 expiries (5-453 days). Best of 5.

| | dh (diffusion) | dhj (jumps) |
|---|---|---|
| project pricer (`dh2.models.price`, the calibration's function) | 17 ms (26 us/option) | 19-24 ms (29-36 us/option) |
| COS, default (width checked) | 82-103 ms (125-156 us/option) | 147-165 ms (222-250 us/option) |
| COS, fixed L = 14 (no width check, ~5e-10 accuracy) | 34 ms (52 us/option) | 48 ms (73 us/option) |
| Lewis adaptive quad | 4.0-4.1 s (6.0-6.2 ms/option) | 5.8-5.9 s (8.8-9.0 ms/option) |

**COS is not faster than the project pricer on a chain**: 5-8x slower in its default (checked) form, 2-2.5x with fixed L (its value is independence, not speed; it needs
512-2048 terms per expiry). Lewis is ~35-50x slower than COS. Bulk (the PINN data use): `cos_price_many` on 2000 parameter sets x 8 strikes, each with its own
parameters and maturity, took 2.1-3.9 s (129-244 us/option); `normalized_price` on 8000 completely independent (x, tau, params) triples took 5.3-8.1 s (660-1020 us/triple).
Roughly 4,000-8,000 options per second per core; for the PINN, pass `L=14.0` for ~2.5-3x more.

## Caveats

- The parameter sets are the diffusion part of the 2016-2026 jump-model calibrations (`outputs/parameters_dhj_main.csv`), i.e. realistic regimes but not
  diffusion-only fits. The synthetic grid uses F = 25,000 and r = 6.5% (only F and DF matter; verified).
- IV errors are reported for options worth >= Rs 0.05. Prices below that (1-day options 20% away, worth 1e-20 Rs) cannot be meaningfully inverted; the
  `max_abs_iv_err_pts_any_price` column (price floored at 1e-12) shows IV errors up to 14 vol points there, which is the project pricer's ~1e-9 Rs absolute noise
  divided by a vega of ~1e-10. Not a market-relevant error.
- "in_domain" = >= 3 days, |ln K/F| <= 0.22, COS price >= Rs 1 (what the calibration uses). The ">400d" maturity bucket is the single 453-day expiry of real chains.
- Monte Carlo has statistical, not exact, resolution: 400k paths give a standard error of 0.1-0.3% of an at-the-money price, so a QE discretisation bias smaller than
  that (252 steps a year, minimum 32) would not be seen. It cannot detect the 1e-5-level effects COS and Lewis resolve; it is a gross-error check of everything else.
- Lewis and COS share the characteristic function (mine), so their agreement does not test it; the mpmath checks and the Monte Carlo do. Agreement of my CF with the
  project's `_log_cf_factor` at 540 points is a consequence of both matching the ODE.
- Out-of-the-money calls beyond the e^6 rule (long maturities, extreme sets) come from parity: absolute accuracy ~1e-15 in normalised price, so relative accuracy
  is lost below u ~ 1e-10 (not relevant for quotes).
- Timing is on a loaded laptop (two runs differ by up to 2x).
