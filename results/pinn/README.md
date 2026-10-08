# Double Heston surrogate pricer (work package B, brief Phase 4)

A neural surrogate for the **diffusion** Double Heston price (no jumps), in normalised coordinates, validated against the project's exact Fourier pricer
(`pricer/models.py`). It **approximates** the exact pricer. Its value is speed when many prices are needed, smooth autograd Greeks, and a physics
(PDE) consistency that can be measured. It is not a replacement for the exact pricer, and it is only valid inside the box it was trained on (section 6).

**Bottom line.** The ablation winner on price accuracy is the **data-only network** (same architecture, `lambda_pde = 0`), not the PINN. The
hybrid PINN is worse on prices (0.077 vs 0.049 vol points, real held-out chains) but is clearly better on the second derivative (gamma), on the PDE residual and on
strike-convexity (butterfly) arbitrage. Physics-only does not work as a pricer (1.3 vol points). The PDE term did not improve data efficiency in this budget.

## 1. What was built (all new files; nothing existing was edited)

| Path | What |
|---|---|
| `dh2/pinn/data.py` | domain from development-period fits, samplers, exact labels (project pricer; own vectorised COS as a cross-check), Black helpers |
| `dh2/pinn/model.py` | `u = u_BS(x, w) + g * NN(x, v1, v2, tau, Theta)`; initial condition exact by construction |
| `dh2/pinn/pde.py` | PDE residual by autograd; independent finite-difference verification of the PDE against exact prices |
| `dh2/pinn/train.py` | one loop for data-only / physics-only / hybrid; Adam; optional adaptive weights, RAR, L-BFGS; progress files |
| `dh2/pinn/experiments.py` | the named training runs (skips a run whose checkpoint exists) |
| `dh2/pinn/validate.py`, `checks.py`, `figures.py` | held-out validation, Greeks, arbitrage, speed, PDE checks, domain flag, figures |
| `dh2/pinn/surrogate.py` | inference API for the next work package (section 7) |
| `configs/pinn.yaml` | every setting and seed |
| `tests/test_pinn.py` | 13 tests (12 fast, 1 slow); the whole fast suite (`pytest tests -m "not slow"`) passes: 122 passed |
| `results/pinn/checkpoints/` | 16 checkpoints, 1.1 MB each (`final.pt` = `dat_s2`, `final_greeks.pt` = `hyb_s2`) |
| `results/pinn/tables/*.csv`, `results/pinn/figures/*.png` (300 dpi) | all numbers below |
| `data/cache/pinn/` (97 MB, not part of results) | cached labelled data (1e6 training samples, validation sets) |

## 2. The PDE was verified before training, and it is correct as written

The spec's PDE (`u_tau = 1/2 (v1+v2)(u_xx - u_x) + sum_j [kappa_j(theta_j - v_j) u_vj + 1/2 sigma_j^2 v_j u_vjvj + rho_j sigma_j v_j u_xvj]`,
`u(x,v,0) = max(e^x - 1, 0)`) was derived by hand from the S-space PDE (derivation in `dh2/pinn/pde.py`) and then checked numerically: seven-point
finite differences of **exact prices from the project's pricer** (and, separately, of my own COS prices) at 40 random points of the domain
(`tables/pde_verification_points.csv`, `pde_verification_summary.csv`).

| step size (fraction of local scale) | median residual / sum of absolute terms, project pricer | 90th pct | max |
|---|---|---|---|
| 0.25 | 5.7e-5 | 7.6e-4 | 1.9e-2 |
| 0.125 | 1e-6 | 2.5e-5 | 1.9e-3 |
| 0.0625 | below 1e-6 | 1e-6 | 3.8e-5 |

The residual falls about 64x per halving of the step (the sixth-order stencil's truncation error), so what remains is discretisation error. The cross term is
10% of the sum of absolute terms (median) and **with its sign flipped the same check is off by 0.199** (median), 5 orders of magnitude worse, so the test would have
caught the classic sign error. **No change to the spec was needed.** `r` and `q` drop out (tested in `tests/test_pinn.py`).

Exact labels: the project pricer agrees with the independent COS of the pricer-validation package (`dh2/cos.py`) to 1.8e-11 (max, 300 random draws), and with my own COS (N = 2048)
to 1.8e-7 (`tables/cos_vs_pricer.csv`). Labels and validation truth both come from the project pricer.

## 3. Design (and the one deliberate deviation from the spec)

* **Domain**, from the calibrated *development-period* fits only (`walk_dh_main.pkl`, days before 2022-01-01) plus a 25% margin (+-0.05 on correlations), clipped to the calibrator's bounds:
  `tables/domain_box.csv`, `results/pinn/domain.json`. tau from 1 day to 0.6 years, |x| <= 0.30, variances 0.0016-0.36. Training parameter sets are 50% jittered development days, 50% log-uniform in the box. The **last 20% of development days (251 days, 2020-12-28 to 2021-12-31) never enter the training sampler.**
* **Labels**: 1e6 exact prices (125,000 parameter/state/maturity combinations x 8 strikes), project pricer, 73 s.
* **Loss**: data = vega-weighted price error (volatility-point units, to first order the IV error; vega floor Rs 0.05 per point on a 25,000 strike); PDE = `tau*R/vega` at collocation points (also volatility points); asymptotics = the correction `g*NN` must vanish 8-12 Black standard deviations out in the wings and at huge variance. The initial condition is satisfied exactly by `g(0) = 0` (tested with `torch.equal`).
* **Deviation: the gate `g`.** The spec's `g = sqrt(tau)` is the "spec gate" row in the ablation. I replaced it by `g = [e^x phi(d1) + 0.02 exp(-(sqrt(1+d1^2)-1)/4)] sqrt(tau)`, the Black vega at the comparison volatility plus a fat exponential tail (`g = "vegat"`). It still vanishes at tau = 0 so the initial condition is still exact, but it depends on (x, w), not on tau alone. Reason, measured: with `sqrt(tau)` the network had to build the Gaussian vega profile itself; the worst errors with a pure vega gate were all 6+ Black standard deviations out in the wing, where the vega gate dies but the model's fat-tailed wing prices do not, hence the tail term. Same steps, same everything else: **0.242 (spec gate) vs 0.053 (this gate)** mean vol points on the monitoring set.
* **Float32 on MPS is accurate enough for the PDE residual** (decided by measurement, `tables/precision_check.csv`): for trained networks the median |difference to the float64 residual| is 2e-5 against a median residual of 0.45-1.3 volatility points (ratio 1e-5 to 3e-5; 99th percentile of the difference 1e-3). All training ran on MPS with 2 CPU threads.
* **Training**: Adam, 30,000 steps, batch 2,048 (PDE batch 1,024 evaluated every 4th step, `lambda_pde = 0.1`), cosine learning rate 4e-3 to 2e-5, width 256, depth 5, SiLU, 267,521 parameters. A data-only run takes 133 s, a hybrid run 390-450 s, physics-only 618 s (15,000 steps, a PDE step every step).

## 4. Headline numbers (held-out, exact pricer as truth)

Implied-vol error in **volatility points** = |surrogate IV - exact IV| of the out-of-the-money option, scored where the exact OTM price is at least 4e-5 of the strike (Rs 1 on a
25,000 index: the data filter) and tau <= 0.6. Mean +- sd over the three seeds. Full tables: `tables/ablation_summary.csv`, `tables/heldout_metrics_all_runs.csv`.

**Real held-out chains**: every quote of the 251 held-out development days at that day's own calibrated parameters (65,465 scored quotes; 7,605 below the price floor and 1,741 beyond 0.6 y are not scored, `tables/quote_counts.csv`).

| variant | mean | median | 95th pct | 99th pct | max | price MAE (Rs) | PDE residual median / 95th / RMS (vol pts) |
|---|---|---|---|---|---|---|---|
| **data-only, 3 seeds** | **0.049 +- 0.004** | 0.039 | **0.130 +- 0.011** | 0.191 | **0.444 +- 0.065** | 0.49 (max 9.2) | 0.73 / 9.8 / 29.7 |
| hybrid (data + PDE + asymptotics), 3 seeds | 0.077 +- 0.012 | 0.064 | 0.196 +- 0.028 | 0.269 | 0.651 +- 0.082 | 0.74 (max 12.4) | 0.36 / 2.1 / 1.8 |
| data-only with the spec gate `sqrt(tau)`, 1 seed | 0.172 | 0.129 | 0.478 | 0.705 | 2.16 | 1.51 | 1.8 / 27.8 / 75.6 |
| physics-only (no labels), 1 seed | 1.31 | 0.97 | 3.21 | 5.71 | 23.8 | 9.86 | 0.26 / 0.91 / 0.63 |

The shipped `final.pt` (`dat_s2`, best data-only seed on the *development validation set*, not on these sets): mean 0.052, 95th 0.137, max 0.385; price MAE Rs 0.49, worst price error Rs 7.3.

**Held-out parameter sets drawn from the box** (1,500 sets x 8 strikes, 10,343 scored, including the extreme corners the real days never visit): data-only mean 0.082 +- 0.011, 95th 0.248, max 2.66 +- 0.93 (up to 1.6 at a single corner point for `final.pt`);
hybrid 0.141 +- 0.008, 95th 0.463, max 2.96.

**Per moneyness x maturity bucket** (`tables/bucket_errors_all_runs.csv`, 20 buckets, `final.pt`, real held-out chains, buckets with at least 100 quotes): worst bucket mean 0.098 vol points
(OTM calls beyond ln(K/F) = 0.10, 7-30 days, 752 quotes; 95th 0.268); worst single-quote error in any bucket 0.385 (OTM calls 0.03-0.10, 7-30 days). On the box set the worst bucket is the deep OTM calls at 90 days - 0.6 y: mean 0.126, 95th 0.40, max 1.59.
`figures/error_heatmaps.png` shows the same on a regular (x, tau) grid.

**Greeks** (autograd of the surrogate against central finite differences of the exact pricer, 336 held-out real quotes, `tables/greeks_errors.csv`, `figures/greeks_scatter.png`): median error

| | delta (abs) | gamma (fraction of the quote's Black gamma) | dC/dv1, dC/dv2 (fraction of Black variance sensitivity) |
|---|---|---|---|
| data-only (3 seeds) | 0.0008-0.0012 (95th 0.004-0.007) | 2.5-6.0% (95th 13-29%) | 1.4-1.8%, 0.85-1.0% |
| hybrid (3 seeds) | 0.0011-0.0015 (95th 0.006-0.007) | **1.8-2.4% (95th 6.4-7.7%)** | 1.7-2.3%, 0.8% |

So prices are better data-only, but the **second derivative is clearly better and more reproducible across seeds in the hybrid**.

**Arbitrage of the surrogate's prices** (`tables/arbitrage_checks.csv`; 150 parameter sets x 5 maturities x 101 strikes, plus 5 strikes x 40 maturities for calendar; tolerance 1e-6 of the forward = Rs 0.025 on NIFTY; the exact pricer has zero violations at any tolerance):

| at tolerance 1e-6 | butterfly (convexity in K) violated | monotone in K | calendar | worst butterfly violation (any tolerance) |
|---|---|---|---|---|
| data-only (3 seeds) | 0.24%, 1.04%, 0.24% of triples | 0.008-0.055% | <= 0.010% | 2.5e-4 to 4.6e-4 of the forward (Rs 6-12) |
| hybrid (3 seeds) | 0.027%, 0.008%, 0.009% | 0.008-0.019% | <= 0.010% | <= 2.5e-5 (Rs 0.6) |

The surrogate is **not arbitrage-free by construction**; violations are rare and tiny but not zero, and the physics term reduces the convexity violations by roughly 10-100x.

**Speed** (`tables/speed.csv`, `figures/speed.png`; a held-out real chain of 283 quotes in 8 expiries, median of 20 repeats on a machine that was also running other jobs; exact pricer = the project's numpy Gil-Pelaez, one thread):

| | exact pricer | surrogate CPU (2 threads) | surrogate MPS |
|---|---|---|---|
| one chain, prices | 10.9 ms | **1.1 ms (10x)** | 3.3 ms (slower than CPU: launch latency) |
| one chain, prices + Black IV | - | 2.0 ms | 4.1 ms |
| 128 parameter sets x chain (36,224 quotes) | 1,556 ms | 75 ms (21x) | **19.6 ms (79x)** |
| Greeks of one chain | 120 ms (finite differences, 9 chain evaluations) | 21.8 ms autograd (5.5x) | - |

Honest reading: the exact pricer is already fast (38 microseconds per quote). The surrogate is 10x faster for a single chain on the CPU and up to 79x for many parameter sets at once on the GPU, which is what a Jacobian or a global search needs; the gain per single price is modest.

**Data efficiency** (`tables/data_efficiency.csv`, `figures/data_efficiency.png`; mean IV error, real held-out chains; sizes below 1e6 are single seeds with the step budget scaled to the data size; 1e6 = mean of 3 seeds):

| supervised samples | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|
| data-only | 0.380 | 0.147 | 0.099 | 0.049 |
| hybrid | 0.399 | 0.217 | 0.163 | 0.077 |

The hybrid is **not** more data-efficient than the data-only network here (equal at 1e3, worse above). On the box set the picture is the same (0.96/1.00, 0.40/0.39, 0.19/0.31, 0.082/0.141).
Both error curves are still falling with data and with steps. The development probes show the error is optimisation-limited (mean IV error 1.16 / 0.49 / 0.24-0.32 vol points at 3k / 12k / 30k Adam steps on the first design), so longer training would likely lower all of these; I stopped at 30k steps because the user's machine was hot.

## 5. What did NOT work (or did not help)

All from `tables/design_probes.csv` (8,000-step probes, seed 0, in-distribution validation set; *these 25 rows are transcribed from the console output of the probe scripts, they were not re-run by the validation code*).

* **Physics-only fails as a pricer**: PDE residual is the lowest of all variants (median 0.26 volatility points) yet the price error is 1.3 vol points on real chains and 4.5 on the box (95th percentile 16). A small normalised PDE residual does not pin down the price; the initial condition (exact) and the weak asymptotic terms leave a family of near-solutions. I did not try to rescue it (more steps, Fourier features, causal training).
* **Hybrid with the PDE weight at 1 or 0.02 or 0.001 was worse than data-only** (mean IV 0.68 / 0.58 / 0.51 against 0.21 for data-only at 8k steps with the pure vega gate): the PDE loss, normalised in vega units, was dominated by far-wing points and stayed at 600-2,800 volatility-point^2. What fixed it was the fat-tail gate and a larger vega floor (1e-4) in the PDE scale; then lambda_pde = 0.1 gave 0.220 (0.260 at lambda = 1) against 0.145 data-only.
* **Wang-Teng-Perdikaris adaptive loss weights**: worse (0.283 vs 0.220). Residual-based adaptive resampling: no gain (0.224 vs 0.220). Neither is used.
* **L-BFGS after Adam**: 0.491 -> 0.429 in 3.5 minutes on a 12k-step model; not worth it, not used.
* **tanh instead of SiLU**: 2.1 vs 0.73 at 8k steps. Large batches (32,768) lose to small batches (2,048) with more steps at equal wall time. Width 384 and width 128 were no better than 256 x 5, depth 7 only marginally (0.65 vs 0.73) at this budget.
* The two first pilots (including a 1,500-step hybrid) were run concurrently on the GPU and slowed each other by 10x; no conclusions were drawn from their speed.

## 6. Limitations

* **Domain only.** The box (`tables/domain_box.csv`) is much narrower than the calibrator's bounds for some parameters: sigma1 from 0.037, rho1 up to 0.457, theta2 up to 0.109, sigma2 from 1.49, rho2 up to -0.154 (the calibrator allows rho up to 0.98, theta2 up to 0.25, sigma2 from 0.02). A PINN-accelerated calibration must stay in the box or fall back to the exact pricer; `Surrogate.in_domain` returns the flag. tau > 0.6 y is outside (2.3% of real quotes).
* **Test-period flag** (not used for any decision): of 1,174 calibrated test-period days, 22 (1.9%) have rho2 slightly above the box (-0.145 against -0.154); every other parameter and both states are inside (`tables/test_period_domain_flags.csv`). Diagnostic only: on every 10th test day (46,682 scored quotes) the mean IV error is 0.0485 (data-only) and 0.0697 (hybrid).
* **Diffusion only.** No jumps: it cannot price the `dhj` headline model of the main study. It approximates the model the exact pricer prices; it says nothing about how well the model fits the market.
* **Tails.** Typical errors are 0.04-0.05 vol points, but the box corners reach 1.6-3 vol points, and the in-distribution monitoring set reaches 4-17 vol points (max over the three data-only seeds: 6.5, 3.9, 16.8) at a handful of points. The 20 worst points of that set are spread over exact prices from 4e-5 to 0.1 of the strike, far wings (|x| up to 7-10 Black standard deviations) and extreme parameter corners; on the real held-out chains the 20 worst are 0.27-0.43 vol points. I did not find a single cause.
* **Second derivatives of the data-only network are noisy** (PDE residual RMS 30 vs 1.8 for the hybrid; gamma 95th-percentile error 13-29%): use the hybrid (`final_greeks.pt`) for gamma-sensitive work.
* Only 1 seed for physics-only, the spec-gate variant and the data-efficiency points below 1e6.
* The held-out real days (2021) are the *last* part of the development period; their box ranges are included in the domain (the box was built from all development days), which is design on the development period only, not on the test period.
* Compute: final runs 48 minutes of GPU time in total (data-only 3 x 133 s, hybrid 3 x ~420 s, physics-only 618 s, spec gate 134 s, data-efficiency 447 s), plus about 25 minutes of development probes and 3 minutes of validation.

## 7. Using it (for the PINN-accelerated calibration package)

```python
from dh2.pinn.surrogate import Surrogate
from dh2.pinn.data import eta_to_theta
s = Surrogate.load("results/pinn/checkpoints/final.pt", device="cpu")        # prices; final_greeks.pt for gamma-sensitive work
theta = eta_to_theta(eta)                              # walk-forward eta (kappa1, theta1, xi1, rho1, dk, theta2, xi2, rho2) -> Theta with kappa2 = kappa1 + dk
out = s.price_surface(F, K, T, DF, theta, v1, v2, is_call)   # dict: call, put, price, iv, u, in_domain  (arrays over quotes)
u   = s.price_normalized(x, v1, v2, tau, theta)        # numpy -> numpy, torch -> torch (differentiable)
iv  = s.implied_vol(x, v1, v2, tau, theta)             # Black IV of the OTM price
g   = s.greeks(F, K, T, DF, theta, v1, v2, S=spot)     # delta_S, gamma_S, dC_dv1, dC_dv2 by autograd
```

Always check `out["in_domain"]`; outside the box the surrogate is an extrapolation. For many candidate parameter sets use one batched call (the speed table).

## 8. Reproduce

```
python3 -m dh2.pinn.experiments dat_s0 dat_s1 dat_s2 hyb_s0 hyb_s1 hyb_s2 phy_s0 spec_dat_s0 de_dat_1000 ...   # each skipped if its checkpoint exists
python3 -m dh2.pinn.validate sets | metrics | greeks | arb | speed | pde | domain | figures
python3 -m pytest tests/test_pinn.py -q            # add -m slow for the held-out accuracy test
```
