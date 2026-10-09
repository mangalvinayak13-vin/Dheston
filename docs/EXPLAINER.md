# EXPLAINER: Double Heston 2.0 on NIFTY 50 options, for the person who has to present it

This document is written for you. It assumes you know Python and C and are still learning finance and statistics. It explains what every part of the
project does, why it is there, how to read every figure, and what to say when a professor pushes back. Nothing here is a claim you cannot trace: every
number has a file next to it in square brackets (paths are relative to the repository root). Where the project found something unflattering, this
document says it plainly, because the honest version is also the one you can defend.

How to use it: read Part A once (the ideas), skim Part B before the talk (the figures), memorise Part E (the hard questions), and keep Part C (the glossary) open.

The three technical write-ups it points to: `RESULTS.md` (the full report), `PRESENTATION_NOTES.md` (slide plan), `results/report.md` (the auto-generated numbers).

---

## The project in one page

An **option** is a contract that pays off depending on where a price ends up. NIFTY 50 is India's main stock index, and NSE (the National Stock Exchange)
trades thousands of NIFTY option contracts every day. Each contract has a price. From a price you can compute the **implied volatility** (IV): the
volatility number you must plug into the Black formula to get that price back. Plot IV against strike price and you get a curve, the **smile** (it
is usually a lopsided U). All the contracts of one day together form the day's **surface** (one smile per expiry date).

The question of this project: **can a mathematical model of the market, the Double Heston model, describe that surface well, and can it forecast tomorrow's
surface better than a trivial method such as "tomorrow's smile equals today's smile"?**

What we did, in order:

1. Built a clean dataset from ten years of NSE files (Oct 2016 to Oct 2026): 2,470 trading days [results/dq/README.md].
2. Checked that the pricing formula we use is correct by comparing it with three independent methods [results/pricer_validation/README.md].
3. Fitted the model every day ("calibration"), and tested it by walk-forward forecasting: fit on days up to *t*, predict the surface of day *t+1*, measure the error.
4. Found which fixes made the old (1.0) recipe better, one at a time (the "waterfall") [results/attribution/README.md].
5. Built a neural-network copy of the pricer (a PINN) and used it to speed up the fitting [results/pinn/README.md, results/pinn_calib/README.md].
6. Compared the model with India VIX, with a flexible curve-fitting reference (SVI), and with a simple trading test.

The headline: **Double Heston with jumps, plus yesterday's leftover error, forecasts the next day's surface with an average error of 1.034 volatility
points, against 1.236 for the best no-model benchmark, over 1,173 test days; the difference is -0.202 [-0.227, -0.181]** [results/tables/forecast_scores.csv].

The honest negatives you must also say (each is explained below): the diffusion-only Double Heston does *not* beat the benchmark; the "shared slow
parameters" idea is *not* shown to help; the PINN is *not* more accurate than a plain network; the PINN-accelerated calibration is a bit worse on
the objective and weaker on some days; the individual parameters are *not* identified; near the money the benchmark is slightly better than the model;
and the trading result is *not* shown to be a real edge.

---

# PART A. What each piece does and why

## A1. The data pipeline

**What the raw data is.** NSE publishes one file per trading day (a "bhavcopy") listing every futures and options contract with its closing price,
traded volume and open interest. We downloaded 2,470 of them, from 3 Oct 2016 to 6 Oct 2026 [results/dq/README.md]. The format changed in 2024 (an old
format until May 2024, a new "UDiFF" format after) [AUDIT.md section 5], so there are two readers in `dh2/fetch_nse_old.py` and `dh2/fetch_nse.py`.
Think of it as ten years of daily CSV files that you must turn into clean tables.

**What "a quote" is here.** One option contract on one day: strike price, expiry date, call or put, closing price, volume. Important caveat: NSE's
"closing price" is an end-of-day print, not a bid/ask quote, so some contracts that did not trade still carry a stale price. That is why we keep
only contracts that traded (volume above zero) [results/dq/README.md, "Things to know"].

**The filters (the quote funnel).** Out of 5,254,005 NIFTY option rows in the files, 1,517,128 traded (28.9%), and after all filters 736,313 remain
(14.0% of all rows, 48.5% of the traded ones) [results/dq/README.md]. The filters, in order, are in `results/tables/data_quality_funnel.md`:

| Filter | Why it exists |
|---|---|
| traded (volume above 0) | untraded contracts carry stale prices |
| expiry at least 3 days away | in the last days an option's price is mostly rounding noise |
| a forward can be found from put-call parity | needed to define moneyness (see A2) |
| out of the money only | puts below the forward, calls above it: the market convention, and these are the liquid ones |
| price at least Rs 1 | the tick size is Rs 0.05, so below Rs 1 the rounding is a large share of the price |
| absolute value of ln(K/F) at most 0.22 | far-away strikes are rarely traded and badly priced |
| implied vol solves | the price must be inside the no-arbitrage range |

Every one of the 2,470 days gives a usable surface, with 298 quotes a day on average (median 311) [results/dq/README.md]. The number of quotes grows through
the years (about 95 a day in 2016, about 551 a day in 2026) [results/dq/README.md, "Surface shape per year"], because the options market got
more active. That matters when you read year-by-year results: the early years are thin.

**The split: development and test.** Days from 2016 to 2021 (1,256 scored days) are the **development** period: every setting was chosen there.
Days from 2022 to 2026 (1,173 scored days) are the **test** period: used only to report [results/tables/forecast_scores.csv]. This split was fixed
before any result existed [DECISIONS.md decision 1]. It is the same idea as the train/test split in machine learning, with one extra rule: time
only goes forward. The walk-forward needs 40 burn-in days first [configs/nifty.yaml], which is why the number of scored days (2,429 in all) is a bit
below 2,470.

**No-arbitrage checks are diagnostics, not filters.** A price list can contain contradictions (for example, a more expensive option that pays
less). We measured how often: with a tolerance of one tick (Rs 0.05), 64.1% of quotes take part in at least one violation; with an extra 0.25 vol point
of slack for closing-print noise, 12.0% do [results/dq/NOARB.md]. We did not delete these quotes, because that would change every surface and
invalidate about 22 hours of finished runs [DECISIONS.md decision 4]. Instead we rescored the forecasts without the violating quotes: the model's
advantage over the benchmark stays between -0.166 and -0.224 vol points in every variant [results/dq/NOARB.md], so the result does not hinge on them.

## A2. The forward price from put-call parity

A **forward price** F is today's price for delivering the index at the option's expiry date. You cannot read it off a screen for every expiry, but you
can find it from the options themselves. **Put-call parity** says that for the same strike K and expiry,

call price - put price = discount factor x (F - K).

So take the strikes near the money (the code uses the 6 nearest: `PARITY_STRIKES = 6`) and solve for F. Do it for each expiry [dh2/data.py]. The same
relation also gives the **discount factor**, and from it an "implied financing rate" that is smoothed with a trailing 10-day median so that it uses
only past days [dh2/data.py docstring]. The rate comes out between about 7 and 12% because it includes the funding spread; only F and the discount
factor matter to the pricer, not the label [AUDIT.md section 4].

Why not just use spot x exp(rT)? Because NSE's option close and the index close are not at the same instant, and over ten years the rate moved a lot. A
forward implied by the option prices themselves is consistent with those prices. Moneyness is then measured as **ln(K/F)** ("log-moneyness"): zero means
at the money, negative means a strike below the forward.

## A3. Implied volatility

The **Black formula** gives an option's price from the forward, strike, time to expiry, discount factor and one volatility number sigma. Run it
backwards: given a market price, find the sigma that reproduces it. That is the implied volatility. There is no closed form for the inverse, so
`dh2/data.py` does it by **bisection** (halve the interval until the price matches), vectorised in numpy, 64 iterations. This is the same trick as
`sqrt` by bisection in a first C course.

Units: **volatility points** (vol points) are percentage points of volatility. An error of 1.0 means 20% versus 21%. All errors in this project are in vol
points, so "1.034" means about one percentage point of volatility.

**Vega** is how much an option's price changes when the volatility changes by one point. Near-the-money options have high vega; far-wing options
have tiny vega. This matters in A8: dividing a price error by vega converts it into a volatility error.

## A4. The Double Heston model and its ten parameters

The simplest model (Black-Scholes) says the index moves randomly with one constant volatility. Real volatility changes through time, so **Heston (1993)**
makes the variance (volatility squared) itself a random process that is pulled back to a long-run level. **Double Heston** uses two such variance
processes, one slow and one fast, so that the model can describe both a slow-moving background level and quick spikes. In equations [AUDIT.md section 2]:

- the index: dS/S = (r - q) dt + sqrt(v1) dW1 + sqrt(v2) dW2
- each variance (j = 1, 2): dv_j = kappa_j (theta_j - v_j) dt + sigma_j sqrt(v_j) dZ_j
- each variance's noise is correlated with the index's noise: corr(dW_j, dZ_j) = rho_j; the two factors are independent of each other.

The ten parameters (the code writes the vol-of-vol as `xi`; the brief calls it sigma) [dh2/models.py]:

| Parameter | Plain meaning | Analogy |
|---|---|---|
| v0_1, v0_2 | today's variance of each factor | the current temperature of two thermometers |
| kappa_1, kappa_2 | how fast each variance snaps back to its long-run level | the strength of a spring |
| theta_1, theta_2 | the long-run level each variance returns to | the room temperature the thermostat targets |
| xi_1, xi_2 (sigma) | how violently each variance itself fluctuates ("vol of vol") | how jittery the thermometer is |
| rho_1, rho_2 | correlation between index moves and variance moves; negative means "falling market, rising volatility" | the leverage effect |

**Slow and fast.** Factor 1 is the slow one and factor 2 the fast one. In the code the fast speed is written kappa_2 = kappa_1 + dk, with dk at least
0.2, so that the labels cannot swap (otherwise the same surface would have two different parameter vectors: swap factor 1 and 2) [dh2/calib.py].

**The Feller condition** (2 kappa theta > xi squared) guarantees that a variance never touches zero. 2.0 does not impose it (see question 10 in Part E).

## A5. The characteristic function, and why a Fourier method prices options

The price of an option is the discounted average payoff over all possible final index prices. To do that average you need the probability
distribution of the final price. For Heston-type models that distribution has no simple formula. But its **Fourier transform** does. The Fourier transform
of a distribution is called its **characteristic function** (CF): phi(u) = E[exp(i u X)], where X is the log of the final price and i is the imaginary
unit. A CF is like a **fingerprint** of the distribution: it carries the same information in another form.

For Heston and Double Heston the CF is known in closed form (it comes from solving a small system of ordinary differential equations, the Riccati
equations). The project's pricer uses the "little trap" form of the CF (Albrecher et al. 2007), which avoids a numerical problem where a complex
logarithm jumps branch [AUDIT.md section 3]. Then you turn the fingerprint back into a price with an integral over u, like the inverse of a Fourier transform. That is why a
Fourier method prices an option: the hard thing (the distribution) is replaced by an easy thing (the CF) plus a one-dimensional integral. In numerical code
the cost is a short sum or integral per expiry (the COS version below needs 512 to 2048 terms per expiry [results/pricer_validation/README.md]),
not hundreds of thousands of simulated paths (the Monte Carlo check used 400,000).

**Two factors are easy.** Because the two variance factors are independent, the log of the CF of the sum is the sum of the logs. The pricer literally adds two
`_log_cf_factor` calls [AUDIT.md section 2]. Jumps add a third term the same way (A7).

**The project's own pricer** (`pricer/models.py`, inherited unchanged from the earlier research project) uses a midpoint rule on the Gil-Pelaez integral with
spacing 0.5 [AUDIT.md section 3]. We did not trust it blindly; see A6.

## A6. The COS method (and the other two checks)

To check the pricer we built three independent ones [results/pricer_validation/README.md]:

- **COS (Fang and Oosterlee 2008)**: pretend the distribution lives on a finite interval [a, b] and write its density as a sum of cosines
  (a cosine series, which is a Fourier series for a function on an interval). The coefficients of the series come straight from the CF, and the option
  payoff's coefficients are known exactly. The price is then a short sum of N terms (hundreds to a couple of thousand). It is very fast and the error falls
  exponentially in N. We wrote our own CF from the Riccati equations, not copied from the project's pricer, so the two are really independent.
- **Lewis (2001)**: a different single integral formula. A second independent inversion.
- **Monte Carlo** with the Andersen (2008) "QE" scheme: simulate the index paths directly. It uses no CF at all, so it can catch an error in the CF itself.

**What they found.** On real chains (18,071 points with the calibrated parameters of 62 sampled days) the project pricer and COS differ by at most
1.5e-6 vol points; Lewis and COS agree to 1.3e-9 relative; Monte Carlo agrees with COS within its statistical error at every one of 70 points
(largest |z| 1.98) [results/pricer_validation/README.md, results/tables/pricer_validation.md]. The characteristic function matches a 30-digit mpmath
solution of the ODE at 540 points.

**The one disagreement.** For 1-year options with extreme parameters (the March-April 2020 fits) the project pricer is off by up to Rs 0.0076,
which is 1.0e-4 vol points. The cause was found: the fixed integration spacing 0.5 makes the midpoint rule alias the density when the return distribution is
very wide; halving the spacing removes it. It is negligible against fit errors of 0.3 to 1 vol point, and it is kept as two documented "expected failure" tests
instead of being hidden [results/pricer_validation/README.md]. Say this one yourself; it makes everything else more credible.

**A lesson from building COS.** The textbook truncation width (L = 10) was not enough for the fitted parameters, so the code checks the width and widens it
until the price stops changing [results/pricer_validation/README.md]. This is an example of "verified, not assumed".

## A7. What jumps add

A diffusion model moves in small steps, so over a few days it cannot produce a big fall. Real index options show expensive far-out-of-the-money
puts at short expiries (the market pays for protection against sudden crashes). The **Merton jump** extension adds sudden jumps to the index: on average
`lam` jumps a year, each with a log-size that is normal with mean `mu_j` and spread `delta_j`; the drift is adjusted ("compensated") so the forward price is
unchanged [dh2/models.py]. The jump part is independent of the rest, so its log CF just adds to the diffusion's (A5). It brings three extra parameters,
13 in all.

The effect is the largest single change in the project: pooled same-day error across the 118 sampled test days falls from 2.91 vol points (Double Heston
without jumps) to 0.77 (with jumps) [results/buckets/README.md]. In the hardest corner, 3 to 7 days to expiry and strikes 12 to 22% below the forward, the error falls from
19.1 to 1.48 [results/buckets/README.md]. In the next-day forecast waterfall the jump step is worth -1.66 vol points [results/attribution/waterfall.csv]. The jumps are an
extension, not part of the brief's base model; the project reports the diffusion model next to it everywhere.

## A8. Calibration is fitting

**Calibration** means choosing the parameters so that the model's prices match the market's. It is curve fitting, the same as `scipy.optimize.least_squares` on a
model function, just with an expensive model function. The ingredients [dh2/calib.py]:

- **The residual**: for each quote, (model price - market price) divided by the quote's vega. That is, to first order, the implied-volatility error in vol
  points. Why divide by vega? Without it, expensive near-the-money options dominate and the cheap wing options (which carry the smile) are ignored. This is called the
  "IV-space objective", and it is worth -1.05 vol points of next-day error in the waterfall [results/attribution/waterfall.csv]. Vega is floored at 0.05
  (rupees per vol point) so a near-worthless option does not create a giant residual [DECISIONS.md decision 10].
- **A robust loss** (soft-L1 with scale 0.8 vol points): large residuals count linearly instead of quadratically, because some closing prints are stale.
- **Bounds**: each parameter is kept inside a box (listed in `configs/nifty.yaml`).
- **Multi-start**: the surface of the objective has many valleys, so the fit is started from several places (a flat-volatility corner plus random draws; 4 starts for
  the daily fit) and the best is kept. This together with the kappa ordering is worth -1.04 vol points [results/attribution/waterfall.csv].
- **Warm start** and the walk-forward structure (A10) reuse yesterday's answer.

## A9. Why the parameters are weakly identified

**Identifiability** asks: do the prices determine the parameters uniquely? Think of tasting a cake and guessing the recipe: very different recipes can taste almost the same.
We measured this on 118 test days with 16 starts each [results/identifiability/README.md]:

- Locally the problem is not singular (the median Jacobian condition number is 76 for the diffusion model, 529 with jumps), so there is *some* curvature in every direction.
- But the standard errors are large: the median is 21% of a parameter's entire allowed range for the diffusion model (12% with jumps), and for kappa1, theta1, rho1 and dk
  the standard error exceeds the whole range on more than a quarter of days.
- **On every one of the 118 days at least one parameter sits on a bound** (within 0.3% of the interval's end), for both models [results/identifiability/summary.csv].
- Today's variance levels (v0_1, v0_2) and a few others are tight (median standard error about 1.4% of the range); the slow factor's speed and level are not.
- The cost surface is not "many separate minima": on average 12.2 of 16 starts end within 1% of the best cost, and 1.27 distinct parameter sets per day (16% of days have two or more).
  What is left is ridges (long flat valleys) and bounds.

**The PINN calibration gives another view**: on 51 of 118 days, the accelerated and exact fits reach nearly the same cost (within 1%) with parameters that differ by more than 5%
of their ranges, and the two fitted surfaces differ by a median of only 0.04 vol points [results/pinn_calib/README.md]. Different parameters, same prices.

**The consequence to say aloud:** interpret fitted surfaces, not individual parameters. Do not tell a story from "kappa1 was 2.3 on Tuesday".

## A10. The walk-forward test, and why it is the honest test

A **walk-forward** (or rolling) test mimics real use. On day *t* you have only days up to *t*. You calibrate on them. Then you predict day *t+1*'s surface and score the error against what the market
actually did. Then you move to *t+1* and repeat. No future information is ever used, and the error is measured on data the fit never saw.

Details [configs/nifty.yaml, dh2/evalpairs.py]:

- Each method is given for free, on day *t+1*, that day's forward price, times to expiry and the quotes' strikes. A method must supply the *shape* of the surface from its own past.
  (Note, for a sharp professor: the forward on day *t+1* is derived from day *t+1*'s option prices. Every method, including the benchmark, gets it equally; it carries the index
  level, not the smile.)
- The model is refitted every 10 days; the walk-forward structure uses a 40-day trailing window. The first 40 days are a burn-in.
- **The benchmarks** that the model must beat: *yesterday's smile* (read yesterday's IV at the same log-moneyness) and the *time-scaled smile* (read it at moneyness times sqrt(T0/T1),
  because a smile gets narrower like sqrt(time) as expiry nears) [dh2/evalpairs.py]. These are the "yesterday's weather" forecast. A weather model is only good if it beats "tomorrow = today". The old 1.0 research
  project never had such a bar to clear [COMPARISON_1.0_vs_2.0.md section 3].
- **"+ yesterday's misfit"**: the model's own error on day *t* (market IV minus model IV) is carried to day *t+1* as a correction. The model supplies the structure and yesterday's leftover supplies what the
  model cannot express. It is worth -0.155 more vol points in the waterfall [results/attribution/waterfall.csv].
- Errors are the root-mean-square IV error over the day's quotes (vol points), averaged over days. Intervals are **block bootstrap** 95% intervals: resample blocks of 20 consecutive days (so autocorrelation is respected), 2,000 times
  [configs/nifty.yaml].
- **Wilcoxon and Diebold-Mariano tests** compare two methods on the same days. Wilcoxon signed-rank: do the daily differences have median zero? Diebold-Mariano: do the mean losses differ, using a variance
  that allows for autocorrelation (Newey-West, 6 lags, with the Harvey-Leybourne-Newbold small-sample correction). The p-values are adjusted for the 10 pairs tested (Holm). "<1e-10" means too small to print
  [results/stats/tests_test_period.md]. A significant test says a difference exists; it does not say the difference is large.

**What "structure and state" means, and the correction you must make.** The project's second version split the ten (or thirteen) parameters into *structure* (slow ones, fitted jointly over a 40-day window)
and *state* (today's two variance levels, fitted daily). The original motivation was that one day of prices pins down the slow parameters poorly [dh2/struct.py docstring]. The attribution waterfall later
showed that this split is **not** the source of the improvement: adding it made the diffusion model slightly worse (+0.08 vol points next day, p = 0.003), and independent daily jump fits forecast better
(1.142) than the walk-forward jump model (1.254) on the same 118 days [results/attribution/waterfall.csv, DECISIONS.md decision 11]. The split is kept only because all finished walk-forward results were produced with it.
The gain comes from the IV-space objective, the global multi-start with the kappa ordering, and the jumps. Do not present the split as the key idea.

## A11. The PINN and the PDE it obeys

A **neural network** is a function with many adjustable numbers (weights) trained to match examples. A **PINN** (Raissi, Perdikaris and Karniadakis 2019) is a network that is trained
to match examples *and* to obey a known physical law, written as a **partial differential equation** (PDE). Here the "law" is the pricing equation of Double Heston: the price u(x, v1, v2, tau)
must satisfy

u_tau = 1/2 (v1+v2)(u_xx - u_x) + sum over j of [ kappa_j (theta_j - v_j) u_vj + 1/2 sigma_j^2 v_j u_vjvj + rho_j sigma_j v_j u_xvj ],

with the starting condition u(x, v, 0) = max(e^x - 1, 0), the payoff at expiry. Here x = ln(F/K), tau is time to expiry, and the subscripts mean derivatives. In words: the PDE says how the price changes
as time to expiry grows, in terms of how it curves with the index and how it responds to the two variances. The **PDE residual** measures how badly a function breaks the equation (zero means it
obeys it exactly).

Before training anything, the project checked the PDE is correct as written: seven-point finite differences of exact prices from the project's pricer satisfy it to a relative residual that falls about 64 times each time the step
is halved, down to 1e-6; with the sign of the cross term flipped the same check is off by 0.199 [results/pinn/README.md, section 2]. So the equation is right.

**The design** [results/pinn/README.md, section 3]: the network works with the normalised price u = C/(K x discount factor), as u_BS(x, w) + g x NN(x, v1, v2, tau, Theta), the Black price at a comparison volatility plus a gate g
times a network correction. The gate vanishes at tau = 0, so the initial condition (the payoff) is satisfied *exactly*, by construction. It has 267,521 weights (width 256, depth 5), trained with Adam for 30,000 steps on 1,000,000 exact prices
(125,000 parameter/state/maturity combinations x 8 strikes) from the project's own pricer. One deliberate change from the brief: the gate is the Black vega plus a fat tail, not sqrt(tau); that gave 0.053
instead of 0.242 vol points in the same number of steps.

**Three trained variants** (the ablation): *data-only* (learns only from the exact prices), *hybrid* (prices plus the PDE term plus asymptotic conditions) and *physics-only* (the PDE alone, no exact prices).

## A12. Why the surrogate is useful, and PINN-accelerated calibration

A **surrogate** is a cheap copy of an expensive function: like replacing a slow exact routine with a lookup-and-interpolate routine that is fast but approximate. The exact pricer is already fast (38 microseconds
per quote). The surrogate pays off when you need very many prices at once: 128 parameter sets on a chain take 1,555 ms exactly versus 75 ms on a CPU (21x) and 19.6 ms on the Apple GPU (79x). It also gives smooth
derivatives (Greeks) by automatic differentiation, 5.5x faster than finite differences [results/pinn/README.md, results/tables/pinn_speed.md].

**Where it is valid.** Only inside the box it was trained on (time to expiry from 1 day to 0.6 years, |ln F/K| up to 0.30, a restricted range of parameters), and only for the diffusion model, not jumps. About 2.3% of real quotes are
longer than 0.6 years [results/pinn/README.md, section 6, DECISIONS.md decision 9].

**PINN-accelerated calibration** [results/pinn_calib/README.md]: use the surrogate to do the global search (many starting points at once, on the surrogate), then polish the best candidate with the exact pricer. Result on 118 test days:
the same-day IV error is 2.560 versus 2.555 for the exact calibration (paired difference +0.0054 [-0.0091, +0.0205]), with 8.0x less wall time (5.85 s to 0.73 s per day) and 11.6x fewer exact-pricer calls.
The honest part: the objective cost is on average 0.8% higher than the exact calibration's (worst day +10.9%), the polish budget runs out on 115 of 118 days, and on the 16% of days whose exact optimum lies outside the surrogate's box the
accelerated fit is more often more than 1% worse (47% of those days). The surrogate alone is not a calibration (cost 18% above exact); the exact polish is essential.

## A13. What the ablation showed

The ablation compared the variants against the exact pricer on held-out data [results/pinn/tables/ablation_summary.csv]:

| Network | Mean IV error, real held-out chains (vol points) | Reading |
|---|---|---|
| data-only | 0.049 | the most accurate on prices |
| hybrid (data + PDE) | 0.077 | worse on prices; better on gamma, PDE residual and butterfly arbitrage |
| data-only with the brief's gate sqrt(tau) | 0.172 | the gate change mattered |
| physics-only | 1.307 | does not work as a pricer |

- **The data-only network wins on price.** On gamma (second derivative in the index) the hybrid is better: median error 1.8-2.4% of the Black gamma versus 2.5-6.0%. The PDE residual median is 0.36 versus 0.73 vol points. At the 1e-6 tolerance the
  surrogate breaks butterfly convexity on 0.24% / 1.04% / 0.24% of triples (data-only, three seeds) against 0.027% / 0.008% / 0.009% (hybrid) [results/pinn/README.md, section 4].
- **Physics alone does not pin down the price**: it reaches the lowest PDE residual (0.26) and still has a 1.3 vol point price error. A small PDE residual is not enough.
- **Data efficiency**: the hybrid is not more data-efficient. With 1,000 exact prices both are about 0.38-0.40; with 1,000,000 data-only reaches 0.049 and hybrid 0.077 [results/pinn/tables/data_efficiency.csv].
- Decision taken (measured, not chosen in advance): ship the data-only network for calibration speed and the hybrid for Greeks [DECISIONS.md decision 8].
- Limits: box only, diffusion only, tails (box corners reach 1.6 to 3 vol points), second derivatives of the data-only network are noisy, 3 seeds only for two variants. The training budget was stopped at 30,000 steps because the machine was hot; errors were still falling,
  so longer training might change the ranking. That was not tested.

## A14. India VIX and SVI, briefly

- **India VIX** is the exchange's model-free measure of 30-day expected volatility, built from the whole strip of NIFTY options. We computed the model-implied 30-day volatility from each day's fitted parameters and compared. The correlation of levels over all years is 0.989
  (jump model); the correlation of daily changes is 0.879 over all years and 0.948 in the test years [results/vix/vix_vs_model_stats.csv]. It is a sanity check, not a forecast: both are fitted to the same day's prices. The mean level gap is +0.44
  vol points for the jump model. On 16 days of the March-April 2020 crash a variance level sits on its upper bound and the model cannot follow the VIX [results/vix/README.md].
- **SVI** (Gatheral and Jacquier 2014, "stochastic volatility inspired") is a 5-parameter curve fitted separately to each expiry's smile. It is a pure curve fit with no dynamics, so it is the yardstick for "the best a smile fit can do on that day". On 118 test days its same-day RMSE is 0.502
  against 1.155 for the jump model and 3.069 for the diffusion model [results/svi/README.md]. 48.8% of its fitted expiries have a butterfly arbitrage somewhere on a wide grid (3.8% inside the quote range); the exact Double Heston pricer showed zero violations in the surrogate's arbitrage test [results/pinn/README.md, section 4].

## A15. The trading test

A "win" is one option trade (bought if the model says it is cheap, sold if it says it is rich; the 5 most-traded contracts a day within 5% of the forward, 5 to 45 days to expiry) that makes a profit at the next close. 41 to 45 of 100 trades profit
(a coin flip would give 50), because the model mostly says "buy" and bought options lose a day of time value; the average is positive only because wins are larger than losses. The edge was positive at the signal day's close, and was **not shown to survive entering one day later**
[results/tables/trading_summary.md, COMPARISON_1.0_vs_2.0.md]. It is a backtest on closing prints, not quotes you can deal at. It is not investment advice.

---

# PART B. How to read every figure in `results/figures/`

Every figure exists as a 300 dpi PNG and a PDF with the same name. The list with sources is `results/figures/INDEX.md`. In all of them the unit of error is the **volatility point** (percentage point of implied volatility) and lower
is better, unless the paragraph says otherwise. "Test period" means 2022 to 2026. "Sampled test days" means every 10th trading day of the test period, 118 days, used for the expensive per-day experiments.

### 01_summary_panel
Six panels on one page. Top left: bars of the mean next-day forecast error on the test period, with 95% intervals (yesterday's smile 1.47, time-scaled smile 1.24, Heston + misfit 1.42, diffusion DH + misfit 1.38, jump model + misfit 1.03). Top middle: the
waterfall from the 1.0 recipe (5.03) to 1.10, with a dotted line at the time-scaled smile. Top right: same-day fit error against SVI for all quotes, near the money and in the wings. Bottom left: seconds per daily calibration, exact 5.85 s versus PINN-accelerated
0.73 s with the same RMSE (2.555 versus 2.560). Bottom middle: correlation with India VIX (solid bars levels, hatched bars daily changes in the test years). Bottom right: key numbers. Good looks like: model bars shorter than the benchmark bars. What it shows: only the jump model
with the misfit is clearly shorter than the best benchmark; the diffusion models are longer. Use it as the opening slide.

### 02_error_by_year
x axis: calendar year; y axis: the average next-day forecast error in that year. Four lines: yesterday's smile, time-scaled smile, the jump model alone, the jump model plus misfit (thick orange). The grey band is the test period. Good looks like: the orange line below the others in every year. It is below the time-scaled smile in
all 11 years (2016 and 2026 are partial years) [results/tables/forecast_by_year.csv]. Errors of every method peak in 2020 (the COVID crash; the time-scaled smile 3.01, jump + misfit 2.69). The jump model alone (blue) is not consistently better than the time-scaled smile
(grey): it is better in some years and worse in others.

### 03_model_comparison
Horizontal bars: mean next-day error on the test period for each model, with 95% bootstrap intervals; the dashed vertical line is the best benchmark (time-scaled smile, 1.236). Hatched bars are the model alone, solid bars are the model plus yesterday's misfit. What it shows: Heston alone 3.532 and diffusion DH alone 2.792 are far
above the line; adding the misfit brings them to 1.416 and 1.377, still above the line; only the jump model (1.201 alone, 1.034 with the misfit) gets to or below it. The interval of the jump model alone (1.053 to 1.372) overlaps the benchmark's, which is why that row is called a tie.

### 04_waterfall_dh1_to_dh2
Two waterfall charts on the 118 sampled test days: left is same-day (in-sample) error, right is next-day (out-of-sample) error. Each bar steps from the previous level; blue/orange means the fix helped, red means it made things worse. Steps: the 1.0 recipe (DH1), clean data, IV-space objective,
global multi-start with the kappa ordering, the structure/state walk-forward, Merton jumps, yesterday's misfit. The numbers under the labels are the changes. What it shows: three fixes are large (IV objective -1.05, multi-start -1.04, jumps -1.66 on the next-day side); the walk-forward step is **red** (+0.08 next day,
+0.13 same day), so the split does not help; the dashed and dotted lines are the benchmarks (1.59 and 1.33 on these days). Note that on these 118 days the model reaches the benchmarks only after the jump step.

### 05_market_vs_model_smiles
Nine small plots: three real test days (low, middle, high volatility: 2023-11-28, 2022-02-01, 2022-05-02) times three expiries each. Dots are market implied vols; the green curve is the diffusion Double Heston, orange is the jump model, the dashed blue is SVI. x axis: ln(K/F); y axis: implied vol in %.
Good looks like: the curve running through the dots. What it shows: at short expiries (9-10 days) the green diffusion curve is far too flat in the wings while the orange one follows the market; at 86-121 days all curves are close. It is not perfect: on 2022-02-01 the orange 9-day curve sags below the quotes near the money
(its error is 2.16 vol points on that expiry [results/tables/smile_figure_days.csv]). Stray dots far below the curves are stale or erroneous closing prints. The three days are chosen by a fixed rule (percentiles of the market's own volatility), not by how well the model fits.

### 06_where_models_miss
Six heat maps (Black-Scholes, Heston, Double Heston, Double Heston with jumps, and the two walk-forward versions). Rows: days to expiry; columns: ln(K/F) from -22% to +22%; colour and number: the same-day IV error in that cell, pooled over the 118 sampled days (square-root colour scale). Good looks like: light cells. What it shows: errors are
concentrated in the wings of the shortest expiries: in the 3-7 day, deep-put cell the errors are 29.2 (Black-Scholes), 22.1 (Heston), 19.1 (Double Heston), 1.48 (with jumps). For expiries beyond 120 days all smile models are between 0.43 and 1.00. Grey cells have fewer than 40 quotes. Look at the walk-forward jump panel too: it is
worse than a fresh daily jump fit in the short wings (3.65 versus 1.48) [results/buckets/README.md].

### 07_quote_counts_by_bucket
The same grid as figure 06, but each cell shows how many quotes are in it (all 118 days pooled). Why you need it: an error number in a cell is only as trustworthy as the number of quotes behind it. What it shows: most quotes sit near the money at 15-30 days (4,128 in the central cell); the corner cells are thin (29 quotes in the 3-7 day, +12 to +22% cell, which is why it is grey in figure 06). Do not over-read the
extreme-wing cells.

### 08_calibrated_volatility_and_jumps
Top: the market's own 30-day at-the-money implied volatility (dark) against the volatility implied by the calibrated diffusion part today, sqrt(v1 + v2) (orange), 2016-2026; y axis in % a year. Bottom: the jump rate (jumps per year) of the jump model through time; it is a step function because the structural parameters are refitted every 10 days.
Good looks like: the orange line tracking the dark line. What it shows: it tracks closely, but in the March 2020 spike the orange line stops below the market peak, because the variance levels hit their upper bounds (see figure 12 and the VIX note). The jump rate is mostly between about 0.05 and 0.4 a year, spikes above 1 in 2020, and is lower and steady after 2022.

### 09_svi_vs_dh_by_day
x axis: date (every 10th test day); y axis: same-day IV RMSE of each fit. Blue: SVI (5 parameters per expiry), orange: Double Heston with jumps, green: Double Heston without jumps. Good looks like: a low line. What it shows: SVI is lowest on essentially every sampled day (mean daily RMSE 0.460), the jump model sits in the middle (0.994) and the diffusion model is highest with big spikes
(2.685 on average, peaks above 8) [results/svi/README.md]. SVI is a yardstick with far more freedom (5 parameters for every expiry, no link between expiries or days), so it should win the same-day fit.

### 10_svi_vs_dh_by_bucket
Three heat maps, SVI, jump model, diffusion model, by time to expiry (rows) and forward log-moneyness (columns), pooled over the sampled days. The colour scale is capped at 4. What it shows: SVI is 0.27-0.68 in every cell. The jump model is within about 1 vol point of SVI except at 7 days or less (1.65 to 3.24). The diffusion model has huge errors in the wings
at 7 days or less (16.76 and 17.68) and in the 8-14 day wings. Near the money and at long expiries the diffusion and jump models are similar to each other.

### 11_svi_example_smile
One expiry (30 days) on one day, 2024-05-28. Dots: market quotes; blue: SVI; green: diffusion Double Heston; orange: jump model. What it shows: SVI follows the quotes closely; the diffusion curve captures the general shape; the orange (walk-forward jump) curve is too flat in the left wing (about 30% where the market is 36%).
This is not a cherry-picked success: it shows a case where the walk-forward jump fit is visibly worse than SVI. Two stray quotes near 18% are bad prints that every method misses.

### 12_vix_timeseries
Top: India VIX (blue) with the model-implied 30-day volatility from the jump model (orange) and the diffusion model (green), 2016 to 2026, in % a year. Bottom: VIX against the surface's own at-the-money 30-day implied vol. Good looks like: the lines on top of each other. What it shows: they move together, including the spike to about 84 in March 2020;
the surface ATM line sits slightly below VIX on average (-1.24 vol points). This is a same-day sanity check, not a forecast.

### 13_vix_scatter_levels
Three scatter plots of India VIX against the model number, levels, each point a day. The dashed line is 45 degrees, the solid line the regression. Titles give R-squared and slope. Jump model R-squared 0.98, slope 1.01; diffusion 0.98 and 1.00; surface ATM 0.97 and 0.98. Good looks like: points hugging the diagonal. What it shows: a very tight link in levels (correlation 0.989).
Levels are always high-correlated because volatility is persistent, so this figure alone proves little; figure 14 is the harder test.

### 14_vix_scatter_changes
The same, for day-over-day changes instead of levels. R-squared 0.77 (slope 0.86) for the jump model, 0.76 (0.81) for the diffusion model, 0.69 (0.69) for the surface ATM. Changes are the stricter test, because you cannot get a high correlation just by both series being persistent. What it shows: the fitted models track the VIX's daily moves better
than the raw surface's ATM point does; the regression slopes below 1 mean that VIX moves a little less than the model number for a given day.

### 15_vix_lead_lag
Bars: correlation between the model's change at day t and the VIX change at day t+L, for lags L from -5 to +5. L greater than 0 means the model moves first. The dashed lines are the approximate 95% band under "no relation" (+-0.04). What it shows: a huge spike at L = 0 (about 0.88) and almost nothing at any other lag (at most about 0.11). The model does not
lead or lag VIX: it moves with it on the same day, which is expected since both are computed from the same day's option prices.

### 16_vix_gap_and_rolling_correlation
Top: the level gap (model minus VIX, vol points) through time for the jump model (orange) and the surface ATM (purple). Bottom: the rolling 120-day correlation of daily changes with VIX. What it shows: the jump model's gap stays within a few vol points of zero, with a dip in 2019 and a sharp spike in March 2020 when the model is capped by its bounds; the surface ATM gap sits around -1 to -2.
The rolling correlation of the jump model stays high, mostly above 0.8 and dips in 2019 and 2020; that of the surface ATM falls to about 0.47 in late 2023. Use it to say where the link is weaker, not just that it is strong on average.

### 17_pricer_validation
Left: the largest implied-vol difference (log scale, vol points) between the project pricer and the independent COS pricer, by time to expiry, for real chains (orange), the synthetic grid (blue), and the Lewis-versus-COS check (green); the dotted line at 0.01 vol point is the tolerance of the tests. Right: microseconds per option for each pricer on one real chain (log scale, diffusion and jump versions).
What it shows: real chains stay below about 2e-6 at every maturity; the synthetic grid is below 1e-5 except for the 121-365 day bucket, which reaches 1e-4 (the documented corner case); Lewis and COS agree to about 1e-9. On the right, the project pricer is the cheapest (about 26 and 29 microseconds per option), COS costs 2 to 8 times more, the Lewis integral about 6,000 to 8,800.

### 18_crosssection
Left: held-out error when a random 20% of the quotes (by expiry) are hidden during the fit. Right: error on expiries longer than 60 days when the fit sees only shorter ones. Bars: mean over 118 days with 95% bootstrap intervals; the dots are single days. What it shows: with jumps the interpolation error is 0.71 versus 2.49 for the diffusion model, 3.24 for Heston and 5.79
for Black-Scholes. On the maturity extrapolation the order changes: Heston 1.37, jump model 1.41, diffusion Double Heston 2.02, Black-Scholes 2.67. So the diffusion Double Heston loses to plain Heston here (33 of 118 days better, p = 4.1e-08) [results/crosssection/README.md].

### 19_identifiability
Four panels, 118 days. (a) Jacobian condition number at the optimum, box plots on a log scale (columns normalised, and scaled by bound width): none is near-singular. (b) Share of days on which each parameter sits on a bound (solid: within 0.3% of the interval; pale: within 2%): xi2 is on its bound on about 91% of days for the diffusion model, rho1 on
64%. (c) Approximate standard error divided by the width of the allowed interval (log scale): the red dashed line means "the standard error is the whole allowed range"; kappa1, theta1, rho1 and dk reach or exceed it. (d) Histogram of how many distinct parameter sets fit equally well among 16 starts: most days have one (mean 1.27 for diffusion, 1.50 with jumps).
Reading: tight v0, loose slow-factor parameters.

### 20_pinn_ablation
Bars (log scale) of the held-out IV error of the surrogate against the exact pricer: left on real held-out chains (65,465 quotes), right on parameter sets drawn from the training box. Solid: mean; hatched: 95th percentile. Variants: hybrid, data-only, physics-only, data-only with the brief's gate sqrt(tau). Error bars are one standard deviation over seeds (3 seeds for hybrid and data-only).
What it shows: data-only has the smallest error (0.049; 95th pct 0.13), hybrid is higher (0.077), the spec gate is 0.172, and physics-only is far worse (1.31, 95th pct 3.21). So the PINN does not beat the plain network on price.

### 21_pinn_data_efficiency
x axis: number of exact prices used for training (1e3 to 1e6, log scale); y axis: held-out IV error (log scale); two panels (real chains, box parameter sets). Orange: data-only, blue: hybrid; dashed lines are 95th percentiles. Good looks like: a low curve that falls fast. What it shows: both curves fall with more data and are still falling at 1e6. The hybrid is not lower anywhere
(equal at 1e3, above data-only from 1e4). The hoped-for advantage of physics when data is scarce was not found in this budget.

### 22_pinn_error_maps
Two heat maps of mean absolute IV error on a regular grid for 120 held-out parameter sets: x = ln(F/K) (negative: out-of-the-money calls, positive: out-of-the-money puts), y = maturity in days (log scale); white means the exact option is worth less than a rupee. Left data-only, right hybrid. Beware: the two colour bars differ (up to 0.14 versus 0.25).
What it shows: errors are smallest at the money and grow into the far wings, worst around 30 to 100 days at the wing edges; the hybrid map is brighter nearly everywhere.

### 23_pinn_pde_residual_maps
Three heat maps of the PDE residual (log10 of tau x |residual| / vega, in vol points; geometric mean over 120 held-out parameter sets) for data-only, hybrid and physics-only. Brighter means the equation is violated more. The black areas are below the bottom of the colour scale or not scored. What it shows: the data-only network violates the equation most (brightest), the hybrid less, and physics-only
least, which is exactly what each was trained to do. Compare with figure 20: the network with the smallest PDE residual (physics-only) has by far the largest price error. Small PDE residual is not the same as accurate price.

### 24_pinn_speed
Horizontal bars, microseconds per priced quote on a log scale, on one real chain (283 quotes, 8 expiries). Black: exact pricer; orange: surrogate (CPU or Apple GPU "mps", with or without computing the IV). One chain: exact 38.4 microseconds per quote (10.9 ms), surrogate on CPU 3.9 (1.1 ms), GPU 11.6 (3.3 ms: slower than CPU because of launch overhead). For 128 parameter sets at once: exact 42.9, CPU 2.1, GPU 0.5 microseconds
per quote (1,555 ms, 75 ms, 19.6 ms). What it shows: a modest gain for a single chain and a large one for many parameter sets, which is what a global search needs.

### 25_pinn_greeks
Scatter plots, surrogate (from automatic differentiation) on the y axis against the exact pricer (finite differences) on the x axis, for delta, gamma and dC/dv1 (the sensitivity to the first variance level), data-only (top) and hybrid (bottom), 300+ held-out quotes. The diagonal is perfect agreement. What it shows: delta lies on the diagonal for both. Gamma, the second derivative, scatters more for data-only than for hybrid (median error 2.5-6.0% versus 1.8-2.4% of the Black gamma), which
is the one area where the physics term clearly paid off.

### 26_pinn_smile_example
One held-out development day (2021-07-01), three expiries (7, 28 and 182 days). Top: smiles from the exact pricer (black), data-only (dashed orange) and hybrid (dotted blue). Bottom: surrogate minus exact in vol points. What it shows: the curves are indistinguishable by eye; the differences are within about 0.1 to 0.4 vol points, and on this day the data-only error is smaller than the hybrid's at all three expiries
(the hybrid reaches +0.43 at the right edge of the 182-day smile). A typical case, not the worst.

### 27_pinn_training_curves
Validation error (mean absolute IV error in vol points, log scale) against the Adam step for the three data-only runs (orange) and three hybrid runs (blue), measured on a validation set from the training distribution (not the real chains). Good looks like: falling and flattening. What it shows: all six curves fall over 30,000 steps and flatten; the data-only runs end lower (about 0.05 to 0.07) than the hybrid runs
(about 0.09 to 0.11). The curves are still declining slowly, which is why the README says longer training might lower all errors.

### 28_pinn_calibration_rmse
Each dot is one test day: x = same-day IV RMSE of the exact calibration, y = of the accelerated (left) or surrogate-only (right) calibration, scored with the exact pricer (log axes). Points on the diagonal are equal. Triangles are days whose exact optimum lies outside the surrogate's box. What it shows: almost all dots lie on the line (mean 2.56 versus 2.55), so the fit quality is the same. The triangles scatter
more. Remember that RMSE is not what the optimiser minimises: figure 29 shows the objective.

### 29_pinn_calibration_cost_ratio
Cumulative distribution over days of (objective cost of a method) / (cost of the exact calibration); 1 means equal, more than 1 means worse. Accelerated (orange) rises steeply at 1 and ends at about 1.11 (the worst day, +10.9%); surrogate-only (green) is spread out up to about 2.2; the control, exact polish with the same budget but without the surrogate (yellow), is in between. What it shows:
82% of days the accelerated fit is within 1% of the exact cost, only 7% for the surrogate alone. So the surrogate alone is not a calibration and the polish is essential.

### 30_pinn_calibration_time
Left: one dot per day, wall time of the exact calibration (x) against the accelerated one (y); the lines mark 3x, 6x and 12x speedups. Right: bars of the mean wall time and where it goes. Exact: 5.8 s and 1,380 exact-pricer calls; accelerated: 0.7 s and 119 calls, of which 0.12 s is the surrogate search, 0.18 s re-ranking and correction, 0.43 s the exact polish. What it shows: the accelerated
fit is faster on every one of the 118 days (the smallest speedup 4.0x, the largest 23.5x); the overall speedup is 8.0x. Most of the remaining time is the exact polish.

### 31_pinn_calibration_parameters
Left: box plots (log scale) of the per-day difference between the accelerated and the exact solution for each parameter. Right: one dot per day, the relative cost difference (y) against the largest parameter gap as a share of its allowed range (x). What it shows: the variance levels and correlations agree closely; kappa1 and theta1 differ a lot (median about 0.4 and 0.2 in log units). In the right panel, 51 of 118 days have cost within 1% and a parameter gap above 5%:
the same price quality with different parameters. This is the weak-identifiability point seen from a second angle.

### 32_pinn_calibration_rmse_over_time
x: date; y: accelerated (or surrogate-only) RMSE minus exact RMSE (symmetric log axis); above zero means worse. Triangles mark days whose exact optimum is outside the box. What it shows: accelerated dots hug zero almost always; a few days are above 0.1 (the worst is +0.52 on 2023-03-17, a triangle), and a few below zero (the accelerated fit found a better optimum than the 4-start exact fit).
The surrogate-only dots are scattered widely, again showing the polish matters.

### 33_pinn_calibration_design
The design experiment on 52 development days: x = mean wall time per day, y = mean log(cost / exact cost) (0 = as good as exact). Three panels: polish the surrogate's own best candidates; re-rank the 8 survivors by exact cost first; re-rank plus two rounds of defect correction. Lines are 1, 2 or 3 candidates polished; numbers are polish budgets (5, 10, 20, 40 evaluations). The circle marks the frozen choice. What it shows: more budget buys accuracy at a cost in time.
The chosen design is the pre-declared "cheapest design within 0.01 of the best". On the test days the plain variant turned out as good, so the re-rank did not demonstrably help; the design was kept as frozen.

---

# PART C. Glossary

- **Arbitrage**: a way to make a riskless profit; a valid price system must have none.
- **At the money (ATM)**: strike equal to the forward price.
- **Black-Scholes / Black formula**: the textbook option price from one constant volatility.
- **Block bootstrap**: resampling blocks of consecutive days to estimate how uncertain an average is when days are not independent.
- **Butterfly arbitrage**: a violation of convexity of price in strike (a "butterfly" spread would have a negative price).
- **Calibration**: choosing model parameters so model prices match market prices.
- **Calendar arbitrage**: a longer-dated option cheaper than a shorter-dated one of the same strike.
- **Characteristic function (CF)**: E[exp(iuX)], the Fourier transform of a distribution; the "fingerprint".
- **Condition number**: a measure of how ill-posed a fitting problem is; large means that some parameter combinations barely affect the fit.
- **COS method**: a pricing method that expands the density in a cosine series using the CF.
- **Development period**: 2016-2021, where all settings were chosen.
- **Diebold-Mariano test**: tests whether two forecasts have the same average loss.
- **Diffusion model**: a model without jumps.
- **Expiry / maturity**: the date an option ends.
- **Feller condition**: 2 kappa theta > xi squared; keeps the variance process above zero.
- **Forward price (F)**: the price today for delivery of the index at the expiry date.
- **Gamma**: the second derivative of the option price with respect to the index.
- **Hybrid (PINN)**: trained on exact prices plus the PDE term.
- **Identifiability**: whether the data determine the parameters uniquely.
- **Implied volatility (IV)**: the volatility that makes the Black formula match a market price.
- **In-sample / same-day**: error on the data the fit used. **Out-of-sample / next-day**: error on data the fit never saw.
- **Jacobian**: the matrix of how each model price changes with each parameter.
- **Jump (Merton)**: a sudden random move in the index price.
- **Kappa**: speed of mean reversion. **Theta**: long-run variance. **Xi (sigma)**: vol of vol. **Rho**: correlation between index and variance. **v0**: today's variance.
- **Log-moneyness**: ln(K/F).
- **Misfit carry**: today's market-minus-model IV error added to tomorrow's forecast.
- **Moneyness bucket**: a range of ln(K/F).
- **Out of the money (OTM)**: a call with strike above the forward, or a put below it.
- **PDE / PDE residual**: partial differential equation; how much a function breaks it.
- **PINN**: physics-informed neural network.
- **Put-call parity**: call - put = discount factor x (forward - strike).
- **QE scheme**: Andersen's quadratic-exponential simulation method for variance processes.
- **Smile**: the curve of implied volatility against strike.
- **Soft-L1 loss**: a loss that is quadratic for small errors and linear for large ones.
- **Surface**: all of a day's quotes (one smile per expiry).
- **Surrogate**: a cheap approximate copy of an expensive function.
- **SVI**: a 5-parameter curve for one expiry's smile.
- **Test period**: 2022-2026, used only to report.
- **Time-scaled smile**: yesterday's smile read at moneyness x sqrt(T0/T1); the strongest no-model benchmark.
- **Vega**: price change per one vol point.
- **Vol point**: one percentage point of volatility.
- **Walk-forward**: fit on days up to t, predict t+1, repeat.
- **Wilcoxon signed-rank test**: a rank test for whether paired differences have median zero.

---

# PART D. How to run it

```
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt        # Python 3.13; pinned versions
python -m dh2.run_all --config configs/nifty.yaml --list     # every stage: cached or missing, estimated cost, outputs
python -m dh2.run_all --config configs/nifty.yaml            # everything the cached data allows (stages with outputs are skipped)
python -m dh2.run_all --config configs/nifty.yaml --quick    # demo-scale run, writes ONLY to results_quick/
python3 -m pytest -q                                         # the tests
```

- **One command**: `python -m dh2.run_all --config configs/nifty.yaml`. It prints a plan with estimated costs first and asks for confirmation before anything expensive. It skips every stage whose outputs already exist, so on a finished checkout it only rebuilds tables, figures and the report in about half a minute
  (measured: 27 s for tests, tables, figures and report together) [results/PIPELINE.md].
- **Quick mode**: `--quick` runs a demo-scale pipeline in about 5.3 minutes on 2 cores (measured) on a 4-month window (2025-01-02 to 2025-04-30) with small budgets. It writes only to `results_quick/`, never to `results/`, and every output is marked as demo-scale.
  It is a smoke test, not the headline result.
- **What is cached** [results/PIPELINE.md]: the raw NSE files `data/nse_index/` (520 MB, downloaded only if missing); per-day surfaces `data/cache/surfaces.pkl` (45 MB); walk-forward fits, forecasts, trading results and ablation reruns in `data/results/` (475 MB; the full walk-forwards take hours: about
  22 hours for the whole programme [README.md]); PINN data and checkpoints (`data/cache/pinn/` 97 MB, `results/pinn/checkpoints/` 16 MB); caches of the heavy experiments in `results/_cache/`; India VIX `data/vix/india_vix.csv`.
- **What is not re-run by default**: the walk-forwards, the pricer validation, the PINN training (only with `--retrain-pinn`), the heavy per-day experiments. A forced rerun with `--force` of a walk-forward takes hours.
- **Tests**: about 170 fast tests pass in about 17 s, 14 slow ones are deselected, and 2 are expected failures (the documented pricer corner case) [results/PIPELINE.md].
- Settings: all in `configs/nifty.yaml` and `configs/pinn.yaml`; `tests/test_run_all.py` checks the config equals the code's defaults.
- The raw data (about 0.5 GB) is not in the repository; the downloaders `dh2.fetch_nse_old` and `dh2.fetch_nse` fetch it and never overwrite.

---

# PART E. The 15 hardest questions, with answers you can say aloud

Each answer starts with a one-sentence version you can say, then the evidence to back it up, then what you should admit.

### 1. Why does the PINN not beat the plain network?

**Say:** "On price accuracy the physics term did not help. The plain network that only learns from exact prices has a mean error of 0.049 vol points; the physics-informed one has 0.077. Physics helped on gamma, on the PDE residual and on convexity, so I use the plain one for speed and the hybrid for Greeks."

Evidence: real held-out chains, 65,465 quotes: data-only 0.049, hybrid 0.077, physics-only 1.307 [results/pinn/tables/ablation_summary.csv]. With only 1,000 labelled prices both are about 0.38-0.40; at a million, 0.049 against 0.077 [results/pinn/tables/data_efficiency.csv].
The hybrid is better on the second derivative (median gamma error 1.8-2.4% against 2.5-6.0%), on the PDE residual (0.36 against 0.73) and on butterfly violations (0.027% / 0.008% / 0.009% of triples against 0.24% / 1.04% / 0.24%) [results/pinn/README.md]. Why: the equation was verified correct, so this is not a bug. Here the exact pricer gives
unlimited correct labels for free, so the physics has little extra information to add; in the design probes the PDE loss was dominated by far-wing points until a gate change fixed that. Physics alone fails (1.3 vol points) because a small residual does not pin down the price. **Admit:** the training was stopped at 30,000 steps and errors were still falling, so with longer training the ranking could change;
that was not tested. And PINNs are usually justified when labels are scarce or expensive; here they are neither.

### 2. If the parameters are not identified, why does the fit matter?

**Say:** "Prices pin down the surface, not the individual parameters. Many parameter sets give nearly the same prices, so I validate the surfaces the parameters produce and I do not interpret any single parameter."

Evidence: on every one of 118 days at least one parameter is on a bound; the median standard error is 21% of a parameter's allowed range (diffusion model); kappa1, theta1, rho1 and dk have standard errors beyond the whole range on more than a quarter of days [results/identifiability/README.md]. In the accelerated-calibration study, 51 of 118 days have the same cost (within 1%)
with parameters that differ by more than 5% of their range, and the fitted surfaces differ by a median of 0.04 vol points [results/pinn_calib/README.md]. What is pinned down: today's variance levels (standard error about 1.4% of the range). **So the model is useful as a surface fitter and forecaster, not as a source of "the market's mean reversion speed".** The 1.0 research found the same
(99.1% of surfaces with several equally good parameter sets [COMPARISON_1.0_vs_2.0.md]).

### 3. Is this overfit?

**Say:** "I checked it five ways, and the test years were never used to choose anything."

(1) All settings were chosen on 2016-2021; 2022-2026 is the test [DECISIONS.md decision 1; README.md]. (2) The forecasts are walk-forward: fit on days up to t, score on t+1, never on the fitted data. (3) Hiding 20% of the quotes at random: the held-out error is within about 0.1 of the training-quote error for every model (jump model 0.705 held-out, 0.683 on the training quotes)
[results/crosssection/README.md]. (4) Fifteen reruns with different windows, refit schedules, priors and bounds give 1.033 to 1.036 for the final forecast [results/tables/robustness_runs.csv]. (5) The jump model + misfit beats the time-scaled smile in 11 of 11 calendar years, and the paired tests are significant (Wilcoxon and Diebold-Mariano, Holm-adjusted) [results/tables/forecast_by_year.csv, results/stats/tests_test_period.md].
**Admit:** some memory and jump settings landed at the edge of the grid searched [COMPARISON_1.0_vs_2.0.md section 5]; the expensive experiments use 118 days, and intervals that resample days may be slightly narrow because volatility regimes persist; test years are calmer than development years, so error levels are lower there.

### 4. Why not a 70/30 split?

**Say:** "I fixed the split before I saw any result and it is about half and half in days: 1,256 development days and 1,173 test days. Moving the line afterwards would itself be a tuning choice."

Evidence: [DECISIONS.md decision 1; results/tables/forecast_scores.csv]. The development period contains the 2020 crash, which gives the settings a hard case to be chosen on; the test period is five years long and spans several regimes (the yearly errors in figure 02). A 70/30 cut would be another arbitrary line; with time series what matters is that time only goes forward and that the test is untouched.
**Admit:** the test period is calmer than the development one (benchmark error 1.236 against 1.624), so absolute errors are not comparable between them; the paired differences are what you should look at.

### 5. How do you know there is no look-ahead?

**Say:** "Every forecast for day t+1 uses only days up to t, plus the three things every method is given for free on t+1: the forward price, the time to expiry and the strikes."

Evidence: the scoring code states this in its docstring [dh2/evalpairs.py]; the financing rate is a trailing 10-day median, so it is causal [dh2/data.py]; a regression test checks the refit schedule (after a bug where slow parameters froze for about 200 days, found and fixed; the frozen run is kept as a labelled ablation) [COMPARISON_1.0_vs_2.0.md section 4];
an independent audit of the trading test found no look-ahead [COMPARISON_1.0_vs_2.0.md]; the PINN's box was built from development-period fits only and the accelerated-calibration design was asserted in code to use days before 2022 [results/pinn/README.md, results/pinn_calib/README.md]. Same-day fits (the SVI comparison, the cross-sectional tests, the VIX link) are in-sample by definition and are labelled as such.
**Admit:** the forward on day t+1 comes from day t+1's option prices. Every method, including the benchmark, gets it equally, and it carries the index level, not the volatility smile.

### 6. Why jumps?

**Say:** "Because without them the model cannot make the steep wings of short-dated options. Jumps are the biggest single improvement in the project."

Evidence: pooled same-day error 6.13 (one volatility), 3.67 (Heston), 2.91 (Double Heston), 0.77 (with jumps) [results/buckets/README.md]; in the 3-7 day deep-put bucket 19.1 falls to 1.48; the waterfall step for jumps is -1.66 vol points next day (p = 8e-21) [results/attribution/waterfall.csv]; hidden-quote error 0.71 against 2.49 [results/crosssection/summary.csv].
Intuition: over a few days a diffusion has too little time to produce a big fall; a jump can. Jumps are an extension, not part of the brief's base model; the diffusion model is reported everywhere. **Admit:** the diffusion Double Heston alone does not beat the benchmark; and the jump model's walk-forward version is worse than a fresh daily jump fit at 3-7 days (3.65 against 1.48 in the deep-put bucket).

### 7. What does the 1.0 comparison really prove, given that DH1 is a rebuild?

**Say:** "It shows how much each fix is worth, not what 1.0's exact script would have scored. DH1 is a documented rebuild of a lost script: close in aggregate, not an exact reproduction."

Evidence: the rebuild matches 1.0's data universe exactly (same quote and expiry counts on all ten dates), but not its held-out set or Black-Scholes sigma; median held-out error 2.53 against 1.0's published 2.37 vol points (+7%), per-date correlation 0.74 [results/attribution/README.md, results/attribution/replication_check.csv]. On those terms the waterfall says: the IV-space objective (-1.05), the global multi-start with kappa ordering (-1.04) and jumps (-1.66) carry the gain; clean data alone is not significant (p = 0.26)
[results/attribution/waterfall.csv]. **Also say:** the shared-slow-parameter design was the original motivation of version 2, and the waterfall shows it does not help (+0.08, p = 0.003); I withdrew that claim [DECISIONS.md decision 11]. 1.0 and 2.0 also differ in data, instrument and metric, so the "old tests side by side" table in the comparison note has rows marked not identical [COMPARISON_1.0_vs_2.0.md].

### 8. Is the trading result an edge?

**Say:** "No, it is not shown to be. About 41 to 45 of 100 trades profit, wins are bigger than losses, and the profit mostly disappears when you enter one day later. It is a backtest on closing prices, not advice."

Evidence: [results/tables/trading_summary.md]: wins per 100 between 41.1 and 45.0 against a blind-trade share of 38.5 to 44.9; at the signal day's close a profit of +4.34 rupees per trade [+0.19, +8.98] in the best hedged case at 1% cost per side; matched trades entered one day later: hedged versions drop from +6.54 to +1.55 and from +7.23 to +1.21 rupees per trade, and none of the one-day-later averages is distinguishable from zero
[COMPARISON_1.0_vs_2.0.md section 2; metrics.json `trading.matched.*`]. Reasons for caution: closing prints are last trades, not quotes you could trade at; costs are modelled as a share of price; and "not shown to survive" is not the same as "shown not to".

### 9. Why use the exact pricer for the polish?

**Say:** "Because the surrogate's small error is harmless as a price error and harmful to an optimiser. The surrogate alone ends 18% above the exact cost; the exact polish brings it to within about 0.8%."

Evidence: surrogate-only: mean log cost ratio +0.166, only 7% of days within 1% of the exact cost; accelerated: +0.008, 82% of days within 1% [results/pinn_calib/accuracy_by_method.csv]. The surrogate's 0.05 vol-point error moves the minimiser far in the flat valleys of the objective [results/pinn_calib/README.md, section 3]. The same exact budget spent from a single exact-pricer start (the control) is 0.28 vol points worse in RMSE.
So the surrogate does the global search and the exact pricer does the last steps.

### 10. What is the Feller condition, and did you impose it?

**Say:** "It is 2 kappa theta greater than xi squared. It guarantees a variance factor never touches zero. I did not impose it, because the pricer does not need it and real fits often violate it."

The pricer's own documentation says the model is still usable when it fails [pricer/models.py, `feller_condition`]. The calibrated fast factor is non-Feller [results/pricer_validation/README.md], and xi2 sits on its upper bound on 91% of daily diffusion fits [results/identifiability/README.md]. The 1.0 recipe did impose Feller through its latent coordinates [results/attribution/README.md]; the waterfall step 3 changes this together with other things (box bounds, multi-start, ordering) and
the effect of Feller alone was not isolated. The Monte Carlo check used the QE scheme, which handles variances touching zero [results/pricer_validation/README.md]. **Admit:** a professor may say that non-Feller makes the variance process hit zero; the answer is that the price comes from the exact characteristic function, which is verified against Monte Carlo and Lewis, so the pricing is not affected; whether imposing it would improve forecasts was not tested.

### 11. What about the shortest expiries?

**Say:** "They are the model's weakest region and also where it adds the most over the benchmark. Options with fewer than 3 days to expiry are dropped, and the walk-forward jump model is clearly worse there than a fresh daily fit."

Evidence: the largest same-day jump-model errors are in the 3-7 day row; in the deep-put bucket the walk-forward jump model scores 3.65 against 1.48 for a fresh fit, and the walk-forward diffusion model 19.4 [results/buckets/README.md]. Against SVI, the jump model at 7 days or less has RMSE 1.65 to 3.24 against 0.35 to 0.54 [results/svi/README.md]. The gain over the benchmark is concentrated in the wings and at short expiries [COMPARISON_1.0_vs_2.0.md section 3].
Limits: expiry time of day (15:30) is not modelled, which matters for 1 to 3 day options [AUDIT.md section 4]; quotes are thin in the far wings (29 quotes in one corner cell) [results/figures/07_quote_counts_by_bucket.png].

### 12. How does this compare with SVI?

**Say:** "SVI fits a single day better, as it should, since it gets five free parameters for every expiry and no connection between expiries or days. It is the yardstick, not a competitor in forecasting."

Evidence: same-day RMSE on 118 days: SVI 0.502, jump model 1.155, diffusion 3.069; near the money 0.533 / 1.042 / 1.107; wings 0.513 / 1.421 / 6.235 [results/svi/README.md]. 48.8% of SVI's fitted expiries violate butterfly no-arbitrage somewhere on a wide grid (3.8% inside their quote range), so SVI's error is an optimistic reference [results/svi/README.md].
A model like Double Heston gives a consistent surface across expiries and a dynamic that can be forecast; SVI as used here is a per-day, per-expiry curve fit. My reading (not tested separately): the gap between 1.155 and 0.502 is the kind of leftover that the "+ misfit" correction carries forward. **Admit:** SVI is the better same-day smile fitter in every bucket.

### 13. Why is the model worse near the money than the time-scaled smile?

**Say:** "Near the money, yesterday's smile is already almost exact, and a model with a handful of parameters cannot beat a curve that has the day's own shape. The difference is small, 0.017 vol points, but it is statistically real."

Evidence: near the money (absolute ln(K/F) at most 0.04) the jump model + misfit scores 1.088 against 1.071 for the time-scaled smile; the model is better on only 47% of days; Wilcoxon (Holm) p = 2.85e-07 and Diebold-Mariano +9.41 [results/stats/tests_test_period.md]. The model's advantage overall (-0.202) comes from the wings and short expiries [COMPARISON_1.0_vs_2.0.md section 3].
The explanation (a restricted model versus a free curve) is plausible but was not tested separately. Say it as "my reading", not as a result.

### 14. Is an 8x speedup meaningful when the exact pricer is already fast?

**Say:** "It is real, but modest in absolute terms, and it only applies to the diffusion model. The exact pricer takes 38 microseconds a quote; the speed matters when you need many fits or many parameter sets at once."

Evidence: 5.85 s against 0.73 s per day, 8.0x [7.4, 8.7], with 11.6x fewer exact-pricer calls [results/pinn_calib/headline.json]; the speedup is capped because the exact pricer is still called 119 times a day; for 128 parameter sets at once the surrogate is 21x faster on CPU and 79x on the Apple GPU [results/pinn/tables/speed.csv]. Costs: about 0.8% higher objective, weaker on the 16% of days whose exact optimum is outside the box, and no jump model [results/pinn_calib/README.md].
**Admit:** for this project, with one index and about 1,200 test days, the exact calibration is perfectly feasible; the surrogate was built because the brief asked for it and it would matter at scale.

### 15. What would you do with more time?

**Say:** "Four things: widen and retrain the surrogate, give it the jumps, untangle what the structure/state design really does, and test trading with real bid/ask."

(1) Retrain the surrogate with a wider box (16% of exact optima lie outside it) and a longer budget, and test whether data-only still beats hybrid [results/pinn/README.md, results/pinn_calib/README.md]. (2) A PINN with a jump term so it can price the headline model [DECISIONS.md decision 3]. (3) Separate the structure/state split from the walk-forward refits (the waterfall cannot), and test independent jump fits plus misfit on the full test period [results/attribution/README.md].
(4) Fix the pricer's integration spacing for 1-year options with extreme parameters [results/pricer_validation/README.md]. (5) Extend the 118-day experiments to all test days and search wider than the grid edges. (6) A trading study with bid/ask quotes. (7) Model the exact expiry time for the shortest options.

### Three extra questions to be ready for

**E1. Is the "shared slow parameters" design your main idea?** No. It was the original motivation of version 2 and the waterfall shows it does not help: the diffusion model gets slightly worse (+0.08 vol points next day, p = 0.003) and independent jump fits forecast better (1.142) than the walk-forward jump model (1.254). The gain comes from the IV-space objective, the global multi-start with the kappa ordering and jumps; yesterday's misfit adds -0.155. It is kept only because the finished results were produced with it [results/attribution/waterfall.csv; DECISIONS.md decision 11].
(Refitting every 10 days instead of every 200 matters: 1.48 against 1.91 over the decade, but that is about refit frequency, not the split [COMPARISON_1.0_vs_2.0.md section 3].)

**E2. Is the pricer accurate?** Yes: at most 1.5e-6 vol points from an independent COS pricer on 18,071 real-chain points; Lewis and COS agree to 1.3e-9 relative; Monte Carlo agrees at all 70 points (largest |z| 1.98). One documented corner case: 1-year options with extreme parameters, up to 1.0e-4 vol points, caused by the integration spacing [results/pricer_validation/README.md].

**E3. What is "yesterday's misfit"?** The model's error on day t (market IV minus model IV), added to the model's forecast for t+1. It is worth -0.155 vol points on the 118 sampled days [results/attribution/waterfall.csv]. Without it the jump model only ties the benchmark (1.201 against 1.236, interval includes zero) [results/tables/forecast_scores.csv].

---

# PART F. Where earlier text in the repository is out of date

Read older documents with these corrections in mind (also listed in the report that accompanies this work):

- `README.md` (before this refresh) described the structure/state split as a key design element, said "near the money ... 2.0 adds nothing" (the correct statement is that the time-scaled smile is slightly *better* near the money: 1.071 against 1.088), listed 12 tests (about 170 fast tests now) and did not mention the new work.
- `dh2/struct.py` (module docstring) and `COMPARISON_1.0_vs_2.0.md` (header table, "What each version is") describe the shared-parameter split as the motivation and design of 2.0; decision 11 withdraws the claim that it helps.
- `AUDIT.md` section 4 says no jumps are the "largest single cause" of DH1's poor calibration. In the waterfall the jump step is the largest (-1.66), but the IV-space objective (-1.05) and the global multi-start (-1.04) are each almost as large.
- `COMPARISON_1.0_vs_2.0.md` and `AUDIT.md` quote 99.1% of 1.0 surfaces with several equally good parameter sets next to 2.0 numbers; the 2.0 identifiability study finds fewer distinct sets (1.27 on average, 16% of days with two or more) because the definitions differ and the kappa ordering removes duplicates [results/identifiability/README.md].
- `results/report.md` section 10 and `results/tables/trading_summary.*` list the "one day later" rows without their signal thresholds; the hedged ones use no threshold (4,525 trades) while the signal-day rows use 1.0 vol point. The matched table in `COMPARISON_1.0_vs_2.0.md` is the fair comparison.
