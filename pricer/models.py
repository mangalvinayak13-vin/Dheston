"""
models.py
=========
All pricing and volatility maths for the app. Plain functions, NumPy/SciPy only,
no Streamlit -- so every function here can be unit-tested (see tests/test_models.py).

The app's story is that each model relaxes one assumption of the one before it:

    Black-Scholes / GBM   volatility is a constant
    Heston                volatility is random and mean-reverting, and it moves
                          with the stock price (rho)
    Double Heston         volatility has a fast AND a slow random component

Conventions used everywhere below
    S      spot price                 K   strike
    T      time to expiry, in years   r   continuously-compounded risk-free rate
    sigma  annualised volatility      q   continuous dividend yield (default 0)
    kind   "call" or "put"
    Heston variance v = sigma**2; kappa, theta, xi, rho, v0 are explained at heston_paths.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.stats import norm


# ===========================================================================
# 1. Black-Scholes: price, Greeks, implied volatility
# ===========================================================================

def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes price of a European option. K may be a single number or an array."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    K = np.asarray(K, dtype=float)

    if T <= 0 or sigma <= 0:
        # No randomness left, so the option is worth its discounted payoff at the forward price.
        # (T = 0 gives plain intrinsic value, max(S - K, 0).)
        fwd_minus_k = S * np.exp(-q * T) - K * np.exp(-r * T)
        price = np.maximum(fwd_minus_k if kind == "call" else -fwd_minus_k, 0.0)
    else:
        d1, d2 = _d1_d2(S, K, T, r, sigma, q)
        if kind == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return float(price) if price.ndim == 0 else price


def greeks(S, K, T, r, sigma, kind="call", q=0.0):
    """
    Black-Scholes Greeks for one option, as a dict. Units are the ones traders quote:
        delta  change in price per $1 move in the stock
        gamma  change in delta per $1 move in the stock
        vega   change in price per 1 volatility POINT (sigma up by 0.01)
        theta  change in price per calendar DAY (usually negative for a long option)
        rho    change in price per 1 percentage POINT move in r
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    # Gamma and vega are the same for calls and puts.
    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T) / 100.0
    decay = -S * disc_q * pdf_d1 * sigma / (2.0 * np.sqrt(T))  # time decay common to both

    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2) / 100.0
    elif kind == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2) / 100.0
    else:
        raise ValueError("kind must be 'call' or 'put'")

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta / 365.0), "rho": float(rho)}


def implied_vol(price, S, K, T, r, kind="call", q=0.0):
    """
    The sigma that makes black_scholes(...) equal `price`, found with Brent's method.

    Black-Scholes price rises steadily with sigma, so there is at most one answer, and
    brentq is guaranteed to find it once we have a sigma below and a sigma above it.
    Returns nan when no answer exists (price below intrinsic value, above the upper
    bound, or T <= 0) -- callers should drop those points rather than plot them.
    """
    lo, hi = 1e-4, 5.0  # 0.01% to 500% volatility
    if T <= 0:
        return np.nan
    price_lo = black_scholes(S, K, T, r, lo, kind, q)
    price_hi = black_scholes(S, K, T, r, hi, kind, q)
    if not (price_lo < price < price_hi):
        return np.nan
    return brentq(lambda s: black_scholes(S, K, T, r, s, kind, q) - price, lo, hi, xtol=1e-10)


# ===========================================================================
# 2. Monte Carlo path simulation
# ===========================================================================
# Antithetic variates: for every batch of random shocks Z we also run -Z. The two
# paths are negatively correlated, so their average has lower variance than two
# independent paths would. Layout: the first half of the paths uses +Z, the second
# half uses the matching -Z, so path i and path i + n/2 are an antithetic pair.
# price_from_paths relies on that layout to compute an honest standard error.

def _even(n_paths, antithetic):
    """Antithetic pairs need an even number of paths."""
    return n_paths + (n_paths % 2) if antithetic else n_paths


def _normals(rng, shape, antithetic):
    """Standard normals of the given shape; if antithetic, second half = minus the first half."""
    if not antithetic:
        return rng.standard_normal(shape)
    half = rng.standard_normal((shape[0] // 2,) + tuple(shape[1:]))
    return np.concatenate([half, -half])


def gbm_paths(S, r, sigma, T, n_paths, n_steps, seed, antithetic=True, q=0.0):
    """
    Geometric Brownian Motion paths under the risk-neutral measure. Returns an array of
    shape (n_paths, n_steps + 1); column 0 is S. n_paths is rounded up to even if antithetic.

    GBM has an exact solution, so there is no discretisation error:
        S(t + dt) = S(t) * exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z)
    The -sigma^2/2 is the Ito correction: it makes E[S(T)] = S e^{(r-q)T}, i.e. the
    stock grows at r on average, which is what no-arbitrage pricing requires.
    """
    n_paths = _even(n_paths, antithetic)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = _normals(rng, (n_paths, n_steps), antithetic)
    log_steps = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return S * np.exp(log_paths)


def _variance_factor_step(v, kappa, theta, xi, rho, dt, rng, antithetic):
    """
    One Full-Truncation-Euler step for ONE stochastic-variance factor. Returns
        v_next        the variance at the next step (may be slightly negative, see below)
        v_pos         the truncated variance max(v, 0) used during this step
        price_shock   this factor's contribution to the stock's log-return shock

    The variance follows dv = kappa (theta - v) dt + xi sqrt(v) dW. Two problems for Euler:
      * sqrt(v) needs v >= 0. The true process stays >= 0 only if the Feller condition
        (2 kappa theta > xi^2) holds; real stocks usually violate it. And even when Feller
        holds, a discrete Euler step can overshoot below zero, and sqrt(negative) = nan,
        which then poisons every later step of that path.
      * "Naive Euler" fixes this by clipping v to 0 after each step -- that biases the
        variance upward and misprices options.
    Full truncation (Lord, Koekkoek & van Dijk 2010) is the standard fix: keep the raw v
    (it may dip negative), but use v_pos = max(v, 0) EVERYWHERE v is read -- in the drift,
    in the sqrt, and in the price step. When v < 0, v_pos = 0, so the diffusion switches off
    and the drift kappa*theta*dt pulls v back up. No nan, and the bias is small.
    """
    n = v.shape[0]
    z_v = _normals(rng, (n,), antithetic)                      # shock to variance
    z_indep = _normals(rng, (n,), antithetic)                  # independent noise
    z_price = rho * z_v + np.sqrt(1.0 - rho**2) * z_indep      # shock to price, corr(z_price, z_v) = rho

    v_pos = np.maximum(v, 0.0)
    sqrt_v_pos = np.sqrt(v_pos)
    v_next = v + kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * np.sqrt(dt) * z_v
    return v_next, v_pos, sqrt_v_pos * np.sqrt(dt) * z_price


def heston_paths(S, r, T, v0, kappa, theta, xi, rho, n_paths, n_steps, seed, antithetic=True, q=0.0):
    """
    Heston stock paths (risk-neutral), Full Truncation Euler. Shape (n_paths, n_steps + 1).

        dS = (r - q) S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,      corr(dW1, dW2) = rho

    v0     today's variance (vol^2)          kappa  how fast v is pulled back to theta
    theta  long-run variance                 xi     "vol of vol": how violently v itself moves
    rho    correlation of price and variance shocks. Negative rho means price drops come
           with volatility rises (the leverage effect) -- this is what tilts the smile.

    The stock uses the log form  log S += (r - q - v/2) dt + sqrt(v dt) Z, so S stays positive.
    The loop below is over TIME steps only; each step is vectorised across all paths.
    """
    n_paths = _even(n_paths, antithetic)
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    log_S = np.full(n_paths, np.log(S))
    v = np.full(n_paths, float(v0))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S

    for step in range(n_steps):
        v, v_pos, shock = _variance_factor_step(v, kappa, theta, xi, rho, dt, rng, antithetic)
        log_S += (r - q - 0.5 * v_pos) * dt + shock
        paths[:, step + 1] = np.exp(log_S)
    return paths


def double_heston_paths(S, r, T, v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2,
                        n_paths, n_steps, seed, antithetic=True, q=0.0):
    """
    Double Heston paths: two INDEPENDENT variance factors, each a Heston variance process
    with its own (v0, kappa, theta, xi, rho). Total variance of the stock is v1 + v2, so
        log S += (r - q - (v1 + v2)/2) dt + shock1 + shock2
    Typically factor 1 is fast (large kappa: short-lived spikes) and factor 2 is slow.
    """
    n_paths = _even(n_paths, antithetic)
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    log_S = np.full(n_paths, np.log(S))
    v1 = np.full(n_paths, float(v0_1))
    v2 = np.full(n_paths, float(v0_2))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S

    for step in range(n_steps):
        v1, v1_pos, shock1 = _variance_factor_step(v1, kappa1, theta1, xi1, rho1, dt, rng, antithetic)
        v2, v2_pos, shock2 = _variance_factor_step(v2, kappa2, theta2, xi2, rho2, dt, rng, antithetic)
        log_S += (r - q - 0.5 * (v1_pos + v2_pos)) * dt + shock1 + shock2
        paths[:, step + 1] = np.exp(log_S)
    return paths


def price_from_paths(paths, K, r, T, kind="call", antithetic=True):
    """
    Monte Carlo option price from simulated paths: the discounted average payoff at expiry.
    Returns (price, standard_error).

    With antithetic paths, path i and path i + n/2 are NOT independent, so we first average
    each pair and take the standard error across the n/2 pair-averages. Using all n paths as
    if independent would overstate the error.
    """
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0) if kind == "call" else np.maximum(K - S_T, 0.0)
    discounted = np.exp(-r * T) * payoff

    if antithetic:
        half = len(discounted) // 2
        samples = 0.5 * (discounted[:half] + discounted[half:])
    else:
        samples = discounted
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples)))


def feller_condition(kappa, theta, xi):
    """
    True if 2 kappa theta > xi^2. Then the variance can never touch zero. When it fails
    the model is still usable (that is why we simulate with full truncation), it just
    means v occasionally hits zero.
    """
    return bool(2.0 * kappa * theta > xi**2)


# ===========================================================================
# 3. Semi-analytic Heston / Double Heston prices (used for the smile and calibration)
# ===========================================================================
# Monte Carlo is noisy, and calibration needs thousands of prices. Heston has a
# closed-form characteristic function, so we can price by numerical integration
# instead: fast and noise-free. Monte Carlo above is the independent cross-check.

_CF_FLOOR = 1e-8    # treat the characteristic function as zero once it is this small
_DU = 0.5           # spacing of the integration grid (see _integration_grid)
_PROBES = np.array([75.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0])


@lru_cache(maxsize=16)
def _integration_points(u_max):
    """Midpoint-rule points and weights on [0, u_max]. Cached so repeated calls reuse the array."""
    u = (np.arange(int(u_max / _DU)) + 0.5) * _DU
    return u, np.full(u.shape, _DU)


def _integration_grid(log_cf):
    """
    Choose how far to integrate, and with what spacing.

    HOW FAR. The price integrals run to u = infinity, so we stop where the characteristic
    function has died away. That distance DEPENDS ON THE PARAMETERS and cannot be a fixed
    number: Heston's characteristic function decays roughly like exp(-c*u), and c shrinks as
    the total variance v0*T shrinks. For a short-dated option on a calm stock it is still
    around 1e-2 at u = 300, and stopping there leaves a tail big enough to swamp the price
    of a far out-of-the-money option -- which is how a "price" of -0.01 appears for an option
    genuinely worth 1e-9. So probe increasing values of u and stop at the first one where
    both the plain and the shifted characteristic function have fallen below _CF_FLOOR.

    WHAT SPACING. An evenly spaced midpoint rule, not anything cleverer. The integrand
    oscillates like exp(-i*u*ln(K/F)); over the strikes we plot, ln(K/F) is at most about
    0.22, so one oscillation takes at least 28 units of u and a spacing of 0.5 samples it
    roughly 56 times. Measured against a much finer grid, that holds the error near 1e-7 of
    a dollar. Gauss-Legendre nodes would be marginally more accurate per point, but
    generating thousands of them costs seconds, which is far too slow for calibration.
    """
    size = np.maximum(np.abs(np.exp(log_cf(_PROBES))), np.abs(np.exp(log_cf(_PROBES - 1j))))
    decayed = np.flatnonzero(np.isfinite(size) & (size < _CF_FLOOR))
    return _integration_points(float(_PROBES[decayed[0]] if decayed.size else _PROBES[-1]))


def _log_cf_factor(u, v0, kappa, theta, xi, rho, T):
    """
    log of one variance factor's contribution to the characteristic function of
    ln(S_T / forward):  C(u) + D(u) * v0,  using the "little trap" form (Albrecher et al.
    2007). The original 1993 Heston formula takes a complex log that can jump across a
    branch cut for long maturities, silently giving wrong prices; writing it with
    g = (c - d)/(c + d) keeps the log's argument well behaved.
    """
    xi = max(xi, 1e-3)  # C and D divide by xi^2; below ~1e-3 rounding error takes over
    c = kappa - rho * xi * 1j * u
    d = np.sqrt(c**2 + xi**2 * (1j * u + u**2))
    g = (c - d) / (c + d)
    e = np.exp(-d * T)
    C = (kappa * theta / xi**2) * ((c - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (c - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    return C + D * v0


def _price_from_log_cf(log_cf, S, K, T, r, q, kind):
    """
    Price a European option from the log characteristic function of ln(S_T / F), F = forward.
    Gil-Pelaez inversion:  call = e^{-rT} [F * P1 - K * P2]  where P1 and P2 are
    probabilities, each written as a 1-D integral over u. Working with ln(K/F) instead of
    ln K makes the integrand oscillate slowly, so a grid of nodes is accurate.
    K may be an array: the grid is reused for every strike.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)[:, None]                           # (n_strikes, 1)

    U, W = _integration_grid(log_cf)
    phi = np.exp(log_cf(U))[None, :]                     # E[exp(iu ln(S_T/F))]
    phi_shift = np.exp(log_cf(U - 1j))[None, :]          # shifted argument gives the share-measure probability
    osc = np.exp(-1j * k * U[None, :]) / (1j * U[None, :])

    P2 = 0.5 + (np.real(osc * phi) @ W) / np.pi
    P1 = 0.5 + (np.real(osc * phi_shift) @ W) / np.pi

    call = np.exp(-r * T) * (F * P1 - K * P2)
    # Floor at the no-arbitrage bound. A call is never worth less than the discounted
    # forward intrinsic value, and never less than zero. Deep out of the money the true
    # price is a rounding error away from zero, so leftover integration error could still
    # make it slightly negative; this guarantees a price we can hand to a student.
    call = np.maximum(call, np.maximum(np.exp(-r * T) * (F - K), 0.0))
    price = call if kind == "call" else call - np.exp(-r * T) * (F - K)   # put-call parity
    return float(price[0]) if price.size == 1 else price


def heston_price(S, K, T, r, v0, kappa, theta, xi, rho, kind="call", q=0.0):
    """Heston European option price (semi-analytic). K may be an array."""
    return _price_from_log_cf(lambda u: _log_cf_factor(u, v0, kappa, theta, xi, rho, T),
                              S, K, T, r, q, kind)


def double_heston_price(S, K, T, r, v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2,
                        kind="call", q=0.0):
    """
    Double Heston price. The two factors are independent, and independent random variables
    multiply their characteristic functions, so the LOGS simply add.
    """
    def log_cf(u):
        return (_log_cf_factor(u, v0_1, kappa1, theta1, xi1, rho1, T)
                + _log_cf_factor(u, v0_2, kappa2, theta2, xi2, rho2, T))
    return _price_from_log_cf(log_cf, S, K, T, r, q, kind)


# ===========================================================================
# 4. Calibration: fitting model parameters to market option prices
# ===========================================================================

# Allowed range for each Heston parameter. The Pricing page's sliders use the same ranges,
# so a calibrated value can always be shown on its slider.
HESTON_BOUNDS = {"v0": (0.0005, 1.0), "kappa": (0.1, 10.0), "theta": (0.0005, 1.0),
                 "xi": (0.05, 2.0), "rho": (-0.99, 0.99)}


def _group_by_maturity_and_kind(maturities, kinds):
    """Indices of quotes sharing a maturity and call/put type, so each group is priced in one vectorised call."""
    groups = {}
    for i, key in enumerate(zip(maturities, kinds)):
        groups.setdefault(key, []).append(i)
    return [(T, kind, np.array(idx)) for (T, kind), idx in groups.items()]


def vega_weights(S, r, strikes, maturities, kinds, market_ivs, floor=1e-2):
    """
    One weight per quote: its Black-Scholes vega at the quote's own market implied volatility,
    per volatility POINT. Dividing a price residual by vega converts it into an approximate
    IMPLIED-VOLATILITY residual, because vega is exactly the derivative of price with respect
    to volatility.

    Why this matters. An unweighted price fit is dominated by whatever is expensive: an
    at-the-money option costs rupees while a far out-of-the-money one costs paise, so the
    wings contribute almost nothing to the objective and the fit ignores the part of the
    smile that carries the skew. Weighting by vega puts every quote on the same volatility
    scale, which is also the scale the smile chart is drawn on. Measured during development
    on SPY chains, before this project moved to NSE, it cut the implied-volatility fit error
    by about a third; that gain comes from the weighting, not from that particular market.

    The floor stops a near-zero vega (deep out of the money, or very short dated) from turning
    one quote into an arbitrarily large residual.
    """
    strikes = np.asarray(strikes, float)
    return np.array([max(greeks(S, k, T, r, iv, kind)["vega"] * 100.0, floor)
                     for k, T, iv, kind in zip(strikes, maturities, market_ivs, kinds)])


def calibrate_heston(S, r, strikes, maturities, kinds, market_prices, x0, weights=None):
    """
    Least-squares fit of Heston (v0, kappa, theta, xi, rho) to market option prices,
    minimising the sum of squared weighted residuals (model - market) / weight, from x0.

    Pass weights=vega_weights(...) to fit in implied-volatility space rather than dollars;
    that is what the app does, and it is standard practice (Gatheral 2006, ch. 2).
    Leave it None to weight every quote equally in dollars.

    Returns a dict: params (tuple), rmse (dollars per option, always unweighted so it stays
    comparable across weighting schemes), weighted_rmse, converged, message, and at_bounds
    (parameters that ended on the edge of their range, which means the data does not identify
    them -- widening the range of maturities is usually what fixes this).
    Raises if the optimiser cannot evaluate the starting point; the caller decides how to recover.
    """
    strikes, market_prices = np.asarray(strikes, float), np.asarray(market_prices, float)
    groups = _group_by_maturity_and_kind(list(maturities), list(kinds))
    names = list(HESTON_BOUNDS)
    lo = np.array([HESTON_BOUNDS[n][0] for n in names])
    hi = np.array([HESTON_BOUNDS[n][1] for n in names])
    w = np.ones(len(market_prices)) if weights is None else np.asarray(weights, float)

    def price_errors(x):
        model = np.empty(len(market_prices))
        for T, kind, idx in groups:
            model[idx] = heston_price(S, strikes[idx], T, r, *x, kind=kind)
        return model - market_prices

    x0 = np.clip(np.asarray(x0, float), lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo))
    result = least_squares(lambda x: price_errors(x) / w, x0, bounds=(lo, hi),
                           x_scale="jac", max_nfev=300)

    tol = 1e-3 * (hi - lo)   # "on the edge" = within 0.1% of the range from either end
    at_bounds = [name for name, v, a, b, t in zip(names, result.x, lo, hi, tol) if v <= a + t or v >= b - t]
    return {"params": tuple(float(v) for v in result.x),
            "rmse": float(np.sqrt(np.mean(price_errors(result.x)**2))),
            "weighted_rmse": float(np.sqrt(np.mean(result.fun**2))),
            "converged": bool(result.success) and bool(np.isfinite(result.fun).all()),
            "message": str(result.message),
            "at_bounds": at_bounds}


def implied_vol_rmse(S, r, strikes, maturities, kinds, market_ivs, model_prices, min_price=0.01):
    """
    Fit error in ANNUALISED VOLATILITY POINTS: invert each model price back to a
    Black-Scholes implied volatility and compare with the market's. This is the honest
    metric for a smile fit, because a dollar error means something different on a $20
    at-the-money option than on a $0.10 wing. Quotes the model prices below min_price
    cannot be inverted and are skipped; the fraction used is returned alongside.
    """
    model_ivs = np.array([implied_vol(p, S, k, T, r, kind) if p >= min_price else np.nan
                          for p, k, T, kind in zip(model_prices, strikes, maturities, kinds)])
    ok = ~np.isnan(model_ivs)
    if not ok.any():
        return float("nan"), 0.0
    return float(100 * np.sqrt(np.mean((model_ivs[ok] - np.asarray(market_ivs)[ok])**2))), float(ok.mean())


def calibrate_black_scholes(S, r, strikes, maturities, kinds, market_prices):
    """
    Best single flat volatility for the same quotes (one parameter, same least-squares loss).
    Returns (sigma, rmse). It is the yardstick for calibrate_heston: Heston has five
    parameters, so it must beat this by a real margin to justify them.
    """
    strikes, market_prices = np.asarray(strikes, float), np.asarray(market_prices, float)
    groups = _group_by_maturity_and_kind(list(maturities), list(kinds))

    def rmse_for(sigma):
        model = np.empty(len(market_prices))
        for T, kind, idx in groups:
            model[idx] = black_scholes(S, strikes[idx], T, r, sigma, kind)
        return float(np.sqrt(np.mean((model - market_prices) ** 2)))

    best = minimize_scalar(rmse_for, bounds=(0.01, 3.0), method="bounded")
    return float(best.x), float(best.fun)


# ===========================================================================
# 5. Volatility forecasting
# ===========================================================================
# Every forecaster below returns a pandas Series of DAILY volatility, indexed by the
# ORIGIN date t and using returns up to and including day t only. Its value at t is
# the forecast of average daily volatility over the days AFTER t. That common alignment
# lets the page compare them directly against forward_realised_vol.
# Returns are log returns as decimals (0.01 = 1%). Multiply by sqrt(252) to annualise.

def naive_vol(returns, window=21):
    """Baseline: tomorrow's vol = today's vol, where 'today's vol' is the trailing `window`-day realised vol."""
    return np.sqrt((returns**2).rolling(window).mean())


def ewma_vol(returns, lam=0.94):
    """
    RiskMetrics exponentially weighted vol:  var_t = lam * var_{t-1} + (1 - lam) * r_t^2.
    Recent days count more; lam = 0.94 is the RiskMetrics default for daily data. The
    forecast is flat: the EWMA has no long-run level to revert to, so it predicts today's
    estimate for every future day.
    """
    return np.sqrt((returns**2).ewm(alpha=1.0 - lam, adjust=False).mean())


def forward_realised_vol(returns, horizon=21):
    """What actually happened: realised daily vol over the `horizon` days after each date (nan at the end)."""
    return np.sqrt((returns**2).rolling(horizon).mean().shift(-horizon))


def garch_vol(returns, horizon=21, n_train=None):
    """
    GARCH(1,1) volatility forecast. Returns (forecast_series, info).

        var_t = omega + alpha * r_{t-1}^2 + beta * var_{t-1}

    Unlike EWMA it has a long-run variance omega / (1 - alpha - beta) that forecasts revert
    to. The parameters are fitted by maximum likelihood with the `arch` package on the first
    `n_train` returns ONLY (all of them if None), then held fixed and run over the whole
    series, so the forecasts after n_train never use the future.

    Averaging the 1..horizon-step-ahead variance forecasts (each pulls back toward the
    long-run level at rate alpha + beta per day) gives the forecast series.

    info has the fitted omega, alpha, beta (returns as decimals, daily variance units),
    'long_run_var_daily', and 'converged'.
    """
    from arch import arch_model  # imported here so the rest of this file works without `arch`

    pct = returns * 100.0  # arch is numerically happier with returns in percent
    fit_data = pct if n_train is None else pct.iloc[:n_train]
    spec = dict(mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    fitted = arch_model(fit_data, **spec).fit(disp="off")
    omega, alpha, beta = fitted.params["omega"], fitted.params["alpha[1]"], fitted.params["beta[1]"]

    # Re-run the recursion over ALL data with the parameters frozen at their training values.
    cond_var = arch_model(pct, **spec).fix(fitted.params).conditional_volatility ** 2   # var for day t, known at t-1
    next_var = omega + alpha * pct**2 + beta * cond_var                                # var for day t+1, known at t

    persistence = alpha + beta
    long_run = omega / (1.0 - persistence)
    # The k-step-ahead variance is  long_run + persistence^(k-1) * (next_var - long_run).
    # Averaging k = 1..horizon uses the geometric-series sum (1 - p^H) / (1 - p). arch keeps
    # alpha + beta < 1, so the denominator is safe.
    avg_decay = (1.0 - persistence**horizon) / (1.0 - persistence)
    avg_var = long_run + (next_var - long_run) * avg_decay / horizon

    info = {"omega": omega / 1e4, "alpha": alpha, "beta": beta,
            "long_run_var_daily": long_run / 1e4,
            "converged": bool(fitted.convergence_flag == 0)}
    return np.sqrt(avg_var) / 100.0, info
