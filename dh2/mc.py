"""Monte Carlo for Double Heston: the third independent check (simulation, no characteristic function at all).

Scheme: Andersen's (2008) quadratic-exponential (QE) scheme for each variance factor, and his log-price update with the martingale
correction, for the process z = ln(S/F) in the forward measure (so the drift is exactly what makes e^z a martingale; no r or q needed):

    v_{t+dt} | v_t  from QE (moment matched non-central chi-square: quadratic branch psi <= 1.5, exponential branch otherwise)
    z_{t+dt} = z_t + sum_j [ K0*_j + K1_j v^j_t + K2_j v^j_{t+dt} ] + sqrt( sum_j [ K3_j v^j_t + K4_j v^j_{t+dt} ] ) Z
    K1 = dt/2 (kappa rho/xi - 1/2) - rho/xi,  K2 = dt/2 (kappa rho/xi - 1/2) + rho/xi,  K3 = K4 = dt (1-rho^2)/2,
    K0* = -ln E[exp(K2 v' + K4 v'/2) | v] - (K1 + K3/2) v      (analytic for QE; this makes E[e^{z_{t+dt}} | z_t] = e^{z_t} exactly)

The two variance factors are independent given the price noise being independent, so the two Gaussian price shocks add into one.
Optional Merton jumps (same convention as dh2/models.py: lam jumps a year, log size N(mu_j, delta_j^2), compensated) are simulated exactly per step.

Variance reduction: antithetic variates (second half of the paths uses -Z and 1-U of the first half) and, optionally, the exactly known
mean E[e^z] = 1 as a control variate. The standard error is computed over the pair averages (the pairs, not the individual paths, are independent).
The generator is numpy's default_rng with a fixed seed, so a run is reproducible.
"""
from __future__ import annotations

import numpy as np

PSI_C = 1.5


def _qe_step(v, kappa, theta, xi, rho, dt, zv, u_unif):
    """One QE step of one variance factor. Returns v_next, K0*, K1, K2, K3, K4 (arrays over paths for K0*; floats for the rest) and a count of
    paths where the martingale correction was not defined (those use the uncorrected K0)."""
    ek = np.exp(-kappa * dt)
    m = theta + (v - theta) * ek
    s2 = v * xi**2 * ek / kappa * (1.0 - ek) + theta * xi**2 / (2.0 * kappa) * (1.0 - ek) ** 2
    psi = s2 / (m * m)
    quad = psi <= PSI_C

    v_next = np.empty_like(v)
    # quadratic branch
    inv = 2.0 / psi[quad]
    b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
    a = m[quad] / (1.0 + b2)
    v_next[quad] = a * (np.sqrt(b2) + zv[quad]) ** 2
    # exponential branch
    ex = ~quad
    p = (psi[ex] - 1.0) / (psi[ex] + 1.0)
    beta = (1.0 - p) / m[ex]
    uu = u_unif[ex]
    v_next[ex] = np.where(uu <= p, 0.0, np.log((1.0 - p) / (1.0 - uu)) / beta)

    r_xi = rho / xi
    K1 = 0.5 * dt * (kappa * r_xi - 0.5) - r_xi
    K2 = 0.5 * dt * (kappa * r_xi - 0.5) + r_xi
    K3 = K4 = 0.5 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4
    # martingale correction: M = E exp(A v')
    lnM = np.empty_like(v)
    ok = np.ones(v.shape, dtype=bool)
    den = 1.0 - 2.0 * A * a
    okq = den > 0
    lnM[quad] = np.where(okq, A * b2 * a / np.where(okq, den, 1.0) - 0.5 * np.log(np.where(okq, den, 1.0)), 0.0)
    ok[quad] = okq
    oke = beta > A
    Me = p + beta * (1.0 - p) / np.where(oke, beta - A, 1.0)
    lnM[ex] = np.where(oke, np.log(np.where(oke, Me, 1.0)), 0.0)
    ok[ex] = oke
    K0 = np.where(ok, -lnM - (K1 + 0.5 * K3) * v, -rho * kappa * theta / xi * dt)
    return v_next, K0, K1, K2, K3, K4, int((~ok).sum())


def mc_terminal(params, T, n_pairs=100_000, n_steps=None, seed=20261008, jump=None):
    """Simulate z_T = ln(S_T/F) for 2 * n_pairs antithetic paths (first half original, second half mirrored). Returns (z, diagnostics)."""
    v0a, ka, ta, xa, ra, v0b, kb, tb, xb, rb = (float(p) for p in params)
    if n_steps is None:
        n_steps = int(max(32, np.ceil(252 * T)))
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    n = n_pairs
    va, vb = np.full(n, v0a), np.full(n, v0b)
    z = np.zeros(n)
    z_mirror = np.zeros(n)
    va_m, vb_m = va.copy(), vb.copy()
    n_bad = 0
    if jump is not None:
        lam, mu_j, dj = (float(j) for j in jump)
        kbar = np.exp(mu_j + 0.5 * dj**2) - 1.0
    for _ in range(n_steps):
        # both halves use the same underlying randoms, the mirror half with Z -> -Z and U -> 1-U
        zva, zvb, zx = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
        ua, ub = rng.random(n), rng.random(n)
        if jump is not None:
            nj = rng.poisson(lam * dt, n)
            zj = rng.standard_normal(n)
        for tag, sg, va_, vb_, zz in (("o", 1.0, va, vb, z), ("m", -1.0, va_m, vb_m, z_mirror)):
            ua_, ub_ = (ua, ub) if sg > 0 else (1.0 - ua, 1.0 - ub)
            va2, K0a, K1a, K2a, K3a, K4a, bad_a = _qe_step(va_, ka, ta, xa, ra, dt, sg * zva, ua_)
            vb2, K0b, K1b, K2b, K3b, K4b, bad_b = _qe_step(vb_, kb, tb, xb, rb, dt, sg * zvb, ub_)
            n_bad += bad_a + bad_b
            drift = K0a + K1a * va_ + K2a * va2 + K0b + K1b * vb_ + K2b * vb2
            var = K3a * va_ + K4a * va2 + K3b * vb_ + K4b * vb2
            zz += drift + np.sqrt(var) * sg * zx
            if jump is not None:
                zz += nj * mu_j + dj * np.sqrt(nj) * sg * zj - lam * dt * kbar
            if tag == "o":
                va, vb = va2, vb2
            else:
                va_m, vb_m = va2, vb2
    return np.concatenate([z, z_mirror]), dict(n_steps=n_steps, n_pairs=n, seed=seed, martingale_correction_failures=n_bad)


def mc_price(params, F, K, T, DF, kind="call", n_pairs=100_000, n_steps=None, seed=20261008, jump=None, control_variate=False, return_z=False):
    """Monte Carlo price and standard error of European options, one parameter set, K scalar or array (all strikes from the same paths).
    Returns (price, standard_error) in the units of F (and a dict of diagnostics). Standard error from the n_pairs pair-averaged payoffs."""
    z, diag = mc_terminal(params, T, n_pairs, n_steps, seed, jump)
    Ka = np.atleast_1d(np.asarray(K, float))
    x = np.log(F / Ka)[:, None]
    ez = np.exp(z)[None, :]
    pay = np.maximum(np.exp(x) * ez - 1.0, 0.0) if kind == "call" else np.maximum(1.0 - np.exp(x) * ez, 0.0)   # normalised: payoff / K
    half = z.size // 2
    pairs = 0.5 * (pay[:, :half] + pay[:, half:])
    if control_variate:
        cv = 0.5 * (ez[:, :half] + ez[:, half:]) - 1.0                      # exact mean zero
        beta = np.array([np.cov(pairs[i], cv[0])[0, 1] for i in range(pairs.shape[0])]) / cv[0].var(ddof=1)
        pairs = pairs - beta[:, None] * cv
    u = pairs.mean(axis=1)
    se = pairs.std(axis=1, ddof=1) / np.sqrt(half)
    price, err = DF * Ka * u, DF * Ka * se
    diag["mean_exp_z"] = float(np.exp(z).mean())              # should be 1 within sampling error
    out = (price if np.ndim(K) else float(price[0]), err if np.ndim(K) else float(err[0]), diag)
    return out + (z,) if return_z else out
