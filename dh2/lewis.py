"""Lewis (2001) single-integral option pricing, adaptive quadrature: the second independent check of the Double Heston price.

Same characteristic function as dh2/cos.py (log_cf_double, written separately from pricer/models.py), but a completely different inversion:
no cosine series, no fixed grid. With X = ln(F/K), z = ln(S_T/F) and phi the characteristic function of z, the normalised call
u = C/(K DF) is

    u_call = e^X - e^{X/2} / pi * I,        I = int_0^inf Re[ e^{i u X} phi(u - i/2) ] / (u^2 + 1/4) du,

(the contour at Im = 1/2 in the middle of the strip of existence, so one real integral and no singularity). Check: for Black-Scholes
phi(u - i/2) = exp(-w (u^2 + 1/4)/2) with w = sigma^2 T, which reproduces the Black formula.

Out-of-the-money options are tiny differences of O(1) numbers in that form, so by default the integrand is a DIFFERENCE from the Black-Scholes
integrand with the same total variance c2 (closed-form price, whose integral is analytic), I_model - I_BS, which is small and keeps the
relative accuracy of far out-of-the-money prices:

    u_model = u_BS(w = c2) - e^{X/2}/pi * (I_model - I_BS).

The integration range [0, U] stops where |phi(u - i/2)|/(u^2+1/4) has fallen under 1e-18 (found by scanning, no fixed cut), and
scipy.integrate.quad (adaptive Gauss-Kronrod) does the integral, one strike at a time. Slow by design: it is a referee, not a production pricer.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from .cos import cumulants, log_cf_double


def _bs_normalised(X, w, kind):
    """Black price / (K DF) for total variance w = sigma^2 T and X = ln(F/K)."""
    sw = np.sqrt(w)
    d1 = (X + 0.5 * w) / sw
    d2 = d1 - sw
    if kind == "call":
        return np.exp(X) * ndtr(d1) - ndtr(d2)
    return ndtr(-d2) - np.exp(X) * ndtr(-d1)


def _upper_limit(phi_half, tiny=1e-18):
    """Smallest power-of-two multiple of 1 where |phi(u - i/2)|/(u^2 + 1/4) is below tiny and stays below at the next two doublings."""
    u = 1.0
    below = 0
    while u < 2.0**22:
        val = abs(phi_half(u)) / (u * u + 0.25)
        below = below + 1 if val < tiny else 0
        if below >= 3:
            return u / 2.0**2
        u *= 2.0
    return u


def lewis_price(params, F, K, T, DF, kind="call", jump=None, control_variate=True, epsabs=1e-15, epsrel=1e-12, limit=2000, return_info=False):
    """Lewis-formula price of European calls/puts, one parameter set, K array or scalar, same units as F. Adaptive quad per strike."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    params = [float(v) for v in params]
    Ka = np.atleast_1d(np.asarray(K, float))
    X = np.log(F / Ka)
    c2 = float(cumulants(T, params, jump)[1][0])

    def phi_half(u):
        return np.exp(log_cf_double(u - 0.5j, T, params, jump))

    U = _upper_limit(phi_half)
    bs_phi = lambda u: np.exp(-0.5 * c2 * (u * u + 0.25))
    out = np.empty(Ka.size)
    n_eval = 0
    err_est = 0.0
    for i, x in enumerate(X):
        def f(u):
            nonlocal n_eval
            n_eval += 1
            ph = phi_half(u)
            if control_variate:
                ph = ph - bs_phi(u)
            return float((np.exp(1j * u * x) * ph).real / (u * u + 0.25))
        # break the range into pieces of at most ~ 4 oscillations of exp(i u x) or 256 units so that quad cannot step over structure
        n_piece = max(1, int(np.ceil(max(U / 256.0, U * abs(x) / (8 * np.pi)))))
        edges = np.linspace(0.0, U, n_piece + 1)
        tot = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            val, err = quad(f, lo, hi, epsabs=epsabs, epsrel=epsrel, limit=limit)
            tot += val
            err_est += err
        if control_variate:
            out[i] = _bs_normalised(x, c2, kind) - np.exp(0.5 * x) / np.pi * tot     # the same correction serves call and put
        else:
            call = np.exp(x) - np.exp(0.5 * x) / np.pi * tot
            out[i] = call if kind == "call" else call - np.expm1(x)
    price = DF * Ka * out
    price = price if np.ndim(K) else float(price[0])
    if return_info:
        return price, dict(U=U, n_eval=n_eval, quad_error_estimate=err_est, c2=c2)
    return price
