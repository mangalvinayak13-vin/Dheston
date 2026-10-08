"""The pricing models, on top of the project's own pricer (pricer/models.py, a copy of the earlier Streamlit site's pricer, imported unchanged).

Parameter vectors (the pricer's own order, factor 1 = slow, factor 2 = fast):
  bs      [sigma]
  heston  [v0, kappa, theta, xi, rho]
  dh      [v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2]
  dhj     dh + [lam, mu_j, delta_j]   Double Heston with Merton-style jumps in the index: lam jumps a year, log jump size
                                      normal(mu_j, delta_j^2), compensated so the forward is unchanged. The jump term is an
                                      independent factor, so its log characteristic function simply adds to the diffusion's
                                      (the same reason the pricer adds the two variance factors' logs).

Everything is priced against one Surface: its implied rate r, and each expiry's own forward (put-call parity) through its carry q.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .data import Surface, black_iv, black_price

_spec = importlib.util.spec_from_file_location("dh_pricer", Path(__file__).resolve().parents[1] / "pricer" / "models.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

NAMES = {
    "bs": ["sigma"],
    "heston": ["v0", "kappa", "theta", "xi", "rho"],
    "dh": ["v0_1", "kappa1", "theta1", "xi1", "rho1", "v0_2", "kappa2", "theta2", "xi2", "rho2"],
    "dhjs": ["v0_1", "kappa1", "theta1", "xi1", "rho1", "v0_2", "kappa2", "theta2", "xi2", "rho2", "lam", "mu_j", "delta_j"],
    "dhj": ["v0_1", "kappa1", "theta1", "xi1", "rho1", "v0_2", "kappa2", "theta2", "xi2", "rho2", "lam", "mu_j", "delta_j"],
}


def groups(S: Surface):
    """(expiry index, quote indices): the quotes that share one pricer call. Cached on the surface."""
    g = getattr(S, "_groups", None)
    if g is None:
        g = [(e, np.flatnonzero(S.e == e)) for e in range(len(S.T))]
        g = [(e, q) for e, q in g if q.size]
        S._groups = g
    return g


def _jump_log_cf(u, T, lam, mu, delta):
    kappa_bar = np.exp(mu + 0.5 * delta**2) - 1.0
    return lam * T * (np.exp(1j * u * mu - 0.5 * delta**2 * u**2) - 1.0 - 1j * u * kappa_bar)


def price(model: str, x, S: Surface, idx: np.ndarray | None = None) -> np.ndarray:
    """Model price of every quote of the surface (or of the quotes in idx), in rupees. Each expiry is priced once as calls and the
    puts follow by put-call parity, exactly what the pricer does for a put (call - exp(-rT)(F - K)), at half the cost."""
    x = np.asarray(x, float)
    out = np.zeros(S.n)
    for e, q in groups(S):
        if idx is not None:
            q = np.intersect1d(q, idx)
            if not q.size:
                continue
        K, T, F, r = S.K[q], S.T[e], S.F[e], S.r
        if model == "bs":
            out[q] = black_price(F, K, T, r, x[0], S.is_call[q])
            continue
        if model == "heston":
            call = np.atleast_1d(P.heston_price(S.spot, K, T, r, *x[:5], kind="call", q=S.q[e]))
        elif model == "dh":
            call = np.atleast_1d(P.double_heston_price(S.spot, K, T, r, *x[:10], kind="call", q=S.q[e]))
        elif model in ("dhj", "dhjs"):
            v1, k1, t1, x1, r1, v2, k2, t2, x2, r2, lam, mu, dl = x[:13]
            log_cf = lambda u: (P._log_cf_factor(u, v1, k1, t1, x1, r1, T) + P._log_cf_factor(u, v2, k2, t2, x2, r2, T)
                                + _jump_log_cf(u, T, lam, mu, dl))
            call = np.atleast_1d(P._price_from_log_cf(log_cf, S.spot, K, T, r, S.q[e], "call"))
        else:
            raise ValueError(model)
        out[q] = np.where(S.is_call[q], call, call - np.exp(-r * T) * (F - K))
    return out if idx is None else out[idx]


def model_iv(prices: np.ndarray, S: Surface) -> np.ndarray:
    """Black implied volatility of model prices (a price below Black's lowest range reads as a 0.1% volatility)."""
    iv = black_iv(np.maximum(prices, 1e-9), S.F[S.e], S.K, S.T[S.e], S.r, S.is_call)
    return np.where(np.isfinite(iv), iv, 1e-3)


def iv_error(model: str, x, S: Surface, mask: np.ndarray | None = None) -> np.ndarray:
    """Model implied volatility minus market, in volatility POINTS, per quote."""
    iv = model_iv(price(model, x, S), S)
    err = 100.0 * (iv - S.iv)
    return err if mask is None else err[mask]


def thin(S: Surface, per_expiry: int = 14) -> np.ndarray:
    """Indices of at most per_expiry quotes per expiry, spread evenly over strikes: the quotes a fit uses
    (evaluation always uses every quote)."""
    keep = []
    for e in range(len(S.T)):
        idx = np.flatnonzero(S.e == e)
        idx = idx[np.argsort(S.K[idx])]
        if idx.size > per_expiry:
            idx = idx[np.unique(np.round(np.linspace(0, idx.size - 1, per_expiry)).astype(int))]
        keep.append(idx)
    return np.sort(np.concatenate(keep))
