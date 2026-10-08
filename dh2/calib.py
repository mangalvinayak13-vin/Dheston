"""Calibration building blocks: bounds, starting points, the implied-volatility residual and a per-day fit.

The residual of a quote is (model price - market price) / vega, which is the implied-volatility error in volatility points to
first order. Fitting prices without that weighting lets the expensive at-the-money options dominate; this puts every quote
on the volatility scale (Gatheral 2006, ch. 2; the same weighting the project's own single-Heston calibrator uses).

Double Heston is fitted in an internal vector u where the fast factor's speed is kappa2 = kappa1 + dk with dk >= 0.2, so
"fast" always means faster (otherwise the two factors can swap labels and the same surface has two parameter vectors).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from . import models as M
from .data import Surface

# internal parameter bounds (lo, hi) per model
BOUNDS = {
    "bs": ([0.02], [1.0]),
    "heston": ([0.002, 0.1, 0.002, 0.02, -0.98], [0.30, 10.0, 0.25, 2.0, 0.98]),
    "dh": ([0.002, 0.05, 0.002, 0.02, -0.98, 0.002, 0.2, 0.002, 0.02, -0.98],
           [0.30, 4.0, 0.25, 1.5, 0.98, 0.30, 15.0, 0.25, 2.0, 0.98]),
}
BOUNDS["dhj"] = (BOUNDS["dh"][0] + [0.0, -0.25, 0.005], BOUNDS["dh"][1] + [8.0, 0.10, 0.25])
# dhjs: jump size (mean, sd of the log jump) fixed at dev-period values, jump RATE lam a daily state: internal vector = dh's ten + lam
BOUNDS["dhjs"] = (BOUNDS["dh"][0] + [0.0], BOUNDS["dh"][1] + [8.0])


import os as _os
if _os.environ.get("DH2_WIDE"):
    # wide bounds (DH2_WIDE=1): the default state/level bounds cap the model near 55% volatility, which cannot reach the March 2020 crash
    # (front-month at-the-money volatility above 130%); allow variance levels up to 3.0 (170% vol), long-run levels to 1.0, vol-of-vol to 3.0
    for _m in ("dh", "dhj", "dhjs"):
        _lo, _hi = BOUNDS[_m]
        _hi = list(_hi)
        for _i in (0, 5):
            _hi[_i] = 3.0
        for _i in (2, 7):
            _hi[_i] = 1.0
        for _i in (3, 8):
            _hi[_i] = 3.0
        BOUNDS[_m] = (_lo, _hi)


def _jump_shape() -> tuple[float, float]:
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "config" / "jump_shape.json"
    if f.exists():
        d = json.loads(f.read_text())
        return float(d["mu_j"]), float(d["delta_j"])
    return -0.07, 0.15


MU_J, DELTA_J = _jump_shape()
VEGA_FLOOR = 0.05          # rupees per volatility point: stops a deep-wing quote from becoming a huge residual
F_SCALE = 0.8              # volatility points: residuals beyond this count linearly (the market's stale quotes), not squared


def to_phys(model: str, u) -> np.ndarray:
    """Internal vector -> the pricer's parameter vector."""
    x = np.array(u, float)
    if model in ("dh", "dhj", "dhjs"):
        x[6] = x[1] + x[6]
    if model == "dhjs":
        x = np.concatenate([x, [MU_J, DELTA_J]])      # the pricer's 13-vector: ten diffusion parameters, lam, mu_j, delta_j
    return x


def from_phys(model: str, x) -> np.ndarray:
    u = np.array(x, float)
    if model in ("dh", "dhj", "dhjs"):
        u[6] = x[6] - x[1]
    return u[:11] if model == "dhjs" else u


def clip(model: str, u) -> np.ndarray:
    lo, hi = (np.array(b) for b in BOUNDS[model])
    return np.clip(u, lo + 1e-9, hi - 1e-9)


def residual(model: str, u, S: Surface, idx: np.ndarray) -> np.ndarray:
    return (M.price(model, to_phys(model, u), S, idx) - S.price[idx]) / np.maximum(S.vega[idx], VEGA_FLOOR)


def guess(S: Surface) -> dict:
    """Plain starting values read off the surface: variances from the at-the-money volatilities of the front and back."""
    iv = [S.atm_iv(e) for e in range(len(S.T))]
    iv = [v for v in iv if np.isfinite(v)]
    front, back = (iv[0] if iv else 0.12), (iv[-1] if iv else 0.14)
    return {"front": front, "back": back, "mean": float(np.mean(iv)) if iv else 0.13}


def starts(model: str, S: Surface, rng: np.random.Generator, n: int = 4) -> list[np.ndarray]:
    """Starting vectors: the flat-volatility corner (every Double Heston nests Black-Scholes, so a fit can never be worse than
    flat volatility), then plausible random draws."""
    g = guess(S)
    f2, b2, m2 = g["front"] ** 2, g["back"] ** 2, g["mean"] ** 2
    if model == "bs":
        return [np.array([g["mean"]])]
    if model == "heston":
        out = [np.array([f2, 2.0, b2, 0.05, 0.0])]
        for _ in range(n - 1):
            out.append(np.array([f2 * rng.uniform(0.6, 1.4), rng.uniform(0.5, 6), b2 * rng.uniform(0.6, 1.5), rng.uniform(0.2, 1.0), rng.uniform(-0.9, -0.1)]))
        return [clip(model, o) for o in out]
    base = [np.array([0.5 * f2, 0.5, 0.5 * b2, 0.05, 0.0, 0.5 * f2, 4.0, 0.5 * b2, 0.05, 0.0])]
    for _ in range(n - 1):
        base.append(np.array([0.5 * f2 * rng.uniform(0.5, 1.5), rng.uniform(0.2, 2.0), 0.5 * b2 * rng.uniform(0.5, 1.5), rng.uniform(0.1, 0.8), rng.uniform(-0.9, -0.1),
                              0.5 * f2 * rng.uniform(0.5, 1.5), rng.uniform(1.0, 10.0), 0.5 * b2 * rng.uniform(0.3, 1.5), rng.uniform(0.2, 1.4), rng.uniform(-0.9, 0.0)]))
    if model == "dhj":
        base = [np.concatenate([b, [rng.uniform(0.2, 3.0), rng.uniform(-0.1, 0.0), rng.uniform(0.02, 0.1)]]) for b in base]
    if model == "dhjs":
        base = [np.concatenate([b, [rng.uniform(0.05, 1.0), MU_J, DELTA_J]]) for b in base]
    return [clip(model, from_phys(model, b)) for b in base]


def fit_day(model: str, S: Surface, n_starts: int = 4, seed: int = 0, per_expiry: int = 14, max_nfev: int = 120,
            x0: np.ndarray | None = None) -> dict:
    """Independent fit of one day (what the project's earlier calibrators do, here in volatility space): best of several starts."""
    rng = np.random.default_rng(seed)
    idx = M.thin(S, per_expiry)
    lo, hi = (np.array(b) for b in BOUNDS[model])
    best = None
    for u0 in ([clip(model, x0)] if x0 is not None else []) + starts(model, S, rng, n_starts):
        try:
            r = least_squares(lambda u: residual(model, u, S, idx), u0, bounds=(lo, hi), loss="soft_l1", f_scale=F_SCALE,
                              x_scale=np.maximum(hi - lo, 1e-9) * 0.05, max_nfev=max_nfev)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    u = best.x
    err = M.iv_error(model, to_phys(model, u), S)
    tol = 0.003 * (hi - lo)
    return {"u": u, "x": to_phys(model, u), "cost": float(best.cost), "rmse": float(np.sqrt(np.mean(err**2))),
            "at_bound": [int(i) for i in np.flatnonzero((u <= lo + tol) | (u >= hi - tol))], "n": S.n}
