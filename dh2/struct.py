"""The 2.0 calibrator: slow STRUCTURAL parameters shared across a window of days, fast STATE parameters fitted daily.

Why. The project's own study found that one day's option prices pin down only today's volatility level (v0); the speeds of mean
reversion, the long-run levels, the vol-of-vol and the price/volatility correlation are only weakly identified from one day, so
independent daily fits wander (99% of surfaces had several equally good parameter sets, 3.9x further apart than random ones) and
the parameters carried to the next day price worse than yesterday's smile. Those parameters describe HOW the market's volatility
behaves, not what it is today, so they should not change day to day. 2.0 therefore estimates them jointly over a window of recent
days (one set of structural parameters, one pair of volatility levels per day) and then, for each day, fits only the daily state.
That is the "variable projection" split: a few parameters that need many days, a few that need one.

  model   structural (shared over the window)                         state (one per day)
  bs      -                                                           sigma
  heston  kappa, theta, xi, rho                                        v0
  dh      kappa1, theta1, xi1, rho1, dk, theta2, xi2, rho2            v0_1, v0_2
  dhj     dh's eight + lam, mu_j, delta_j (the jumps)                  v0_1, v0_2

The window fit can be pulled toward a prior (MAP): the previous window's estimate (persistence), or what worked in the past when
the market looked like today (pattern memory, memory.py). The prior is a Gaussian on a transformed parameter (log for positive
ones, the value for rho and the jump mean), with a strength in "equivalent residuals".
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import calib, models as M
from .data import Surface

# (structural indices in the internal vector u, state indices)
IDX = {"bs": ([], [0]), "heston": ([1, 2, 3, 4], [0]), "dh": ([1, 2, 3, 4, 6, 7, 8, 9], [0, 5]),
       "dhj": ([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], [0, 5]), "dhjs": ([1, 2, 3, 4, 6, 7, 8, 9], [0, 5, 10])}
# how each structural entry is measured by the prior: ('log', scale) or ('lin', scale), in the order of IDX[model][0]
_KIND = {1: ("log", 0.7), 2: ("log", 0.7), 3: ("log", 0.7), 4: ("lin", 0.3), 6: ("log", 0.7), 7: ("log", 0.7), 8: ("log", 0.7),
         9: ("lin", 0.3), 10: ("lin", 2.0), 11: ("lin", 0.05), 12: ("log", 0.6)}


def assemble(model: str, eta: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Internal vector u from structural eta and state."""
    ei, si = IDX[model]
    n = len(ei) + len(si)
    u = np.empty(n if model != "bs" else 1)
    u[ei] = eta
    u[si] = state
    return u


def split(model: str, u: np.ndarray):
    ei, si = IDX[model]
    return np.asarray(u)[ei].copy(), np.asarray(u)[si].copy()


def _bounds(model: str):
    lo, hi = (np.array(b, float) for b in calib.BOUNDS[model])
    ei, si = IDX[model]
    return lo[ei], hi[ei], lo[si], hi[si]


def prior_residual(model: str, eta: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    """Gaussian prior on the structural parameters about mu, one residual per parameter, strength lam (equivalent quotes)."""
    ei, _ = IDX[model]
    out = np.empty(len(ei))
    for k, j in enumerate(ei):
        kind, s = _KIND[j]
        a, b = (np.log(max(eta[k], 1e-8)), np.log(max(mu[k], 1e-8))) if kind == "log" else (eta[k], mu[k])
        out[k] = np.sqrt(lam) * (a - b) / s
    return out


def fit_window(model: str, surfaces: list[Surface], eta0: np.ndarray, state0: np.ndarray, weights: np.ndarray | None = None,
               prior: tuple[np.ndarray, float] | None = None, per_expiry: int = 10, max_nfev: int = 25) -> dict:
    """Jointly fit one set of structural parameters and a state per day over the window of surfaces.

    state0 has one row per day. Returns eta, states (D x n_state), the mean squared weighted residual and the data residual RMSE."""
    ei, si = IDX[model]
    nE, nS, D = len(ei), len(si), len(surfaces)
    lo_e, hi_e, lo_s, hi_s = _bounds(model)
    idxs = [M.thin(S, per_expiry) for S in surfaces]
    w = np.ones(D) if weights is None else np.asarray(weights, float)
    sizes = [len(ix) for ix in idxs]
    n_prior = nE if prior is not None else 0
    n_res = sum(sizes) + n_prior
    z0 = np.concatenate([eta0, np.asarray(state0, float).ravel()])
    lo = np.concatenate([lo_e, np.tile(lo_s, D)]); hi = np.concatenate([hi_e, np.tile(hi_s, D)])
    z0 = np.clip(z0, lo + 1e-9, hi - 1e-9)

    def fun(z):
        eta, st = z[:nE], z[nE:].reshape(D, nS)
        parts = [np.sqrt(w[d]) * calib.residual(model, assemble(model, eta, st[d]), surfaces[d], idxs[d]) for d in range(D)]
        if prior is not None:
            parts.append(prior_residual(model, eta, prior[0], prior[1]))
        return np.concatenate(parts)

    sp = lil_matrix((n_res, nE + nS * D), dtype=int)
    r0 = 0
    for d in range(D):
        sp[r0:r0 + sizes[d], :nE] = 1
        for j in range(nS):
            sp[r0:r0 + sizes[d], nE + d * nS + j] = 1
        r0 += sizes[d]
    if n_prior:
        sp[r0:, :nE] = 1
    scale = np.maximum(hi - lo, 1e-9) * 0.05
    r = least_squares(fun, z0, bounds=(lo, hi), jac_sparsity=sp, loss="soft_l1", f_scale=calib.F_SCALE, x_scale=scale,
                      max_nfev=max_nfev)
    eta, st = r.x[:nE], r.x[nE:].reshape(D, nS)
    data = fun(r.x)[:sum(sizes)]
    return {"eta": eta, "states": st, "cost": float(r.cost), "rmse": float(np.sqrt(np.mean(data ** 2))), "nfev": int(r.nfev)}


def fit_state(model: str, S: Surface, eta: np.ndarray, state0: np.ndarray, per_expiry: int = 14, max_nfev: int = 40,
              prior: tuple[np.ndarray, float] | None = None) -> dict:
    """The day's state (volatility levels) given the structural parameters. prior = (expected state, strength): a Gaussian in
    log variance about the expected value (the persistence forecast), strength in equivalent quotes."""
    ei, si = IDX[model]
    _, _, lo_s, hi_s = _bounds(model)
    idx = M.thin(S, per_expiry)

    def fun(st):
        r = calib.residual(model, assemble(model, eta, st), S, idx)
        if prior is not None:
            r = np.concatenate([r, np.sqrt(prior[1]) * (np.log(np.maximum(st, 1e-8)) - np.log(np.maximum(prior[0], 1e-8))) / 0.5])
        return r

    s0 = np.clip(state0, lo_s + 1e-9, hi_s - 1e-9)
    best = None
    cands = [s0] + ([s0 * 0.8, s0 * 1.25] if model != "bs" else [])
    for c in cands:
        r = least_squares(fun, np.clip(c, lo_s + 1e-9, hi_s - 1e-9), bounds=(lo_s, hi_s), loss="soft_l1", f_scale=calib.F_SCALE,
                          x_scale=np.maximum(hi_s - lo_s, 1e-9) * 0.05, max_nfev=max_nfev)
        if best is None or r.cost < best.cost:
            best = r
    u = assemble(model, eta, best.x)
    err = M.iv_error(model, calib.to_phys(model, u), S)
    return {"state": best.x, "u": u, "x": calib.to_phys(model, u), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def expected_state(model: str, eta: np.ndarray, state: np.ndarray, dt_years: float) -> np.ndarray:
    """The model's own forecast of tomorrow's variance levels: each factor reverts toward its long-run level theta at speed kappa,
    E[v(t+dt)] = theta + (v - theta) exp(-kappa dt). (Black-Scholes has no dynamics: sigma carries.)"""
    if model == "bs":
        return state.copy()
    if model == "heston":
        kappa, theta = eta[0], eta[1]
        return np.array([theta + (state[0] - theta) * np.exp(-kappa * dt_years)])
    k1, th1, k2, th2 = eta[0], eta[1], eta[0] + eta[4], eta[5]
    out = [th1 + (state[0] - th1) * np.exp(-k1 * dt_years), th2 + (state[1] - th2) * np.exp(-k2 * dt_years)]
    return np.array(out + ([state[2]] if model == "dhjs" else []))        # the jump rate has no dynamics of its own: it carries
