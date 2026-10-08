"""PINN-accelerated calibration of the diffusion Double Heston (model 'dh'): surrogate for the GLOBAL search, exact pricer for the POLISH.

The problem is the one of dh2.calib.fit_day, unchanged: internal vector u = (v0_1, kappa1, theta1, xi1, rho1, v0_2, dk, theta2, xi2, rho2)
with kappa2 = kappa1 + dk, the quotes of dh2.models.thin(S, 14), residual (model price - market price) / max(vega, 0.05) (implied-vol points
to first order), scipy's soft-L1 loss with f_scale 0.8, the calibrator's bounds.  The methods differ only in WHERE the optimiser starts and
WHICH pricer evaluates the early iterations:

  1. GLOBAL (surrogate).  Scrambled-Sobol starting points in the intersection of the calibrator's bounds and the surrogate's training box
     (results/pinn/README.md: the surrogate is only valid inside that box), optimised all at once by a batched, bounded, damped Gauss-Newton
     (Levenberg-Marquardt) iteration in torch on the CPU.  The Jacobian is the autograd derivative of the surrogate (one backward pass for
     every start and quote), the loss is scipy's soft-L1 with its own robust-loss Hessian weights.  Successive halving: a few iterations for
     all starts, more for the best, most for the best of those.  Quotes the surrogate cannot price (tau > 0.6 y, |ln F/K| > 0.3) are left
     out of the global stage ONLY.
  1b. (optional, chosen on development days) RE-RANK / CORRECT.  The surrogate's minimiser can sit where its small error matters (flat valleys), so
     the survivors can be ranked by their exact cost (one exact residual evaluation each, cfg.rerank) and refined by a few rounds of additive
     defect correction (cfg.n_correct, correct()); both use exact residual evaluations but never an exact Jacobian.
  2. POLISH (exact).  The best few distinct surrogate solutions are the starting points of scipy's least_squares on the exact Fourier
     pricer, with the identical objective, bounds and scaling as dh2.calib.fit_day and a small function-evaluation budget.  The polish
     can leave the surrogate's box (it is bounded only by the calibrator's bounds), and it only ever accepts steps that lower the exact cost.

Every reported fit metric is computed with the exact pricer (score()); the surrogate never appears in a metric.  Exact-pricer calls are counted
by a wrapper around the residual, finite-difference Jacobian columns included (scipy's own nfev does not count them).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import qmc

from . import calib, models as M
from .data import Surface
from .pinn.surrogate import Surrogate

MODEL = "dh"
NAMES = ["v0_1", "kappa1", "theta1", "xi1", "rho1", "v0_2", "dk", "theta2", "xi2", "rho2"]
LOG_PARAM = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], bool)       # sampled / compared on a log scale (the correlations linearly)
FS = calib.F_SCALE
# surrogate theta = (kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2) <- internal u indices (kappa2 = u[1] + u[6])
_U_OF_THETA = {0: 1, 1: 2, 2: 3, 3: 4, 5: 7, 6: 8, 7: 9}
_SMALL = 1e-12


# ------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class PinnConfig:
    """Everything the accelerated method is free to choose.  The values in DEFAULT were frozen on development-period days only
    (exp_pinn_calib.py design stage, results/pinn_calib/design_*.csv)."""
    n_starts: int = 128                  # Sobol starting points (a power of two keeps the sequence balanced)
    iters: tuple = (6, 10, 20)           # batched LM iterations in stage 1, 2, 3 ...
    keep: tuple = (32, 8)                # survivors after stage 1, 2 ... (len(keep) == len(iters) - 1)
    n_polish: int = 2                    # distinct surrogate solutions polished with the exact pricer
    polish_nfev: int = 20                # scipy max_nfev of each polish (a Jacobian costs 10 further pricer calls on top)
    min_sep: float = 0.05                # two candidates are "distinct" if some parameter differs by > this share of its bound width
    start_mode: str = "data"             # 'data': v0 and theta ranges scaled to the day's at-the-money variance; 'box': whole box
    per_expiry: int = 14                 # quotes per expiry (identical to dh2.calib.fit_day)
    seed: int = 0
    lm_lambda0: float = 1e-2             # initial Marquardt damping (relative to the diagonal of J'WJ)
    rerank: bool = False                 # rank the surrogate's survivors by their EXACT cost (one exact evaluation each) before choosing whom to polish
    n_correct: int = 0                   # defect-correction rounds between the global stage and the polish (0 = none); see correct()
    correct_inner: int = 15              # batched-LM iterations on the corrected surrogate in each round
    correct_rho: float = 0.25            # initial trust radius of a round, as a share of each parameter's calibrator bound width


DEFAULT = PinnConfig()


# ------------------------------------------------------------------ bounds
def box_bounds(sur: Surrogate) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of u for the GLOBAL stage: the calibrator's bounds intersected with the surrogate's training box."""
    lo, hi = (np.array(b, float) for b in calib.BOUNDS[MODEL])
    d = sur.domain
    tlo, thi = np.array(d["theta_lo"], float), np.array(d["theta_hi"], float)
    for k, j in _U_OF_THETA.items():
        lo[j] = max(lo[j], tlo[k]); hi[j] = min(hi[j], thi[k])
    for j in (0, 5):
        lo[j] = max(lo[j], float(d["v_lo"][0 if j == 0 else 1])); hi[j] = min(hi[j], float(d["v_hi"][0 if j == 0 else 1]))
    # kappa2 = kappa1 + dk must stay inside [theta_lo[4], theta_hi[4]]
    lo[6] = max(lo[6], tlo[4] - hi[1]); hi[6] = min(hi[6], thi[4] - lo[1])
    return lo, hi


def in_box(u: np.ndarray, sur: Surrogate, tol: float = 1e-9) -> np.ndarray:
    """True where the internal vector(s) u lie inside the global-stage box (the surrogate's valid region for the parameters)."""
    lo, hi = box_bounds(sur)
    u = np.atleast_2d(u)
    k2 = u[:, 1] + u[:, 6]
    d = sur.domain
    return np.all((u >= lo - tol) & (u <= hi + tol), 1) & (k2 >= d["theta_lo"][4] - tol) & (k2 <= d["theta_hi"][4] + tol)


# ------------------------------------------------------------------ the exact objective (what is minimised and reported)
def exact_residual(u, S: Surface, idx: np.ndarray) -> np.ndarray:
    return calib.residual(MODEL, u, S, idx)


def soft_l1_cost(r) -> float:
    """scipy's cost for loss='soft_l1', f_scale=0.8:  f_scale^2 * sum(sqrt(1 + (r/f_scale)^2) - 1)."""
    z = (np.asarray(r, float) / FS) ** 2
    return float(FS ** 2 * np.sum(np.sqrt(1.0 + z) - 1.0))


def exact_cost(u, S: Surface, idx: np.ndarray) -> float:
    return soft_l1_cost(exact_residual(u, S, idx))


def score(u, S: Surface, idx: np.ndarray | None = None) -> dict:
    """Metrics of a solution with the EXACT pricer: same-day implied-vol RMSE in vol points over every quote, the objective's cost, bound hits."""
    idx = M.thin(S, 14) if idx is None else idx
    lo, hi = (np.array(b) for b in calib.BOUNDS[MODEL])
    u = np.asarray(u, float)
    err = M.iv_error(MODEL, calib.to_phys(MODEL, u), S)
    tol = 0.003 * (hi - lo)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "cost": exact_cost(u, S, idx),
            "at_bound": [int(i) for i in np.flatnonzero((u <= lo + tol) | (u >= hi - tol))]}


class Counter:
    """Wraps a residual function and counts calls (finite-difference Jacobian columns included)."""

    def __init__(self, fun):
        self.fun, self.n = fun, 0

    def __call__(self, u):
        self.n += 1
        return self.fun(u)


# ------------------------------------------------------------------ the surrogate objective (batched, with autograd Jacobian)
class SurrogateObjective:
    """Residuals and Jacobians of the calibration objective, for B parameter vectors at once, from the surrogate.

    Only the quotes of `idx` that the surrogate can price are used (|ln F/K| <= x_max and tau_lo <= tau <= tau_hi); the others are counted in
    n_excluded.  Residual = (K DF u_sur - [put] DF (F - K) - market price) / max(vega, 0.05), the exact objective's residual with the exact price
    replaced by the surrogate price."""

    def __init__(self, sur: Surrogate, S: Surface, idx: np.ndarray):
        self.sur = sur
        d = sur.domain
        e = S.e[idx]
        T, F, K = S.T[e], S.F[e], S.K[idx]
        x = np.log(F / K)
        ok = (np.abs(x) <= d["x_max"]) & (T >= d["tau_lo"] - 1e-12) & (T <= d["tau_hi"])
        self.ok = ok                       # which quotes of idx the surrogate can price
        self.offset = None                 # optional (B, n) additive correction to the residuals (defect correction)
        self.n_excluded = int((~ok).sum())
        self.n = int(ok.sum())
        df = np.exp(-S.r * T)
        f32 = lambda a: torch.as_tensor(np.asarray(a, float)[ok], dtype=torch.float32)
        self.x, self.tau, self.kd = f32(x), f32(T), f32(K * df)
        self.put = f32(np.where(S.is_call[idx], 0.0, df * (F - K)))
        self.px = f32(S.price[idx])
        self.w = f32(1.0 / np.maximum(S.vega[idx], calib.VEGA_FLOOR))
        self.device = sur.device
        self.calls = 0                     # (start, quote) surrogate evaluations, forward or forward+backward
        self.x, self.tau, self.kd, self.put, self.px, self.w = (a.to(self.device) for a in (self.x, self.tau, self.kd, self.put, self.px, self.w))

    def _price(self, ur):
        """ur (N, 10) -> surrogate prices (N,) for N rows, row i using quote i % n of the surface."""
        v1, v2 = ur[:, 0], ur[:, 5]
        th = torch.stack([ur[:, 1], ur[:, 2], ur[:, 3], ur[:, 4], ur[:, 1] + ur[:, 6], ur[:, 7], ur[:, 8], ur[:, 9]], 1)
        return th, v1, v2

    def eval(self, U, jac: bool = True, rows=None):
        """U (B, 10) float64 numpy -> r (B, n) and, if jac, J (B, n, 10) = dr/du, both float64 numpy.  If an offset is set, row b of U is
        the offset's row rows[b] (rows defaults to all rows in order)."""
        B = U.shape[0]
        n = self.n
        Ut = torch.as_tensor(np.asarray(U, float), dtype=torch.float32, device=self.device)
        ur = Ut[:, None, :].expand(B, n, 10).reshape(B * n, 10).clone()
        if jac:
            ur.requires_grad_(True)
        th, v1, v2 = self._price(ur)
        rep = lambda a: a.repeat(B)
        with torch.set_grad_enabled(jac):
            uu = self.sur.model(rep(self.x), v1, v2, rep(self.tau), th)
            r = (rep(self.kd) * uu - rep(self.put) - rep(self.px)) * rep(self.w)
        self.calls += B * n
        rn = r.detach().reshape(B, n).double().cpu().numpy()
        if self.offset is not None:
            rn = rn + (self.offset if rows is None else self.offset[rows])
        if not jac:
            return rn, None
        (g,) = torch.autograd.grad(r.sum(), ur)
        return rn, g.reshape(B, n, 10).double().cpu().numpy()


def _robust(r):
    """(cost, gradient weights, Hessian weights) of scipy's soft-L1: cost = fs^2 sum(sqrt(1+z)-1), z = (r/fs)^2; the Gauss-Newton matrix is
    J' diag((1+z)^-1.5) J (rho' + 2 rho'' r^2) and the gradient J' ((1+z)^-0.5 r)."""
    z = (r / FS) ** 2
    s = np.sqrt(1.0 + z)
    return FS ** 2 * np.sum(s - 1.0, axis=1), 1.0 / s, 1.0 / s ** 3


def batched_lm(obj: SurrogateObjective, U0: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_iter: int, lam0: float = 1e-2, xs: np.ndarray | None = None,
               tol: float = 1e-7) -> dict:
    """Bounded Levenberg-Marquardt on B independent problems at once (variables scaled by xs = 0.05 * calibrator bound width, as in dh2.calib).

    Each iteration solves (J'HJ + lam diag(J'HJ)) d = -J'g on the variables that are not held at a bound by the gradient, steps to
    clip(u + xs d, lo, hi) and keeps the step only if the cost fell (Nielsen's rule updates lam).  A start stops when a step changes the cost by
    less than tol (relative) or the step is tiny.  Returns the final U, cost (scipy's soft-L1 cost on the surrogate residuals) and iteration counts."""
    B = U0.shape[0]
    if xs is None:
        xs = np.maximum(np.array(calib.BOUNDS[MODEL][1]) - np.array(calib.BOUNDS[MODEL][0]), 1e-9) * 0.05
    lo = np.broadcast_to(lo, (B, 10)).copy()                      # bounds may differ from row to row (trust regions)
    hi = np.broadcast_to(hi, (B, 10)).copy()
    U = np.clip(U0, lo + 1e-9, hi - 1e-9)
    r, J = obj.eval(U)
    cost, gw, hw = _robust(r)
    lam = np.full(B, lam0)
    nu = np.full(B, 2.0)
    active = np.ones(B, bool)
    its = np.zeros(B, int)
    eye = np.eye(10)
    for _ in range(n_iter):
        a = np.flatnonzero(active)
        if a.size == 0:
            break
        Js = J[a] * xs                                           # scaled variables
        A = np.einsum("bqi,bq,bqj->bij", Js, hw[a], Js)
        grad = np.einsum("bqi,bq->bi", Js, gw[a] * r[a])
        Ua, lo_a, hi_a = U[a], lo[a], hi[a]
        blocked = ((Ua <= lo_a + 1e-9) & (grad > 0)) | ((Ua >= hi_a - 1e-9) & (grad < 0))
        free = ~blocked
        Af = A * (free[:, :, None] & free[:, None, :])
        diag = np.einsum("bii->bi", Af)
        dg = np.maximum(diag, 1e-9 * diag.max(1, keepdims=True) + 1e-300)
        Af = Af + lam[a][:, None, None] * dg[:, :, None] * eye + (~free)[:, :, None] * eye
        rhs = -np.where(free, grad, 0.0)
        try:
            d = np.linalg.solve(Af, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:                                           # extremely rare: the whole batch then takes a short steepest step
            d = rhs * 1e-3
        Un = np.clip(Ua + d * xs, lo_a, hi_a)
        dn = (Un - Ua) / xs
        rn, Jn = obj.eval(Un, rows=a)
        cn, gwn, hwn = _robust(rn)
        pred = -(np.einsum("bi,bi->b", grad, dn) + 0.5 * np.einsum("bi,bij,bj->b", dn, A, dn))
        gain = cost[a] - cn
        acc = np.isfinite(cn) & (gain > 0)
        rho = np.where(pred > 0, gain / np.maximum(pred, 1e-300), -1.0)
        ia = a[acc]
        U[ia], r[ia], J[ia], cost[ia], gw[ia], hw[ia] = Un[acc], rn[acc], Jn[acc], cn[acc], gwn[acc], hwn[acc]
        lam[ia] *= np.maximum(1.0 / 3.0, 1.0 - (2.0 * rho[acc] - 1.0) ** 3); nu[ia] = 2.0
        ib = a[~acc]
        lam[ib] *= nu[ib]; nu[ib] *= 2.0
        its[a] += 1
        step_small = np.max(np.abs(dn), 1) < 1e-6
        conv = (acc & (gain <= tol * np.maximum(cost[a], 1e-12))) | step_small | (lam[a] > 1e12)
        active[a[conv]] = False
    return {"U": U, "cost": cost, "iters": its}


# ------------------------------------------------------------------ starts
def sobol_starts(S: Surface, lo: np.ndarray, hi: np.ndarray, n: int, seed: int, mode: str = "data") -> np.ndarray:
    """n scrambled-Sobol points in [lo, hi] (log scale for the positive parameters, linear for the correlations).  mode 'data' narrows the
    variance-like parameters (v0_1, v0_2, theta1, theta2) to 0.03-1.5 times the day's largest at-the-money variance, mode 'box' uses the whole box."""
    lo, hi = lo.copy(), hi.copy()
    if mode == "data":
        g = calib.guess(S)
        ref = max(g["front"] ** 2, g["back"] ** 2, g["mean"] ** 2)
        for j in (0, 5, 2, 7):
            lo[j] = min(max(lo[j], 0.03 * ref), hi[j] * 0.999)
            hi[j] = max(min(hi[j], 1.5 * ref), lo[j] * 1.001)
    q = qmc.Sobol(d=10, scramble=True, seed=seed).random(n)
    a = lo.copy(); b = hi.copy()
    a[LOG_PARAM] = np.log(lo[LOG_PARAM]); b[LOG_PARAM] = np.log(hi[LOG_PARAM])
    z = a + q * (b - a)
    z[:, LOG_PARAM] = np.exp(z[:, LOG_PARAM])
    return np.clip(z, lo, hi)


def _distinct(U: np.ndarray, order: np.ndarray, k: int, width: np.ndarray, sep: float) -> list[int]:
    """Indices (into U, in the given order) of up to k candidates that differ pairwise by more than sep of some parameter's bound width."""
    out: list[int] = []
    for i in order:
        if all(np.max(np.abs(U[i] - U[j]) / width) > sep for j in out):
            out.append(int(i))
            if len(out) == k:
                break
    return out


# ------------------------------------------------------------------ stage 1: the global search
def global_search(sur: Surrogate, S: Surface, idx: np.ndarray, cfg: PinnConfig, obj: SurrogateObjective | None = None) -> dict:
    """Many starts, batched LM with successive halving.  Returns the surviving solutions sorted by surrogate cost (best first)."""
    lo, hi = box_bounds(sur)
    obj = obj or SurrogateObjective(sur, S, idx)
    U = sobol_starts(S, lo, hi, cfg.n_starts, cfg.seed, cfg.start_mode)
    n_start = len(U)
    survivors = list(cfg.keep) + [None]
    cost = None
    for stage, (it, k) in enumerate(zip(cfg.iters, survivors)):
        res = batched_lm(obj, U, lo, hi, it, cfg.lm_lambda0)
        U, cost = res["U"], res["cost"]
        order = np.argsort(cost)
        if k is not None:
            order = order[:k]
        U, cost = U[order], cost[order]
    return {"U": U, "cost": cost, "obj": obj, "n_start": n_start, "n_excluded": obj.n_excluded, "n_global_quotes": obj.n}


# ------------------------------------------------------------------ stage 1b: defect correction (optional, cfg.n_correct > 0)
def correct(sur: Surrogate, S: Surface, idx: np.ndarray, obj: SurrogateObjective, U0: np.ndarray, cfg: PinnConfig) -> dict:
    """Additive defect correction ("space mapping") of the surrogate's solutions, a few exact residual evaluations and NO exact Jacobian.

    The surrogate's error is small and varies slowly with the parameters, but a valley of the objective is flat, so the surrogate's minimiser can sit
    where its small error matters (the exact cost there can be several times the surrogate's).  For each candidate u_k one exact residual vector
    r_e(u_k) is computed; the surrogate residual is shifted by the constant d = r_e(u_k) - r_s(u_k), so the corrected surrogate equals the exact
    residual at u_k and keeps the surrogate's (autograd) derivatives; it is minimised in a trust region around u_k (batched LM, inside the
    surrogate's box).  The new point is accepted only if the EXACT cost fell (one more exact evaluation); the radius grows after a success and
    halves after a failure.  Quotes the surrogate cannot price (tau > 0.6 y) keep their exact value at u_k and are held constant in the inner solve.
    Returns the candidates sorted by exact cost."""
    lo_b, hi_b = box_bounds(sur)
    width = np.array(calib.BOUNDS[MODEL][1]) - np.array(calib.BOUNDS[MODEL][0])
    B = len(U0)
    U = np.clip(U0, lo_b, hi_b)
    R = [exact_residual(U[b], S, idx) for b in range(B)]
    cost = np.array([soft_l1_cost(r) for r in R])
    n_calls = B
    rho = np.full(B, cfg.correct_rho)
    active = np.ones(B, bool)
    for _ in range(cfg.n_correct):
        a = np.flatnonzero(active)
        if a.size == 0:
            break
        obj.offset = None
        rs, _j = obj.eval(U[a], jac=False)
        obj.offset = np.stack([R[b][obj.ok] for b in a]) - rs
        tlo = np.maximum(lo_b, U[a] - rho[a, None] * width)
        thi = np.minimum(hi_b, U[a] + rho[a, None] * width)
        res = batched_lm(obj, U[a], tlo, thi, cfg.correct_inner, cfg.lm_lambda0)
        obj.offset = None
        for j, b in enumerate(a):
            un = res["U"][j]
            rn = exact_residual(un, S, idx)
            n_calls += 1
            cn = soft_l1_cost(rn)
            if cn < cost[b] * (1.0 - 1e-9):
                rel = (cost[b] - cn) / cost[b]
                U[b], R[b], cost[b] = un, rn, cn
                rho[b] = min(rho[b] * 1.5, 0.5)
                if rel < 1e-4:
                    active[b] = False
            else:
                rho[b] *= 0.5
                if rho[b] < 5e-3:
                    active[b] = False
    order = np.argsort(cost)
    return {"U": U[order], "cost": cost[order], "n_calls": n_calls}


# ------------------------------------------------------------------ stage 2: the exact polish
def polish(S: Surface, idx: np.ndarray, u0: np.ndarray, max_nfev: int) -> dict:
    """scipy least_squares on the EXACT pricer from u0, with dh2.calib.fit_day's objective, bounds and scaling."""
    lo, hi = (np.array(b) for b in calib.BOUNDS[MODEL])
    f = Counter(lambda u: exact_residual(u, S, idx))
    u0 = calib.clip(MODEL, u0)
    r = least_squares(f, u0, bounds=(lo, hi), loss="soft_l1", f_scale=FS, x_scale=np.maximum(hi - lo, 1e-9) * 0.05, max_nfev=max_nfev)
    return {"u": r.x.copy(), "cost": float(r.cost), "nfev": int(r.nfev), "njev": int(r.njev), "n_calls": f.n, "status": int(r.status)}


# ------------------------------------------------------------------ the whole method
def propose(sur: Surrogate, S: Surface, idx: np.ndarray, cfg: PinnConfig) -> dict:
    """Stage 1 (and 1b): the candidates to polish, best first.  Without cfg.rerank / cfg.n_correct the order is the surrogate's own cost order;
    otherwise the survivors are ranked (and possibly refined) by their exact cost, which spends B (+ rounds x B) exact residual evaluations."""
    t0, c0 = time.perf_counter(), time.process_time()
    g = global_search(sur, S, idx, cfg)
    t1, c1 = time.perf_counter(), time.process_time()
    out = {"U": g["U"], "u_sur": g["U"][0].copy(), "sur_cost": float(g["cost"][0]), "sur_costs": g["cost"], "n_excluded": g["n_excluded"],
           "n_global_quotes": g["n_global_quotes"], "n_correct_calls": 0, "u_corr": None, "corr_cost": float("nan")}
    if cfg.n_correct > 0 or cfg.rerank:
        cr = correct(sur, S, idx, g["obj"], g["U"], cfg)
        out.update(U=cr["U"], n_correct_calls=cr["n_calls"], u_corr=cr["U"][0].copy(), corr_cost=float(cr["cost"][0]))
    t2, c2 = time.perf_counter(), time.process_time()
    out.update(n_sur_evals=g["obj"].calls, t_global=t1 - t0, t_correct=t2 - t1, cpu_global=c1 - c0, cpu_correct=c2 - c1)
    return out


def fit_day_pinn(sur: Surrogate, S: Surface, cfg: PinnConfig = DEFAULT, polish_starts: np.ndarray | None = None) -> dict:
    """PINN-accelerated fit of one day: propose() then polish() of the first n_polish distinct candidates.  polish_starts: optional replacement for
    the surrogate stage (used by the control 'exact polish from the Phase-3 starts'), an (m, 10) array.  Timing covers the global stage, the
    correction and the polish; the scoring is done by the caller (score())."""
    t0, c0 = time.perf_counter(), time.process_time()
    idx = M.thin(S, cfg.per_expiry)
    width = np.array(calib.BOUNDS[MODEL][1]) - np.array(calib.BOUNDS[MODEL][0])
    if polish_starts is None:
        p = propose(sur, S, idx, cfg)
        starts = p["U"][_distinct(p["U"], np.arange(len(p["U"])), cfg.n_polish, width, cfg.min_sep)]
        out = {k: v for k, v in p.items() if k != "U"}
    else:
        starts = np.atleast_2d(polish_starts)
        out = dict(u_sur=starts[0].copy(), sur_cost=float("nan"), n_excluded=0, n_global_quotes=0, n_sur_evals=0, n_correct_calls=0, u_corr=None,
                   corr_cost=float("nan"), t_global=0.0, t_correct=0.0, cpu_global=0.0, cpu_correct=0.0)
    t1, c1 = time.perf_counter(), time.process_time()
    cands = [polish(S, idx, u, cfg.polish_nfev) for u in starts]
    t2, c2 = time.perf_counter(), time.process_time()
    best = min(cands, key=lambda c: c["cost"])
    out.update(idx=idx, u=best["u"], cost=best["cost"], cands=cands, starts=starts, n_polish_calls=int(sum(c["n_calls"] for c in cands)),
               n_exact_calls=int(out["n_correct_calls"] + sum(c["n_calls"] for c in cands)),
               t_polish=t2 - t1, t_total=t2 - t0, cpu_polish=c2 - c1, cpu_total=c2 - c0)
    return out


# ------------------------------------------------------------------ the Phase-3 baseline with counters (identical to dh2.calib.fit_day)
def fit_day_exact(S: Surface, n_starts: int = 4, seed: int = 0, per_expiry: int = 14, max_nfev: int = 120) -> dict:
    """dh2.calib.fit_day('dh', ...) with exact-pricer call counting, wall and CPU timing and every start kept.  The loop, thinning, starts,
    bounds, loss and scaling are those of calib.fit_day, so the best start equals its result (checked in exp_pinn_calib.py against the C2 cache)."""
    t0, c0 = time.perf_counter(), time.process_time()
    rng = np.random.default_rng(seed)
    idx = M.thin(S, per_expiry)
    lo, hi = (np.array(b) for b in calib.BOUNDS[MODEL])
    runs, n_calls = [], 0
    best = None
    for u0 in calib.starts(MODEL, S, rng, n_starts):
        f = Counter(lambda u: exact_residual(u, S, idx))
        try:
            r = least_squares(f, u0, bounds=(lo, hi), loss="soft_l1", f_scale=FS, x_scale=np.maximum(hi - lo, 1e-9) * 0.05, max_nfev=max_nfev)
        except Exception:
            n_calls += f.n
            continue
        n_calls += f.n
        runs.append({"u": r.x.copy(), "cost": float(r.cost), "nfev": int(r.nfev), "n_calls": f.n})
        if best is None or r.cost < best.cost:
            best = r
    return {"u": best.x.copy(), "cost": float(best.cost), "runs": runs, "n_exact_calls": n_calls,
            "t_total": time.perf_counter() - t0, "cpu_total": time.process_time() - c0}
