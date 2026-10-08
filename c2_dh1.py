"""DH1: the original 1.0 research project's Double Heston calibration on NIFTY, rebuilt on the 2.0 pricer (pricer/models.py).

Recovered from the old repository (read-only): double-heston-inverse-calibration/.claude/worktrees/complete-training
  data   src/mentor_dh_pinn/nifty_panel.py          (surface())     NSE F&O files, SettlementPrice (not the last-trade close), open
                                                    interest >= 10,000 per contract, an expiry kept only if it has volume and enough
                                                    open interest, forward + discount per expiry from a Huber-reweighted put-call
                                                    parity regression (single_heston.robust_carry), OTM side, 0.78..1.22 of forward,
                                                    implied vols by Brent, 1 <= days to expiry (no minimum), no minimum price.
  quotes scripts/mentor_dh_pinn/evaluate_real_markets.py (nifty_cases())  every third strike of every expiry (sorted, positions 2::3)
                                                    is held out of the fit; the fit uses the other two thirds; the held-out quotes with
                                                    20 <= days <= 800 are scored in implied-vol RMSE. Everything is in the forward
                                                    measure: spot = the quote's own forward, r = q = 0, price = forward call
                                                    (SttlmPrice / discount, puts converted by parity).
  loss   src/mentor_dh_pinn/baselines.py           (fit_double_heston)   residual = (model forward call - market forward call) / forward,
                                                    PRICE space, no vega weights, plain least squares (no robust loss).
  optim  same file + params_v2.py                  scipy least_squares (trf), diff_step 1e-6, max_nfev 220, in unbounded latent
                                                    coordinates that build in the ordering kappa_fast > kappa_slow, the Feller condition
                                                    (sigma = eta sqrt(2 kappa theta), eta in (0,1)) and the correlation disk
                                                    (rho_s^2 + rho_f^2 < 1); five fixed starting vectors (DH_STARTS); keep the lowest
                                                    mean squared data residual. NOTE: the first start violates Feller, encode() raises
                                                    ValueError, and 1.0's loop skips it silently, so four starts actually run.
What could not be reproduced exactly: 1.0 priced with its own Gauss-Legendre pricer (src/double_heston.py / torch_pricer.py); here the
pricer is pricer/models.py as instructed (the same model, a different quadrature), and the pre-June-2024 NSE files carry no index level
(UndrlygPric), so the reference level used only for robust_carry's validity gates is the strike where call = put (as dh2.data does).
Black-Scholes (one sigma) is fitted as in baselines.fit_black_scholes (price space, grid + bounded Brent).
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

import c2_common as C
from dh2 import models as M
from dh2.data import black_iv, black_price

RAW = C.ROOT / "data" / "nse_index"
P = M.P
ROW_OI = 0           # per-contract open-interest floor: the surviving read_quotes() has 10,000, but only 0 reproduces 1.0's own counts (833 quotes, 10 expiries on 2026-04-08)
AGG_OI = 10_000      # per-expiry open-interest floor (nifty_panel.surface)
MONEYNESS = (0.78, 1.22)
SIG_LO, SIG_HI = 0.01, 4.0

DH_STARTS = (   # kappa_s, theta_s, sigma_s, rho_s, v0_s, kappa_f, theta_f, sigma_f, rho_f, v0_f  (baselines.DH_STARTS)
    (0.8, 0.04, 0.30, -0.6, 0.04, 4.0, 0.03, 0.35, -0.3, 0.03),
    (1.5, 0.09, 0.45, -0.4, 0.09, 6.0, 0.06, 0.55, -0.2, 0.06),
    (0.4, 0.16, 0.35, -0.8, 0.16, 2.5, 0.10, 0.45, -0.4, 0.10),
    (2.2, 0.02, 0.20, -0.5, 0.03, 9.0, 0.02, 0.30, -0.1, 0.02),
    (1.0, 0.25, 0.55, -0.3, 0.30, 5.0, 0.18, 0.70, -0.3, 0.25),
)
_TINY, _SHARE_EPS = 1e-12, 1e-12


# ------------------------------------------------------------------ 1.0's latent coordinates (params_v2.py, numpy path)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _share(x):
    return _SHARE_EPS + (1.0 - 2.0 * _SHARE_EPS) * _sig(x)


def decode(z):
    """R^10 -> canonical (kappa_s, theta_s, sigma_s, rho_s, v0_s, kappa_f, theta_f, sigma_f, rho_f, v0_f), as params_v2.decode."""
    z = np.asarray(z, float)
    ks = np.exp(z[0])
    kf = ks * (1.0 + np.exp(z[1]) + _TINY)
    th_tot = np.exp(z[2]); a_th = _share(z[3]); ts, tf = th_tot * a_th, th_tot * (1.0 - a_th)
    v_tot = np.exp(z[4]); a_v = _share(z[5]); vs, vf = v_tot * a_v, v_tot * (1.0 - a_v)
    ss = _share(z[6]) * np.sqrt(2.0 * ks * ts)
    sf = _share(z[7]) * np.sqrt(2.0 * kf * tf)
    a, b = z[8], z[9]
    n = np.sqrt(a * a + b * b) + _TINY
    r = np.tanh(n) * (1.0 - 1e-12)
    return [ks, ts, ss, a / n * r, vs, kf, tf, sf, b / n * r, vf]


def encode(params):
    """Canonical ten-vector -> R^10 (params_v2.encode); raises ValueError if the vector is outside the model class."""
    ks, ts, ss, rs, vs, kf, tf, sf, rf, vf = (float(v) for v in params)
    bad = []
    if not ks < kf:
        bad.append("ordering")
    if 2 * ks * ts - ss ** 2 <= 0 or 2 * kf * tf - sf ** 2 <= 0:
        bad.append("Feller")
    if rs ** 2 + rf ** 2 >= 1.0:
        bad.append("correlation disk")
    if min(ts, tf, vs, vf, ss, sf, ks) <= 0:
        bad.append("positivity")
    if bad:
        raise ValueError(f"outside the model class: {bad}")

    def logit(u):
        u = (min(max(u, 1e-15), 1 - 1e-15) - _SHARE_EPS) / (1.0 - 2.0 * _SHARE_EPS)
        u = min(max(u, 1e-15), 1 - 1e-15)
        return math.log(u / (1 - u))
    z = np.zeros(10)
    z[0] = math.log(ks)
    z[1] = math.log(max(kf / ks - 1.0, 1e-12))
    z[2] = math.log(ts + tf)
    z[3] = logit(ts / (ts + tf))
    z[4] = math.log(vs + vf)
    z[5] = logit(vs / (vs + vf))
    z[6] = logit(ss / math.sqrt(2 * ks * ts))
    z[7] = logit(sf / math.sqrt(2 * kf * tf))
    r = math.hypot(rs, rf)
    if r < 1e-15:
        z[8] = z[9] = 0.0
    else:
        n = math.atanh(min(r, 1 - 1e-15))
        z[8], z[9] = rs / r * n, rf / r * n
    return z


def to_pricer(c) -> np.ndarray:
    """Canonical (slow first) -> the pricer's order (v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2)."""
    ks, ts, ss, rs, vs, kf, tf, sf, rf, vf = c
    return np.array([vs, ks, ts, ss, rs, vf, kf, tf, sf, rf], float)


def n_valid_starts() -> int:
    k = 0
    for s in DH_STARTS:
        try:
            encode(s)
            k += 1
        except ValueError:
            pass
    return k


# ------------------------------------------------------------------ robust parity (single_heston.robust_carry, copied)
def robust_carry(strike, call_minus_put, spot, maturity):
    design = np.column_stack([np.ones(len(strike)), strike])
    beta = np.linalg.lstsq(design, call_minus_put, rcond=None)[0]
    for _ in range(6):
        error = call_minus_put - design @ beta
        scale = max(np.median(np.abs(error)) * 1.4826, 0.05)
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(error), 1e-12))
        beta = np.linalg.lstsq(design * weights[:, None], call_minus_put * weights, rcond=None)[0]
    discount = -float(beta[1])
    forward = float(beta[0] / discount) if discount > 0 else np.nan
    error = call_minus_put - discount * (forward - strike)
    nrmse = float(np.sqrt(np.mean(error ** 2)) / spot)
    valid = (np.isfinite(forward) and math.exp(-0.25 * maturity) <= discount <= math.exp(0.10 * maturity)
             and 0.75 <= forward / spot <= 1.25 and nrmse <= 0.025 and (strike.max() - strike.min()) / spot >= 0.04)
    if not valid:
        return None
    rate = -math.log(discount) / maturity
    dividend = rate - math.log(forward / spot) / maturity
    if not (-0.50 <= dividend <= 0.50):
        return None
    return forward, discount, rate, dividend, nrmse


# ------------------------------------------------------------------ 1.0's NIFTY surface for one day
def dh1_surface(day: date) -> dict | None:
    """The quotes 1.0's NIFTY pipeline would use on this day (nifty_panel.surface + nifty_cases), as arrays; None if it yields none."""
    path = RAW / f"{day.isoformat()}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    d = df[(df.TckrSymb == "NIFTY") & df.OptnTp.isin(["CE", "PE"])].copy()
    for c in ("StrkPric", "SttlmPric", "UndrlygPric", "TtlTradgVol", "OpnIntrst", "ClsPric"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.drop_duplicates()
    d["dte"] = (pd.to_datetime(d.XpryDt) - pd.to_datetime(d.TradDt)).dt.days
    has_und = bool(d.UndrlygPric.notna().any())
    keep = (d.dte > 0) & (d.SttlmPric > 0) & (d.StrkPric > 0) & (d.OpnIntrst >= ROW_OI)
    if has_und:
        keep &= d.UndrlygPric > 0
    d = d[keep]
    d = d.drop_duplicates(subset=["XpryDt", "StrkPric", "OptnTp"])
    if d.empty:
        return None
    spot_file = float(d.UndrlygPric.median()) if has_und else float("nan")
    rows = []
    for dte, g in d.groupby("dte"):
        if g.TtlTradgVol.fillna(0).sum() <= 0 or g.OpnIntrst.fillna(0).sum() < AGG_OI:
            continue
        c = g[g.OptnTp == "CE"].set_index("StrkPric").SttlmPric
        p = g[g.OptnTp == "PE"].set_index("StrkPric").SttlmPric
        k = c.index.intersection(p.index).sort_values()
        if len(k) < 6:
            continue
        K = k.to_numpy(float)
        cmp_ = (c[k] - p[k]).to_numpy(float)
        spot = spot_file if np.isfinite(spot_file) else float(np.median(K[np.argsort(np.abs(cmp_))[:3]]))
        tau = dte / 365.0
        carry = robust_carry(K, cmp_, spot, tau)
        if carry is None:
            continue
        F, df_, rate, div, nrmse = carry
        q = g[(g.StrkPric >= MONEYNESS[0] * F) & (g.StrkPric <= MONEYNESS[1] * F)].copy()
        q = q[np.where(q.StrkPric >= F, q.OptnTp == "CE", q.OptnTp == "PE")]
        if len(q) < 4:
            continue
        Kq = q.StrkPric.to_numpy(float)
        is_call = (q.OptnTp.to_numpy() == "CE")
        px = q.SttlmPric.to_numpy(float)
        vol_q = q.TtlTradgVol.fillna(0).to_numpy(float)
        oi_q = q.OpnIntrst.fillna(0).to_numpy(float)
        fwd_call = np.where(is_call, px / df_, px / df_ + F - Kq)
        gate = (np.maximum(F - Kq, 0.0) < fwd_call) & (fwd_call < F)
        iv = np.full(len(q), np.nan)
        if gate.any():
            iv[gate] = black_iv(fwd_call[gate], F, Kq[gate], tau, 0.0, True)
        for j in range(len(q)):
            rows.append((int(dte), tau, F, df_, float(Kq[j]), bool(is_call[j]), float(px[j]), float(fwd_call[j]), float(iv[j]),
                         q.XpryDt.iloc[j], float(vol_q[j]), float(oi_q[j])))
    if not rows:
        return None
    a = list(zip(*rows))
    out = {"dte": np.array(a[0], int), "tau": np.array(a[1]), "F": np.array(a[2]), "df": np.array(a[3]), "K": np.array(a[4]),
           "is_call": np.array(a[5], bool), "px": np.array(a[6]), "fwd_call": np.array(a[7]), "iv": np.array(a[8]),
           "xpry": np.array(a[9]), "vol": np.array(a[10]), "oi": np.array(a[11])}
    ok = np.isfinite(out["iv"]) & (out["iv"] > SIG_LO * 1.01) & (out["iv"] < SIG_HI * 0.99)
    out = {k: v[ok] for k, v in out.items()}
    n = len(out["K"])
    if n < 8:
        return None
    fit = np.ones(n, bool)
    for dte in np.unique(out["dte"]):
        ix = np.flatnonzero(out["dte"] == dte)
        ix = ix[np.argsort(out["K"][ix])]
        fit[ix[2::3]] = False
    out["fit"] = fit
    out["hold"] = (~fit) & (out["dte"] >= 20) & (out["dte"] <= 800)
    out["spot"] = spot_file
    out["day"] = day
    return out


def _groups(sf: dict, mask: np.ndarray):
    """(F, T, positions-within-mask) per expiry for the quotes in mask, for vectorised pricing."""
    ix = np.flatnonzero(mask)
    g = []
    for dte in np.unique(sf["dte"][ix]):
        pos = np.flatnonzero(sf["dte"][ix] == dte)
        g.append((float(sf["F"][ix[pos[0]]]), float(sf["tau"][ix[pos[0]]]), sf["K"][ix[pos]], pos))
    return g, len(ix)


def price_fwd_call(x10, groups, n) -> np.ndarray:
    """Forward-measure call prices (spot = forward, r = q = 0) of every quote, in the pricer's parameter order."""
    out = np.empty(n)
    for F, T, K, pos in groups:
        out[pos] = np.atleast_1d(P.double_heston_price(F, K, T, 0.0, *x10, kind="call", q=0.0))
    return out


# ------------------------------------------------------------------ the 1.0 fits
def fit_latent(resid_of_x, m: int, starts=DH_STARTS, max_nfev: int = 220, loss: str = "linear", f_scale: float = 1.0) -> dict | None:
    """1.0's fit_double_heston loop: for each (valid) fixed start, scipy least_squares (trf) in the unbounded latent coordinates with
    diff_step 1e-6 and max_nfev 220; keep the start with the lowest cost. resid_of_x maps the pricer-order x to a residual vector."""
    best, used = None, 0
    for s in starts:
        try:
            z0 = encode(s)
        except ValueError:
            continue

        def fun(z):
            try:
                r = resid_of_x(to_pricer(decode(z)))
            except Exception:
                return np.ones(m)
            return r if np.all(np.isfinite(r)) else np.ones(m)
        try:
            r = least_squares(fun, z0, max_nfev=max_nfev, diff_step=1e-6, loss=loss, f_scale=f_scale)
        except Exception:
            continue
        used += 1
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    return {"x": to_pricer(decode(best.x)), "z": best.x, "cost": float(best.cost), "nfev": int(best.nfev), "n_starts": used}


def fit_dh1(sf: dict) -> dict | None:
    """Step 0: 1.0's data, objective and optimiser."""
    g, n = _groups(sf, sf["fit"])
    obs, F = sf["fwd_call"][sf["fit"]], sf["F"][sf["fit"]]
    f = fit_latent(lambda x: (price_fwd_call(x, g, n) - obs) / F, n)
    return f


def holdout_iv_rmse_dh(sf: dict, x10) -> float:
    """1.0's score: IV RMSE (vol points) of the model on the held-out quotes (20..800 days), implied at the quote's own forward."""
    g, n = _groups(sf, sf["hold"])
    if n < 4:
        return float("nan")
    ix = np.flatnonzero(sf["hold"])
    m = price_fwd_call(x10, g, n)
    iv = black_iv(m, sf["F"][ix], sf["K"][ix], sf["tau"][ix], 0.0, True)
    e = (iv - sf["iv"][ix])
    e = e[np.isfinite(e)]
    return 100.0 * float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def fit_bs1(sf: dict) -> float:
    """1.0's one-sigma Black-Scholes fit (price space, grid + bounded Brent) on the fit quotes."""
    obs, F, K, T = sf["fwd_call"][sf["fit"]], sf["F"][sf["fit"]], sf["K"][sf["fit"]], sf["tau"][sf["fit"]]
    obj = lambda s: float(np.mean((black_price(F, K, T, 0.0, s, True) - obs) ** 2))
    grid = np.exp(np.linspace(math.log(0.02), math.log(3.0), 140))
    vals = [obj(s) for s in grid]
    j = int(np.argmin(vals))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    r = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
    return float(r.x) if r.fun <= vals[j] else float(grid[j])


def holdout_iv_rmse_bs(sf: dict, sigma: float) -> float:
    e = sigma - sf["iv"][sf["hold"]]
    return 100.0 * float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def replicate_day(day: date) -> dict:
    """Run the 1.0 recipe on one day and score it as 1.0 did (held-out third); used to check the replication on 1.0's own dates."""
    sf = dh1_surface(day)
    if sf is None:
        return {"day": day, "error": "no surface"}
    f = fit_dh1(sf)
    sg = fit_bs1(sf)
    return {"day": day, "quotes": int(len(sf["K"])), "holdout": int(sf["hold"].sum()), "expiries": int(len(np.unique(sf["dte"]))),
            "bs1_sigma": sg, "bs1_iv": holdout_iv_rmse_bs(sf, sg) / 100.0,
            "dh_iv": (holdout_iv_rmse_dh(sf, f["x"]) / 100.0 if f else float("nan")), "x": (f["x"] if f else None),
            "n_starts": (f["n_starts"] if f else 0)}


# ------------------------------------------------------------------ steps 0, 1, 2 of the attribution
def fit_step(step: int, S0, day: date) -> dict:
    """step 0: DH1 as 1.0 (its data, price objective, latent multi-start). step 1: the same objective/optimiser on the 2.0 clean surface.
    step 2: step 1 with the IV-space, vega-weighted, soft-L1 objective. All use the two-thirds-of-strikes fit set of the 1.0 protocol."""
    import time
    t = time.time()
    out = {"step": step}
    if step == 0:
        sf = dh1_surface(day)
        if sf is None:
            return {"step": 0, "error": "no 1.0 surface"}
        f = fit_dh1(sf)
        out.update({"n_quotes": int(len(sf["K"])), "n_fit": int(sf["fit"].sum()), "n_exp": int(len(np.unique(sf["dte"]))),
                    "holdout_iv_rmse_pts": (holdout_iv_rmse_dh(sf, f["x"]) if f else float("nan")),
                    "sf_keys": {"xpry": sf["xpry"], "K": sf["K"], "is_call": sf["is_call"], "px": sf["px"], "F": sf["F"], "dte": sf["dte"]}})
    else:
        S = S0
        fit = np.ones(S.n, bool)
        for e in range(len(S.T)):
            ix = np.flatnonzero(S.e == e)
            ix = ix[np.argsort(S.K[ix])]
            fit[ix[2::3]] = False
        idx = np.flatnonzero(fit)
        e_ = S.e[idx]
        if step == 1:
            scale = np.exp(-S.r * S.T[e_]) * S.F[e_]
            resid = lambda x: (M.price("dh", x, S, idx) - S.price[idx]) / scale
            f = fit_latent(resid, len(idx))
        elif step == 2:
            w = np.maximum(S.vega[idx], C.calib.VEGA_FLOOR)
            resid = lambda x: (M.price("dh", x, S, idx) - S.price[idx]) / w
            f = fit_latent(resid, len(idx), loss="soft_l1", f_scale=C.calib.F_SCALE)
        else:
            raise ValueError(step)
        out.update({"n_fit": int(len(idx))})
    if f is None:
        out["error"] = "all starts failed"
        return out
    out.update({"x": f["x"], "cost": f["cost"], "nfev": f["nfev"], "n_starts": f["n_starts"], "sec": time.time() - t})
    return out
