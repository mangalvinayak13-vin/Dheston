"""Out-of-sample scoring: what was known on day t, used to predict day t+1's option prices (in implied-volatility points).

A "prediction" for day t+1 only ever uses information from days <= t PLUS the market state that every method is given for free on
day t+1: the index forward of each expiry (the parity forward), the time to expiry, the rate. What a method must supply from its
own past is the SHAPE of the volatility surface. That is the standard persistence test: carry the calibration forward one day.

The error of a quote is (predicted implied vol - market implied vol) in volatility points. Reported per day as an RMSE over
that day's quotes (all of them, and the near-the-money core |ln(K/F)| <= 0.04), then summarised across days.
"""
from __future__ import annotations

import numpy as np

from . import models as M
from .data import Surface

CORE = 0.04


def consecutive(S: list[Surface], max_gap_days: int = 5) -> list[tuple[int, int]]:
    """Index pairs (i, i+1) of surfaces on consecutive trading days (a gap of more than a long weekend means missing data)."""
    return [(i, i + 1) for i in range(len(S) - 1) if (S[i + 1].day - S[i].day).days <= max_gap_days]


def _log_m(S: Surface) -> np.ndarray:
    return np.log(S.K / S.F[S.e])


def core_mask(S: Surface) -> np.ndarray:
    return np.abs(_log_m(S)) <= CORE


# ------------------------------------------------------------------ predictors: each returns predicted implied vols for S1's quotes
def pred_flat(sigma: float, S1: Surface) -> np.ndarray:
    return np.full(S1.n, sigma)


def _match_expiry(S0: Surface, S1: Surface, e1: int) -> int:
    """The expiry of S0 that day t+1's expiry e1 corresponds to: the same expiry date if listed, else the nearest in time."""
    for e0, d in enumerate(S0.expiry):
        if d == S1.expiry[e1]:
            return e0
    return int(np.argmin(np.abs(S0.T - (S1.T[e1] + (S1.day - S0.day).days / 365.0))))


def carry_values(S0: Surface, S1: Surface, values0: np.ndarray) -> np.ndarray:
    """Day t's per-quote values (e.g. its implied vols, or a model's misfit) carried onto day t+1's quotes: expiry by expiry
    (same expiry date, else nearest in time), interpolated in log-moneyness."""
    m0, m1 = _log_m(S0), _log_m(S1)
    out = np.full(S1.n, np.nan)
    for e1 in range(len(S1.T)):
        q1 = np.flatnonzero(S1.e == e1)
        e0 = _match_expiry(S0, S1, e1)
        q0 = np.flatnonzero(S0.e == e0)
        if q0.size < 2:
            continue
        o = np.argsort(m0[q0])
        out[q1] = np.interp(m1[q1], m0[q0][o], values0[q0][o])
    return out


def pred_last_smile(S0: Surface, S1: Surface) -> np.ndarray:
    """Persistence of yesterday's smile in log-moneyness, expiry by expiry (the model-free benchmark a model must beat)."""
    return carry_values(S0, S1, S0.iv)


def pred_smile_scaled(S0: Surface, S1: Surface) -> np.ndarray:
    """A stronger model-free benchmark: yesterday's smile, carried in time-scaled moneyness. A smile's width shrinks like sqrt(T), so a
    strike's log-moneyness m on day t+1 is read from yesterday's same-expiry smile at m * sqrt(T0 / T1), where T0 and T1 are the
    expiry's time left on the two days. (Plain persistence reads it at m, ignoring that the expiry is a day closer.)"""
    m0, m1 = _log_m(S0), _log_m(S1)
    out = np.full(S1.n, np.nan)
    for e1 in range(len(S1.T)):
        q1 = np.flatnonzero(S1.e == e1)
        e0 = _match_expiry(S0, S1, e1)
        q0 = np.flatnonzero(S0.e == e0)
        if q0.size < 2:
            continue
        o = np.argsort(m0[q0])
        scale = np.sqrt(S0.T[e0] / S1.T[e1])
        out[q1] = np.interp(m1[q1] * scale, m0[q0][o], S0.iv[q0][o])
    return out


def fwd_return(S0: Surface, S1: Surface) -> float:
    """Log return of the index forward between the two days, on the nearest expiry (at least 10 days out) both days list."""
    for e1 in range(len(S1.T)):
        if S1.T[e1] < 10 / 365:
            continue
        e0 = next((k for k, d in enumerate(S0.expiry) if d == S1.expiry[e1]), None)
        if e0 is not None:
            return float(np.log(S1.F[e1] / S0.F[e0]))
    return 0.0


def pred_atm_by_expiry(S0: Surface, S1: Surface) -> np.ndarray:
    """Yesterday's at-the-money volatility of the matching expiry, held flat across strikes (Black-Scholes per expiry)."""
    out = np.full(S1.n, np.nan)
    for e1 in range(len(S1.T)):
        v = S0.atm_iv(_match_expiry(S0, S1, e1))
        out[S1.e == e1] = v
    return out


def pred_model(model: str, x, S1: Surface) -> np.ndarray:
    return M.model_iv(M.price(model, x, S1), S1)


# ------------------------------------------------------------------ scoring
def score(iv_pred: np.ndarray, S1: Surface) -> dict:
    """Per-day summary of one prediction: RMSE in vol points over all quotes and over the core, plus mean signed error."""
    ok = np.isfinite(iv_pred)
    if ok.sum() < 5:
        return {"rmse": np.nan, "core": np.nan, "bias": np.nan, "n": int(ok.sum())}
    e = 100.0 * (iv_pred - S1.iv)
    c = ok & core_mask(S1)
    return {"rmse": float(np.sqrt(np.mean(e[ok] ** 2))), "core": float(np.sqrt(np.mean(e[c] ** 2))) if c.sum() >= 3 else np.nan,
            "bias": float(np.mean(e[ok])), "n": int(ok.sum())}


def summarize(rows: list[dict], key: str = "rmse") -> dict:
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "days": int(v.size)}
