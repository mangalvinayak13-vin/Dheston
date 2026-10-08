"""Pattern memory: when today looks like an old pattern, learn from what happened after it.

The question the memory answers is "has the market been in a spot like this before, and if so how did my forecast do afterward?"

1. DESCRIBE each day by a handful of numbers read off the option surface and the index: the volatility level (30-day at the money),
   the term-structure slope (90-day minus 30-day), the skew and curvature at 30 days, the front-expiry skew, the index's recent
   moves (1 and 5 days) and recent realised volatility (5 and 20 days), and the volatility risk premium (implied minus realised).
2. A PATTERN is the last L = 5 days of those numbers, each standardised with only what was known by today (expanding mean and
   standard deviation), and compared with every earlier 5-day stretch, recent days weighted more (0.7 per day back).
3. IS IT REPEATING? A match counts only when the closest old stretch is unusually close: closer than the q-quantile (default 5%)
   of all old-vs-old distances known today. Otherwise the memory stays silent and the forecast is untouched. When it does fire,
   the k closest stretches vote, nearer ones more.
4. LEARN FROM THE OLD PATTERN. For each voting stretch, take the day after it, and the errors the SAME predictor made on that
   day's quotes (market implied vol minus predicted). Today's correction at a quote (log-moneyness m, expiry T) is the vote-weighted,
   kernel-smoothed average of those old errors near (m, T), shrunk by a strength beta. The memory therefore corrects the
   predictor's systematic misses in situations that look like the current one, whatever the predictor is.

Look-ahead is impossible by construction: a stretch ending at day i votes for day t only if i + 1 <= t (its aftermath is known),
and the standardisation uses days <= t.
"""
from __future__ import annotations

import numpy as np

from .data import Surface

FEATURES = ["atm30", "term", "skew30", "curv30", "skew_front", "ret1", "ret5", "rv5", "rv20", "vrp"]


def _civ(S: Surface, T_target: float, m: float) -> float:
    """Implied vol at log-moneyness m and maturity T_target: each expiry's smile interpolated at m (where it has quotes), total
    variance interpolated linearly in T between the two expiries around T_target (flat volatility beyond the ends)."""
    pts = []
    for e in range(len(S.T)):
        q = np.flatnonzero(S.e == e)
        if q.size < 4:
            continue
        lm = np.log(S.K[q] / S.F[e])
        if lm.min() - 0.01 > m or lm.max() + 0.01 < m:
            continue
        o = np.argsort(lm)
        pts.append((S.T[e], float(np.interp(m, lm[o], S.iv[q][o]))))
    if not pts:
        return float("nan")
    pts.sort()
    T, iv = np.array(pts).T
    w = iv**2 * T
    if T_target <= T[0]:
        return float(iv[0])
    if T_target >= T[-1]:
        return float(iv[-1])
    return float(np.sqrt(np.interp(T_target, T, w) / T_target))


def features(surfaces: list[Surface]) -> np.ndarray:
    """One row of FEATURES per surface (NaN where a number cannot be read). Index moves come from the parity forward of the nearest
    expiry the two consecutive days share, so a roll between expiries never looks like a jump."""
    n = len(surfaces)
    f = np.full((n, len(FEATURES)), np.nan)
    ret = np.full(n, np.nan)
    for i, S in enumerate(surfaces):
        a30, a90 = _civ(S, 30 / 365, 0.0), _civ(S, 90 / 365, 0.0)
        lo, hi = _civ(S, 30 / 365, -0.05), _civ(S, 30 / 365, 0.05)
        f[i, 0] = a30
        f[i, 1] = a90 - a30
        f[i, 2] = lo - hi
        f[i, 3] = 0.5 * (lo + hi) - a30
        f[i, 4] = _civ(S, S.T[0], -0.03) - _civ(S, S.T[0], 0.03)
        if i > 0 and (S.day - surfaces[i - 1].day).days <= 5:
            P = surfaces[i - 1]
            for e1 in range(len(S.T)):
                if S.T[e1] < 10 / 365:
                    continue
                e0 = next((k for k, d in enumerate(P.expiry) if d == S.expiry[e1]), None)
                if e0 is not None:
                    ret[i] = np.log(S.F[e1] / P.F[e0])
                    break
    f[:, 5] = ret
    for i in range(n):
        r = ret[max(0, i - 4): i + 1]
        r20 = ret[max(0, i - 19): i + 1]
        if np.isfinite(r).sum() >= 3:
            f[i, 6] = np.nansum(r)
            f[i, 7] = np.sqrt(252 * np.nanmean(r**2))
        if np.isfinite(r20).sum() >= 10:
            f[i, 8] = np.sqrt(252 * np.nanmean(r20**2))
    f[:, 9] = f[:, 0] - f[:, 8]
    return f


class Memory:
    """Causal pattern matcher over a fixed list of surfaces. match(t) uses only days <= t."""

    def __init__(self, feat: np.ndarray, L: int = 5, k: int = 8, q: float = 0.05, decay: float = 0.7, min_history: int = 120):
        self.f, self.L, self.k, self.q, self.decay, self.min_history = feat, L, k, q, decay, min_history
        self.w = decay ** np.arange(L)           # lag 0 (today) first
        # expanding statistics, so the standardisation at day t sees only days <= t
        ok = np.isfinite(feat)
        c = np.cumsum(np.where(ok, feat, 0.0), axis=0)
        c2 = np.cumsum(np.where(ok, feat**2, 0.0), axis=0)
        n = np.maximum(np.cumsum(ok, axis=0), 1)
        self.mean = c / n
        self.std = np.sqrt(np.maximum(c2 / n - self.mean**2, 1e-12))

    def _z(self, t: int) -> np.ndarray:
        z = (self.f - self.mean[t]) / self.std[t]
        return np.where(np.isfinite(z), z, 0.0)   # a missing number counts as 'average': it neither helps nor hurts a match

    def match(self, t: int):
        """(indices of the matched stretches' last days, weights, closest distance, threshold) or None when nothing is close."""
        L = self.L
        last = t - L - 1                      # a stretch ending at i must be wholly before today's, and its aftermath i+1 <= t
        if last < L + self.min_history:
            return None
        z = self._z(t)
        ends = np.arange(L - 1, last + 1)
        cand = np.stack([z[ends - l] for l in range(L)])             # (L, n, F)
        cur = np.stack([z[t - l] for l in range(L)])                 # (L, F)
        d = np.sqrt((self.w[:, None] * ((cand - cur[:, None, :]) ** 2).sum(-1)).sum(0) / self.w.sum())
        # how close is "unusually close"? the q-quantile of old-vs-old distances (a sample of reference stretches against all others)
        rng = np.random.default_rng(t)
        ref = rng.choice(len(ends), size=min(60, len(ends)), replace=False)
        dd = np.sqrt((self.w[:, None, None] * ((cand[:, ref, None, :] - cand[:, None, :, :]) ** 2).sum(-1)).sum(0) / self.w.sum())
        dd = dd[dd > 0]
        tau = float(np.quantile(dd, self.q))
        order = np.argsort(d)[: self.k]
        if d[order[0]] > tau:
            return None
        sel = order[d[order] <= 1.5 * tau]          # the nearest stretches that are themselves reasonably close
        h = max(float(np.median(d[sel])), 1e-6)
        wts = np.exp(-0.5 * (d[sel] / h) ** 2)
        return ends[sel], wts / wts.sum(), float(d[order[0]]), tau


def correct(pred_quotes: tuple[np.ndarray, np.ndarray], old: list[tuple[np.ndarray, np.ndarray, np.ndarray]], weights: np.ndarray,
            h_m: float = 0.012, h_logT: float = 0.25) -> np.ndarray:
    """Vote-weighted old errors at (m, T): old = per voting stretch (m_q, T_q, err_q) of the aftermath day's quotes.
    Gaussian kernel in log-moneyness and log maturity; a point with no old quote nearby gets 0 (no opinion)."""
    m, T = pred_quotes
    out = np.zeros(len(m))
    wsum = np.zeros(len(m))
    for (mo, To, eo), w in zip(old, weights):
        ok = np.isfinite(eo)
        if not ok.any():
            continue
        k = np.exp(-0.5 * (((m[:, None] - mo[ok][None, :]) / h_m) ** 2 + ((np.log(T)[:, None] - np.log(To[ok])[None, :]) / h_logT) ** 2))
        s = k.sum(1)
        has = s > 1e-3
        out[has] += w * (k[has] @ eo[ok]) / s[has]
        wsum[has] += w
    return np.where(wsum > 0, out / np.maximum(wsum, 1e-12), 0.0)
