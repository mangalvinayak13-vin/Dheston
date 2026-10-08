"""Daily NIFTY 50 option surfaces from NSE's own end-of-day files (data/nse_index/*.csv), ten years of them.

One Surface per day:
  spot        the index level at the close when the file has it (the 2024+ files do); in the older files, which carry no index
              level, a proxy: the front forward discounted back (so the front carry is 0). Spot only anchors the carry, so
              prices never depend on which it is.
  r           the financing rate that makes put-call parity hold in that day's option prices (see implied_rate); a trailing
              10-day median so it uses only the past, and a causal series is safe for walk-forward tests
  expiries    each listed expiry with its time to expiry T (calendar days / 365), the forward price F implied by put-call
              parity at that expiry and the carry q that makes spot * exp((r-q)T) = F
  quotes      the out-of-the-money options only (puts below the forward, calls above it: the market's convention): strike,
              expiry index, call/put, closing price, Black implied volatility, Black vega, volume, open interest

Why parity forwards and an implied rate: NSE's option close and the index close are not the same instant, and over ten years the
rate went from about 3% to 7%, so neither a spot * exp((r-q)T) with guessed numbers nor a constant rate would be right. The
forward and the discount factor that parity implies from the options themselves are consistent with the option prices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "nse_index"
SYMBOL = "NIFTY"
DEFAULT_RATE = 0.06      # only when a day's parity gives no usable estimate and there is no earlier one

MIN_DAYS = 3             # expiries closer than this are dropped: an expiry-week quote is mostly noise and rounding
MIN_PRICE = 1.0          # rupees: below this the 5-paise tick is a big share of the price
MAX_LOG_MONEYNESS = 0.22
PARITY_STRIKES = 6       # strikes nearest the money used to estimate each expiry's forward
RATE_MIN_DAYS = 30       # expiries at least this far out are used to read the rate (the discount factor is invisible below)
RATE_SMOOTH_DAYS = 10


# ------------------------------------------------------------------ Black-76 helpers (vectorised)
def black_price(F, K, T, r, sigma, is_call):
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    call = np.exp(-r * T) * (F * ndtr(d1) - K * ndtr(d2))
    return np.where(is_call, call, call - np.exp(-r * T) * (F - K))


def black_vega(F, K, T, r, sigma):
    """Rupee change in price for a 1.00 (100 point) change in volatility; divide by 100 for per volatility point."""
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.exp(-r * T) * F * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)


def black_iv(price, F, K, T, r, is_call, lo=1e-3, hi=5.0, iters=64):
    """Implied volatility by bisection, vectorised; NaN where the price is outside the no-arbitrage range."""
    price, F, K, T = (np.asarray(a, float) for a in (price, F, K, T))
    is_call = np.asarray(is_call, bool)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = black_price(F, K, T, r, m, is_call) < price
        a, b = np.where(up, m, a), np.where(up, b, m)
    iv = 0.5 * (a + b)
    ok = (black_price(F, K, T, r, lo, is_call) <= price) & (black_price(F, K, T, r, hi, is_call) >= price)
    return np.where(ok, iv, np.nan)


# ------------------------------------------------------------------ the surface
@dataclass
class Surface:
    day: date
    spot: float
    r: float
    T: np.ndarray                 # per expiry, years
    F: np.ndarray                 # per expiry, parity forward
    q: np.ndarray                 # per expiry, carry with spot * exp((r - q) T) = F
    expiry: list                  # per expiry, date
    K: np.ndarray = field(default_factory=lambda: np.empty(0))     # per quote
    e: np.ndarray = field(default_factory=lambda: np.empty(0, int))  # per quote, index into T/F/q
    is_call: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    price: np.ndarray = field(default_factory=lambda: np.empty(0))
    iv: np.ndarray = field(default_factory=lambda: np.empty(0))
    vega: np.ndarray = field(default_factory=lambda: np.empty(0))  # rupees per volatility POINT
    volume: np.ndarray = field(default_factory=lambda: np.empty(0))
    oi: np.ndarray = field(default_factory=lambda: np.empty(0))
    spot_source: str = "file"

    @property
    def n(self) -> int:
        return len(self.K)

    def atm_iv(self, e: int) -> float:
        """Implied volatility at the money for expiry e (the quote closest to the forward), NaN if none."""
        m = self.e == e
        if not m.any():
            return float("nan")
        i = np.flatnonzero(m)[np.argmin(np.abs(np.log(self.K[m] / self.F[e])))]
        return float(self.iv[i])


# ------------------------------------------------------------------ reading a day
def _read(day: date) -> pd.DataFrame | None:
    path = RAW / f"{day.isoformat()}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    o = df[(df.TckrSymb == SYMBOL) & (df.FinInstrmTp == "IDO")].copy()
    if o.empty:
        return None
    o["K"] = pd.to_numeric(o.StrkPric, errors="coerce")
    o["px"] = pd.to_numeric(o.ClsPric, errors="coerce")
    o["vol"] = pd.to_numeric(o.TtlTradgVol, errors="coerce").fillna(0.0)
    o["oi"] = pd.to_numeric(o.OpnIntrst, errors="coerce").fillna(0.0)
    o["und"] = pd.to_numeric(o.UndrlygPric, errors="coerce")
    o["xp"] = pd.to_datetime(o.XpryDt).dt.date
    o["call"] = o.OptnTp == "CE"
    return o.dropna(subset=["K", "px"])


def _pairs(g: pd.DataFrame):
    """Strikes at which both the call and the put traded: the strike, call price, put price."""
    live = g[(g.px > 0) & (g.vol > 0)]
    calls, puts = live[live.call].set_index("K").px, live[~live.call].set_index("K").px
    calls, puts = calls[~calls.index.duplicated()], puts[~puts.index.duplicated()]
    both = calls.index.intersection(puts.index).values
    return both, calls[both].values, puts[both].values


def implied_rate(o: pd.DataFrame, day: date) -> float:
    """The continuously compounded rate at which put-call parity, C - P = exp(-rT)(F - K), holds in the day's prices: for each
    expiry at least RATE_MIN_DAYS out, regress C - P on K over the ~12 strikes nearest the money (slope = -exp(-rT)), then take
    the median over expiries. NaN if no expiry gives a sane reading."""
    out = []
    for xp, g in o.groupby("xp"):
        days = (xp - day).days
        if days < RATE_MIN_DAYS:
            continue
        K, c, p = _pairs(g)
        if len(K) < 8:
            continue
        guess = float(np.median(K[np.argsort(np.abs(c - p))[:3]]))  # the strike where call = put sits at the forward
        near = np.argsort(np.abs(K - guess))[:12]
        slope = np.polyfit(K[near], (c - p)[near], 1)[0]
        df = -slope
        if 0.80 < df < 1.0:
            out.append(-math.log(df) / (days / 365.0))
    r = float(np.median(out)) if out else float("nan")
    return r if (0.0 <= r <= 0.15) else float("nan")


def _surface(day: date, o: pd.DataFrame, r: float) -> Surface | None:
    spot_file = float(o.und.median()) if o.und.notna().any() else float("nan")
    expiries, Ts, Fs = [], [], []
    rows = []
    for xp, g in o.groupby("xp"):
        days = (xp - day).days
        if days < MIN_DAYS:
            continue
        T = days / 365.0
        K, c, p = _pairs(g)
        if len(K) < 3:
            continue
        ref = spot_file if np.isfinite(spot_file) else float(np.median(K[np.argsort(np.abs(c - p))[:3]]))
        near = np.argsort(np.abs(K - ref))[:PARITY_STRIKES]
        F = float(np.median(K[near] + math.exp(r * T) * (c[near] - p[near])))
        if not (0.85 < F / ref < 1.2):  # a broken parity read: skip the expiry rather than price off it
            continue
        ei = len(expiries)
        expiries.append(xp); Ts.append(T); Fs.append(F)
        otm = g[(g.px >= MIN_PRICE) & (g.vol > 0)]
        otm = otm[(otm.call & (otm.K >= F)) | (~otm.call & (otm.K < F))]
        otm = otm[np.abs(np.log(otm.K / F)) <= MAX_LOG_MONEYNESS]
        for k, call, px, vol, oi in zip(otm.K, otm.call, otm.px, otm.vol, otm.oi):
            rows.append((k, ei, bool(call), px, vol, oi))
    if not expiries or not rows:
        return None
    K, e, call, px, vol, oi = (np.array(c) for c in zip(*rows))
    e = e.astype(int); call = call.astype(bool)
    T, F = np.array(Ts), np.array(Fs)
    iv = black_iv(px, F[e], K, T[e], r, call)
    keep = np.isfinite(iv) & (iv > 0.03) & (iv < 2.0)
    if keep.sum() < 8:
        return None
    K, e, call, px, vol, oi, iv = K[keep], e[keep], call[keep], px[keep], vol[keep], oi[keep], iv[keep]
    used = np.unique(e)
    remap = {int(old): new for new, old in enumerate(used)}
    e = np.array([remap[int(i)] for i in e])
    T, F = T[used], F[used]
    if np.isfinite(spot_file):
        spot, source = spot_file, "file"
    else:
        spot, source = float(F[0] * math.exp(-r * T[0])), "proxy"
    q = r - np.log(F / spot) / T
    vega = black_vega(F[e], K, T[e], r, iv) / 100.0
    return Surface(day, spot, r, T, F, q, [expiries[i] for i in used], K, e, call, px, iv, vega, vol, oi, source)


def load_day(day: date, r: float | None = None) -> Surface | None:
    """One day's surface. r defaults to that day's own implied rate (not the causal smoothed one: use load_all for tests)."""
    o = _read(day)
    if o is None:
        return None
    if r is None:
        r = implied_rate(o, day)
        r = r if np.isfinite(r) else DEFAULT_RATE
    return _surface(day, o, r)


def _raw_rate(day: date) -> float:
    o = _read(day)
    return float("nan") if o is None else implied_rate(o, day)


def smooth_rates(days: list[date], raw: np.ndarray) -> np.ndarray:
    """Trailing median of the daily readings (today and the previous RATE_SMOOTH_DAYS - 1): causal, so no look-ahead."""
    out = np.empty(len(raw))
    last = DEFAULT_RATE
    for i in range(len(raw)):
        w = raw[max(0, i - RATE_SMOOTH_DAYS + 1): i + 1]
        w = w[np.isfinite(w)]
        last = float(np.median(w)) if w.size else last
        out[i] = last
    return out


def trading_days(start: str | None = None, end: str | None = None) -> list[date]:
    days = sorted(date.fromisoformat(p.stem) for p in RAW.glob("*.csv"))
    if start:
        days = [d for d in days if d >= date.fromisoformat(start)]
    if end:
        days = [d for d in days if d <= date.fromisoformat(end)]
    return days


def _build(day: date, r: float):
    o = _read(day)
    return None if o is None else _surface(day, o, r)


def load_all(start: str | None = None, end: str | None = None, workers: int = 9, cache: bool = True) -> list[Surface]:
    """Every day's surface, oldest first (days without a usable surface are skipped), each priced at its causal implied rate.
    Built once and cached (data/cache/surfaces.pkl, rebuilt when a day's file is added), because building takes minutes."""
    import pickle
    path = ROOT / "data" / "cache" / "surfaces.pkl"
    days_all = trading_days()
    stamp = (len(days_all), days_all[0], days_all[-1])
    out = None
    if cache and path.exists():
        saved = pickle.load(open(path, "rb"))
        if saved["stamp"] == stamp:
            out = saved["surfaces"]
    if out is None:
        from joblib import Parallel, delayed
        raw = np.array(Parallel(n_jobs=workers)(delayed(_raw_rate)(d) for d in days_all))
        rates = smooth_rates(days_all, raw)
        out = [s for s in Parallel(n_jobs=workers)(delayed(_build)(d, r) for d, r in zip(days_all, rates)) if s is not None]
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({"stamp": stamp, "surfaces": out}, open(path, "wb"))
    lo = date.fromisoformat(start) if start else None
    hi = date.fromisoformat(end) if end else None
    return [s for s in out if (lo is None or s.day >= lo) and (hi is None or s.day <= hi)]
