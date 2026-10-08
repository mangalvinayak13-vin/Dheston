"""If you follow the model, how many individual trades profit out of 100?

    python3 trade_test.py [--model dhj --tag main --top 5 --cost 0.01]

THE TRADE (one option, one day)
  Day t, at the close: the model has been calibrated on days <= t (the 2.0 walk-forward). For each of the `top` most-traded liquid option
  contracts of that day (out of the money, within 5% of the forward, 5 to 45 days to expiry) it forecasts the contract's implied volatility for
  tomorrow, assuming the index does not move (nothing about tomorrow is used). If the forecast is above today's market volatility by more than a
  threshold, BUY the option at today's close; if below, SELL it. Close the position at tomorrow's close, only for contracts that traded both days.
  Two signals: 'forecast' = the model's own one-day volatility forecast (mean reversion and the passage of time on today's surface);
  'rich/cheap' = market volatility minus the model's fitted volatility today (sell what the model calls rich, buy what it calls cheap).
  Two ways of holding: 'plain' = just the option; 'hedged' = the option with the index move hedged out through the forward of the same expiry
  (the option's delta times the forward's change), which isolates volatility from the direction of the market.

WHAT COUNTS AS A WIN: profit after costs is greater than zero. Cost = a percentage of the option's price per side (entry and exit), run at several
levels because the closing-price data has no bid and ask. Controls: random buy/sell, always buy, always sell (same contracts, same days).
Settings (the signal threshold) are picked on 2016-2021 only; 2022-2026 is reported.
"""
import argparse
import dataclasses
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, models as M, struct as ST
from dh2.data import _read, black_iv, load_all

HERE = Path(__file__).resolve().parent
RES = HERE / "data" / "results"
SPLIT = date(2022, 1, 1)
COSTS = [0.0, 0.01, 0.02, 0.05]       # per side, as a share of the option's price


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def delta_f(F, K, T, r, iv, is_call):
    """Derivative of a Black-76 option's price with respect to the forward (discount factor included)."""
    sd = iv * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    return np.exp(-r * T) * np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)


def trades_for_day(S0, S1, rec, model, top, Sa=None):
    """All candidate trades of one day t -> t+1 (before choosing a direction): contract, today's/tomorrow's price, signals, hedge P&L pieces."""
    # the signal always looks one trading day ahead, lagged or not, so both tests pick exactly the same contracts and signals
    gap = ((S1 if Sa is None else Sa).day - S0.day).days / 365.0
    eta, state = rec["eta"], rec["state"]
    x = lambda st: calib.to_phys(model, ST.assemble(model, eta, st))
    m0 = np.log(S0.K / S0.F[S0.e])
    Td = S0.T[S0.e] * 365
    ok = (np.abs(m0) <= 0.05) & (Td >= 5) & (Td <= 45) & (S0.volume > 0)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    idx = idx[np.argsort(-S0.volume[idx])][:top]
    # the same surface one day later at an UNCHANGED forward (nothing about tomorrow's index level is used)
    T1 = S0.T - gap
    keep_exp = T1 > 2 / 365
    S0s = dataclasses.replace(S0, T=np.where(keep_exp, T1, S0.T), q=np.where(keep_exp, S0.r - np.log(S0.F / S0.spot) / np.where(keep_exp, T1, S0.T), S0.q))
    S0s._groups = None
    iv_now = M.model_iv(M.price(model, x(state), S0), S0)
    rev = ST.expected_state(model, eta, state, gap)
    iv_next = M.model_iv(M.price(model, x(rev), S0s), S0s)
    # lagged test (Sa is not S0): the signal is from day t but the trade is entered at the NEXT day's close and exited a day later,
    # so a misfit that is only the bounce of one closing trade cannot earn anything
    Sa = S0 if Sa is None else Sa
    books = []
    for Sx in ((Sa, S1) if Sa is not S0 else (S1,)):
        raw = _read(Sx.day)
        if raw is None:
            return []
        books.append({(r.xp, float(r.K), bool(r.call)): (float(r.px), float(r.vol)) for r in raw.itertuples()})
    out = []
    for i in idx:
        e = S0.e[i]
        if not keep_exp[e]:
            continue
        key = (S0.expiry[e], float(S0.K[i]), bool(S0.is_call[i]))
        nxt = books[-1].get(key)
        ea = next((k for k, d in enumerate(Sa.expiry) if d == S0.expiry[e]), None)
        e1 = next((k for k, d in enumerate(S1.expiry) if d == S0.expiry[e]), None)
        if nxt is None or nxt[1] <= 0 or nxt[0] < 0.5 or e1 is None or ea is None:
            continue
        if Sa is S0:
            p0, iv0 = float(S0.price[i]), S0.iv[i]
        else:
            ent = books[0].get(key)
            if ent is None or ent[1] <= 0 or ent[0] < 0.5:
                continue
            p0 = ent[0]
            iv0 = float(black_iv(p0, Sa.F[ea], S0.K[i], Sa.T[ea], Sa.r, S0.is_call[i]))
            if not np.isfinite(iv0):
                continue
        dF = delta_f(Sa.F[ea], S0.K[i], Sa.T[ea], Sa.r, iv0, S0.is_call[i])
        out.append({"day": S0.day, "p0": p0, "p1": nxt[0], "dF": float(dF), "dFwd": float(S1.F[e1] - Sa.F[ea]), "key": key,
                    "forecast_edge": float(100 * (iv_next[i] - S0.iv[i])), "rich_cheap_edge": float(100 * (iv_now[i] - S0.iv[i])),
                    "is_call": bool(S0.is_call[i]), "T_days": float(S0.T[e] * 365)})
    return out


def evaluate(df, signal, hedged, thr, cost, rng):
    """Win count etc. for the model's direction (+1 buy, -1 sell) on trades whose |edge| passes thr; controls on the same trades."""
    d = df[np.abs(df[signal]) >= thr]
    if len(d) == 0:
        return None
    gross = (d.p1 - d.p0) - (d.dF * d.dFwd if hedged else 0.0)
    fee = cost * (d.p0 + d.p1)                       # entry and exit, each cost x price
    dirn = np.sign(d[signal])
    res = {}
    for name, sgn in (("model", dirn.values), ("always buy", np.ones(len(d))), ("always sell", -np.ones(len(d)))):
        net = sgn * gross.values - fee.values
        wins = int((net > 0).sum())
        lo, hi = wilson(wins, len(d))
        res[name] = {"trades": len(d), "wins_per_100": 100 * wins / len(d), "ci95_per_100": [100 * lo, 100 * hi],
                     "avg_win": float(net[net > 0].mean()) if wins else 0.0, "avg_loss": float(net[net <= 0].mean()) if wins < len(d) else 0.0,
                     "expectancy": float(net.mean()), "expectancy_pct_of_premium": float(100 * np.mean(net / d.p0.values)),
                     "profit_factor": float(net[net > 0].sum() / max(-net[net <= 0].sum(), 1e-9))}
    # random buy/sell, exactly: each trade is a buy or a sell with probability one half
    res["random"] = {k: (res["always buy"][k] + res["always sell"][k]) / 2 for k in ("wins_per_100", "expectancy", "expectancy_pct_of_premium")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dhj"); ap.add_argument("--tag", default="main"); ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--lag", type=int, default=0, choices=[0, 1], help="1 = enter at the close one day after the signal (bid-ask bounce check)")
    a = ap.parse_args()
    S = load_all()
    byday = {s.day: i for i, s in enumerate(S)}
    W = {r["day"]: r for r in pickle.load(open(RES / f"walk_{a.model}_{a.tag}.pkl", "rb"))}
    sfx = f"{a.model}_{a.tag}" + ("_lag1" if a.lag else "")
    rows = []
    prog = HERE / "logs" / "progress"
    prog.mkdir(parents=True, exist_ok=True)
    pf = prog / f"trade_{sfx}_00.json"
    L = a.lag
    days = [d for d in sorted(W) if byday.get(d) is not None and byday[d] + 1 + L < len(S) and (S[byday[d] + 1 + L].day - S[byday[d]].day).days <= 5 + 2 * L]
    for n, d in enumerate(days):
        i = byday[d]
        rows += trades_for_day(S[i], S[i + 1 + L], W[d], a.model, a.top, Sa=S[i + 1] if L else None)
        if n % 5 == 0 or n == len(days) - 1:          # a tiny file the live dashboard reads
            pf.write_text('{"done": %d, "total": %d}' % (n + 1, len(days)))
    df = pd.DataFrame(rows)
    df.to_pickle(RES / f"trades_{sfx}.pkl")
    dev, test = df[df.day < SPLIT], df[df.day >= SPLIT]
    print(f"{len(df):,} candidate trades ({len(dev):,} dev, {len(test):,} test), {df.day.nunique()} days; top {a.top} contracts a day by volume")
    out = {}
    rng = np.random.default_rng(0)
    for signal in ("forecast_edge", "rich_cheap_edge"):
        for hedged in (False, True):
            tag = f"{signal} / {'hedged' if hedged else 'plain'}"
            # the threshold is the only setting: chosen on the development years by expectancy at a 1% cost
            best = max([0.0, 0.25, 0.5, 1.0], key=lambda th: (evaluate(dev, signal, hedged, th, 0.01, rng) or {"model": {"expectancy_pct_of_premium": -1e9}})["model"]["expectancy_pct_of_premium"])
            out[tag] = {"threshold_vol_points": best}
            print(f"\n== {tag}: threshold {best} vol points (chosen on 2016-2021)")
            print(f"   {'cost/side':>9s} {'trades':>7s} {'model wins/100':>15s} {'[95% range]':>14s} {'random':>7s} {'always buy':>11s} {'always sell':>12s} {'model avg win':>14s} {'avg loss':>9s} {'expectancy % prem':>18s} {'profit factor':>14s}")
            for c in COSTS:
                r = evaluate(test, signal, hedged, best, c, rng)
                if r is None:
                    continue
                m = r["model"]
                out[tag][f"cost_{c}"] = r
                print(f"   {100 * c:8.0f}% {m['trades']:7d} {m['wins_per_100']:15.1f} [{m['ci95_per_100'][0]:5.1f},{m['ci95_per_100'][1]:5.1f}] {r['random']['wins_per_100']:7.1f} {r['always buy']['wins_per_100']:11.1f} "
                      f"{r['always sell']['wins_per_100']:12.1f} {m['avg_win']:14.2f} {m['avg_loss']:9.2f} {m['expectancy_pct_of_premium']:18.2f} {m['profit_factor']:14.2f}")
    (RES / f"trade_test_{sfx}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
