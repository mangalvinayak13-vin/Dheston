"""Robust read of the trading test (no costs): is the average profit real, or a handful of lucky trades / bid-ask bounce?

    python3 trade_robust.py [trades_dhj_main.pkl trades_dhj_main_lag1.pkl ...]

For each saved trade file, test years 2022-2026, thresholds as chosen on 2016-2021 by trade_test.py:
  wins per 100 (model vs always buy / always sell on the SAME buy/sell mix), profit per trade in rupees (mean, median, 1%-trimmed mean),
  share of total profit from the 5 best trades, profit by year, and day-resampled 95% ranges (trades of one day move together).
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "data" / "results"
SPLIT = date(2022, 1, 1)
COSTS = [0.0, 0.005, 0.01, 0.02]      # per side (entry and exit), as a share of the option's price; 1% is about realistic for these contracts


def trimmed(a, p=0.01):
    lo, hi = np.quantile(a, [p, 1 - p])
    return float(a[(a >= lo) & (a <= hi)].mean())


def day_boot(net, days, n=2000, seed=1):
    u = np.unique(days)
    grp = [net[days == k] for k in u]
    rng = np.random.default_rng(seed)
    w, m = [], []
    for _ in range(n):
        a = np.concatenate([grp[j] for j in rng.integers(0, len(u), len(u))])
        w.append(100 * (a > 0).mean()); m.append(a.mean())
    return np.percentile(w, [2.5, 97.5]), np.percentile(m, [2.5, 97.5])


def report(fname):
    df = pd.read_pickle(RES / fname)
    test = df[df.day >= SPLIT]
    stem = fname.replace("trades_", "").replace(".pkl", "")
    cfg = json.loads((RES / f"trade_test_{stem}.json").read_text()) if (RES / f"trade_test_{stem}.json").exists() else {}
    out = {}
    print(f"\n######## {fname}: {len(test):,} test trades on {test.day.nunique()} days")
    for signal in ("forecast_edge", "rich_cheap_edge"):
        for hedged in (False, True):
            tag = f"{signal} / {'hedged' if hedged else 'plain'}"
            thr = cfg.get(tag, {}).get("threshold_vol_points", 0.0)
            x = test[np.abs(test[signal]) >= thr]
            s = np.sign(x[signal]).values
            gross = (x.p1 - x.p0).values - ((x.dF * x.dFwd).values if hedged else 0.0)
            buy = s > 0
            yr = x.day.map(lambda d: d.year).values
            out[tag] = {"threshold": thr, "trades": int(len(x)), "buy_share": float(buy.mean())}
            print(f"\n== {tag} (threshold {thr}): {len(x)} trades, {100 * buy.mean():.0f}% buys")
            print(f"   {'cost/side':>9s} {'wins/100':>9s} {'[95% range]':>13s} {'blind same mix':>15s} {'buys win':>9s} {'sells win':>10s} {'Rs/trade':>9s} {'[95% range]':>16s} {'median':>7s} {'trimmed':>8s} {'best 5':>7s}  total Rs by year")
            for cost in COSTS:
                fee = cost * (x.p0 + x.p1).values             # entry and exit, each cost x the option's price
                net = s * gross - fee
                # blind trades with the model's own buy/sell mix: each buy wins at the always-buy rate, each sell at the always-sell rate
                mix = 100 * (buy.mean() * (gross - fee > 0).mean() + (~buy).mean() * (-gross - fee > 0).mean())
                (wl, wh), (ml, mh) = day_boot(net, x.day.values)
                top5 = np.sort(net)[-5:].sum() / net.sum() if net.sum() > 0 else float("nan")
                yrs = {int(y): round(float(net[yr == y].sum())) for y in sorted(set(yr))}
                r = {"wins_per_100": float(100 * (net > 0).mean()), "wins_ci": [float(wl), float(wh)], "blind_same_mix_wins_per_100": float(mix),
                     "buy_wins": float(100 * (net[buy] > 0).mean()), "sell_wins": float(100 * (net[~buy] > 0).mean()) if (~buy).any() else None,
                     "mean_rs": float(net.mean()), "mean_rs_ci": [float(ml), float(mh)], "median_rs": float(np.median(net)), "trimmed_rs": trimmed(net),
                     "total_rs": float(net.sum()), "top5_share": float(top5), "by_year_rs": yrs}
                out[tag][f"cost_{cost}"] = r
                print(f"   {100 * cost:8.1f}% {r['wins_per_100']:9.1f} [{wl:5.1f},{wh:5.1f}] {mix:15.1f} {r['buy_wins']:9.1f} {r['sell_wins'] or float('nan'):10.1f} "
                      f"{r['mean_rs']:9.2f} [{ml:6.2f},{mh:6.2f}] {r['median_rs']:7.2f} {r['trimmed_rs']:8.2f} {100 * top5:6.0f}%  {yrs}")
    (RES / f"trade_robust_{stem}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for f in sys.argv[1:] or ["trades_dhj_main.pkl"]:
        report(f)
