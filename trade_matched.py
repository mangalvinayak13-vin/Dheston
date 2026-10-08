"""Same trades, two entry days: is the edge still there if you enter one day after the signal?

    python3 trade_matched.py

Takes the trades that exist in BOTH trades_dhj_main.pkl (enter at the signal day's close) and trades_dhj_main_lag1.pkl (enter one close later),
matched on (signal day, contract); the signal, side and threshold are identical, only the prices change. Test years 2022-2026, thresholds
as chosen on 2016-2021 by trade_test.py. Reports wins per 100 and rupees per trade (no cost and 1% a side) for each entry day, and the paired
difference with a 95% interval that resamples whole days.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "data" / "results"
SPLIT = date(2022, 1, 1)


def main():
    a = pd.read_pickle(RES / "trades_dhj_main.pkl")
    b = pd.read_pickle(RES / "trades_dhj_main_lag1.pkl")
    cfg = json.loads((RES / "trade_test_dhj_main.json").read_text())
    for d in (a, b):
        d["k"] = d["key"].map(str)
    m = a.merge(b, on=["day", "k"], suffixes=("0", "1"))
    m = m[m.day >= SPLIT]
    print(f"{len(m):,} test trades present in both ({len(m) / len(a[a.day >= SPLIT]):.0%} of the same-day trades); lost to 'must also trade two days later': {len(a[a.day >= SPLIT]) - len(m):,}")
    rng = np.random.default_rng(1)
    out = {}
    for signal in ("forecast_edge", "rich_cheap_edge"):
        for hedged in (False, True):
            tag = f"{signal} / {'hedged' if hedged else 'plain'}"
            thr = cfg[tag]["threshold_vol_points"]
            x = m[np.abs(m[f"{signal}0"]) >= thr]
            s = np.sign(x[f"{signal}0"]).values
            res = {}
            for c in (0.0, 0.01):
                net = []
                for L in ("0", "1"):
                    g = (x[f"p1{L}"] - x[f"p0{L}"]).values - ((x[f"dF{L}"] * x[f"dFwd{L}"]).values if hedged else 0.0)
                    net.append(s * g - c * (x[f"p0{L}"] + x[f"p1{L}"]).values)
                days = x.day.values
                u = np.unique(days)
                idx = {k: np.flatnonzero(days == k) for k in u}
                diffs = []
                for _ in range(2000):
                    ii = np.concatenate([idx[k] for k in rng.choice(u, len(u))])
                    diffs.append(net[0][ii].mean() - net[1][ii].mean())
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                res[f"cost_{c}"] = {"same_day_wins": 100 * float((net[0] > 0).mean()), "next_day_wins": 100 * float((net[1] > 0).mean()),
                                    "same_day_rs": float(net[0].mean()), "next_day_rs": float(net[1].mean()), "lost_rs": float(net[0].mean() - net[1].mean()), "lost_ci": [float(lo), float(hi)]}
            out[tag] = {"threshold": thr, "trades": int(len(x)), **res}
            print(f"\n{tag} (edge >= {thr} vol pts): {len(x)} matched trades")
            for ck, r in res.items():
                print(f"   cost {100 * float(ck[5:]):.0f}%: enter at signal close: {r['same_day_wins']:.1f} wins/100, {r['same_day_rs']:+.2f} Rs/trade | one day later: {r['next_day_wins']:.1f} wins/100, {r['next_day_rs']:+.2f} Rs/trade | lost {r['lost_rs']:+.2f} [{r['lost_ci'][0]:+.2f}, {r['lost_ci'][1]:+.2f}]")
    (RES / "trade_matched_dhj_main.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
