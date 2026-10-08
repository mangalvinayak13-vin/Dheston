"""Why 'the model does not work', measured. Next-day out-of-sample implied-volatility error for each way of carrying
yesterday's calibration to today, on NIFTY option quotes.

    python3 diagnose.py --start 2025-01-01 --tag test2025 --models bs,heston,dh
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import evalpairs as EP
from dh2.data import load_all

RESULTS = Path(__file__).resolve().parent / "data" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--tag", default="all")
    ap.add_argument("--models", default="bs,heston,dh")
    a = ap.parse_args()
    S = load_all(a.start, a.end)
    byday = {s.day: i for i, s in enumerate(S)}
    fits = {}
    for m in a.models.split(","):
        fits[m] = {r["day"]: r for r in pickle.load(open(RESULTS / f"daily_{m}_{a.tag}.pkl", "rb")) if "x" in r}
    pairs = EP.consecutive(S)
    rows = {"last smile (model-free)": [], "BS per expiry (yesterday's ATM)": []}
    for m in fits:
        rows[f"{m}: yesterday's daily fit carried"] = []
    for i, j in pairs:
        S0, S1 = S[i], S[j]
        rows["last smile (model-free)"].append(EP.score(EP.pred_last_smile(S0, S1), S1))
        rows["BS per expiry (yesterday's ATM)"].append(EP.score(EP.pred_atm_by_expiry(S0, S1), S1))
        for m, f in fits.items():
            r = f.get(S0.day)
            rows[f"{m}: yesterday's daily fit carried"].append(
                EP.score(EP.pred_model(m, r["x"], S1), S1) if r is not None else {"rmse": np.nan, "core": np.nan, "bias": np.nan, "n": 0})
    print(f"{len(pairs)} consecutive-day pairs, {S[pairs[0][0]].day}..{S[pairs[-1][1]].day}\n")
    print(f"{'next-day prediction':42s} {'all quotes: mean / median / p90':>34s} {'near-ATM core: mean / median':>32s}")
    for k, v in rows.items():
        a1, c1 = EP.summarize(v, "rmse"), EP.summarize(v, "core")
        print(f"{k:42s} {a1['mean']:9.2f} {a1['median']:9.2f} {a1['p90']:9.2f}   {c1['mean']:12.2f} {c1['median']:9.2f}")
    # in-sample, and how much the parameters move
    for m, f in fits.items():
        days = [d for d in sorted(f)]
        rm = np.array([f[d]["rmse"] for d in days])
        print(f"\n{m}: in-sample IV rmse median {np.median(rm):.2f} pts; at a bound on {100 * np.mean([len(f[d]['at_bound']) > 0 for d in days]):.0f}% of days")
        if m in ("heston", "dh"):
            X = np.array([f[d]["x"] for d in days])
            jump = np.abs(np.diff(np.log(np.maximum(np.abs(X), 1e-6)), axis=0))
            print("   median day-to-day change in log|parameter|:", np.round(np.median(jump, axis=0), 2).tolist())


if __name__ == "__main__":
    main()
