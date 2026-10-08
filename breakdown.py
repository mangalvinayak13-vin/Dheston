"""Where does a forecast win and lose? Next-day implied-vol error pooled by expiry length, moneyness, option side and market regime.

    python3 breakdown.py --model dhj --tag main [--period test|dev|all]

Regime = the day's 30-day at-the-money volatility against its own expanding median up to that day (calm below it, stressed above). Errors
are pooled over quotes (root mean square, vol points), so a bucket with many quotes counts for what it is. Writes
data/results/breakdown_<model>_<tag>.json for the report.
"""
import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import evalpairs as EP, memory as MEM
from dh2.data import load_all

RESULTS = Path(__file__).resolve().parent / "data" / "results"
SPLIT = date(2022, 1, 1)
EXPIRY = [("3-7 days", 3, 7), ("8-14", 8, 14), ("15-35", 15, 35), ("36-100", 36, 100), ("100+ days", 101, 10000)]
MONEY = [("|m| < 2%", 0, 0.02), ("2-5%", 0.02, 0.05), ("5-10%", 0.05, 0.10), ("10-22%", 0.10, 0.23)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dhj"); ap.add_argument("--tag", default="main"); ap.add_argument("--period", default="test")
    a = ap.parse_args()
    S = load_all()
    hyb = pickle.load(open(RESULTS / f"hyb_{a.model}_{a.tag}.pkl", "rb"))
    ts = sorted(hyb["smile"])
    ts = [t for t in ts if (a.period == "all") or (a.period == "test") == (S[t].day >= SPLIT)]
    atm = np.array([MEM._civ(s, 30 / 365, 0.0) for s in S])
    stressed = np.array([bool(i > 120 and np.isfinite(atm[i]) and atm[i] > np.nanmedian(atm[:i + 1])) for i in range(len(S))])
    preds = {"smile_scaled": lambda t: EP.pred_smile_scaled(S[t], S[t + 1]), f"{a.model} carry": lambda t: hyb["carry"][t].astype(float),
             f"{a.model} carry_res": lambda t: hyb["carry_res"][t].astype(float)}
    acc = {k: {"e": [], "T": [], "m": [], "call": [], "stress": []} for k in preds}
    for t in ts:
        S1 = S[t + 1]
        m, T = np.log(S1.K / S1.F[S1.e]), S1.T[S1.e] * 365
        for k, f in preds.items():
            e = 100 * (f(t) - S1.iv)
            ok = np.isfinite(e)
            for key, v in (("e", e[ok]), ("T", T[ok]), ("m", np.abs(m[ok])), ("call", S1.is_call[ok]), ("stress", np.full(ok.sum(), stressed[t]))):
                acc[k][key].append(v)
    A = {k: {kk: np.concatenate(vv) for kk, vv in d.items()} for k, d in acc.items()}
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
    out = {}
    print(f"{a.period}: {len(ts)} days, {len(A['smile_scaled']['e']):,} quotes; pooled next-day IV rmse (vol points)")
    for title, buckets, field in (("expiry length", EXPIRY, "T"), ("moneyness |ln K/F|", MONEY, "m")):
        print(f"\nby {title}:")
        print(f"{'':14s} {'quotes':>9s} " + " ".join(f"{k:>14s}" for k in A))
        for name, lo, hi in buckets:
            row = {}
            for k, d in A.items():
                s = (d[field] >= lo) & (d[field] <= hi)
                row[k] = rmse(d["e"][s])
                n = int(s.sum())
            out.setdefault(title, {})[name] = {"quotes": n, **row}
            print(f"{name:14s} {n:9,d} " + " ".join(f"{row[k]:14.3f}" for k in A))
    for title, mask in (("calls", lambda d: d["call"]), ("puts", lambda d: ~d["call"]), ("calm days", lambda d: ~d["stress"]), ("stressed days", lambda d: d["stress"])):
        row = {k: rmse(d["e"][mask(d)]) for k, d in A.items()}
        n = int(mask(A["smile_scaled"]).sum())
        out.setdefault("side / regime", {})[title] = {"quotes": n, **row}
        print(f"{title:14s} {n:9,d} " + " ".join(f"{row[k]:14.3f}" for k in A))
    (RESULTS / f"breakdown_{a.model}_{a.tag}_{a.period}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
