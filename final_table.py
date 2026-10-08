"""One table for every finished run: next-day implied-vol error (vol points) on the SAME days, development years first.

    python3 final_table.py [name ...]        # names are the hyb_<name>.pkl stems; default = all runs

Per run and forecast variant (carry = the model with today's state, carry_res = plus yesterday's misfit): mean error on 2016-2021 (dev) and
2022-2026 (test), and on test the paired difference against the better model-free benchmark with a 95% moving-block bootstrap interval.
Selection of the final configuration uses the dev column only; test is read once, for the chosen configuration and for the record.
"""
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import evalpairs as EP
from dh2.data import load_all
from tournament import boot

RES = Path(__file__).resolve().parent / "data" / "results"
SPLIT = date(2022, 1, 1)
VARIANTS = ("carry", "carry_res")


def main():
    names = sys.argv[1:] or sorted(p.stem[4:] for p in RES.glob("hyb_*.pkl"))
    S = load_all()
    H = {n: pickle.load(open(RES / f"hyb_{n}.pkl", "rb")) for n in names}
    common = sorted(set.intersection(*[set(h["carry"]) for h in H.values()]))
    dev = [t for t in common if S[t].day < SPLIT]
    test = [t for t in common if S[t].day >= SPLIT]
    print(f"{len(common)} common days ({len(dev)} dev, {len(test)} test)\n")
    sm = {t: EP.pred_smile_scaled(S[t], S[t + 1]) for t in common}
    s0 = {t: H[names[0]]["smile"][t].astype(float) for t in common}
    score = lambda p: {t: EP.score(p[t], S[t + 1])["rmse"] for t in common}
    bench = {"smile": score(s0), "smile_scaled": score(sm)}
    best = min(bench, key=lambda k: np.nanmean([bench[k][t] for t in test]))
    m = lambda d, ix: float(np.nanmean([d[t] for t in ix]))
    print(f"{'run':24s} {'variant':10s} {'dev':>7s} {'test':>7s}   test difference vs {best} (negative = better)")
    for k in bench:
        print(f"{k + ' (no model)':24s} {'':10s} {m(bench[k], dev):7.3f} {m(bench[k], test):7.3f}")
    rows = []
    for n in names:
        for v in VARIANTS:
            sc = score({t: H[n][v][t].astype(float) for t in common})
            d = [sc[t] - bench[best][t] for t in test]
            mm, lo, hi = boot(d)
            rows.append((n, v, m(sc, dev), m(sc, test), mm, lo, hi))
            print(f"{n:24s} {v:10s} {m(sc, dev):7.3f} {m(sc, test):7.3f}   {mm:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    best_dev = min((r for r in rows if r[1] == "carry_res"), key=lambda r: r[2])
    print(f"\nbest carry_res on the DEVELOPMENT years: {best_dev[0]} ({best_dev[2]:.3f})")


if __name__ == "__main__":
    main()
