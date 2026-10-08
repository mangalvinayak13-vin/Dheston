"""Score the stored forecast variants against the model-free benchmarks, by year and on the development / test split.

    python3 score_variants.py --model dhj --tag main

Benchmarks (no model): 'smile' = yesterday's smile in log-moneyness; 'smile_scaled' = the same read in sqrt(T)-scaled moneyness, so the
expiry being a day closer is accounted for. The paired difference of a variant against the BEST benchmark is reported with a 95%
moving-block bootstrap interval on the test period (2022 onward), after variants were chosen using the development period only.
"""
import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import evalpairs as EP
from dh2.data import load_all
from tournament import boot

RESULTS = Path(__file__).resolve().parent / "data" / "results"
SPLIT = date(2022, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dhj"); ap.add_argument("--tag", default="main")
    ap.add_argument("--extra", default="", help="comma list of extra variant files hyb_<name>.pkl to merge (name:variant)")
    a = ap.parse_args()
    S = load_all()
    hyb = pickle.load(open(RESULTS / f"hyb_{a.model}_{a.tag}.pkl", "rb"))
    ts = sorted(hyb["smile"])
    scaled = {t: EP.pred_smile_scaled(S[t], S[t + 1]) for t in ts}
    preds = {"smile (no model)": {t: hyb["smile"][t].astype(float) for t in ts}, "smile_scaled (no model)": scaled}
    for v in ("carry", "carry_res", "lev_res"):
        preds[f"{a.model}: {v}"] = {t: hyb[v][t].astype(float) for t in ts}
    for spec in [x for x in a.extra.split(",") if x]:
        nm, v = spec.split(":")
        h = pickle.load(open(RESULTS / f"hyb_{nm}.pkl", "rb"))
        preds[spec] = {t: h[v][t].astype(float) for t in ts if t in h[v]}
    common = sorted(set.intersection(*[set(p) for p in preds.values()]))
    sc = {k: {t: EP.score(p[t], S[t + 1]) for t in common} for k, p in preds.items()}
    years = sorted({S[t].day.year for t in common})
    print(f"{len(common)} days {S[common[0]].day}..{S[common[-1]].day}\nmean next-day IV rmse (vol points):")
    print(f"{'':28s} {'all':>6s} {'dev':>6s} {'test':>6s} {'core':>6s} " + " ".join(f"{y:>5d}" for y in years))
    dev = [t for t in common if S[t].day < SPLIT]
    test = [t for t in common if S[t].day >= SPLIT]
    for k in preds:
        r = lambda ix, key="rmse": np.nanmean([sc[k][t][key] for t in ix])
        print(f"{k:28s} {r(common):6.3f} {r(dev):6.3f} {r(test):6.3f} {r(common, 'core'):6.3f} " +
              " ".join(f"{r([t for t in common if S[t].day.year == y]):5.2f}" for y in years))
    bench = min(("smile (no model)", "smile_scaled (no model)"), key=lambda k: np.nanmean([sc[k][t]["rmse"] for t in test]))
    print(f"\nTEST period ({len(test)} days), paired difference against the better benchmark '{bench}' (negative = better):")
    for k in preds:
        if k == bench:
            continue
        d = [sc[k][t]["rmse"] - sc[bench][t]["rmse"] for t in test]
        m, lo, hi = boot(d)
        print(f"  {k:28s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
