"""The head-to-head: every method scored on the SAME days, next-day implied-volatility error in vol points.

    python3 tournament.py --start 2025-01-01 --daily test2025 --walk y2526 --models dh,dhj

Methods: yesterday's smile (model-free), each model's independent daily fit carried forward (the repo-style recalibration), and
each model's 2.0 walk-forward calibration (shared structural parameters + daily state; 'carry' = today's state, 'revert' = the
model's own one-day mean-reversion forecast). A paired comparison against the model-free benchmark is given with a 95%
moving-block bootstrap interval (blocks of 20 days), so a difference is only called a difference if the interval excludes zero.
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


def boot(diff, block=20, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    m = [np.mean(np.concatenate([d[s:s + block] for s in rng.integers(0, max(1, len(d) - block), size=max(1, len(d) // block))])) for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--daily", default=None); ap.add_argument("--walk", default=None)
    ap.add_argument("--models", default="dh,dhj")
    a = ap.parse_args()
    S = load_all(a.start, a.end)
    pairs = {i: j for i, j in EP.consecutive(S)}
    methods: dict[str, dict[int, dict]] = {"yesterday's smile (no model)": {i: EP.score(EP.pred_last_smile(S[i], S[j]), S[j]) for i, j in pairs.items()}}
    byday = {s.day: i for i, s in enumerate(S)}
    for m in a.models.split(","):
        if a.daily:
            try:
                fits = {r["day"]: r for r in pickle.load(open(RESULTS / f"daily_{m}_{a.daily}.pkl", "rb")) if "x" in r}
                methods[f"{m}: independent daily fit, carried"] = {
                    i: EP.score(EP.pred_model(m, fits[S[i].day]["x"], S[j]), S[j]) for i, j in pairs.items() if S[i].day in fits}
            except FileNotFoundError:
                pass
        if a.walk:
            try:
                w = pickle.load(open(RESULTS / f"walk_{m}_{a.walk}.pkl", "rb"))
                # S is indexed within this run's date range; the walk file indexes its own surfaces by day
                for name in ("carry", "revert"):
                    methods[f"{m}: 2.0 structural + state ({name})"] = {byday[r["day"]]: r[f"score_{name}"] for r in w if f"score_{name}" in r and r["day"] in byday}
            except FileNotFoundError:
                pass
    common = sorted(set.intersection(*[set(v) for v in methods.values()]))
    print(f"{len(common)} common days, {S[common[0]].day}..{S[common[-1]].day}\n")
    ref = np.array([methods["yesterday's smile (no model)"][i]["rmse"] for i in common])
    print(f"{'method':46s} {'mean':>6s} {'median':>7s} {'p90':>6s} {'core mean':>10s}   {'mean difference vs no-model, 95% interval':>42s}")
    for name, v in methods.items():
        r = np.array([v[i]["rmse"] for i in common]); c = np.array([v[i]["core"] for i in common])
        d, lo, hi = boot(r - ref)
        tag = "" if name.startswith("yesterday") else f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
        print(f"{name:46s} {np.nanmean(r):6.3f} {np.nanmedian(r):7.3f} {np.nanpercentile(r, 90):6.3f} {np.nanmean(c):10.3f}   {tag:>42s}")
    print("\nby year (mean next-day IV rmse):")
    years = sorted({S[i].day.year for i in common})
    print(f"{'method':46s} " + " ".join(f"{y:>6d}" for y in years))
    for name, v in methods.items():
        print(f"{name:46s} " + " ".join(f"{np.nanmean([v[i]['rmse'] for i in common if S[i].day.year == y]):6.2f}" for y in years))


if __name__ == "__main__":
    main()
