"""Fit a model independently to every day (the repo-style recalibration, but in implied-volatility space on every quote) and save it.

    python3 run_daily.py --model dh --start 2025-01-01 --end 2026-10-06 [--tag name]

Days are split into contiguous chunks fitted in parallel; inside a chunk each day starts from yesterday's solution as well as the
flat-volatility corner and random starts. Output: data/results/daily_<model>_<tag>.pkl, a list of one dict per day.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib
from dh2.data import load_all

RESULTS = Path(__file__).resolve().parent / "data" / "results"


def fit_chunk(model, surfaces, seed, n_starts, max_nfev):
    out, prev = [], None
    for k, S in enumerate(surfaces):
        t = time.time()
        try:
            f = calib.fit_day(model, S, n_starts=n_starts if prev is None else max(1, n_starts - 2), seed=seed + k, x0=prev, max_nfev=max_nfev)
            prev = f["u"]
            out.append({"day": S.day, "u": f["u"], "x": f["x"], "rmse": f["rmse"], "cost": f["cost"], "at_bound": f["at_bound"],
                        "n": f["n"], "sec": time.time() - t})
        except Exception as e:  # keep going: a failed day is recorded, not fatal
            out.append({"day": S.day, "error": repr(e)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bs", "heston", "dh", "dhj"])
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--tag", default="all")
    ap.add_argument("--workers", type=int, default=9); ap.add_argument("--chunks", type=int, default=36)
    ap.add_argument("--starts", type=int, default=4); ap.add_argument("--nfev", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    S = load_all(a.start, a.end)
    print(f"{len(S)} surfaces {S[0].day}..{S[-1].day}; fitting {a.model} on {a.workers} workers", flush=True)
    chunks = [c for c in np.array_split(np.arange(len(S)), min(a.chunks, len(S))) if len(c)]
    res = Parallel(n_jobs=a.workers, verbose=5)(delayed(fit_chunk)(a.model, [S[i] for i in c], 1000 * j, a.starts, a.nfev) for j, c in enumerate(chunks))
    flat = [r for chunk in res for r in chunk]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"daily_{a.model}_{a.tag}.pkl"
    pickle.dump(flat, open(path, "wb"))
    ok = [r for r in flat if "rmse" in r]
    print(f"done in {time.time() - t0:.0f}s: {len(ok)}/{len(flat)} days fitted; in-sample IV rmse median {np.median([r['rmse'] for r in ok]):.3f} pts -> {path}")


if __name__ == "__main__":
    main()
