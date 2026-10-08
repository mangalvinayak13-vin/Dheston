"""Does pattern memory help? Test it on any base predictor: add the correction learned from how that predictor erred after old,
similar patterns, and compare next-day error with and without it.

    python3 eval_memory.py --base smile            # the model-free 'yesterday's smile' predictor, all ten years
    python3 eval_memory.py --base walk:dhj:y2526   # a calibrated model's walk-forward predictions (data/results/walk_*.pkl)

Memory settings (k, q, L, beta) are picked on the DEVELOPMENT period only and the TEST period is then scored once with them.
"""
import argparse
import itertools
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import evalpairs as EP, memory as MEM
from dh2.data import load_all

RESULTS = Path(__file__).resolve().parent / "data" / "results"
SPLIT = date(2022, 1, 1)           # test period starts here; nothing below is tuned on it


def base_predictions(S, spec):
    """{t: predicted IV array for the quotes of S[t+1]} for the chosen base predictor."""
    if spec == "smile":
        return {i: EP.pred_last_smile(S[i], S[j]) for i, j in EP.consecutive(S)}
    if spec == "smile_scaled":
        return {i: EP.pred_smile_scaled(S[i], S[j]) for i, j in EP.consecutive(S)}
    if spec.startswith("hyb:"):    # hyb:<model>:<tag>:<variant> -> the stored forecast variants from hybrids.py
        _, model, tag, variant = spec.split(":")
        return {t: p.astype(float) for t, p in pickle.load(open(RESULTS / f"hyb_{model}_{tag}.pkl", "rb"))[variant].items()}
    kind, model, tag = spec.split(":")
    key = "pred_revert" if kind == "walk" else "pred_carry"
    byday = {s.day: i for i, s in enumerate(S)}   # the walk file indexes its own date range; key everything by day
    return {byday[r["day"]]: r[key].astype(float) for r in pickle.load(open(RESULTS / f"walk_{model}_{tag}.pkl", "rb"))
            if key in r and r["day"] in byday}


def geometry(S1):
    return np.log(S1.K / S1.F[S1.e]), S1.T[S1.e]


def run(S, feat, preds, k, q, L, betas, days):
    """{beta: [score per day]} and the share of days the memory fired, over the given day indices t (predicting S[t+1])."""
    mem = MEM.Memory(feat, L=L, k=k, q=q)
    err = {}
    for t, p in preds.items():
        S1 = S[t + 1]
        e = S1.iv - p
        err[t + 1] = (*geometry(S1), np.where(np.isfinite(e), e, np.nan))
    out = {b: [] for b in betas}
    base, fired, n = [], 0, 0
    for t in days:
        if t not in preds:
            continue
        S1 = S[t + 1]
        n += 1
        res = mem.match(t)
        corr = np.zeros(S1.n)
        if res is not None:
            ends, w, d0, tau = res
            old = [err[i + 1] for i in ends if (i + 1) in err]
            ww = np.array([wi for i, wi in zip(ends, w) if (i + 1) in err])
            if old and ww.sum() > 0:
                corr = MEM.correct(geometry(S1), old, ww / ww.sum())
                fired += 1
        base.append(EP.score(preds[t], S1))
        for b in betas:
            out[b].append(EP.score(preds[t] + b * corr, S1))
    return base, out, fired / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="smile")
    ap.add_argument("--grid", default="wide", choices=["narrow", "wide"])
    a = ap.parse_args()
    S = load_all()
    print(f"{len(S)} surfaces; computing features ...", flush=True)
    feat = MEM.features(S)
    preds = base_predictions(S, a.base)
    ts = sorted(preds)
    dev = [t for t in ts if S[t].day < SPLIT]
    test = [t for t in ts if S[t].day >= SPLIT]
    betas = [0.5, 1.0] if a.grid == "narrow" else [1.0, 1.5]
    print(f"base '{a.base}': {len(dev)} development days, {len(test)} test days", flush=True)
    best = None
    print("\ndevelopment period (choose the memory settings here):")
    grid = itertools.product([5, 10], [0.02, 0.05, 0.10], [3, 5]) if a.grid == "narrow" else itertools.product([10, 20, 40], [0.1, 0.2, 0.4], [1, 2, 3])
    for k, q, L in grid:
        base, out, fire = run(S, feat, preds, k, q, L, betas, dev)
        b0 = EP.summarize(base)["mean"]
        for b in betas:
            m = EP.summarize(out[b])["mean"]
            print(f"  k={k:2d} q={q:.2f} L={L} beta={b}: mean rmse {m:.4f} vs base {b0:.4f} ({m - b0:+.4f}), memory fired on {100 * fire:.0f}% of days")
            if best is None or m < best[0]:
                best = (m, k, q, L, b)
    _, k, q, L, b = best
    print(f"\nchosen on development: k={k} q={q} L={L} beta={b}")
    base, out, fire = run(S, feat, preds, k, q, L, [b], test)
    sb, sm = EP.summarize(base), EP.summarize(out[b])
    diff = np.array([m["rmse"] - x["rmse"] for m, x in zip(out[b], base)])
    diff = diff[np.isfinite(diff)]
    # moving-block bootstrap of the mean paired difference (blocks of 20 days)
    rng = np.random.default_rng(0)
    bs = [np.mean(np.concatenate([diff[s:s + 20] for s in rng.integers(0, max(1, len(diff) - 20), size=max(1, len(diff) // 20))])) for _ in range(2000)]
    print(f"\nTEST {len(test)} days (fired on {100 * fire:.0f}%): without memory mean {sb['mean']:.4f} (median {sb['median']:.4f}); "
          f"with memory {sm['mean']:.4f} (median {sm['median']:.4f}); paired difference {diff.mean():+.4f} "
          f"[95% block-bootstrap interval {np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    # keep the result: the dashboard and the final report read it
    import json
    by_year = {}
    for t, m, x in zip([t for t in test if t in preds], out[b], base):
        by_year.setdefault(S[t].day.year, []).append((x["rmse"], m["rmse"]))
    res = {"base": a.base, "chosen_on_development": {"k": k, "q": q, "L": L, "beta": b, "dev_mean_rmse": best[0]},
           "test": {"days": len(test), "fired_share": fire, "without": sb, "with": sm, "paired_diff": float(diff.mean()),
                    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]},
           "test_by_year": {int(y): {"without": float(np.nanmean([v[0] for v in vs])), "with": float(np.nanmean([v[1] for v in vs]))} for y, vs in sorted(by_year.items())}}
    (RESULTS / f"memory_{a.base.replace(':', '_')}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
