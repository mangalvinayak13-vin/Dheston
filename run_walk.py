"""The 2.0 walk-forward run: structural parameters refitted on a trailing window, a daily state, a one-day-ahead forecast.

    python3 run_walk.py --model dhj --start 2016-10-03 --end 2026-10-06 --tag main [--window 40 --refit 10 --prior 0]

For each day t (using only days <= t):
  * every `refit` days, fit one set of structural parameters (shared) + one state per day over the last `window` days, weighted
    toward the recent days (half-life window/3), warm-started from the previous refit; optionally pulled toward the previous
    estimate (--prior strength)
  * fit the day's state (volatility levels) given the structural parameters
  * forecast day t+1's implied volatilities for every quote of that day twice: carrying today's state ('carry') and letting each
    variance factor mean-revert toward its long-run level for one day, the model's own forecast ('revert')
Output: data/results/walk_<model>_<tag>.pkl, one dict per day, including the predicted implied vols for the next day's quotes.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, evalpairs as EP, struct as ST
from dh2.data import load_all

RESULTS = Path(__file__).resolve().parent / "data" / "results"
PROG = Path(__file__).resolve().parent / "logs" / "progress"


def worker(model, S, a, b, W, R, prior_lam, nfev_cold, nfev_warm, seed, per_expiry, burn=0, progress=None, mem=None):
    ei, si = ST.IDX[model]
    rng = np.random.default_rng(seed)
    out, store, eta_prev, eta = [], {}, None, None
    H = W / 3.0
    a0 = max(W, a - burn)       # burn-in days before the chunk: fitted (so no cold start shows) but not recorded
    total = b - a0
    for t in range(a0, b):
        if t - a0 == 0 or (t - a0) % R == 0:
            idx = list(range(t - W + 1, t + 1))
            win = [S[i] for i in idx]
            w = 0.5 ** ((t - np.array(idx)) / H)
            if eta_prev is None:
                u0 = calib.starts(model, win[-1], rng, 1)[0]
                eta0 = u0[ei]
                front0 = calib.guess(win[-1])["front"]
                st0 = np.array([[u0[j] * (calib.guess(s)["front"] / front0) ** 2 for j in si] for s in win])
                nfev = nfev_cold
            else:
                eta0 = eta_prev
                st0 = np.array([store[i] if i in store else ST.fit_state(model, S[i], eta_prev, store.get(i - 1, np.full(len(si), 0.01)),
                                                                          per_expiry=per_expiry)["state"] for i in idx])
                nfev = nfev_warm
            prior = (eta_prev, prior_lam) if (prior_lam > 0 and eta_prev is not None) else None
            if mem is not None:    # pattern memory: if today's pattern has an unusually close precedent, start from what was calibrated then
                mm, eta_hist, lam_mem = mem
                hit = mm.match(t)
                if hit is not None:
                    ends, wts, _, _ = hit
                    known = [(i, w_) for i, w_ in zip(ends, wts) if int(i) in eta_hist]
                    if known:
                        ww = np.array([w_ for _, w_ in known]); ww = ww / ww.sum()
                        mu = np.sum([w_ * eta_hist[int(i)] for (i, _), w_ in zip(known, ww)], axis=0)
                        prior = (mu, lam_mem)
            f = ST.fit_window(model, win, eta0, st0, w, prior, per_expiry=per_expiry, max_nfev=nfev)
            eta = eta_prev = f["eta"]
            for k, i in enumerate(idx):
                store[i] = f["states"][k]
        init = store.get(t - 1, store.get(t))
        fs = ST.fit_state(model, S[t], eta, init if init is not None else np.full(len(si), 0.01), per_expiry=per_expiry)
        store[t] = fs["state"]
        rec = {"t": t, "day": S[t].day, "eta": eta.copy(), "state": fs["state"].copy(), "rmse_in": fs["rmse"], "n": S[t].n}
        if t + 1 < len(S) and (S[t + 1].day - S[t].day).days <= 5:
            dt = (S[t + 1].day - S[t].day).days / 365.0
            for name, st in (("carry", fs["state"]), ("revert", ST.expected_state(model, eta, fs["state"], dt))):
                iv = EP.pred_model(model, calib.to_phys(model, ST.assemble(model, eta, st)), S[t + 1])
                rec[f"pred_{name}"] = iv.astype(np.float32)
                rec[f"score_{name}"] = EP.score(iv, S[t + 1])
        if t >= a:
            out.append(rec)
        if progress is not None:      # a tiny file per chunk, read by progress.py for the live bar
            Path(progress).write_text('{"done": %d, "total": %d}' % (t - a0 + 1, total))
        for i in [k for k in store if k < t - W - 2]:
            del store[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["heston", "dh", "dhj", "dhjs"])
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--tag", default="main")
    ap.add_argument("--window", type=int, default=40); ap.add_argument("--refit", type=int, default=10)
    ap.add_argument("--prior", type=float, default=0.0); ap.add_argument("--per-expiry", type=int, default=10)
    ap.add_argument("--workers", type=int, default=9); ap.add_argument("--chunks", type=int, default=18)
    ap.add_argument("--memory-prior", type=float, default=0.0, help="strength of the pattern-memory prior (0 = off)")
    ap.add_argument("--memory-from", default="main", help="tag of the earlier walk-forward whose calibrations are remembered")
    ap.add_argument("--mem-k", type=int, default=10); ap.add_argument("--mem-q", type=float, default=0.10); ap.add_argument("--mem-L", type=int, default=3)
    ap.add_argument("--burn", type=int, default=40); ap.add_argument("--nfev-cold", type=int, default=60); ap.add_argument("--nfev-warm", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    PROG.mkdir(parents=True, exist_ok=True)
    for f in PROG.glob(f"{a.model}_{a.tag}_*.json"):
        f.unlink()
    S = load_all(a.start, a.end)
    print(f"{len(S)} surfaces {S[0].day}..{S[-1].day}; {a.model} window {a.window} refit {a.refit} prior {a.prior}", flush=True)
    first = a.window
    mem = None
    if a.memory_prior > 0:
        from dh2 import memory as MEM
        feat = MEM.features(S)
        byday = {sf.day: i for i, sf in enumerate(S)}
        past = pickle.load(open(RESULTS / f"walk_{a.model}_{a.memory_from}.pkl", "rb"))
        eta_hist = {byday[r["day"]]: np.asarray(r["eta"]) for r in past if r["day"] in byday}
        mem = (MEM.Memory(feat, L=a.mem_L, k=a.mem_k, q=a.mem_q), eta_hist, a.memory_prior)
        print(f"pattern-memory prior on: strength {a.memory_prior}, k={a.mem_k} q={a.mem_q} L={a.mem_L}, remembering {len(eta_hist)} past calibrations", flush=True)
    chunks = [(int(c[0]), int(c[-1]) + 1) for c in np.array_split(np.arange(first, len(S)), a.chunks) if len(c)]
    res = Parallel(n_jobs=a.workers, verbose=5)(
        delayed(worker)(a.model, S, lo, hi, a.window, a.refit, a.prior, a.nfev_cold, a.nfev_warm, 7 + j, a.per_expiry, a.burn,
                        str(PROG / f"{a.model}_{a.tag}_{j:02d}.json"), mem)
        for j, (lo, hi) in enumerate(chunks))
    flat = [r for c in res for r in c]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"walk_{a.model}_{a.tag}.pkl"
    pickle.dump(flat, open(path, "wb"))
    for name in ("carry", "revert"):
        v = [r[f"score_{name}"]["rmse"] for r in flat if f"score_{name}" in r]
        print(f"{a.model} next-day IV rmse ({name}): mean {np.nanmean(v):.3f} median {np.nanmedian(v):.3f} over {len(v)} days")
    print(f"done in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
