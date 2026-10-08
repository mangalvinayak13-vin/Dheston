"""Better one-day-ahead forecasts from the stored 2.0 calibrations (no refitting): what to do with the model's output.

    python3 hybrids.py --model dhj --tag main

All variants use only day t's calibration (structural parameters eta_t, state v_t) plus what is known about day t+1 when pricing
(its forwards, expiries and quotes' strikes), exactly like the baselines.

  smile        yesterday's smile carried in log-moneyness (no model)
  carry        the model with today's state
  revert       ... with each variance factor mean-reverting one day (the model's own forecast)
  lev          ... plus the LEVERAGE update: the index move to day t+1 is known when pricing, and in these models volatility moves with
               the index through the price/variance correlation: E[dv_i | index return r] = rho_i * xi_i * r (so a fall raises variance
               when rho < 0). This is the model's conditional expectation, with no new parameter.
  carry_res    carry + yesterday's remaining misfit (market minus model on day t) carried in log-moneyness: the model supplies the
               structure and the smile supplies what the model cannot express
  lev_res      lev + the same misfit carry
  smile_lev    yesterday's smile + the model-implied change (lev - carry): persistence, corrected only by what the model says changes
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, evalpairs as EP, models as M, struct as ST
from dh2.data import load_all

RESULTS = Path(__file__).resolve().parent / "data" / "results"
VARIANTS = ["smile", "carry", "revert", "lev", "carry_res", "lev_res", "smile_lev"]


def one(model, S0, S1, eta, state):
    dt = (S1.day - S0.day).days / 365.0
    r = EP.fwd_return(S0, S1)
    x = lambda st: calib.to_phys(model, ST.assemble(model, eta, st))
    iv = lambda st, S: M.model_iv(M.price(model, x(st), S), S)
    carry = iv(state, S1)
    rev_state = ST.expected_state(model, eta, state, dt)
    revert = iv(rev_state, S1)
    lev_state = rev_state.copy()
    if model in ("dhj", "dh", "dhjs"):
        lev_state[0] += eta[3] * eta[2] * r          # rho1 * xi1 * r
        lev_state[1] += eta[7] * eta[6] * r          # rho2 * xi2 * r
    elif model == "heston":
        lev_state[0] += eta[3] * eta[2] * r
    lev_state = np.maximum(lev_state, 5e-4)
    lev = iv(lev_state, S1)
    resid0 = S0.iv - iv(state, S0)                   # yesterday's misfit on yesterday's quotes
    res = EP.carry_values(S0, S1, resid0)
    res = np.where(np.isfinite(res), res, 0.0)
    smile = EP.pred_last_smile(S0, S1)
    return {"smile": smile, "carry": carry, "revert": revert, "lev": lev, "carry_res": carry + res, "lev_res": lev + res,
            "smile_lev": smile + (lev - carry)}


def chunk(model, S, recs):
    out = {}
    for r in recs:
        t = r["t_global"]
        out[t] = {k: v.astype(np.float32) for k, v in one(model, S[t], S[t + 1], r["eta"], r["state"]).items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dhj"); ap.add_argument("--tag", default="main"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S = load_all()
    byday = {s.day: i for i, s in enumerate(S)}
    W = pickle.load(open(RESULTS / f"walk_{a.model}_{a.tag}.pkl", "rb"))
    recs = []
    for r in W:
        t = byday.get(r["day"])
        if t is not None and t + 1 < len(S) and (S[t + 1].day - S[t].day).days <= 5:
            recs.append({**r, "t_global": t})
    parts = np.array_split(np.arange(len(recs)), a.workers * 4)
    res = Parallel(n_jobs=a.workers, verbose=2)(delayed(chunk)(a.model, S, [recs[i] for i in p]) for p in parts if len(p))
    preds = {v: {} for v in VARIANTS}
    for d in res:
        for t, vs in d.items():
            for v in VARIANTS:
                preds[v][t] = vs[v]
    path = RESULTS / f"hyb_{a.model}_{a.tag}.pkl"
    pickle.dump(preds, open(path, "wb"))
    ts = sorted(preds["smile"])
    years = sorted({S[t].day.year for t in ts})
    print(f"\n{len(ts)} days -> {path}\nmean next-day IV rmse by year (vol points):")
    print(f"{'variant':12s} {'all':>6s} " + " ".join(f"{y:>5d}" for y in years))
    for v in VARIANTS:
        sc = {t: EP.score(preds[v][t].astype(float), S[t + 1])["rmse"] for t in ts}
        row = [np.nanmean([sc[t] for t in ts if S[t].day.year == y]) for y in years]
        print(f"{v:12s} {np.nanmean(list(sc.values())):6.3f} " + " ".join(f"{x:5.2f}" for x in row))


if __name__ == "__main__":
    main()
