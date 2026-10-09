"""Is the model's lead over the time-scaled smile just a wing-extrapolation effect?

    python3 exp_wing_benchmark.py        # writes results/benchmark_check/

The time-scaled smile (the best model-free benchmark so far) reads yesterday's smile by interpolation and, for a strike outside the
range yesterday quoted, repeats the edge value. The verification (results/VERIFICATION.md, L1) found that inside the quoted range the
benchmark is slightly BETTER than jump model + misfit, and the whole lead comes from the 6.5% of quotes outside it. This script adds a
third, equally model-free benchmark that extrapolates the wings linearly (the line through yesterday's 3 outermost quotes, floored at
2 vol points) and scores everything on the test period (2022-01-03 onward), split by whether the quote lies inside yesterday's range.
The benchmark's only setting (3 edge quotes) was fixed before looking at any score; nothing was tuned.
"""
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dh2 import evalpairs as EP
from dh2.data import load_all
from tournament import boot

OUT = HERE / "results" / "benchmark_check"
SPLIT = date(2022, 1, 1)
K_EDGE = 3


def smile_linear_wings(S0, S1):
    """Time-scaled smile, but beyond yesterday's quoted range continue the line through the K_EDGE outermost quotes."""
    m0, m1 = EP._log_m(S0), EP._log_m(S1)
    out = np.full(S1.n, np.nan)
    for e1 in range(len(S1.T)):
        q1 = np.flatnonzero(S1.e == e1)
        e0 = EP._match_expiry(S0, S1, e1)
        q0 = np.flatnonzero(S0.e == e0)
        if q0.size < 2:
            continue
        o = np.argsort(m0[q0])
        xs, ys = m0[q0][o], S0.iv[q0][o]
        x = m1[q1] * np.sqrt(S0.T[e0] / S1.T[e1])
        y = np.interp(x, xs, ys)
        k = min(K_EDGE, len(xs))
        sl, sr = np.polyfit(xs[:k], ys[:k], 1)[0], np.polyfit(xs[-k:], ys[-k:], 1)[0]
        lo, hi = x < xs[0], x > xs[-1]
        y[lo] = ys[0] + sl * (x[lo] - xs[0])
        y[hi] = ys[-1] + sr * (x[hi] - xs[-1])
        out[q1] = np.maximum(y, 0.02)
    return out


def inside_mask(S0, S1):
    """True for quotes of day t+1 whose time-scaled strike lies within the range the matched expiry quoted on day t."""
    m0, m1 = EP._log_m(S0), EP._log_m(S1)
    inside = np.zeros(S1.n, bool)
    for e1 in range(len(S1.T)):
        q1 = np.flatnonzero(S1.e == e1)
        e0 = EP._match_expiry(S0, S1, e1)
        q0 = np.flatnonzero(S0.e == e0)
        if q0.size < 2:
            continue
        x = m1[q1] * np.sqrt(S0.T[e0] / S1.T[e1])
        inside[q1] = (x >= m0[q0].min()) & (x <= m0[q0].max())
    return inside


def main():
    S = load_all()
    h = pickle.load(open(HERE / "data" / "results" / "hyb_dhj_main.pkl", "rb"))
    days = [t for t in sorted(h["carry_res"]) if S[t].day >= SPLIT]
    names = ("edge_value_benchmark", "linear_wing_benchmark", "jump_model_plus_misfit")
    rows = []
    for t in days:
        S0, S1 = S[t], S[t + 1]
        preds = (EP.pred_smile_scaled(S0, S1), smile_linear_wings(S0, S1), np.asarray(h["carry_res"][t], float))
        iv1, ins = S1.iv, inside_mask(S0, S1)
        ok = np.all([np.isfinite(p) for p in preds], axis=0) & np.isfinite(iv1)
        row = {"day": S1.day, "n_inside": int((ok & ins).sum()), "n_outside": int((ok & ~ins).sum())}
        for nm, p in zip(names, preds):
            for part, mk in (("all", ok), ("inside", ok & ins), ("outside", ok & ~ins)):
                row[f"{nm}__{part}"] = float(np.sqrt(np.mean((100 * (p[mk] - iv1[mk])) ** 2))) if mk.sum() else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "per_day.csv", index=False)
    res = []
    for part in ("all", "inside", "outside"):
        m = d[f"{names[0]}__{part}"].notna()
        a, b, c = (d.loc[m, f"{n}__{part}"].values for n in names)
        for lab, x, y in (("jump_plus_misfit minus edge_value_benchmark", c, a), ("jump_plus_misfit minus linear_wing_benchmark", c, b),
                          ("linear_wing_benchmark minus edge_value_benchmark", b, a)):
            diff = x - y
            mean, lo, hi = boot(list(diff))
            res.append({"quotes": part, "comparison": lab, "days": int(m.sum()), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                        "mean_difference": float(mean), "ci95_lo": float(lo), "ci95_hi": float(hi), "model_ahead_on_share_of_days": float((diff < 0).mean()),
                        "wilcoxon_p": float(wilcoxon(x, y).pvalue)})
    r = pd.DataFrame(res)
    r.to_csv(OUT / "summary.csv", index=False)
    share = d.n_inside.sum() / (d.n_inside.sum() + d.n_outside.sum())
    print(f"{len(d)} test days; quotes inside yesterday's range: {share:.1%}")
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
