"""Paired significance tests for forecast comparisons, and their application to the test period (2022-01-01 onward).

    python3 -m dh2.stats [--out results/stats]

Wilcoxon signed-rank (scipy) and the Diebold-Mariano test on a loss differential, here the DAILY next-day implied-vol RMSE (vol
points) of two methods scored on the same days:

  d_t = L_A,t - L_B,t           negative = A better
  DM  = mean(d) / sqrt(LRV / T)  LRV = long-run variance of d_t, Newey-West (Bartlett kernel) with `lags` lags
  HLN: DM* = DM * sqrt((T + 1 - 2h + h(h-1)/T) / T), compared with a t distribution on T-1 degrees of freedom
       (Harvey, Leybourne and Newbold 1997). With the horizon h = 1 and lags = 0 this is exactly the one-sample t-test
       on d_t (a unit test checks that); more lags widen the variance estimate when the daily losses are autocorrelated,
       which they are (volatility regimes persist), so the default uses the Newey-West (1994) rule of thumb
       lags = floor(4 (T/100)^(2/9)) (never less than h-1). The textbook h=1 version (lags=0) is reported next to it.

The tests say whether the MEAN daily error differs; they do not make the days independent, so the HAC variance is the honest one.
Wilcoxon looks at the median of the paired differences and is robust to a few huge days (March 2020-type outliers).
"""
from __future__ import annotations

import argparse
import math
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[1]
SPLIT = date(2022, 1, 1)


# ------------------------------------------------------------------ the tests
def _paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def wilcoxon_signed_rank(a, b, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test on the paired differences a - b (scipy, zero differences dropped, normal approximation for n > 50).
    alternative 'less': a tends to be smaller than b (a better when the series are errors)."""
    a, b = _paired(a, b)
    d = a - b
    n_eff = int((d != 0).sum())
    out = {"n": int(d.size), "n_nonzero": n_eff, "median_diff": float(np.median(d)) if d.size else float("nan"),
           "share_a_lower": float(np.mean(d < 0)) if d.size else float("nan")}
    if n_eff == 0:
        return {**out, "statistic": 0.0, "pvalue": 1.0}
    r = sst.wilcoxon(a, b, zero_method="wilcox", alternative=alternative)
    return {**out, "statistic": float(r.statistic), "pvalue": float(r.pvalue)}


def newey_west_lrv(d, lags: int) -> float:
    """Long-run variance of a series by Newey-West: gamma_0 + 2 sum_{j=1..lags} (1 - j/(lags+1)) gamma_j (divisor T). Never negative."""
    d = np.asarray(d, float)
    T = d.size
    x = d - d.mean()
    lrv = float(x @ x) / T
    for j in range(1, min(lags, T - 1) + 1):
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * float(x[j:] @ x[:-j]) / T
    return lrv


def nw_auto_lags(T: int) -> int:
    """Newey-West (1994) rule of thumb for the Bartlett kernel."""
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(loss_a, loss_b, h: int = 1, lags: int | None = None, hln: bool = True, alternative: str = "two-sided") -> dict:
    """Diebold-Mariano test of equal expected loss. d_t = loss_a - loss_b. lags=None -> Newey-West rule (not below h-1).
    Returns the statistic (HLN-corrected when hln), the p-value (t with T-1 df if hln, else standard normal) and the mean difference.
    alternative 'less': A has lower expected loss than B."""
    la, lb = _paired(loss_a, loss_b)
    d = la - lb
    T = d.size
    if T < 3:
        return {"n": T, "mean_diff": float("nan"), "dm_stat": float("nan"), "pvalue": float("nan"), "lags": 0, "lrv": float("nan")}
    L = max(h - 1, nw_auto_lags(T)) if lags is None else int(lags)
    lrv = newey_west_lrv(d, L)
    dbar = float(d.mean())
    if lrv <= 0:
        return {"n": T, "mean_diff": dbar, "dm_stat": float("nan"), "pvalue": float("nan"), "lags": L, "lrv": lrv}
    dm = dbar / math.sqrt(lrv / T)
    if hln:
        dm *= math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dist = sst.t(df=T - 1)
    else:
        dist = sst.norm()
    p = {"two-sided": 2.0 * dist.sf(abs(dm)), "less": dist.cdf(dm), "greater": dist.sf(dm)}[alternative]
    return {"n": T, "mean_diff": dbar, "dm_stat": float(dm), "pvalue": float(p), "lags": L, "lrv": lrv}


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control over a set of comparisons)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ the test-period comparison
def daily_losses(res_dir: Path | None = None) -> tuple[dict, dict, list]:
    """Per-method daily next-day IV RMSE and near-the-money RMSE on the common test days. Returns (rmse, core, days)."""
    from . import evalpairs as EP
    from .data import load_all
    res_dir = res_dir or ROOT / "data" / "results"
    S = load_all()
    H = {n: pickle.load(open(res_dir / f"hyb_{n}.pkl", "rb")) for n in ("dhj_main", "dh_main", "heston_main")}
    common = sorted(set.intersection(*[set(h["carry_res"]) for h in H.values()]))
    test = [t for t in common if S[t].day >= SPLIT]
    preds = {
        "jump model + misfit": lambda t: H["dhj_main"]["carry_res"][t],
        "diffusion DH + misfit": lambda t: H["dh_main"]["carry_res"][t],
        "Heston + misfit": lambda t: H["heston_main"]["carry_res"][t],
        "time-scaled smile": lambda t: EP.pred_smile_scaled(S[t], S[t + 1]),
        "yesterday's smile": lambda t: H["dhj_main"]["smile"][t],
    }
    rm, co = {}, {}
    for name, f in preds.items():
        sc = [EP.score(np.asarray(f(t), float), S[t + 1]) for t in test]
        rm[name] = np.array([s["rmse"] for s in sc])
        co[name] = np.array([s["core"] for s in sc])
    return rm, co, [S[t + 1].day for t in test]


def compare(losses: dict[str, np.ndarray], order: list[str], loss_name: str) -> pd.DataFrame:
    rows = []
    for i, A in enumerate(order):
        for B in order[i + 1:]:
            a, b = _paired(losses[A], losses[B])
            w = wilcoxon_signed_rank(a, b)
            dm = diebold_mariano(a, b)
            dm0 = diebold_mariano(a, b, lags=0)
            rows.append({"loss": loss_name, "A": A, "B": B, "n_days": dm["n"], "mean_A": float(a.mean()), "mean_B": float(b.mean()),
                         "mean_diff_A_minus_B": dm["mean_diff"], "share_days_A_better": w["share_a_lower"],
                         "median_diff_A_minus_B": w["median_diff"], "wilcoxon_stat": w["statistic"], "wilcoxon_p": w["pvalue"],
                         "dm_stat_hln": dm["dm_stat"], "dm_p": dm["pvalue"], "dm_lags": dm["lags"],
                         "dm_stat_hln_lags0": dm0["dm_stat"], "dm_p_lags0": dm0["pvalue"]})
    df = pd.DataFrame(rows)
    df["wilcoxon_p_holm"] = holm(df.wilcoxon_p)
    df["dm_p_holm"] = holm(df.dm_p)
    df["verdict"] = [("A better" if r.mean_diff_A_minus_B < 0 else "B better") + (" (significant, Holm 5%)" if max(r.wilcoxon_p_holm, r.dm_p_holm) < 0.05
                     else " (not significant on both tests, Holm 5%)") for r in df.itertuples()]
    return df


def _fmt_p(p: float) -> str:
    return "<1e-10" if p < 1e-10 else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "stats"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rm, co, days = daily_losses()
    order = list(rm)
    # keep only the days every method has a score on (identical days for every comparison)
    for dct in (rm, co):
        ok = np.all([np.isfinite(v) for v in dct.values()], axis=0)
        for k in dct:
            dct[k] = dct[k][ok]
    n = int(len(next(iter(rm.values()))))
    df = pd.concat([compare(rm, order, "daily IV RMSE, all quotes"), compare(co, order, "daily IV RMSE, near the money |ln K/F|<=0.04")], ignore_index=True)
    df.to_csv(out / "tests_test_period.csv", index=False)

    lines = [f"""# Paired tests, test period 2022-01-01 onward

Generated by `python3 -m dh2.stats`. {n} test days, the same days for every comparison. The loss is the day's next-day implied-vol
RMSE in vol points (mean over the day's quotes, as everywhere in this project); "A - B" below zero means A has the lower error.
Wilcoxon: scipy signed-rank on the paired daily differences. Diebold-Mariano: h = 1, Newey-West HAC variance (Bartlett kernel,
{int(df.dm_lags.iloc[0])} lags by the Newey-West 1994 rule), Harvey-Leybourne-Newbold small-sample correction, t distribution with
T-1 df; the `lags 0` column is the textbook h = 1 version without autocorrelation correction. Holm: adjusted over the 10 pairs of
each loss. A significant test says the mean (DM) or median (Wilcoxon) daily error differs; it does not say by how much it matters.
"""]
    for loss in df["loss"].unique():
        g = df[df["loss"] == loss]
        lines.append(f"\n## {loss}\n")
        lines.append("| A | B | mean A | mean B | A - B | days A better | Wilcoxon p | DM stat | DM p (HAC) | DM p (lags 0) | Holm p (W / DM) | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in g.itertuples():
            lines.append(f"| {r.A} | {r.B} | {r.mean_A:.3f} | {r.mean_B:.3f} | {r.mean_diff_A_minus_B:+.3f} | {100 * r.share_days_A_better:.0f}% | "
                         f"{_fmt_p(r.wilcoxon_p)} | {r.dm_stat_hln:+.2f} | {_fmt_p(r.dm_p)} | {_fmt_p(r.dm_p_lags0)} | {_fmt_p(r.wilcoxon_p_holm)} / {_fmt_p(r.dm_p_holm)} | {r.verdict} |")
    (out / "tests_test_period.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
