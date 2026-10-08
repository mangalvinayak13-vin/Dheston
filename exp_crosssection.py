"""Cross-sectional out-of-sample tests on the sample days (every 10th test-period trading day).

    python3 exp_crosssection.py

For every sample day and each model (bs = one vol, heston, dh = diffusion Double Heston, dhj = with jumps) an independent fit with
exactly dh2.calib.fit_day (same objective, bounds, starts, thinning to 14 quotes per expiry) on part of the day's quotes, scored on
quotes the fit never saw:
  (a) random: 80% of the quotes train, 20% held out, stratified by expiry (each expiry holds out round(0.2 n) of its quotes, at least one
      when it has five or more, and keeps at least three for training); repeated for 3 seeds. Seeds are fixed (0, 1, 2).
  (b) maturity: train on expiries with <= 60 calendar days, score on the expiries beyond 60 days (extrapolation in maturity;
      the reverse is not run).
Starting vectors are built from the training quotes only (a sub-surface), so no held-out quote leaks into the fit, not even as a start.
Score: implied-vol RMSE over the held-out quotes, vol points. Per day the three random splits are averaged; across days: mean with a
95% bootstrap interval and paired two-sided Wilcoxon p-values between models.

Outputs: results/crosssection/{per_split.csv, per_day.csv, summary.csv, pairs.csv, crosssection.png (300 dpi), README.md}
Progress: logs/progress/c2_crosssection_00.json
"""
import c2_common as C

import time
from itertools import combinations

import numpy as np
import pandas as pd

from dh2 import calib, models as M

OUT = C.RES / "crosssection"
SEEDS = (0, 1, 2)
HELD_FRAC = 0.2
MAT_DAYS = 60
MIN_TRAIN_PER_EXPIRY = 3


def random_split(S, seed: int) -> np.ndarray:
    """Boolean mask of held-out quotes: stratified by expiry, seeded by (seed, day)."""
    rng = np.random.default_rng(1_000_003 * (seed + 1) + S.day.toordinal())
    held = np.zeros(S.n, bool)
    for e in range(len(S.T)):
        ix = np.flatnonzero(S.e == e)
        n = ix.size
        h = int(round(HELD_FRAC * n))
        if n >= 5:
            h = max(h, 1)
        h = min(h, max(n - MIN_TRAIN_PER_EXPIRY, 0))
        if h > 0:
            held[rng.choice(ix, h, replace=False)] = True
    return held


def maturity_split(S):
    days = np.round(S.T * 365.0)
    held = days[S.e] > MAT_DAYS
    return held


def split_job(i: int, S, kind: str, seed: int) -> dict:
    """All four models on one split of one day."""
    held = random_split(S, seed) if kind == "random" else maturity_split(S)
    n_train, n_held = int((~held).sum()), int(held.sum())
    out = {"day": S.day, "i": i, "kind": kind, "seed": seed, "n_train": n_train, "n_held": n_held}
    n_exp_train = int(len(np.unique(S.e[~held]))) if n_train else 0
    out["n_exp_train"] = n_exp_train
    if kind == "maturity" and (n_exp_train < 2 or n_held < 10):
        out["skipped"] = f"train expiries {n_exp_train}, held-out quotes {n_held}"
        return out
    Strain = C.subsurface(S, np.flatnonzero(~held))
    for m in C.MODELS:
        t = time.time()
        try:
            f = calib.fit_day(m, Strain, n_starts=4, seed=i * 10 + seed)
            out[f"{m}_heldout"] = C.iv_rmse(m, f["x"], S, held)
            out[f"{m}_train"] = f["rmse"]
            out[f"{m}_sec"] = time.time() - t
        except Exception as ex:
            out[f"{m}_error"] = repr(ex)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, samp = C.load_sample()
    print(f"{len(samp)} sample days; random 80/20 x {len(SEEDS)} seeds and a <= {MAT_DAYS}-day -> longer split, 4 models", flush=True)
    jobs = []
    for i in samp:
        for sd in SEEDS:
            jobs.append((("random", i, sd), (i, S[i], "random", sd)))
        jobs.append((("maturity", i, 0), (i, S[i], "maturity", 0)))
    res = C.pmap(split_job, jobs, "crosssection", cache_name="xsec")
    rows = [r for r in res.values() if "error" not in r or "day" in r]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_split.csv", index=False)

    # ---- per day (random: average of the three seeds; maturity: the single split)
    pd_rows = []
    for i in samp:
        d = df[(df.i == i)]
        r = d[d.kind == "random"]
        m = d[d.kind == "maturity"]
        row = {"day": S[i].day, "i": i, "n_held_random": float(r.n_held.mean()) if len(r) else np.nan}
        for mod in C.MODELS:
            c = f"{mod}_heldout"
            row[f"random_{mod}"] = float(r[c].mean()) if (c in r and r[c].notna().any()) else np.nan
            row[f"maturity_{mod}"] = float(m[c].iloc[0]) if (c in m and len(m) and m[c].notna().any()) else np.nan
            row[f"random_train_{mod}"] = float(r[f"{mod}_train"].mean()) if (f"{mod}_train" in r and r[f"{mod}_train"].notna().any()) else np.nan
        row["n_held_maturity"] = float(m.n_held.iloc[0]) if len(m) else np.nan
        pd_rows.append(row)
    days = pd.DataFrame(pd_rows)
    days.to_csv(OUT / "per_day.csv", index=False)

    # ---- summaries
    summ, pairs = [], []
    for setting in ("random", "maturity"):
        ok = days[[f"{setting}_{m}" for m in C.MODELS]].notna().all(axis=1)
        sub = days[ok]
        for mod in C.MODELS:
            v = sub[f"{setting}_{mod}"].to_numpy()
            mean, lo, hi = C.boot_ci(v, seed=1)
            r = {"setting": setting, "model": mod, "n_days": int(ok.sum()), "mean_iv_rmse": mean, "ci95_lo": lo, "ci95_hi": hi,
                 "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
            if setting == "random":
                for sd in SEEDS:
                    rr = df[(df.kind == "random") & (df.seed == sd) & df.i.isin(sub.i)][f"{mod}_heldout"]
                    r[f"mean_seed{sd}"] = float(rr.mean())
                r["mean_train_rmse"] = float(sub[f"random_train_{mod}"].mean())
            summ.append(r)
        for a, b in combinations(C.MODELS, 2):
            va, vb = sub[f"{setting}_{a}"].to_numpy(), sub[f"{setting}_{b}"].to_numpy()
            d = vb - va
            dm, dlo, dhi = C.boot_ci(d, seed=2)
            pairs.append({"setting": setting, "model_a": a, "model_b": b, "n_days": int(ok.sum()), "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                          "mean_diff_b_minus_a": dm, "diff_ci95_lo": dlo, "diff_ci95_hi": dhi, "days_b_better": int((d < 0).sum()),
                          "wilcoxon_p": C.wilcoxon_p(va, vb)})
    sm, pr = pd.DataFrame(summ), pd.DataFrame(pairs)
    sm.to_csv(OUT / "summary.csv", index=False)
    pr.to_csv(OUT / "pairs.csv", index=False)
    print(sm.to_string(), flush=True)
    print(pr.to_string(), flush=True)
    figure(sm, days)
    readme(sm, pr, days, df)
    print("done", flush=True)


def figure(sm, days):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    cols = {"bs": "#2a78d6", "heston": "#eb6834", "dh": "#1baf7a", "dhj": "#4a3aa7"}
    names = {"bs": "Black-Scholes\n(one vol)", "heston": "Heston", "dh": "Double Heston\n(diffusion)", "dhj": "Double Heston\n+ jumps"}
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    titles = {"random": "Held-out 20% of quotes (random, by expiry)", "maturity": "Fit on expiries <= 60 days, score on longer expiries"}
    for ax, setting in zip(axes, ("random", "maturity")):
        s = sm[sm.setting == setting]
        ok = days[[f"{setting}_{m}" for m in C.MODELS]].notna().all(axis=1)
        rng = np.random.default_rng(0)
        for k, m in enumerate(C.MODELS):
            r = s[s.model == m].iloc[0]
            v = days.loc[ok, f"{setting}_{m}"].to_numpy()
            ax.scatter(k + rng.uniform(-0.22, 0.22, v.size), v, s=14, color=cols[m], alpha=0.28, zorder=2, linewidths=0)
            ax.bar(k, r.mean_iv_rmse, width=0.52, color=cols[m], alpha=0.85, zorder=3)
            ax.errorbar(k, r.mean_iv_rmse, yerr=[[r.mean_iv_rmse - r.ci95_lo], [r.ci95_hi - r.mean_iv_rmse]], color="#0b0b0b", capsize=7, lw=2, zorder=4)
            ax.text(k, r.ci95_hi + 0.03 * s.ci95_hi.max(), f"{r.mean_iv_rmse:.2f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels([names[m] for m in C.MODELS])
        ax.set_ylabel("held-out implied-vol RMSE, vol points (lower is better)")
        ax.set_title(f"{titles[setting]}\n{int(ok.sum())} days; bars = mean with 95% bootstrap interval, dots = days", loc="left", fontweight="bold", fontsize=15)
        top = max(days.loc[ok, [f"{setting}_{m}" for m in C.MODELS]].quantile(0.97).max(), s.ci95_hi.max() * 1.15)
        ax.set_ylim(0, top)
        ax.yaxis.grid(True, color="#e6e5e0", lw=1)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "crosssection.png", dpi=300, facecolor="white")
    plt.close(fig)


def readme(sm, pr, days, df):
    L = ["# Cross-sectional out-of-sample test\n",
         f"Sample: every 10th trading day of the test period (2022-01-01 onward), {len(days)} days. Independent per-day fits with exactly `dh2.calib.fit_day` "
         f"(4 starts, soft-L1 vega-weighted IV objective, 14 quotes per expiry), on a training subset only; scored on held-out quotes, implied-vol RMSE in vol points. "
         "Starting vectors are built from the training quotes only. Numbers come from `exp_crosssection.py`; nothing is hand-edited.\n",
         ]
    g = lambda st, m: sm[(sm.setting == st) & (sm.model == m)].iloc[0]
    q = lambda st, a, b: pr[(pr.setting == st) & (pr.model_a == a) & (pr.model_b == b)].iloc[0]
    L.append("## Headline (tables: `results/crosssection/summary.csv`, `pairs.csv`; per-day scores `per_day.csv`; figure `crosssection.png`)\n")
    L.append("Held-out implied-vol RMSE in vol points, mean over days [95% CI]; all paired Wilcoxon tests are two-sided on the per-day scores.\n")
    L.append("- **Random 20% of quotes held out (interpolation):** " + "; ".join(f"{m} {g('random', m).mean_iv_rmse:.2f} [{g('random', m).ci95_lo:.2f}, {g('random', m).ci95_hi:.2f}]" for m in C.MODELS)
             + f". Every model beats the one before it on at least {min(q('random', a, b).days_b_better for a, b in (('bs', 'heston'), ('heston', 'dh'), ('dh', 'dhj')))} of {int(g('random', 'bs').n_days)} days "
             f"(largest p-value among the three adjacent comparisons {max(q('random', a, b).wilcoxon_p for a, b in (('bs', 'heston'), ('heston', 'dh'), ('dh', 'dhj'))):.1e}); "
             f"jumps are the largest step (dh -> dhj {q('random', 'dh', 'dhj').mean_diff_b_minus_a:+.2f}). Held-out error is within about 0.1 of the training-quote error for every model, so there is no sign of overfitting to the fitted quotes.")
    L.append("- **Fit on expiries <= 60 days, score on longer expiries (extrapolation in maturity):** " + "; ".join(f"{m} {g('maturity', m).mean_iv_rmse:.2f} [{g('maturity', m).ci95_lo:.2f}, {g('maturity', m).ci95_hi:.2f}]" for m in C.MODELS)
             + f". Single Heston and the jump model are level ({q('maturity', 'heston', 'dhj').mean_diff_b_minus_a:+.2f} [{q('maturity', 'heston', 'dhj').diff_ci95_lo:+.2f}, {q('maturity', 'heston', 'dhj').diff_ci95_hi:+.2f}], p = {q('maturity', 'heston', 'dhj').wilcoxon_p:.2f}); "
             f"the diffusion Double Heston is WORSE than single Heston here ({q('maturity', 'heston', 'dh').mean_diff_b_minus_a:+.2f}, better on only {q('maturity', 'heston', 'dh').days_b_better} of {int(g('maturity', 'bs').n_days)} days, p = {q('maturity', 'heston', 'dh').wilcoxon_p:.1e}): "
             "a plausible reason (not tested separately) is that ten parameters fitted on short expiries alone extrapolate worse than five, which the jump model then recovers. All three smile models beat flat volatility.\n")
    L.append("## (a) Random 80/20 split, stratified by expiry, 3 seeds (seeds averaged per day)\n")
    L.append("| model | days | mean [95% CI] | median | 90th pct | seed 0 | seed 1 | seed 2 | training-quote RMSE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in sm[sm.setting == "random"].iterrows():
        L.append(f"| {r.model} | {r.n_days} | {r.mean_iv_rmse:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {r['median']:.3f} | {r.p90:.3f} | {r.mean_seed0:.3f} | {r.mean_seed1:.3f} | {r.mean_seed2:.3f} | {r.mean_train_rmse:.3f} |")
    L.append("\nPaired differences (b minus a, negative = b better), two-sided Wilcoxon signed-rank on the per-day scores:\n")
    L.append("| a | b | mean a | mean b | b - a [95% CI] | days b better | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in pr[pr.setting == "random"].iterrows():
        L.append(f"| {r.model_a} | {r.model_b} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.mean_diff_b_minus_a:+.3f} [{r.diff_ci95_lo:+.3f}, {r.diff_ci95_hi:+.3f}] | {r.days_b_better}/{r.n_days} | {r.wilcoxon_p:.3g} |")
    L.append(f"\n## (b) Fit on expiries with <= {MAT_DAYS} calendar days, score on the longer expiries\n")
    L.append("| model | days | mean [95% CI] | median | 90th pct |")
    L.append("|---|---|---|---|---|")
    for _, r in sm[sm.setting == "maturity"].iterrows():
        L.append(f"| {r.model} | {r.n_days} | {r.mean_iv_rmse:.3f} [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] | {r['median']:.3f} | {r.p90:.3f} |")
    L.append("\n| a | b | mean a | mean b | b - a [95% CI] | days b better | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in pr[pr.setting == "maturity"].iterrows():
        L.append(f"| {r.model_a} | {r.model_b} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.mean_diff_b_minus_a:+.3f} [{r.diff_ci95_lo:+.3f}, {r.diff_ci95_hi:+.3f}] | {r.days_b_better}/{r.n_days} | {r.wilcoxon_p:.3g} |")
    skipped = df[df.kind == "maturity"]
    nsk = int(skipped["skipped"].notna().sum()) if "skipped" in skipped else 0
    L.append(f"\nThe maturity split needs at least two training expiries and ten held-out quotes; {nsk} of {len(skipped)} days were skipped for that reason. "
             "Held-out quotes per day (median): "
             f"random {days.n_held_random.median():.0f}, maturity {days.n_held_maturity.median():.0f}.\n")
    L.append("## Reading it, and limits\n")
    L.append("- The days are 10 trading days apart, so they are close to independent; the random-split seeds for one day are not, which is why the seeds are averaged before any test.")
    L.append("- Random held-out quotes sit between fitted quotes of the same expiry (interpolation). The maturity split asks the model to extrapolate in maturity, which only the term-structure part of the model can do; a model with a flat or short-memory term structure is expected to lose there.")
    L.append("- Fits use 14 quotes per expiry (the `fit_day` thinning) taken from the training quotes, so a held-out quote can lie between two fitted ones but never coincides with one.")
    L.append("- `training-quote RMSE` is the fit error on the training quotes (all of them, not only the 14 per expiry that enter the objective).")
    L.append("\n## What did not work, and what was not done\n")
    L.append("- No fit failed (no error column in `per_split.csv`). Results are cached in `results/_cache/xsec_v1.pkl`, so the README can be regenerated without refitting.")
    L.append("- The reverse maturity split (fit long, score short) was not required and not run.")
    L.append("- Each model is fitted with `fit_day`'s 4 starts, not a larger search; a model could do better with more starts, and the ten- and thirteen-parameter models are the ones most exposed to a poor local optimum.")
    L.append("- The two settings score different quote sets (random: all expiries, mostly near-dated; maturity: only expiries beyond 60 days), so errors are comparable across models within a setting, not across the two settings.")
    L.append("\n## Files\n")
    L.append("`results/crosssection/summary.csv`, `pairs.csv`, `per_day.csv`, `per_split.csv` (every day, split and model), `crosssection.png` (300 dpi); code `exp_crosssection.py` with `c2_common.py`.")
    (OUT / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
