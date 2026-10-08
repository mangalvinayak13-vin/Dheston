"""DH1 -> DH2.0 attribution waterfall on the sample (every 10th test-period trading day).

    python3 exp_attribution.py

Seven steps, each adds ONE fix to the one before; at every step the same quantities are measured on the same days:
  in-sample    IV RMSE (vol points) of the day-t fit over ALL of day t's clean-surface quotes
  next-day     IV RMSE of the same parameters carried to day t+1 (day t+1's forwards and expiries are given), over all of its quotes
  0  DH1                 the 1.0 research project's recipe (c2_dh1.py): its quotes (settlement prices, open-interest filter, no minimum
                         price or days), its price-space objective, its latent-coordinate five-start optimiser, pricer pricer/models.py
  1  + clean data        the same objective/optimiser on the 2.0 surface (traded OTM quotes, parity forwards, price >= Rs 1, >= 3 days)
  2  + IV objective      vega-weighted implied-vol residual with the soft-L1 loss (dh2.calib.residual and loss)
  3  + global multi-start + kappa2 > kappa1: dh2.calib.fit_day (flat-corner + random starts, box bounds with kappa2 - kappa1 >= 0.2,
                         14 quotes per expiry)
  4  + structure/state   walk-forward: slow parameters shared over a 40-day window and refit every 10 days, daily variance levels
                         (saved walk_dh_main.pkl)
  5  + Merton jumps      saved walk_dhj_main.pkl
  6  + yesterday's misfit carried   saved hyb_dhj_main.pkl 'carry_res' (a next-day device: the in-sample number is step 5's)
Context rows (not steps): yesterday's smile, time-scaled smile, flat Black-Scholes, independent daily Heston and DHJ.

Outputs: results/attribution/{waterfall.csv, per_day.csv, waterfall.png (300 dpi), replication_check.csv, data_differences.csv, README.md}
Progress: logs/progress/c2_attribution_00.json (steps 0-2 fits), c2_replication_00.json
"""
import c2_common as C
import c2_dh1 as D

import json
import pickle
from datetime import date

import numpy as np
import pandas as pd

OUT = C.RES / "attribution"
APRIL = [date(2026, 4, d) for d in (8, 9, 10, 13, 15, 16, 17, 20, 21, 22)]      # 1.0's ten NIFTY dates (nifty_selection.json)
PUB = {"bs1": 0.03039462980021644, "dh": 0.023660590935924722}                 # 1.0's published median held-out IV RMSE (decimal)

STEPS = [
    (0, "DH1 (1.0 recipe)", "1.0 data + price objective + latent 5-start optimiser"),
    (1, "+ clean data", "traded OTM quotes, parity forwards, price >= Rs 1, >= 3 days"),
    (2, "+ IV-space vega-weighted objective", "vega-weighted implied-vol residual, soft-L1"),
    (3, "+ global multi-start, kappa2 > kappa1", "dh2.calib.fit_day: flat-corner + random starts, box bounds, ordering"),
    (4, "+ structure/state split, walk-forward", "slow parameters over 40-day window, refit every 10 days; daily state"),
    (5, "+ Merton jumps", "jump intensity, mean and size in the structure"),
    (6, "+ yesterday's misfit carried", "next-day forecast = model + yesterday's remaining misfit"),
]


def build_pub_table():
    """1.0's published per-date rows (nifty_comparison_detail.csv), in the order of APRIL."""
    p = ("/Users/vinayakmangal/Everything related to my projects/double-heston-inverse-calibration/.claude/worktrees/complete-training/"
         "outputs/real_markets/nifty_comparison/nifty_comparison_detail.csv")
    df = pd.read_csv(p, usecols=["date", "quotes", "holdout", "expiries", "spot", "bs1_iv", "bs1_sigma", "dh_iv"])
    df["day"] = pd.to_datetime(df["date"]).dt.date
    return df.rename(columns={"quotes": "pub_quotes", "holdout": "pub_holdout", "expiries": "pub_expiries", "bs1_iv": "pub_bs1_iv",
                              "bs1_sigma": "pub_bs1_sigma", "dh_iv": "pub_dh_iv"}).drop(columns=["date", "spot"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, samp = C.load_sample()
    N = len(samp)
    print(f"{N} sample days {S[samp[0]].day} .. {S[samp[-1]].day}", flush=True)

    # ---------- replication check on 1.0's own ten dates
    rep_res = C.pmap(D.replicate_day, [(d, (d,)) for d in APRIL], "replication", cache_name="repl")
    rep = pd.DataFrame([{k: v for k, v in rep_res[d].items() if k != "x"} for d in APRIL])
    rep = rep.merge(build_pub_table(), on="day", how="left")
    rep.to_csv(OUT / "replication_check.csv", index=False)
    print(rep.to_string(), flush=True)

    # ---------- shared same-day independent fits (step 3 and context rows)
    fits = C.daily_fits(S, samp)

    # ---------- steps 0-2 fits
    jobs = []
    for step in (0, 1, 2):
        for i in samp:
            jobs.append(((step, i), (step, S[i], S[i].day)))
    res = C.pmap(D.fit_step, jobs, "attribution", cache_name="attr_steps")

    # ---------- per-day numbers
    walks = {}
    for name in ("dh_main", "dhj_main"):
        W = pickle.load(open(C.WALK / f"walk_{name}.pkl", "rb"))
        walks[name] = {r["day"]: r for r in W}
    H = pickle.load(open(C.WALK / "hyb_dhj_main.pkl", "rb"))
    rows = []
    for i in samp:
        s0, j = S[i], C.next_index(S, i)
        s1 = S[j] if j is not None else None
        row = {"day": s0.day, "i": i, "n_quotes": s0.n, "has_next": j is not None}
        for step in (0, 1, 2):
            r = res[(step, i)]
            if "error" in r or "x" not in r:
                row[f"in{step}"] = row[f"oos{step}"] = np.nan
                continue
            row[f"in{step}"] = C.iv_rmse("dh", r["x"], s0)
            row[f"oos{step}"] = C.iv_rmse("dh", r["x"], s1) if s1 is not None else np.nan
        f3 = fits[("dh", i)]
        row["in3"], row["oos3"] = f3["rmse_in"], f3["rmse_next"]
        for step, nm in ((4, "dh_main"), (5, "dhj_main")):
            rec = walks[nm].get(s0.day)
            row[f"in{step}"] = rec["rmse_in"] if rec else np.nan
            row[f"oos{step}"] = (C.EP.score(rec["pred_carry"].astype(float), s1)["rmse"] if (rec and s1 is not None and "pred_carry" in rec) else np.nan)
        row["in6"] = row["in5"]
        row["oos6"] = C.EP.score(H["carry_res"][i].astype(float), s1)["rmse"] if (s1 is not None and i in H["carry_res"]) else np.nan
        # context rows
        row["oos_smile"] = C.EP.score(C.EP.pred_last_smile(s0, s1), s1)["rmse"] if s1 is not None else np.nan
        row["oos_smile_scaled"] = C.EP.score(C.EP.pred_smile_scaled(s0, s1), s1)["rmse"] if s1 is not None else np.nan
        for m in ("bs", "heston", "dhj"):
            row[f"in_{m}_daily"], row[f"oos_{m}_daily"] = fits[(m, i)]["rmse_in"], fits[(m, i)]["rmse_next"]
        r0 = res[(0, i)]
        row["dh1_quotes"] = r0.get("n_quotes", np.nan)
        row["dh1_expiries"] = r0.get("n_exp", np.nan)
        row["dh1_holdout_iv_rmse_pts"] = r0.get("holdout_iv_rmse_pts", np.nan)
        row["dh1_nstarts"] = r0.get("n_starts", np.nan)
        for step in (0, 1, 2):
            row[f"sec{step}"] = res[(step, i)].get("sec", np.nan)
        rows.append(row)
    pdays = pd.DataFrame(rows)
    pdays.to_csv(OUT / "per_day.csv", index=False)

    # ---------- the waterfall table
    cols_in = [f"in{k}" for k in range(7)]
    cols_oos = [f"oos{k}" for k in range(7)]
    common_in = pdays[cols_in].notna().all(axis=1)
    common_oos = pdays[cols_oos + ["oos_smile", "oos_smile_scaled"]].notna().all(axis=1)
    print(f"common days: in-sample {int(common_in.sum())}/{N}, next-day {int(common_oos.sum())}/{N}", flush=True)
    tab = []
    for k, label, what in STEPS:
        a_in, a_oo = pdays.loc[common_in, f"in{k}"].to_numpy(), pdays.loc[common_oos, f"oos{k}"].to_numpy()
        row = {"step": k, "label": label, "what_changes": what, "n_days_in": int(common_in.sum()), "n_days_oos": int(common_oos.sum())}
        for tag, a, allv in (("in", a_in, pdays.loc[common_in]), ("oos", a_oo, pdays.loc[common_oos])):
            m, lo, hi = C.boot_ci(a)
            row.update({f"{tag}_mean": m, f"{tag}_lo": lo, f"{tag}_hi": hi, f"{tag}_median": float(np.median(a))})
            if k > 0:
                prev = allv[f"{tag}{k - 1}"].to_numpy()
                d = a - prev
                dm, dlo, dhi = C.boot_ci(d, seed=k)
                row.update({f"{tag}_delta_vs_prev": dm, f"{tag}_delta_lo": dlo, f"{tag}_delta_hi": dhi,
                            f"{tag}_wilcoxon_p": C.wilcoxon_p(a, prev), f"{tag}_days_improved": int((d < 0).sum())})
                d0 = a - allv[f"{tag}0"].to_numpy()
                row[f"{tag}_cum_vs_step0"] = float(np.mean(d0))
        tab.append(row)
    # context rows (next-day or in-sample as available), on the same common days
    ctx = []
    for key, label, kind in (("oos_smile", "yesterday's smile (model-free)", "oos"), ("oos_smile_scaled", "time-scaled smile (model-free)", "oos"),
                             ("oos_bs_daily", "flat Black-Scholes, daily refit, carried", "oos"),
                             ("oos_heston_daily", "single Heston, independent daily fits", "oos"),
                             ("oos_dhj_daily", "DH + jumps, independent daily fits", "oos")):
        a = pdays.loc[common_oos, key].to_numpy()
        m, lo, hi = C.boot_ci(a)
        r = {"step": "ref", "label": label, "what_changes": "reference row (not a step)", "n_days_oos": int(common_oos.sum()),
             "oos_mean": m, "oos_lo": lo, "oos_hi": hi, "oos_median": float(np.median(a))}
        base = key.replace("oos_", "in_")
        if base in pdays:
            ai = pdays.loc[common_in, base].to_numpy()
            r["n_days_in"] = int(common_in.sum())
            r["in_mean"], r["in_lo"], r["in_hi"] = C.boot_ci(ai)
            r["in_median"] = float(np.median(ai))
        ctx.append(r)
    wf = pd.DataFrame(tab + ctx)
    wf.to_csv(OUT / "waterfall.csv", index=False)
    print(wf[["step", "label", "in_mean", "in_median", "oos_mean", "oos_median"]].to_string(), flush=True)

    # ---------- data differences between 1.0's quote set and the 2.0 clean surface (explains step 1)
    dd = []
    for i in samp:
        r0 = res[(0, i)]
        if "sf_keys" not in r0:
            continue
        k0 = r0["sf_keys"]
        s = S[i]
        clean = {}
        for q in range(s.n):
            clean[(s.expiry[s.e[q]], float(s.K[q]), bool(s.is_call[q]))] = float(s.price[q])
        old = {}
        for xp, K, c, px in zip(k0["xpry"], k0["K"], k0["is_call"], k0["px"]):
            old[(pd.Timestamp(str(xp)).date(), float(K), bool(c))] = float(px)
        common = set(old) & set(clean)
        # forwards by expiry date
        f_old = {}
        for xp, F in zip(k0["xpry"], k0["F"]):
            f_old[pd.Timestamp(str(xp)).date()] = float(F)
        fdiff = [abs(np.log(f_old[s.expiry[e]] / s.F[e])) * 1e4 for e in range(len(s.T)) if s.expiry[e] in f_old]
        pdiff = [abs(np.log(old[k] / clean[k])) for k in common]
        dd.append({"day": s.day, "quotes_1.0": len(old), "quotes_2.0": len(clean), "in_both": len(common),
                   "share_of_1.0_in_2.0": len(common) / max(len(old), 1), "share_of_2.0_in_1.0": len(common) / max(len(clean), 1),
                   "expiries_1.0": len({k[0] for k in old}), "expiries_2.0": len(s.T),
                   "median_abs_forward_diff_bp": float(np.median(fdiff)) if fdiff else np.nan,
                   "median_abs_log_price_diff_settle_vs_close": float(np.median(pdiff)) if pdiff else np.nan,
                   "mean_abs_log_price_diff_settle_vs_close": float(np.mean(pdiff)) if pdiff else np.nan,
                   "share_price_diff_gt_1pct": float(np.mean(np.array(pdiff) > 0.01)) if pdiff else np.nan})
        sfx = D.dh1_surface(s.day)
        dd[-1].update({"share_1.0_zero_volume": float(np.mean(sfx["vol"] == 0)), "share_1.0_price_below_Rs1": float(np.mean(sfx["px"] < 1.0)),
                       "share_1.0_under_3_days": float(np.mean(sfx["dte"] < 3)), "share_1.0_zero_open_interest": float(np.mean(sfx["oi"] == 0))})
    ddf = pd.DataFrame(dd)
    ddf.to_csv(OUT / "data_differences.csv", index=False)

    make_figure(wf, pdays, common_in, common_oos)
    write_readme(wf, pdays, rep, ddf, common_in, common_oos, S, samp)
    print("done", flush=True)


def make_figure(wf, pdays, common_in, common_oos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    BLUE, ORANGE, RED, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#e34948", "#b9b8b2", "#0b0b0b", "#52514e"
    steps = wf[wf.step.apply(lambda v: v != "ref")].copy()
    short = ["DH1\n(1.0 recipe)", "clean\ndata", "IV-space\nobjective", "global multi-\nstart + order", "structure/state\nwalk-forward", "Merton\njumps", "yesterday's\nmisfit carried"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.2), sharey=False)
    for ax, tag, title, colour in ((axes[0], "in", "Same-day fit error (in-sample)", BLUE), (axes[1], "oos", "Next-day error (out-of-sample)", ORANGE)):
        mean = steps[f"{tag}_mean"].to_numpy(float)
        lo, hi = steps[f"{tag}_lo"].to_numpy(float), steps[f"{tag}_hi"].to_numpy(float)
        x = np.arange(len(steps))
        # floating bars: each step is a bar from the previous mean to its own mean; the first and last are totals
        for k in range(len(steps)):
            if k == 0:
                ax.bar(k, mean[0], width=0.62, color=colour, zorder=3)
            else:
                top, bot = mean[k - 1], mean[k]
                good = bot <= top
                ax.bar(k, abs(top - bot), bottom=min(top, bot), width=0.62, color=(colour if good else RED), alpha=0.85 if good else 0.9, zorder=3)
                ax.plot([k - 1 + 0.31, k - 0.31], [mean[k - 1], mean[k - 1]], color=MUTED, lw=1.2, zorder=2)
            ax.errorbar(k, mean[k], yerr=[[mean[k] - lo[k]], [hi[k] - mean[k]]], color=INK, capsize=5, lw=1.6, zorder=4)
            ytop = max(hi[k], mean[k - 1] if k > 0 else 0.0)
            ax.text(k, ytop + 0.025 * np.nanmax(hi), f"{mean[k]:.2f}", ha="center", va="bottom", fontsize=14, color=INK, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, zorder=6)
        if tag == "oos":
            for key, lab, ls in (("oos_smile", "yesterday's smile (no model)", "--"), ("oos_smile_scaled", "time-scaled smile (no model)", ":")):
                v = float(wf.loc[wf.label.str.startswith("yesterday's smile" if key == "oos_smile" else "time-scaled"), "oos_mean"].iloc[0])
                ax.axhline(v, color=MUTED, ls=ls, lw=1.8, zorder=1, label=f"{lab}: {v:.2f}")
            ax.legend(loc="upper right", frameon=False, fontsize=13.5)
        deltas = ["" if k == 0 else f"\n{mean[k] - mean[k - 1]:+.2f}" for k in range(len(steps))]
        ax.set_xticks(x)
        ax.set_xticklabels([sh + d for sh, d in zip(short, deltas)])
        ax.set_ylabel("implied-volatility RMSE, vol points (lower is better)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, np.nanmax(hi) * 1.22)
        ax.yaxis.grid(True, color="#e6e5e0", lw=1)
        ax.set_axisbelow(True)
    n_in, n_oos = int(common_in.sum()), int(common_oos.sum())
    fig.suptitle(f"From the 1.0 calibration to Double Heston 2.0: one fix at a time (NIFTY, {n_in} sampled test days, mean with 95% bootstrap interval)",
                 fontsize=18, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Blue/orange bar: improvement over the previous step; red bar: the step made the error larger. The signed number under each label is the change in vol points.\nStep 6 changes only the next-day forecast, so its same-day value equals step 5.",
             fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUT / "waterfall.png", dpi=300, facecolor="white")
    plt.close(fig)


def fmt(v, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_readme(wf, pdays, rep, ddf, common_in, common_oos, S, samp):
    steps = wf[wf.step.apply(lambda v: v != "ref")]
    refs = wf[wf.step.apply(lambda v: v == "ref")]
    L = []
    L.append("# DH1 -> DH2.0 attribution waterfall\n")
    L.append(f"Sample: every 10th trading day of the test period (2022-01-01 onward), {len(samp)} days ({S[samp[0]].day} to {S[samp[-1]].day}); "
             f"all numbers below use the {int(common_in.sum())} days (same-day) and {int(common_oos.sum())} days (next-day) on which every step has a value. "
             "The sample is fixed by that rule, identical for every step and model, and no design choice was tuned on it. "
             "Numbers are produced by `exp_attribution.py`; nothing here is hand-edited.\n")
    L.append("Error = implied-volatility RMSE in volatility points over all of a day's clean-surface quotes (the same scoring set for every step). "
             "Same-day: the day-t fit scored on day t. Next-day: the same parameters carried to day t+1 (its forwards and expiries given), scored on day t+1. "
             "Intervals are 95% bootstrap intervals over days; the p-values are paired two-sided Wilcoxon tests against the previous step.\n")
    L.append("## The waterfall\n")
    L.append("| step | what changes | same-day mean [95%] | same-day median | step change (paired) | next-day mean [95%] | next-day median | step change (paired) | next-day p |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in steps.iterrows():
        k = int(r.step)
        din = "" if k == 0 else f"{r.in_delta_vs_prev:+.3f} [{r.in_delta_lo:+.3f}, {r.in_delta_hi:+.3f}]" + (" (carried misfit: same fit)" if k == 6 else "")
        dout = "" if k == 0 else f"{r.oos_delta_vs_prev:+.3f} [{r.oos_delta_lo:+.3f}, {r.oos_delta_hi:+.3f}]"
        p = "" if k == 0 else (f"{r.oos_wilcoxon_p:.3g}" if np.isfinite(r.oos_wilcoxon_p) else "n/a")
        L.append(f"| {k} {r.label} | {r.what_changes} | {r.in_mean:.3f} [{r.in_lo:.3f}, {r.in_hi:.3f}] | {r.in_median:.3f} | {din} | "
                 f"{r.oos_mean:.3f} [{r.oos_lo:.3f}, {r.oos_hi:.3f}] | {r.oos_median:.3f} | {dout} | {p} |")
    L.append("\nContext rows, same days:\n")
    L.append("| reference | same-day mean | next-day mean [95%] | next-day median |")
    L.append("|---|---|---|---|")
    for _, r in refs.iterrows():
        L.append(f"| {r.label} | {fmt(r.get('in_mean'), 3)} | {r.oos_mean:.3f} [{r.oos_lo:.3f}, {r.oos_hi:.3f}] | {r.oos_median:.3f} |")
    s0 = steps.iloc[0]; s6 = steps.iloc[-1]
    L.append(f"\nOverall, from DH1 to the full DH2.0 forecast: same-day {s0.in_mean:.3f} -> {s6.in_mean:.3f} vol points, "
             f"next-day {s0.oos_mean:.3f} -> {s6.oos_mean:.3f} vol points. Figure: `waterfall.png` (300 dpi); per-day numbers: `per_day.csv`; table: `waterfall.csv`.\n")
    L.append("## How DH1 was rebuilt (and what could not be recovered)\n")
    L.append(f"DH1 is the 1.0 research project's Double Heston calibration of NIFTY. The script that produced `outputs/real_markets/nifty_comparison/nifty_comparison_summary.json` "
             "was lost with a session scratchpad (the header of `src/mentor_dh_pinn/baselines.py` in the old repository says so). The recipe was rebuilt, read-only, from the surviving modules "
             "(`src/mentor_dh_pinn/nifty_panel.py`, `scripts/mentor_dh_pinn/evaluate_real_markets.py`, `src/mentor_dh_pinn/baselines.py`, `params_v2.py`; `legacy_streamlit_site/models.py` holds only a single-Heston vega-weighted calibrator, not the Double Heston one):\n")
    L.append("- **Data**: NSE settlement prices (not last-trade closes), expiries kept when they have traded volume and at least 10,000 open interest in total, any days to expiry, no minimum price, forward and discount per expiry from a Huber-reweighted put-call-parity regression, out-of-the-money side, 0.78 to 1.22 of the forward.")
    L.append("- **Quotes**: every third strike of each expiry left out of the fit (the fit uses the other two thirds).")
    L.append("- **Objective**: price space, (model forward call - market forward call) / forward, no vega weights, plain least squares.")
    L.append("- **Optimiser**: scipy `least_squares` (trf), diff_step 1e-6, max_nfev 220, in unbounded latent coordinates that build in kappa_fast > kappa_slow, the Feller condition and the correlation disk; five fixed start vectors, keep the lowest cost. The first start violates Feller and is skipped silently by 1.0's own loop, so four starts run.")
    L.append("- **Pricer**: `pricer/models.py` (as instructed), not 1.0's own quadrature pricer.\n")
    L.append("What does not match: (1) the old module requires 10,000 open interest per contract, but only a per-expiry floor reproduces 1.0's own counts on 2026-04-08 (833 quotes, 10 expiries); that is what is used. "
             "(2) Pre-June-2024 NSE files carry no index level, so the level used only for the parity-validity gates is the strike where call = put. (3) The held-out set and the Black-Scholes fit could not be matched exactly (below).\n")
    L.append("### Replication check on 1.0's own ten dates (8-22 April 2026): how close is it?\n")
    medd, medb = float(rep.dh_iv.median()), float(rep.bs1_iv.median())
    cd = float(np.corrcoef(rep.dh_iv, rep.pub_dh_iv)[0, 1]); cb = float(np.corrcoef(rep.bs1_iv, rep.pub_bs1_iv)[0, 1])
    same_q = bool((rep.quotes == rep.pub_quotes).all()); same_e = bool((rep.expiries == rep.pub_expiries).all())
    L.append("| quantity (10 dates) | rebuilt | published by 1.0 | verdict |")
    L.append("|---|---|---|---|")
    L.append(f"| quotes per date | {int(rep.quotes.min())}-{int(rep.quotes.max())} | {int(rep.pub_quotes.min())}-{int(rep.pub_quotes.max())} | " + ("identical on every date" if same_q else "differs") + " |")
    L.append(f"| expiries per date | {int(rep.expiries.min())}-{int(rep.expiries.max())} | {int(rep.pub_expiries.min())}-{int(rep.pub_expiries.max())} | " + ("identical on every date" if same_e else "differs") + " |")
    L.append(f"| held-out quotes per date | {int(rep.holdout.min())}-{int(rep.holdout.max())} | {int(rep.pub_holdout.min())}-{int(rep.pub_holdout.max())} | NOT matched (rebuilt set is larger) |")
    L.append(f"| Black-Scholes one-sigma fit | {rep.bs1_sigma.mean():.4f} (mean) | {rep.pub_bs1_sigma.mean():.4f} (mean) | NOT matched: {100 * (rep.bs1_sigma.mean() / rep.pub_bs1_sigma.mean() - 1):.0f}% higher |")
    L.append(f"| median held-out IV RMSE, Black-Scholes (vol points) | {medb * 100:.2f} | {PUB['bs1'] * 100:.2f} | close in aggregate; per-date correlation {cb:.2f} |")
    L.append(f"| median held-out IV RMSE, cold Double Heston (vol points) | {medd * 100:.2f} | {PUB['dh'] * 100:.2f} | {100 * (medd / PUB['dh'] - 1):+.0f}% in aggregate; per-date correlation {cd:.2f} |")
    own = float(pdays.dh1_holdout_iv_rmse_pts.mean())
    L.append(f"\n**Plain reading.** The data universe of 1.0 is reproduced exactly (same quote and expiry counts on all ten dates, same index level), which fixes the quote set, but the "
             "1.0 held-out third and its Black-Scholes sigma are not: the script that produced the published table was lost, and I could not recover which quotes it fitted and held out. "
             f"The rebuilt Double Heston optimiser lands within {abs(100 * (medd / PUB['dh'] - 1)):.0f}% of 1.0's published median (the pricer and the exact held-out set differ), so it behaves like the 1.0 recipe in aggregate, "
             "but this is a close rebuild, not a verified reproduction; date by date it agrees only moderately. "
             f"On the {len(pdays)} sample days the same rebuilt recipe scores {own:.2f} vol points mean on 1.0's own metric (held-out third, 20-800 days), "
             f"in line with the 2.37 that 1.0 published for April 2026, whereas step 0 above scores {steps.iloc[0].in_mean:.2f} same-day: the difference is the scoring set, "
             "not a different fit. The waterfall scores every quote of the 2.0 clean surface, including 3-19 day expiries and the far wings (|ln K/F| up to 0.22), which a price-space "
             "fit with no jump component fits worst; 1.0's metric excluded expiries under 20 days. Table: `replication_check.csv`.\n")
    L.append("## What each data step changes\n")
    L.append(f"On the sampled days DH1's data has on average {ddf['quotes_1.0'].mean():.0f} quotes over {ddf['expiries_1.0'].mean():.1f} expiries; the 2.0 clean surface has {ddf['quotes_2.0'].mean():.0f} quotes over {ddf['expiries_2.0'].mean():.1f} expiries. "
             f"{100*ddf['share_of_1.0_in_2.0'].mean():.0f}% of DH1's quotes are also in the clean surface and {100*ddf['share_of_2.0_in_1.0'].mean():.0f}% of the clean quotes are in DH1's. "
             f"Of DH1's quotes, {100*ddf['share_1.0_zero_volume'].mean():.0f}% did not trade that day (zero volume), {100*ddf['share_1.0_zero_open_interest'].mean():.0f}% have zero open interest, "
             f"{100*ddf['share_1.0_price_below_Rs1'].mean():.1f}% are priced below Rs 1 and {100*ddf['share_1.0_under_3_days'].mean():.1f}% expire in under 3 days: these are what the clean data removes. "
             f"For the same expiry the two forwards differ by a median of {ddf['median_abs_forward_diff_bp'].median():.1f} basis points. "
             f"For quotes in both sets the settlement price and the close are identical in the median ({100*ddf['median_abs_log_price_diff_settle_vs_close'].median():.2f}% log difference) "
             f"but not always: {100*ddf['share_price_diff_gt_1pct'].mean():.1f}% of the common quotes differ by more than 1% (mean absolute log difference {100*ddf['mean_abs_log_price_diff_settle_vs_close'].mean():.2f}%). "
             "The change at step 1 is therefore mainly which quotes and expiries enter (untraded, near-zero-priced and very-short-dated ones leave), plus a minority of different prices. (`data_differences.csv`)\n")
    L.append("### By year (mean IV RMSE, vol points)\n")
    pdays2 = pdays.copy()
    pdays2["year"] = pd.to_datetime(pdays2.day).dt.year
    yr = pdays2.groupby("year").agg(n=("day", "size"), s0=("in0", "mean"), s3=("in3", "mean"), s5=("in5", "mean"), o0=("oos0", "mean"), o3=("oos3", "mean"), o4=("oos4", "mean"), o5=("oos5", "mean"), o6=("oos6", "mean"), sm=("oos_smile_scaled", "mean"))
    L.append("| year | days | same-day: DH1 | step 3 | step 5 | next-day: DH1 | step 3 | step 4 | step 5 | step 6 | time-scaled smile |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for y, r in yr.iterrows():
        L.append(f"| {y} | {int(r.n)} | {r.s0:.2f} | {r.s3:.2f} | {r.s5:.2f} | {r.o0:.2f} | {r.o3:.2f} | {r.o4:.2f} | {r.o5:.2f} | {r.o6:.2f} | {r.sm:.2f} |")
    L.append("")
    L.append("## Caveats\n")
    L.append("- Step 4 changes two things at once that cannot be separated with the saved results: the structural/state split and the walk-forward refits (shared parameters over a window). Same-day fit gets worse there by construction (a day's fit has two free numbers instead of ten); the gain is in the next-day column.")
    L.append("- Steps 0-2 fit a two-thirds subset of strikes (1.0's protocol); step 3 uses `fit_day`'s own subset (14 quotes per expiry). Scoring always uses every quote of the clean surface, including the ones a step did not fit.")
    L.append("- Step 2 changes the loss space, the weights and the robust loss together (they are one objective in `dh2/calib.py`).")
    L.append("- The sample is 118 days spaced 10 trading days apart, so a per-step difference can be small compared with day-to-day noise; use the paired intervals, not the means alone.")
    L.append("- Next-day for steps 0-3 carries all ten parameters; for steps 4-6 the structure is held and only the state is carried (that is the DH2.0 forecast).")
    L.append("\n## What did not work\n")
    L.append("- The exact 1.0 script behind `nifty_comparison_summary.json` could not be recovered (lost with a scratchpad); DH1 is a documented rebuild. The held-out set and the Black-Scholes sigma of 1.0 were not matched (table above); data universe and quote counts were.")
    L.append("- The per-contract open-interest floor in 1.0's surviving code (10,000) does not reproduce 1.0's own quote counts; zero is used for contracts and 10,000 for the expiry total.")
    L.append("- A first run of this script crashed at its last step (a numpy string passed to pandas Timestamp, in the data-differences table, after all fits were done); fixed, and the outputs here were regenerated from the cached fits without refitting.")
    L.append("- The waterfall cannot separate the structure/state split from the walk-forward refits (step 4), nor the loss space from the robust loss (step 2); no saved result isolates them.")
    L.append("\n## Files\n")
    L.append("- `results/attribution/waterfall.csv` the table above (means, medians, bootstrap intervals, paired step changes, Wilcoxon p, days improved); `results/attribution/per_day.csv` every day, every step; `results/attribution/waterfall.png` (300 dpi); "
             "`results/attribution/replication_check.csv`; `results/attribution/data_differences.csv`.")
    L.append("- Code: `exp_attribution.py` (this table), `c2_dh1.py` (the DH1 rebuild), `c2_common.py` (sample, fits, statistics). Cached fits: `results/_cache/attr_steps_v1.pkl`, `daily_fits_v1.pkl`, `repl_v1.pkl`.")
    (OUT / "README.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
