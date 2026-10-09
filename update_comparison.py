"""Regenerate COMPARISON_1.0_vs_2.0.md: the running note of how version 1.0 (the original research and website) and version 2.0 (this
folder) differ, on every old test and every new one. Every 2.0 number is computed here from the saved results; every 1.0 number is
read from the research outputs (paths shown). Re-run any time:   python3 update_comparison.py

Comparability is stated for every row. Version 1.0 measured price errors on 210 stocks with at most 20 quotes per surface; 2.0 measures
implied-volatility errors on NIFTY 50 with the whole option chain, over ten years. Where the two questions are the same the rows are put
side by side; where they are not, the note says so rather than pretend.
"""
import glob
import json
import math
import pickle
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, evalpairs as EP, memory as MEM, models as M, struct as ST
from dh2.data import black_price, load_all
from tournament import boot

HERE = Path(__file__).resolve().parent
V1 = HERE / "v1_results"        # the few version 1.0 result files the comparison reads, copied from the research project
RES = HERE / "data" / "results"
SPLIT = date(2022, 1, 1)


def j(path):
    return json.loads((V1 / path).read_text())


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def rel_price_err(iv_pred, S1):
    """Version 1.0's metric: root-mean-square price error over the day's quotes divided by their mean price."""
    p = black_price(S1.F[S1.e], S1.K, S1.T[S1.e], S1.r, np.where(np.isfinite(iv_pred), iv_pred, np.nan), S1.is_call)
    ok = np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - S1.price[ok]) ** 2)) / np.mean(S1.price[ok])) if ok.sum() > 5 else float("nan")


def main():
    S = load_all()
    byday = {s.day: i for i, s in enumerate(S)}
    out, md = {}, []
    w = md.append

    # ---------------------------------------------------------------------------------------------- version 1.0, from the research outputs
    amb, cons = j("outputs/ambiguity/ambiguity_summary.json"), j("outputs/consolidated_results.json")
    nc = j("outputs/real_markets/nifty_comparison/nifty_comparison_summary.json")
    ft = j("outputs/unified_v6/finetune_real_summary.json")["NIFTY"]
    sel = j("outputs/real_markets/nifty_selection.json")
    bt = pd.read_csv(V1 / "outputs/option_backtest/option_backtest_pairs.csv")
    av = pd.read_csv(V1 / "outputs/ambiguity/ambiguity_vs_flat.csv")
    v1 = {
        "stale_dh_beats_stale_bs": float((bt.err_stale_dh < bt.err_stale_bs).mean()), "fresh_dh_beats_same_day_bs": float((bt.err_fresh_dh < bt.err_bs_same_day).mean()),
        "rel_err_stale_dh": float(bt.err_stale_dh.median()), "rel_err_stale_bs": float(bt.err_stale_bs.median()), "rel_err_bs_same_day": float(bt.err_bs_same_day.median()),
        "dh_over_flat_median": float(av.dh_over_flat.median()), "dh_beats_flat_share": float((av.dh_over_flat < 1).mean()),
        "multi_equiv": amb["share_with_multiple_equivalents"], "any_near_bound": amb["share_with_any_param_near_bound"],
        "g8_net": cons["g8"]["median_network_relative"], "g8_fit": cons["g8"]["median_best_fit_relative"],
        "synthetic_skill_opt": cons["classical_baseline"]["mean_median_skill"] if "mean_median_skill" in cons["classical_baseline"] else None,
        "nifty": {k: v["median_holdout_iv_rmse"] * 100 for k, v in nc["arms"].items()}, "nifty_ft": {k: v * 100 for k, v in ft["median_iv_rmse"].items()},
    }
    out["v1"] = v1

    # ---------------------------------------------------------------------------------------------- 2.0 results
    W = {r["day"]: r for r in pickle.load(open(RES / "walk_dhj_main.pkl", "rb"))}
    hyb = pickle.load(open(RES / "hyb_dhj_main.pkl", "rb"))
    ts = sorted(hyb["smile"])
    scaled = {t: EP.pred_smile_scaled(S[t], S[t + 1]) for t in ts}
    preds = {"yesterday's smile": {t: hyb["smile"][t].astype(float) for t in ts}, "time-scaled smile": scaled,
             "model (calibrated, refit every 10 days)": {t: hyb["carry"][t].astype(float) for t in ts},
             "model + yesterday's misfit": {t: hyb["carry_res"][t].astype(float) for t in ts}}
    sc = {k: {t: EP.score(p[t], S[t + 1])["rmse"] for t in ts} for k, p in preds.items()}
    dev, test = [t for t in ts if S[t].day < SPLIT], [t for t in ts if S[t].day >= SPLIT]

    # like-for-like 1: same-day fit against flat Black-Scholes, and the carried-forward test, on NIFTY 2025-26 (the daily-fit baselines exist there)
    fits = {m: {r["day"]: r for r in pickle.load(open(RES / f"daily_{m}_test2025.pkl", "rb")) if "x" in r} for m in ("bs", "heston", "dh")}
    days = [t for t in ts if S[t].day >= date(2025, 1, 1) and S[t].day in fits["bs"] and S[t].day in fits["dh"] and S[t].day in W]
    row = {}
    for t in days:
        S0, S1 = S[t], S[t + 1]
        d = {"bs_daily_carried": EP.pred_flat(float(fits["bs"][S0.day]["x"][0]), S1), "dh_daily_carried": EP.pred_model("dh", fits["dh"][S0.day]["x"], S1),
             "model_walk": preds["model (calibrated, refit every 10 days)"][t], "model_plus_misfit": preds["model + yesterday's misfit"][t], "scaled_smile": scaled[t]}
        for k, p in d.items():
            row.setdefault(k, []).append((EP.score(p, S1)["rmse"], rel_price_err(p, S1)))
    L = {k: np.array(v) for k, v in row.items()}
    same_day = {"bs": np.array([fits["bs"][S[t].day]["rmse"] for t in days]), "dh_daily": np.array([fits["dh"][S[t].day]["rmse"] for t in days]),
                "model_walk": np.array([W[S[t].day]["rmse_in"] for t in days])}
    out["like_for_like_2025_26"] = {"days": len(days), "same_day_median_iv_rmse": {k: float(np.median(v)) for k, v in same_day.items()},
                                    "next_day_iv_rmse_mean": {k: float(v[:, 0].mean()) for k, v in L.items()},
                                    "next_day_rel_price_err_median": {k: float(np.nanmedian(v[:, 1])) for k, v in L.items()},
                                    "beats_bs_carried": {k: float(np.mean(v[:, 0] < L["bs_daily_carried"][:, 0])) for k, v in L.items() if k != "bs_daily_carried"},
                                    "dh_beats_bs_same_day": float(np.mean(same_day["dh_daily"] < same_day["bs"])),
                                    "dh_over_bs_same_day_median": float(np.median(same_day["dh_daily"] / same_day["bs"]))}

    # like-for-like 2: the old NIFTY comparison's own ten high-volatility days
    sd = [date.fromisoformat(x) for x in eval(sel["dates"]) if not isinstance(sel["dates"], list)] if isinstance(sel["dates"], str) else [date.fromisoformat(x) for x in sel["dates"]]
    nd = [byday[d] for d in sd if d in byday and d in W]
    out["nifty_april_2026"] = {"days": len(nd), "same_day_iv_rmse_median": float(np.median([W[S[i].day]["rmse_in"] for i in nd])),
                               "next_day_iv_rmse_median": {k: float(np.median([sc[k][i] for i in nd if i in sc[k]])) for k in sc}}

    # the website's own example: NIFTY 23150 call, 27 Oct 2026, on 25 Sep 2026
    d0 = date(2026, 9, 25)
    if d0 in W and d0 in byday:
        s = S[byday[d0]]
        e = next(k for k, x in enumerate(s.expiry) if x == date(2026, 10, 27))
        raw_path = HERE / "data" / "nse_index" / f"{d0.isoformat()}.csv"
        # the NSE close of this contract (370.15), kept so the note rebuilds without the raw files; read from the file when it is there
        c = pd.DataFrame({"ClsPric": [370.15]})
        if raw_path.exists():
            raw = pd.read_csv(raw_path, dtype=str)
            c = raw[(raw.TckrSymb == "NIFTY") & (raw.FinInstrmTp == "IDO") & (raw.OptnTp == "CE") & (raw.XpryDt == "2026-10-27") & (raw.StrkPric.astype(float) == 23150)]
        x = calib.to_phys("dhj", ST.assemble("dhj", W[d0]["eta"], W[d0]["state"]))
        v1_, k1, t1, x1, r1, v2, k2, t2, x2, r2, lam, mu, dl = x
        T = s.T[e]
        log_cf = lambda u: (M.P._log_cf_factor(u, v1_, k1, t1, x1, r1, T) + M.P._log_cf_factor(u, v2, k2, t2, x2, r2, T) + M._jump_log_cf(u, T, lam, mu, dl))
        px = float(M.P._price_from_log_cf(log_cf, s.spot, 23150.0, T, s.r, s.q[e], "call"))
        mkt = float(c.ClsPric.iloc[0]) if len(c) else float("nan")
        out["site_example"] = {"market_close": mkt, "v1_default_model": 600.40, "v2_calibrated_model": px}

    # identifiability: how often is a calibrated structural parameter pinned at a bound?
    lo_b, hi_b = (np.array(b) for b in calib.BOUNDS["dhj"])
    ei = ST.IDX["dhj"][0]
    E = np.array([r["eta"] for r in W.values()])
    tol = 0.02 * (hi_b[ei] - lo_b[ei])
    at = (E <= lo_b[ei] + tol) | (E >= hi_b[ei] - tol)
    out["identifiability"] = {"days_with_any_param_near_bound": float(at.any(1).mean()), "per_param_share_at_bound": [float(x) for x in at.mean(0)]}

    # the full tournament over every saved run
    runs = {}
    for f in sorted(glob.glob(str(RES / "hyb_*.pkl"))):
        name = Path(f).stem[4:]
        h = pickle.load(open(f, "rb"))
        for v in ("carry", "carry_res"):
            p = {t: h[v][t].astype(float) for t in h[v] if t in scaled}
            s_ = {t: EP.score(p[t], S[t + 1])["rmse"] for t in p}
            tt = [t for t in p if S[t].day >= SPLIT]
            dd = [t for t in p if S[t].day < SPLIT]
            m, lo, hi = boot([s_[t] - sc["time-scaled smile"][t] for t in tt]) if len(tt) > 100 else (float("nan"),) * 3
            runs[f"{name} {v}"] = {"all": float(np.mean(list(s_.values()))), "dev": float(np.mean([s_[t] for t in dd])), "test": float(np.mean([s_[t] for t in tt])),
                                   "test_vs_scaled": [m, lo, hi]}
    out["runs"] = runs
    mem = {}
    for f in sorted(glob.glob(str(RES / "memory_*.json"))):
        if not f.endswith("final.json"):
            d = json.load(open(f)); mem[d["base"]] = d
    out["memory"] = mem
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "comparison.json").write_text(json.dumps(out, indent=1, default=float))

    # ---------------------------------------------------------------------------------------------- the note
    ll = out["like_for_like_2025_26"]
    w(f"# Double Heston: version 1.0 against version 2.0\n\n*Living note, regenerated by `update_comparison.py` on {datetime.now():%d %b %Y %H:%M}. "
      "Every 2.0 number is computed from the saved results; every 1.0 number is read from the research outputs (paths given).*\n")
    w("## What each version is\n")
    w("| | Version 1.0 (research + website) | Version 2.0 (this folder) |\n|---|---|---|\n"
      "| Question | Can Double Heston's ten parameters be recovered from option prices, and does the model price better than flat volatility? | Can a recalibrated model price tomorrow's options better than simple persistence, with a memory of repeating patterns? |\n"
      "| Data | 210 NSE stocks, 60 days (Jul-Sep 2026), at most 20 quotes per surface, 12,480 surfaces | NIFTY 50, ten years (Oct 2016 to Oct 2026), 2,470 days, whole chain (about 90 to 550 quotes a day) |\n"
      "| Error measure | price error relative to the mean price | implied-volatility error in volatility points (and, for comparison, the same relative price error) |\n"
      "| Calibration | each day independently (ANN, PINN, or multistart optimiser) | slow parameters shared over a 40-day window, only today's volatility levels fitted daily (this sharing is not shown to help: DECISIONS.md 11); implied-volatility objective, global multi-start and jumps added |\n"
      "| Pricer | the project's own `models.py` | the same file, unchanged; jumps added by passing a different characteristic function |\n")
    w("## 1. Old tests, side by side (where the question is the same)\n")
    w("| # | Test | Version 1.0 | Version 2.0 | Comparable? |\n|---|---|---|---|---|")
    w(f"| 1 | Does a Double Heston fit beat flat Black-Scholes? (same-day, in-sample) | beat it on {pct(v1['dh_beats_flat_share'], 0)} of 2,400 stock surfaces; median error {pct(1 - v1['dh_over_flat_median'], 0)} lower (price) `outputs/ambiguity/ambiguity_vs_flat.csv` | "
      f"independent daily Double Heston beats Black-Scholes on {pct(ll['dh_beats_bs_same_day'], 0)} of {ll['days']} NIFTY days, median error {pct(1 - ll['dh_over_bs_same_day_median'], 0)} lower; the 2.0 jump model's same-day error is {ll['same_day_median_iv_rmse']['model_walk']:.2f} vol points vs {ll['same_day_median_iv_rmse']['bs']:.2f} for Black-Scholes | same question, different data and measure |")
    w(f"| 2 | Parameters carried to the next day beat flat Black-Scholes carried (\"stale\" test) | Double Heston won on {pct(v1['stale_dh_beats_stale_bs'])} of 12,265 stock-days (network parameters; fresh parameters vs same-day BS: {pct(v1['fresh_dh_beats_same_day_bs'])}) `option_backtest_pairs.csv` | "
      f"independent daily Double Heston carried wins on {pct(ll['beats_bs_carried']['dh_daily_carried'], 0)} of {ll['days']} NIFTY days; 2.0 model {pct(ll['beats_bs_carried']['model_walk'], 0)}; model + misfit {pct(ll['beats_bs_carried']['model_plus_misfit'], 0)} | same question; 1.0 used network parameters on stocks, here optimiser fits on NIFTY. **Caution:** a flat volatility is a weak bar on an index whose smile runs from about 10% to 40%, so 100% says little; the real bar here is yesterday's smile (section 2) |")
    w(f"| 3 | Next-day relative price error (1.0's metric) | stale Double Heston {pct(v1['rel_err_stale_dh'])}, stale Black-Scholes {pct(v1['rel_err_stale_bs'])}, BS refit same day {pct(v1['rel_err_bs_same_day'])} (medians, 12,265 pairs) | "
      f"median over {ll['days']} NIFTY days: Black-Scholes carried {pct(ll['next_day_rel_price_err_median']['bs_daily_carried'])}, daily Double Heston carried {pct(ll['next_day_rel_price_err_median']['dh_daily_carried'])}, "
      f"2.0 model {pct(ll['next_day_rel_price_err_median']['model_walk'])}, 2.0 model + misfit {pct(ll['next_day_rel_price_err_median']['model_plus_misfit'])} | metric is the same, instruments differ (NIFTY has far more liquid quotes). Note the metric weights cheap far-out options heavily, so it understates the volatility-point gains in section 2 |")
    w(f"| 4 | Are the parameters identifiable from prices? | {pct(v1['multi_equiv'])} of surfaces had several equally good parameter sets; {pct(v1['any_near_bound'])} had a parameter near a bound `ambiguity_summary.json` | "
      f"{pct(out['identifiability']['days_with_any_param_near_bound'], 0)} of 2.0's calibrations have at least one structural parameter within 2% of a bound (1.0 used 3%), even with 40 days of data | same conclusion: the individual parameters are not pinned down; the prices they produce are |")
    nf = v1["nifty"]
    nd_ = out["nifty_april_2026"]
    w(f"| 5 | NIFTY, the old test's own 10 high-volatility days (8-22 Apr 2026) | held-out-quote IV error: Black-Scholes {nf['Black-Scholes, 1 sigma (1 param)']:.2f} pts, single Heston {nf['single Heston (5 params)']:.2f}, Double Heston {nf['Double Heston, cold 5-start (10 params)']:.2f}, "
      f"PINN + ridge {nf['dual PINN + ridge 3 (out of distribution)']:.2f} (`nifty_comparison_summary.json`) | same days, whole chain: 2.0 same-day fit error {nd_['same_day_iv_rmse_median']:.2f} pts; next-day forecast error, time-scaled smile {nd_['next_day_iv_rmse_median']['time-scaled smile']:.2f}, "
      f"2.0 model {nd_['next_day_iv_rmse_median']['model (calibrated, refit every 10 days)']:.2f}, model + misfit {nd_['next_day_iv_rmse_median']['model + yesterday\'s misfit']:.2f} | **not identical**: 1.0 fitted some quotes and scored the held-out ones; 2.0 scores a next-day forecast |")
    se = out.get("site_example")
    if se:
        w(f"| 6 | The website's example: NIFTY 23150 call, 27 Oct 2026, on 25 Sep 2026 | default settings: model ₹{se['v1_default_model']:.2f} vs market close ₹{se['market_close']:.2f}, gap +₹{se['v1_default_model'] - se['market_close']:.2f} | "
          f"2.0 calibrated on that day: model ₹{se['v2_calibrated_model']:.2f}, gap {se['v2_calibrated_model'] - se['market_close']:+.2f} | same contract; 2.0 is calibrated to that day's own chain (in-sample), 1.0 used untouched defaults |")
    w(f"| 7 | Held-out repricing of a neural-network calibrator (G8, NTPC) | network {pct(v1['g8_net'])} vs classical best fit {pct(v1['g8_fit'])} median relative price error | not repeated: 2.0 does not use a neural network to calibrate | n/a |")
    w("| 8 | Synthetic recovery (known parameters) | optimiser prices match to 9e-8 but recovered parameters score worse than guessing (skill 1.80) `consolidated_results.json` | not repeated: needs a synthetic generator for the jump model; the pytest suite checks that prices, not parameters, are recovered | n/a |")
    w("| 9 | PINN vs plain network (3D demo) | 0.8% vs 5.7% price error on a fixed demonstration | not repeated | n/a |\n")
    w("## 2. New tests in 2.0\n")
    w("All scored **next day**, using only what was known on the day, in implied-volatility points (lower is better). Settings are chosen on 2016-2021 only; 2022-2026 is the test period.\n")
    w("| Forecast | all 2,429 days | 2016-2021 (dev) | 2022-2026 (test) |\n|---|---|---|---|")
    for k in preds:
        a_ = np.mean([sc[k][t] for t in ts]); d_ = np.mean([sc[k][t] for t in dev]); t_ = np.mean([sc[k][t] for t in test])
        w(f"| {k} | {a_:.3f} | {d_:.3f} | {t_:.3f} |")
    w("\n**Every saved run** (mean next-day error; paired difference against the time-scaled smile on the test period, with a 95% block-bootstrap interval; negative = better):\n")
    w("| Run | all | dev | test | test difference vs time-scaled smile |\n|---|---|---|---|---|")
    for k, v in runs.items():
        m, lo, hi = v["test_vs_scaled"]
        w(f"| {k} | {v['all']:.3f} | {v['dev']:.3f} | {v['test']:.3f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    w("\n### Pattern memory (the repeating-pattern feature)\n")
    w("| Base forecast | memory settings chosen on dev | test without memory | with memory | paired difference | fired on |\n|---|---|---|---|---|---|")
    for b, d in mem.items():
        t = d["test"]; c = d["chosen_on_development"]
        w(f"| {b} | k={c['k']} q={c['q']} L={c['L']} β={c['beta']} | {t['without']['mean']:.4f} | {t['with']['mean']:.4f} | {t['paired_diff']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {pct(t['fired_share'], 0)} of days |")
    jr = {k: v["test"] for k, v in runs.items() if k.endswith("carry_res") and k.startswith("dhj")}
    if jr:
        w("\n### Do the settings matter? (robustness of the final forecast)\n")
        w(f"The final forecast (the model with today's state plus yesterday's misfit) was scored in {len(jr)} jump-model runs (the headline run itself plus {len(jr) - 1} reruns, one of them the known-buggy frozen-refit run) with different slow-parameter windows (20 to 120 days), refit schedules (every 5 to 20 days, and the accidental ~200-day run), prior strengths, a pattern-memory prior (with a plain-shrinkage control), wide bounds and the jump rate as a daily state. "
          f"Test error ranges only from {min(jr.values()):.3f} to {max(jr.values()):.3f} vol points, against {np.mean([sc['time-scaled smile'][t] for t in test]):.3f} for the time-scaled smile. The final forecast is almost insensitive to the model's slow settings, which is consistent with decision 11 (the misfit carried from yesterday supplies most of the forecast); it shows insensitivity to settings, not that the model's structure matters. "
          "The pure model forecast does move with these settings (more frequent refits and shorter windows help a little), but never clearly beats the benchmark. "
          "**Pattern memory as a prior for the parameters changes nothing**: all three memory runs and the plain-shrinkage control (`dhj_p10`) score the same to three decimals.\n")
    tr = {nm: json.loads((RES / f"trade_robust_{nm}.json").read_text()) for nm in ("dhj_main", "dhj_main_lag1") if (RES / f"trade_robust_{nm}.json").exists()}
    if tr:
        w("\n### Trading test: following the model one trade at a time\n")
        w("A trade is one NIFTY option (the 5 most-traded contracts a day within 5% of the forward, 5 to 45 days to expiry), bought if the model says it is cheap and sold if rich (or its volatility forecast is higher / lower than today's market volatility), at the day's close, closed at the next close; a win is a profit above zero after cost. "
          "'Hedged' removes the index move with the option's delta on the forward. Thresholds were chosen on 2016-2021; the rows are the 2022-2026 test years. Cost is a share of the option price on each of entry and exit. Ranges are 95% intervals resampling whole days (trades of one day move together).\n")
        w("| Signal / holding | Entry | Cost per side | Trades | Wins per 100 | Blind trades, same buy/sell mix | Rs per trade [95% range] | Median trade | Trimmed mean |\n|---|---|---|---|---|---|---|---|---|")
        for nm, lab in (("dhj_main", "signal day close"), ("dhj_main_lag1", "one day later")):
            for tag, v in tr.get(nm, {}).items():
                for ck in ("cost_0.0", "cost_0.01"):
                    r = v.get(ck)
                    if r:
                        w(f"| {tag} (edge ≥ {v['threshold']} vol pts) | {lab} | {float(ck[5:]) * 100:.0f}% | {v['trades']:,} | {r['wins_per_100']:.1f} | {r['blind_same_mix_wins_per_100']:.1f} | {r['mean_rs']:+.2f} [{r['mean_rs_ci'][0]:+.2f}, {r['mean_rs_ci'][1]:+.2f}] | {r['median_rs']:+.2f} | {r['trimmed_rs']:+.2f} |")
        mt = RES / "trade_matched_dhj_main.json"
        if mt.exists():
            M_ = json.loads(mt.read_text())
            w("\n**The same trades, entered one day later** (matched on signal day and contract; only the entry price changes; threshold as above). "
              "The one-day-later file keeps only contracts that still have at least 3 days to expiry two days on, which removes the nearest weekly contracts on Thursday and Friday signals, so the matched comparison is the fair one.\n")
            w("| Signal / holding | Matched trades | Wins per 100: signal close / one day later | Rs per trade (no cost): signal close | one day later | Profit lost by waiting [95% range] | Rs per trade at 1% cost: signal close / one day later |\n|---|---|---|---|---|---|---|")
            for tag, v in M_.items():
                a0, a1 = v["cost_0.0"], v["cost_0.01"]
                w(f"| {tag} | {v['trades']:,} | {a0['same_day_wins']:.1f} / {a0['next_day_wins']:.1f} | {a0['same_day_rs']:+.2f} | {a0['next_day_rs']:+.2f} | {a0['lost_rs']:+.2f} [{a0['lost_ci'][0]:+.2f}, {a0['lost_ci'][1]:+.2f}] | {a1['same_day_rs']:+.2f} / {a1['next_day_rs']:+.2f} |")
        w("\n**Reading it.** In the first table 41.1 to 45.0 of 100 trades profit (in the matched table below, 40.6 to 48.2 over both entry days; 37.7 at 1% cost). A coin flip is not the right yardstick: the relevant control is the 'blind trades, same buy/sell mix' column (38.5 to 44.9), and the model beats it by 1.5 to 3.0 wins per 100 in the signal-day rows; always selling the same options wins 61 to 65 of 100 trades (results/VERIFICATION.md, L7), so the win count alone is a weak signal. The model mostly says buy, and bought options lose a day of time value, so the typical trade loses; the profit comes from wins being larger than losses. "
          "Entered at the signal day's closing price, all four versions have a positive average whose range stays above zero with no cost (lower ends +1.35, +1.83, +0.22, +2.33 Rs), and only the index-hedged rich/cheap version stays above zero at a 1% cost (+0.19). "
          "Entered one day later on the same trades, the two hedged versions lose most of their profit (about 5 to 6 Rs per trade, from about +6.5 / +7.2 to +1.5 / +1.2) and turn negative at 1% cost, but the size of that loss is itself not statistically certain (its range includes zero); the plain forecast version loses nothing; and none of the matched one-day-later averages is distinguishable from zero. This is not true of the first table: its 4,525-trade hedged rows (no signal threshold) are -2.04 [-3.31, -0.76] and -2.02 [-3.25, -0.70] Rs per trade at a 1% cost, which exclude zero (negative). "
          "So the evidence is: a possible edge at the closing print that has not been shown to survive a one-day delay. An independent audit found no look-ahead; closing prices are last trades, not quotes you can deal at. This is a backtest, not investment advice.\n")
    w("\n## 3. Findings that changed along the way\n")
    w("- **1.0 had no benchmark that a model had to beat.** In 2.0, yesterday's smile carried forward is the bar; a plain independent daily Double Heston fit is much worse than it (2025-26: 2.09 vs 1.23 vol points).\n"
      "- **The jump component is what makes the model fit.** Same-day error drops from about 1.7 (Double Heston) to about 0.6 (with jumps) over 2025-26; it prices the crash premium of far-out options (gain concentrated beyond 5% moneyness and at 3-7 day expiries).\n"
      "- **The pure model forecast ties the time-scaled smile; the model plus yesterday's remaining misfit beats it** (about 16% on the test period), but mostly at strikes that yesterday did not quote: against a benchmark that continues the wings linearly the lead is -0.045 [-0.061, -0.032] instead of -0.202, and inside yesterday's quoted range (93.5% of quotes) the benchmark is slightly better (results/benchmark_check/; DECISIONS.md decision 12).\n"
      "- **Refitting the slow parameters often matters** (every 10 days vs about every 200 days: 1.48 vs 1.91 over the decade).\n"
      "- **Pattern memory helps weak forecasts a little and the best forecast not at all** as a forecast correction; its second use, as a starting belief for the parameters, changed nothing (three memory runs and the plain-shrinkage control all score alike).\n"
      "- **The model's own dynamics add nothing for the jump model:** one-day mean reversion -0.002 [-0.006, +0.001], the leverage update (volatility moves with the index through the correlation) +0.090 (worse), leverage plus misfit +0.102 (worse), yesterday's smile plus the model-implied change +0.084 (worse) [results/tables/forecast_variants.md].\n"
      "- **Jump rate as a daily state:** no benefit over a structural jump rate. **Wide bounds:** help the pure model in the 2020 crash year, but not the final forecast (dev 1.323 vs 1.317), so the final configuration keeps the default bounds.\n"
      "- **The jumps, not the second factor, carry the gain:** without jumps the same recipe scores 1.377 (Double Heston) and 1.416 (single Heston) on test, worse than the time-scaled smile (1.236).\n")
    w("## 4. Bugs found and fixed in 2.0 (kept so no result is trusted blindly)\n")
    w("- A fit-subset bug (`price()` returned uninitialised values for the unrequested quotes): found by the first failed fit, fixed before any result.\n"
      "- A variable-name clash in the walk-forward driver froze the slow parameters for about 200 days at a time; found by counting distinct parameter sets per year, fixed and rerun. The frozen run is kept as the `slowrefit` ablation (it is a valid causal run) and a regression test now checks the refit schedule.\n"
      "- An early assumption that the 10-year run would take minutes; the machine has 4 fast and 6 slow cores, so nine workers give about 3x, and runs are queued one at a time.\n")
    w("## 5. What 2.0 does not claim\n")
    w("- The individual calibrated parameters are not identified and should not be interpreted one by one; the surfaces they produce are what is validated.\n"
      "- Near the money, where yesterday's smile is already excellent, the time-scaled smile is slightly better than the jump model + misfit; inside the strike range yesterday quoted (93.5% of quotes) it is also slightly better (+0.012 vol points). The gains are at strikes yesterday did not quote and at the shortest expiries.\n"
      "- Closing prices of illiquid options can be stale, which flatters persistence in the thin early years (2016-2018); the floor under all errors is not zero.\n"
      "- Several memory and jump settings landed at the edge of the grid searched; they were chosen on the development years only, but a wider search could move them.\n"
      "- 1.0 and 2.0 differ in data, instrument and measure; rows marked not identical should not be read as a like-for-like win or loss.\n")
    (HERE / "COMPARISON_1.0_vs_2.0.md").write_text("\n".join(md))
    # dated change log: one line each time the set of results or the headline numbers change
    best = min((v["test"] for k, v in runs.items() if k.endswith("carry_res")), default=float("nan"))
    best_pure = min((v["test"] for k, v in runs.items() if k.endswith(" carry")), default=float("nan"))
    digest = f"{len(runs)} runs; test mean error: best model+misfit {best:.3f}, best pure model {best_pure:.3f}, time-scaled smile {np.mean([sc['time-scaled smile'][t] for t in test]):.3f}; memory runs {len(mem)}"
    logf = HERE / "COMPARISON_LOG.md"
    old = logf.read_text() if logf.exists() else "# Comparison log (one line each time the results change)\n\n"
    if digest not in old:
        logf.write_text(old + f"- {datetime.now():%d %b %Y %H:%M}: {digest}\n")
    print("written", HERE / "COMPARISON_1.0_vs_2.0.md")


if __name__ == "__main__":
    main()
