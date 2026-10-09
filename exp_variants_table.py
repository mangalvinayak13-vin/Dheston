"""The forecast variants of hybrids.py, all scored on the test period (the null results that the main tables did not show).

    python3 exp_variants_table.py        # writes results/tables/forecast_variants.csv and forecast_variants.md

hybrids.py stores seven forecasts per model (data/results/hyb_<run>.pkl):

  smile       yesterday's smile carried in log-moneyness (no model)
  carry       the model with today's calibrated state
  revert      carry, with each variance factor mean-reverting for one day (the model's own forecast of its dynamics)
  lev         revert plus the leverage update (variance moves with the index through rho; the model's own conditional expectation)
  carry_res   carry plus yesterday's misfit (the headline forecast)
  lev_res     lev plus the same misfit carry
  smile_lev   yesterday's smile plus the model-implied change (lev - carry)

The main tables (final_table.py, results/tables/forecast_scores.csv) show only carry and carry_res. The independent verification
(results/VERIFICATION.md, L2) found that the other four were computed and not reported. This script scores them exactly as final_table.py
does: days = the intersection of the headline runs, dev = origin day before 2022-01-01, test = origin day from 2022-01-01, error of a day =
dh2.evalpairs.score(...)["rmse"], intervals = tournament.boot (95% moving-block bootstrap, blocks of 20 days, 2,000 resamples, seed 0).
Each variant is compared with the variant it is meant to improve on (its "baseline" column) and with the time-scaled smile.
Nothing is refitted: every number is read from the stored forecasts.
"""
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dh2 import evalpairs as EP
from dh2.data import load_all
from tournament import boot

RES = HERE / "data" / "results"
OUT = HERE / "results" / "tables"
SPLIT = date(2022, 1, 1)
RUNS = ("dhj_main", "dh_main", "heston_main")                      # the headline runs (configs/nifty.yaml, forecasts.headline_runs)
RUN_LABEL = {"dhj_main": "jump model", "dh_main": "diffusion Double Heston", "heston_main": "single Heston"}
VARIANTS = ("smile", "carry", "revert", "lev", "carry_res", "lev_res", "smile_lev")
LABEL = {"smile": "yesterday's smile (no model)", "carry": "model, today's state", "revert": "model + its one-day mean reversion",
         "lev": "model + mean reversion + leverage update", "carry_res": "model + yesterday's misfit (headline forecast)",
         "lev_res": "model + leverage update + yesterday's misfit", "smile_lev": "yesterday's smile + model-implied change"}
BASELINE = {"smile": None, "carry": None, "revert": "carry", "lev": "carry", "carry_res": "carry", "lev_res": "carry_res", "smile_lev": "smile"}


def main():
    S = load_all()
    H = {n: pickle.load(open(RES / f"hyb_{n}.pkl", "rb")) for n in RUNS}
    common = sorted(set.intersection(*[set(h["carry"]) for h in H.values()]))
    is_test = np.array([S[t].day >= SPLIT for t in common])
    is_dev = ~is_test
    score = lambda p: np.array([EP.score(p[t], S[t + 1])["rmse"] for t in common])
    scaled = score({t: EP.pred_smile_scaled(S[t], S[t + 1]) for t in common})
    series = {n: {v: score({t: H[n][v][t].astype(float) for t in common}) for v in VARIANTS} for n in RUNS}

    rows = []
    for n in RUNS:
        for v in VARIANTS:
            sc = series[n][v]
            mean, lo, hi = boot(sc[is_test])
            r = {"run": n, "model": RUN_LABEL[n], "variant": v, "label": LABEL[v], "n_dev_days": int(is_dev.sum()), "n_test_days": int(is_test.sum()),
                 "dev_mean": float(np.nanmean(sc[is_dev])), "test_mean": float(np.nanmean(sc[is_test])), "test_mean_ci_lo": lo, "test_mean_ci_hi": hi,
                 "baseline": BASELINE[v] or "", "dev_diff_vs_baseline": np.nan, "diff_vs_baseline": np.nan, "diff_baseline_ci_lo": np.nan, "diff_baseline_ci_hi": np.nan,
                 "share_test_days_better_than_baseline": np.nan}
            if BASELINE[v]:
                base = series[n][BASELINE[v]]
                d = sc[is_test] - base[is_test]
                m_, dlo, dhi = boot(d)
                r.update(dev_diff_vs_baseline=float(np.nanmean(sc[is_dev] - base[is_dev])), diff_vs_baseline=m_, diff_baseline_ci_lo=dlo, diff_baseline_ci_hi=dhi,
                         share_test_days_better_than_baseline=float(np.mean(d < 0)))
            d2 = sc[is_test] - scaled[is_test]
            m2, l2, h2 = boot(d2)
            r.update(diff_vs_time_scaled_smile=m2, diff_scaled_ci_lo=l2, diff_scaled_ci_hi=h2)
            rows.append(r)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "forecast_variants.csv", index=False)

    f3 = lambda x: "" if (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    fs = lambda x: "" if (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"
    ci = lambda a, b: "" if np.isnan(a) else f"[{a:+.3f}, {b:+.3f}]"
    md = ["**Forecast variants of hybrids.py on the test period (mean next-day implied-volatility error, vol points, lower is better)**", "",
          "| Model | Forecast | Dev | Test | Compared with | Test minus that forecast [95% interval] | Days better | Test minus time-scaled smile [95% interval] |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        base = LABEL[r.baseline] if r.baseline else ""
        share = "" if np.isnan(r.share_test_days_better_than_baseline) else f"{r.share_test_days_better_than_baseline:.0%}"
        md.append(f"| {r.model} | {r.label} | {f3(r.dev_mean)} | {f3(r.test_mean)} | {base} | {fs(r.diff_vs_baseline)} {ci(r.diff_baseline_ci_lo, r.diff_baseline_ci_hi)} | {share} | "
                  f"{fs(r.diff_vs_time_scaled_smile)} {ci(r.diff_scaled_ci_lo, r.diff_scaled_ci_hi)} |")
    t = float(np.nanmean(scaled[is_test]))
    md += ["", f"*Test = origin days from {SPLIT} ({int(is_test.sum())} days), Dev = before ({int(is_dev.sum())} days); days are the intersection of the three headline runs, as in "
           f"final_table.py. The time-scaled smile (no model) scores {t:.3f} on the test days. Intervals: 95% moving-block bootstrap (blocks of 20 days, 2,000 resamples). "
           "A negative difference means the forecast in that row is better than the one it is compared with. Nothing is refitted: the numbers are read from data/results/hyb_<run>.pkl "
           "and scored with dh2/evalpairs.py. Written by exp_variants_table.py.*", ""]
    (OUT / "forecast_variants.md").write_text("\n".join(md))
    print(f"{len(common)} common days ({int(is_dev.sum())} dev, {int(is_test.sum())} test); time-scaled smile test mean {t:.3f}")
    show = out[["model", "variant", "dev_mean", "test_mean", "baseline", "diff_vs_baseline", "diff_baseline_ci_lo", "diff_baseline_ci_hi", "diff_vs_time_scaled_smile"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
