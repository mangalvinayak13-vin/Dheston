"""Presentation figures of the one-command pipeline  ->  results/figures/*.png (300 dpi) and *.pdf, numbered, plus INDEX.md.

Two kinds of figure, both listed in REGISTRY (the single place that knows a figure's file name, caption and source, and from which INDEX.md is written):

  new   built here from result files, with the shared large-type style of dh2/plotstyle.py (16:9, opaque light panel so it reads on dark and
        light slide backgrounds): 01 summary panel, 02 error by year, 03 model comparison bars, 05 market-versus-model smiles on three
        real days, 08 calibrated volatility and jump rate over ten years, 17 pricer validation, 34 the two model-free benchmarks inside and outside
        the strike range that yesterday quoted.
  copy  figures that the other work packages already produced and that are good as they are: copied byte for byte (PNG) and also written
        as PDF (the raster embedded at its 300 dpi), under a numbered name.

Numbers on the new figures are read from results/metrics.json or from the result files named in the registry; none is typed here.
Run through the pipeline (stage `figures`) or directly:  python -m dh2.report_figures [--config configs/nifty.yaml]
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import plotstyle as PS  # noqa: E402
from .report import LABELS, MissingSource, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAL = PS.PAL


@dataclass
class Fig:
    key: str
    stem: str                 # numbered file name without extension
    caption: str              # one plain sentence, no numbers
    source: str               # script / module that produces it
    data: str                 # data files it is drawn from
    kind: str                 # new | copy
    src: str | None = None    # for copies: the existing PNG (relative to the repository root)


REGISTRY: list[Fig] = [
    Fig("summary", "01_summary_panel", "One page with the headline results: forecast error against the benchmarks, the step-by-step improvement from the 1.0 recipe, the SVI yardstick, "
        "the PINN-accelerated calibration and the link to India VIX.", "dh2/report_figures.py", "results/metrics.json", "new"),
    Fig("by_year", "02_error_by_year", "Average next-day forecast error in each calendar year for the two model-free benchmarks and the jump model with and without yesterday's misfit.",
        "dh2/report_figures.py", "results/tables/forecast_by_year.csv (from data/results/hyb_*_main.pkl)", "new"),
    Fig("bars", "03_model_comparison", "Mean next-day error on the test period for every model and both benchmarks, with 95% bootstrap intervals.",
        "dh2/report_figures.py", "results/tables/forecast_scores.csv (from data/results/hyb_*_main.pkl)", "new"),
    Fig("waterfall", "04_waterfall_dh1_to_dh2", "How the error falls from the 1.0 recipe to the 2.0 forecast, one fix at a time, same-day and next-day.",
        "exp_attribution.py", "results/attribution/waterfall.csv", "copy", "results/attribution/waterfall.png"),
    Fig("smiles", "05_market_vs_model_smiles", "Market implied volatilities against the diffusion Double Heston, the jump model and the SVI curve, for three expiries on three real test days.",
        "dh2/report_figures.py", "data/cache/surfaces.pkl, data/results/walk_dh_main.pkl, walk_dhj_main.pkl, results/svi/sample_days.csv", "new"),
    Fig("miss", "06_where_models_miss", "Same-day implied-volatility error by moneyness and time to expiry for six models, as heat maps.",
        "exp_buckets.py", "results/buckets/bucket_rmse.csv", "copy", "results/buckets/heatmaps_rmse.png"),
    Fig("counts", "07_quote_counts_by_bucket", "How many quotes sit in each moneyness and maturity cell of the heat maps.",
        "exp_buckets.py", "results/buckets/bucket_rmse.csv", "copy", "results/buckets/heatmaps_counts.png"),
    Fig("paths", "08_calibrated_volatility_and_jumps", "The calibrated volatility level and the jump rate of the jump model over ten years against the market's own 30-day at-the-money volatility.",
        "dh2/report_figures.py", "data/results/walk_dhj_main.pkl, data/cache/surfaces.pkl", "new"),
    Fig("svi_day", "09_svi_vs_dh_by_day", "Same-day smile fit: SVI reference against the two Double Heston models on the sampled test days.",
        "dh2/svi.py", "results/svi/svi_vs_dh_by_day.csv", "copy", "results/svi/svi_vs_dh_rmse_by_day.png"),
    Fig("svi_bucket", "10_svi_vs_dh_by_bucket", "In-sample implied-volatility error of SVI and the Double Heston models by time to expiry and forward log-moneyness.",
        "dh2/svi.py", "results/svi/svi_vs_dh_by_bucket.csv", "copy", "results/svi/svi_vs_dh_buckets.png"),
    Fig("svi_ex", "11_svi_example_smile", "One expiry on one day: market quotes, SVI and the two Double Heston fits.",
        "dh2/svi.py", "results/svi/svi_fits.csv", "copy", "results/svi/svi_example_smile.png"),
    Fig("vix_ts", "12_vix_timeseries", "India VIX against the model-implied and surface at-the-money 30-day volatility over ten years.",
        "dh2/vixcompare.py", "results/vix/vix_vs_model_daily.csv", "copy", "results/vix/fig_timeseries.png"),
    Fig("vix_lv", "13_vix_scatter_levels", "India VIX against the model-implied 30-day volatility in levels, with R-squared and slope.",
        "dh2/vixcompare.py", "results/vix/vix_vs_model_daily.csv", "copy", "results/vix/fig_scatter_levels.png"),
    Fig("vix_ch", "14_vix_scatter_changes", "India VIX against the model-implied 30-day volatility in daily changes, with R-squared and slope.",
        "dh2/vixcompare.py", "results/vix/vix_vs_model_daily.csv", "copy", "results/vix/fig_scatter_changes.png"),
    Fig("vix_ll", "15_vix_lead_lag", "Correlation of the model's daily change with the VIX change at different lags.",
        "dh2/vixcompare.py", "results/vix/vix_leadlag.csv", "copy", "results/vix/fig_leadlag.png"),
    Fig("vix_gap", "16_vix_gap_and_rolling_correlation", "The gap between model-implied volatility and India VIX over time, and their rolling correlation.",
        "dh2/vixcompare.py", "results/vix/vix_vs_model_daily.csv", "copy", "results/vix/fig_gap_and_rolling_corr.png"),
    Fig("pricer", "17_pricer_validation", "Largest difference between the project pricer and the independent COS pricer by maturity, and the cost per option of each pricer.",
        "dh2/report_figures.py", "results/pricer_validation/errors.csv, timing.json", "new"),
    Fig("xsec", "18_crosssection", "Held-out implied-volatility error of four models on a random 80/20 split of the quotes and on a split by maturity.",
        "exp_crosssection.py", "results/crosssection/summary.csv", "copy", "results/crosssection/crosssection.png"),
    Fig("ident", "19_identifiability", "Identifiability: Jacobian condition numbers, parameters sitting on a bound, approximate standard errors and the number of equally good parameter sets.",
        "exp_identifiability.py", "results/identifiability/summary.csv", "copy", "results/identifiability/identifiability.png"),
    Fig("pinn_abl", "20_pinn_ablation", "PINN ablation: held-out implied-volatility error of the hybrid, data-only, physics-only and spec-gate networks.",
        "dh2/pinn/figures.py", "results/pinn/tables/ablation_summary.csv", "copy", "results/pinn/figures/ablation_bars.png"),
    Fig("pinn_de", "21_pinn_data_efficiency", "PINN data efficiency: error against the number of supervised samples for the data-only and hybrid networks.",
        "dh2/pinn/figures.py", "results/pinn/tables/data_efficiency.csv", "copy", "results/pinn/figures/data_efficiency.png"),
    Fig("pinn_err", "22_pinn_error_maps", "PINN error maps over moneyness and maturity on a regular grid at held-out parameter sets, for the data-only and hybrid networks.",
        "dh2/pinn/figures.py", "results/pinn/tables/heldout_metrics_all_runs.csv", "copy", "results/pinn/figures/error_heatmaps.png"),
    Fig("pinn_pde", "23_pinn_pde_residual_maps", "PDE residual maps of the data-only, hybrid and physics-only networks on held-out parameter sets.",
        "dh2/pinn/figures.py", "results/pinn/tables/ablation_summary.csv", "copy", "results/pinn/figures/pde_residual_maps.png"),
    Fig("pinn_speed", "24_pinn_speed", "Microseconds per priced quote: surrogate on CPU and GPU against the exact Fourier pricer on one real chain.",
        "dh2/pinn/figures.py", "results/pinn/tables/speed.csv", "copy", "results/pinn/figures/speed.png"),
    Fig("pinn_gr", "25_pinn_greeks", "Greeks of the surrogate by automatic differentiation against finite differences of the exact pricer, for the data-only and hybrid networks.",
        "dh2/pinn/figures.py", "results/pinn/greeks_points.npz, results/pinn/tables/greeks_errors.csv", "copy", "results/pinn/figures/greeks_scatter.png"),
    Fig("pinn_sm", "26_pinn_smile_example", "One real held-out day at three expiries: the exact smile, both surrogates, and their differences.",
        "dh2/pinn/figures.py", "results/pinn/tables/heldout_metrics_all_runs.csv", "copy", "results/pinn/figures/smile_example.png"),
    Fig("pinn_tr", "27_pinn_training_curves", "Training curves of the shipped networks on the validation set of the training distribution.",
        "dh2/pinn/figures.py", "results/pinn/tables/train_*.csv", "copy", "results/pinn/figures/training_curves.png"),
    Fig("cal1", "28_pinn_calibration_rmse", "Same-day implied-volatility error of the exact and the PINN-accelerated calibration on each test day, scored with the exact pricer.",
        "exp_pinn_calib.py", "results/pinn_calib/per_day.csv", "copy", "results/pinn_calib/fig1_rmse_paired.png"),
    Fig("cal2", "29_pinn_calibration_cost_ratio", "Share of days that reach each objective cost relative to the exact calibration, for the accelerated, surrogate-only and control methods.",
        "exp_pinn_calib.py", "results/pinn_calib/per_day.csv", "copy", "results/pinn_calib/fig2_cost_ratio.png"),
    Fig("cal3", "30_pinn_calibration_time", "Wall time per day of the exact and accelerated calibration, and where the accelerated time goes.",
        "exp_pinn_calib.py", "results/pinn_calib/timing_by_method.csv", "copy", "results/pinn_calib/fig3_time.png"),
    Fig("cal4", "31_pinn_calibration_parameters", "Per-parameter difference between the accelerated and exact solutions, and cost difference against the largest parameter gap day by day.",
        "exp_pinn_calib.py", "results/pinn_calib/parameter_agreement.csv", "copy", "results/pinn_calib/fig4_parameters.png"),
    Fig("cal5", "32_pinn_calibration_rmse_over_time", "Difference in same-day error between the accelerated and the exact calibration through time (above zero means worse).",
        "exp_pinn_calib.py", "results/pinn_calib/per_day.csv", "copy", "results/pinn_calib/fig5_rmse_difference_over_time.png"),
    Fig("cal6", "33_pinn_calibration_design", "Wall time against objective cost for the design variants and polish budgets on development days, with the chosen design marked.",
        "exp_pinn_calib.py", "results/pinn_calib/design_per_day.csv", "copy", "results/pinn_calib/fig6_design_development_days.png"),
    Fig("wing", "34_benchmark_inside_outside", "Next-day error of two model-free benchmarks and the jump model with yesterday's misfit, for all quotes and for the quotes inside and outside the strike range "
        "that yesterday quoted.", "dh2/report_figures.py", "results/benchmark_check/summary.csv (via results/metrics.json)", "new"),
]
FIG = {f.key: f.stem for f in REGISTRY}


# ============================================================================================================== output helpers
def save_new(fig, stem: str, out_dir: Path, dpi: int = 300, formats=("png", "pdf")) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor())
        paths.append(p)
    plt.close(fig)
    return paths


def copy_with_pdf(src: Path, stem: str, out_dir: Path) -> list[Path]:
    """Copy a finished PNG under its numbered name and write a PDF that embeds the same raster at its own resolution."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise MissingSource(f"missing figure {src} (the stage that draws it has not run)")
    png = out_dir / f"{stem}.png"
    shutil.copyfile(src, png)
    im = Image.open(src)
    dpi = im.info.get("dpi", (300, 300))[0] or 300
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, PS.SURFACE)
        bg.paste(im, mask=im.split()[3])
        im = bg
    else:
        im = im.convert("RGB")
    pdf = out_dir / f"{stem}.pdf"
    im.save(pdf, "PDF", resolution=float(max(dpi, 72)))
    return [png, pdf]


def _fig(cfg, nrows=1, ncols=1, **kw):
    PS.setup()
    w, h = cfg["figures"]["size_inches"]
    return plt.subplots(nrows, ncols, figsize=(w, h), constrained_layout=True, **kw)


def _mvals(root: Path) -> dict:
    p = root / "results" / "metrics.json"
    if not p.exists():
        raise MissingSource("results/metrics.json is missing (stage metrics_tables has not run)")
    d = json.loads(p.read_text())["metrics"]
    return {k: r["value"] for k, r in d.items()}


def _v(mv: dict, key: str):
    try:
        return mv[key]
    except KeyError:
        raise MissingSource(f"metric {key!r} not in results/metrics.json") from None


# ============================================================================================================== 02 error by year
def fig_by_year(cfg, root, out_dir):
    f = root / cfg["paths"]["tables"] / "forecast_by_year.csv"
    if not f.exists():
        raise MissingSource(f"missing {f}")
    d = pd.read_csv(f)
    series = [("smile", "yesterday's smile", PS.INK2, "-", 2.2), ("smile_scaled", "time-scaled smile", "#8a8884", "-", 2.2),
              ("dhj_main:carry", "jump model", PAL[0], "-", 2.6), ("dhj_main:carry_res", "jump model + misfit", PAL[1], "-", 3.6)]
    for s, *_ in series:
        if s not in d.columns:
            raise MissingSource(f"{f}: no column {s!r}")
    fig, ax = _fig(cfg)
    test0 = cfg["split"]["test"]["start"].year
    ax.axvspan(test0 - 0.5, d["year"].max() + 0.5, color="#efede7", zorder=0)
    ax.text((test0 + d["year"].max()) / 2, 0.97, "test period", transform=ax.get_xaxis_transform(), ha="center", va="top", color=PS.INK2, fontsize=15)
    ends = {}
    for s, lab, c, ls, lw in series:
        ax.plot(d["year"], d[s], ls, color=c, lw=lw, marker="o", ms=6, label=lab)
        ends[lab] = (d[s].iloc[-1], c)
    # direct labels at the right edge, spread so they do not overlap
    last = None
    for lab, (y, c) in sorted(ends.items(), key=lambda kv: -kv[1][0]):
        ly = y if last is None else min(y, last - 0.17)
        ax.annotate(lab, (d["year"].iloc[-1], y), xytext=(d["year"].iloc[-1] + 0.45, ly), textcoords="data", color=PS.INK, fontsize=15, va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.0, shrinkA=0, shrinkB=2))
        last = ly
    ax.set_xlim(d["year"].min() - 0.3, d["year"].max() + 3.0)
    ax.set_xticks(d["year"])
    ax.set_ylim(0, None)
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Average next-day error (implied-volatility points)")
    ax.set_title("Next-day forecast error by year, NIFTY 50 options")
    return save_new(fig, FIG["by_year"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])


# ============================================================================================================== 03 model comparison bars
def _bar_series(cfg):
    runs = ["heston_main", "dh_main", "dhj_main"]
    out = [("smile", "none", "yesterday's smile", "#a9a7a1"), ("smile_scaled", "none", "time-scaled smile", PS.INK2)]
    cols = {"heston_main": PAL[3], "dh_main": PAL[2], "dhj_main": PAL[1]}
    for r in runs:
        out.append((r, "carry", LABELS[f"{r}:carry"], cols[r]))
        out.append((r, "carry_res", LABELS[f"{r}:carry_res"], cols[r]))
    return out


def fig_bars(cfg, root, out_dir):
    f = root / cfg["paths"]["tables"] / "forecast_scores.csv"
    if not f.exists():
        raise MissingSource(f"missing {f}")
    d = pd.read_csv(f)
    rows = []
    for run, var, lab, col in _bar_series(cfg):
        r = d[(d.run == run) & (d.variant == var)]
        if len(r) != 1:
            raise MissingSource(f"{f}: expected one row for {run}/{var}, found {len(r)}")
        rows.append((lab, col, var == "carry", r.iloc[0]))
    best = d.best_benchmark.iloc[0]
    best_mean = float(d[(d.run == best) & (d.variant == "none")].test_mean.iloc[0])
    fig, ax = _fig(cfg)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, col, pure, r) in zip(y, rows):
        ax.barh(yi, r.test_mean, color=col, alpha=0.55 if pure else 1.0, hatch="//" if pure else None, edgecolor="white", height=0.68)
        ax.errorbar(r.test_mean, yi, xerr=[[r.test_mean - r.test_mean_ci_lo], [r.test_mean_ci_hi - r.test_mean]], fmt="none", ecolor=PS.INK, capsize=6, lw=2)
        ax.text(r.test_mean_ci_hi + 0.06, yi, f"{r.test_mean:.3f}", va="center", fontsize=16, color=PS.INK, bbox=dict(facecolor=PS.SURFACE, edgecolor="none", pad=1.5), zorder=5)
    ax.axvline(best_mean, color=PS.INK, ls="--", lw=1.6)
    ax.text(best_mean + 0.03, 0.99, f"best benchmark: {LABELS[best]}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=14, color=PS.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, float(max(r[3].test_mean_ci_hi for r in rows)) * 1.14)
    ax.set_xlabel("Mean next-day error on the test period (implied-volatility points; lower is better)")
    ax.set_title("Next-day forecast error on the test period (95% bootstrap intervals)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PS.INK2, alpha=0.55, hatch="//", label="model alone (today's calibrated state)"), Patch(facecolor=PS.INK2, label="model + yesterday's remaining misfit")],
              loc="lower right")
    return save_new(fig, FIG["bars"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])


# ============================================================================================================== 05 smiles on three real days
def smile_days(cfg, root, S):
    """The three days of the smile figure, fixed by rule: the paired-sample days ranked by the market's own 30-day ATM volatility; the days at the configured percentiles."""
    from . import memory as MEM
    sample = pd.read_csv(root / "results" / "svi" / "sample_days.csv")["day"].tolist()
    byday = {s.day.isoformat(): i for i, s in enumerate(S)}
    vol = np.array([MEM._civ(S[byday[d]], 30 / 365, 0.0) for d in sample])
    ok = np.isfinite(vol)
    days = [d for d, o in zip(sample, ok) if o]
    vv = vol[ok]
    order = np.argsort(vv)
    picks = []
    for p in cfg["evaluation"]["smile_figure_days"]["percentiles"]:
        k = order[int(round(p / 100.0 * (len(order) - 1)))]
        picks.append((days[k], float(vv[k]), p))
    return picks, byday


def fig_smiles(cfg, root, out_dir):
    from . import calib, models as M, struct as ST, svi
    from .data import load_all
    S = load_all()
    picks, byday = smile_days(cfg, root, S)
    res = root / cfg["paths"]["walk_results"]
    W = {}
    for m in ("dh", "dhj"):
        p = res / f"walk_{m}_main.pkl"
        if not p.exists():
            raise MissingSource(f"missing {p}")
        W[m] = {r["day"]: r for r in pickle.load(open(p, "rb"))}
    PS.setup()
    w, h = cfg["figures"]["size_inches"]
    fig, axs = plt.subplots(3, 3, figsize=(w, h), constrained_layout=True)
    info = []
    for row, (dstr, atm, pct) in enumerate(picks):
        s = S[byday[dstr]]
        fit_svi, _ = svi.fit_surface(s)
        iv = {}
        for m in ("dh", "dhj"):
            r = W[m][s.day]
            iv[m] = M.model_iv(M.price(m, calib.to_phys(m, ST.assemble(m, r["eta"], r["state"])), s), s)
        fitted = [e for e in range(len(s.T)) if np.isfinite(fit_svi[s.e == e]).all() and (s.e == e).sum() >= svi.MIN_QUOTES]
        want = []
        for tgt in (7, 30, 90):
            e = min(fitted, key=lambda k: abs(s.T[k] * 365 - tgt))
            if e not in want:
                want.append(e)
        while len(want) < 3:
            want.append(next(k for k in fitted if k not in want))
        want = sorted(want, key=lambda k: s.T[k])
        for col, e in enumerate(want):
            ax = axs[row, col]
            q = np.flatnonzero(s.e == e)
            q = q[np.argsort(s.K[q])]
            k = np.log(s.K[q] / s.F[e])
            ax.scatter(k, 100 * s.iv[q], s=26, color=PS.INK, label="market quotes", zorder=4)
            ax.plot(k, 100 * iv["dh"][q], color=PAL[2], lw=2.4, label="Double Heston (no jumps)")
            ax.plot(k, 100 * iv["dhj"][q], color=PAL[1], lw=2.6, label="Double Heston + jumps")
            ax.plot(k, 100 * fit_svi[q], color=PAL[0], lw=2.2, ls="--", label="SVI (5 parameters)")
            ax.set_title(f"{s.day}, {int(round(s.T[e] * 365))}-day expiry", fontsize=15)
            ax.tick_params(labelsize=12)
            if col == 0:
                ax.set_ylabel("Implied volatility (%)", fontsize=13)
            if row == 2:
                ax.set_xlabel("Forward log-moneyness ln(K/F)", fontsize=13)
            info.append({"day": str(s.day), "percentile_rank_of_market_30d_atm_vol": pct, "market_30d_atm_vol_pct": 100 * atm, "expiry_days": int(round(s.T[e] * 365)),
                         "quotes": int(q.size), "svi_rmse_vol_pts": float(np.sqrt(np.mean((100 * (fit_svi[q] - s.iv[q])) ** 2))),
                         "dh_rmse_vol_pts": float(np.sqrt(np.mean((100 * (iv['dh'][q] - s.iv[q])) ** 2))), "dhj_rmse_vol_pts": float(np.sqrt(np.mean((100 * (iv['dhj'][q] - s.iv[q])) ** 2)))})
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=4, fontsize=15)
    fig.suptitle("Market against model on three real test days (low, middle, high volatility)", fontsize=19, x=0.01, ha="left")
    paths = save_new(fig, FIG["smiles"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])
    tdir = root / cfg["paths"]["tables"]
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(info).to_csv(tdir / "smile_figure_days.csv", index=False)
    return paths


# ============================================================================================================== 08 parameter paths
def fig_paths(cfg, root, out_dir):
    from . import memory as MEM
    from .data import load_all
    S = load_all()
    p = root / cfg["paths"]["walk_results"] / "walk_dhj_main.pkl"
    if not p.exists():
        raise MissingSource(f"missing {p}")
    W = {r["day"]: r for r in pickle.load(open(p, "rb"))}
    byday = {s.day: i for i, s in enumerate(S)}
    days = sorted(d for d in W if d in byday)
    atm = np.array([100 * MEM._civ(S[byday[d]], 30 / 365, 0.0) for d in days])
    vol = np.array([100 * np.sqrt(W[d]["state"][0] + W[d]["state"][1]) for d in days])
    lam = np.array([W[d]["eta"][8] for d in days])
    fig, axs = _fig(cfg, 2, 1, sharex=True)
    axs[0].plot(days, atm, color=PS.INK2, lw=1.6, label="market: 30-day at-the-money implied volatility")
    axs[0].plot(days, vol, color=PAL[1], lw=1.6, label="calibrated diffusion volatility today, sqrt(v1 + v2)")
    axs[0].set_ylabel("Volatility (% a year)")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Calibrated volatility level and jump rate of the jump model, 2016-2026")
    axs[1].plot(days, lam, color=PAL[0], lw=1.6)
    axs[1].set_ylabel("Jumps a year")
    axs[1].set_xlabel("Date")
    return save_new(fig, FIG["paths"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])


# ============================================================================================================== 17 pricer validation
def _bucket_key(s: str) -> int:
    return int("".join(ch for ch in s.split("-")[0].split("d")[0] if ch.isdigit()) or 0)


def fig_pricer(cfg, root, out_dir):
    e = root / "results" / "pricer_validation" / "errors.csv"
    t = root / "results" / "pricer_validation" / "timing.json"
    for p in (e, t):
        if not p.exists():
            raise MissingSource(f"missing {p}")
    d = pd.read_csv(e)
    tim = json.loads(t.read_text())
    fig, axs = _fig(cfg, 1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    base = d[(d.model == "dh") & (d.domain == "all") & (d.moneyness_bucket == "all") & (d.maturity_bucket != "all")]
    buckets = sorted(base.maturity_bucket.unique(), key=_bucket_key)
    pos = {b: i for i, b in enumerate(buckets)}
    for src, comp, lab, col, mk in (("real_chains", "project_vs_cos", "real chains, project vs COS", PAL[1], "o"), ("synthetic_grid", "project_vs_cos", "synthetic grid, project vs COS", PAL[0], "s"),
                                    ("synthetic_grid", "lewis_vs_cos", "synthetic grid, Lewis vs COS", PAL[2], "^")):
        b = base[(base.source == src) & (base.comparison == comp)].sort_values("maturity_bucket", key=lambda s_: s_.map(_bucket_key))
        ax.plot([pos[x_] for x_ in b.maturity_bucket], b.max_abs_iv_err_pts, marker=mk, ms=9, color=col, lw=2.6, label=lab)
    ax.set_yscale("log")
    ax.axhline(0.01, color=PS.INK2, ls=":", lw=1.6)
    ax.text(0.01, 0.01, "  tolerance of the pricer tests: 0.01 vol point", transform=ax.get_yaxis_transform(), va="bottom", ha="left", color=PS.INK2, fontsize=14)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=30)
    ax.set_xlabel("Time to expiry")
    ax.set_ylabel("Largest implied-vol difference (vol points)")
    ax.set_title("Agreement with the independent COS pricer")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=13)
    ax2 = axs[1]
    names = [("project_pricer_dh2.models.price", "project\npricer"), ("cos_fixed_L14_no_width_check", "COS,\nfixed width"), ("cos_default_width_checked", "COS, width\nchecked"),
             ("lewis_adaptive_quad", "Lewis\nintegral")]
    x = np.arange(len(names))
    for j, (m_, col, lab) in enumerate((("dh", PAL[2], "diffusion"), ("dhj", PAL[1], "with jumps"))):
        vals = [tim[m_][k]["microseconds_per_option_best"] for k, _ in names]
        ax2.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=col, label=lab)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n for _, n in names], fontsize=13)
    ax2.set_ylabel("Microseconds per option (one core)")
    ax2.set_title("Cost per option on one real chain")
    ax2.legend(loc="upper left", fontsize=13)
    ax2.grid(axis="x", visible=False)
    return save_new(fig, FIG["pricer"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])


# ============================================================================================================== 01 summary panel
def fig_summary(cfg, root, out_dir):
    mv = _mvals(root)
    v = lambda k: _v(mv, k)
    PS.setup()
    w, h = cfg["figures"]["size_inches"]
    fig = plt.figure(figsize=(w, h), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axD, axE, axF = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    small = dict(labelsize=11)

    # A: forecast bars
    items = [("smile", "none", "yesterday's smile", "#a9a7a1"), ("smile_scaled", "none", "time-scaled smile", PS.INK2)]
    for r, c in (("heston_main", PAL[3]), ("dh_main", PAL[2]), ("dhj_main", PAL[1])):
        items.append((r, "carry_res", LABELS[f"{r}:carry_res"], c))
    ys = np.arange(len(items))[::-1]
    for yi, (r, var, lab, c) in zip(ys, items):
        k = f"forecast.{r}.{var}" if var != "none" else f"forecast.{r}"
        m_, lo, hi = v(k + ".test_mean"), v(k + ".test_mean_ci_lo"), v(k + ".test_mean_ci_hi")
        axA.barh(yi, m_, color=c, height=0.66)
        axA.errorbar(m_, yi, xerr=[[m_ - lo], [hi - m_]], fmt="none", ecolor=PS.INK, capsize=4, lw=1.6)
        axA.text(hi + 0.05, yi, f"{m_:.2f}", va="center", fontsize=12)
    axA.set_yticks(ys)
    axA.set_yticklabels([i[2] for i in items], fontsize=11)
    axA.set_xlabel("Next-day error, test period (vol points)", fontsize=12)
    axA.set_title("Forecast error", fontsize=15)
    axA.tick_params(axis="x", **small)
    axA.grid(axis="y", visible=False)

    # B: waterfall next-day
    steps = [("step0", "1.0 recipe"), ("step1", "+ clean data"), ("step2", "+ IV objective"), ("step3", "+ global starts"), ("step4", "+ walk-forward"), ("step5", "+ jumps"),
             ("step6", "+ yesterday's misfit")]
    vals = [v(f"attribution.{k}.oos_mean") for k, _ in steps]
    yb = np.arange(len(steps))[::-1]
    axB.barh(yb, vals, color=PAL[1], height=0.68)
    for yi, val in zip(yb, vals):
        axB.text(val + 0.06, yi, f"{val:.2f}", va="center", fontsize=11)
    ref = v("attribution.ref.time_scaled_smile.oos_mean")
    axB.axvline(ref, color=PS.INK, ls=":", lw=1.8)
    axB.text(ref + 0.05, 1.0, "time-scaled smile", transform=axB.get_xaxis_transform(), ha="left", va="top", fontsize=10, color=PS.INK2)
    axB.set_yticks(yb)
    axB.set_yticklabels([s for _, s in steps], fontsize=11)
    axB.set_xlabel("Next-day error (vol points)", fontsize=12)
    axB.set_xlim(0, max(vals) * 1.15)
    axB.set_title("From the 1.0 recipe to 2.0", fontsize=15)
    axB.tick_params(axis="x", **small)
    axB.grid(axis="y", visible=False)

    # C: SVI vs DH same-day
    regs = [("all", "all\nquotes"), ("core", "near the\nmoney"), ("wings", "wings")]
    xx = np.arange(len(regs))
    for j, (col, lab, c) in enumerate((("svi_rmse", "SVI, 5 params per expiry", PAL[0]), ("dhj_rmse", "Double Heston + jumps", PAL[1]), ("dh_rmse", "Double Heston", PAL[2]))):
        bv = [v(f"svi.pooled.{r}.{col}") for r, _ in regs]
        axC.bar(xx + (j - 1) * 0.27, bv, 0.26, color=c, label=lab)
    axC.set_xticks(xx)
    axC.set_xticklabels([l for _, l in regs], fontsize=11)
    axC.set_ylabel("Same-day error (vol points)", fontsize=12)
    axC.legend(fontsize=10, loc="upper left")
    axC.set_title("Fit quality against SVI", fontsize=15)
    axC.tick_params(axis="y", **small)

    # D: PINN-accelerated calibration
    ms = [("exact", "exact", PS.INK2), ("accel", "PINN-\naccelerated", PAL[1])]
    t = [v("pinn_calib.time.exact.mean_sec"), v("pinn_calib.time.accel.mean_sec")]
    axD.bar([0, 1], t, color=[c for _, _, c in ms], width=0.6)
    for i, (k, _, _) in enumerate(ms):
        axD.text(i, t[i] + 0.1, f"{t[i]:.2f} s\nRMSE {v(f'pinn_calib.method.{k}.mean_rmse'):.3f}", ha="center", fontsize=11)
    axD.set_xticks([0, 1])
    axD.set_xticklabels([n for _, n, _ in ms], fontsize=11)
    axD.set_ylabel("Seconds per daily calibration", fontsize=12)
    axD.set_ylim(0, max(t) * 1.3)
    axD.set_title(f"Faster calibration, same fit ({v('pinn_calib.speedup_wall_total.est'):.1f}x)", fontsize=15)
    axD.tick_params(axis="y", **small)

    # E: VIX
    sers = [("jump", "jump\nmodel", PAL[1]), ("diff", "diffusion\nDH", PAL[2]), ("atm", "surface\nATM", "#4a3aa7")]
    xx = np.arange(len(sers))
    for j, (per, lab) in enumerate((("all", "levels, 2016-2026"), ("test", "daily changes, test period"))):
        key = "corr_levels" if per == "all" else "corr_changes"
        bv = [v(f"vix.{s}.{per}.{key}") for s, _, _ in sers]
        axE.bar(xx + (j - 0.5) * 0.38, bv, 0.36, color=[c for _, _, c in sers], alpha=1.0 if j == 0 else 0.55, hatch=None if j == 0 else "//", edgecolor="white")
        for xi, val in zip(xx + (j - 0.5) * 0.38, bv):
            axE.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=10)
    axE.set_xticks(xx)
    axE.set_xticklabels([n for _, n, _ in sers], fontsize=11)
    axE.set_ylim(0, 1.12)
    axE.set_ylabel("Correlation with India VIX", fontsize=12)
    axE.set_xlabel("solid: levels, 2016-2026;  hatched: daily changes, test period", fontsize=10)
    axE.set_title("Link to India VIX", fontsize=15)
    axE.tick_params(axis="y", **small)

    # F: key numbers
    axF.axis("off")
    mc = "forecast.dhj_main.carry_res"
    lines = [("Jump + misfit vs best benchmark, test", f"{v(mc + '.test_mean'):.3f} vs {v('forecast.smile_scaled.test_mean'):.3f} vol pts"),
             ("Paired difference, 95% interval", f"{v(mc + '.diff_vs_best_benchmark'):+.3f} [{v(mc + '.diff_ci_lo'):+.3f}, {v(mc + '.diff_ci_hi'):+.3f}]"),
             ("Project pricer vs COS, real chains", f"{v('pricer.real_chains.dh.project_vs_cos.max_abs_iv_err_pts'):.1e} vol pts at most"),
             ("PINN surrogate vs exact pricer, mean error", f"{v('pinn.ablation.data_only.real.iv_err_mean'):.3f} vol pts"),
             ("Surrogate speed-up, 128 parameter sets", f"{v('pinn.speed.speedup_128_sets_cpu'):.0f}x on CPU, {v('pinn.speed.speedup_128_sets_mps'):.0f}x on GPU"),
             ("Held-out quotes: jump model vs diffusion", f"{v('crosssection.random.dhj.mean_iv_rmse'):.2f} vs {v('crosssection.random.dh.mean_iv_rmse'):.2f} vol pts")]
    axF.set_title("Key numbers", fontsize=15)
    for i, (a, b) in enumerate(lines):
        y = 0.90 - i * 0.165
        axF.text(0.0, y, a, fontsize=11, color=PS.INK2, transform=axF.transAxes, va="center")
        axF.text(0.0, y - 0.065, b, fontsize=14, color=PS.INK, transform=axF.transAxes, va="center", fontweight="bold")
    fig.suptitle("Double Heston 2.0 on NIFTY 50 options: summary of the results", fontsize=20, x=0.01, ha="left")
    return save_new(fig, FIG["summary"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])


# ============================================================================================================== 34 benchmark inside / outside yesterday's quoted range
def fig_wing(cfg, root, out_dir):
    mv = _mvals(root)
    v = lambda k: _v(mv, k)
    PS.setup()
    fig, ax = _fig(cfg)
    parts = [("all", "all quotes", None), ("inside", "inside yesterday's\nquoted range", "inside"), ("outside", "outside it", "outside")]
    bars = [("edge-value benchmark (time-scaled smile)", "model_vs_edge", "mean_b", "#a9a7a1"), ("linear-wing benchmark", "model_vs_linear", "mean_b", PS.INK2),
            ("jump model + yesterday's misfit", "model_vs_edge", "mean_a", PAL[1])]
    w = 0.26
    top = 0.0
    for j, (lab, cmp_, col, c) in enumerate(bars):
        vals = [v(f"benchmark_check.{p}.{cmp_}.{col}") for p, _, _ in parts]
        xs = np.arange(len(parts)) + (j - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=c, label=lab, edgecolor="white")
        for xi, val in zip(xs, vals):
            ax.text(xi, val + 0.04, f"{val:.3f}", ha="center", va="bottom", fontsize=14, color=PS.INK)
        top = max(top, max(vals))
    labels = []
    for p, lab, sh in parts:
        share = 1.0 if sh is None else v(f"benchmark_check.share_quotes_{sh}")
        k = f"benchmark_check.{p}.model_vs_linear"
        labels.append(f"{lab}\n{share:.1%} of quotes\nmodel minus linear-wing: {v(k + '.mean_difference'):+.3f}")
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, top * 1.2)
    ax.set_ylabel("Mean next-day error, test period (vol points)")
    ax.set_title("Where the model's lead comes from: quotes outside the strike range that yesterday quoted")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return save_new(fig, FIG["wing"], out_dir, cfg["figures"]["dpi"], cfg["figures"]["formats"])


# ============================================================================================================== driver and INDEX.md
NEW_BUILDERS = {"by_year": fig_by_year, "bars": fig_bars, "smiles": fig_smiles, "paths": fig_paths, "pricer": fig_pricer, "summary": fig_summary, "wing": fig_wing}


def write_index(out_dir: Path, built: dict[str, list[Path]]) -> Path:
    L = ["# Figures", "",
         "Every figure is a 300 dpi PNG and a PDF with the same stem, 16:9-friendly, on an opaque light panel so it reads on dark and light slide backgrounds. "
         "This file is written by `dh2/report_figures.py` (the registry `REGISTRY`), so it cannot drift from the files. Figures marked *new* are drawn by "
         "`dh2/report_figures.py` from the result files named in the last column and from `results/metrics.json`; figures marked *copy* are the work packages' own figures, copied "
         "and also written as PDF (the raster at 300 dpi).", "",
         "| # | File | Kind | What it shows | Source script | Data |", "|---|---|---|---|---|---|"]
    for f in REGISTRY:
        n = f.stem.split("_")[0]
        have = "png + pdf" if ((out_dir / f"{f.stem}.png").exists() and (out_dir / f"{f.stem}.pdf").exists()) else "not built"
        L.append(f"| {n} | `{f.stem}` ({have}) | {f.kind} | {f.caption} | `{f.source}` | {f.data} |")
    L += ["", "The three days of figure 05 are chosen by a fixed rule (the paired sample's days ranked by the market's own 30-day at-the-money volatility, taken at the configured "
          "percentiles), not by how well the models fit; they are listed with their errors in `results/tables/smile_figure_days.csv`.", ""]
    p = out_dir / "INDEX.md"
    p.write_text("\n".join(L))
    return p


def build_all(cfg: dict, root: Path = ROOT, only: list[str] | None = None) -> dict[str, list[Path]]:
    out_dir = root / cfg["paths"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    built: dict[str, list[Path]] = {}
    for f in REGISTRY:
        if only and f.key not in only:
            continue
        if f.kind == "new":
            built[f.key] = NEW_BUILDERS[f.key](cfg, root, out_dir)
        else:
            built[f.key] = copy_with_pdf(root / f.src, f.stem, out_dir)
    write_index(out_dir, built)
    return built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nifty.yaml")
    ap.add_argument("--only", default="", help="comma-separated figure keys")
    a = ap.parse_args()
    cfg = load_config(a.config)
    built = build_all(cfg, ROOT, [k for k in a.only.split(",") if k] or None)
    print(f"{len(built)} figures -> {ROOT / cfg['paths']['figures']}")


if __name__ == "__main__":
    main()
