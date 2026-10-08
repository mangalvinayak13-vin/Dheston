"""Figures for the Double Heston 2.0 report -> outputs/figures/*.png   (python3 make_figures.py [--model dhj --tag main])

 1 error_by_year     next-day implied-vol error by year: the two model-free benchmarks, the pure model, the model + misfit carry
 2 paired_test       the test period's paired differences against the better benchmark, with 95% block-bootstrap intervals
 3 where_it_wins     pooled error by moneyness and by expiry length (test period)
 4 surface_fits      the calibrated model against market implied volatilities on three very different days
 5 parameter_paths   the calibrated volatility level and jump rate over ten years, against the market's own at-the-money volatility
"""
import argparse
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, evalpairs as EP, memory as MEM, models as M, struct as ST
from dh2.data import load_all
from tournament import boot

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "data" / "results", HERE / "outputs" / "figures"
SPLIT = date(2022, 1, 1)
INK, MUTED, GRID = "#1b1c1e", "#6a6e75", "#e4e2dc"
C = {"smile": "#b9b6ad", "scaled": "#6a6e75", "carry": "#4d7fb8", "res": "#c27a12", "mem": "#2c6e63"}
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "text.color": INK})


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(5, dy), textcoords="offset points", color=color, fontsize=8.8, va="center", fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dhj"); ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    S = load_all()
    hyb = pickle.load(open(RES / f"hyb_{a.model}_{a.tag}.pkl", "rb"))
    ts = sorted(hyb["smile"])
    pr = {"smile": {t: hyb["smile"][t].astype(float) for t in ts}, "scaled": {t: EP.pred_smile_scaled(S[t], S[t + 1]) for t in ts},
          "carry": {t: hyb["carry"][t].astype(float) for t in ts}, "res": {t: hyb["carry_res"][t].astype(float) for t in ts}}
    sc = {k: {t: EP.score(p[t], S[t + 1])["rmse"] for t in ts} for k, p in pr.items()}
    years = sorted({S[t].day.year for t in ts})
    names = {"smile": "yesterday's smile", "scaled": "yesterday's smile, time-scaled", "carry": "calibrated model", "res": "model + yesterday's misfit"}

    # 1 ------------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.axvspan(2021.5, 2026.9, color="#f1efe9", zorder=0)
    ax.text(2024.2, 3.55, "test period (never used to choose anything)", ha="center", color=MUTED, fontsize=8.3)
    ends = {}
    for k in ("smile", "scaled", "carry", "res"):
        y = [np.mean([sc[k][t] for t in ts if S[t].day.year == yr]) for yr in years]
        ax.plot(years, y, "-", color=C[k], lw=2.3 if k == "res" else 1.6, marker="o", ms=3.2)
        ends[k] = y[-1]
    last = None                                   # direct labels at the right edge, spread so they never overlap
    for k in sorted(ends, key=ends.get, reverse=True):
        ly = ends[k] if last is None else min(ends[k], last - 0.16)
        ax.annotate(names[k].replace("yesterday's smile, time-scaled", "smile, time-scaled"), (years[-1], ends[k]), xytext=(years[-1] + 0.35, ly),
                    textcoords="data", color=C[k], fontsize=8.8, va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C[k], lw=.7, shrinkA=0, shrinkB=1))
        last = ly
    ax.set_xlim(2016, 2029.3); ax.set_ylim(0, 3.8); ax.set_xticks(years)
    ax.set_ylabel("average next-day error (implied-vol points)"); ax.set_title("Next-day forecast error by year, NIFTY 50 options", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "error_by_year.png"); plt.close(fig)

    # 2 ------------------------------------------------------------------------------------------------------------------
    test = [t for t in ts if S[t].day >= SPLIT]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    rows = [("calibrated model", "carry"), ("model + yesterday's misfit", "res"), ("yesterday's smile (plain)", "smile")]
    for i, (nm, k) in enumerate(rows):
        m, lo, hi = boot([sc[k][t] - sc["scaled"][t] for t in test])
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=C[k], capsize=4, lw=2)
        ax.text(hi + 0.012, i, f"{m:+.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(-0.26, 0.34)
    ax.set_xlabel("difference in error vs the time-scaled smile (negative = better), with 95% interval")
    ax.set_title(f"Test period 2022-2026, {len(test):,} days", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "paired_test.png"); plt.close(fig)

    # 3 ------------------------------------------------------------------------------------------------------------------
    bd = json.loads((RES / f"breakdown_{a.model}_{a.tag}_test.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, key, ttl in ((axs[0], "moneyness |ln K/F|", "by how far the strike is from the money"), (axs[1], "expiry length", "by time to expiry")):
        labs = list(bd[key]); x = np.arange(len(labs)); w = 0.27
        for j, (k, nm) in enumerate((("smile_scaled", "scaled smile"), (f"{a.model} carry", "model"), (f"{a.model} carry_res", "model + misfit"))):
            ax.bar(x + (j - 1) * w, [bd[key][l][k] for l in labs], w, color=[C["scaled"], C["carry"], C["res"]][j], label=nm)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.2); ax.set_title(ttl, loc="left", fontsize=10, color=INK); ax.set_ylabel("error (vol points)")
    axs[0].legend(frameon=False, fontsize=8.2, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "where_it_wins.png"); plt.close(fig)

    # 4 ------------------------------------------------------------------------------------------------------------------
    W = {r["day"]: r for r in pickle.load(open(RES / f"walk_{a.model}_{a.tag}.pkl", "rb"))}
    picks = [(date(2020, 3, 23), "23 Mar 2020, the crash"), (date(2022, 6, 17), "17 Jun 2022"), (date(2025, 3, 20), "20 Mar 2025")]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=False)
    for ax, (d, ttl) in zip(axs, picks):
        d = min((x for x in W if x >= d), default=None)
        i = next(k for k, s in enumerate(S) if s.day == d)
        s, r = S[i], W[d]
        x = calib.to_phys(a.model, ST.assemble(a.model, r["eta"], r["state"]))
        iv = M.model_iv(M.price(a.model, x, s), s)
        order = np.argsort(s.T)[:6]
        for rank, e in enumerate((0, len(s.T) // 3, len(s.T) // 2)):
            q = np.flatnonzero(s.e == e); o = np.argsort(s.K[q]); mm = np.log(s.K[q][o] / s.F[e])
            col = ["#2c6e63", "#4d7fb8", "#c27a12"][rank]
            ax.plot(mm, 100 * s.iv[q][o], "o", ms=2.8, color=col, alpha=.7)
            ax.plot(mm, 100 * iv[q][o], "-", color=col, lw=1.6, label=f"{round(s.T[e] * 365)}-day expiry")
        ax.set_title(f"{ttl}\nsame-day error {r['rmse_in']:.2f} pts", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("log-moneyness ln(K/F)"); ax.legend(frameon=False, fontsize=7.8)
    axs[0].set_ylabel("implied volatility (%)")
    fig.suptitle("Dots: market. Lines: the calibrated jump model", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "surface_fits.png"); plt.close(fig)

    # 5 ------------------------------------------------------------------------------------------------------------------
    days = sorted(W)
    idx = [next(k for k, s in enumerate(S) if s.day == d) for d in days]
    atm = np.array([100 * MEM._civ(S[k], 30 / 365, 0.0) for k in idx])
    vol = np.array([100 * np.sqrt(W[d]["state"][0] + W[d]["state"][1]) for d in days])
    lam = np.array([W[d]["eta"][8] if a.model == "dhj" else np.nan for d in days])
    fig, axs = plt.subplots(2, 1, figsize=(8.4, 5.0), sharex=True)
    axs[0].plot(days, atm, color=C["scaled"], lw=1.1, label="market: 30-day at-the-money volatility")
    axs[0].plot(days, vol, color=C["res"], lw=1.1, label="calibrated: diffusion volatility today, sqrt(v1+v2)")
    axs[0].set_ylabel("%"); axs[0].legend(frameon=False, fontsize=8.5, loc="upper right"); axs[0].set_ylim(0, 60)
    axs[1].plot(days, lam, color=C["carry"], lw=1.1); axs[1].set_ylabel("jumps a year"); axs[1].set_title("calibrated jump rate (rare, large, crash-like jumps)", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "parameter_paths.png"); plt.close(fig)
    print("figures ->", FIG, [p.name for p in sorted(FIG.glob("*.png"))])


if __name__ == "__main__":
    main()
