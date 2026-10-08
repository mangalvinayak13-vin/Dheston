"""Figures (300 dpi) and the headline tables of the surrogate study, from the saved tables and checkpoints.  python3 -m dh2.pinn.validate figures"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import data as D
from . import pde as PDE
from . import train as TR
from .surrogate import Surrogate
from .validate import FIG, TAB, _mask, eval_sets, evaluate_model

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
C = {"hyb": "#0072B2", "dat": "#D55E00", "phy": "#009E73", "spec": "#CC79A7", "exact": "#222222"}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name, flush=True)


def headline_tables() -> dict:
    d = pd.read_csv(TAB / "heldout_metrics_all_runs.csv")
    out = {}
    rows = []
    for label, runs in (("hybrid (data + PDE + asymptotics), 3 seeds", ["hyb_s0", "hyb_s1", "hyb_s2"]), ("data-only (same architecture), 3 seeds", ["dat_s0", "dat_s1", "dat_s2"]),
                        ("physics-only (no labels), 1 seed", ["phy_s0"]), ("data-only with the spec gate g = sqrt(tau), 1 seed", ["spec_dat_s0"])):
        for st in ("real", "box", "mix", "test"):
            sub = d[(d.run.isin(runs)) & (d.set == st)]
            r = {"variant": label, "held-out set": st, "n_models": len(sub)}
            for k in ("mean", "median", "p95", "p99", "max", "bias"):
                r[f"IV err {k} (vol pts) avg over seeds"] = sub[k].mean()
                if len(sub) > 1:
                    r[f"IV err {k} sd over seeds"] = sub[k].std(ddof=1)
            if st in ("real", "box"):
                for k in ("pde_res_median", "pde_res_p95", "pde_res_rms"):
                    r[k] = sub[k].mean()
            if st == "real":
                r["price MAE (rupees)"] = sub["price_mae_rupees"].mean()
                r["price max abs err (rupees)"] = sub["price_max_rupees"].mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "ablation_summary.csv", index=False)
    out["ablation"] = t
    # data efficiency
    de = []
    for st in ("real", "box"):
        for mode, pre in (("data-only", "dat"), ("hybrid", "hyb")):
            for n in (1000, 10000, 100000, 1000000):
                runs = [f"de_{pre}_{n}"] if n < 1000000 else [f"{pre}_s{s}" for s in range(3)]
                sub = d[(d.run.isin(runs)) & (d.set == st)]
                de.append({"held-out set": st, "variant": mode, "supervised samples": n, "models": len(sub), "IV err mean": sub["mean"].mean(), "IV err p95": sub["p95"].mean(),
                           "IV err max": sub["max"].mean(), "sd of mean over seeds": sub["mean"].std(ddof=1) if len(sub) > 1 else np.nan})
    de = pd.DataFrame(de)
    de.to_csv(TAB / "data_efficiency.csv", index=False)
    out["de"] = de
    return out


def fig_ablation(t):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labs = ["hybrid", "data-only", "physics-only", "data-only,\nspec gate"]
    cols = [C["hyb"], C["dat"], C["phy"], C["spec"]]
    for ax, st, ttl in zip(axes, ("real", "box"), ("Real held-out chains (65,465 quotes)", "Held-out parameter sets from the box")):
        sub = t[t["held-out set"] == st].reset_index(drop=True)
        x = np.arange(len(sub))
        mean = sub["IV err mean (vol pts) avg over seeds"].values; p95 = sub["IV err p95 (vol pts) avg over seeds"].values
        sd = sub["IV err mean sd over seeds"].fillna(0).values
        ax.bar(x - 0.2, mean, 0.4, yerr=sd, color=cols, capsize=4, label="mean |IV error|")
        ax.bar(x + 0.2, p95, 0.4, color=cols, alpha=0.45, hatch="//", label="95th percentile")
        for xi, m, p in zip(x, mean, p95):
            ax.text(xi - 0.2, m * 1.15, f"{m:.3f}" if m < 1 else f"{m:.2f}", ha="center", fontsize=12)
            ax.text(xi + 0.2, p * 1.15, f"{p:.2f}", ha="center", fontsize=12)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=13)
        ax.set_ylabel("implied-vol error (vol points)"); ax.set_title(ttl, fontsize=15)
        ax.grid(axis="y", alpha=0.3, which="both")
    axes[0].legend(loc="upper left")
    fig.suptitle("Ablation: held-out IV error vs the exact pricer (bars = seed average, error bars = 1 sd)", fontsize=17, y=1.02)
    _save(fig, "ablation_bars.png")


def fig_data_efficiency(de):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, st, ttl in zip(axes, ("real", "box"), ("Real held-out chains", "Held-out box parameter sets")):
        for mode, key in (("data-only", "dat"), ("hybrid", "hyb")):
            s = de[(de["held-out set"] == st) & (de.variant == mode)]
            ax.errorbar(s["supervised samples"], s["IV err mean"], yerr=s["sd of mean over seeds"].fillna(0), marker="o", ms=9, lw=2.5, capsize=4, color=C[key], label=f"{mode} (mean error)")
            ax.plot(s["supervised samples"], s["IV err p95"], marker="s", ms=7, lw=1.8, ls="--", color=C[key], alpha=0.6, label=f"{mode} (95th pct)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("supervised samples (exact prices)"); ax.set_ylabel("implied-vol error (vol points)"); ax.set_title(ttl)
    axes[0].legend()
    fig.suptitle("Data efficiency: error vs number of labelled samples (1e6 = mean of 3 seeds)", fontsize=17, y=1.02)
    _save(fig, "data_efficiency.png")


def fig_speed():
    s = pd.read_csv(TAB / "speed.csv")
    keep = s[s["what"].str.contains("one chain|128")]
    keep = keep[~keep["what"].str.contains("Greeks")]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    labels = [f"{w.replace(', prices + Black IV', ' + IV')}\n[{dv.split(' ')[0]}]" for w, dv in zip(keep["what"], keep["device"])]
    cols = [C["exact"] if "exact" in w else C["dat"] for w in keep["what"]]
    y = np.arange(len(keep))
    ax.barh(y, keep["ms_per_quote"] * 1e3, color=cols)
    for yi, v, ms in zip(y, keep["ms_per_quote"] * 1e3, keep["median_ms"]):
        ax.text(v * 1.05, yi, f"{v:.1f} us/quote  ({ms:.1f} ms)", va="center", fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("microseconds per priced quote (log scale)"); ax.set_title("Speed: surrogate vs the exact Fourier pricer, one NSE chain (283 quotes, 8 expiries)")
    ax.grid(axis="x", alpha=0.3, which="both"); ax.set_xlim(right=ax.get_xlim()[1] * 8)
    _save(fig, "speed.png")


def fig_maps(cfg, dom):
    sets = eval_sets(cfg, dom)
    g = sets["grid"]
    xs = np.unique(g["x"]); taus = np.unique(g["tau"])
    ix = np.searchsorted(xs, g["x"]); it = np.searchsorted(taus, g["tau"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    for ax, (name, ttl) in zip(axes, (("dat_s0", "data-only"), ("hyb_s0", "hybrid"))):
        ev = evaluate_model(TR.CKPT / f"{name}.pt", sets, cfg["device"], names=("grid",))["grid"]
        e = np.abs(ev["err"]); m = ev["mask"]
        H = np.full((len(taus), len(xs)), np.nan); cnt = np.zeros_like(H); S = np.zeros_like(H)
        np.add.at(S, (it[m], ix[m]), e[m]); np.add.at(cnt, (it[m], ix[m]), 1)
        H = np.where(cnt >= 10, S / np.maximum(cnt, 1), np.nan)
        im = ax.pcolormesh(xs, taus * 365, np.ma.masked_invalid(H), cmap="viridis", shading="nearest", vmin=0, vmax=np.nanpercentile(H, 98))
        ax.set_yscale("log"); ax.set_xlabel("x = ln(F/K)  (negative: OTM calls; positive: OTM puts)"); ax.set_title(f"{ttl}: mean |IV error| (vol pts)")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("maturity (days)")
    fig.suptitle("Held-out error maps (120 held-out parameter sets; white = exact OTM price below Rs 1-equivalent)", fontsize=17, y=1.02)
    _save(fig, "error_heatmaps.png")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2), sharey=True)
    for ax, (name, ttl) in zip(axes, (("dat_s0", "data-only"), ("hyb_s0", "hybrid"), ("phy_s0", "physics-only"))):
        sur = Surrogate.load(TR.CKPT / f"{name}.pt", cfg["device"])
        n = len(g["x"]); res = np.empty(n)
        for s0 in range(0, n, 2000):
            sl = slice(s0, s0 + 2000)
            b = {k: torch.tensor(g[k][sl], dtype=torch.float32, device=sur.device) for k in ("x", "v1", "v2", "tau", "theta")}
            with torch.enable_grad():
                nr, _, _ = PDE.normalised_residual(sur.model, b["x"], b["v1"], b["v2"], b["tau"], b["theta"], 1e-4, create_graph=False)
            res[sl] = np.abs(nr.detach().cpu().numpy())
        S = np.zeros((len(taus), len(xs))); cnt = np.zeros_like(S)
        lr = np.log10(np.maximum(res, 1e-6))
        np.add.at(S, (it, ix), lr); np.add.at(cnt, (it, ix), 1)
        H = S / cnt
        im = ax.pcolormesh(xs, taus * 365, H, cmap="magma", shading="nearest", vmin=-1.5, vmax=1.5)
        ax.set_yscale("log"); ax.set_xlabel("x = ln(F/K)"); ax.set_title(f"{ttl}")
        fig.colorbar(im, ax=ax, label="log10 |tau R / vega|  (vol points)")
    axes[0].set_ylabel("maturity (days)")
    fig.suptitle("PDE residual of the network (geometric mean over 120 held-out parameter sets), in volatility points", fontsize=17, y=1.03)
    _save(fig, "pde_residual_maps.png")


def fig_greeks(cfg, dom):
    z = np.load(TAB.parent / "greeks_points.npz")
    sets = eval_sets(cfg, dom)
    real = sets["real"]; idx = z["idx"]
    x, tau, v1, v2, th = real["x"][idx], real["tau"][idx], real["v1"][idx], real["v2"][idx], real["theta"][idx]
    ex = z["exact"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for row, (name, key) in enumerate((("dat_s0", "dat"), ("hyb_s0", "hyb"))):
        g = Surrogate.load(TR.CKPT / f"{name}.pt", "cpu").greeks(np.ones(len(idx)), np.exp(-x), tau, np.ones(len(idx)), th, v1, v2, S=1.0)
        for ax, (k, j, ttl) in zip(axes[row], (("delta_S", 0, "delta"), ("gamma_S", 1, "gamma"), ("dC_dv1", 2, "dC/dv1"))):
            ax.scatter(ex[:, j], g[k], s=18, color=C[key], alpha=0.65)
            lo, hi = min(ex[:, j].min(), g[k].min()), max(ex[:, j].max(), g[k].max())
            ax.plot([lo, hi], [lo, hi], color="k", lw=1.2)
            ax.set_xlabel(f"{ttl}: finite differences of the exact pricer"); ax.set_ylabel(f"{ttl}: autograd of the surrogate")
            ax.set_title(f"{'data-only' if key == 'dat' else 'hybrid'}: {ttl}"); ax.grid(alpha=0.3)
    fig.suptitle("Greeks: surrogate (autograd) vs exact (finite differences), 300+ held-out quotes", fontsize=18, y=1.0)
    _save(fig, "greeks_scatter.png")


def fig_smile(cfg, dom):
    sets = eval_sets(cfg, dom)
    real = sets["real"]
    days = np.unique(real["day"])
    d = days[len(days) // 2]
    m = (real["day"] == d) & _mask(real)
    sd = {k: Surrogate.load(TR.CKPT / f"{k}.pt", "cpu") for k in ("dat_s0", "hyb_s0")}
    exps = np.unique(real["exp"][m])
    pick = exps[[0, len(exps) // 2, -1]] if len(exps) >= 3 else exps
    fig, axes = plt.subplots(2, len(pick), figsize=(6.2 * len(pick), 9), sharex="col", gridspec_kw={"height_ratios": [3, 1.4]})
    for j, e in enumerate(pick):
        mm = m & (real["exp"] == e)
        o = np.argsort(real["x"][mm])
        x = real["x"][mm][o]; tau = real["tau"][mm][o]
        axes[0, j].plot(-x, 100 * real["iv"][mm][o], color=C["exact"], lw=3, label="exact pricer")
        for k, key, ls in (("dat_s0", "dat", "--"), ("hyb_s0", "hyb", ":")):
            iv = sd[k].implied_vol(x, real["v1"][mm][0], real["v2"][mm][0], tau, real["theta"][mm][0])
            axes[0, j].plot(-x, 100 * iv, color=C[key], lw=2.5, ls=ls, label=f"surrogate ({'data-only' if key == 'dat' else 'hybrid'})")
            axes[1, j].plot(-x, 100 * (iv - real["iv"][mm][o]), color=C[key], lw=2, label=k)
        axes[0, j].set_title(f"{tau[0] * 365:.0f} days"); axes[1, j].axhline(0, color="k", lw=1)
        axes[1, j].set_xlabel("ln(K/F)"); axes[0, j].grid(alpha=0.3); axes[1, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("implied vol (%)"); axes[1, 0].set_ylabel("surrogate - exact (vol pts)"); axes[0, 0].legend()
    fig.suptitle(f"One held-out day ({dom['ho_days'][d]}): model smile, exact vs surrogate", fontsize=17, y=0.99)
    _save(fig, "smile_example.png")


def fig_training():
    fig, ax = plt.subplots(figsize=(11, 6))
    for pre, key, lab in (("dat", "dat", "data-only"), ("hyb", "hyb", "hybrid")):
        for s in range(3):
            f = TAB / f"train_{pre}_s{s}.csv"
            if f.exists():
                h = pd.read_csv(f)
                ax.plot(h["step"], h["val_iv_mean"], color=C[key], alpha=0.8, lw=2, label=lab if s == 0 else None)
    ax.set_yscale("log"); ax.set_xlabel("Adam step"); ax.set_ylabel("validation mean |IV error| (vol points)"); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_title("Training curves (validation set of the training distribution)")
    _save(fig, "training_curves.png")


def stage_figures(cfg, dom):
    t = headline_tables()
    print(t["ablation"].round(4).to_string())
    print(t["de"].round(4).to_string())
    fig_ablation(t["ablation"]); fig_data_efficiency(t["de"]); fig_speed(); fig_training()
    fig_maps(cfg, dom); fig_greeks(cfg, dom); fig_smile(cfg, dom)
