"""Raw-SVI reference: the best smile fit a five-parameter-per-expiry curve can give, with no dynamics.

    python3 -m dh2.svi [--step 10] [--out results/svi]

Per day and per expiry, raw SVI (Gatheral) in total variance w = IV^2 T against forward log-moneyness k = ln(K/F):

    w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + s^2) )

fitted by least squares on the IMPLIED-VOL error in vol points (the metric every table here reports), to ALL the expiry's quotes (the
Double Heston fits use at most 10 quotes per expiry and share most parameters across expiries and days, so SVI has the
advantage in data and in freedom: 5 free numbers per expiry, 40+ per surface). Constraints (Gatheral-Jacquier style):
b >= 0, |rho| < 1, s > 0, a + b s sqrt(1 - rho^2) >= 0 (the minimum total variance is non-negative; enforced exactly by fitting
the minimum variance a' = a + b s sqrt(1-rho^2) >= 0 instead of a). The butterfly condition g(k) >= 0 is NOT imposed; it is checked
afterwards on a grid and violations are reported (g from Gatheral-Jacquier 2014; also Lee's wing bound b(1 + |rho|) <= 2).
Expiries with fewer than MIN_QUOTES quotes are not fitted (5 parameters through 5 or fewer points fit anything); both models are
compared on the quotes of the fitted expiries only, and the number left out is reported.

The comparison with Double Heston is in-sample, the same day's fit: the walk-forward 'state' fit of walk_dh_main / walk_dhj_main
(structural parameters from the trailing window, today's two variance levels fitted to today's surface), priced on every quote.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
MIN_QUOTES = 6
SPLIT = date(2022, 1, 1)
K_EDGES = [-0.10, -0.04, 0.04, 0.10]                    # forward log-moneyness buckets: (< -0.10), -0.10..-0.04, core, 0.04..0.10, (> 0.10)
K_LABELS = ["k<-0.10", "-0.10..-0.04", "core (abs k<=0.04)", "0.04..0.10", "k>0.10"]
M_EDGES = [7, 14, 30, 60]
M_LABELS = ["<=7d", "8-14d", "15-30d", "31-60d", ">60d"]


# ------------------------------------------------------------------ the curve
def svi_w(k, p) -> np.ndarray:
    """Raw SVI total variance; p = (a, b, rho, m, s)."""
    a, b, rho, m, s = p
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def butterfly_g(k, p) -> np.ndarray:
    """Gatheral-Jacquier's g(k): the risk-neutral density of the slice is proportional to g(k)/sqrt(w); g >= 0 everywhere = no butterfly arbitrage."""
    a, b, rho, m, s = p
    k = np.asarray(k, float)
    w = svi_w(k, p)
    r = np.sqrt((k - m) ** 2 + s**2)
    w1 = b * (rho + (k - m) / r)
    w2 = b * s**2 / r**3
    w = np.maximum(w, 1e-12)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2


def _to_abrms(q) -> np.ndarray:
    """Fit vector (a_min, b, rho, m, s) -> raw SVI (a, b, rho, m, s) with a = a_min - b s sqrt(1 - rho^2)."""
    amin, b, rho, m, s = q
    return np.array([amin - b * s * np.sqrt(1.0 - rho**2), b, rho, m, s])


def fit_expiry(k: np.ndarray, iv: np.ndarray, T: float) -> dict:
    """Constrained raw-SVI least-squares fit of one expiry to implied vols (error measured in vol points)."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    w = iv**2 * T
    lo = [0.0, 1e-4, -0.999, k.min() - 0.25, 1e-3]
    hi = [max(1.5, 2 * w.max()), 20.0, 0.999, k.max() + 0.25, 1.5]

    def res(q):
        return 100.0 * (np.sqrt(np.maximum(svi_w(k, _to_abrms(q)), 1e-12) / T) - iv)

    # starts: a grid over (rho, m, s); for each, (a, b) by linear least squares of w on the SVI shape, then the cheapest few are polished
    km = float(k[np.argmin(w)])
    cands = []
    for rho in (-0.8, -0.4, 0.0, 0.4, 0.8):
        for m in sorted({0.0, km, km - 0.05, km + 0.05}):
            for s in (0.02, 0.08, 0.25, 0.7):
                phi = rho * (k - m) + np.sqrt((k - m) ** 2 + s**2)
                A = np.column_stack([np.ones_like(k), phi])
                (a_, b_), *_ = np.linalg.lstsq(A, w, rcond=None)
                b_ = max(float(b_), 1e-4)
                amin = float(np.clip(a_ + b_ * s * np.sqrt(1 - rho**2), 0.0, hi[0]))
                q0 = np.clip([amin, b_, rho, m, s], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
                cands.append((float(np.sum(res(q0) ** 2)), q0))
    cands.sort(key=lambda c: c[0])
    best = None
    for _, q0 in cands[:5]:
        try:
            r = least_squares(res, q0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=300)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = _to_abrms(best.x)
    e = res(best.x)
    grid = np.linspace(min(k.min() - 0.1, -0.5), max(k.max() + 0.1, 0.5), 400)
    inside = (grid >= k.min()) & (grid <= k.max())
    g = butterfly_g(grid, p)
    return {"params": p, "rmse": float(np.sqrt(np.mean(e**2))), "iv_fit": np.sqrt(np.maximum(svi_w(k, p), 1e-12) / T), "g_min": float(g.min()),
            "g_min_in_range": float(g[inside].min()), "butterfly_violation": bool(g.min() < -1e-6),
            "butterfly_violation_in_range": bool(g[inside].min() < -1e-6), "lee_ok": bool(p[1] * (1 + abs(p[2])) <= 2.0 + 1e-9),
            "at_bound": [i for i, (v, l, h) in enumerate(zip(best.x, lo, hi)) if v <= l + 1e-6 * (h - l) or v >= h - 1e-6 * (h - l)]}


def fit_surface(S) -> tuple[np.ndarray, list[dict]]:
    """SVI implied vols for every quote of the surface (NaN where the expiry has too few quotes) and one record per fitted expiry."""
    fit = np.full(S.n, np.nan)
    recs = []
    for e in range(len(S.T)):
        q = np.flatnonzero(S.e == e)
        if q.size < MIN_QUOTES:
            continue
        k = np.log(S.K[q] / S.F[e])
        f = fit_expiry(k, S.iv[q], S.T[e])
        fit[q] = f["iv_fit"]
        recs.append({"day": S.day, "expiry": S.expiry[e], "days_to_expiry": int(round(S.T[e] * 365)), "n_quotes": int(q.size),
                     "a": f["params"][0], "b": f["params"][1], "rho": f["params"][2], "m": f["params"][3], "s": f["params"][4],
                     "rmse_volpts": f["rmse"], "g_min": f["g_min"], "g_min_in_quote_range": f["g_min_in_range"],
                     "butterfly_violation_anywhere": f["butterfly_violation"], "butterfly_violation_in_quote_range": f["butterfly_violation_in_range"],
                     "lee_wing_bound_ok": f["lee_ok"], "params_at_fit_bound": ",".join(map(str, f["at_bound"]))})
    return fit, recs


# ------------------------------------------------------------------ Double Heston same-day fits
def dh_errors(model: str, rec: dict, S) -> np.ndarray:
    """Model implied vol minus market, vol points, per quote, for the walk-forward same-day fit of day rec['day']."""
    from . import calib, models as M, struct as ST
    x = calib.to_phys(model, ST.assemble(model, np.asarray(rec["eta"]), np.asarray(rec["state"])))
    return M.iv_error(model, x, S)


def bucket_index(S) -> tuple[np.ndarray, np.ndarray]:
    k = np.log(S.K / S.F[S.e])
    days = np.round(S.T[S.e] * 365)
    return _kbucket(k), np.searchsorted(M_EDGES, days, side="left")


def _kbucket(k: np.ndarray) -> np.ndarray:
    return np.where(k < K_EDGES[0], 0, np.where(k < K_EDGES[1], 1, np.where(k <= K_EDGES[2], 2, np.where(k <= K_EDGES[3], 3, 4))))


def sample_days(S: list, step: int = 10) -> list[int]:
    """The paired sample: every `step`-th trading day of the test period (2022-01-01 onward), starting at its first day."""
    test = [i for i, s in enumerate(S) if s.day >= SPLIT]
    return test[::step]


def _one(S, W, i: int) -> dict:
    """One sample day: SVI fit of every expiry plus the Double Heston same-day errors."""
    s = S[i]
    fit, recs = fit_surface(s)
    d = {"i": i, "svi_iv": fit, "recs": recs}
    for m in ("dh", "dhj"):
        d[m] = dh_errors(m, W[m][s.day], s)
    return d


def run(step: int = 10, workers: int = 2):
    from joblib import Parallel, delayed
    from .data import load_all
    S = load_all()
    byday = {s.day: i for i, s in enumerate(S)}
    W = {m: {r["day"]: r for r in pickle.load(open(ROOT / "data" / "results" / f"walk_{m}_main.pkl", "rb"))} for m in ("dh", "dhj")}
    idx = [i for i in sample_days(S, step) if S[i].day in W["dh"] and S[i].day in W["dhj"]]

    out = Parallel(n_jobs=workers, verbose=2)(delayed(_one)(S, W, i) for i in idx)
    return S, W, idx, out


def tables(S, W, idx, out):
    rows_day, fits, quotes = [], [], []
    for d in out:
        s = S[d["i"]]
        ok = np.isfinite(d["svi_iv"])
        e_svi = 100.0 * (d["svi_iv"] - s.iv)
        kb, mb = bucket_index(s)
        k = np.log(s.K / s.F[s.e])
        quotes.append(pd.DataFrame({"day": s.day, "k": k, "days": np.round(s.T[s.e] * 365), "kb": kb, "mb": mb, "fitted_expiry": ok, "err_svi": e_svi,
                                    "err_dh": d["dh"], "err_dhj": d["dhj"]}))
        rms = lambda e, m: float(np.sqrt(np.mean(e[m] ** 2))) if m.sum() else np.nan
        allq = np.ones(s.n, bool)
        rows_day.append({"day": s.day, "quotes": s.n, "quotes_in_svi_expiries": int(ok.sum()), "expiries": len(s.T), "expiries_svi_fitted": len(d["recs"]),
                         "svi_rmse": rms(e_svi, ok), "dh_rmse_same_quotes": rms(d["dh"], ok), "dhj_rmse_same_quotes": rms(d["dhj"], ok),
                         "dh_rmse_all_quotes": rms(d["dh"], allq), "dhj_rmse_all_quotes": rms(d["dhj"], allq),
                         "dh_rmse_in_walk": W["dh"][s.day]["rmse_in"], "dhj_rmse_in_walk": W["dhj"][s.day]["rmse_in"]})
        fits += d["recs"]
    return pd.DataFrame(rows_day), pd.DataFrame(fits), pd.concat(quotes, ignore_index=True)


def bucket_table(q: pd.DataFrame) -> pd.DataFrame:
    q = q[q.fitted_expiry]
    rows = []
    for mb in range(5):
        for kb in range(5):
            g = q[(q.mb == mb) & (q.kb == kb)]
            r = {"maturity": M_LABELS[mb], "moneyness": K_LABELS[kb], "quotes": len(g)}
            for name, c in (("svi", "err_svi"), ("dhj", "err_dhj"), ("dh", "err_dh")):
                r[f"{name}_rmse"] = float(np.sqrt(np.mean(g[c] ** 2))) if len(g) else np.nan
                r[f"{name}_bias"] = float(np.mean(g[c])) if len(g) else np.nan
            rows.append(r)
    for label, g in (("ALL", q), ("core (abs k<=0.04)", q[q.kb == 2]), ("wings (abs k>0.10)", q[(q.kb == 0) | (q.kb == 4)])):
        r = {"maturity": label, "moneyness": "all" if label == "ALL" else label, "quotes": len(g)}
        for name, c in (("svi", "err_svi"), ("dhj", "err_dhj"), ("dh", "err_dh")):
            r[f"{name}_rmse"] = float(np.sqrt(np.mean(g[c] ** 2)))
            r[f"{name}_bias"] = float(np.mean(g[c]))
        rows.append(r)
    return pd.DataFrame(rows)


def figures(daily: pd.DataFrame, fits: pd.DataFrame, bt: pd.DataFrame, S, out_dir: Path, out):
    import matplotlib.pyplot as plt
    from . import plotstyle as PS
    PS.setup()
    C = {"svi": PS.ENT["svi"], "dhj": PS.ENT["jump"], "dh": PS.ENT["diff"]}
    # 1. per-day in-sample RMSE
    fig, ax = plt.subplots(figsize=(14, 7))
    x = pd.to_datetime(daily.day)
    for col, key, lab in (("svi_rmse", "svi", "SVI per expiry (5 parameters each)"), ("dhj_rmse_same_quotes", "dhj", "Double Heston + jumps (same-day fit)"),
                          ("dh_rmse_same_quotes", "dh", "Double Heston, no jumps (same-day fit)")):
        ax.plot(x, daily[col], color=C[key], lw=2.4, label=lab)
    ax.set_ylabel("In-sample IV RMSE (vol points)"); ax.set_xlabel("Date (every 10th test-period trading day)")
    ax.set_title("Same-day smile fit: SVI reference vs Double Heston")
    ax.legend(loc="upper left"); ax.set_xlim(x.iloc[0], x.iloc[-1])
    fig.tight_layout(); fig.savefig(out_dir / "svi_vs_dh_rmse_by_day.png", dpi=300); plt.close(fig)
    # 2. bucket heat maps (one sequential ramp, shared scale)
    b = bt[bt.maturity.isin(M_LABELS)]
    fig, axs = plt.subplots(1, 3, figsize=(22, 7.5), sharey=True)
    vmax = 4.0                                   # colour scale capped at 4 vol points so SVI and the jump model are distinguishable; the cells print the value
    cm = PS.seq_cmap()
    for ax, (name, title) in zip(axs, (("svi", "SVI"), ("dhj", "Double Heston + jumps"), ("dh", "Double Heston, no jumps"))):
        mat = b.pivot(index="maturity", columns="moneyness", values=f"{name}_rmse").reindex(index=M_LABELS, columns=K_LABELS)
        im = ax.imshow(np.minimum(mat.values, vmax), cmap=cm, vmin=0, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(5)); ax.set_xticklabels(["< -0.10", "-0.10\nto -0.04", "core\n+-0.04", "0.04\nto 0.10", "> 0.10"], fontsize=14)
        ax.set_yticks(range(5)); ax.set_yticklabels(M_LABELS)
        for i in range(5):
            for j in range(5):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=PS.cell_text_color(min(v, vmax) / vmax), fontsize=17)
        ax.set_title(title)
    axs[0].set_ylabel("Time to expiry")
    for ax in axs:
        ax.set_xlabel("Forward log-moneyness k = ln(K/F)")
    fig.colorbar(im, ax=axs, shrink=0.85, label="RMSE (vol points; colour capped at 4)", extend="max")
    fig.suptitle("In-sample IV RMSE by maturity and forward log-moneyness k (pooled over the sample days)", fontsize=21, x=0.02, ha="left")
    fig.savefig(out_dir / "svi_vs_dh_buckets.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # 3. one example surface: a mid-sample day, the expiry nearest 30 days
    d = out[len(out) // 2]
    s = S[d["i"]]
    e = int(np.argmin(np.abs(s.T * 365 - 30)))
    q = np.flatnonzero(s.e == e)
    q = q[np.argsort(s.K[q])]
    k = np.log(s.K[q] / s.F[e])
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.scatter(k, 100 * s.iv[q], s=55, color=PS.INK, label="market", zorder=3)
    ax.plot(k, 100 * d["svi_iv"][q], color=C["svi"], lw=2.8, label="SVI")
    ax.plot(k, 100 * (s.iv[q] + d["dhj"][q] / 100), color=C["dhj"], lw=2.8, label="Double Heston + jumps")
    ax.plot(k, 100 * (s.iv[q] + d["dh"][q] / 100), color=C["dh"], lw=2.8, label="Double Heston, no jumps")
    ax.set_xlabel("Forward log-moneyness ln(K/F)"); ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{s.day}, expiry in {int(round(s.T[e] * 365))} days")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "svi_example_smile.png", dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "svi"))
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from the saved CSVs (refits only the one example day)")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.figures_only:
        from .data import load_all
        S = load_all()
        W = {m: {r["day"]: r for r in pickle.load(open(ROOT / "data" / "results" / f"walk_{m}_main.pkl", "rb"))} for m in ("dh", "dhj")}
        days = pd.read_csv(out_dir / "sample_days.csv")["day"].tolist()
        byday = {s_.day.isoformat(): i for i, s_ in enumerate(S)}
        ex = _one(S, W, byday[days[len(days) // 2]])
        figures(pd.read_csv(out_dir / "svi_vs_dh_by_day.csv"), pd.read_csv(out_dir / "svi_fits.csv"), pd.read_csv(out_dir / "svi_vs_dh_by_bucket.csv"), S, out_dir, [ex])
        print("figures redrawn in", out_dir)
        return
    S, W, idx, out = run(a.step, a.workers)
    daily, fits, quotes = tables(S, W, idx, out)
    bt = bucket_table(quotes)
    quotes.round(5).to_csv(out_dir / "svi_vs_dh_quote_errors.csv.gz", index=False)
    daily.to_csv(out_dir / "svi_vs_dh_by_day.csv", index=False)
    fits.to_csv(out_dir / "svi_fits.csv", index=False)
    bt.to_csv(out_dir / "svi_vs_dh_by_bucket.csv", index=False)
    pd.DataFrame({"day": [S[i].day for i in idx]}).to_csv(out_dir / "sample_days.csv", index=False)
    figures(daily, fits, bt, S, out_dir, out)

    allq = bt[bt.maturity == "ALL"].iloc[0]
    chk = (daily.dh_rmse_all_quotes - daily.dh_rmse_in_walk).abs().max(), (daily.dhj_rmse_all_quotes - daily.dhj_rmse_in_walk).abs().max()
    n_excl = int(daily.quotes.sum() - daily.quotes_in_svi_expiries.sum())
    mean_ = lambda c: float(daily[c].mean())
    viol = fits.butterfly_violation_anywhere.mean(), fits.butterfly_violation_in_quote_range.mean(), 1 - fits.lee_wing_bound_ok.mean()
    md = f"""# SVI reference vs Double Heston (same-day, in-sample)

Generated by `python3 -m dh2.svi`. SVI (raw, per expiry, 5 free parameters each, no dynamics; module docstring of `dh2/svi.py`) is the
"best achievable smile fit" yardstick. The Double Heston fits are the walk-forward same-day fits (`walk_dh_main`, `walk_dhj_main`:
structural parameters from the trailing 40-day window, today's two variance levels), priced on every quote of the day.

Sample: {len(idx)} days, every {a.step}th trading day of the test period 2022-01-01 onward (`sample_days.csv`), {len(fits)} expiries fitted with SVI
(needs >= {MIN_QUOTES} quotes; {n_excl:,} of {int(daily.quotes.sum()):,} quotes sit in expiries too thin to fit and are excluded for ALL models in the same-quote columns).

## In-sample implied-vol RMSE (vol points, pooled over quotes)

| | SVI (5 params/expiry) | Double Heston + jumps | Double Heston, no jumps |
|---|---|---|---|
| all quotes of SVI-fitted expiries | {allq.svi_rmse:.3f} | {allq.dhj_rmse:.3f} | {allq.dh_rmse:.3f} |
| mean of the daily RMSEs, same quotes | {mean_('svi_rmse'):.3f} | {mean_('dhj_rmse_same_quotes'):.3f} | {mean_('dh_rmse_same_quotes'):.3f} |
| mean of the daily RMSEs, ALL quotes (DH as in the walk-forward) | - | {mean_('dhj_rmse_all_quotes'):.3f} | {mean_('dh_rmse_all_quotes'):.3f} |

Check: the recomputed Double Heston RMSE on all quotes reproduces the stored `rmse_in` of the walk-forward files (largest difference {chk[0]:.2e} diffusion, {chk[1]:.2e} jumps).

## By time to expiry and forward log-moneyness (RMSE, vol points; quotes of SVI-fitted expiries)

{_md_buckets(bt)}

By year (mean of the daily RMSEs, all quotes of SVI-fitted expiries; days in the sample):

{_md_years(daily)}

Quote-level errors (every quote of the sample days, vol points): `svi_vs_dh_quote_errors.csv.gz`.

Figures: `svi_vs_dh_rmse_by_day.png`, `svi_vs_dh_buckets.png`, `svi_example_smile.png`. Tables: `svi_vs_dh_by_day.csv`, `svi_vs_dh_by_bucket.csv`, `svi_fits.csv` (parameters per expiry).

## No-arbitrage check of the SVI fits (butterfly g(k) >= 0, not imposed)

- {100 * viol[0]:.1f}% of fitted expiries have g(k) < 0 somewhere on the grid ln(K/F) in [-0.5, 0.5] (wider if the quotes are), {100 * viol[1]:.1f}% inside their own quote range.
- {100 * viol[2]:.1f}% break Lee's wing bound b(1+|rho|) <= 2.
- Fits with a parameter on a fitting bound: {100 * (fits.params_at_fit_bound.fillna('').astype(str).str.len() > 0).mean():.1f}%.
Violations mean the unconstrained-in-g SVI error is a slightly optimistic reference (the arbitrage-free SVI could only fit worse).
"""
    (out_dir / "README.md").write_text(md)
    print(md)


def _md_years(daily: pd.DataFrame) -> str:
    d = daily.assign(year=[x.year for x in daily.day])
    s = "| year | days | SVI | Double Heston + jumps | Double Heston, no jumps |\n|---|---|---|---|---|\n"
    for y, g in d.groupby("year"):
        s += f"| {y} | {len(g)} | {g.svi_rmse.mean():.3f} | {g.dhj_rmse_same_quotes.mean():.3f} | {g.dh_rmse_same_quotes.mean():.3f} |\n"
    return s


def _md_buckets(bt: pd.DataFrame) -> str:
    s = "| time to expiry | moneyness | quotes | SVI | DH + jumps | DH no jumps | bias SVI | bias DH+J | bias DH |\n|---|---|---|---|---|---|---|---|---|\n"
    for r in bt.itertuples():
        s += f"| {r.maturity} | {r.moneyness} | {r.quotes:,} | {r.svi_rmse:.2f} | {r.dhj_rmse:.2f} | {r.dh_rmse:.2f} | {r.svi_bias:+.2f} | {r.dhj_bias:+.2f} | {r.dh_bias:+.2f} |\n"
    return s


if __name__ == "__main__":
    main()
