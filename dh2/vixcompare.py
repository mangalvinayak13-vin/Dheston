"""Model-implied 30-day volatility of the Double Heston fits against India VIX.

    python3 -m dh2.vixcompare [--out results/vix]          (needs data/vix/india_vix.csv: python3 fetch_vix.py)

For every day the calibrated parameters of the walk-forward fits (walk_dh_main = diffusion Double Heston, walk_dhj_main = with
Merton jumps: structural parameters eta of the trailing window + that day's two variance levels v_j) give the model's risk-neutral
expected average variance over the next T = 30/365 years. For one Heston factor with mean-reversion speed kappa_j and long-run
level theta_j the expected integrated variance is  theta_j T + (v_j - theta_j)(1 - exp(-kappa_j T))/kappa_j, so

    sigma30^2 = (1/T) sum_j [ theta_j T + (v_j - theta_j) (1 - exp(-kappa_j T)) / kappa_j ]          (diffusion part)

Jump models add the variance the jumps contribute, per year, to that sum (three versions are kept, the first is the headline):
  qv         lam (mu_j^2 + delta_j^2)                      the quadratic variation of the jumps: the second moment of the log jump,
                                                           what a realised-variance measure would see
  logcontract 2 lam (exp(mu_j + delta_j^2/2) - 1 - mu_j)   what a model-free (log-contract) variance such as VIX's formula
                                                           assigns to a Merton jump: close to qv for small jumps
  diffusion  0                                             the jump model's diffusion variance alone
The surface's own ATM 30-day implied vol comes from the day's quotes: each expiry's implied vol at forward moneyness k = 0
(robust local quadratic fit over |k| <= 0.05), then total variance interpolated linearly in time to 30 days (NaN when 30 days lies outside
the listed expiries).

India VIX is NOT built from this: it is a model-free variance-swap style measure over a whole strip of NIFTY options (near and next
month, interpolated to 30 days, risk-free rate from T-bills), quoted in percent. A level offset between it and a model-implied
number is expected; the questions are co-movement (changes), the size and stability of the offset, and who leads.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[1]
T30 = 30.0 / 365.0
SPLIT = date(2022, 1, 1)
SERIES = {                                   # column -> (label, kind)
    "dhj_qv": "Jump model + jump variance",
    "dh": "Diffusion Double Heston",
    "atm30": "Surface ATM 30-day IV",
    "dhj_logc": "Jump model + log-contract jump variance",
    "dhj_diff": "Jump model, diffusion part only",
}


def expected_variance(kappa, theta, v, T=T30) -> float:
    """Risk-neutral expected average variance over [0, T] of one square-root factor."""
    return theta + (v - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T)


def implied_vol30(model: str, eta, state, T: float = T30) -> dict:
    """30-day model-implied volatilities (decimals) from one day's walk-forward record."""
    eta, state = np.asarray(eta, float), np.asarray(state, float)
    k1, th1, _, _, dk, th2 = eta[0], eta[1], eta[2], eta[3], eta[4], eta[5]
    var = expected_variance(k1, th1, state[0], T) + expected_variance(k1 + dk, th2, state[1], T)
    out = {"diff": float(np.sqrt(var))}
    if model == "dhj":
        lam, mu, dl = eta[8], eta[9], eta[10]
        out["qv"] = float(np.sqrt(var + lam * (mu**2 + dl**2)))
        out["logc"] = float(np.sqrt(var + 2.0 * lam * (np.exp(mu + 0.5 * dl**2) - 1.0 - mu)))
    return out


def atm_iv_expiry(S, e: int, half: float = 0.05, min_n: int = 5) -> float:
    """The expiry's implied vol at forward moneyness k = 0: a robust local quadratic fit of IV on k over the quotes with |k| <= `half`
    (outliers beyond 3 MAD, and at least 0.5 vol point, are dropped and the fit repeated), because one stale at-the-money print would
    otherwise move a plain interpolation by many vol points (seen on 2022-05-12: 37.9% against 22% neighbours). Fewer than `min_n`
    quotes in the window, or quotes on one side only: the expiry is skipped (NaN)."""
    q = np.flatnonzero(S.e == e)
    if q.size < min_n:
        return float("nan")
    k = np.log(S.K[q] / S.F[e])
    m = np.abs(k) <= half
    if m.sum() < min_n or not ((k[m] < 0).any() and (k[m] >= 0).any()):
        return float("nan")
    kk, vv = k[m], S.iv[q][m]
    w = np.ones(len(kk), bool)
    for _ in range(3):
        c = np.polyfit(kk[w], vv[w], 2 if w.sum() >= 6 else 1)
        res = vv - np.polyval(c, kk)
        r_in = res[w]
        mad = 1.4826 * np.median(np.abs(r_in - np.median(r_in)))
        new = np.abs(res) <= max(3 * mad, 0.005)
        if new.sum() < 4 or (new == w).all():
            break
        w = new
    c = np.polyfit(kk[w], vv[w], 2 if w.sum() >= 6 else 1)
    return float(np.polyval(c, 0.0))


def atm30(S, T: float = T30) -> float:
    pts = [(S.T[e], atm_iv_expiry(S, e)) for e in range(len(S.T))]
    pts = [(t, v) for t, v in pts if np.isfinite(v)]
    if len(pts) < 2:
        return float("nan")
    t = np.array([p[0] for p in pts]); w = np.array([p[1] ** 2 * p[0] for p in pts])
    if T < t.min() or T > t.max():
        return float("nan")
    return float(np.sqrt(np.interp(T, t, w) / T))


# ------------------------------------------------------------------ statistics
def ols_hac(y, x, lags: int) -> dict:
    """y = a + b x by OLS with Newey-West (Bartlett) standard errors."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    X = np.column_stack([np.ones_like(x), x])
    n = len(y)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu / n
    for j in range(1, lags + 1):
        G = Xu[j:].T @ Xu[:-j] / n
        S += (1 - j / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X / n)
    cov = XtXi @ S @ XtXi / n
    r2 = 1.0 - float(u @ u) / float(((y - y.mean()) ** 2).sum())
    return {"n": n, "intercept": float(beta[0]), "slope": float(beta[1]), "se_intercept": float(np.sqrt(cov[0, 0])),
            "se_slope": float(np.sqrt(cov[1, 1])), "r2": r2, "resid_sd": float(u.std(ddof=2))}


def pair_stats(vix: pd.Series, x: pd.Series, label: str, period: str, lags: int = 21) -> dict:
    d = pd.concat([vix, x], axis=1).dropna()
    v, m = d.iloc[:, 0].values, d.iloc[:, 1].values
    if len(d) < 30:
        return {"series": label, "period": period, "n_levels": len(d)}
    reg = ols_hac(v, m, lags)
    dv, dm = np.diff(v), np.diff(m)
    # changes only across consecutive rows that are consecutive trading days in the merged frame (rows with gaps are rare)
    regc = ols_hac(dv, dm, 5)
    return {"series": label, "period": period, "n_levels": len(d),
            "mean_model": float(m.mean()), "mean_vix": float(v.mean()), "mean_gap_model_minus_vix": float((m - v).mean()),
            "median_ratio_model_to_vix": float(np.median(m / v)), "rmse_gap": float(np.sqrt(np.mean((m - v) ** 2))),
            "corr_levels": float(np.corrcoef(v, m)[0, 1]), "spearman_levels": float(sst.spearmanr(v, m)[0]),
            "corr_changes": float(np.corrcoef(dv, dm)[0, 1]), "spearman_changes": float(sst.spearmanr(dv, dm)[0]),
            "corr_log_changes": float(np.corrcoef(np.diff(np.log(v)), np.diff(np.log(m)))[0, 1]),
            "reg_level_intercept": reg["intercept"], "reg_level_slope": reg["slope"], "reg_level_slope_se_hac": reg["se_slope"],
            "reg_level_r2": reg["r2"], "reg_level_resid_sd": reg["resid_sd"],
            "reg_change_slope": regc["slope"], "reg_change_slope_se_hac": regc["se_slope"], "reg_change_r2": regc["r2"]}


def lead_lag(vix: pd.Series, x: pd.Series, maxlag: int = 5) -> pd.DataFrame:
    """corr(change in x at t, change in VIX at t+L) for L = -maxlag..maxlag; L > 0: the model's change comes first."""
    dv, dx = vix.diff(), x.diff()
    rows = []
    for L in range(-maxlag, maxlag + 1):
        d = pd.concat([dx, dv.shift(-L)], axis=1).dropna()
        rows.append({"lag_days": L, "corr": float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]), "n": len(d), "band95": 1.96 / np.sqrt(len(d))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ build and report
def build(vix_path: Path) -> pd.DataFrame:
    from .data import load_all
    S = load_all()
    W = {m: {r["day"]: r for r in pickle.load(open(ROOT / "data" / "results" / f"walk_{m}_main.pkl", "rb"))} for m in ("dh", "dhj")}
    rows = []
    for s in S:
        row = {"date": s.day, "atm30": 100 * atm30(s), "saturated": False}
        for m_ in ("dh", "dhj"):
            if s.day in W[m_] and (np.asarray(W[m_][s.day]["state"]) >= 0.2999).any():
                row["saturated"] = True      # a variance level sits on its upper bound (0.30, about 55% vol): the model cannot go higher
        if s.day in W["dh"]:
            row["dh"] = 100 * implied_vol30("dh", W["dh"][s.day]["eta"], W["dh"][s.day]["state"])["diff"]
        if s.day in W["dhj"]:
            r = implied_vol30("dhj", W["dhj"][s.day]["eta"], W["dhj"][s.day]["state"])
            row.update({"dhj_qv": 100 * r["qv"], "dhj_logc": 100 * r["logc"], "dhj_diff": 100 * r["diff"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    v = pd.read_csv(vix_path)
    v["date"] = pd.to_datetime(v["date"]).dt.date
    df = df.merge(v[["date", "close", "nifty50_close"]].rename(columns={"close": "vix"}), on="date", how="left")
    return df


def figures(df: pd.DataFrame, stats_all: pd.DataFrame, ll: dict, out: Path):
    import matplotlib.pyplot as plt
    from . import plotstyle as PS
    PS.setup()
    P = {"vix": PS.ENT["vix"], "dhj_qv": PS.ENT["jump"], "dh": PS.ENT["diff"], "atm30": PS.ENT["atm"]}      # one colour per entity in every figure
    d = df.dropna(subset=["vix"]).copy()
    d["dt"] = pd.to_datetime(d["date"])
    # 1. time series, two panels, at most three lines each
    fig, axs = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    ax = axs[0]
    for col, lab, c in (("vix", "India VIX", P["vix"]), ("dhj_qv", "Jump model, 30-day implied", P["dhj_qv"]), ("dh", "Diffusion Double Heston, 30-day implied", P["dh"])):
        ax.plot(d.dt, d[col], color=c, lw=1.8 if col != "vix" else 2.2, label=lab)
    ax.set_ylabel("Annualised volatility (%)"); ax.set_title("India VIX and the model-implied 30-day volatility"); ax.legend(loc="upper right")
    ax = axs[1]
    for col, lab, c in (("vix", "India VIX", P["vix"]), ("atm30", "Surface ATM 30-day implied vol", P["atm30"])):
        ax.plot(d.dt, d[col], color=c, lw=1.8 if col != "vix" else 2.2, label=lab)
    ax.set_ylabel("Annualised volatility (%)"); ax.set_title("India VIX and the surface's own ATM 30-day implied vol"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig_timeseries.png", dpi=300); plt.close(fig)
    # 2/3. scatter, levels and daily changes
    for kind, fname, title in (("level", "fig_scatter_levels.png", "Levels"), ("change", "fig_scatter_changes.png", "Daily changes")):
        fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
        for ax, col, lab in zip(axs, ("dhj_qv", "dh", "atm30"), ("Jump model", "Diffusion Double Heston", "Surface ATM 30-day")):
            dd = d[["vix", col]].dropna()
            if kind == "change":
                dd = dd.diff().dropna()
            ax.scatter(dd[col], dd["vix"], s=14, color=P[col], alpha=0.35, linewidths=0)
            lo, hi = float(min(dd.min())), float(max(dd.max()))
            ax.plot([lo, hi], [lo, hi], color=PS.INK2, lw=1.4, ls=(0, (4, 3)))
            b = ols_hac(dd["vix"], dd[col], 21 if kind == "level" else 5)
            xs = np.array([lo, hi])
            ax.plot(xs, b["intercept"] + b["slope"] * xs, color=PS.INK, lw=2.2)
            ax.set_xlabel(f"{lab} (vol points)"); ax.set_ylabel("India VIX (vol points)" if col == "dhj_qv" else "")
            ax.set_title(f"{lab}: R² {b['r2']:.2f}, slope {b['slope']:.2f}")
            ax.text(0.04, 0.95, "dashed: 45 degrees\nsolid dark: regression", transform=ax.transAxes, va="top", fontsize=14, color=PS.INK2)
        fig.suptitle(f"India VIX against model-implied 30-day volatility: {title.lower()}", fontsize=21, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(out / fname, dpi=300); plt.close(fig)
    # 4. lead-lag
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for (col, lab), c, off in zip((("dhj_qv", "Jump model"), ("dh", "Diffusion Double Heston"), ("atm30", "Surface ATM 30-day")), (P["dhj_qv"], P["dh"], P["atm30"]), (-0.2, 0.0, 0.2)):
        t = ll[col]
        ax.bar(t.lag_days + off, t["corr"], width=0.2, color=c, label=lab)
    band = float(ll["dhj_qv"].band95.iloc[0])
    ax.axhline(band, color=PS.INK2, lw=1.2, ls=(0, (4, 3)), label="approx. 95% band if unrelated"); ax.axhline(-band, color=PS.INK2, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("Lag L in days (L > 0: the model's change comes first)")
    ax.set_ylabel("Correlation of daily changes"); ax.set_title("Lead-lag: model change at t vs VIX change at t+L")
    ax.set_xticks(range(-5, 6)); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig_leadlag.png", dpi=300); plt.close(fig)
    # 5. gap and rolling correlation
    fig, axs = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    ax = axs[0]
    ax.plot(d.dt, d.dhj_qv - d.vix, color=P["dhj_qv"], lw=1.6, label="Jump model minus VIX")
    ax.plot(d.dt, d.atm30 - d.vix, color=P["atm30"], lw=1.6, label="Surface ATM 30-day minus VIX")
    ax.axhline(0, color=PS.INK2, lw=1.0)
    ax.set_ylabel("Gap (vol points)"); ax.set_title("Level gap to India VIX"); ax.legend(loc="upper right")
    ax = axs[1]
    dd = d.set_index("dt")[["vix", "dhj_qv", "atm30"]].diff()
    for col, lab, c in (("dhj_qv", "Jump model", P["dhj_qv"]), ("atm30", "Surface ATM 30-day", P["atm30"])):
        ax.plot(dd.index, dd["vix"].rolling(120).corr(dd[col]), color=c, lw=2.0, label=lab)
    ax.set_ylim(-0.2, 1.0); ax.set_ylabel("Correlation of daily changes"); ax.set_title("Rolling 120-day correlation of daily changes with VIX"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_gap_and_rolling_corr.png", dpi=300); plt.close(fig)


def _tab(df: pd.DataFrame, cols: list[str], names: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    s = "| " + " | ".join(names) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c, "{:.2f}")
            if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool):
                cells.append(f.format(int(v)) if f.endswith("d}") else f.format(v))
            else:
                cells.append(str(v))
        s += "| " + " | ".join(cells) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "vix"))
    ap.add_argument("--vix", default=str(ROOT / "data" / "vix" / "india_vix.csv"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = build(Path(a.vix))
    df.to_csv(out / "vix_vs_model_daily.csv", index=False)
    d = df.dropna(subset=["vix"]).reset_index(drop=True)
    periods = {"2016-2026 (all)": d, "2016-2021 (development)": d[d.date < SPLIT], "2022-2026 (test)": d[d.date >= SPLIT]}
    rows = []
    for pname, g in periods.items():
        for col, lab in SERIES.items():
            rows.append(pair_stats(g["vix"], g[col], lab, pname))
    n_sat = int(d.saturated.sum())
    ns = d[~d.saturated.astype(bool)]
    for col in ("dhj_qv", "dh"):
        rows.append(pair_stats(ns["vix"], ns[col], SERIES[col], f"2016-2026 without the {n_sat} bound-saturated days"))
    st = pd.DataFrame(rows)
    st.to_csv(out / "vix_vs_model_stats.csv", index=False)
    sat_days = d[d.saturated.astype(bool)]
    # by year
    ry = []
    for y, g in d.groupby([x.year for x in d.date]):
        for col in ("dhj_qv", "dh", "atm30"):
            r = pair_stats(g["vix"], g[col], SERIES[col], str(y), lags=10)
            ry.append(r)
    sy = pd.DataFrame(ry)
    sy.to_csv(out / "vix_vs_model_by_year.csv", index=False)
    ll = {col: lead_lag(d.set_index("date")["vix"], d.set_index("date")[col]) for col in ("dhj_qv", "dh", "atm30")}
    pd.concat([t.assign(series=SERIES[c]) for c, t in ll.items()]).to_csv(out / "vix_leadlag.csv", index=False)
    figures(d, st, ll, out)

    # README
    al = st[st.period == "2016-2026 (all)"].set_index("series")
    te = st[st.period == "2022-2026 (test)"].set_index("series")
    main_ = SERIES["dhj_qv"]
    tbl_cols = ["series", "period", "n_levels", "mean_model", "mean_vix", "mean_gap_model_minus_vix", "rmse_gap", "corr_levels", "corr_changes", "spearman_changes", "reg_level_slope", "reg_level_intercept", "reg_level_r2", "reg_change_slope", "reg_change_r2"]
    tbl_names = ["series", "period", "days", "mean model", "mean VIX", "model - VIX", "RMSE of gap", "corr levels", "corr changes", "Spearman changes", "level slope", "level intercept", "level R2", "change slope", "change R2"]
    best_lag = {SERIES[c]: int(t.loc[t["corr"].idxmax(), "lag_days"]) for c, t in ll.items()}
    ll_tab = pd.DataFrame({"lag": ll["dhj_qv"].lag_days, **{SERIES[c]: t["corr"].values for c, t in ll.items()}})
    n_atm_nan = int(d.atm30.isna().sum())
    src_file = Path(a.vix).parent / "SOURCE.txt"
    SRC = ("`data/vix/india_vix.csv`, fetched by `fetch_vix.py` from NSE's own daily index-close archive "
           "(`nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv`, the 'India VIX' row of each trading day: open, high, low, close; "
           "the 'Nifty 50' close is kept too). Yahoo's `^INDIAVIX` chart endpoint answered HTTP 429 (Too Many Requests) on every attempt from this network, and "
           "niftyindices.com's historical-data endpoint returned an error, so the NSE archive was used; it is the exchange's own record. "
           "Fetch record:\n\n```\n" + (src_file.read_text().strip() if src_file.exists() else "SOURCE.txt not found") + "\n```")
    txt = f"""# Model-implied 30-day volatility against India VIX

Generated by `python3 -m dh2.vixcompare` (VIX download: `fetch_vix.py`; source and dates in `data/vix/SOURCE.txt`). {len(d):,} trading days with both a surface
and a VIX print ({d.date.min()} to {d.date.max()}). Definitions are in the docstring of `dh2/vixcompare.py`; short version:

- **Model-implied 30-day vol** = sqrt of the model's risk-neutral expected average variance over the next 30/365 years,
  sum over the two factors of theta T + (v - theta)(1 - exp(-kappa T))/kappa, from each day's calibrated parameters (walk-forward fits `walk_dh_main` and `walk_dhj_main`).
  The jump model adds lam (mu_j^2 + delta_j^2) to the variance per year (the jumps' quadratic variation; the headline "{main_}"). Two variants are also reported:
  the exact log-contract jump variance 2 lam (exp(mu_j + delta_j^2/2) - 1 - mu_j), and the jump model's diffusion part alone.
- **Surface ATM 30-day IV**: the day's own quotes, ATM implied vol per expiry (k = 0, robust local quadratic fit of the quotes with |k| <= 0.05, stale outliers dropped), total variance interpolated linearly in time to 30 days
  ({n_atm_nan} days have no expiry on one side of 30 days and are NaN).
- **India VIX** is a different, model-free construction (variance-swap style strip of NIFTY options, near and next month interpolated to 30 days), in percent. Its level is not expected to equal any of these.

## Headline

- Levels, all years: correlation of India VIX with the jump model's 30-day vol {al.loc[main_, 'corr_levels']:.3f}, with the diffusion Double Heston {al.loc[SERIES['dh'], 'corr_levels']:.3f}, with the surface's ATM 30-day IV {al.loc[SERIES['atm30'], 'corr_levels']:.3f}.
- Daily changes, all years: {al.loc[main_, 'corr_changes']:.3f} (jump model), {al.loc[SERIES['dh'], 'corr_changes']:.3f} (diffusion), {al.loc[SERIES['atm30'], 'corr_changes']:.3f} (surface ATM). Test years only: {te.loc[main_, 'corr_changes']:.3f}, {te.loc[SERIES['dh'], 'corr_changes']:.3f}, {te.loc[SERIES['atm30'], 'corr_changes']:.3f}.
- Average level gap (model minus VIX): jump model {al.loc[main_, 'mean_gap_model_minus_vix']:+.2f} vol points, diffusion {al.loc[SERIES['dh'], 'mean_gap_model_minus_vix']:+.2f}, surface ATM {al.loc[SERIES['atm30'], 'mean_gap_model_minus_vix']:+.2f}.
- Regression VIX = a + b x model (Newey-West standard errors, 21 lags) for the jump model: b = {al.loc[main_, 'reg_level_slope']:.2f} (s.e. {al.loc[main_, 'reg_level_slope_se_hac']:.2f}), a = {al.loc[main_, 'reg_level_intercept']:.2f}, R2 = {al.loc[main_, 'reg_level_r2']:.2f}.

## Statistics

{_tab(st, tbl_cols, tbl_names, {"n_levels": "{:d}"})}
"corr changes" is the correlation of day-over-day changes in vol points; "level slope/intercept" are from VIX regressed on the model number; "change slope" is the same for daily changes.
By year: `vix_vs_model_by_year.csv`.

## Lead-lag (correlation of the model's daily change at t with VIX's daily change at t+L)

L > 0 means the model's move comes first. Lag with the highest correlation: {", ".join(f"{k}: {v}" for k, v in best_lag.items())}.
Approximate 95% band under no relation: +-{float(ll['dhj_qv'].band95.iloc[0]):.3f}.

{_tab(ll_tab, list(ll_tab.columns), ["lag (days)"] + list(ll_tab.columns[1:]), {"lag": "{:d}"})}

## Figures (300 dpi)

`fig_timeseries.png`, `fig_scatter_levels.png`, `fig_scatter_changes.png`, `fig_leadlag.png`, `fig_gap_and_rolling_corr.png`. Daily data: `vix_vs_model_daily.csv`.

## Where the model cannot follow: bound-saturated days

On {n_sat} days ({sat_days.date.min()} to {sat_days.date.max()}, the March-April 2020 crash) at least one variance level of the walk-forward fit sits on its upper bound (0.30, about 55% volatility),
so the model-implied 30-day vol is capped (jump model {sat_days.dhj_qv.min():.1f}-{sat_days.dhj_qv.max():.1f}%, diffusion {sat_days.dh.min():.1f}-{sat_days.dh.max():.1f}% on those days, while VIX on them ranged {sat_days.vix.min():.1f}-{sat_days.vix.max():.1f}) and stops responding. The bottom rows of the statistics table repeat the jump-model and diffusion rows without those days.

## Source of the VIX data

{SRC}

## Reading it honestly

- The comparison is with a model-free number on the same day. The model-implied vols are fitted to the same-day option prices (filtered to traded OTM quotes within |ln K/F| <= 0.22 and priced at least Rs 1), so a high co-movement says the fit tracks the market's volatility level, not that the model forecasts VIX.
- The model-implied number is a risk-neutral expectation of a mean-reverting variance; VIX's formula weights the whole strip including far wings that the filtered quotes do not cover, so a level gap of a few vol points is built in.
- Jump contribution: the three jump-variance choices give the same ordering; see `vix_vs_model_stats.csv` for all.
"""
    (out / "README.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
