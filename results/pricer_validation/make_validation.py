"""Builds results/pricer_validation/{errors.csv, worst_cases.csv, timing.json}: the project pricer (pricer/models.py, midpoint Gil-Pelaez) against
the three independent pricers (dh2/cos.py COS, dh2/lewis.py Lewis single integral, dh2/mc.py Andersen-QE Monte Carlo).

    python3 results/pricer_validation/make_validation.py [--quick]

Three sources of test points, all reported separately in errors.csv:
  synthetic_grid  about 27 realistic parameter sets (every 122nd calibrated NIFTY day of outputs/parameters_dhj_main.csv, plus the site default and the
                  most extreme calibrated sets) x 16 maturities (1 day .. 1 year) x 21 strikes (|ln K/F| <= 0.25), out-of-the-money options only
                  (calls above the forward, puts below: the quotes the calibration uses), diffusion model (dh) and with Merton jumps (dhj).
  real_chains     the real option chains of ~60 trading days (every 40th day of dh2.data.load_all()), each priced with that day's calibrated
                  parameters, through the very function the calibration uses (dh2.models.price), strikes and expiries exactly as in the data.
  monte_carlo     the 10 parameter sets of the slow test, 7 strikes each, 200k antithetic pairs (400k QE paths).
Single process, one BLAS thread.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from dh2 import cos, data, lewis, mc, models as M          # noqa: E402
import test_pricer_cross as TP                             # noqa: E402  (shared parameter sets)

OUT = Path(__file__).resolve().parent
P = M.P
S0, R0, Q0 = 25000.0, 0.065, 0.0                           # nominal NIFTY-like level / financing rate for the synthetic grid
DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365]
X_GRID = np.linspace(-0.25, 0.25, 21)                      # x = ln(F/K); negative = out-of-the-money call
TICK = 0.05                                                # rupees: below one NSE tick relative / IV errors are not meaningful

MAT_EDGES = [(0, 2, "1-2d"), (3, 7, "3-7d"), (8, 14, "8-14d"), (15, 30, "15-30d"), (31, 60, "31-60d"), (61, 120, "61-120d"), (121, 400, "121-365d"), (401, 3000, ">400d")]
MON_EDGES = [(0.0, 0.02, "0-0.02"), (0.02, 0.05, "0.02-0.05"), (0.05, 0.10, "0.05-0.10"), (0.10, 0.15, "0.10-0.15"), (0.15, 0.2201, "0.15-0.22"), (0.2201, 1.0, "0.22-0.25")]


def bucket(v, edges, closed_low=True):
    for lo, hi, name in edges:
        if lo <= v <= hi if closed_low else lo < v <= hi:
            return name
    return "other"


def project_prices(model, p13, spot, K, T, r, q, is_call):
    """The project pricer exactly as dh2.models.price calls it: one call-price vector per expiry, puts by parity."""
    F = spot * np.exp((r - q) * T)
    if model == "dh":
        call = np.atleast_1d(P.double_heston_price(spot, K, T, r, *p13[:10], kind="call", q=q))
    else:
        v1, k1, t1, x1, r1, v2, k2, t2, x2, r2, lam, mu, dl = p13[:13]
        log_cf = lambda u: (P._log_cf_factor(u, v1, k1, t1, x1, r1, T) + P._log_cf_factor(u, v2, k2, t2, x2, r2, T) + M._jump_log_cf(u, T, lam, mu, dl))
        call = np.atleast_1d(P._price_from_log_cf(log_cf, spot, K, T, r, q, "call"))
    return np.where(is_call, call, call - np.exp(-r * T) * (F - K))


def cos_prices(p13, model, F, K, T, DF, is_call):
    jump = None if model == "dh" else p13[10:13]
    out = np.empty(K.size)
    if is_call.any():
        out[is_call] = cos.cos_price(p13[:10], F, K[is_call], T, DF, "call", jump=jump)
    if (~is_call).any():
        out[~is_call] = cos.cos_price(p13[:10], F, K[~is_call], T, DF, "put", jump=jump)
    return out


def lewis_prices(p13, model, F, K, T, DF, is_call):
    jump = None if model == "dh" else p13[10:13]
    out = np.empty(K.size)
    if is_call.any():
        out[is_call] = lewis.lewis_price(p13[:10], F, K[is_call], T, DF, "call", jump=jump)
    if (~is_call).any():
        out[~is_call] = lewis.lewis_price(p13[:10], F, K[~is_call], T, DF, "put", jump=jump)
    return out


def ivs(price, F, K, T, r, is_call):
    with np.errstate(all="ignore"):
        return data.black_iv(np.maximum(price, 1e-12), F, K, T, r, is_call)


def synthetic_sets(df):
    idx = list(np.arange(0, len(df), 122))
    for col, fn in (("xi_fast", np.argmax), ("v0_slow", np.argmax), ("kappa_slow", np.argmin), ("jump_rate", np.argmax), ("xi_slow", np.argmax), ("theta_fast", np.argmin)):
        idx.append(int(fn(df[col].values)))
    idx = sorted(set(idx))
    cols = ["v0_slow", "kappa_slow", "theta_slow", "xi_slow", "rho_slow", "v0_fast", "kappa_fast", "theta_fast", "xi_fast", "rho_fast", "jump_rate", "jump_mean", "jump_sd"]
    sets = [("site_default", np.array(TP.DEFAULT + [0.10, -0.06, 0.20]))]
    sets += [(str(df.date.iloc[i])[:10], df[cols].iloc[i].values.astype(float)) for i in idx]
    return sets


def run_synthetic(df, quick):
    rows = []
    sets = synthetic_sets(df)
    if quick:
        sets = sets[::6]
    t0 = time.time()
    for si, (name, p13) in enumerate(sets):
        for model in ("dh", "dhj"):
            for d in DAYS:
                T = d / 365.0
                F = S0 * np.exp((R0 - Q0) * T)
                DF = np.exp(-R0 * T)
                K = F * np.exp(-X_GRID)
                is_call = X_GRID <= 0
                pc = cos_prices(p13, model, F, K, T, DF, is_call)
                pl = lewis_prices(p13, model, F, K, T, DF, is_call)
                pp = project_prices(model, p13, S0, K, T, R0, Q0, is_call)
                iv_c, iv_l, iv_p = (ivs(v, F, K, T, R0, is_call) for v in (pc, pl, pp))
                for j in range(K.size):
                    rows.append(dict(source="synthetic_grid", model=model, set=name, days=d, x=X_GRID[j], K=K[j], F=F, is_call=bool(is_call[j]),
                                     cos=pc[j], lewis=pl[j], project=pp[j], iv_cos=iv_c[j], iv_lewis=iv_l[j], iv_project=iv_p[j], T=T))
        print(f"  synthetic set {si + 1}/{len(sets)} {name}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def run_real(df, quick):
    surfaces = data.load_all()
    step = 40 if not quick else 150
    picks = surfaces[::step]
    cols = ["v0_slow", "kappa_slow", "theta_slow", "xi_slow", "rho_slow", "v0_fast", "kappa_fast", "theta_fast", "xi_fast", "rho_fast", "jump_rate", "jump_mean", "jump_sd"]
    dfi = df.assign(date=pd.to_datetime(df.date).dt.date).set_index("date")
    rows = []
    t0 = time.time()
    for si, S in enumerate(picks):
        if S.day not in dfi.index:
            continue
        p13 = dfi.loc[S.day, cols].values.astype(float)
        for model in ("dh", "dhj"):
            proj = M.price(model, p13 if model == "dhj" else p13[:10], S)
            for e in range(len(S.T)):
                q = np.flatnonzero(S.e == e)
                if not q.size:
                    continue
                T, F = S.T[e], S.F[e]
                DF = np.exp(-S.r * T)
                K, ic = S.K[q], S.is_call[q]
                pc = cos_prices(p13, model, F, K, T, DF, ic)
                pl = lewis_prices(p13, model, F, K, T, DF, ic)
                iv_c, iv_l, iv_p = (ivs(v, F, K, T, S.r, ic) for v in (pc, pl, proj[q]))
                for j in range(q.size):
                    rows.append(dict(source="real_chains", model=model, set=str(S.day), days=int(round(T * 365)), x=float(np.log(F / K[j])), K=K[j], F=F,
                                     is_call=bool(ic[j]), cos=pc[j], lewis=pl[j], project=proj[q[j]], iv_cos=iv_c[j], iv_lewis=iv_l[j], iv_project=iv_p[j], T=T))
        if (si + 1) % 10 == 0:
            print(f"  real day {si + 1}/{len(picks)} {S.day}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), surfaces


def run_mc(quick):
    rows = []
    n_pairs = 200_000 if not quick else 40_000      # (the slow test uses 100k pairs)
    xs = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
    t0 = time.time()
    for name, p10, T in TP.MC_CASES:
        F, DF = 100.0, np.exp(-0.06 * T)
        K = F * np.exp(-xs)
        is_call = xs <= 0
        pc = cos_prices(np.array(p10 + [0, 0, 0.1]), "dh", F, K, T, DF, is_call)
        pm = np.empty(K.size); se = np.empty(K.size)
        for kind, sel in (("call", is_call), ("put", ~is_call)):
            m, s, _ = mc.mc_price(p10, F, K[sel], T, DF, kind, n_pairs=n_pairs, seed=TP.MC_SEED)
            pm[sel], se[sel] = m, s
        spot = F * np.exp(-0.06 * T)                                    # r = 0.06, q = 0  =>  F = spot e^{rT}
        pp = project_prices("dh", np.array(p10 + [0, 0, 0.1]), spot, K, T, 0.06, 0.0, is_call)
        iv_c, iv_m, iv_p = (ivs(v, F, K, T, 0.06, is_call) for v in (pc, pm, pp))
        for j in range(K.size):
            rows.append(dict(source="monte_carlo", model="dh", set=name, days=int(round(T * 365)), x=xs[j], K=K[j], F=F, is_call=bool(is_call[j]), cos=pc[j], mc=pm[j], mc_se=se[j],
                             project=pp[j], iv_cos=iv_c[j], iv_mc=iv_m[j], iv_project=iv_p[j], T=T))
        print(f"  mc case {name} T={T:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ aggregation
def _stats(d, a, b, iv_a, iv_b, tag):
    """Error of pricer a against reference b over the rows of d."""
    ref = d[b].values
    err = np.abs(d[a].values - ref)
    tradeable = ref >= TICK
    rel = np.where(tradeable, err / np.where(tradeable, ref, 1.0), np.nan)
    ive = 100.0 * np.abs(d[iv_a].values - d[iv_b].values)
    ok = np.isfinite(ive)
    iv_t = np.where(tradeable & ok, ive, np.nan)
    out = dict(comparison=tag, n=len(d), n_tradeable=int(tradeable.sum()),
               max_abs_price_err_rs=err.max(), mean_abs_price_err_rs=err.mean(),
               max_rel_price_err=np.nanmax(rel) if tradeable.any() else np.nan, mean_rel_price_err=np.nanmean(rel) if tradeable.any() else np.nan,
               max_abs_iv_err_pts=np.nanmax(iv_t) if np.isfinite(iv_t).any() else np.nan, mean_abs_iv_err_pts=np.nanmean(iv_t) if np.isfinite(iv_t).any() else np.nan,
               max_abs_iv_err_pts_any_price=np.nanmax(np.where(ok, ive, np.nan)) if ok.any() else np.nan,
               n_iv_not_invertible=int((~ok).sum()))
    if "mc_se" in d and tag in ("mc_vs_cos", "project_vs_mc"):
        z = (d[a].values - ref) / d["mc_se"].values
        out.update(max_abs_z=np.nanmax(np.abs(z)), mean_abs_z=np.nanmean(np.abs(z)))
    return out


def aggregate(det):
    det = det.copy()
    det["maturity_bucket"] = [bucket(d, MAT_EDGES) for d in det.days]
    det["moneyness_bucket"] = [bucket(abs(x), MON_EDGES) for x in det.x]
    det["in_calibration_domain"] = (det.days >= 3) & (det.x.abs() <= 0.22) & (det.cos >= 1.0 if det.source.iloc[0] != "monte_carlo" else True)
    pairs = {"project_vs_cos": ("project", "cos", "iv_project", "iv_cos"), "lewis_vs_cos": ("lewis", "cos", "iv_lewis", "iv_cos"),
             "mc_vs_cos": ("mc", "cos", "iv_mc", "iv_cos"), "project_vs_mc": ("project", "mc", "iv_project", "iv_mc")}
    out = []
    for (src, model), g in det.groupby(["source", "model"]):
        for tag, (a, b, ia, ib) in pairs.items():
            if a not in g or b not in g or g[a].isna().all() or g[b].isna().all():
                continue
            views = [("all", "all", None)]
            views += [(m, "all", g.maturity_bucket == m) for m in g.maturity_bucket.unique()]
            views += [("all", mo, g.moneyness_bucket == mo) for mo in g.moneyness_bucket.unique()]
            views += [(m, mo, (g.maturity_bucket == m) & (g.moneyness_bucket == mo)) for m in g.maturity_bucket.unique() for mo in g.moneyness_bucket.unique()]
            for dom in ("all", "in_domain"):
                gd = g if dom == "all" else g[g.in_calibration_domain]
                for mb, mo, mask in views:
                    sub = gd if mask is None else gd[mask.reindex(gd.index, fill_value=False)]
                    if len(sub):
                        out.append(dict(source=src, model=model, maturity_bucket=mb, moneyness_bucket=mo, domain=dom, **_stats(sub, a, b, ia, ib, tag)))
    cols = ["source", "model", "comparison", "domain", "maturity_bucket", "moneyness_bucket"]
    res = pd.DataFrame(out)
    return res[cols + [c for c in res.columns if c not in cols]].sort_values(cols).reset_index(drop=True), det


# ------------------------------------------------------------------------------------------------ timing
def best_of(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def run_timing(surfaces, df):
    cols = ["v0_slow", "kappa_slow", "theta_slow", "xi_slow", "rho_slow", "v0_fast", "kappa_fast", "theta_fast", "xi_fast", "rho_fast", "jump_rate", "jump_mean", "jump_sd"]
    dfi = df.assign(date=pd.to_datetime(df.date).dt.date).set_index("date")
    cand = [s for s in surfaces if s.day in dfi.index and s.day.year >= 2025]
    S = max(cand[-60:], key=lambda s: s.n)                                  # the busiest of the last 60 days
    p13 = dfi.loc[S.day, cols].values.astype(float)
    res = dict(day=str(S.day), n_quotes=int(S.n), n_expiries=int(len(S.T)), maturities_days=[int(round(t * 365)) for t in S.T],
               params_source="outputs/parameters_dhj_main.csv for that day", threads="1 (OMP/OPENBLAS/VECLIB limited to 1), single process",
               cpu="Apple M-series laptop (other jobs running concurrently: timings are indicative)")

    def chain_cos(model, **kw):                                             # default: the truncation width is checked (priced at 14 L and 21 L)
        jump = None if model == "dh" else p13[10:13]
        for e in range(len(S.T)):
            q = np.flatnonzero(S.e == e)
            for kind, sel in (("call", S.is_call[q]), ("put", ~S.is_call[q])):
                if sel.any():
                    cos.cos_price(p13[:10], S.F[e], S.K[q][sel], S.T[e], np.exp(-S.r * S.T[e]), kind, jump=jump, **kw)

    def chain_lewis(model):
        for e in range(len(S.T)):
            q = np.flatnonzero(S.e == e)
            lewis_prices(p13, model, S.F[e], S.K[q], S.T[e], np.exp(-S.r * S.T[e]), S.is_call[q])

    for model in ("dh", "dhj"):
        x = p13 if model == "dhj" else p13[:10]
        t = {}
        t["project_pricer_dh2.models.price"] = best_of(lambda: M.price(model, x, S))
        t["cos_default_width_checked"] = best_of(lambda: chain_cos(model))
        t["cos_fixed_L14_no_width_check"] = best_of(lambda: chain_cos(model, L=14.0))
        t["lewis_adaptive_quad"] = best_of(lambda: chain_lewis(model), n=1)
        res[model] = {k: dict(seconds_best=v[0], seconds_median=v[1], microseconds_per_option_best=1e6 * v[0] / S.n) for k, v in t.items()}
        res[model]["cos_time_ratio_project_over_cos"] = t["project_pricer_dh2.models.price"][0] / t["cos_default_width_checked"][0]
        res[model]["lewis_time_ratio_over_cos"] = t["lewis_adaptive_quad"][0] / t["cos_default_width_checked"][0]
    # terms used by COS on that chain
    Ns = []
    for e in range(len(S.T)):
        q = np.flatnonzero(S.e == e)
        _, info = cos.cos_price(p13[:10], S.F[e], S.K[q], S.T[e], np.exp(-S.r * S.T[e]), "call", return_info=True)
        Ns.append(int(info["N"][0]))
    res["cos_terms_per_expiry"] = Ns
    # bulk throughput: many parameter sets (the PINN training-data use)
    rng = np.random.default_rng(7)
    Mset = 2000
    sel = rng.integers(0, len(df), Mset)
    PM = df[cols[:10]].values[sel]
    Ts = rng.uniform(3, 365, Mset) / 365.0
    Kx = np.exp(-np.linspace(-0.2, 0.2, 8))
    t_many = best_of(lambda: cos.cos_price_many(PM, 100.0, 100.0 * Kx, Ts, np.exp(-0.06 * Ts)), n=2)
    res["bulk_cos_price_many"] = dict(parameter_sets=Mset, strikes_each=8, seconds_best=t_many[0], microseconds_per_option=1e6 * t_many[0] / (Mset * 8),
                                      note="every set has its own parameters and maturity; one process; the PINN data generator can call this directly")
    xr = rng.uniform(-0.25, 0.25, 8000)
    PR = df[cols[:10]].values[rng.integers(0, len(df), 8000)]
    tr = rng.uniform(3, 365, 8000) / 365.0
    t_ind = best_of(lambda: cos.normalized_price(xr, tr, PR), n=2)
    res["bulk_normalized_price_independent_triples"] = dict(triples=8000, seconds_best=t_ind[0], microseconds_per_triple=1e6 * t_ind[0] / 8000,
                                                           note="every triple (x, tau, params) distinct: no sharing of the characteristic function")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="a small subset, for a smoke test")
    ap.add_argument("--only-timing", action="store_true", help="rerun timing.json only")
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "outputs" / "parameters_dhj_main.csv")
    if a.only_timing:
        old = json.load(open(OUT / "timing.json")) if (OUT / "timing.json").exists() else {}
        timing = run_timing(data.load_all(), df)
        timing["wall_seconds_for_whole_validation_script"] = old.get("wall_seconds_for_whole_validation_script")
        json.dump(timing, open(OUT / "timing.json", "w"), indent=2)
        sys.exit(0)
    t0 = time.time()
    print("synthetic grid ...", flush=True)
    syn = run_synthetic(df, a.quick)
    print("real chains ...", flush=True)
    real, surfaces = run_real(df, a.quick)
    print("monte carlo ...", flush=True)
    mcd = run_mc(a.quick)
    parts = [aggregate(d) for d in (syn, real, mcd)]
    errors = pd.concat([p[0] for p in parts], ignore_index=True)
    detail = pd.concat([p[1] for p in parts], ignore_index=True)
    suffix = "_quick" if a.quick else ""
    errors.to_csv(OUT / f"errors{suffix}.csv", index=False, float_format="%.6g")
    # the worst project-vs-COS cases (tradeable prices) for the README
    d = detail[(detail.source != "monte_carlo") & (detail.cos >= TICK)].copy()
    d["iv_err_pts"] = 100.0 * (d.iv_project - d.iv_cos)
    d["price_err_rs"] = d.project - d.cos
    d["lewis_minus_cos_rs"] = d.lewis - d.cos
    worst = d.reindex(d.iv_err_pts.abs().sort_values(ascending=False).index).head(40)
    worst[["source", "model", "set", "days", "x", "K", "is_call", "cos", "lewis", "project", "price_err_rs", "lewis_minus_cos_rs", "iv_cos", "iv_project", "iv_err_pts", "in_calibration_domain"]].to_csv(
        OUT / f"worst_cases{suffix}.csv", index=False, float_format="%.8g")
    print("timing ...", flush=True)
    timing = run_timing(surfaces, df)
    timing["wall_seconds_for_whole_validation_script"] = time.time() - t0
    json.dump(timing, open(OUT / f"timing{suffix}.json", "w"), indent=2)
    print("done in %.0fs" % (time.time() - t0))
