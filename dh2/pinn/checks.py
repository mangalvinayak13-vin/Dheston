"""Checks of the surrogate beyond price accuracy: Greeks against finite differences of the exact pricer, no-arbitrage structure of its prices,
speed against the exact pricer, and the independent verification of the PDE (and of the exact labels).  Stages: greeks | arb | speed | pde."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from . import data as D
from . import pde as PDE
from . import train as TR
from .surrogate import Surrogate
from .validate import TAB, _mask, _P, eval_sets

GREEK_MODELS = ["dat_s0", "dat_s1", "dat_s2", "hyb_s0", "hyb_s1", "hyb_s2", "spec_dat_s0"]


# ------------------------------------------------------------------ Greeks vs finite differences of the exact pricer
def exact_greeks(theta, v1, v2, tau, x, h_s=2e-3, h_g=4e-3, h_v=0.02):
    """Central finite differences of the project's pricer in the normalised world S = F = 1, r = q = 0 (K = e^-x).
    Returns delta_S, gamma_S, dC/dv1, dC/dv2 (C = K u)."""
    P = _P()
    K = float(np.exp(-x))

    def c(S, a, b):
        return float(P.double_heston_price(S, K, tau, 0.0, *D.theta_to_pricer(theta, a, b), kind="call", q=0.0))
    c0 = c(1.0, v1, v2)
    delta = (c(1 + h_s, v1, v2) - c(1 - h_s, v1, v2)) / (2 * h_s)
    gamma = (c(1 + h_g, v1, v2) - 2 * c0 + c(1 - h_g, v1, v2)) / h_g**2
    hv1, hv2 = h_v * v1, h_v * v2
    d1 = (c(1.0, v1 + hv1, v2) - c(1.0, v1 - hv1, v2)) / (2 * hv1)
    d2 = (c(1.0, v1, v2 + hv2) - c(1.0, v1, v2 - hv2)) / (2 * hv2)
    return delta, gamma, d1, d2


def stage_greeks(cfg, dom):
    sets = eval_sets(cfg, dom)
    real = sets["real"]
    rng = np.random.default_rng(5)
    days = np.unique(real["day"])
    days = days[:: max(1, len(days) // cfg["validate"]["fd_greeks_points"])]
    idx = []
    ok = np.flatnonzero(_mask(real) & (real["tau"] >= 5 / 365))
    for d in days:
        cand = ok[real["day"][ok] == d]
        if cand.size:
            idx.extend(rng.choice(cand, min(8, cand.size), replace=False))
    idx = np.array(idx)
    x, tau, v1, v2, th = real["x"][idx], real["tau"][idx], real["v1"][idx], real["v2"][idx], real["theta"][idx]
    ex = np.array([exact_greeks(th[i], v1[i], v2[i], tau[i], x[i]) for i in range(len(idx))])
    # reference scales: Black gamma and variance sensitivity at the quote's own implied vol
    sig = real["iv"][idx]
    d1b = (x + 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    g_bs = norm.pdf(d1b) / (sig * np.sqrt(tau))
    v_bs = norm.pdf(d1b) * np.sqrt(tau) / (2 * sig)
    rows = []
    for name in GREEK_MODELS:
        p = TR.CKPT / f"{name}.pt"
        if not p.exists():
            continue
        s = Surrogate.load(p, "cpu")
        g = s.greeks(np.ones(len(idx)), np.exp(-x), tau, np.ones(len(idx)), th, v1, v2, S=1.0)
        err = {"delta (abs)": np.abs(g["delta_S"] - ex[:, 0]), "gamma (frac of Black gamma)": np.abs(g["gamma_S"] - ex[:, 1]) / g_bs,
               "dC/dv1 (frac of Black var-sens.)": np.abs(g["dC_dv1"] - ex[:, 2]) / v_bs, "dC/dv2 (frac of Black var-sens.)": np.abs(g["dC_dv2"] - ex[:, 3]) / v_bs}
        for k, e in err.items():
            rows.append({"model": name, "greek": k, "n": len(e), "mean": e.mean(), "median": np.median(e), "p95": np.percentile(e, 95), "max": e.max()})
        print(name, {k: round(float(np.median(e)), 4) for k, e in err.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "greeks_errors.csv", index=False)
    np.savez(TAB.parent / "greeks_points.npz", idx=idx, exact=ex, g_bs=g_bs, v_bs=v_bs, x=x, tau=tau)
    # paired summary: median over seeds
    print(df.groupby(["model", "greek"])[["median", "p95"]].first().round(4).unstack().to_string())


# ------------------------------------------------------------------ no-arbitrage structure of the surrogate's prices
def _curves(sur, theta, v1, v2, taus, xs, exact: bool):
    """Call prices C = K u on rows (one parameter set and maturity each) over the strike grid xs (S = F = 1)."""
    n = len(theta)
    th = np.repeat(theta, len(taus), 0); a1 = np.repeat(v1, len(taus)); a2 = np.repeat(v2, len(taus)); tt = np.tile(taus, n)
    X = np.tile(xs, (len(tt), 1))
    if exact:
        u = D.pricer_call(th, a1, a2, tt, X)
    else:
        u = sur.price_normalized(X.ravel(), np.repeat(a1, len(xs)), np.repeat(a2, len(xs)), np.repeat(tt, len(xs)), np.repeat(th, len(xs), 0)).reshape(X.shape)
    return np.exp(-X) * u


def stage_arb(cfg, dom):
    rng = np.random.default_rng(21)
    th_b, a_b, b_b = D.sample_params(75, dom, rng, p_emp=0.0)
    sets = eval_sets(cfg, dom)
    real = sets["real"]
    days = rng.choice(np.unique(real["day"]), 75, replace=False)
    ii = np.array([np.flatnonzero(real["day"] == d)[0] for d in days])
    theta = np.vstack([th_b, real["theta"][ii]]); v1 = np.concatenate([a_b, real["v1"][ii]]); v2 = np.concatenate([b_b, real["v2"][ii]])
    xs = np.linspace(0.25, -0.25, 101)                      # K = e^-x increasing along the grid
    taus = np.array([5, 14, 30, 90, 180]) / 365.0
    K = np.exp(-xs)
    # calendar grid
    xk = np.log(1.0 / np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    tc = np.exp(np.linspace(np.log(5 / 365), np.log(0.6), 40))
    tols = [1e-7, 1e-6, 1e-5]
    rows = []
    for name in ["exact"] + ["dat_s0", "dat_s1", "dat_s2", "hyb_s0", "hyb_s1", "hyb_s2"]:
        sur = None if name == "exact" else Surrogate.load(TR.CKPT / f"{name}.pt", "cpu")
        if name != "exact" and sur is None:
            continue
        C = _curves(sur, theta, v1, v2, taus, xs, name == "exact")                 # (sets*taus, 101), K increasing
        d1 = np.diff(C, axis=1)                                                      # should be <= 0
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        bf = lam * C[:, :-2] + (1 - lam) * C[:, 2:] - C[:, 1:-1]                     # butterfly, should be >= 0 (convexity in K)
        lower = np.maximum(1.0 - K, 0.0)
        # calendar
        th2 = np.repeat(theta, len(tc), 0); a1 = np.repeat(v1, len(tc)); a2 = np.repeat(v2, len(tc)); t2 = np.tile(tc, len(theta))
        X = np.tile(xk, (len(t2), 1))
        if name == "exact":
            u = D.pricer_call(th2, a1, a2, t2, X)
        else:
            u = sur.price_normalized(X.ravel(), np.repeat(a1, len(xk)), np.repeat(a2, len(xk)), np.repeat(t2, len(xk)), np.repeat(th2, len(xk), 0)).reshape(X.shape)
        Cc = (np.exp(-X) * u).reshape(len(theta), len(tc), len(xk))
        cal = np.diff(Cc, axis=1)                                                    # should be >= 0 in maturity
        for tol in tols:
            rows.append({"price source": name, "tolerance (price per unit forward)": tol,
                         "monotone in strike: share violated": float((d1 > tol).mean()), "convex in strike: share violated": float((bf < -tol).mean()),
                         "below intrinsic: share": float((C < lower - tol).mean()), "above forward: share": float((C > 1 + tol).mean()),
                         "calendar: share violated": float((cal < -tol).mean()),
                         "max monotone violation": float(max(d1.max(), 0)), "max convexity violation": float(max((-bf).max(), 0)), "max calendar violation": float(max((-cal).max(), 0))})
        print(name, "violations at 1e-6: mono %.4f convex %.4f cal %.4f" % (rows[-3]["monotone in strike: share violated"], rows[-3]["convex in strike: share violated"], rows[-3]["calendar: share violated"]), flush=True)
    pd.DataFrame(rows).to_csv(TAB / "arbitrage_checks.csv", index=False)


# ------------------------------------------------------------------ speed
def stage_speed(cfg, dom):
    from dh2 import data as DD, models as M
    sets = eval_sets(cfg, dom)
    surfs = {s.day.isoformat(): s for s in DD.load_all()}
    real = sets["real"]
    # the held-out day whose in-domain chain size is closest to the median
    days = np.unique(real["day"])
    sizes = np.array([(_mask(real) & (real["day"] == d)).sum() for d in days])
    d = days[np.argsort(np.abs(sizes - np.median(sizes)))[0]]
    s = surfs[dom["ho_days"][d]]
    th = D.eta_to_theta(dom["ho_eta"][d]); v1, v2 = dom["ho_state"][d]
    args = D.theta_to_pricer(th, v1, v2)
    P = _P()
    sel = np.flatnonzero(s.T[s.e] <= D.TAU_MAX)
    groups = [(e, sel[s.e[sel] == e]) for e in range(len(s.T)) if np.any(s.e[sel] == e)]
    K = s.K[sel]; T = s.T[s.e[sel]]; F = s.F[s.e[sel]]; DF = np.exp(-s.r * T)

    def exact_chain(a):
        out = np.empty(len(sel))
        pos = 0
        for e, q in groups:
            n = len(q)
            out[pos:pos + n] = P.double_heston_price(s.spot, s.K[q], s.T[e], s.r, *a, kind="call", q=s.q[e])
            pos += n
        return out

    def timeit(f, n=20, warm=3):
        for _ in range(warm):
            f()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
        return float(np.median(t)), float(np.min(t))
    rows = []
    t_ex, _ = timeit(lambda: exact_chain(args), cfg["validate"]["speed_repeats"])
    rows.append({"what": "exact pricer, one chain", "device": "cpu (1 thread, numpy)", "quotes": len(sel), "expiries": len(groups), "median_ms": 1e3 * t_ex})
    for dev in ("cpu", "mps"):
        sur = Surrogate.load(TR.CKPT / "dat_s0.pt", dev)

        def run():
            r = sur.price_surface(F, K, T, DF, th, v1, v2)
            return r
        # price only (what a price-space calibration needs)
        def price_only():
            x = np.log(F / K)
            u = sur.price_normalized(x, v1, v2, T, th)
            if dev == "mps":
                torch.mps.synchronize()
            return K * DF * u
        t1, _ = timeit(price_only, cfg["validate"]["speed_repeats"])
        t2, _ = timeit(run, cfg["validate"]["speed_repeats"])
        rows.append({"what": "surrogate, one chain, prices", "device": dev, "quotes": len(sel), "expiries": len(groups), "median_ms": 1e3 * t1})
        rows.append({"what": "surrogate, one chain, prices + Black IV", "device": dev, "quotes": len(sel), "expiries": len(groups), "median_ms": 1e3 * t2})
    # many parameter sets at once (what a Jacobian or a global search needs): B parameter sets, the same chain
    rng = np.random.default_rng(3)
    for B in (16, 128, 1024):
        th_b, a_b, b_b = D.sample_params(B, dom, rng, p_emp=0.5)
        if B <= 128:
            def ex_many():
                for i in range(B):
                    exact_chain(D.theta_to_pricer(th_b[i], a_b[i], b_b[i]))
            t_e, _ = timeit(ex_many, 3, 1)
            rows.append({"what": f"exact pricer, {B} parameter sets x chain", "device": "cpu (1 thread, numpy)", "quotes": len(sel) * B, "expiries": len(groups), "median_ms": 1e3 * t_e})
        for dev in ("cpu", "mps"):
            sur = Surrogate.load(TR.CKPT / "dat_s0.pt", dev)
            x = np.tile(np.log(F / K), B); TT = np.tile(T, B); A1 = np.repeat(a_b, len(sel)); A2 = np.repeat(b_b, len(sel)); TH = np.repeat(th_b, len(sel), 0)

            def many():
                u = sur.price_normalized(x, A1, A2, TT, TH)
                if dev == "mps":
                    torch.mps.synchronize()
            t_s, _ = timeit(many, 5, 2)
            rows.append({"what": f"surrogate, {B} parameter sets x chain", "device": dev, "quotes": len(sel) * B, "expiries": len(groups), "median_ms": 1e3 * t_s})
    # Greeks of one chain: autograd (one pass) vs finite differences of the exact pricer (7 chain evaluations)
    sur = Surrogate.load(TR.CKPT / "dat_s0.pt", "cpu")
    t_g, _ = timeit(lambda: sur.greeks(F, K, T, DF, th, v1, v2, S=s.spot), 10)
    rows.append({"what": "surrogate Greeks (delta, gamma, dC/dv1, dC/dv2), one chain", "device": "cpu", "quotes": len(sel), "expiries": len(groups), "median_ms": 1e3 * t_g})

    def ex_greeks():
        for dS in (0, 1, -1, 2, -2):
            exact_chain(args)
        exact_chain([v1 * 1.02] + list(args[1:5]) + [v2] + list(args[6:]))
        exact_chain([v1 * 0.98] + list(args[1:5]) + [v2] + list(args[6:]))
        exact_chain([v1] + list(args[1:5]) + [v2 * 1.02] + list(args[6:]))
        exact_chain([v1] + list(args[1:5]) + [v2 * 0.98] + list(args[6:]))
    t_eg, _ = timeit(ex_greeks, 5, 1)
    rows.append({"what": "exact Greeks by finite differences (9 chain evaluations), one chain", "device": "cpu (1 thread, numpy)", "quotes": len(sel), "expiries": len(groups), "median_ms": 1e3 * t_eg})
    df = pd.DataFrame(rows)
    df["ms_per_quote"] = df["median_ms"] / df["quotes"]
    df.to_csv(TAB / "speed.csv", index=False)
    print(df.round(4).to_string())


# ------------------------------------------------------------------ PDE verification, precision, label cross-checks
def stage_pde(cfg, dom):
    rng = np.random.default_rng(11)
    n = cfg["validate"]["pde_check_points"]
    theta, v1, v2 = D.sample_params(n, dom, rng, p_emp=0.5)
    tau = np.exp(rng.uniform(np.log(0.02), np.log(0.5), n))
    sd = np.sqrt(D.total_variance(theta, v1, v2, tau))
    x = sd * rng.uniform(-1.5, 1.5, n)
    cosf = lambda th, a, b, t, xs: D.cos_call(np.repeat(th[None], len(a), 0), a, b, t, np.asarray(xs).reshape(len(a), -1), N=4096, L=14)
    rows = []
    for i in range(n):
        for f in (0.25, 0.125, 0.0625):
            st = dict(hx=f * sd[i], h1=f * v1[i], h2=f * v2[i], ht=0.4 * f * tau[i])
            for src, fn in (("project pricer", None), ("COS (own)", cosf)):
                r = PDE.verify_pde_fd(theta[i], v1[i], v2[i], tau[i], x[i], steps=st, price_fn=fn)
                t = r["terms"]
                flipped = r["ut"] - (t["diffusion"] + t["drift1"] + t["drift2"] + t["volvol1"] + t["volvol2"] - t["cross1"] - t["cross2"])
                rows.append({"point": i, "tau": tau[i], "v1": v1[i], "v2": v2[i], "x_over_sd": x[i] / sd[i], "step_factor": f, "price source": src,
                             "u_tau": r["ut"], "rhs": r["rhs"], "abs_residual": abs(r["residual"]), "sum_abs_terms": r["scale"],
                             "rel_residual": abs(r["residual"]) / r["scale"], "rel_residual_cross_sign_flipped": abs(flipped) / r["scale"],
                             "cross_share_of_scale": (abs(t["cross1"]) + abs(t["cross2"])) / r["scale"]})
        if i % 10 == 9:
            print("pde check point", i + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "pde_verification_points.csv", index=False)
    summ = df.groupby(["price source", "step_factor"]).agg(median_rel_residual=("rel_residual", "median"), p90_rel_residual=("rel_residual", lambda v: np.percentile(v, 90)),
                                                          max_rel_residual=("rel_residual", "max"), median_rel_residual_cross_sign_flipped=("rel_residual_cross_sign_flipped", "median"),
                                                          median_cross_share=("cross_share_of_scale", "median")).reset_index()
    summ.to_csv(TAB / "pde_verification_summary.csv", index=False)
    print(summ.round(6).to_string())

    # --- label cross-check: project pricer vs own COS vs dh2/cos.py (the other agent's independent COS)
    rng = np.random.default_rng(31)
    m = 300
    th, a, b = D.sample_params(m, dom, rng, p_emp=0.5)
    ta = D.sample_tau(m, rng)
    xx = np.stack([D.sample_x(th, a, b, ta, rng) for _ in range(8)], 1)
    ref = D.pricer_call(th, a, b, ta, xx)
    out = []
    for lab, u in (("own COS N=2048 L=12", D.cos_call(th, a, b, ta, xx, N=2048, L=12.0)), ("own COS N=4096 L=16", D.cos_call(th, a, b, ta, xx, N=4096, L=16.0))):
        e = np.abs(u - ref); out.append({"comparison": "project pricer vs " + lab, "max_abs": e.max(), "p99_abs": np.percentile(e, 99), "median_abs": np.median(e)})
    try:
        from dh2 import cos as C2
        u2 = np.empty_like(ref)
        for i in range(m):
            u2[i] = C2.normalized_price(xx[i], ta[i], np.array(D.theta_to_pricer(th[i], a[i], b[i])))
        e = np.abs(u2 - ref); out.append({"comparison": "project pricer vs dh2/cos.py (independent COS by the pricer-validation package)", "max_abs": e.max(), "p99_abs": np.percentile(e, 99), "median_abs": np.median(e)})
    except Exception as ex:                                                  # noqa: BLE001
        out.append({"comparison": f"dh2/cos.py not usable here: {ex}", "max_abs": np.nan, "p99_abs": np.nan, "median_abs": np.nan})
    pd.DataFrame(out).to_csv(TAB / "cos_vs_pricer.csv", index=False)
    print(pd.DataFrame(out).to_string())

    # --- precision of the PDE residual: fp64 CPU vs fp32 CPU vs fp32 MPS, for trained networks
    sets = eval_sets(cfg, dom)
    inp = {k: sets["box"][k][:4000] for k in ("x", "v1", "v2", "tau", "theta")}
    rows = []
    for name in ("dat_s0", "hyb_s0"):
        res = {}
        for lab, dev, dt in (("cpu float64", "cpu", torch.float64), ("cpu float32", "cpu", torch.float32), ("mps float32", "mps", torch.float32)):
            sur = Surrogate.load(TR.CKPT / f"{name}.pt", dev)
            m_ = sur.model.to(dt)
            b_ = {k: torch.tensor(v, dtype=dt, device=dev) for k, v in inp.items()}
            r_, d_ = [], None
            for s0 in range(0, 4000, 1000):
                c = {k: v[s0:s0 + 1000] for k, v in b_.items()}
                with torch.enable_grad():
                    nr, rr, _ = PDE.normalised_residual(m_, c["x"], c["v1"], c["v2"], c["tau"], c["theta"], 1e-4, create_graph=False)
                r_.append(nr.detach().cpu().double().numpy())
            res[lab] = np.concatenate(r_)
        for lab in ("cpu float32", "mps float32"):
            d = np.abs(res[lab] - res["cpu float64"])
            rows.append({"network": name, "arithmetic": lab, "median |residual| (float64)": float(np.median(np.abs(res["cpu float64"]))), "median |difference to float64|": float(np.median(d)),
                         "p95 |difference|": float(np.percentile(d, 95)), "p99 |difference|": float(np.percentile(d, 99)),
                         "ratio median difference / median residual": float(np.median(d) / np.median(np.abs(res["cpu float64"])))})
    pd.DataFrame(rows).to_csv(TAB / "precision_check.csv", index=False)
    print(pd.DataFrame(rows).round(5).to_string())


# ------------------------------------------------------------------ domain table and test-period flag
def stage_domain(cfg, dom):
    """The box (from development days only), and a FLAG for calibrated test-period parameters outside it. Nothing is tuned on this."""
    lo, hi = D.domain_theta_bounds(dom)
    rows = []
    for i, n in enumerate(D.THETA_NAMES):
        rows.append({"quantity": n, "box low": lo[i], "box high": hi[i],
                     "dev min": np.nan if i == 4 else dom["dev_eta_min"][i], "dev max": np.nan if i == 4 else dom["dev_eta_max"][i]})
    rows += [{"quantity": "v1, v2 (state)", "box low": dom["v_lo"][0], "box high": dom["v_hi"][0], "dev min": dom["dev_state_min"].min(), "dev max": dom["dev_state_max"].max()},
             {"quantity": "tau (years)", "box low": dom["tau_lo"], "box high": dom["tau_hi"], "dev min": np.nan, "dev max": np.nan},
             {"quantity": "|x| = |ln(F/K)|", "box low": 0.0, "box high": dom["x_max"], "dev min": np.nan, "dev max": np.nan}]
    pd.DataFrame(rows).to_csv(TAB / "domain_box.csv", index=False)
    te, ts = dom["test_eta"], dom["test_state"]
    inside = D.in_domain(dom, te, ts)
    flag = []
    for i, n in enumerate(D.ETA_NAMES):
        out = (te[:, i] < dom["eta_lo"][i] - 1e-12) | (te[:, i] > dom["eta_hi"][i] + 1e-12)
        flag.append({"item": n + " (eta order; dk = kappa2 - kappa1)", "test days outside the box": int(out.sum()), "share of test days": float(out.mean())})
    for j, n in enumerate(("v1", "v2")):
        out = (ts[:, j] < dom["v_lo"][j]) | (ts[:, j] > dom["v_hi"][j])
        flag.append({"item": n + " (state)", "test days outside the box": int(out.sum()), "share of test days": float(out.mean())})
    flag.append({"item": "ANY of the above (day flagged)", "test days outside the box": int((~inside).sum()), "share of test days": float((~inside).mean())})
    pd.DataFrame(flag).to_csv(TAB / "test_period_domain_flags.csv", index=False)
    sets = eval_sets(cfg, dom)
    r = sets["real"]; t = sets["test"]
    q = []
    for nm, s in (("held-out development days (real)", r), ("test period, every 10th day (diagnostic)", t)):
        q.append({"set": nm, "quotes": len(s["x"]), "tau > 0.6 y (outside box)": int((s["tau"] > D.TAU_MAX).sum()), "|x| > 0.30": int((np.abs(s["x"]) > D.X_MAX).sum()),
                  "exact OTM price below Rs-1 equivalent (not scored)": int((s["u_otm"] < D.P_MIN).sum()), "scored quotes": int(_mask(s).sum())})
    pd.DataFrame(q).to_csv(TAB / "quote_counts.csv", index=False)
    D.save_domain({k: v for k, v in dom.items() if k not in ("test_eta", "test_state", "test_days")}, TAB.parent / "domain.json")
    print(pd.read_csv(TAB / "domain_box.csv").round(4).to_string()); print(pd.DataFrame(flag).to_string()); print(pd.DataFrame(q).to_string())
