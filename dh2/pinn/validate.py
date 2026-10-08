"""Validation of the surrogate against the project's exact pricer (pricer/models.py) on parameter sets it never saw.

Held-out sets (all labelled by the EXACT pricer):
  mix   in-distribution monitor (fresh draws of the training distribution; used during training for monitoring only)
  box   1500 parameter sets drawn from the box, never seen, 8 strikes each
  real  the real option chains of the 251 held-out development days (the last 20% of 2016-2021, never used by the training sampler),
        priced at each day's own calibrated (eta, state): every quote of the day, exactly as the calibrator would ask for it
  grid  a regular (x, tau) grid at 120 held-out parameter sets, for the error and PDE-residual maps
  test  every 10th TEST-period day (2022-2026) at its calibrated parameters: a DIAGNOSTIC and domain flag only; nothing was tuned on it
Stages (each is quick; run separately):  python3 -m dh2.pinn.validate <stage>
  sets | metrics | greeks | arb | speed | pde | figures
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from . import pde as PDE
from . import train as TR
from .surrogate import Surrogate

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "pinn"
TAB = RES / "tables"
FIG = RES / "figures"
CACHE = ROOT / "data" / "cache" / "pinn"

MONEY_EDGES = [-0.30, -0.10, -0.03, 0.03, 0.10, 0.30]            # in k = ln(K/F)
MONEY_LABELS = ["k < -0.10 (OTM puts, deep)", "-0.10 .. -0.03 (OTM puts)", "|k| < 0.03 (ATM)", "0.03 .. 0.10 (OTM calls)", "k > 0.10 (OTM calls, deep)"]
TAU_EDGES = [0.0, 7 / 365, 30 / 365, 90 / 365, 0.6001]
TAU_LABELS = ["1-7 days", "7-30 days", "30-90 days", "90 days - 0.6 y"]


def _P():
    return D._project_pricer()


# ------------------------------------------------------------------ evaluation sets
def _real_set(dom: dict, which: str, stride: int = 1) -> dict:
    from dh2 import data as DD
    surfs = {s.day.isoformat(): s for s in DD.load_all()}
    if which == "ho":
        days, eta, state = dom["ho_days"], dom["ho_eta"], dom["ho_state"]
    else:
        days, eta, state = dom["test_days"], dom["test_eta"], dom["test_state"]
    P = _P()
    rows = {k: [] for k in ("x", "tau", "v1", "v2", "kdf", "u", "day", "exp", "spot")}
    th_rows = []
    for i in range(0, len(days), stride):
        s = surfs.get(days[i])
        if s is None:
            continue
        th = D.eta_to_theta(eta[i]); v1, v2 = state[i]
        args = D.theta_to_pricer(th, v1, v2)
        for e in range(len(s.T)):
            q = np.flatnonzero(s.e == e)
            if not q.size:
                continue
            K = s.K[q]
            call = np.atleast_1d(P.double_heston_price(s.spot, K, s.T[e], s.r, *args, kind="call", q=s.q[e]))
            kdf = K * np.exp(-s.r * s.T[e])
            n = q.size
            rows["x"].append(np.log(s.F[e] / K)); rows["tau"].append(np.full(n, s.T[e])); rows["v1"].append(np.full(n, v1)); rows["v2"].append(np.full(n, v2))
            rows["kdf"].append(kdf); rows["u"].append(call / kdf); rows["day"].append(np.full(n, i)); rows["exp"].append(np.full(n, e)); rows["spot"].append(np.full(n, s.spot))
            th_rows.append(np.tile(th, (n, 1)))
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["theta"] = np.concatenate(th_rows)
    return _finish_labels(out)


def _finish_labels(d: dict) -> dict:
    d["u_otm"] = D.exact_otm(d["u"], d["x"])
    d["iv"] = D.implied_vol_unit(np.maximum(d["u_otm"], 1e-14), d["x"], d["tau"])
    d["vega"] = np.maximum(D.vega_per_point(d["x"], d["tau"], np.where(np.isfinite(d["iv"]), d["iv"], 0.15)), D.VEGA_FLOOR)
    return d


def _grid_set(cfg: dict, dom: dict, real: dict, n_box: int = 60, n_real: int = 60) -> dict:
    """Regular (x, tau) grid at held-out parameter sets: n_box from the box, n_real real held-out days' parameters."""
    rng = np.random.default_rng(777)
    th_b, a_b, b_b = D.sample_params(n_box, dom, rng, p_emp=0.0)
    days = np.unique(real["day"]); pick = rng.choice(days, n_real, replace=False)
    th_r, a_r, b_r = [], [], []
    for d in pick:
        i = np.flatnonzero(real["day"] == d)[0]
        th_r.append(real["theta"][i]); a_r.append(real["v1"][i]); b_r.append(real["v2"][i])
    th = np.vstack([th_b, np.array(th_r)]); v1 = np.concatenate([a_b, a_r]); v2 = np.concatenate([b_b, b_r])
    xs = np.linspace(-0.28, 0.28, 29); taus = np.exp(np.linspace(np.log(D.TAU_MIN), np.log(D.TAU_MAX), 12))
    n = len(th)
    X = np.tile(xs, n * len(taus)); T = np.repeat(np.tile(taus, n), len(xs)); TH = np.repeat(th, len(taus) * len(xs), 0)
    V1 = np.repeat(v1, len(taus) * len(xs)); V2 = np.repeat(v2, len(taus) * len(xs))
    u = np.empty(len(X))
    blk = len(xs)
    P = _P()
    for j in range(n * len(taus)):
        sl = slice(j * blk, (j + 1) * blk)
        u[sl] = D.pricer_call(TH[sl][:1], V1[sl][:1], V2[sl][:1], T[sl][:1], X[sl].reshape(1, -1))[0]
    out = {"x": X, "tau": T, "v1": V1, "v2": V2, "theta": TH, "u": u, "set": np.repeat(np.arange(n), len(taus) * len(xs)), "is_box": np.repeat(np.arange(n) < n_box, len(taus) * len(xs))}
    out["kdf"] = np.ones(len(X))
    return _finish_labels(out)


def eval_sets(cfg: dict, dom: dict, refresh: bool = False) -> dict:
    f = CACHE / "eval_sets.pkl"
    if f.exists() and not refresh:
        return pickle.load(open(f, "rb"))
    t = time.time()
    sets = {"mix": TR.labelled(cfg, dom, "val"), "box": TR.labelled(cfg, dom, "box")}
    sets["real"] = _real_set(dom, "ho")
    sets["grid"] = _grid_set(cfg, dom, sets["real"])
    sets["test"] = _real_set(dom, "test", stride=10)
    for k in ("mix", "box"):
        sets[k]["kdf"] = np.ones(len(sets[k]["x"]))
    pickle.dump(sets, open(f, "wb"))
    print("eval sets built in %.0fs: " % (time.time() - t) + ", ".join(f"{k}={len(v['x'])}" for k, v in sets.items()), flush=True)
    return sets


# ------------------------------------------------------------------ metrics
def _mask(ref: dict, p_min: float = D.P_MIN) -> np.ndarray:
    return (ref["u_otm"] >= p_min) & np.isfinite(ref["iv"]) & (ref["tau"] <= D.TAU_MAX + 1e-12) & (np.abs(ref["x"]) <= D.X_MAX)


def predict_set(sur: Surrogate, ref: dict) -> np.ndarray:
    return sur.price_normalized(ref["x"], ref["v1"], ref["v2"], ref["tau"], ref["theta"])


def summarize(err: np.ndarray, mask: np.ndarray) -> dict:
    e = np.abs(err[mask]); nan = int(np.isnan(e).sum()); e = e[np.isfinite(e)]
    return {"n": int(mask.sum()), "n_nan": nan, "mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "p99": float(np.percentile(e, 99)), "max": float(e.max()), "bias": float(np.nanmean(err[mask]))}


def bucket_table(ref: dict, err: np.ndarray, mask: np.ndarray, price_err: np.ndarray | None = None) -> pd.DataFrame:
    k = -ref["x"]
    rows = []
    for mi, ml in enumerate(MONEY_LABELS):
        for ti, tl in enumerate(TAU_LABELS):
            m = mask & (k >= MONEY_EDGES[mi]) & (k < MONEY_EDGES[mi + 1]) & (ref["tau"] >= TAU_EDGES[ti]) & (ref["tau"] < TAU_EDGES[ti + 1])
            e = np.abs(err[m]); e = e[np.isfinite(e)]
            r = {"moneyness": ml, "maturity": tl, "n": int(m.sum())}
            if e.size:
                r.update(mean=float(e.mean()), p95=float(np.percentile(e, 95)), max=float(e.max()), bias=float(np.nanmean(err[m])))
                if price_err is not None:
                    r["mean_abs_price_err_rupees"] = float(np.abs(price_err[m]).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def evaluate_model(path: Path, sets: dict, device: str = "mps", names=("mix", "box", "real", "grid", "test")) -> dict:
    sur = Surrogate.load(path, device)
    out = {}
    for nm in names:
        ref = sets[nm]
        u = predict_set(sur, ref)
        otm = D.exact_otm(u, ref["x"])
        iv = D.implied_vol_unit(np.maximum(otm, 1e-14), ref["x"], ref["tau"])
        err = 100.0 * (iv - ref["iv"])
        m = _mask(ref)
        s = summarize(err, m)
        pe = (u - ref["u"]) * ref["kdf"]
        s["price_mae_rupees"] = float(np.abs(pe[m]).mean()) if nm in ("real", "test") else float("nan")
        s["price_max_rupees"] = float(np.abs(pe[m]).max()) if nm in ("real", "test") else float("nan")
        out[nm] = {"summary": s, "err": err, "mask": m, "price_err": pe, "u": u}
    return out


def residual_stats(sur: Surrogate, inp: dict, n: int = 6000, floor: float = 1e-4) -> dict:
    """Normalised PDE residual (tau*R/vega in volatility points) of the surrogate at n points (fp32 on the surrogate's device)."""
    sl = slice(0, n)
    b = {k: torch.tensor(np.asarray(inp[k][sl]), dtype=torch.float32, device=sur.device) for k in ("x", "v1", "v2", "tau", "theta")}
    res = []
    for s0 in range(0, n, 2000):
        c = {k: v[s0:s0 + 2000] for k, v in b.items()}
        with torch.enable_grad():
            nr, _, _ = PDE.normalised_residual(sur.model, c["x"], c["v1"], c["v2"], c["tau"], c["theta"], floor, create_graph=False)
        res.append(nr.detach().cpu().numpy().astype(np.float64))
    r = np.abs(np.concatenate(res))
    return {"res_median": float(np.median(r)), "res_p95": float(np.percentile(r, 95)), "res_p99": float(np.percentile(r, 99)), "res_rms": float(np.sqrt((r ** 2).mean()))}


# ------------------------------------------------------------------ stage: metrics (ablation tables)
RUNS = ["hyb_s0", "hyb_s1", "hyb_s2", "dat_s0", "dat_s1", "dat_s2", "phy_s0", "spec_dat_s0",
        "de_dat_1000", "de_dat_10000", "de_dat_100000", "de_hyb_1000", "de_hyb_10000", "de_hyb_100000"]


def stage_metrics(cfg, dom):
    sets = eval_sets(cfg, dom)
    TAB.mkdir(parents=True, exist_ok=True)
    rows, buckets, store = [], {}, {}
    for r in RUNS:
        p = TR.CKPT / f"{r}.pt"
        if not p.exists():
            print("missing", r); continue
        ev = evaluate_model(p, sets, cfg["device"])
        sur = Surrogate.load(p, cfg["device"])
        res_box = residual_stats(sur, sets["box"]); res_real = residual_stats(sur, sets["real"])
        info = torch.load(p, map_location="cpu", weights_only=False)["info"]
        for nm, e in ev.items():
            rows.append({"run": r, "mode": info["mode"], "seed": info["seed"], "n_data": info["n_data"], "steps": info["steps"], "train_wall_s": info["wall_s"], "set": nm, **e["summary"],
                         **({f"pde_{k}": v for k, v in (res_box if nm == "box" else res_real).items()} if nm in ("box", "real") else {})})
        if r in ("hyb_s0", "dat_s0", "hyb_s1", "dat_s1", "hyb_s2", "dat_s2", "spec_dat_s0", "phy_s0"):
            for nm in ("real", "box"):
                buckets[(r, nm)] = bucket_table(sets[nm], ev[nm]["err"], ev[nm]["mask"], ev[nm]["price_err"] if nm == "real" else None)
        print(r, {nm: round(e["summary"]["mean"], 4) for nm, e in ev.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "heldout_metrics_all_runs.csv", index=False)
    bl = []
    for (r, nm), b in buckets.items():
        b = b.copy(); b.insert(0, "set", nm); b.insert(0, "run", r); bl.append(b)
    pd.concat(bl).to_csv(TAB / "bucket_errors_all_runs.csv", index=False)
    print("wrote metrics tables")


if __name__ == "__main__":
    cfg = TR.load_cfg(); dom = TR.get_domain(cfg)
    torch.set_num_threads(cfg.get("threads", 2))
    st = sys.argv[1] if len(sys.argv) > 1 else "sets"
    if st == "sets":
        eval_sets(cfg, dom, refresh="--refresh" in sys.argv)
    elif st == "metrics":
        stage_metrics(cfg, dom)
    elif st == "figures":
        from . import figures
        figures.stage_figures(cfg, dom)
    else:                                              # greeks | arb | speed | pde
        from . import checks
        getattr(checks, "stage_" + st)(cfg, dom)
