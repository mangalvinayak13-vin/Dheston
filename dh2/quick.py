"""The demo-scale run of the pipeline (python -m dh2.run_all --quick): the same chain of steps as the full run on a short stretch of REAL data, with small budgets.

Everything is written to results_quick/ and nothing in results/ is touched. Every output says it is a demo-scale run and not the headline result
(banner on the markdown files, a first comment line on the CSVs, a notice key in the JSON, a footer on the figure).

Steps (one function each, called by the stages q_*):
  q_data      the real surfaces of the window (cached surfaces, no download)
  q_walk      a mini walk-forward of Heston, Double Heston and the jump model: dh2/../run_walk.py's own `worker`, small window / refit / budgets (configs/nifty.yaml: quick)
  q_forecast  the forecast variants of hybrids.py on those fits, scored against yesterday's smile and the time-scaled smile, with paired block-bootstrap intervals
              and, for reference, the shipped full-run score of the same forecasts on the same days
  q_pricer    the project pricer against the independent COS pricer (and Lewis on a thin subset) on the real chains of 3 days, at the mini calibrations
  q_pinn      the shipped PINN surrogates (results/pinn/checkpoints) against the exact pricer on the same 3 real days
  q_svi       the SVI yardstick fitted on a few of the days, against the models' same-day fits
  q_stats     Wilcoxon and Diebold-Mariano on the mini forecast errors (dh2.stats)
  q_report    results_quick/report.md (same layout as results/report.md), metrics_quick.json and one figure
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTICE = "DEMO-SCALE RUN, NOT THE HEADLINE RESULT"
LONG_NOTICE = ("This is a demo-scale run of the pipeline: a short stretch of real NIFTY data, a small window, few refits and small optimiser budgets. "
               "It shows that every step runs end to end; its numbers are not the project's results (those are in results/).")
MODEL_LABEL = {"heston": "Heston", "dh": "diffusion DH", "dhj": "jump model"}


def _q(cfg, root=ROOT) -> Path:
    return Path(root) / cfg["paths"]["results_quick"]


def _paths(cfg, root=ROOT):
    q = _q(cfg, root)
    return q, q / "work", q / "tables", q / "figures"


def demo_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {NOTICE}. {LONG_NOTICE}\n")
        df.to_csv(f, index=False)


def read_demo_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def _surfaces(cfg):
    from .data import load_all
    qc = cfg["quick"]
    return load_all(str(qc["start"]), str(qc["end"]))


def _walk_job(root: str, model: str, S, lo: int, hi: int, qc: dict, j: int):
    import sys as _sys
    if root not in _sys.path:
        _sys.path.insert(0, root)
    import run_walk
    t0 = time.time()
    out = run_walk.worker(model, S, lo, hi, qc["window"], qc["refit"], 0.0, qc["nfev_cold"], qc["nfev_warm"], qc["seed_base"] + j, qc["per_expiry"], qc["burn"], None, None)
    return model, j, out, time.time() - t0


def _hyb_job(root: str, model: str, S, recs):
    import sys as _sys
    if root not in _sys.path:
        _sys.path.insert(0, root)
    import hybrids
    return hybrids.chunk(model, S, recs)


# =================================================================================================================== stages
def q_data(cfg, root=ROOT, workers=2):
    S = _surfaces(cfg)
    qc = cfg["quick"]
    if len(S) <= qc["window"] + 5:
        raise RuntimeError(f"only {len(S)} surfaces in {qc['start']}..{qc['end']}: the quick window needs at least {qc['window'] + 6}")
    df = pd.DataFrame({"day": [s.day for s in S], "quotes": [s.n for s in S], "expiries": [len(s.T) for s in S]})
    demo_csv(df, _paths(cfg, root)[2] / "quick_window.csv")
    return {"days": len(S), "first": str(S[0].day), "last": str(S[-1].day)}


def q_walk(cfg, root=ROOT, workers=2):
    from joblib import Parallel, delayed
    S = _surfaces(cfg)
    qc = cfg["quick"]
    q, work, _, _ = _paths(cfg, root)
    work.mkdir(parents=True, exist_ok=True)
    first = qc["window"]
    chunks = [(int(c[0]), int(c[-1]) + 1) for c in np.array_split(np.arange(first, len(S)), qc["chunks"]) if len(c)]
    jobs = [(m, j, lo, hi) for m in qc["models"] for j, (lo, hi) in enumerate(chunks)]
    res = Parallel(n_jobs=workers)(delayed(_walk_job)(str(root), m, S, lo, hi, qc, j) for m, j, lo, hi in jobs)
    secs = {}
    for m in qc["models"]:
        flat = [r for mm, j, out, s in sorted(res, key=lambda x: (x[0], x[1])) if mm == m for r in out]
        secs[m] = float(sum(s for mm, _, _, s in res if mm == m))
        with open(work / f"walk_{m}.pkl", "wb") as f:
            pickle.dump(flat, f)
    t = pd.DataFrame([{"model": m, "forecast_days": len([1 for r in pickle.load(open(work / f'walk_{m}.pkl', 'rb'))]), "worker_seconds": secs[m]} for m in qc["models"]])
    demo_csv(t, _paths(cfg, root)[2] / "quick_walk_forward.csv")
    return secs


def _load_walks(cfg, root):
    q, work, _, _ = _paths(cfg, root)
    return {m: pickle.load(open(work / f"walk_{m}.pkl", "rb")) for m in cfg["quick"]["models"]}


def q_forecast(cfg, root=ROOT, workers=2):
    from joblib import Parallel, delayed
    from . import evalpairs as EP
    from .data import load_all
    sys.path.insert(0, str(root))
    from tournament import boot
    S = _surfaces(cfg)
    qc = cfg["quick"]
    q, work, tabs, _ = _paths(cfg, root)
    byday = {s.day: i for i, s in enumerate(S)}
    W = _load_walks(cfg, root)
    hyb = {}
    for m in qc["models"]:
        recs = []
        for r in W[m]:
            t = byday.get(r["day"])
            if t is not None and t + 1 < len(S) and (S[t + 1].day - S[t].day).days <= 5:
                recs.append({**r, "t_global": t})
        parts = np.array_split(np.arange(len(recs)), max(1, workers * 2))
        res = Parallel(n_jobs=workers)(delayed(_hyb_job)(str(root), m, S, [recs[i] for i in p]) for p in parts if len(p))
        preds = {}
        for d in res:
            for t, vs in d.items():
                preds[t] = vs
        hyb[m] = preds
        with open(work / f"hyb_{m}.pkl", "wb") as f:
            pickle.dump(preds, f)
    common = sorted(set.intersection(*[set(h) for h in hyb.values()]))
    series = {"smile": [], "smile_scaled": []}
    for m in qc["models"]:
        series[f"{m}:carry"] = []
        series[f"{m}:carry_res"] = []
    for t in common:
        series["smile"].append(EP.score(hyb[qc["models"][0]][t]["smile"].astype(float), S[t + 1])["rmse"])
        series["smile_scaled"].append(EP.score(EP.pred_smile_scaled(S[t], S[t + 1]), S[t + 1])["rmse"])
        for m in qc["models"]:
            for v in ("carry", "carry_res"):
                series[f"{m}:{v}"].append(EP.score(hyb[m][t][v].astype(float), S[t + 1])["rmse"])
    daily = pd.DataFrame({"day": [S[t].day for t in common], **{k: np.round(v, 6) for k, v in series.items()}})
    demo_csv(daily, tabs / "forecast_daily_rmse.csv")
    bc = qc["bootstrap"]
    bt = lambda d: boot(d, block=bc["block"], n=bc["n"], seed=bc["seed"])
    best = min(("smile", "smile_scaled"), key=lambda k: float(np.mean(series[k])))
    # the shipped full-run forecasts of the same models on the same days (for reference only)
    Sf = load_all()
    byf = {s.day: i for i, s in enumerate(Sf)}
    ship = {}
    for m in qc["models"]:
        p = Path(root) / cfg["paths"]["walk_results"] / f"hyb_{m}_main.pkl"
        if p.exists():
            H = pickle.load(open(p, "rb"))
            for v in ("carry", "carry_res"):
                vals = []
                for t in common:
                    tf = byf.get(S[t].day)
                    vals.append(EP.score(H[v][tf].astype(float), Sf[tf + 1])["rmse"] if (tf is not None and tf in H[v]) else np.nan)
                ship[f"{m}:{v}"] = float(np.nanmean(vals))
    rows = []
    labels = {"smile": "yesterday's smile", "smile_scaled": "time-scaled smile"}
    for k, v in series.items():
        v = np.array(v)
        mean, lo, hi = bt(v)
        r = {"series": k, "label": labels.get(k) or (MODEL_LABEL[k.split(':')[0]] + (" + misfit" if k.endswith("carry_res") else "")), "kind": "benchmark" if ":" not in k else "model",
             "n_days": len(v), "mean": mean, "ci_lo": lo, "ci_hi": hi, "best_benchmark": best, "diff_vs_best_benchmark": np.nan, "diff_lo": np.nan, "diff_hi": np.nan,
             "shipped_full_run_same_days": ship.get(k, np.nan)}
        if ":" in k:
            d, dlo, dhi = bt(v - np.array(series[best]))
            r.update(diff_vs_best_benchmark=d, diff_lo=dlo, diff_hi=dhi)
        rows.append(r)
    demo_csv(pd.DataFrame(rows), tabs / "forecast_models.csv")
    return {"days": len(common)}


def _exact_chain(model, x, s):
    from . import models as M
    return M.price(model, x, s)


def _pick_days(cfg, S, W, n):
    byday = {s.day: i for i, s in enumerate(S)}
    days = sorted(r["day"] for r in W["dh"] if byday.get(r["day"]) is not None)
    idx = np.unique(np.round(np.linspace(0, len(days) - 1, n)).astype(int))
    return [days[i] for i in idx], byday


def q_pricer(cfg, root=ROOT, workers=2):
    from . import calib, cos, data as D, lewis, models as M, struct as ST
    S = _surfaces(cfg)
    qc = cfg["quick"]
    W = _load_walks(cfg, root)
    picks, byday = _pick_days(cfg, S, W, qc["pricer_days"])
    rows = []
    tick = 0.05
    for model in ("dh", "dhj"):
        recs = {r["day"]: r for r in W[model]}
        for d in picks:
            s = S[byday[d]]
            r = recs[d]
            x = calib.to_phys(model, ST.assemble(model, r["eta"], r["state"]))
            t0 = time.time()
            proj = M.price(model, x, s)
            t_proj = time.time() - t0
            jump = None if model == "dh" else x[10:13]
            cosp = np.empty(s.n)
            t0 = time.time()
            for e in range(len(s.T)):
                q = np.flatnonzero(s.e == e)
                F, T = s.F[e], s.T[e]
                DF = np.exp(-s.r * T)
                for kind, sel in (("call", s.is_call[q]), ("put", ~s.is_call[q])):
                    if sel.any():
                        cosp[q[sel]] = cos.cos_price(x[:10], F, s.K[q[sel]], T, DF, kind, jump=jump)
            t_cos = time.time() - t0
            F_q, T_q = s.F[s.e], s.T[s.e]
            with np.errstate(all="ignore"):
                ivp = D.black_iv(np.maximum(proj, 1e-12), F_q, s.K, T_q, s.r, s.is_call)
                ivc = D.black_iv(np.maximum(cosp, 1e-12), F_q, s.K, T_q, s.r, s.is_call)
            ok = (cosp >= tick) & np.isfinite(ivp) & np.isfinite(ivc)
            # Lewis on a thin subset (it is about 40x slower than COS)
            sub = np.unique(np.round(np.linspace(0, s.n - 1, 12)).astype(int))
            lew = []
            for i in sub:
                e = s.e[i]
                lew.append(float(lewis.lewis_price(x[:10], s.F[e], np.array([s.K[i]]), s.T[e], np.exp(-s.r * s.T[e]), "call" if s.is_call[i] else "put", jump=jump)[0]))
            lew = np.array(lew)
            rows.append({"day": str(d), "model": model, "quotes": s.n, "tradeable_quotes": int(ok.sum()),
                         "max_abs_price_diff_project_vs_cos_rs": float(np.max(np.abs(proj - cosp))),
                         "max_abs_iv_diff_project_vs_cos_vol_pts": float(np.max(100 * np.abs(ivp[ok] - ivc[ok]))) if ok.any() else np.nan,
                         "lewis_points": int(sub.size), "max_abs_price_diff_lewis_vs_cos_rs": float(np.max(np.abs(lew - cosp[sub]))),
                         "project_seconds": t_proj, "cos_seconds": t_cos})
    demo_csv(pd.DataFrame(rows), _paths(cfg, root)[2] / "pricer_cross_check.csv")
    return {"rows": len(rows)}


def q_pinn(cfg, root=ROOT, workers=2):
    import torch
    from . import calib, data as D, models as M, struct as ST
    from .pinn.data import eta_to_theta
    from .pinn.surrogate import Surrogate
    torch.set_num_threads(workers)
    S = _surfaces(cfg)
    qc = cfg["quick"]
    W = _load_walks(cfg, root)
    picks, byday = _pick_days(cfg, S, W, qc["pinn_days"])
    ck = Path(root) / cfg["paths"]["pinn_checkpoints"]
    nets = {"data-only (final.pt)": Surrogate.load(ck / "final.pt", device="cpu"), "hybrid (final_greeks.pt)": Surrogate.load(ck / "final_greeks.pt", device="cpu")}
    recs = {r["day"]: r for r in W["dh"]}
    rows = []
    for d in picks:
        s = S[byday[d]]
        r = recs[d]
        x = calib.to_phys("dh", ST.assemble("dh", r["eta"], r["state"]))
        t0 = time.time()
        exact = M.price("dh", x, s)
        t_exact = time.time() - t0
        with np.errstate(all="ignore"):
            iv_exact = D.black_iv(np.maximum(exact, 1e-12), s.F[s.e], s.K, s.T[s.e], s.r, s.is_call)
        theta = eta_to_theta(r["eta"])
        v1, v2 = r["state"]
        for name, net in nets.items():
            t0 = time.time()
            iv = np.full(s.n, np.nan)
            dom = np.zeros(s.n, bool)
            for e in range(len(s.T)):
                q = np.flatnonzero(s.e == e)
                F, T = s.F[e], s.T[e]
                o = net.price_surface(np.full(q.size, F), s.K[q], np.full(q.size, T), np.full(q.size, np.exp(-s.r * T)), theta, v1, v2, s.is_call[q])
                iv[q] = o["iv"]
                dom[q] = o["in_domain"]
            t_sur = time.time() - t0
            ok = dom & np.isfinite(iv) & np.isfinite(iv_exact) & (exact >= 0.05)
            err = 100 * np.abs(iv[ok] - iv_exact[ok])
            rows.append({"day": str(d), "network": name, "quotes": s.n, "in_domain_quotes": int(dom.sum()), "scored_quotes": int(ok.sum()),
                         "mean_abs_iv_err_vol_pts": float(err.mean()) if err.size else np.nan, "max_abs_iv_err_vol_pts": float(err.max()) if err.size else np.nan,
                         "surrogate_seconds_cpu": t_sur, "exact_pricer_seconds": t_exact})
    demo_csv(pd.DataFrame(rows), _paths(cfg, root)[2] / "pinn_surrogate_check.csv")
    return {"rows": len(rows)}


def q_svi(cfg, root=ROOT, workers=2):
    from . import svi
    S = _surfaces(cfg)
    qc = cfg["quick"]
    W = _load_walks(cfg, root)
    picks, byday = _pick_days(cfg, S, W, qc["svi_days"])
    recs = {m: {r["day"]: r for r in W[m]} for m in W}
    rows = []
    for d in picks:
        s = S[byday[d]]
        fit, fr = svi.fit_surface(s)
        ok = np.isfinite(fit)
        e_svi = 100.0 * (fit - s.iv)
        row = {"day": str(d), "quotes": s.n, "quotes_in_svi_expiries": int(ok.sum()), "expiries_fitted": len(fr), "svi_rmse_vol_pts": float(np.sqrt(np.mean(e_svi[ok] ** 2)))}
        for m in W:
            e = svi.dh_errors(m, recs[m][d], s)
            row[f"{m}_rmse_same_quotes_vol_pts"] = float(np.sqrt(np.mean(e[ok] ** 2)))
        rows.append(row)
    df = pd.DataFrame(rows)
    demo_csv(df, _paths(cfg, root)[2] / "svi_reference.csv")
    return {"days": len(df)}


def q_stats(cfg, root=ROOT, workers=2):
    from . import stats as ST_
    tabs = _paths(cfg, root)[2]
    d = read_demo_csv(tabs / "forecast_daily_rmse.csv")
    names = {"dhj:carry_res": "jump model + misfit", "dh:carry_res": "diffusion DH + misfit", "heston:carry_res": "Heston + misfit", "smile_scaled": "time-scaled smile", "smile": "yesterday's smile"}
    for k in names:
        if k not in d.columns:
            raise RuntimeError(f"quick forecast table has no column {k!r}")
    loss = {v: d[k].to_numpy(float) for k, v in names.items()}
    df = ST_.compare(loss, list(names.values()), "daily IV RMSE, all quotes (demo-scale days)")
    demo_csv(df, tabs / "tests_quick.csv")
    return {"days": len(d)}


# =================================================================================================================== report
def _fmt_p(p):
    from .stats import _fmt_p as f
    return f(p)


def _table_md(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    """cols: (column in df, header, format)"""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c, _, f in cols:
            v = r[c]
            cells.append("" if (isinstance(v, float) and np.isnan(v)) else (f(v) if callable(f) else (f.format(v) if f else str(v))))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def q_report(cfg, root=ROOT, workers=2, timings: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plotstyle as PS
    from .report import Metrics
    q, work, tabs, figs = _paths(cfg, root)
    qc = cfg["quick"]
    fm = read_demo_csv(tabs / "forecast_models.csv")
    st = read_demo_csv(tabs / "tests_quick.csv")
    pc = read_demo_csv(tabs / "pricer_cross_check.csv")
    pn = read_demo_csv(tabs / "pinn_surrogate_check.csv")
    sv = read_demo_csv(tabs / "svi_reference.csv")
    wk = read_demo_csv(tabs / "quick_walk_forward.csv")
    win = read_demo_csv(tabs / "quick_window.csv")
    fc = read_demo_csv(tabs / "forecast_daily_rmse.csv")

    # ---- metrics (same collector as the full run: every number read from a CSV of results_quick/ with its column)
    m = Metrics(q, comment="#")
    rel = "tables/forecast_models.csv"
    for _, r in fm.iterrows():
        for col, unit in (("mean", "vol points"), ("ci_lo", "vol points"), ("ci_hi", "vol points"), ("n_days", "days")):
            m.cell(f"quick.forecast.{r.series.replace(':', '.')}.{col}", rel, col, unit, {"series": r.series})
        if r.kind == "model":
            for col in ("diff_vs_best_benchmark", "diff_lo", "diff_hi"):
                m.cell(f"quick.forecast.{r.series.replace(':', '.')}.{col}", rel, col, "vol points", {"series": r.series})
    for col, unit in (("max_abs_iv_diff_project_vs_cos_vol_pts", "vol points"), ("max_abs_price_diff_project_vs_cos_rs", "rupees"), ("max_abs_price_diff_lewis_vs_cos_rs", "rupees")):
        for model in ("dh", "dhj"):
            sub = pc[pc.model == model]
            m.derived(f"quick.pricer.{model}.{col}", float(sub[col].max()), unit, ["tables/pricer_cross_check.csv"], f"max over the {len(sub)} days (rows with model == {model!r}) of column {col!r}")
    for net in pn.network.unique():
        sub = pn[pn.network == net]
        key = "data_only" if net.startswith("data") else "hybrid"
        m.derived(f"quick.pinn.{key}.mean_abs_iv_err_vol_pts", float(sub.mean_abs_iv_err_vol_pts.mean()), "vol points", ["tables/pinn_surrogate_check.csv"],
                  f"mean over days of column 'mean_abs_iv_err_vol_pts' (rows with network == {net!r})")
        m.derived(f"quick.pinn.{key}.in_domain_share", float(sub.in_domain_quotes.sum() / sub.quotes.sum()), "share", ["tables/pinn_surrogate_check.csv"], "sum(in_domain_quotes) / sum(quotes)")
    m.derived("quick.svi.svi_rmse_mean", float(sv.svi_rmse_vol_pts.mean()), "vol points", ["tables/svi_reference.csv"], "mean over days of 'svi_rmse_vol_pts'")
    for mod in qc["models"]:
        m.derived(f"quick.svi.{mod}_rmse_mean", float(sv[f"{mod}_rmse_same_quotes_vol_pts"].mean()), "vol points", ["tables/svi_reference.csv"], f"mean over days of '{mod}_rmse_same_quotes_vol_pts'")
    m.derived("quick.window.days", int(len(win)), "days", ["tables/quick_window.csv"], "number of rows")
    m.derived("quick.forecast_days", int(len(fc)), "days", ["tables/forecast_daily_rmse.csv"], "number of rows")
    js = m.to_json("configs/nifty.yaml")
    js["notice"] = f"{NOTICE}. {LONG_NOTICE}"
    (q / "metrics_quick.json").write_text(json.dumps(js, indent=1, ensure_ascii=False))
    v = m.v

    # ---- figure: forecast bars with intervals + one real smile
    PS.setup()
    fig, axs = plt.subplots(1, 2, figsize=(cfg["figures"]["size_inches"][0], cfg["figures"]["size_inches"][1] - 0.3), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    order = ["smile", "smile_scaled"] + [f"{mm}:{vv}" for mm in qc["models"] for vv in ("carry", "carry_res")]
    cols = {"smile": "#a9a7a1", "smile_scaled": PS.INK2, "heston": PS.PAL[3], "dh": PS.PAL[2], "dhj": PS.PAL[1]}
    ys = np.arange(len(order))[::-1]
    for yi, k in zip(ys, order):
        r = fm[fm.series == k].iloc[0]
        c = cols[k.split(":")[0]]
        pure = k.endswith(":carry")
        ax.barh(yi, r["mean"], color=c, alpha=0.55 if pure else 1.0, hatch="//" if pure else None, edgecolor="white", height=0.68)
        ax.errorbar(r["mean"], yi, xerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]], fmt="none", ecolor=PS.INK, capsize=5, lw=1.8)
        ax.text(r.ci_hi + 0.05, yi, f"{r['mean']:.2f}", va="center", fontsize=13)
    ax.set_yticks(ys)
    ax.set_yticklabels([fm[fm.series == k].iloc[0].label for k in order], fontsize=13)
    ax.set_xlabel("Mean next-day error on the demo window (vol points)", fontsize=14)
    ax.set_title("Next-day forecast error, demo window", fontsize=17)
    ax.grid(axis="y", visible=False)
    # smile of the last quick day
    from . import calib, models as M_, struct as ST, svi
    S = _surfaces(cfg)
    W = _load_walks(cfg, root)
    byday = {s.day: i for i, s in enumerate(S)}
    d = max(r["day"] for r in W["dh"])
    s = S[byday[d]]
    fit, _ = svi.fit_surface(s)
    e = int(np.argmin(np.abs(s.T * 365 - 30)))
    qi = np.flatnonzero(s.e == e)
    qi = qi[np.argsort(s.K[qi])]
    k = np.log(s.K[qi] / s.F[e])
    ax2 = axs[1]
    ax2.scatter(k, 100 * s.iv[qi], s=34, color=PS.INK, label="market quotes", zorder=4)
    for mod, lab, c in (("dh", "Double Heston (mini fit)", PS.PAL[2]), ("dhj", "jump model (mini fit)", PS.PAL[1])):
        if mod in W:
            rr = next(r for r in W[mod] if r["day"] == d)
            iv = M_.model_iv(M_.price(mod, calib.to_phys(mod, ST.assemble(mod, rr["eta"], rr["state"])), s), s)
            ax2.plot(k, 100 * iv[qi], color=c, lw=2.6, label=lab)
    if np.isfinite(fit[qi]).all():
        ax2.plot(k, 100 * fit[qi], color=PS.PAL[0], lw=2.2, ls="--", label="SVI")
    ax2.set_xlabel("Forward log-moneyness ln(K/F)", fontsize=14)
    ax2.set_ylabel("Implied volatility (%)", fontsize=14)
    ax2.set_title(f"{d}, {int(round(s.T[e] * 365))}-day expiry", fontsize=17)
    ax2.legend(fontsize=12)
    fig.suptitle(f"{NOTICE}: short window, small budgets", fontsize=19, color="#b3261e", fontweight="bold", x=0.01, ha="left")
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"quick_summary.{ext}", dpi=cfg["figures"]["dpi"], facecolor=fig.get_facecolor())
    plt.close(fig)

    # ---- report.md (same layout as results/report.md)
    best = fm.best_benchmark.iloc[0]
    jr = fm[fm.series == "dhj:carry_res"].iloc[0]
    tm = timings or {}
    L = [f"# Double Heston 2.0 on NIFTY 50: quick-run report", "",
         f"> **{NOTICE}.** {LONG_NOTICE}", "",
         f"*Window {qc['start']} to {qc['end']} ({v('quick.window.days')} real trading days, {v('quick.forecast_days')} scored forecast days); mini walk-forward: window {qc['window']} days, refit every {qc['refit']} days, "
         f"burn-in {qc['burn']}, {qc['per_expiry']} quotes per expiry, {qc['nfev_cold']}/{qc['nfev_warm']} optimiser evaluations (full run: window {cfg['walk_forward']['window']}, refit {cfg['walk_forward']['refit']}, burn {cfg['walk_forward']['burn']}, "
         f"{cfg['walk_forward']['per_expiry']}, {cfg['walk_forward']['nfev_cold']}/{cfg['walk_forward']['nfev_warm']}). Every number is read from `results_quick/tables/*.csv` (listed with column in `metrics_quick.json`).*", "",
         "## 1. Headline: next-day forecast error (demo window)", "",
         f"On the {int(jr.n_days)} demo days the jump model plus yesterday's misfit has a mean next-day error of {jr['mean']:.3f} vol points; the best model-free benchmark ({'yesterday' + chr(39) + 's smile' if best == 'smile' else 'time-scaled smile'}) "
         f"has {fm[fm.series == best].iloc[0]['mean']:.3f}. Paired difference {jr.diff_vs_best_benchmark:+.3f} [{jr.diff_lo:+.3f}, {jr.diff_hi:+.3f}] (block bootstrap, demo-sized sample: wide and not a result).", "",
         _table_md(fm, [("label", "Forecast", ""), ("kind", "Kind", ""), ("mean", "Demo mean", "{:.3f}"), ("ci_lo", "lo", "{:.3f}"), ("ci_hi", "hi", "{:.3f}"), ("diff_vs_best_benchmark", "Difference vs best benchmark", "{:+.3f}"),
                         ("shipped_full_run_same_days", "Shipped full-run forecast, same days (reference)", "{:.3f}")]), "",
         "## 2. Is the difference real? Paired tests (demo days)", "",
         _table_md(st, [("A", "A", ""), ("B", "B", ""), ("mean_diff_A_minus_B", "A minus B", "{:+.3f}"), ("share_days_A_better", "Days A better", "{:.0%}"), ("wilcoxon_p", "Wilcoxon p", _fmt_p), ("dm_p", "DM p (HAC)", _fmt_p)]), "",
         "## 3. From the 1.0 recipe to 2.0", "", "Not part of the quick run (needs the independent daily fits of the 1.0 recipe on 118 days); see `results/attribution/`.", "",
         "## 4. SVI yardstick (demo days)", "",
         _table_md(sv, [("day", "Day", ""), ("quotes", "Quotes", "{:.0f}"), ("svi_rmse_vol_pts", "SVI", "{:.3f}")] + [(f"{mm}_rmse_same_quotes_vol_pts", MODEL_LABEL[mm], "{:.3f}") for mm in qc["models"]]),
         "Same-day implied-vol error, vol points, on the quotes of the expiries SVI fitted.", "",
         "## 5. Pricer cross-check (project pricer against the independent COS pricer, real chains)", "",
         _table_md(pc, [("day", "Day", ""), ("model", "Model", ""), ("tradeable_quotes", "Quotes >= Rs 0.05", "{:.0f}"), ("max_abs_iv_diff_project_vs_cos_vol_pts", "Max IV difference (vol pts)", "{:.2e}"),
                        ("max_abs_price_diff_project_vs_cos_rs", "Max price difference (Rs)", "{:.2e}"), ("max_abs_price_diff_lewis_vs_cos_rs", f"Lewis vs COS on a thin subset (Rs)", "{:.2e}"),
                        ("project_seconds", "Project pricer (s)", "{:.3f}"), ("cos_seconds", "COS (s)", "{:.3f}")]), "",
         "## 6. PINN surrogate (shipped checkpoints, real chains of the demo days)", "",
         _table_md(pn, [("day", "Day", ""), ("network", "Network", ""), ("in_domain_quotes", "In-domain quotes", "{:.0f}"), ("quotes", "Quotes", "{:.0f}"), ("mean_abs_iv_err_vol_pts", "Mean IV error (vol pts)", "{:.3f}"),
                        ("max_abs_iv_err_vol_pts", "Max IV error (vol pts)", "{:.3f}"), ("surrogate_seconds_cpu", "Surrogate (s, CPU)", "{:.3f}"), ("exact_pricer_seconds", "Exact pricer (s)", "{:.3f}")]),
         "Scored on in-domain quotes only (the surrogate is valid inside its training box); the mini calibration's parameters can lie outside it, which the in-domain column shows.", "",
         "## 7. PINN-accelerated calibration", "", "Not part of the quick run; see `results/pinn_calib/`.", "",
         "## 8. Link to India VIX", "", "Not part of the quick run; see `results/vix/`.", "",
         "## 9. Data quality and no-arbitrage", "", "Not part of the quick run; see `results/dq/`.", "",
         "## 10. Trading test", "", "Not part of the quick run; see `data/results/trade_*.json` and `results/tables/trading_summary.md`.", "",
         "## 11. Robustness and pattern memory", "", "Not part of the quick run; see `results/tables/robustness_memory.md`.", "",
         "## Timing of this run", "",
         _table_md(wk, [("model", "Model", ""), ("forecast_days", "Forecast days", "{:.0f}"), ("worker_seconds", "Walk-forward worker seconds", "{:.0f}")]), ""]
    if tm:
        L += ["Stage wall times: " + ", ".join(f"{k} {v_:.0f} s" for k, v_ in tm.items()) + ".", ""]
    L += ["> " + NOTICE + ".", ""]
    (q / "report.md").write_text("\n".join(L))
    (q / "README.md").write_text(f"# results_quick\n\n**{NOTICE}.** {LONG_NOTICE}\n\nWritten by `python -m dh2.run_all --quick`; `report.md` is the entry point, `tables/` holds every number, `work/` the mini walk-forward pickles, "
                                 "`figures/` one figure. Nothing here overwrites `results/`.\n")
    return {"metrics": len(m.items)}
