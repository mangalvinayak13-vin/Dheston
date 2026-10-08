"""Work package D (brief Phase 5): PINN-accelerated calibration of the diffusion Double Heston against the pure exact-pricer calibration.

    python3 exp_pinn_calib.py design    # DEVELOPMENT days only (every 25th dev day): choose starts / surrogate budget / polish budget, freeze them
    python3 exp_pinn_calib.py test      # the 118 paired test days (results/svi/sample_days.csv): run the frozen method once, time everything
    python3 exp_pinn_calib.py report    # tables, figures, headline.json  (from the cached per-day results; nothing is recomputed)

One process, torch on the CPU with 2 threads (numpy BLAS 1 thread).  Resumable: every day is cached in results/pinn_calib/_cache/.
Progress: logs/progress/pinncal_00.json  {"done": n, "total": m}  (design days + test days).

Rules kept: design choices use days before 2022-01-01 only (asserted); the test stage refuses to start without results/pinn_calib/frozen_config.json,
which the design stage writes; every fit metric is computed by the EXACT pricer (dh2.calib_pinn.score); nothing in results/_cache is touched.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import pickle
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dh2 import calib, calib_pinn as CP, models as M          # noqa: E402
from dh2.data import load_all                                 # noqa: E402
from dh2.pinn.surrogate import Surrogate                      # noqa: E402

SPLIT = date(2022, 1, 1)
OUT = Path(os.environ.get("PINNCAL_OUT", ROOT / "results" / "pinn_calib"))       # PINNCAL_OUT / PINNCAL_NDAYS: smoke tests in a scratch folder
CACHE = OUT / "_cache"
_NSMOKE = int(os.environ.get("PINNCAL_NDAYS", "0"))
PROG = ROOT / "logs" / "progress" / "pinncal_00.json"
C2_CACHE = ROOT / "results" / "_cache" / "daily_fits_v1.pkl"      # read only
SAMPLE_CSV = ROOT / "results" / "svi" / "sample_days.csv"
N_DESIGN_DAYS, N_TEST_DAYS = 52, 118
LO, HI = (np.array(b, float) for b in calib.BOUNDS["dh"])
WIDTH = HI - LO
LOG = CP.LOG_PARAM
SCALE = np.where(LOG, 0.7, 0.3)                                # struct._KIND scales: log-parameters 0.7, correlations 0.3

GLOBAL_CONFIGS = {
    "A_32": CP.PinnConfig(n_starts=32, iters=(8, 20), keep=(8,)),
    "B_64": CP.PinnConfig(n_starts=64, iters=(6, 12, 20), keep=(16, 8)),
    "C_128": CP.PinnConfig(n_starts=128, iters=(6, 10, 20), keep=(32, 8)),
    "D_256": CP.PinnConfig(n_starts=256, iters=(6, 10, 25), keep=(64, 8)),
    "E_128_box": CP.PinnConfig(n_starts=128, iters=(6, 10, 20), keep=(32, 8), start_mode="box"),
    "F_128_long": CP.PinnConfig(n_starts=128, iters=(6, 10, 60), keep=(32, 8)),
}
FIXED_POLISH = dict(n_polish=2, polish_nfev=20)                # used to compare the global configurations
VARIANTS = {"plain": dict(rerank=False, n_correct=0),          # polish the surrogate's own best candidates
            "rerank": dict(rerank=True, n_correct=0),          # rank the 8 survivors by their exact cost first (8 exact evaluations)
            "rerank_corr2": dict(rerank=True, n_correct=2)}    # ... and refine them by two rounds of defect correction first
POLISH_K, POLISH_B = (1, 2, 3), (5, 10, 20, 40)
RULE_TOL = 0.01                                                # "within 1% (mean log cost ratio) of the best"


# ------------------------------------------------------------------ small helpers
def progress(done: int, total: int) -> None:
    if _NSMOKE:
        return
    PROG.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done": int(done), "total": int(total)}))
    tmp.replace(PROG)


def load_cache(name: str) -> dict:
    p = CACHE / name
    return pickle.load(open(p, "rb")) if p.exists() else {}


def save_cache(obj: dict, name: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(p)


def load_world():
    S = load_all()
    dev = [i for i, s in enumerate(S) if s.day < SPLIT]
    test = [i for i, s in enumerate(S) if s.day >= SPLIT][::10]
    return S, dev[::25], test


def machine_load() -> dict:
    """1-minute load average and the CPU share of every other python process above 5% (what else was running)."""
    la = os.getloadavg()[0]
    try:
        ps = subprocess.run(["ps", "-Ao", "pid,pcpu,command"], capture_output=True, text=True, timeout=10).stdout.splitlines()[1:]
        busy = [float(l.split(None, 2)[1]) for l in ps if "python" in l.lower() and float(l.split(None, 2)[1]) > 5.0 and str(os.getpid()) != l.split(None, 1)[0]]
    except Exception:
        busy = []
    return {"loadavg1": float(la), "other_python_procs_over_5pct": len(busy), "other_python_cpu_pct": float(sum(busy))}


def params_row(u) -> dict:
    return {n: float(v) for n, v in zip(CP.NAMES, u)}


def to_scaled(u) -> np.ndarray:
    """Parameters on the scale of struct.py's prior: log for the positive ones, linear for the correlations, divided by 0.7 / 0.3."""
    u = np.asarray(u, float)
    return np.where(LOG, np.log(np.maximum(u, 1e-12)), u) / SCALE


def dist_scaled(a, b) -> float:
    return float(np.sqrt(np.mean((to_scaled(a) - to_scaled(b)) ** 2)))


def dist_frac(a, b) -> float:
    """C2's identifiability measure: the largest parameter difference as a share of that parameter's bound-interval width."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / WIDTH))


def surface_gap(S, ua, ub) -> float:
    """RMS difference (vol points) between the two fitted implied-vol surfaces over all quotes."""
    ia = M.model_iv(M.price("dh", calib.to_phys("dh", ua), S), S)
    ib = M.model_iv(M.price("dh", calib.to_phys("dh", ub), S), S)
    return float(100.0 * np.sqrt(np.mean((ia - ib) ** 2)))


def warm_up(sur, S) -> None:
    """One throw-away accelerated fit so that thread pools and allocator caches are warm before anything is timed."""
    CP.fit_day_pinn(sur, S, replace(CP.DEFAULT, polish_nfev=2, n_polish=1, n_starts=16, iters=(2, 2), keep=(4,)))


# ------------------------------------------------------------------ design stage (development days only)
def design() -> None:
    S, dsgn, _ = load_world()
    assert len(dsgn) == N_DESIGN_DAYS and all(S[i].day < SPLIT for i in dsgn), "design days must all be development-period days"
    if _NSMOKE:
        dsgn = dsgn[:_NSMOKE]
    OUT.mkdir(parents=True, exist_ok=True)
    sur = Surrogate.load(device="cpu")
    warm_up(sur, S[dsgn[0]])
    total = N_DESIGN_DAYS + N_TEST_DAYS
    base = load_cache("design_exact.pkl")
    glob = load_cache("design_global.pkl")
    pol = load_cache("design_polish.pkl")
    print(f"design: {len(dsgn)} development days {S[dsgn[0]].day} .. {S[dsgn[-1]].day}", flush=True)
    t_start = time.time()
    for n, i in enumerate(dsgn):
        s = S[i]
        idx = M.thin(s, 14)
        if i not in base:
            f = CP.fit_day_exact(s, 4, seed=i)
            sc = CP.score(f["u"], s, idx)
            base[i] = {"u": f["u"], "cost": f["cost"], "rmse": sc["rmse"], "calls": f["n_exact_calls"], "sec": f["t_total"], "cpu": f["cpu_total"],
                       "n": s.n, "n_exp": len(s.T)}
            save_cache(base, "design_exact.pkl")
        # global stage for every configuration, then the fixed polish
        for name, cfg in GLOBAL_CONFIGS.items():
            key = (i, name)
            if key in glob:
                continue
            t0 = time.perf_counter()
            g = CP.global_search(sur, s, idx, cfg)
            t_g = time.perf_counter() - t0
            width = HI - LO
            pick = CP._distinct(g["U"], np.arange(len(g["U"])), 3, width, cfg.min_sep)
            sc = CP.score(g["U"][0], s, idx)
            ex_surv = [CP.exact_cost(g["U"][k], s, idx) for k in range(min(4, len(g["U"])))]
            # the fixed polish (2 distinct candidates, 20 nfev) from this configuration's survivors
            t1 = time.perf_counter()
            cands = [CP.polish(s, idx, g["U"][k], FIXED_POLISH["polish_nfev"]) for k in pick[:FIXED_POLISH["n_polish"]]]
            t_p = time.perf_counter() - t1
            best = min(cands, key=lambda c: c["cost"])
            glob[key] = {"t_global": t_g, "t_polish": t_p, "sur_evals": g["obj"].calls, "n_excluded": g["n_excluded"], "U": g["U"], "sur_cost": g["cost"],
                         "pick": pick, "sur_exact_cost": sc["cost"], "sur_rmse": sc["rmse"], "ex_cost_surv": ex_surv,
                         "polished_cost": best["cost"], "polished_u": best["u"], "polished_rmse": CP.score(best["u"], s, idx)["rmse"],
                         "calls": int(sum(c["n_calls"] for c in cands))}
            save_cache(glob, "design_global.pkl")
        progress(n + 1, total)
        print(f"  design day {n + 1}/{len(dsgn)} {s.day} done  ({time.time() - t_start:.0f}s)", flush=True)
    # ---- choose the global configuration (rule 1)
    rows = []
    lb = np.array([np.log(base[i]["cost"]) for i in dsgn])
    for name, cfg in GLOBAL_CONFIGS.items():
        g = [glob[(i, name)] for i in dsgn]
        rows.append({
            "config": name, "n_starts": cfg.n_starts, "iters": str(cfg.iters), "keep": str(cfg.keep), "start_mode": cfg.start_mode,
            "mean_t_global_s": np.mean([x["t_global"] for x in g]), "mean_sur_evals": np.mean([x["sur_evals"] for x in g]),
            "sur_only_mean_log_cost_ratio": float(np.mean([np.log(x["sur_exact_cost"]) for x in g]) - lb.mean()),
            "sur_only_mean_rmse": np.mean([x["sur_rmse"] for x in g]),
            "sur_only_share_cost_within_5pct_of_exact_baseline": float(np.mean([x["sur_exact_cost"] <= 1.05 * base[i]["cost"] for x, i in zip(g, dsgn)])),
            "polished_mean_log_cost_ratio": float(np.mean([np.log(x["polished_cost"]) for x in g]) - lb.mean()),
            "polished_mean_rmse": np.mean([x["polished_rmse"] for x in g]),
            "polished_share_cost_within_1pct_of_exact_baseline": float(np.mean([x["polished_cost"] <= 1.01 * base[i]["cost"] for x, i in zip(g, dsgn)])),
            "polished_mean_exact_calls": np.mean([x["calls"] for x in g]),
            "mean_t_total_s": np.mean([x["t_global"] + x["t_polish"] for x in g]),
            "share_top1_is_best_exact_among_top4": float(np.mean([int(np.argmin(x["ex_cost_surv"])) == 0 for x in g])),
        })
    gt = pd.DataFrame(rows)
    gt["exact_baseline_mean_rmse"] = np.mean([base[i]["rmse"] for i in dsgn])
    gt["exact_baseline_mean_sec"] = np.mean([base[i]["sec"] for i in dsgn])
    gt["exact_baseline_mean_calls"] = np.mean([base[i]["calls"] for i in dsgn])
    best_ratio = gt["polished_mean_log_cost_ratio"].min()
    ok = gt[gt["polished_mean_log_cost_ratio"] <= best_ratio + RULE_TOL]
    chosen_name = ok.sort_values("mean_t_total_s").iloc[0]["config"]
    gt["chosen"] = gt["config"] == chosen_name
    gt.to_csv(OUT / "design_global_configs.csv", index=False)
    cfg0 = GLOBAL_CONFIGS[chosen_name]
    print("chosen global configuration:", chosen_name, flush=True)
    # ---- variants x polish grid on the chosen configuration (rule 2): plain / exact re-ranking / re-ranking + 2 defect-correction rounds
    width = HI - LO
    vgrid = load_cache("design_variants.pkl")
    t_start = time.time()
    for n, i in enumerate(dsgn):
        s = S[i]
        idx = M.thin(s, 14)
        for vname, vkw in VARIANTS.items():
            if (i, vname) in vgrid:
                continue
            cfg_v = replace(cfg0, **vkw)
            p = CP.propose(sur, s, idx, cfg_v)
            pick = CP._distinct(p["U"], np.arange(len(p["U"])), max(POLISH_K), width, cfg_v.min_sep)
            rec = {"U": p["U"][pick], "t_global": p["t_global"], "t_correct": p["t_correct"], "n_correct_calls": p["n_correct_calls"], "sur_evals": p["n_sur_evals"],
                   "corr_cost": p["corr_cost"], "polish": {}}
            for c, u0 in enumerate(rec["U"]):
                for b in POLISH_B:
                    t0 = time.perf_counter()
                    q = CP.polish(s, idx, u0, b)
                    rec["polish"][(c, b)] = {"cost": q["cost"], "calls": q["n_calls"], "u": q["u"], "nfev": q["nfev"], "sec": time.perf_counter() - t0}
            vgrid[(i, vname)] = rec
        save_cache(vgrid, "design_variants.pkl")
        print(f"  variant grid day {n + 1}/{len(dsgn)} {s.day}  ({time.time() - t_start:.0f}s)", flush=True)
    rows = []
    for vname in VARIANTS:
        for kk in POLISH_K:
            for b in POLISH_B:
                costs, calls, secs, rmses = [], [], [], []
                for i in dsgn:
                    rec = vgrid[(i, vname)]
                    m = min(kk, len(rec["U"]))
                    cs = [rec["polish"][(c, b)] for c in range(m)]
                    best = min(cs, key=lambda q: q["cost"])
                    costs.append(best["cost"])
                    calls.append(rec["n_correct_calls"] + sum(q["calls"] for q in cs))
                    secs.append(rec["t_global"] + rec["t_correct"] + sum(q["sec"] for q in cs))
                    rmses.append(CP.score(best["u"], S[i])["rmse"])
                rows.append({"variant": vname, "n_polish": kk, "polish_nfev": b, "mean_log_cost_ratio": float(np.mean(np.log(costs)) - lb.mean()), "mean_rmse": float(np.mean(rmses)),
                             "mean_exact_calls": float(np.mean(calls)), "mean_sec": float(np.mean(secs)),
                             "share_cost_within_1pct_of_exact_baseline": float(np.mean([c <= 1.01 * base[i]["cost"] for c, i in zip(costs, dsgn)])),
                             "share_cost_worse_than_5pct": float(np.mean([c > 1.05 * base[i]["cost"] for c, i in zip(costs, dsgn)]))})
    pt = pd.DataFrame(rows)
    best_ratio = pt["mean_log_cost_ratio"].min()
    ok = pt[pt["mean_log_cost_ratio"] <= best_ratio + RULE_TOL]
    pick = ok.sort_values("mean_sec").iloc[0]
    pt["chosen"] = (pt["variant"] == pick["variant"]) & (pt["n_polish"] == pick["n_polish"]) & (pt["polish_nfev"] == pick["polish_nfev"])
    pt["exact_baseline_mean_rmse"] = np.mean([base[i]["rmse"] for i in dsgn])
    pt["exact_baseline_mean_calls"] = np.mean([base[i]["calls"] for i in dsgn])
    pt["exact_baseline_mean_sec"] = np.mean([base[i]["sec"] for i in dsgn])
    pt.to_csv(OUT / "design_variant_polish_grid.csv", index=False)
    frozen = replace(cfg0, n_polish=int(pick["n_polish"]), polish_nfev=int(pick["polish_nfev"]), **VARIANTS[pick["variant"]])
    out = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(frozen).items()}, "global_config_name": chosen_name, "variant": pick["variant"],
           "rule": f"global configuration: cheapest mean wall time with mean log exact-cost ratio (after the fixed polish {FIXED_POLISH}) within {RULE_TOL} of the best configuration; "
                   f"variant / number of polished candidates / polish budget: cheapest mean wall time with mean log cost ratio within {RULE_TOL} of the best cell of the variant x polish grid",
           "design_days": [S[i].day.isoformat() for i in dsgn], "n_design_days": len(dsgn)}
    (OUT / "frozen_config.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame([{"day": S[i].day, "index": i, **{f"exact_{k}": base[i][k] for k in ("cost", "rmse", "calls", "sec", "n", "n_exp")},
                   **{f"{name}_{k}": glob[(i, name)][k] for name in GLOBAL_CONFIGS for k in ("sur_exact_cost", "sur_rmse", "polished_cost", "polished_rmse", "t_global", "calls")}}
                  for i in dsgn]).to_csv(OUT / "design_per_day.csv", index=False)
    print("frozen:", frozen, flush=True)


# ------------------------------------------------------------------ test stage (frozen method, once)
def load_frozen() -> CP.PinnConfig:
    p = OUT / "frozen_config.json"
    if not p.exists():
        raise SystemExit("results/pinn_calib/frozen_config.json is missing: run the design stage (development days) first")
    c = json.loads(p.read_text())["config"]
    c["iters"], c["keep"] = tuple(c["iters"]), tuple(c["keep"])
    return CP.PinnConfig(**c)


def test() -> None:
    cfg = load_frozen()
    S, _, tst = load_world()
    sd = pd.read_csv(SAMPLE_CSV)["day"].tolist()
    assert [S[i].day.isoformat() for i in tst] == sd and len(tst) == N_TEST_DAYS, "test sample differs from results/svi/sample_days.csv"
    c2 = pickle.load(open(C2_CACHE, "rb"))
    sur = Surrogate.load(device="cpu")
    warm_up(sur, S[tst[0]])
    res = load_cache("test_days.pkl")
    print(f"test: {len(tst)} days, frozen config {cfg}", flush=True)
    t_start = time.time()
    for n, i in enumerate(tst):
        if i not in res:
            s = S[i]
            idx = M.thin(s, 14)
            order = [0, 1] if n % 2 == 0 else [1, 0]          # alternate which method runs first so a slow minute does not favour one
            r: dict = {"index": i, "day": s.day.isoformat(), "n": s.n, "n_thin": len(idx), "n_exp": len(s.T), "load_before": machine_load()}
            for which in order:
                if which == 0:
                    r["exact"] = CP.fit_day_exact(s, 4, seed=i)
                else:
                    r["accel"] = CP.fit_day_pinn(sur, s, cfg)
            # control: exact polish from the Phase-3 starts with the same polish budget (no surrogate)
            rng = np.random.default_rng(i)
            st = np.array(calib.starts("dh", s, rng, 4))[:cfg.n_polish]
            r["matched"] = CP.fit_day_pinn(sur, s, cfg, polish_starts=st)
            if cfg.rerank or cfg.n_correct:                      # ablation: the same method without the exact re-ranking / correction
                r["plain"] = CP.fit_day_pinn(sur, s, replace(cfg, rerank=False, n_correct=0))
            r["load_after"] = machine_load()
            # scoring (exact pricer, never timed)
            for k in ("exact", "accel", "matched", "plain"):
                if k in r:
                    r[k]["score"] = CP.score(r[k]["u"], s, idx)
            r["sur_only"] = CP.score(r["accel"]["u_sur"], s, idx)
            ce = c2[("dh", i)]
            r["c2"] = {"cost": ce["cost"], "rmse": ce["rmse_in"], "sec": ce["sec"], "u": ce["u"]}
            r["gap_accel_exact"] = surface_gap(s, r["accel"]["u"], r["exact"]["u"])
            r["gap_sur_exact"] = surface_gap(s, r["accel"]["u_sur"], r["exact"]["u"])
            r["gap_exact_c2"] = surface_gap(s, r["exact"]["u"], ce["u"])
            for k in ("exact", "accel", "matched", "plain"):
                if k in r:
                    r[k].pop("idx", None)
            res[i] = r
            save_cache(res, "test_days.pkl")
            e, a = r["exact"], r["accel"]
            print(f"  {n + 1}/{len(tst)} {s.day}  exact rmse {e['score']['rmse']:.3f} ({e['t_total']:.1f}s, {e['n_exact_calls']} calls)  "
                  f"accel rmse {a['score']['rmse']:.3f} ({a['t_total']:.1f}s, {a['n_exact_calls']} calls)  sur-only {r['sur_only']['rmse']:.3f}  "
                  f"load {r['load_before']['loadavg1']:.1f}  [{time.time() - t_start:.0f}s]", flush=True)
        progress(N_DESIGN_DAYS + n + 1, N_DESIGN_DAYS + N_TEST_DAYS)


# ------------------------------------------------------------------ report stage (tables, figures, headline.json from the cached per-day results)
def boot_ci(v, stat=np.mean, n_boot: int = 10000, seed: int = 0) -> tuple:
    """(estimate, lo, hi): percentile bootstrap 95% interval, resampling days."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, v.size, size=(n_boot, v.size))
    bs = stat(v[ix], axis=1)
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio_of_means(num, den, n_boot: int = 10000, seed: int = 0) -> tuple:
    """Ratio of the sums (total time of A / total time of B), days resampled in pairs."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, num.size, size=(n_boot, num.size))
    bs = num[ix].sum(1) / den[ix].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_day_table(res: dict, sur) -> pd.DataFrame:
    rows = []
    for i in sorted(res):
        r = res[i]
        e, a, m = r["exact"], r["accel"], r["matched"]
        ue, ua, us, um = e["u"], a["u"], a["u_sur"], m["u"]
        row = {
            "day": r["day"], "index": i, "n_quotes": r["n"], "n_fit_quotes": r["n_thin"], "n_expiries": r["n_exp"], "n_quotes_outside_surrogate_box": a["n_excluded"],
            "rmse_exact": e["score"]["rmse"], "rmse_accel": a["score"]["rmse"], "rmse_surrogate_only": r["sur_only"]["rmse"], "rmse_exact_polish_from_phase3_starts": m["score"]["rmse"],
            "rmse_c2_cached_exact": r["c2"]["rmse"],
            "cost_exact": e["score"]["cost"], "cost_accel": a["score"]["cost"], "cost_surrogate_only": r["sur_only"]["cost"], "cost_exact_polish_from_phase3_starts": m["score"]["cost"],
            "cost_c2_cached_exact": r["c2"]["cost"],
            "sec_exact": e["t_total"], "sec_accel": a["t_total"], "sec_accel_surrogate_stage": a["t_global"], "sec_accel_correction": a["t_correct"], "sec_accel_polish": a["t_polish"],
            "sec_exact_polish_from_phase3_starts": m["t_total"], "sec_c2_recorded": r["c2"]["sec"],
            "cpu_exact": e["cpu_total"], "cpu_accel": a["cpu_total"],
            "calls_exact": e["n_exact_calls"], "calls_accel": a["n_exact_calls"], "calls_accel_correction": a["n_correct_calls"], "calls_accel_polish": a["n_polish_calls"],
            "calls_exact_polish_from_phase3_starts": m["n_exact_calls"],
            "nfev_exact_total": int(sum(q["nfev"] for q in e["runs"])), "nfev_accel_total": int(sum(q["nfev"] for q in a["cands"])),
            "polish_budget_exhausted": bool(any(q["status"] == 0 for q in a["cands"])),
            "surrogate_evals_accel": a["n_sur_evals"],
            "exact_solution_inside_box": bool(CP.in_box(ue, sur)[0]), "accel_solution_inside_box": bool(CP.in_box(ua, sur)[0]),
            "n_params_at_bound_exact": len(e["score"]["at_bound"]), "n_params_at_bound_accel": len(a["score"]["at_bound"]),
            "dist_frac_accel_vs_exact": dist_frac(ua, ue), "dist_scaled_accel_vs_exact": dist_scaled(ua, ue),
            "dist_frac_surrogate_only_vs_exact": dist_frac(us, ue), "dist_scaled_surrogate_only_vs_exact": dist_scaled(us, ue),
            "surface_gap_accel_vs_exact_volpts": r["gap_accel_exact"], "surface_gap_surrogate_only_vs_exact_volpts": r["gap_sur_exact"],
            "reproduces_c2_cache_cost_diff": abs(e["score"]["cost"] - r["c2"]["cost"]), "reproduces_c2_cache_surface_gap_volpts": r["gap_exact_c2"],
            "loadavg1_before": r["load_before"]["loadavg1"], "other_python_cpu_pct_before": r["load_before"]["other_python_cpu_pct"],
            "loadavg1_after": r["load_after"]["loadavg1"],
        }
        if "plain" in r:
            p = r["plain"]
            row.update(rmse_accel_without_rerank=p["score"]["rmse"], cost_accel_without_rerank=p["score"]["cost"], sec_accel_without_rerank=p["t_total"],
                       calls_accel_without_rerank=p["n_exact_calls"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["cost_ratio_accel_vs_exact"] = df["cost_accel"] / df["cost_exact"]
    df["cost_ratio_surrogate_only_vs_exact"] = df["cost_surrogate_only"] / df["cost_exact"]
    df["cost_ratio_phase3_starts_vs_exact"] = df["cost_exact_polish_from_phase3_starts"] / df["cost_exact"]
    df["speedup_wall"] = df["sec_exact"] / df["sec_accel"]
    df["speedup_cpu"] = df["cpu_exact"] / df["cpu_accel"]
    df["call_ratio_exact_over_accel"] = df["calls_exact"] / df["calls_accel"]
    return df


def paired_row(name: str, a, b, la: str, lb: str, unit: str) -> dict:
    """Mean difference a - b with a day-resampling 95% interval, medians, Wilcoxon signed-rank p (dh2.stats) and win counts."""
    from dh2 import stats as ST
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    est, lo, hi = boot_ci(d)
    w = ST.wilcoxon_signed_rank(a, b)
    return {"comparison": name, "metric": unit, f"mean_{la}": float(np.mean(a)), f"mean_{lb}": float(np.mean(b)), "mean_diff": est, "ci95_lo": lo, "ci95_hi": hi,
            f"median_{la}": float(np.median(a)), f"median_{lb}": float(np.median(b)), "median_diff": float(np.median(d)),
            "wilcoxon_p": w["pvalue"], "n_days": int(d.size), f"days_{la}_lower": int((d < 0).sum()), f"days_{lb}_lower": int((d > 0).sum()), "days_equal": int((d == 0).sum())}


def make_tables(df: pd.DataFrame, res: dict, cfg: CP.PinnConfig) -> dict:
    h: dict = {}
    # ---- accuracy
    rows = [paired_row("accelerated vs exact", df.rmse_accel, df.rmse_exact, "accel", "exact", "same-day IV RMSE (vol points), exact pricer"),
            paired_row("surrogate-only vs exact", df.rmse_surrogate_only, df.rmse_exact, "sur", "exact", "same-day IV RMSE (vol points), exact pricer"),
            paired_row("surrogate-only vs accelerated", df.rmse_surrogate_only, df.rmse_accel, "sur", "accel", "same-day IV RMSE (vol points), exact pricer"),
            paired_row("exact polish from Phase-3 starts (no surrogate, same polish budget) vs accelerated", df.rmse_exact_polish_from_phase3_starts, df.rmse_accel, "ctrl", "accel",
                       "same-day IV RMSE (vol points), exact pricer"),
            paired_row("exact polish from Phase-3 starts vs exact", df.rmse_exact_polish_from_phase3_starts, df.rmse_exact, "ctrl", "exact", "same-day IV RMSE (vol points), exact pricer")]
    if "rmse_accel_without_rerank" in df:
        rows.append(paired_row("accelerated without exact re-ranking vs accelerated", df.rmse_accel_without_rerank, df.rmse_accel, "plain", "accel", "same-day IV RMSE (vol points), exact pricer"))
    lc = lambda x: np.log(x)
    rows += [paired_row("accelerated vs exact", lc(df.cost_accel), lc(df.cost_exact), "accel", "exact", "log of the objective cost (soft-L1)"),
             paired_row("surrogate-only vs exact", lc(df.cost_surrogate_only), lc(df.cost_exact), "sur", "exact", "log of the objective cost (soft-L1)")]
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / "accuracy_paired.csv", index=False)
    methods = {"exact (Phase-3 recipe, 4 starts)": "exact", "accelerated (surrogate global + exact polish)": "accel", "surrogate-only (no polish)": "surrogate_only",
               "exact polish from Phase-3 starts (control)": "exact_polish_from_phase3_starts"}
    if "rmse_accel_without_rerank" in df:
        methods["accelerated without exact re-ranking (ablation)"] = "accel_without_rerank"
    srows = []
    for lab, k in methods.items():
        rm, co = df[f"rmse_{k}"], df[f"cost_{k}"]
        m, lo, hi = boot_ci(rm)
        md, mlo, mhi = boot_ci(rm, np.median)
        cm, clo, chi = boot_ci(np.log(co / df.cost_exact), np.mean)
        srows.append({"method": lab, "mean_rmse": m, "ci95_lo": lo, "ci95_hi": hi, "median_rmse": md, "median_ci95_lo": mlo, "median_ci95_hi": mhi,
                      "p90_rmse": float(np.percentile(rm, 90)), "max_rmse": float(rm.max()), "mean_log_cost_ratio_vs_exact": cm, "log_cost_ratio_ci95_lo": clo, "log_cost_ratio_ci95_hi": chi,
                      "share_days_cost_within_1pct_of_exact_or_better": float(np.mean(co <= 1.01 * df.cost_exact)), "share_days_cost_worse_than_5pct": float(np.mean(co > 1.05 * df.cost_exact)),
                      "share_days_cost_better_than_exact_by_more_than_1pct": float(np.mean(co < 0.99 * df.cost_exact)), "n_days": len(df)})
    pd.DataFrame(srows).to_csv(OUT / "accuracy_by_method.csv", index=False)
    h["rmse_mean"] = {k: float(df[f"rmse_{k}"].mean()) for k in methods.values()}
    h["rmse_median"] = {k: float(df[f"rmse_{k}"].median()) for k in methods.values()}
    h["accuracy_by_method"] = srows
    h["accuracy_paired"] = rows
    # ---- timing
    trows = []
    tcols = [("exact (Phase-3 recipe, 4 starts)", "sec_exact", "cpu_exact", "calls_exact"), ("accelerated", "sec_accel", "cpu_accel", "calls_accel"),
             ("exact polish from Phase-3 starts (control)", "sec_exact_polish_from_phase3_starts", None, "calls_exact_polish_from_phase3_starts")]
    for lab, s_, c_, n_ in tcols:
        m, lo, hi = boot_ci(df[s_])
        md, mlo, mhi = boot_ci(df[s_], np.median)
        trows.append({"method": lab, "mean_sec": m, "ci95_lo": lo, "ci95_hi": hi, "median_sec": md, "median_ci95_lo": mlo, "median_ci95_hi": mhi, "p90_sec": float(np.percentile(df[s_], 90)),
                      "total_sec": float(df[s_].sum()), "mean_cpu_sec": float(df[c_].mean()) if c_ else float("nan"),
                      "mean_exact_pricer_calls": float(df[n_].mean()), "median_exact_pricer_calls": float(df[n_].median())})
    pd.DataFrame(trows).to_csv(OUT / "timing_by_method.csv", index=False)
    rr, rlo, rhi = boot_ratio_of_means(df.sec_exact.values, df.sec_accel.values)
    sm, slo, shi = boot_ci(df.speedup_wall)
    smd, smlo, smhi = boot_ci(df.speedup_wall, np.median)
    cr, crlo, crhi = boot_ratio_of_means(df.calls_exact.values, df.calls_accel.values)
    cpr, cplo, cphi = boot_ratio_of_means(df.cpu_exact.values, df.cpu_accel.values)
    sp = pd.DataFrame([
        {"quantity": "speedup in total wall time (sum over days exact / sum accelerated)", "estimate": rr, "ci95_lo": rlo, "ci95_hi": rhi},
        {"quantity": "speedup in total CPU time (process time, includes torch threads)", "estimate": cpr, "ci95_lo": cplo, "ci95_hi": cphi},
        {"quantity": "mean of the per-day wall-time speedups", "estimate": sm, "ci95_lo": slo, "ci95_hi": shi},
        {"quantity": "median of the per-day wall-time speedups", "estimate": smd, "ci95_lo": smlo, "ci95_hi": smhi},
        {"quantity": "ratio of exact-pricer function evaluations (sum exact / sum accelerated)", "estimate": cr, "ci95_lo": crlo, "ci95_hi": crhi},
        {"quantity": "share of days on which the accelerated fit was faster", "estimate": float((df.speedup_wall > 1).mean()), "ci95_lo": float("nan"), "ci95_hi": float("nan")},
        {"quantity": "minimum per-day speedup", "estimate": float(df.speedup_wall.min()), "ci95_lo": float("nan"), "ci95_hi": float("nan")},
        {"quantity": "maximum per-day speedup", "estimate": float(df.speedup_wall.max()), "ci95_lo": float("nan"), "ci95_hi": float("nan")}])
    sp.to_csv(OUT / "speedup.csv", index=False)
    h["speedup_wall_total"] = {"est": rr, "lo": rlo, "hi": rhi}
    h["speedup_cpu_total"] = {"est": cpr, "lo": cplo, "hi": cphi}
    h["speedup_wall_median_per_day"] = {"est": smd, "lo": smlo, "hi": smhi}
    h["speedup_wall_mean_per_day"] = {"est": sm, "lo": slo, "hi": shi}
    h["call_ratio"] = {"est": cr, "lo": crlo, "hi": crhi}
    h["timing_by_method"] = trows
    h["accel_sec_breakdown_mean"] = {"surrogate_stage": float(df.sec_accel_surrogate_stage.mean()), "correction": float(df.sec_accel_correction.mean()), "polish": float(df.sec_accel_polish.mean())}
    # ---- timing by machine load (the other job's three workers were running during the first days of the test run only)
    rows_l = []
    for lo_, hi_, lab in ((0, 2, "1-minute load average < 2"), (2, 3.5, "2 to 3.5"), (3.5, 99, ">= 3.5")):
        d = df[(df.loadavg1_before >= lo_) & (df.loadavg1_before < hi_)]
        if len(d):
            rows_l.append({"load_before_the_day": lab, "n_days": len(d), "mean_sec_exact": d.sec_exact.mean(), "mean_sec_accel": d.sec_accel.mean(), "speedup_total_wall": d.sec_exact.sum() / d.sec_accel.sum(),
                           "median_speedup": d.speedup_wall.median(), "mean_cpu_over_wall_exact": (d.cpu_exact / d.sec_exact).mean(), "mean_cpu_over_wall_accel": (d.cpu_accel / d.sec_accel).mean()})
    pd.DataFrame(rows_l).to_csv(OUT / "timing_by_load.csv", index=False)
    h["timing_by_load"] = rows_l
    # ---- what the optimiser minimises: plain RMS of the vega-weighted residuals on the fit quotes (thinned set), versus the all-quote IV RMSE reported elsewhere
    fr = []
    for lab, k in (("exact", "exact"), ("accelerated", "accel"), ("surrogate-only", "surrogate_only")):
        fr.append({"method": lab, "mean_fit_quote_rms_resid": float(df[f"fitset_rms_{k}"].mean()), "median_fit_quote_rms_resid": float(df[f"fitset_rms_{k}"].median()),
                   "mean_all_quote_iv_rmse": float(df[f"rmse_{k}"].mean())})
    pd.DataFrame(fr).to_csv(OUT / "fit_quote_vs_all_quote_error.csv", index=False)
    h["fit_quote_vs_all_quote_error"] = fr
    # ---- the re-timed subset of 20 days (every 6th test day), against the C2 package's recorded times
    sub = df.iloc[::6].head(20)
    sub[["day", "sec_c2_recorded", "sec_exact", "sec_accel", "speedup_wall", "calls_exact", "calls_accel", "loadavg1_before", "other_python_cpu_pct_before", "cost_exact", "cost_c2_cached_exact",
         "reproduces_c2_cache_surface_gap_volpts"]].to_csv(OUT / "retimed_20_days.csv", index=False)
    h["retimed_20"] = {"n": len(sub), "median_c2_recorded_sec": float(sub.sec_c2_recorded.median()), "median_retimed_exact_sec": float(sub.sec_exact.median()),
                       "median_accel_sec": float(sub.sec_accel.median()), "speedup_total_wall": float(sub.sec_exact.sum() / sub.sec_accel.sum()), "speedup_median": float(sub.speedup_wall.median()),
                       "mean_loadavg1": float(sub.loadavg1_before.mean())}
    h["load"] = {"loadavg1_before_mean": float(df.loadavg1_before.mean()), "loadavg1_before_min": float(df.loadavg1_before.min()), "loadavg1_before_max": float(df.loadavg1_before.max()),
                 "other_python_cpu_pct_mean": float(df.other_python_cpu_pct_before.mean())}
    h["c2_recorded_sec"] = {"mean": float(df.sec_c2_recorded.mean()), "median": float(df.sec_c2_recorded.median()), "max": float(df.sec_c2_recorded.max())}
    h["reproduction_of_c2_cache"] = {"days_cost_identical_1e-6": int((df.reproduces_c2_cache_cost_diff < 1e-6).sum()), "n": len(df),
                                     "max_cost_diff": float(df.reproduces_c2_cache_cost_diff.max()), "max_surface_gap_volpts": float(df.reproduces_c2_cache_surface_gap_volpts.max())}
    # ---- parameter agreement and weak identifiability
    names = CP.NAMES
    pa = []
    for k, nm in enumerate(names):
        da = np.array([ (np.log(max(res[i]["accel"]["u"][k], 1e-12)) - np.log(max(res[i]["exact"]["u"][k], 1e-12))) if LOG[k] else (res[i]["accel"]["u"][k] - res[i]["exact"]["u"][k]) for i in sorted(res)])
        ds = np.array([ (np.log(max(res[i]["accel"]["u_sur"][k], 1e-12)) - np.log(max(res[i]["exact"]["u"][k], 1e-12))) if LOG[k] else (res[i]["accel"]["u_sur"][k] - res[i]["exact"]["u"][k]) for i in sorted(res)])
        wa = np.abs(np.array([res[i]["accel"]["u"][k] - res[i]["exact"]["u"][k] for i in sorted(res)])) / WIDTH[k]
        pa.append({"parameter": nm, "scale": "log" if LOG[k] else "linear", "median_abs_diff_accel_vs_exact": float(np.median(np.abs(da))), "p90_abs_diff_accel_vs_exact": float(np.percentile(np.abs(da), 90)),
                   "median_abs_diff_surrogate_only_vs_exact": float(np.median(np.abs(ds))),
                   "median_diff_as_share_of_bound_width": float(np.median(wa)), "share_days_within_5pct_of_bound_width": float(np.mean(wa <= 0.05)),
                   "share_days_exact_at_bound": float(np.mean([k in res[i]["exact"]["score"]["at_bound"] for i in sorted(res)])),
                   "share_days_accel_at_bound": float(np.mean([k in res[i]["accel"]["score"]["at_bound"] for i in sorted(res)]))})
    pd.DataFrame(pa).to_csv(OUT / "parameter_agreement.csv", index=False)
    h["parameter_agreement"] = pa
    ags = []
    for lab, col_f, col_s in (("accelerated vs exact", "dist_frac_accel_vs_exact", "dist_scaled_accel_vs_exact"),
                              ("surrogate-only vs exact", "dist_frac_surrogate_only_vs_exact", "dist_scaled_surrogate_only_vs_exact")):
        ags.append({"pair": lab, "median_dist_frac": float(df[col_f].median()), "mean_dist_frac": float(df[col_f].mean()), "p90_dist_frac": float(np.percentile(df[col_f], 90)),
                    "median_dist_scaled_rms": float(df[col_s].median()), "mean_dist_scaled_rms": float(df[col_s].mean()), "p90_dist_scaled_rms": float(np.percentile(df[col_s], 90)),
                    **{f"share_same_solution_tol{int(t * 100)}pct": float((df[col_f] < t).mean()) for t in (0.02, 0.05, 0.10)}})
    pd.DataFrame(ags).to_csv(OUT / "parameter_distance.csv", index=False)
    h["parameter_distance"] = ags
    rel = (df.cost_accel - df.cost_exact).abs() / np.minimum(df.cost_accel, df.cost_exact)
    wk = []
    for ct in (0.001, 0.01, 0.05):
        for pt_ in (0.02, 0.05, 0.10):
            eq = rel <= ct
            diff = eq & (df.dist_frac_accel_vs_exact > pt_)
            wk.append({"cost_tolerance": ct, "param_tolerance_share_of_bound_width": pt_, "share_days_equal_cost": float(eq.mean()), "share_days_equal_cost_same_params": float((eq & ~diff).mean()),
                       "share_days_equal_cost_DIFFERENT_params": float(diff.mean()), "n_days_equal_cost_different_params": int(diff.sum()),
                       "median_surface_gap_volpts_on_those_days": float(df.loc[diff, "surface_gap_accel_vs_exact_volpts"].median()) if diff.any() else float("nan"),
                       "p90_surface_gap_volpts_on_those_days": float(np.percentile(df.loc[diff, "surface_gap_accel_vs_exact_volpts"], 90)) if diff.any() else float("nan")})
    pd.DataFrame(wk).to_csv(OUT / "weak_identifiability.csv", index=False)
    h["weak_identifiability"] = wk
    # ---- where the accelerated path is worse
    worse = df[(df.rmse_accel > df.rmse_exact + 0.02) | (df.cost_accel > 1.01 * df.cost_exact)].copy()
    worse["rmse_excess"] = worse.rmse_accel - worse.rmse_exact
    worse.sort_values("rmse_excess", ascending=False)[["day", "rmse_exact", "rmse_accel", "rmse_surrogate_only", "rmse_excess", "cost_exact", "cost_accel", "cost_ratio_accel_vs_exact",
                                                       "exact_solution_inside_box", "accel_solution_inside_box", "n_quotes_outside_surrogate_box", "polish_budget_exhausted",
                                                       "n_params_at_bound_exact", "n_params_at_bound_accel", "dist_frac_accel_vs_exact", "calls_accel", "sec_accel", "sec_exact"]].to_csv(OUT / "days_where_accelerated_is_worse.csv", index=False)
    better = df[(df.rmse_accel < df.rmse_exact - 0.02) | (df.cost_accel < 0.99 * df.cost_exact)]
    h["worse_days"] = {"n_worse": int(len(worse)), "n_better": int(len(better)), "n": len(df),
                       "worse_rmse_by_more_than_0.1": int(((df.rmse_accel - df.rmse_exact) > 0.1).sum()), "worse_rmse_by_more_than_0.5": int(((df.rmse_accel - df.rmse_exact) > 0.5).sum()),
                       "max_rmse_excess": float((df.rmse_accel - df.rmse_exact).max()), "max_cost_ratio": float(df.cost_ratio_accel_vs_exact.max()),
                       "min_cost_ratio": float(df.cost_ratio_accel_vs_exact.min())}
    seg = []
    for lab, mask in (("exact solution inside the surrogate box", df.exact_solution_inside_box), ("exact solution outside the surrogate box", ~df.exact_solution_inside_box),
                      ("no quote beyond tau 0.6 y", df.n_quotes_outside_surrogate_box == 0), ("some quotes beyond tau 0.6 y", df.n_quotes_outside_surrogate_box > 0),
                      ("polish budget exhausted", df.polish_budget_exhausted), ("polish converged inside its budget", ~df.polish_budget_exhausted), ("all days", df.index >= 0)):
        d = df[mask]
        if len(d):
            seg.append({"segment": lab, "n_days": len(d), "mean_rmse_exact": d.rmse_exact.mean(), "mean_rmse_accel": d.rmse_accel.mean(), "mean_rmse_surrogate_only": d.rmse_surrogate_only.mean(),
                        "median_cost_ratio_accel": d.cost_ratio_accel_vs_exact.median(), "median_cost_ratio_surrogate_only": d.cost_ratio_surrogate_only_vs_exact.median(),
                        "share_accel_cost_worse_than_1pct": float((d.cost_ratio_accel_vs_exact > 1.01).mean()), "median_speedup": d.speedup_wall.median()})
    pd.DataFrame(seg).to_csv(OUT / "segments.csv", index=False)
    h["segments"] = seg
    return h


def make_figures(df: pd.DataFrame, res: dict, h: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    from dh2 import plotstyle as PS
    PS.setup()
    BLUE, ORANGE, AQUA, YELLOW = PS.PAL
    INK2 = PS.INK2
    date_ = pd.to_datetime(df.day)
    inbox = df.exact_solution_inside_box.values
    # -------- figure 1: same-day IV RMSE, paired
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.4))
    lim = (0.45, max(df.rmse_exact.max(), df.rmse_accel.max(), df.rmse_surrogate_only.max()) * 1.15)
    for a, col, lab, c in ((ax[0], "rmse_accel", "accelerated (surrogate global + exact polish)", ORANGE), (ax[1], "rmse_surrogate_only", "surrogate-only (no polish)", AQUA)):
        a.plot(lim, lim, color=INK2, lw=1.4, zorder=1)
        a.scatter(df.rmse_exact[inbox], df[col][inbox], s=70, color=c, edgecolor=PS.SURFACE, lw=1.2, marker="o", zorder=3, label="exact solution inside the surrogate box")
        a.scatter(df.rmse_exact[~inbox], df[col][~inbox], s=90, color=c, edgecolor=PS.INK, lw=1.2, marker="^", zorder=3, label="exact solution outside the box")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel("exact calibration, same-day IV RMSE (vol points)"); a.set_ylabel(f"{lab.split(' (')[0]}, IV RMSE (vol points)")
        m_e, m_m = df.rmse_exact.mean(), df[col].mean()
        a.set_title(f"{lab.split(' (')[0].capitalize()}: mean {m_m:.2f} vs exact {m_e:.2f}", fontsize=18)
        a.text(0.97, 0.05, "below the line = better than exact", transform=a.transAxes, ha="right", fontsize=14, color=INK2)
        a.legend(loc="upper left", fontsize=13)
        a.set_xticks([0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.5", "1", "2", "4", "8"]); a.set_yticks([0.5, 1, 2, 4, 8]); a.set_yticklabels(["0.5", "1", "2", "4", "8"])
        a.xaxis.set_minor_formatter(NullFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Same-day implied-vol error of the final solution, scored with the exact pricer (118 test days)", fontsize=20, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig1_rmse_paired.png"); plt.close(fig)
    # -------- figure 2: cost ratio distribution
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    series = [("accelerated", df.cost_ratio_accel_vs_exact, ORANGE), ("surrogate-only", df.cost_ratio_surrogate_only_vs_exact, AQUA),
              ("exact polish from Phase-3 starts\n(control, same polish budget)", df.cost_ratio_phase3_starts_vs_exact, YELLOW)]
    for lab, v, c in series:
        x = np.sort(v.values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, lw=2.6, label=lab.replace("\n", " "))
    ax.axvline(1.0, color=PS.INK, lw=1.2)
    ax.set_xscale("log"); ax.set_xlim(0.9, max(2.6, df.cost_ratio_surrogate_only_vs_exact.max() * 1.05))
    ax.set_xticks([0.9, 1, 1.1, 1.25, 1.5, 2, 2.5]); ax.set_xticklabels(["0.9", "1", "1.1", "1.25", "1.5", "2", "2.5"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("objective cost / cost of the exact calibration (1 = same; below 1 = lower cost than exact)")
    ax.set_ylabel("share of days")
    ax.set_title("Cost reached relative to the exact multi-start calibration")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig2_cost_ratio.png"); plt.close(fig)
    # -------- figure 3: time
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    xlo, xhi = df.sec_exact.min() * 0.75, df.sec_exact.max() * 1.35
    ylo, yhi = df.sec_accel.min() * 0.8, df.sec_accel.max() * 1.35
    xs = np.array([xlo, xhi])
    for k, ls in ((3, "-"), (6, "--"), (12, ":")):                   # lines of constant speedup: accelerated time = exact time / k
        a.plot(xs, xs / k, color=INK2, lw=1.3, ls=ls)
        xl, yl = (xhi, xhi / k) if xhi / k <= yhi else (yhi * k, yhi)
        a.text(xl, yl, f"{k}x faster", color=INK2, fontsize=14, ha="right", va="top")
    a.scatter(df.sec_exact, df.sec_accel, s=70, color=ORANGE, edgecolor=PS.SURFACE, lw=1.2, zorder=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(xlo, xhi); a.set_ylim(ylo, yhi)
    a.set_xticks([2, 5, 10, 20]); a.set_xticklabels(["2", "5", "10", "20"]); a.set_yticks([0.5, 1, 2]); a.set_yticklabels(["0.5", "1", "2"])
    a.xaxis.set_minor_formatter(NullFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("exact calibration, wall time per day (s)"); a.set_ylabel("accelerated, wall time per day (s)")
    a.set_title(f"Per day: total speedup {h['speedup_wall_total']['est']:.1f}x")
    b = ax[1]
    bd = h["accel_sec_breakdown_mean"]
    me = df.sec_exact.mean()
    b.bar([0], [me], color=BLUE, width=0.55, label="exact (4 starts)")
    bottom = 0.0
    for lab, v, c in (("surrogate global search", bd["surrogate_stage"], AQUA), ("exact re-rank / correction", bd["correction"], YELLOW), ("exact polish", bd["polish"], ORANGE)):
        b.bar([1], [v], bottom=bottom, color=c, width=0.55, label=lab, edgecolor=PS.SURFACE, lw=2)
        bottom += v
    b.set_xticks([0, 1]); b.set_xticklabels(["exact\ncalibration", "accelerated"])
    b.set_ylabel("mean wall time per day (s)")
    b.annotate(f"{me:.1f} s\n{df.calls_exact.mean():.0f} exact-pricer calls", xy=(0, me), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=14)
    b.annotate(f"{bottom:.1f} s\n{df.calls_accel.mean():.0f} exact-pricer calls", xy=(1, bottom), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=14)
    b.set_ylim(0, me * 1.3)
    b.legend(loc="upper right", fontsize=13)
    b.set_title("Where the time goes")
    fig.tight_layout(); fig.savefig(OUT / "fig3_time.png"); plt.close(fig)
    # -------- figure 4: parameters
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    names = CP.NAMES
    data = []
    for k in range(10):
        d = np.array([(np.log(max(res[i]["accel"]["u"][k], 1e-12)) - np.log(max(res[i]["exact"]["u"][k], 1e-12))) if LOG[k] else (res[i]["accel"]["u"][k] - res[i]["exact"]["u"][k]) for i in sorted(res)])
        data.append(np.abs(d))
    bp = a.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6, showfliers=True, flierprops=dict(marker="o", markersize=4, markerfacecolor=ORANGE, markeredgecolor=ORANGE, alpha=0.6),
                   medianprops=dict(color=PS.INK, lw=2.2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for p in bp["boxes"]:
        p.set(facecolor=ORANGE, alpha=0.55, edgecolor=INK2)
    a.set_yticks(range(1, 11)); a.set_yticklabels([n + (" (log)" if LOG[k] else "") for k, n in enumerate(names)])
    a.invert_yaxis(); a.set_xscale("symlog", linthresh=0.002); a.set_xlim(0, 8)
    a.set_xticks([0, 0.01, 0.1, 1, 5]); a.set_xticklabels(["0", "0.01", "0.1", "1", "5"]); a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("|accelerated - exact| (log units for positive parameters, plain for correlations)")
    a.set_title("Parameter difference per day")
    c = ax[1]
    c.scatter(df.dist_frac_accel_vs_exact, (df.cost_accel - df.cost_exact).abs() / np.minimum(df.cost_accel, df.cost_exact) + 1e-6, s=70, color=ORANGE, edgecolor=PS.SURFACE, lw=1.2, zorder=3)
    c.axvline(0.05, color=INK2, ls="--", lw=1.4); c.axhline(0.01, color=INK2, ls="--", lw=1.4)
    c.set_xscale("symlog", linthresh=0.01); c.set_xlim(0, 1.05); c.set_yscale("log")
    c.set_xticks([0, 0.01, 0.05, 0.1, 0.5, 1]); c.set_xticklabels(["0", "0.01", "0.05", "0.1", "0.5", "1"]); c.xaxis.set_minor_formatter(NullFormatter())
    c.set_xlabel("largest parameter gap (share of its bound width)"); c.set_ylabel("|relative cost difference| (+1e-6)")
    wk = [q for q in h["weak_identifiability"] if q["cost_tolerance"] == 0.01 and q["param_tolerance_share_of_bound_width"] == 0.05][0]
    c.set_title("Same cost, different parameters?")
    c.text(0.96, 0.04, f"cost within 1% and gap > 5%:\n{wk['n_days_equal_cost_different_params']} of {len(df)} days", transform=c.transAxes, ha="right", va="bottom", fontsize=14, color=PS.INK)
    fig.tight_layout(); fig.savefig(OUT / "fig4_parameters.png"); plt.close(fig)
    # -------- figure 5: over time
    fig, ax = plt.subplots(figsize=(15, 7.2))
    d_acc = (df.rmse_accel - df.rmse_exact).values
    d_sur = (df.rmse_surrogate_only - df.rmse_exact).values
    ax.axhline(0, color=PS.INK, lw=1.2)
    ax.scatter(date_[inbox], d_acc[inbox], s=70, color=ORANGE, edgecolor=PS.SURFACE, lw=1.2, marker="o", zorder=3)
    ax.scatter(date_[~inbox], d_acc[~inbox], s=90, color=ORANGE, edgecolor=PS.INK, lw=1.2, marker="^", zorder=3)
    ax.scatter(date_, d_sur, s=34, color=AQUA, alpha=0.9, zorder=2, marker="s")
    ax.set_yscale("symlog", linthresh=0.1); ax.set_yticks([-1, -0.3, -0.1, 0, 0.1, 0.3, 1, 3]); ax.set_yticklabels(["-1", "-0.3", "-0.1", "0", "0.1", "0.3", "1", "3"])
    ax.set_ylabel("IV RMSE minus exact calibration's (vol points)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, markersize=10, label="accelerated"), Line2D([], [], marker="^", ls="", color=ORANGE, markeredgecolor=PS.INK, markersize=11, label="accelerated, exact solution outside the box"),
                       Line2D([], [], marker="s", ls="", color=AQUA, markersize=8, label="surrogate-only")], loc="upper left", fontsize=14)
    ax.set_title("Above zero = worse than the exact calibration")
    fig.tight_layout(); fig.savefig(OUT / "fig5_rmse_difference_over_time.png"); plt.close(fig)
    # -------- figure 6: design (development days only): one panel per variant, one line per number of polished candidates, points = polish budgets 5, 10, 20, 40
    p = OUT / "design_variant_polish_grid.csv"
    if p.exists():
        g = pd.read_csv(p)
        ramp = {1: "#f6bf9b", 2: "#ee8450", 3: "#bd4318"}              # one hue (the accelerated method's orange), light to dark = more candidates
        titles = {"plain": "plain: polish the surrogate's\nown best candidates", "rerank": "re-rank the 8 survivors\nby exact cost first", "rerank_corr2": "re-rank + 2 rounds of\ndefect correction"}
        fig, axs = plt.subplots(1, 3, figsize=(17.5, 7.4), sharey=True)
        for a, v in zip(axs, ["plain", "rerank", "rerank_corr2"]):
            for k in (1, 2, 3):
                q = g[(g.variant == v) & (g.n_polish == k)].sort_values("polish_nfev")
                a.plot(q.mean_sec, q.mean_log_cost_ratio, color=ramp[k], lw=2.4, marker="o", ms=9, mec=PS.SURFACE, mew=1.2, label=f"{k} candidate{'s' if k > 1 else ''} polished", zorder=3)
                for _, r in q.iterrows():
                    if k == 1:
                        a.annotate(f"{int(r.polish_nfev)}", (r.mean_sec, r.mean_log_cost_ratio), xytext=(7, 7), textcoords="offset points", fontsize=12, color=INK2)
            ch = g[g.chosen]
            if (ch.variant == v).any():
                c0 = ch.iloc[0]
                a.scatter([c0.mean_sec], [c0.mean_log_cost_ratio], s=520, facecolor="none", edgecolor=PS.INK, lw=2.4, zorder=4)
                a.annotate("chosen (frozen)", (c0.mean_sec, c0.mean_log_cost_ratio), xytext=(18, 26), textcoords="offset points", fontsize=14, color=PS.INK, arrowprops=dict(arrowstyle="-", color=PS.INK))
            a.axhline(0, color=PS.INK, lw=1.2)
            a.set_title(titles[v], fontsize=17)
            a.set_xlabel("mean wall time per development day (s)")
            a.set_xlim(0, 3.6)
        axs[0].set_ylabel("mean log(cost / exact cost); 0 = as good as exact")
        axs[0].legend(loc="upper right", fontsize=13)
        fig.suptitle("Design on 52 development days (exact calibration: 6.6 s per day, off the right edge). Numbers = polish budget in evaluations", fontsize=18, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / "fig6_design_development_days.png"); plt.close(fig)

def report() -> None:
    cfg = load_frozen()
    res = load_cache("test_days.pkl")
    assert len(res) == N_TEST_DAYS, f"only {len(res)} of {N_TEST_DAYS} test days are cached: run the test stage first"
    sur = Surrogate.load(device="cpu")
    df = per_day_table(res, sur)
    S, _, _ = load_world()
    for key, col in (("exact", "exact"), ("accel", "accel"), ("u_sur", "surrogate_only")):           # the optimised criterion's own residuals (fit quotes, vega-weighted, plain RMS)
        vals = []
        for i in sorted(res):
            ix = M.thin(S[i], 14)
            u = res[i]["accel"]["u_sur"] if key == "u_sur" else res[i][key]["u"]
            vals.append(float(np.sqrt(np.mean(CP.exact_residual(u, S[i], ix) ** 2))))
        df[f"fitset_rms_{col}"] = vals
    df.to_csv(OUT / "per_day.csv", index=False)
    for key, nm in (("exact", "params_exact.csv"), ("accel", "params_accelerated.csv"), ("matched", "params_exact_polish_from_phase3_starts.csv")):
        pd.DataFrame([{"day": res[i]["day"], **params_row(res[i][key]["u"])} for i in sorted(res)]).to_csv(OUT / nm, index=False)
    pd.DataFrame([{"day": res[i]["day"], **params_row(res[i]["accel"]["u_sur"])} for i in sorted(res)]).to_csv(OUT / "params_surrogate_only.csv", index=False)
    h = make_tables(df, res, cfg)
    h["frozen_config"] = json.loads((OUT / "frozen_config.json").read_text())
    h["n_days"] = len(df)
    h["box_exact_inside_share"] = float(df.exact_solution_inside_box.mean())
    h["share_days_with_quotes_beyond_tau_0.6"] = float((df.n_quotes_outside_surrogate_box > 0).mean())
    h["mean_share_quotes_beyond_box"] = float((df.n_quotes_outside_surrogate_box / df.n_fit_quotes).mean())
    (OUT / "headline.json").write_text(json.dumps(h, indent=1, default=float))
    make_figures(df, res, h)
    print(json.dumps({k: h[k] for k in ("rmse_mean", "speedup_wall_total", "call_ratio", "worse_days", "box_exact_inside_share")}, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "design":
        design()
    elif cmd == "test":
        test()
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
