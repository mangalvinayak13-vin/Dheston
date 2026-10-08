"""Shared helpers for work package C2 (attribution, cross-sectional tests, identifiability, per-bucket errors).

Nothing here changes an existing file: it imports dh2.calib / dh2.models / dh2.struct and re-uses them.

The sample. Every 10th trading day of the TEST period (2022-01-01 onward), taken in order from the cached surfaces and indexed
i % 10 == 0. It is fixed by this rule alone (no result was looked at), and it is identical for every model and every step so that
all comparisons are paired.

Parallelism. At most 4 worker processes (joblib/loky), one heavy job at a time. Each job writes {"done": n, "total": m} to
logs/progress/c2_<task>_00.json, which the live dashboard reads.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dh2 import calib, evalpairs as EP, models as M, struct as ST  # noqa: E402
from dh2.data import Surface, load_all  # noqa: E402

SPLIT = date(2022, 1, 1)
RES = ROOT / "results"
CACHE = RES / "_cache"
PROG = ROOT / "logs" / "progress"
WALK = ROOT / "data" / "results"
MODELS = ["bs", "heston", "dh", "dhj"]
N_WORKERS = 3
CACHE_VERSION = "v1"


# ------------------------------------------------------------------ the sample
def load_sample():
    """(all surfaces, indices of the sample days). Every 10th test-period trading day, in order."""
    S = load_all()
    test = [i for i, s in enumerate(S) if s.day >= SPLIT]
    return S, test[::10]


def next_index(S, i: int):
    """Index of the next trading day if it is a consecutive pair (gap of at most a long weekend), else None."""
    return i + 1 if (i + 1 < len(S) and (S[i + 1].day - S[i].day).days <= 5) else None


# ------------------------------------------------------------------ progress + parallel map with resume
def write_progress(task: str, done: int, total: int) -> None:
    PROG.mkdir(parents=True, exist_ok=True)
    p = PROG / f"c2_{task}_00.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done": int(done), "total": int(total)}))
    tmp.replace(p)


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def _run(fn, key, args):
    try:
        return key, fn(*args)
    except Exception as e:  # a failed job is recorded, never fatal
        return key, {"error": repr(e)}


def pmap(fn, jobs, task: str, cache_name: str | None = None, n_jobs: int = N_WORKERS, save_every: int = 10) -> dict:
    """Run fn(*args) for every (key, args) in jobs on at most n_jobs workers; returns {key: result}.

    Results are cached to results/_cache/<cache_name>_<version>.pkl as they arrive, so an interrupted run resumes."""
    cpath = (CACHE / f"{cache_name}_{CACHE_VERSION}.pkl") if cache_name else None
    done = pickle.load(open(cpath, "rb")) if (cpath and cpath.exists()) else {}
    todo = [(k, a) for k, a in jobs if k not in done]
    total, base = len(jobs), len(jobs) - len(todo)
    write_progress(task, base, total)
    if todo:
        gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered", batch_size=1)(delayed(_run)(fn, k, a) for k, a in todo)
        n = 0
        for k, r in gen:
            done[k] = r
            n += 1
            write_progress(task, base + n, total)
            if cpath and n % save_every == 0:
                _dump(done, cpath)
        if cpath:
            _dump(done, cpath)
    return {k: done[k] for k, _ in jobs}


# ------------------------------------------------------------------ surfaces
def subsurface(S: Surface, keep: np.ndarray) -> Surface:
    """The surface restricted to the quotes in `keep` (expiries without a quote are dropped and the expiry index is remapped)."""
    keep = np.sort(np.asarray(keep, int))
    used = np.unique(S.e[keep])
    remap = {int(o): n for n, o in enumerate(used)}
    e = np.array([remap[int(i)] for i in S.e[keep]], int)
    return Surface(S.day, S.spot, S.r, S.T[used], S.F[used], S.q[used], [S.expiry[i] for i in used], S.K[keep], e, S.is_call[keep],
                   S.price[keep], S.iv[keep], S.vega[keep], S.volume[keep], S.oi[keep], S.spot_source)


def rmse(err: np.ndarray) -> float:
    err = np.asarray(err, float)
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def iv_rmse(model: str, x, S: Surface, mask: np.ndarray | None = None) -> float:
    """Model implied-vol RMSE over the quotes of S (vol points), optionally only the quotes in mask."""
    return rmse(M.iv_error(model, x, S, mask))


# ------------------------------------------------------------------ the dh2.calib.fit_day fit, keeping every start
def fit_day_full(model: str, S: Surface, n_starts: int = 4, seed: int = 0, per_expiry: int = 14, max_nfev: int = 120,
                 x0: np.ndarray | None = None, want_jac: bool = False) -> dict:
    """Exactly the loop of dh2.calib.fit_day (same thinning, starts, bounds, soft-L1 loss, scaling), but every start's result is kept
    so that the multi-start solutions can be examined. The best entry equals calib.fit_day's result for the same arguments."""
    rng = np.random.default_rng(seed)
    idx = M.thin(S, per_expiry)
    lo, hi = (np.array(b) for b in calib.BOUNDS[model])
    runs = []
    for u0 in ([calib.clip(model, x0)] if x0 is not None else []) + calib.starts(model, S, rng, n_starts):
        try:
            r = least_squares(lambda u: calib.residual(model, u, S, idx), u0, bounds=(lo, hi), loss="soft_l1", f_scale=calib.F_SCALE,
                              x_scale=np.maximum(hi - lo, 1e-9) * 0.05, max_nfev=max_nfev)
        except Exception:
            continue
        runs.append({"u": r.x.copy(), "cost": float(r.cost), "nfev": int(r.nfev), "status": int(r.status),
                     "jac": (np.array(r.jac) if want_jac else None)})
    best = runs[int(np.argmin([q["cost"] for q in runs]))]      # first minimum, as fit_day's strict '<' keeps the first
    u = best["u"]
    err = M.iv_error(model, calib.to_phys(model, u), S)
    tol = 0.003 * (hi - lo)
    return {"u": u, "x": calib.to_phys(model, u), "cost": best["cost"], "rmse": rmse(err),
            "at_bound": [int(i) for i in np.flatnonzero((u <= lo + tol) | (u >= hi - tol))], "n": S.n,
            "runs": runs, "idx": idx, "best": best}


def daily_fit_job(model: str, S0: Surface, S1: Surface | None, seed: int) -> dict:
    """Independent same-day fit (dh2.calib.fit_day objective) + the same parameters carried to the next day."""
    t = time.time()
    f = fit_day_full(model, S0, n_starts=4, seed=seed)
    x = f["x"]
    return {"u": f["u"], "x": x, "cost": f["cost"], "rmse_in": f["rmse"], "rmse_next": (iv_rmse(model, x, S1) if S1 is not None else float("nan")),
            "at_bound": f["at_bound"], "n": f["n"], "sec": time.time() - t}


def daily_fits(S, samp, task: str = "fits") -> dict:
    """{(model, i): result} for the four models on every sample day (cached; shared by attribution, buckets and cross-checks)."""
    jobs = []
    for i in samp:
        j = next_index(S, i)
        for m in MODELS:
            jobs.append(((m, i), (m, S[i], S[j] if j is not None else None, i)))
    return pmap(daily_fit_job, jobs, task, cache_name="daily_fits")


# ------------------------------------------------------------------ statistics
def boot_ci(v, n_boot: int = 10000, seed: int = 0, stat=np.mean):
    """(estimate, lo, hi): percentile bootstrap 95% interval, resampling days."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, v.size, size=(n_boot, v.size))
    bs = stat(v[ix], axis=1)
    return float(stat(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wilcoxon_p(a, b) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value on the days where both are finite (NaN if fewer than 6 or all differences zero)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size < 6 or np.all(d == 0):
        return float("nan")
    return float(wilcoxon(d, zero_method="wilcox").pvalue)
