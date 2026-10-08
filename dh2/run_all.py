"""One command that reproduces every table, figure and headline number of the project from cached data, and a demo-scale mode for a live demonstration.

    python -m dh2.run_all --config configs/nifty.yaml                # every stage; a stage whose outputs exist is skipped
    python -m dh2.run_all --config configs/nifty.yaml --quick        # demo-scale run on a short real window, writes ONLY to results_quick/
    python -m dh2.run_all --config configs/nifty.yaml --list         # every stage: cached or missing, estimated cost, outputs, dependencies
    python -m dh2.run_all --stages benchmark_table,metrics_tables    # selected stages (and any missing dependency)
    python -m dh2.run_all --stages walk_dh --force --yes             # rerun a stage although its outputs exist (hours for a walk-forward)
    python -m dh2.run_all --retrain-pinn                             # train the PINN checkpoints that are MISSING (the shipped ones are only validated)

Stages run in dependency order. Each declares its outputs; a stage is skipped when all of them exist unless --force. Existing scripts are never edited: a stage calls
them as subprocesses (run_walk.py, hybrids.py, trade_*.py, exp_*.py, dh2.dq, dh2.svi, ...) or imports their functions; the log of every subprocess is in
results/run_logs/<stage>.log. A plan with estimated costs is printed first; a stage estimated above execution.confirm_over_minutes (or needing the network) asks for
confirmation (--yes to skip). Worker processes: execution.workers (2). Nothing runs in the background: every stage finishes before the next starts.

The estimates are the compute times recorded in this repository (README, work-package READMEs, file times of the cached results), scaled where the recorded run used more workers;
the basis of each is shown by --list. The real elapsed time of every stage is printed at the end and written to results/run_logs/last_run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]


class StageError(RuntimeError):
    pass


@dataclass
class Stage:
    name: str
    title: str
    outputs: list                    # paths relative to the repository root; all present = cached
    deps: list = field(default_factory=list)
    run: Callable | None = None      # run(ctx)
    est_min: float = 1.0             # estimated minutes (at execution.workers)
    basis: str = ""                  # where the estimate comes from
    group: str = ""
    optional: bool = False           # only runs when named in --stages
    network: object = False          # True, or a callable(ctx) -> bool
    note: str = ""


# ============================================================================================================== config
REQUIRED = {"paths": ["raw_data", "surface_cache", "walk_results", "results", "results_quick", "tables", "figures", "run_logs", "pinn_checkpoints", "today_json", "vix_csv"],
            "data": ["first_day", "last_day", "expected_files", "min_raw_files", "filters"],
            "split": ["development", "test"],
            "walk_forward": ["tag", "models", "window", "refit", "burn", "per_expiry", "nfev_cold", "nfev_warm", "prior", "chunks", "bounds"],
            "forecasts": ["variants", "benchmarks", "headline_runs", "report_variants"],
            "evaluation": ["paired_sample", "bootstrap", "smile_figure_days"],
            "trading": ["model", "tag", "top", "lags"],
            "pinn": ["settings", "shipped_checkpoints", "validation_stages"],
            "execution": ["workers", "max_workers", "confirm_over_minutes"],
            "figures": ["dpi", "size_inches", "formats"],
            "quick": ["start", "end", "models", "window", "refit", "burn", "per_expiry", "nfev_cold", "nfev_warm", "chunks", "bootstrap", "pricer_days", "pinn_days", "svi_days"]}


def load_config(path) -> dict:
    import yaml
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"config file not found: {p}")
    cfg = yaml.safe_load(p.read_text())
    for sec, keys in REQUIRED.items():
        if sec not in cfg:
            raise SystemExit(f"config {p.name}: missing section {sec!r}")
        for k in keys:
            if k not in cfg[sec]:
                raise SystemExit(f"config {p.name}: missing key {sec}.{k}")
    return cfg


# ============================================================================================================== context and helpers
class Ctx:
    def __init__(self, cfg: dict, root: Path = ROOT, quick: bool = False, force: bool = False, retrain_pinn: bool = False, workers: int | None = None, out=print):
        self.cfg, self.root, self.quick, self.force, self.retrain_pinn, self.out = cfg, Path(root), quick, force, retrain_pinn, out
        ex = cfg["execution"]
        w = ex["workers"] if workers is None else min(int(workers), int(ex["max_workers"]))
        self.workers = max(1, min(int(w), int(ex["max_workers"])))
        base = self.root / (cfg["paths"]["results_quick"] if quick else cfg["paths"]["results"])
        self.logdir = base / "run_logs" if quick else self.root / cfg["paths"]["run_logs"]
        self.timings: dict[str, float] = {}

    def log(self, msg: str) -> None:
        self.out(msg)

    def env(self, threads: str = "1") -> dict:
        e = dict(os.environ)
        e["PYTHONPATH"] = str(self.root) + os.pathsep + e.get("PYTHONPATH", "")
        e["PYTHONUNBUFFERED"] = "1"
        e["MPLBACKEND"] = "Agg"
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            e.setdefault(k, threads)
        return e

    def sh(self, name: str, cmd: list, env_extra: dict | None = None, logname: str | None = None) -> None:
        """Run one subprocess, output to results/run_logs/<name>.log; raise StageError (with the tail of the log) on a non-zero exit."""
        self.logdir.mkdir(parents=True, exist_ok=True)
        path = self.logdir / (logname or f"{name}.log")
        env = self.env()
        env.update(env_extra or {})
        with open(path, "a") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}\n")
            f.flush()
            r = subprocess.run([str(c) for c in cmd], cwd=self.root, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = "".join(path.read_text().splitlines(keepends=True)[-25:])
            raise StageError(f"{name}: command failed with exit code {r.returncode}: {' '.join(map(str, cmd))}\n--- last lines of {path} ---\n{tail}")

    def py(self, name: str, *args, **kw) -> None:
        self.sh(name, [sys.executable, *args], **kw)


def fmt_min(m: float) -> str:
    if m < 1:
        return "<1 min"
    if m < 90:
        return f"{m:.0f} min"
    return f"{m / 60:.1f} h"


def fmt_sec(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def exists(root: Path, rel: str) -> bool:
    p = Path(root) / rel
    if p.is_dir():
        return any(p.iterdir())
    return p.exists() and p.stat().st_size > 0


def missing_outputs(stage: Stage, root: Path) -> list:
    return [o for o in stage.outputs if not exists(root, o)]


def is_cached(stage: Stage, root: Path) -> bool:
    return not missing_outputs(stage, root)


# ============================================================================================================== stage graph
def _wf_args(cfg: dict, model: str, workers: int, tag: str | None = None) -> list:
    w = cfg["walk_forward"]
    return ["run_walk.py", "--model", model, "--tag", tag or w["tag"], "--window", w["window"], "--refit", w["refit"], "--prior", w["prior"], "--per-expiry", w["per_expiry"],
            "--workers", workers, "--chunks", w["chunks"], "--burn", w["burn"], "--nfev-cold", w["nfev_cold"], "--nfev-warm", w["nfev_warm"]]


def build_stages(cfg: dict, quick: bool = False) -> dict:
    return build_quick_stages(cfg) if quick else build_full_stages(cfg)


def build_full_stages(cfg: dict) -> dict:
    P, wf, tr, ex = cfg["paths"], cfg["walk_forward"], cfg["trading"], cfg["execution"]
    res, R, tag = P["walk_results"], P["results"], wf["tag"]
    S: dict[str, Stage] = {}

    def add(name, title, outputs, deps, run, est, basis, group, **kw):
        if name in S:
            raise ValueError(f"duplicate stage {name}")
        S[name] = Stage(name, title, list(outputs), list(deps), run, est, basis, group, **kw)

    # ---------------------------------------------------------------- data and tests
    def run_data(c: Ctx):
        raw = c.root / P["raw_data"]
        n = len(list(raw.glob("*.csv"))) if raw.exists() else 0
        if n < cfg["data"]["min_raw_files"]:
            c.log(f"  only {n} raw files (< {cfg['data']['min_raw_files']}): running the resumable downloaders")
            c.py("data_check", "-m", "dh2.fetch_nse_old", "--workers", c.workers)
            c.py("data_check", "-m", "dh2.fetch_nse", "--workers", c.workers)
        if str(c.root) not in sys.path:
            sys.path.insert(0, str(c.root))
        from dh2.data import load_all
        Sx = load_all(workers=c.workers)
        c.log(f"  {len(Sx)} surfaces, {Sx[0].day} .. {Sx[-1].day} ({n} raw files); configured: {cfg['data']['expected_files']} files, {cfg['data']['first_day']} .. {cfg['data']['last_day']}")
        if str(Sx[0].day) != str(cfg["data"]["first_day"]) or str(Sx[-1].day) != str(cfg["data"]["last_day"]):
            c.log("  NOTE: the data range differs from data.first_day / data.last_day in the config (new files downloaded?); the shipped results cover the configured range")

    def raw_missing(c: Ctx) -> bool:
        raw = c.root / P["raw_data"]
        return (len(list(raw.glob("*.csv"))) if raw.exists() else 0) < cfg["data"]["min_raw_files"]

    add("data_check", "Check the raw NSE files and the surface cache (never downloads unless raw files are missing)", [P["surface_cache"]], [], run_data, 0.5,
        "seconds when cached; building the cache from the raw files takes minutes (dh2/data.py); the downloaders (only if raw files are missing) take hours", "data", network=raw_missing)

    def run_tests(c: Ctx):
        (c.logdir / "tests_fast.log").unlink(missing_ok=True)
        c.py("tests_fast", "-m", "pytest", "tests", "-m", "not slow", "-q", "-p", "no:cacheprovider", "--color=no")      # the output file appears only on success
        (c.logdir / "tests_fast.txt").write_text((c.logdir / "tests_fast.log").read_text())

    add("tests_fast", "Fast tests (pricer cross-checks, PINN, calibration, statistics, this pipeline)", [f"{P['run_logs']}/tests_fast.txt"], ["data_check"], run_tests, 0.5,
        "measured: about 17 s on this machine with warm caches (about 170 fast tests pass, 14 slow ones are deselected, 2 are expected failures); a cold cache adds the one-off surface build", "tests")

    # ---------------------------------------------------------------- walk-forwards and forecasts
    for m in wf["models"]:
        add(f"walk_{m}", f"Walk-forward of the {m} model ({wf['window']}-day window, refit every {wf['refit']} days)", [f"{res}/walk_{m}_{tag}.pkl"], ["data_check"],
            (lambda c, m=m: c.py(f"walk_{m}", *_wf_args(cfg, m, c.workers))), 150,
            "about 1 h each with 8-9 workers on an idle machine (README, brief: whole programme about 22 h); about 2.5x that with 2 workers (extrapolated, not measured)", "walk-forward")
        add(f"forecasts_{m}", f"Forecast variants of the {m} model (hybrids.py: smile, carry, revert, lev, carry_res, ...)", [f"{res}/hyb_{m}_{tag}.pkl"], [f"walk_{m}"],
            (lambda c, m=m: c.py(f"forecasts_{m}", "hybrids.py", "--model", m, "--tag", tag, "--workers", c.workers)), 5,
            "the hyb_* file was written about 1 minute after its walk file with 8 workers (file times); about 4x with 2 workers", "walk-forward")

    heads = [f"forecasts_{m}" for m in wf["models"]]

    def run_benchmark(c: Ctx):
        sys.path.insert(0, str(c.root))
        from dh2 import report
        out = report.build_forecast_tables(cfg, c.root)
        runs = " ".join(cfg["forecasts"]["headline_runs"])
        (c.logdir / "final_table_run.log").unlink(missing_ok=True)
        c.py("benchmark_table", "final_table.py", *cfg["forecasts"]["headline_runs"], logname="final_table_run.log")
        text = (c.logdir / "final_table_run.log").read_text()
        (c.logdir / "final_table.txt").write_text(text)
        bad = report.crosscheck_final_table(out["models"], text)
        if bad:
            raise StageError("results/tables/forecast_scores.csv disagrees with the existing final_table.py:\n  " + "\n  ".join(bad))
        c.log(f"  forecast_scores.csv reproduces final_table.py ({runs}) to 3 decimals")

    add("benchmark_table", "Model and benchmark table on the test period (final_table.py logic, paired block-bootstrap intervals), per-year means, robustness over saved runs",
        [f"{R}/tables/forecast_scores.csv", f"{R}/tables/forecast_by_year.csv", f"{R}/tables/forecast_daily_rmse.csv", f"{R}/tables/robustness_runs.csv", f"{P['run_logs']}/final_table.txt"],
        heads, run_benchmark, 0.2, "measured: 4.3 s (the table, the per-year means, 40 robustness bootstraps and the final_table.py cross-check)", "evaluation")

    def run_trading(c: Ctx):
        t = [tr["model"], tr["tag"]]
        base = ["--model", tr["model"], "--tag", tr["tag"], "--top", tr["top"]]
        c.py("trading_tests", "trade_test.py", *base)
        c.py("trading_tests", "trade_test.py", *base, "--lag", 1)
        c.py("trading_tests", "trade_robust.py", f"trades_{t[0]}_{t[1]}.pkl", f"trades_{t[0]}_{t[1]}_lag1.pkl")
        c.py("trading_tests", "trade_matched.py")

    sfx = f"{tr['model']}_{tr['tag']}"
    add("trading_tests", "Trading test: one trade at a time, signal-day close and one day later, robustness, matched comparison",
        [f"{res}/trade_test_{sfx}.json", f"{res}/trade_test_{sfx}_lag1.json", f"{res}/trade_robust_{sfx}.json", f"{res}/trade_robust_{sfx}_lag1.json", f"{res}/trade_matched_{sfx}.json"],
        [f"walk_{tr['model']}"], run_trading, 3, "measured: 2.2 min for the four commands (re-run in this session; all five JSON files and both trade pickles came out byte-identical)", "evaluation")

    # ---------------------------------------------------------------- pricer validation, PINN
    def run_pricer(c: Ctx):
        c.py("pricer_validation", "results/pricer_validation/make_validation.py")
        c.py("pricer_validation", "results/pricer_validation/cf_scan_mpmath.py", logname="cf_scan_mpmath.log")

    add("pricer_validation", "Pricer validation: project pricer against COS, Lewis and Monte Carlo (and the mpmath characteristic-function scan)",
        [f"{R}/pricer_validation/errors.csv", f"{R}/pricer_validation/worst_cases.csv", f"{R}/pricer_validation/timing.json"], ["data_check"], run_pricer, 16,
        "results/pricer_validation/README.md: make_validation.py about 10 min, cf_scan_mpmath.py about 6 min, one process", "pricer")

    ck_names = cfg["pinn"]["shipped_checkpoints"]
    ck = P["pinn_checkpoints"]

    def run_ckpt(c: Ctx):
        miss = [n for n in ck_names if not exists(c.root, f"{ck}/{n}.pt")]
        if not c.retrain_pinn:
            raise StageError(f"shipped PINN checkpoints missing: {miss} in {ck}. The pipeline only validates the shipped checkpoints; pass --retrain-pinn to train the missing ones "
                             "(about 48 GPU minutes in all, dh2/pinn/experiments.py)")
        trainable = [n for n in miss if n not in ("final", "final_greeks")]
        if trainable:
            c.py("pinn_checkpoints", "-m", "dh2.pinn.experiments", *trainable, env_extra={"OMP_NUM_THREADS": str(c.workers)})
        import shutil
        for dst, src in (("final", "dat_s2"), ("final_greeks", "hyb_s2")):          # results/pinn/README.md: final.pt = dat_s2, final_greeks.pt = hyb_s2
            if not exists(c.root, f"{ck}/{dst}.pt") and exists(c.root, f"{ck}/{src}.pt"):
                shutil.copyfile(c.root / ck / f"{src}.pt", c.root / ck / f"{dst}.pt")

    add("pinn_checkpoints", "PINN checkpoints (shipped; trained only with --retrain-pinn)", [f"{ck}/{n}.pt" for n in ck_names], ["walk_dh"], run_ckpt, 48,
        "results/pinn/README.md: 48 GPU minutes in all for the final runs (only when --retrain-pinn and a checkpoint is missing)", "pinn")

    pt = f"{R}/pinn/tables"

    def run_pinn_val(c: Ctx):
        for st in cfg["pinn"]["validation_stages"]:
            c.py("pinn_validation", "-m", "dh2.pinn.validate", st, env_extra={"OMP_NUM_THREADS": str(c.workers)})

    add("pinn_validation", "PINN validation from the shipped checkpoints: held-out accuracy, Greeks, arbitrage, speed, PDE residual, domain, figures",
        [f"{pt}/ablation_summary.csv", f"{pt}/data_efficiency.csv", f"{pt}/heldout_metrics_all_runs.csv", f"{pt}/bucket_errors_all_runs.csv", f"{pt}/greeks_errors.csv",
         f"{pt}/arbitrage_checks.csv", f"{pt}/speed.csv", f"{pt}/test_period_domain_flags.csv", f"{pt}/pde_verification_summary.csv"] + [f"{R}/pinn/figures/{n}.png" for n in (
             "ablation_bars", "data_efficiency", "error_heatmaps", "greeks_scatter", "pde_residual_maps", "smile_example", "speed", "training_curves")],
        ["pinn_checkpoints", "walk_dh"], run_pinn_val, 4, "results/pinn/README.md: about 3 minutes of validation", "pinn")

    # ---------------------------------------------------------------- C1 stages
    d = f"{R}/dq"
    add("dq", "Data-quality report: quotes surviving each filter, per year and on ten sample days", [f"{d}/data_quality_by_year.csv", f"{d}/data_quality_by_day.csv", f"{d}/data_quality_sample_days.csv"],
        ["data_check"], (lambda c: c.py("dq", "-m", "dh2.dq", "--workers", c.workers)), 1, "measured: 46 s with 2 workers (re-run in this session; all outputs byte-identical to the saved ones)", "C1")
    add("noarb", "No-arbitrage diagnostics (monotonicity, butterfly, calendar) and the scoring sensitivity", [f"{d}/noarb_by_year.csv", f"{d}/noarb_sensitivity.csv", f"{d}/NOARB.md"],
        ["data_check", "forecasts_dhj"], (lambda c: c.py("noarb", "-m", "dh2.noarb")), 0.2, "measured: 2.2 s (re-run in this session; outputs byte-identical)", "C1")
    add("vix_data", "India VIX history (downloaded only if data/vix/india_vix.csv is missing)", [P["vix_csv"]], [], (lambda c: c.py("vix_data", "fetch_vix.py", "--source", "auto", "--workers", c.workers)), 2,
        "fetch_vix.py reads one NSE file per trading day (network)", "C1", network=True)
    v = f"{R}/vix"
    add("vix", "Model-implied 30-day volatility against India VIX", [f"{v}/vix_vs_model_stats.csv", f"{v}/vix_vs_model_daily.csv", f"{v}/vix_leadlag.csv", f"{v}/vix_vs_model_by_year.csv"] + [
        f"{v}/{n}.png" for n in ("fig_timeseries", "fig_scatter_levels", "fig_scatter_changes", "fig_leadlag", "fig_gap_and_rolling_corr")], ["vix_data", "walk_dh", "walk_dhj"],
        (lambda c: c.py("vix", "-m", "dh2.vixcompare")), 0.2, "measured: 2.5 s (re-run in this session; CSVs and figures byte-identical)", "C1")
    s = f"{R}/svi"
    add("svi", "Raw-SVI reference per expiry against the walk-forward Double Heston fits (118 sampled test days)", [f"{s}/sample_days.csv", f"{s}/svi_fits.csv", f"{s}/svi_vs_dh_by_day.csv", f"{s}/svi_vs_dh_by_bucket.csv",
        f"{s}/svi_vs_dh_quote_errors.csv.gz"] + [f"{s}/{n}.png" for n in ("svi_example_smile", "svi_vs_dh_buckets", "svi_vs_dh_rmse_by_day")], ["walk_dh", "walk_dhj"],
        (lambda c: c.py("svi", "-m", "dh2.svi", "--step", cfg["evaluation"]["paired_sample"]["every_nth"], "--workers", c.workers)), 1,
        "measured: 55 s with 2 workers (re-run in this session; outputs identical, the .csv.gz only differs in its gzip timestamp)", "C1")
    add("stats", "Paired Wilcoxon and Diebold-Mariano tests on the test period", [f"{R}/stats/tests_test_period.csv", f"{R}/stats/tests_test_period.md"], heads,
        (lambda c: c.py("stats", "-m", "dh2.stats")), 0.2, "measured: 0.7 s (re-run in this session; outputs byte-identical)", "C1")

    # ---------------------------------------------------------------- C2 stages (the scripts fix their own 3 workers in c2_common.py)
    c2 = ex.get("workers_c2", 3)
    add("daily_fits", "Independent same-day fits of four models on the paired sample (prerequisite of the C2 experiments)", [f"{R}/_cache/daily_fits_v1.pkl"], ["data_check"],
        (lambda c: c.py("daily_fits", "exp_daily_fits.py")), 20,
        f"cache file times: C2 queue started 12:17, daily_fits_v1.pkl written 12:30 (3 workers, machine shared with other jobs); the script uses {c2} workers", "C2")
    a = f"{R}/attribution"
    add("attribution", "DH1 to DH2.0 step-by-step attribution waterfall", [f"{a}/waterfall.csv", f"{a}/per_day.csv", f"{a}/replication_check.csv", f"{a}/data_differences.csv", f"{a}/waterfall.png"],
        ["daily_fits", "walk_dh", "walk_dhj", "forecasts_dhj"], (lambda c: c.py("attribution", "exp_attribution.py")), 100,
        "cache file times: about 1.5 h for the steps 0-2 fits (3 workers, shared machine); needs the old research repository's nifty_comparison_detail.csv for the replication table", "C2")
    x = f"{R}/crosssection"
    add("crosssection", "Cross-sectional out-of-sample tests (random 80/20 split and maturity split)", [f"{x}/summary.csv", f"{x}/pairs.csv", f"{x}/per_day.csv", f"{x}/per_split.csv", f"{x}/crosssection.png"],
        ["data_check"], (lambda c: c.py("crosssection", "exp_crosssection.py")), 280,
        "cache file times: about 4.7 h (3 workers, shared machine, upper bound)", "C2")
    b = f"{R}/buckets"
    add("buckets", "Per-bucket same-day implied-vol RMSE for all models", [f"{b}/bucket_rmse.csv", f"{b}/bucket_summary.csv", f"{b}/heatmaps_rmse.png", f"{b}/heatmaps_counts.png"],
        ["daily_fits", "walk_dh", "walk_dhj"], (lambda c: c.py("buckets", "exp_buckets.py")), 10, "progress file times: minutes after the daily fits (reads the cached fits)", "C2")
    i = f"{R}/identifiability"
    add("identifiability", "Identifiability: Jacobian condition numbers, standard errors, multi-start solutions", [f"{i}/summary.csv", f"{i}/per_parameter.csv", f"{i}/per_day.csv", f"{i}/identifiability.png"],
        ["daily_fits"], (lambda c: c.py("identifiability", "exp_identifiability.py")), 90, "cache file times: about 1.5 h (3 workers, shared machine)", "C2")

    # ---------------------------------------------------------------- PINN-accelerated calibration
    pc = f"{R}/pinn_calib"
    add("pinn_calib_design", "PINN-accelerated calibration: design on development days", [f"{pc}/frozen_config.json", f"{pc}/design_global_configs.csv", f"{pc}/design_variant_polish_grid.csv", f"{pc}/design_per_day.csv"],
        ["pinn_checkpoints", "walk_dh"], (lambda c: c.py("pinn_calib_design", "exp_pinn_calib.py", "design", env_extra={"OMP_NUM_THREADS": str(c.workers)})), 30,
        "results/pinn_calib/README.md: about 30 minutes, one process", "pinn calibration")
    add("pinn_calib_test", "PINN-accelerated calibration: the frozen design once on the paired test days", [f"{pc}/_cache/test_days.pkl"], ["pinn_calib_design", "svi"],
        (lambda c: c.py("pinn_calib_test", "exp_pinn_calib.py", "test", env_extra={"OMP_NUM_THREADS": str(c.workers)})), 15, "results/pinn_calib/README.md: about 15 minutes", "pinn calibration")
    add("pinn_calib_report", "PINN-accelerated calibration: tables and figures", [f"{pc}/headline.json", f"{pc}/accuracy_by_method.csv", f"{pc}/accuracy_paired.csv", f"{pc}/timing_by_method.csv", f"{pc}/speedup.csv",
        f"{pc}/per_day.csv"] + [f"{pc}/{n}.png" for n in ("fig1_rmse_paired", "fig2_cost_ratio", "fig3_time", "fig4_parameters", "fig5_rmse_difference_over_time", "fig6_design_development_days")],
        ["pinn_calib_test"], (lambda c: c.py("pinn_calib_report", "exp_pinn_calib.py", "report")), 0.2, "measured: 3.9 s from the cached per-day results (re-run in this session; outputs byte-identical)", "pinn calibration")

    # ---------------------------------------------------------------- UI adapter, metrics, figures, report
    add("ui_adapter", "UI adapter: today.json with the latest calibrated parameters in the website's format", [P["today_json"]], ["walk_dh", "walk_dhj"],
        (lambda c: c.py("ui_adapter", "-m", "dh2.ui_adapter", "--tag", tag)), 0.2, "measured: 0.5 s (re-run in this session; today.json byte-identical)", "report")

    from .report import TABLE_NAMES
    upstream = ["benchmark_table", "trading_tests", "pricer_validation", "pinn_validation", "dq", "noarb", "vix", "svi", "stats", "attribution", "crosssection", "identifiability", "buckets",
                "pinn_calib_report", "ui_adapter"]

    def run_metrics(c: Ctx):
        sys.path.insert(0, str(c.root))
        from dh2 import report
        m, tabs = report.build_all(cfg, c.root, "configs/nifty.yaml")
        c.log(f"  {len(m.items)} numbers ({sum(1 for r in m.items.values() if r['kind'] == 'read')} read, {sum(1 for r in m.items.values() if r['kind'] == 'derived')} derived), {len(tabs)} tables")

    add("metrics_tables", "Metrics collector (results/metrics.json, every number with its source and column) and the headline tables as CSV, Markdown and LaTeX",
        [f"{R}/metrics.json"] + [f"{R}/tables/{n}.{e}" for n in TABLE_NAMES for e in ("csv", "md", "tex")], upstream, run_metrics, 0.2, "measured: 0.3 s", "report")

    from .report_figures import REGISTRY

    def run_figs(c: Ctx):
        sys.path.insert(0, str(c.root))
        from dh2 import report_figures
        built = report_figures.build_all(cfg, c.root)
        c.log(f"  {len(built)} figures (png + pdf) and INDEX.md in {P['figures']}")

    add("figures", "Presentation figures (300 dpi PNG and PDF): new summary / comparison / smile / pricer figures and re-exports of the work packages' figures, with INDEX.md",
        [f"{P['figures']}/INDEX.md", f"{P['tables']}/smile_figure_days.csv"] + [f"{P['figures']}/{f.stem}.{e}" for f in REGISTRY for e in ("png", "pdf")], ["metrics_tables"], run_figs, 0.2, "measured: 5.4 s for 33 figures (png + pdf)", "report")

    def run_report(c: Ctx):
        sys.path.insert(0, str(c.root))
        from dh2 import report
        m = report.build_metrics(cfg, c.root)
        tabs = report.build_tables(m, cfg)
        (c.root / R / "report.md").write_text(report.render_report(m, cfg, tabs))

    add("report", "Plain-numbers report (results/report.md), every number from results/metrics.json's sources", [f"{R}/report.md"], ["figures"], run_report, 0.2, "measured: 0.2 s", "report")

    # ---------------------------------------------------------------- optional (named explicitly with --stages)
    ab = cfg.get("ablations", [])

    def run_ablations(c: Ctx):
        for a_ in ab:
            if exists(c.root, f"{res}/hyb_dhj_{a_['tag']}.pkl"):
                continue
            c.py(f"ablation_{a_['tag']}", *_wf_args(cfg, "dhj", c.workers, a_["tag"]), *a_["args"])      # a later flag overrides the default one
            c.py(f"ablation_{a_['tag']}", "hybrids.py", "--model", "dhj", "--tag", a_["tag"], "--workers", c.workers)

    add("ablation_runs", "Robustness reruns of the jump model (windows, refit schedules, priors, memory priors): the saved runs behind the robustness table", [f"{res}/hyb_dhj_{a_['tag']}.pkl" for a_ in ab],
        ["forecasts_dhj"], run_ablations, 150 * max(1, len(ab)), f"about 1 h each with 8-9 workers (queue1/queue4.sh), {len(ab)} runs; about 2.5x with 2 workers (extrapolated)", "walk-forward", optional=True,
        note="the 'slowrefit' and 'wb' (wide bounds, DH2_WIDE=1) and 'dhjs' runs are not regenerated here")
    add("comparison_note", "Regenerate COMPARISON_1.0_vs_2.0.md and outputs/comparison.json (update_comparison.py; overwrites the note)", ["outputs/comparison.json"], ["forecasts_dhj"],
        (lambda c: c.py("comparison_note", "update_comparison.py")), 3, "seconds to minutes (estimate)", "legacy", optional=True)
    add("legacy_figures", "The five original figures of make_figures.py (outputs/figures); superseded by the numbered figures", ["outputs/figures/error_by_year.png"], ["forecasts_dhj"],
        (lambda c: c.py("legacy_figures", "make_figures.py")), 2, "seconds (estimate)", "legacy", optional=True)
    return S


def build_quick_stages(cfg: dict) -> dict:
    Q = cfg["paths"]["results_quick"]
    qc = cfg["quick"]
    S: dict[str, Stage] = {}

    def add(name, title, outputs, deps, fn, est, basis):
        def run(c: Ctx, fn=fn, name=name):
            sys.path.insert(0, str(c.root))
            from dh2 import quick
            kw = {"timings": dict(c.timings)} if name == "q_report" else {}
            getattr(quick, fn)(cfg, c.root, c.workers, **kw)
        S[name] = Stage(name, title, outputs, deps, run, est, basis, "quick")

    t = f"{Q}/tables"
    add("q_data", f"Real surfaces of the window {qc['start']} .. {qc['end']} (cached; no download)", [f"{t}/quick_window.csv"], [], "q_data", 0.2, "seconds (cached surfaces)")
    add("q_walk", f"Mini walk-forward of {', '.join(qc['models'])} (window {qc['window']}, refit {qc['refit']}, small budgets)", [f"{t}/quick_walk_forward.csv"] + [f"{Q}/work/walk_{m}.pkl" for m in qc["models"]],
        ["q_data"], "q_walk", 5, "measured: 5.1 min with 2 workers (3 models x 2 chunks of the 79-day window; about 500 s of CPU)")
    add("q_forecast", "Forecast variants and the mini forecast table against the model-free benchmarks (paired bootstrap)", [f"{t}/forecast_models.csv", f"{t}/forecast_daily_rmse.csv"], ["q_walk"], "q_forecast", 0.2, "measured: 5 s (7 forecast variants per day for 3 models, 2 workers)")
    add("q_pricer", "Mini pricer cross-check: project pricer against COS (and Lewis on a subset) on real chains", [f"{t}/pricer_cross_check.csv"], ["q_walk"], "q_pricer", 0.1, "measured: 0.6 s")
    add("q_pinn", "PINN surrogate validation from the shipped checkpoints on real chains", [f"{t}/pinn_surrogate_check.csv"], ["q_walk"], "q_pinn", 0.1, "measured: 0.4 s (CPU)")
    add("q_svi", "Mini SVI reference on a few real days", [f"{t}/svi_reference.csv"], ["q_walk"], "q_svi", 0.1, "measured: 1.6 s")
    add("q_stats", "Wilcoxon and Diebold-Mariano on the mini forecast errors", [f"{t}/tests_quick.csv"], ["q_forecast"], "q_stats", 0.1, "measured: under 0.1 s")
    add("q_report", "Quick report (same layout as results/report.md), metrics and figure, all marked demo-scale", [f"{Q}/report.md", f"{Q}/metrics_quick.json", f"{Q}/figures/quick_summary.png", f"{Q}/figures/quick_summary.pdf"],
        ["q_forecast", "q_pricer", "q_pinn", "q_svi", "q_stats"], "q_report", 0.1, "measured: 1.0 s")
    return S


# ============================================================================================================== graph checks
def topo_order(stages: dict, names=None) -> list:
    """Dependency order (Kahn) of `names` (default: all stages). Raises on a cycle or an unknown dependency."""
    names = list(stages) if names is None else list(names)
    pos = {n: i for i, n in enumerate(stages)}
    names.sort(key=pos.get)
    indeg = {n: 0 for n in names}
    for n in names:
        for d in stages[n].deps:
            if d not in stages:
                raise ValueError(f"stage {n} depends on unknown stage {d}")
            if d in indeg:
                indeg[n] += 1
    out, ready = [], [n for n in names if indeg[n] == 0]
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in names:
            if n in stages[m].deps:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        ready.sort(key=pos.get)
    if len(out) != len(names):
        raise ValueError(f"cycle in the stage graph among {sorted(set(names) - set(out))}")
    return out


def validate_graph(stages: dict) -> None:
    topo_order(stages)
    seen = {}
    for n, s in stages.items():
        if not s.outputs:
            raise ValueError(f"stage {n} declares no outputs")
        for o in s.outputs:
            if o in seen:
                raise ValueError(f"output {o} declared by both {seen[o]} and {n}")
            seen[o] = n


def closure(stages: dict, names) -> list:
    out, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n in out:
            continue
        if n not in stages:
            raise SystemExit(f"unknown stage {n!r}; known stages: {', '.join(stages)}")
        out.add(n)
        todo += stages[n].deps
    return list(out)


@dataclass
class Item:
    stage: Stage
    action: str        # run | skip
    reason: str


def make_plan(stages: dict, requested: list | None, force: bool, root: Path) -> list:
    """Plan in dependency order. Requested stages (all non-optional ones when none are named) run when forced or when an output is missing;
    stages that are only dependencies run only when an output is missing."""
    explicit = list(requested) if requested else [n for n, s in stages.items() if not s.optional]
    for n in explicit:
        if n not in stages:
            raise SystemExit(f"unknown stage {n!r}; known stages: {', '.join(stages)}")
    names = topo_order(stages, closure(stages, explicit))
    plan = []
    for n in names:
        s = stages[n]
        miss = missing_outputs(s, root)
        if n in explicit and force:
            plan.append(Item(s, "run", "forced"))
        elif miss:
            plan.append(Item(s, "run", f"missing: {', '.join(miss[:2])}" + (f" (+{len(miss) - 2} more)" if len(miss) > 2 else "")))
        else:
            plan.append(Item(s, "skip", "cached (all outputs exist)"))
    return plan


def stale_downstream(stages: dict, plan: list) -> list:
    """Stages that are skipped although a dependency of theirs will run (their outputs may be stale)."""
    runs = {i.stage.name for i in plan if i.action == "run"}
    out = []
    for i in plan:
        if i.action == "skip" and any(d in runs for d in i.stage.deps):
            out.append(i.stage.name)
    return out


def needs_network(stage: Stage, ctx: Ctx) -> bool:
    n = stage.network
    return bool(n(ctx)) if callable(n) else bool(n)


# ============================================================================================================== output
def print_list(stages: dict, ctx: Ctx, root: Path, quick: bool) -> None:
    thr = ctx.cfg["execution"]["confirm_over_minutes"]
    ctx.log(f"{'Quick (demo-scale)' if quick else 'Full'} pipeline: {len(stages)} stages, workers {ctx.workers}; estimated minutes are for this configuration; 'state' = outputs present now\n")
    ctx.log(f"{'#':>2} {'stage':<20}{'state':<14}{'est. time':<11}{'outputs':>7}  {'depends on':<34}")
    for k, (n, s) in enumerate(stages.items(), 1):
        miss = missing_outputs(s, root)
        state = "cached" if not miss else f"MISSING {len(miss)}/{len(s.outputs)}"
        flags = (" [optional]" if s.optional else "") + (" [network]" if needs_network(s, ctx) else "") + (" [EXPENSIVE]" if s.est_min >= thr else "")
        ctx.log(f"{k:>2} {n:<20}{state:<14}{fmt_min(s.est_min):<11}{len(s.outputs):>7}  {', '.join(s.deps) or '-':<34}{flags}")
        ctx.log(f"     {s.title}")
        ctx.log(f"     cost basis: {s.basis}" + (f"; {s.note}" if s.note else ""))
    tot = sum(s.est_min for s in stages.values() if not s.optional and missing_outputs(s, root))
    ctx.log(f"\nEstimated time of the stages that would run now (cached stages are skipped): {fmt_min(tot)}. --force reruns a stage; --stages a,b selects stages.")


def print_plan(plan: list, ctx: Ctx, stages: dict) -> float:
    thr = ctx.cfg["execution"]["confirm_over_minutes"]
    ctx.log(f"Plan ({'quick, demo-scale' if ctx.quick else 'full'}; workers {ctx.workers}; logs in {ctx.logdir.relative_to(ctx.root)}):")
    total = 0.0
    for k, it in enumerate(plan, 1):
        s = it.stage
        if it.action == "run":
            total += s.est_min
            flag = "  <-- EXPENSIVE" if s.est_min >= thr else ""
            flag += "  <-- needs the network" if needs_network(s, ctx) else ""
            ctx.log(f"  {k:>2}. RUN  {s.name:<20} est {fmt_min(s.est_min):<9} {it.reason}{flag}")
        else:
            ctx.log(f"  {k:>2}. skip {s.name:<20} {'':<13} {it.reason}")
    ctx.log(f"  Estimated total of the stages that will run: {fmt_min(total)}")
    stale = stale_downstream(stages, plan)
    if stale:
        ctx.log(f"  WARNING: {', '.join(stale)} are cached but depend on a stage that will rerun; rerun them with --stages ... --force if their inputs change.")
    return total


# ============================================================================================================== execution
@dataclass
class Result:
    name: str
    status: str            # ran | skipped | failed | blocked
    seconds: float
    est_min: float
    detail: str = ""


def execute_plan(plan: list, ctx: Ctx, keep_going: bool = False) -> list:
    results, failed = [], set()
    todo = [i for i in plan if i.action == "run"]
    n = 0
    for it in plan:
        s = it.stage
        if it.action == "skip":
            results.append(Result(s.name, "skipped", 0.0, s.est_min, it.reason))
            continue
        n += 1
        if any(d in failed for d in s.deps):
            results.append(Result(s.name, "blocked", 0.0, s.est_min, "a dependency failed"))
            failed.add(s.name)
            continue
        ctx.log(f"\n[{n}/{len(todo)}] {s.name}: {s.title}\n      estimated {fmt_min(s.est_min)}; started {datetime.now():%H:%M:%S}")
        t0 = time.time()
        try:
            if s.run is None:
                raise StageError(f"{s.name} has no run function")
            s.run(ctx)
            miss = missing_outputs(s, ctx.root)
            if miss:
                raise StageError(f"{s.name} finished but its declared outputs are missing: {miss[:4]}")
            dt = time.time() - t0
            ctx.timings[s.name] = dt
            results.append(Result(s.name, "ran", dt, s.est_min))
            ctx.log(f"      done in {fmt_sec(dt)}")
        except (StageError, Exception) as e:           # noqa: BLE001 (a failed stage is reported with its message, never swallowed)
            dt = time.time() - t0
            results.append(Result(s.name, "failed", dt, s.est_min, str(e)))
            failed.add(s.name)
            ctx.log(f"      FAILED after {fmt_sec(dt)}: {e}")
            if not keep_going:
                break
    return results


def print_summary(results: list, ctx: Ctx, wall: float) -> None:
    ctx.log("\nSummary (real elapsed time; 'est.' is the plan's estimate):")
    ctx.log(f"  {'stage':<20}{'status':<9}{'elapsed':<12}{'est.':<9}")
    for r in results:
        ctx.log(f"  {r.name:<20}{r.status:<9}{fmt_sec(r.seconds) if r.status in ('ran', 'failed') else '-':<12}{fmt_min(r.est_min) if r.status in ('ran', 'failed') else '-':<9}")
    ran = [r for r in results if r.status == "ran"]
    ctx.log(f"  ran {len(ran)} stage(s), skipped {sum(r.status == 'skipped' for r in results)} (cached), failed {sum(r.status == 'failed' for r in results)}, "
            f"blocked {sum(r.status == 'blocked' for r in results)}; wall time {fmt_sec(wall)}")


def write_last_run(results: list, ctx: Ctx, wall: float, args) -> Path:
    ctx.logdir.mkdir(parents=True, exist_ok=True)
    p = ctx.logdir / "last_run.json"
    rec = {"finished": datetime.now().isoformat(timespec="seconds"), "quick": ctx.quick, "force": ctx.force, "workers": ctx.workers, "wall_seconds": wall,
           "stages": [{"name": r.name, "status": r.status, "seconds": r.seconds, "estimated_minutes": r.est_min, "detail": r.detail[:300]} for r in results]}
    if ctx.quick:
        rec = {"notice": "DEMO-SCALE RUN, NOT THE HEADLINE RESULT", **rec}
    p.write_text(json.dumps(rec, indent=1))
    with open(ctx.logdir / "run_history.jsonl", "a") as f:          # one line per invocation, so earlier timings are not lost
        f.write(json.dumps({**rec, "stages": [{k: v for k, v in s.items() if k != "detail"} for s in rec["stages"] if s["status"] != "skipped"]}) + "\n")
    return p


# ============================================================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dh2.run_all", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/nifty.yaml")
    ap.add_argument("--quick", action="store_true", help="demo-scale run on a short real window; writes only to results_quick/")
    ap.add_argument("--force", action="store_true", help="rerun the selected stages even if their outputs exist (a forced walk-forward takes hours)")
    ap.add_argument("--stages", default="", help="comma-separated stage names (plus any dependency whose outputs are missing)")
    ap.add_argument("--list", action="store_true", help="list every stage with its state, estimated cost, outputs and dependencies, then exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--retrain-pinn", action="store_true", help="train the PINN checkpoints that are missing (the shipped ones are only validated)")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before expensive stages")
    ap.add_argument("--keep-going", action="store_true", help="continue with independent stages after a failure")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (cannot exceed execution.max_workers)")
    return ap


def main(argv=None, out=print) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    ctx = Ctx(cfg, ROOT, quick=args.quick, force=args.force, retrain_pinn=args.retrain_pinn, workers=args.workers, out=out)
    stages = build_stages(cfg, args.quick)
    validate_graph(stages)
    if args.list:
        print_list(stages, ctx, ctx.root, args.quick)
        return 0
    requested = [x.strip() for x in args.stages.split(",") if x.strip()]
    plan = make_plan(stages, requested, args.force, ctx.root)
    total = print_plan(plan, ctx, stages)
    if args.dry_run:
        return 0
    thr = cfg["execution"]["confirm_over_minutes"]
    big = [it.stage.name for it in plan if it.action == "run" and (it.stage.est_min >= thr or needs_network(it.stage, ctx))]
    if big and not args.yes:
        msg = f"\nThe plan contains expensive or network stages: {', '.join(big)}."
        if sys.stdin is not None and sys.stdin.isatty():
            ans = input(msg + " Continue? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                ctx.log("Not started.")
                return 1
        else:
            ctx.log(msg + " Not started: re-run with --yes to confirm (or --stages to choose cheaper stages).")
            return 2
    if not any(i.action == "run" for i in plan):
        ctx.log("\nNothing to run: every selected stage is cached (--force reruns, --list shows the state).")
        return 0
    t0 = time.time()
    results = execute_plan(plan, ctx, keep_going=args.keep_going)
    wall = time.time() - t0
    print_summary(results, ctx, wall)
    p = write_last_run(results, ctx, wall, args)
    ctx.log(f"  timing record: {p.relative_to(ctx.root)}")
    return 1 if any(r.status in ("failed", "blocked") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
