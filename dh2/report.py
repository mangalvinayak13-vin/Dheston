"""Metrics collector, headline tables and the plain-numbers report of the one-command pipeline (python -m dh2.run_all).

Three jobs, all reading files that other stages wrote (nothing here is typed in):

1. build_forecast_tables(): the model-comparison table with the paired block-bootstrap intervals of final_table.py's logic, per-day scores,
   per-year means and the robustness table over every saved run  ->  results/tables/forecast_*.csv, robustness_runs.csv
2. build_metrics(): results/metrics.json. Every number is READ from a result file (CSV column or JSON key) or DERIVED from such cells by an
   arithmetic rule written next to it; each record carries its source file and the exact column / key (or the rule). A missing file, column,
   key or row raises MissingSource: there are no defaults and no silent fallbacks.
3. write_tables() / render_report(): results/tables/*.csv|md|tex (LaTeX booktabs) and results/report.md, all from the metrics above.

Run it through the pipeline (stages `benchmark_table`, `metrics_tables`, `report`) or directly:  python -m dh2.report [--config configs/nifty.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LABELS = {"smile": "yesterday's smile", "smile_scaled": "time-scaled smile",
          "heston_main:carry": "Heston", "heston_main:carry_res": "Heston + misfit",
          "dh_main:carry": "diffusion DH", "dh_main:carry_res": "diffusion DH + misfit",
          "dhj_main:carry": "jump model", "dhj_main:carry_res": "jump model + misfit"}


class MissingSource(RuntimeError):
    """A source file, column, key or row that the metrics need is not there (the collector never invents a value)."""


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


# =================================================================================================================== config
def load_config(path=None) -> dict:
    import yaml
    p = Path(path) if path else ROOT / "configs" / "nifty.yaml"
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def _split_date(cfg) -> date:
    d = cfg["split"]["test"]["start"]
    return d if isinstance(d, date) else date.fromisoformat(str(d))


# =================================================================================================================== 1. forecast tables
def _load_pkl(path: Path):
    if not path.exists():
        raise MissingSource(f"missing {path} (walk-forward / forecast-variant stage not run)")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_forecast_tables(cfg: dict, root: Path = ROOT, out_dir: Path | None = None) -> dict:
    """Model comparison on the test period exactly as final_table.py does it, plus per-day scores, per-year means and the robustness table.

    final_table logic: days = intersection over the headline runs; dev = origin day < 2022-01-01, test = origin day >= 2022-01-01; the
    benchmarks are yesterday's smile and the time-scaled smile; a model's difference is against the benchmark with the lower TEST mean,
    with the 95% moving-block bootstrap interval of tournament.boot."""
    from . import evalpairs as EP
    from .data import load_all
    sys.path.insert(0, str(root))
    from tournament import boot

    out_dir = Path(out_dir) if out_dir else root / cfg["paths"]["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res = root / cfg["paths"]["walk_results"]
    split = _split_date(cfg)
    bc = cfg["evaluation"]["bootstrap"]
    bt = lambda d: boot(d, block=bc["block"], n=bc["n"], seed=bc["seed"])
    runs, variants = cfg["forecasts"]["headline_runs"], cfg["forecasts"]["report_variants"]

    S = load_all()
    H = {n: _load_pkl(res / f"hyb_{n}.pkl") for n in runs}
    common = sorted(set.intersection(*[set(h["carry"]) for h in H.values()]))
    dev = [t for t in common if S[t].day < split]
    test = [t for t in common if S[t].day >= split]
    scaled = {t: EP.pred_smile_scaled(S[t], S[t + 1]) for t in common}
    smile0 = {t: H[runs[0]]["smile"][t].astype(float) for t in common}
    score = lambda p: np.array([EP.score(p[t], S[t + 1])["rmse"] for t in common])
    series = {"smile": score(smile0), "smile_scaled": score(scaled)}
    for n in runs:
        for v in variants:
            series[f"{n}:{v}"] = score({t: H[n][v][t].astype(float) for t in common})
    is_test = np.array([S[t].day >= split for t in common])
    is_dev = ~is_test
    best = min(("smile", "smile_scaled"), key=lambda k: float(np.nanmean(series[k][is_test])))

    rows = []
    for key, sc in series.items():
        is_bench = ":" not in key
        run, var = (key, "none") if is_bench else key.split(":")
        mean, lo, hi = bt(sc[is_test])
        r = {"run": run, "variant": var, "label": LABELS[key], "kind": "benchmark" if is_bench else "model",
             "n_dev_days": int(is_dev.sum()), "n_test_days": int(is_test.sum()), "dev_mean": float(np.nanmean(sc[is_dev])),
             "test_mean": float(np.nanmean(sc[is_test])), "test_mean_ci_lo": lo, "test_mean_ci_hi": hi, "best_benchmark": best,
             "diff_vs_best_benchmark": np.nan, "diff_ci_lo": np.nan, "diff_ci_hi": np.nan}
        if not is_bench:
            d, dlo, dhi = bt(sc[is_test] - series[best][is_test])
            r.update(diff_vs_best_benchmark=d, diff_ci_lo=dlo, diff_ci_hi=dhi)
        rows.append(r)
    models = pd.DataFrame(rows)
    models.to_csv(out_dir / "forecast_scores.csv", index=False)

    years = np.array([S[t].day.year for t in common])
    by_year = pd.DataFrame({"year": sorted(set(years))})
    for key, sc in series.items():
        by_year[key] = [float(np.nanmean(sc[years == y])) for y in by_year["year"]]
    by_year.to_csv(out_dir / "forecast_by_year.csv", index=False)

    daily = pd.DataFrame({"day": [S[t].day for t in common], "split": np.where(is_test, "test", "development")})
    for key, sc in series.items():
        daily[key] = np.round(sc, 6)
    daily.to_csv(out_dir / "forecast_daily_rmse.csv", index=False)

    # robustness: every saved run of the walk-forward (ablations included), as update_comparison.py's "every saved run" table
    rob = []
    for f in sorted(res.glob("hyb_*.pkl")):
        name = f.stem[4:]
        h = _load_pkl(f)
        for v in ("carry", "carry_res"):
            ts = [t for t in h[v] if t in scaled]
            tt = [t for t in ts if S[t].day >= split]
            dd = [t for t in ts if S[t].day < split]
            sc = {t: EP.score(h[v][t].astype(float), S[t + 1])["rmse"] for t in ts}
            ref = {t: EP.score(scaled[t], S[t + 1])["rmse"] for t in tt}
            m_, lo, hi = bt([sc[t] - ref[t] for t in tt]) if len(tt) > 100 else (float("nan"),) * 3
            rob.append({"run": name, "variant": v, "all_mean": float(np.mean(list(sc.values()))), "dev_mean": float(np.mean([sc[t] for t in dd])),
                        "test_mean": float(np.mean([sc[t] for t in tt])), "n_test_days": len(tt),
                        "test_diff_vs_time_scaled_smile": m_, "diff_ci_lo": lo, "diff_ci_hi": hi})
    pd.DataFrame(rob).to_csv(out_dir / "robustness_runs.csv", index=False)
    return {"models": models, "n_common": len(common), "best_benchmark": best}


_FT_LINE = re.compile(r"^(\S+)\s+(carry_res|carry)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]\s*$")


def parse_final_table(text: str) -> dict:
    """{(run, variant): (dev, test, diff, lo, hi)} from the printout of final_table.py."""
    out = {}
    for line in text.splitlines():
        m = _FT_LINE.match(line)
        if m:
            out[(m.group(1), m.group(2))] = tuple(float(m.group(i)) for i in range(3, 8))
    return out


def crosscheck_final_table(models: pd.DataFrame, text: str, tol: float = 6e-4) -> list[str]:
    """Compare results/tables/forecast_scores.csv with the printout of the existing final_table.py (rounded to 3 decimals there).
    Returns the list of disagreements (empty = the table reproduces the existing script's numbers)."""
    ft = parse_final_table(text)
    if not ft:
        return ["could not parse any model row from the final_table.py printout"]
    bad = []
    for (run, var), vals in ft.items():
        r = models[(models.run == run) & (models.variant == var)]
        if len(r) != 1:
            bad.append(f"{run} {var}: not in forecast_scores.csv")
            continue
        r = r.iloc[0]
        mine = (r.dev_mean, r.test_mean, r.diff_vs_best_benchmark, r.diff_ci_lo, r.diff_ci_hi)
        for nm, a, b in zip(("dev", "test", "diff", "lo", "hi"), mine, vals):
            if abs(a - b) > tol:
                bad.append(f"{run} {var} {nm}: table {a:.4f} vs final_table.py {b:.3f}")
    return bad


# =================================================================================================================== 2. metrics
def _py(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (str, int)):
        return v
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return str(v)


class Metrics:
    """The collected numbers. Every add goes through cell / jcell / derived; each stores value, unit, source and the exact locator."""

    def __init__(self, root: Path = ROOT, comment: str | None = None):
        self.root = Path(root)
        self.comment = comment          # results_quick CSVs carry a "# DEMO-SCALE RUN" first line
        self.items: dict[str, dict] = {}
        self._tab: dict[str, pd.DataFrame] = {}
        self._js: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}

    # ---- files
    def _path(self, rel: str) -> Path:
        p = self.root / rel
        if not p.exists():
            raise MissingSource(f"missing source file {rel}")
        if rel not in self.sources:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            self.sources[rel] = {"bytes": p.stat().st_size, "sha256": h}
        return p

    def table(self, rel: str) -> pd.DataFrame:
        if rel not in self._tab:
            self._tab[rel] = pd.read_csv(self._path(rel), comment=self.comment)
        return self._tab[rel]

    def json(self, rel: str) -> dict:
        if rel not in self._js:
            self._js[rel] = json.loads(self._path(rel).read_text())
        return self._js[rel]

    # ---- adds
    def _put(self, mid, value, unit, source, key, kind):
        if mid in self.items:
            raise ValueError(f"duplicate metric id {mid}")
        value = _py(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise MissingSource(f"{mid}: value from {source} ({key}) is not finite")
        self.items[mid] = {"value": value, "unit": unit, "source": source, "key": key, "kind": kind}
        return value

    @staticmethod
    def _match(col: pd.Series, v):
        if isinstance(v, str):
            return col.astype(str) == v
        if isinstance(v, float):
            return pd.Series(np.isclose(col.astype(float), v, rtol=1e-9, atol=0.0), index=col.index)
        return col == v

    def rows(self, rel: str, where: dict | None = None) -> pd.DataFrame:
        df = self.table(rel)
        sel = df
        for c, v in (where or {}).items():
            if c not in df.columns:
                raise MissingSource(f"{rel}: no column {c!r} (columns: {list(df.columns)})")
            sel = sel[self._match(sel[c], v)]
        return sel

    def cell(self, mid: str, rel: str, col: str, unit: str = "", where: dict | None = None):
        df = self.table(rel)
        if col not in df.columns:
            raise MissingSource(f"{rel}: no column {col!r} (columns: {list(df.columns)})")
        sel = self.rows(rel, where)
        if len(sel) != 1:
            raise MissingSource(f"{rel}: expected exactly one row for {where}, found {len(sel)} (metric {mid})")
        key = f"column {col!r}" + (f" where {', '.join(f'{c} == {v!r}' for c, v in where.items())}" if where else "")
        return self._put(mid, sel[col].iloc[0], unit, rel, key, "read")

    def jcell(self, mid: str, rel: str, path: list, unit: str = ""):
        node = self.json(rel)
        for k in path:
            try:
                node = node[k]
            except (KeyError, IndexError, TypeError):
                raise MissingSource(f"{rel}: key path {path} not found (stopped at {k!r})") from None
        return self._put(mid, node, unit, rel, "key " + " > ".join(map(str, path)), "read")

    def derived(self, mid: str, value, unit: str, sources: list[str], rule: str):
        for s in sources:
            self._path(s)
        return self._put(mid, value, unit, sources if len(sources) > 1 else sources[0], rule, "derived")

    def col(self, rel: str, col: str, where: dict | None = None) -> np.ndarray:
        """A whole column (for derived rules); the caller records the rule in derived()."""
        sel = self.rows(rel, where)
        if col not in sel.columns:
            raise MissingSource(f"{rel}: no column {col!r}")
        if len(sel) == 0:
            raise MissingSource(f"{rel}: no rows for {where}")
        return sel[col].to_numpy()

    def v(self, mid: str):
        try:
            return self.items[mid]["value"]
        except KeyError:
            raise MissingSource(f"metric {mid!r} was not collected") from None

    # ---- output
    def to_json(self, cfg_path: str = "configs/nifty.yaml") -> dict:
        return {"schema": 1, "generated": datetime.now().isoformat(timespec="seconds"), "config": cfg_path, "n_metrics": len(self.items),
                "n_read": sum(1 for r in self.items.values() if r["kind"] == "read"),
                "n_derived": sum(1 for r in self.items.values() if r["kind"] == "derived"),
                "rule": "every value is read from its source file (column / key) or derived by the stated rule from such cells; a missing source stops the build",
                "metrics": self.items, "sources": self.sources}


def _mean_over(m: Metrics, rel: str, col: str, where: dict, prefix_col: str, prefix: str) -> float:
    df = m.rows(rel, where)
    df = df[df[prefix_col].astype(str).str.startswith(prefix)]
    if len(df) == 0:
        raise MissingSource(f"{rel}: no rows with {prefix_col} starting {prefix!r} for {where}")
    return float(df[col].mean()), int(len(df))


# ------------------------------------------------------------------ the collectors
T = "results/tables"


def collect_data(m: Metrics, cfg):
    f = "results/dq/data_quality_by_year.csv"
    df = m.table(f)
    allw = {"year": "ALL"}                       # the file's own whole-history row
    for c in ("files", "days_with_surface", "nifty_options", "traded", "expiry_ge3d", "has_forward", "otm", "price_ge1", "moneyness_ok", "iv_ok", "final"):
        m.cell(f"data.total.{c}", f, c, "count", allw)
    m.cell("data.quotes_per_surface_mean", f, "quotes_per_surface_mean", "quotes", allw)
    m.cell("data.quotes_per_surface_median", f, "quotes_per_surface_median", "quotes", allw)
    m.cell("data.final_share_of_traded", f, "final_share_of_traded", "share", allw)
    m.cell("data.final_share_of_nifty_options", f, "final_share_of_nifty_options", "share", allw)
    m.derived("data.traded_share_of_nifty_options", m.v("data.total.traded") / m.v("data.total.nifty_options"), "share", [f], "total 'traded' / total 'nifty_options' (rows year == ALL)")
    m.derived("data.usable_day_share", m.v("data.total.days_with_surface") / m.v("data.total.files"), "share", [f], "'days_with_surface' / 'files' (row year == ALL)")
    for y in df["year"].astype(str):
        if y == "ALL":
            continue
        m.cell(f"data.year.{y}.final_quotes", f, "final", "count", {"year": y})
        m.cell(f"data.year.{y}.quotes_per_surface_mean", f, "quotes_per_surface_mean", "quotes", {"year": y})
        m.cell(f"data.year.{y}.final_share_of_traded", f, "final_share_of_traded", "share", {"year": y})
    m.derived("data.n_years", int((df["year"].astype(str) != "ALL").sum()), "years", [f], "number of rows with year != ALL")
    s = "data/vix/SOURCE.txt"
    m._path(s)
    # no-arbitrage violations (diagnostics, not filters)
    nb = "results/dq/noarb_by_year.csv"
    for tol, tk in (("1 tick", "1tick"), ("1 tick + 0.25 vol pt", "1tick_025vol")):
        d = m.rows(nb, {"tolerance": tol})
        for col, short in (("share_quotes_a_monotonic", "a_monotonic"), ("share_quotes_b_butterfly", "b_butterfly"),
                           ("share_quotes_c_calendar", "c_calendar"), ("share_quotes_any", "any"), ("share_surfaces_any", "surfaces_any")):
            m.cell(f"noarb.{tk}.all_years.{short}", nb, col, "share", {"tolerance": tol, "year": "ALL"})
        for y in d["year"].astype(str):
            if y != "ALL":
                m.cell(f"noarb.{tk}.year.{y}.share_quotes_any", nb, "share_quotes_any", "share", {"tolerance": tol, "year": y})
    sf = "results/dq/noarb_sensitivity.csv"
    for tol, tk in (("1 tick", "1tick"), ("1 tick + 0.25 vol pt", "1tick_025vol")):
        for ex in ("none (all quotes, as published)", "a monotonic", "b butterfly", "c calendar", "any violation"):
            w = {"tolerance": tol, "period": "2022-2026", "excluded": ex}
            for col, unit in (("mean_share_of_quotes_excluded", "share"), ("smile_scaled_rmse", "vol points"), ("jump_carry_res_rmse", "vol points"),
                              ("diff_model_minus_benchmark", "vol points"), ("model_better_days_share", "share"), ("days_scored", "days")):
                m.cell(f"noarb.sens.{tk}.{slug(ex)}.{col}", sf, col, unit, w)


def collect_forecast(m: Metrics, cfg):
    f = f"{T}/forecast_scores.csv"
    runs, variants = cfg["forecasts"]["headline_runs"], cfg["forecasts"]["report_variants"]
    bench_fields = [("dev_mean", "vol points"), ("test_mean", "vol points"), ("test_mean_ci_lo", "vol points"), ("test_mean_ci_hi", "vol points"),
                    ("n_dev_days", "days"), ("n_test_days", "days")]
    model_fields = bench_fields + [("diff_vs_best_benchmark", "vol points"), ("diff_ci_lo", "vol points"), ("diff_ci_hi", "vol points")]
    for b in cfg["forecasts"]["benchmarks"]:
        for fld, unit in bench_fields:
            m.cell(f"forecast.{b}.{fld}", f, fld, unit, {"run": b, "variant": "none"})
    for r in runs:
        for v in variants:
            for fld, unit in model_fields:
                m.cell(f"forecast.{r}.{v}.{fld}", f, fld, unit, {"run": r, "variant": v})
    m.cell("forecast.best_benchmark", f, "best_benchmark", "name", {"run": runs[0], "variant": variants[0]})
    fy = f"{T}/forecast_by_year.csv"
    by = m.table(fy)
    series = list(cfg["forecasts"]["benchmarks"]) + [f"{r}:{v}" for r in runs for v in variants]
    for y in by["year"]:
        for s in series:
            m.cell(f"forecast_by_year.{slug(s)}.{y}", fy, s, "vol points", {"year": int(y)})
    mc, bs = "dhj_main:carry_res", "smile_scaled"
    for col, nm in ((mc, "jump_misfit"),):
        diff = by[mc].to_numpy(float) - by[bs].to_numpy(float)
        m.derived("forecast_by_year.years_total", int(len(diff)), "years", [fy], "number of rows (calendar years with forecasts)")
        m.derived("forecast_by_year.years_jump_misfit_beats_time_scaled_smile", int((diff < 0).sum()), "years", [fy],
                  f"count of years with column {mc!r} < column {bs!r}")
    # robustness over every saved run
    fr = f"{T}/robustness_runs.csv"
    rb = m.table(fr)
    for col in ("run", "variant", "test_mean"):
        if col not in rb.columns:
            raise MissingSource(f"{fr}: no column {col!r}")
    for v, tag in (("carry_res", "final_forecast"), ("carry", "pure_model_forecast")):
        sel = rb[(rb.variant == v) & (rb.run.astype(str).str.startswith("dhj"))]
        if len(sel) == 0:
            raise MissingSource(f"{fr}: no jump-model runs for variant {v}")
        m.derived(f"robustness.{tag}.n_runs", int(len(sel)), "runs", [fr], f"count of rows with run starting 'dhj' and variant == {v!r}")
        m.derived(f"robustness.{tag}.test_min", float(sel.test_mean.min()), "vol points", [fr], f"min of 'test_mean' over those rows")
        m.derived(f"robustness.{tag}.test_max", float(sel.test_mean.max()), "vol points", [fr], f"max of 'test_mean' over those rows")
    # pattern memory
    mem = "data/results/memory_final.json"
    for base, tag in (("hyb_dhj_main_carry_res", "jump_misfit"), ("smile_scaled", "smile_scaled")):
        for path, nm, unit in ((["test", "without", "mean"], "test_without", "vol points"), (["test", "with", "mean"], "test_with", "vol points"),
                               (["test", "paired_diff"], "paired_diff", "vol points"), (["test", "fired_share"], "fired_share", "share"),
                               (["test", "days"], "days", "days")):
            m.jcell(f"memory.{tag}.{nm}", mem, [base] + path, unit)
        m.jcell(f"memory.{tag}.ci95_lo", mem, [base, "test", "ci95", 0], "vol points")
        m.jcell(f"memory.{tag}.ci95_hi", mem, [base, "test", "ci95", 1], "vol points")


STATS_PAIRS = [("jump model + misfit", "diffusion DH + misfit"), ("jump model + misfit", "Heston + misfit"), ("jump model + misfit", "time-scaled smile"),
               ("jump model + misfit", "yesterday's smile"), ("diffusion DH + misfit", "Heston + misfit"), ("diffusion DH + misfit", "time-scaled smile"),
               ("diffusion DH + misfit", "yesterday's smile"), ("Heston + misfit", "time-scaled smile"), ("Heston + misfit", "yesterday's smile"),
               ("time-scaled smile", "yesterday's smile")]
STATS_LOSSES = {"all": "daily IV RMSE, all quotes", "core": "daily IV RMSE, near the money |ln K/F|<=0.04"}


def collect_stats(m: Metrics, cfg):
    f = "results/stats/tests_test_period.csv"
    for lk, loss in STATS_LOSSES.items():
        for a, b in STATS_PAIRS:
            w = {"loss": loss, "A": a, "B": b}
            base = f"stats.{lk}.{slug(a)}__{slug(b)}"
            for col, unit in (("mean_diff_A_minus_B", "vol points"), ("share_days_A_better", "share"), ("wilcoxon_p_holm", "p-value"), ("dm_stat_hln", "statistic"),
                              ("dm_p_holm", "p-value"), ("n_days", "days")):
                m.cell(f"{base}.{col}", f, col, unit, w)
            m.cell(f"{base}.verdict", f, "verdict", "text", w)


def collect_pricer(m: Metrics, cfg):
    e = "results/pricer_validation/errors.csv"
    allw = {"domain": "all", "maturity_bucket": "all", "moneyness_bucket": "all"}
    fields = [("max_abs_price_err_rs", "rupees"), ("max_rel_price_err", "relative"), ("max_abs_iv_err_pts", "vol points"), ("mean_abs_iv_err_pts", "vol points"), ("n", "points")]
    for src in ("real_chains", "synthetic_grid"):
        for model in ("dh", "dhj"):
            for comp in ("project_vs_cos", "lewis_vs_cos"):
                for col, unit in fields:
                    m.cell(f"pricer.{src}.{model}.{comp}.{col}", e, col, unit, {"source": src, "model": model, "comparison": comp, **allw})
    for col, unit in (("max_abs_z", "standard errors"), ("mean_abs_z", "standard errors"), ("n", "points")):
        m.cell(f"pricer.monte_carlo.dh.mc_vs_cos.{col}", e, col, unit, {"source": "monte_carlo", "model": "dh", "comparison": "mc_vs_cos", **allw})
    m.cell("pricer.monte_carlo.dh.project_vs_cos.max_abs_price_err_rs", e, "max_abs_price_err_rs", "rupees",
           {"source": "monte_carlo", "model": "dh", "comparison": "project_vs_cos", **allw})
    wc = "results/pricer_validation/worst_cases.csv"
    wd = m.table(wc)
    if "iv_err_pts" not in wd.columns:
        raise MissingSource(f"{wc}: no column 'iv_err_pts'")
    m.derived("pricer.worst_cases.n_rows", int(len(wd)), "rows", [wc], "number of rows")
    m.derived("pricer.worst_cases.max_abs_iv_err_pts", float(np.abs(wd["iv_err_pts"]).max()), "vol points", [wc], "max of |column 'iv_err_pts'|")
    m.derived("pricer.worst_cases.n_in_calibration_domain", int(wd["in_calibration_domain"].astype(bool).sum()), "rows", [wc],
              "count of rows with column 'in_calibration_domain' true")
    t = "results/pricer_validation/timing.json"
    for model in ("dh", "dhj"):
        for k, tag in (("project_pricer_dh2.models.price", "project"), ("cos_default_width_checked", "cos_checked"), ("cos_fixed_L14_no_width_check", "cos_fixed_L14"),
                       ("lewis_adaptive_quad", "lewis")):
            m.jcell(f"pricer.timing.{model}.{tag}_us_per_option", t, [model, k, "microseconds_per_option_best"], "microseconds")
        m.jcell(f"pricer.timing.{model}.cos_time_ratio_project_over_cos", t, [model, "cos_time_ratio_project_over_cos"], "ratio")
        m.jcell(f"pricer.timing.{model}.lewis_time_ratio_over_cos", t, [model, "lewis_time_ratio_over_cos"], "ratio")
    m.jcell("pricer.timing.n_quotes", t, ["n_quotes"], "quotes")
    m.jcell("pricer.timing.n_expiries", t, ["n_expiries"], "expiries")
    m.jcell("pricer.timing.day", t, ["day"], "date")


PINN_VARIANTS = {"hybrid": "hybrid (data + PDE + asymptotics), 3 seeds", "data_only": "data-only (same architecture), 3 seeds",
                 "physics_only": "physics-only (no labels), 1 seed", "spec_gate": "data-only with the spec gate g = sqrt(tau), 1 seed"}


SPEED_ROWS = {"exact_one_chain": ("exact pricer, one chain", "cpu (1 thread, numpy)"), "sur_one_chain_cpu": ("surrogate, one chain, prices", "cpu"),
              "sur_one_chain_mps": ("surrogate, one chain, prices", "mps"), "exact_128": ("exact pricer, 128 parameter sets x chain", "cpu (1 thread, numpy)"),
              "sur_128_cpu": ("surrogate, 128 parameter sets x chain", "cpu"), "sur_128_mps": ("surrogate, 128 parameter sets x chain", "mps"),
              "exact_greeks_fd": ("exact Greeks by finite differences (9 chain evaluations), one chain", "cpu (1 thread, numpy)"),
              "sur_greeks_ad": ("surrogate Greeks (delta, gamma, dC/dv1, dC/dv2), one chain", "cpu")}


def collect_pinn(m: Metrics, cfg):
    a = "results/pinn/tables/ablation_summary.csv"
    fl = [("IV err mean (vol pts) avg over seeds", "mean", "vol points"), ("IV err median (vol pts) avg over seeds", "median", "vol points"),
          ("IV err p95 (vol pts) avg over seeds", "p95", "vol points"), ("IV err max (vol pts) avg over seeds", "max", "vol points")]
    for vk, vn in PINN_VARIANTS.items():
        for st in ("real", "box"):
            for col, short, unit in fl:
                m.cell(f"pinn.ablation.{vk}.{st}.iv_err_{short}", a, col, unit, {"variant": vn, "held-out set": st})
            m.cell(f"pinn.ablation.{vk}.{st}.pde_res_median", a, "pde_res_median", "vol points", {"variant": vn, "held-out set": st})
            if vk in ("hybrid", "data_only"):          # a spread over seeds exists only for the 3-seed variants
                m.cell(f"pinn.ablation.{vk}.{st}.iv_err_mean_sd_over_seeds", a, "IV err mean sd over seeds", "vol points", {"variant": vn, "held-out set": st})
    for vk in ("hybrid", "data_only"):
        m.cell(f"pinn.ablation.{vk}.real.price_mae_rs", a, "price MAE (rupees)", "rupees", {"variant": PINN_VARIANTS[vk], "held-out set": "real"})
        m.cell(f"pinn.ablation.{vk}.test.iv_err_mean", a, "IV err mean (vol pts) avg over seeds", "vol points", {"variant": PINN_VARIANTS[vk], "held-out set": "test"})
    d = "results/pinn/tables/data_efficiency.csv"
    for vk, vn in (("data_only", "data-only"), ("hybrid", "hybrid")):
        for n in (1000, 10000, 100000, 1000000):
            for st in ("real", "box"):
                m.cell(f"pinn.data_efficiency.{vk}.{st}.n{n}.iv_err_mean", d, "IV err mean", "vol points", {"held-out set": st, "variant": vn, "supervised samples": n})
    s = "results/pinn/tables/speed.csv"
    rows = SPEED_ROWS
    for k, (what, dev) in rows.items():
        m.cell(f"pinn.speed.{k}.median_ms", s, "median_ms", "milliseconds", {"what": what, "device": dev})
    for num, den, nm in (("exact_one_chain", "sur_one_chain_cpu", "one_chain_cpu"), ("exact_128", "sur_128_cpu", "128_sets_cpu"), ("exact_128", "sur_128_mps", "128_sets_mps"),
                         ("exact_greeks_fd", "sur_greeks_ad", "greeks")):
        m.derived(f"pinn.speed.speedup_{nm}", m.v(f"pinn.speed.{num}.median_ms") / m.v(f"pinn.speed.{den}.median_ms"), "ratio", [s],
                  f"median_ms of {rows[num]} divided by median_ms of {rows[den]}")
    g = "results/pinn/tables/greeks_errors.csv"
    for greek, gk in (("delta (abs)", "delta_abs"), ("gamma (frac of Black gamma)", "gamma_frac"), ("dC/dv1 (frac of Black var-sens.)", "dcdv1_frac"),
                      ("dC/dv2 (frac of Black var-sens.)", "dcdv2_frac")):
        for pre, vk in (("dat_s", "data_only"), ("hyb_s", "hybrid")):
            val, n = _mean_over(m, g, "median", {"greek": greek}, "model", pre)
            m.derived(f"pinn.greeks.{vk}.{gk}.median_error", val, "see greek", [g], f"mean over the {n} seeds (rows with model starting {pre!r}, greek == {greek!r}) of column 'median'")
    ar = "results/pinn/tables/arbitrage_checks.csv"
    tol = 1e-6
    tcol = "tolerance (price per unit forward)"
    for col, ck in (("convex in strike: share violated", "convexity"), ("monotone in strike: share violated", "monotone")):
        m.cell(f"pinn.arbitrage.exact.{ck}.share_violated", ar, col, "share", {"price source": "exact", tcol: tol})
        for pre, vk in (("dat_s", "data_only"), ("hyb_s", "hybrid")):
            val, n = _mean_over(m, ar, col, {tcol: tol}, "price source", pre)
            m.derived(f"pinn.arbitrage.{vk}.{ck}.share_violated", val, "share", [ar], f"mean over the {n} seeds (price source starting {pre!r}, tolerance 1e-6) of {col!r}")
    m.cell("pinn.domain.test_days_flagged_share", "results/pinn/tables/test_period_domain_flags.csv", "share of test days", "share",
           {"item": "ANY of the above (day flagged)"})


PINN_CALIB_METHODS = {"exact": "exact (Phase-3 recipe, 4 starts)", "accel": "accelerated (surrogate global + exact polish)", "surrogate_only": "surrogate-only (no polish)",
                      "control": "exact polish from Phase-3 starts (control)"}


def collect_pinn_calib(m: Metrics, cfg):
    h = "results/pinn_calib/headline.json"
    for k in ("exact", "accel", "surrogate_only", "exact_polish_from_phase3_starts", "accel_without_rerank"):
        m.jcell(f"pinn_calib.rmse_mean.{k}", h, ["rmse_mean", k], "vol points")
        m.jcell(f"pinn_calib.rmse_median.{k}", h, ["rmse_median", k], "vol points")
    for k, unit in (("speedup_wall_total", "ratio"), ("speedup_cpu_total", "ratio"), ("speedup_wall_median_per_day", "ratio"), ("call_ratio", "ratio")):
        for sub in ("est", "lo", "hi"):
            m.jcell(f"pinn_calib.{k}.{sub}", h, [k, sub], unit)
    for k, unit in (("surrogate_stage", "seconds"), ("correction", "seconds"), ("polish", "seconds")):
        m.jcell(f"pinn_calib.accel_seconds.{k}", h, ["accel_sec_breakdown_mean", k], unit)
    m.jcell("pinn_calib.n_days", h, ["n_days"], "days")
    m.jcell("pinn_calib.box_exact_inside_share", h, ["box_exact_inside_share"], "share")
    m.jcell("pinn_calib.share_days_with_quotes_beyond_tau_0_6", h, ["share_days_with_quotes_beyond_tau_0.6"], "share")
    m.jcell("pinn_calib.worse_days.n_worse", h, ["worse_days", "n_worse"], "days")
    m.jcell("pinn_calib.worse_days.max_cost_ratio", h, ["worse_days", "max_cost_ratio"], "ratio")
    a = "results/pinn_calib/accuracy_by_method.csv"
    for k, nm in PINN_CALIB_METHODS.items():
        for col, unit in (("mean_rmse", "vol points"), ("ci95_lo", "vol points"), ("ci95_hi", "vol points"), ("mean_log_cost_ratio_vs_exact", "log ratio"),
                          ("share_days_cost_within_1pct_of_exact_or_better", "share")):
            m.cell(f"pinn_calib.method.{k}.{col}", a, col, unit, {"method": nm})
    p = "results/pinn_calib/accuracy_paired.csv"
    w = {"comparison": "accelerated vs exact", "metric": "same-day IV RMSE (vol points), exact pricer"}
    for col, unit in (("mean_diff", "vol points"), ("ci95_lo", "vol points"), ("ci95_hi", "vol points"), ("wilcoxon_p", "p-value"), ("days_accel_lower", "days"), ("days_exact_lower", "days")):
        m.cell(f"pinn_calib.paired_accel_vs_exact.{col}", p, col, unit, w)
    tm = "results/pinn_calib/timing_by_method.csv"
    for k, nm in (("exact", "exact (Phase-3 recipe, 4 starts)"), ("accel", "accelerated")):
        m.cell(f"pinn_calib.time.{k}.mean_sec", tm, "mean_sec", "seconds", {"method": nm})
        m.cell(f"pinn_calib.time.{k}.mean_exact_pricer_calls", tm, "mean_exact_pricer_calls", "calls", {"method": nm})


ATTR_STEPS = [("step0", "DH1 (1.0 recipe)"), ("step1", "+ clean data"), ("step2", "+ IV-space vega-weighted objective"),
              ("step3", "+ global multi-start, kappa2 > kappa1"), ("step4", "+ structure/state split, walk-forward"), ("step5", "+ Merton jumps"),
              ("step6", "+ yesterday's misfit carried")]
ATTR_REFS = [("yesterdays_smile", "yesterday's smile (model-free)"), ("time_scaled_smile", "time-scaled smile (model-free)"),
             ("flat_bs", "flat Black-Scholes, daily refit, carried"), ("single_heston", "single Heston, independent daily fits"),
             ("dhj_independent", "DH + jumps, independent daily fits")]


def collect_attribution(m: Metrics, cfg):
    f = "results/attribution/waterfall.csv"
    for k, lab in ATTR_STEPS:
        for col, unit in (("in_mean", "vol points"), ("in_lo", "vol points"), ("in_hi", "vol points"), ("oos_mean", "vol points"), ("oos_lo", "vol points"),
                          ("oos_hi", "vol points"), ("n_days_in", "days")):
            m.cell(f"attribution.{k}.{col}", f, col, unit, {"label": lab})
        if k != "step0":
            for col, unit in (("in_delta_vs_prev", "vol points"), ("oos_delta_vs_prev", "vol points"), ("oos_wilcoxon_p", "p-value")):
                m.cell(f"attribution.{k}.{col}", f, col, unit, {"label": lab})
    for k, lab in ATTR_REFS:
        m.cell(f"attribution.ref.{k}.oos_mean", f, "oos_mean", "vol points", {"label": lab})
    for k, lab in ATTR_REFS[2:]:
        m.cell(f"attribution.ref.{k}.in_mean", f, "in_mean", "vol points", {"label": lab})
    m.derived("attribution.total_oos_change", m.v("attribution.step6.oos_mean") - m.v("attribution.step0.oos_mean"), "vol points", [f],
              "oos_mean of the last step minus oos_mean of step 0")
    m.derived("attribution.total_in_change", m.v("attribution.step5.in_mean") - m.v("attribution.step0.in_mean"), "vol points", [f],
              "in_mean of step 5 minus in_mean of step 0 (step 6 carries the same fit)")


def collect_cross_ident_buckets(m: Metrics, cfg):
    f = "results/crosssection/summary.csv"
    for setting in ("random", "maturity"):
        for model in ("bs", "heston", "dh", "dhj"):
            for col, unit in (("mean_iv_rmse", "vol points"), ("ci95_lo", "vol points"), ("ci95_hi", "vol points"), ("median", "vol points"), ("n_days", "days")):
                m.cell(f"crosssection.{setting}.{model}.{col}", f, col, unit, {"setting": setting, "model": model})
    pf = "results/crosssection/pairs.csv"
    for setting, pairs in (("random", [("bs", "dhj"), ("heston", "dh"), ("dh", "dhj")]), ("maturity", [("heston", "dh"), ("heston", "dhj"), ("dh", "dhj")])):
        for a, b in pairs:
            w = {"setting": setting, "model_a": a, "model_b": b}
            for col, unit in (("mean_diff_b_minus_a", "vol points"), ("diff_ci95_lo", "vol points"), ("diff_ci95_hi", "vol points"), ("days_b_better", "days"),
                              ("wilcoxon_p", "p-value"), ("n_days", "days")):
                m.cell(f"crosssection.pair.{setting}.{a}_to_{b}.{col}", pf, col, unit, w)
    i = "results/identifiability/summary.csv"
    for model in ("dh", "dhj"):
        for col, unit in (("days", "days"), ("n_params", "parameters"), ("median_cond_colnorm_ls", "condition number"), ("median_cond_range_ls", "condition number"),
                          ("median_effective_rank_ls", "rank"), ("share_params_within_2pct_of_bound", "share"), ("share_days_with_a_param_at_bound", "share"),
                          ("median_param_se_over_range_ls", "share of bound width"), ("mean_distinct_sets_tol5", "sets"),
                          ("share_days_2plus_distinct_sets_tol5", "share"), ("mean_starts_within_1pct_of_best", "starts"), ("starts", "starts"),
                          ("median_worst_over_best_cost", "ratio")):
            m.cell(f"identifiability.{model}.{col}", i, col, unit, {"model": model})
    b = "results/buckets/bucket_summary.csv"
    for model in ("bs", "heston", "dh", "dhj", "walk_dh", "walk_dhj"):
        m.cell(f"buckets.pooled.{model}.iv_rmse", b, "pooled_iv_rmse_pts", "vol points", {"model": model})
    m.cell("buckets.pooled.n_quotes", b, "n_quotes", "quotes", {"model": "dh"})
    br = "results/buckets/bucket_rmse.csv"
    for model in ("bs", "heston", "dh", "dhj", "walk_dh", "walk_dhj"):
        m.cell(f"buckets.deep_puts_3_7d.{model}.rmse", br, "rmse_pts", "vol points", {"model": model, "maturity_bucket": "3-7 d", "moneyness_bucket": "-22 to -12%"})


def collect_vix_svi(m: Metrics, cfg):
    f = "results/vix/vix_vs_model_stats.csv"
    series = {"jump": "Jump model + jump variance", "diff": "Diffusion Double Heston", "atm": "Surface ATM 30-day IV"}
    periods = {"all": "2016-2026 (all)", "dev": "2016-2021 (development)", "test": "2022-2026 (test)"}
    for sk, sn in series.items():
        for pk, pn in periods.items():
            for col, unit in (("n_levels", "days"), ("corr_levels", "correlation"), ("corr_changes", "correlation"), ("mean_gap_model_minus_vix", "vol points"),
                              ("rmse_gap", "vol points"), ("reg_level_slope", "slope"), ("reg_level_r2", "R2"), ("reg_change_slope", "slope")):
                m.cell(f"vix.{sk}.{pk}.{col}", f, col, unit, {"series": sn, "period": pn})
    for sk in ("jump", "diff"):
        m.cell(f"vix.{sk}.without_saturated.corr_changes", f, "corr_changes", "correlation",
               {"series": series[sk], "period": "2016-2026 without the 16 bound-saturated days"})
    m.cell("vix.reg_level_slope_se_jump_all", f, "reg_level_slope_se_hac", "slope s.e.", {"series": series["jump"], "period": periods["all"]})
    s = "results/svi/svi_vs_dh_by_bucket.csv"
    for key, mat in (("all", "ALL"), ("core", "core (abs k<=0.04)"), ("wings", "wings (abs k>0.10)")):
        for col, unit in (("svi_rmse", "vol points"), ("dhj_rmse", "vol points"), ("dh_rmse", "vol points"), ("quotes", "quotes")):
            m.cell(f"svi.pooled.{key}.{col}", s, col, unit, {"maturity": mat})
    d = "results/svi/svi_vs_dh_by_day.csv"
    dd = m.table(d)
    for col, tag in (("svi_rmse", "svi"), ("dh_rmse_same_quotes", "dh"), ("dhj_rmse_same_quotes", "dhj")):
        if col not in dd.columns:
            raise MissingSource(f"{d}: no column {col!r}")
        m.derived(f"svi.daily_mean.{tag}", float(dd[col].mean()), "vol points", [d], f"mean over days of column {col!r}")
    m.derived("svi.n_days", int(len(dd)), "days", [d], "number of rows")
    sf = "results/svi/svi_fits.csv"
    sd = m.table(sf)
    for col, tag in (("butterfly_violation_anywhere", "butterfly_violation_anywhere"), ("butterfly_violation_in_quote_range", "butterfly_violation_in_quote_range"),
                     ("lee_wing_bound_ok", "lee_wing_bound_ok")):
        if col not in sd.columns:
            raise MissingSource(f"{sf}: no column {col!r}")
        m.derived(f"svi.fits.share_{tag}", float(sd[col].astype(bool).mean()), "share", [sf], f"mean of boolean column {col!r} over the fitted expiries")
    m.derived("svi.fits.n_expiries", int(len(sd)), "expiries", [sf], "number of rows")


def collect_trading(m: Metrics, cfg):
    tags = ["forecast_edge / plain", "forecast_edge / hedged", "rich_cheap_edge / plain", "rich_cheap_edge / hedged"]
    for nm, lab in (("dhj_main", "signal_close"), ("dhj_main_lag1", "one_day_later")):
        f = f"data/results/trade_robust_{nm}.json"
        for tag in tags:
            base = f"trading.{lab}.{slug(tag)}"
            m.jcell(f"{base}.threshold", f, [tag, "threshold"], "vol points")
            m.jcell(f"{base}.trades", f, [tag, "trades"], "trades")
            m.jcell(f"{base}.buy_share", f, [tag, "buy_share"], "share")
            for ck, cn in (("cost_0.0", "cost0"), ("cost_0.01", "cost1pct")):
                for key, unit in (("wins_per_100", "wins per 100"), ("blind_same_mix_wins_per_100", "wins per 100"), ("mean_rs", "rupees per trade"),
                                  ("median_rs", "rupees per trade"), ("trimmed_rs", "rupees per trade")):
                    m.jcell(f"{base}.{cn}.{key}", f, [tag, ck, key], unit)
                m.jcell(f"{base}.{cn}.mean_rs_ci_lo", f, [tag, ck, "mean_rs_ci", 0], "rupees per trade")
                m.jcell(f"{base}.{cn}.mean_rs_ci_hi", f, [tag, ck, "mean_rs_ci", 1], "rupees per trade")
    f = "data/results/trade_matched_dhj_main.json"
    for tag in tags:
        base = f"trading.matched.{slug(tag)}"
        m.jcell(f"{base}.trades", f, [tag, "trades"], "trades")
        for ck, cn in (("cost_0.0", "cost0"), ("cost_0.01", "cost1pct")):
            for key, unit in (("same_day_wins", "wins per 100"), ("next_day_wins", "wins per 100"), ("same_day_rs", "rupees per trade"), ("next_day_rs", "rupees per trade"),
                              ("lost_rs", "rupees per trade")):
                m.jcell(f"{base}.{cn}.{key}", f, [tag, ck, key], unit)
            m.jcell(f"{base}.{cn}.lost_ci_lo", f, [tag, ck, "lost_ci", 0], "rupees per trade")
            m.jcell(f"{base}.{cn}.lost_ci_hi", f, [tag, ck, "lost_ci", 1], "rupees per trade")
    u = "today.json"
    m.jcell("ui.asof", u, ["asof"], "date")
    m.jcell("ui.diffusion_same_day_iv_rmse", u, ["same_day_iv_rmse_vol_points"], "vol points")
    m.jcell("ui.jump_same_day_iv_rmse", u, ["jump_model_for_information", "same_day_iv_rmse_vol_points"], "vol points")


def collect_pinn_calib_and_rest(m, cfg):  # kept for readability of build_metrics
    collect_pinn_calib(m, cfg)
    collect_attribution(m, cfg)
    collect_cross_ident_buckets(m, cfg)


def build_metrics(cfg: dict, root: Path = ROOT) -> Metrics:
    """Collect every number. Raises MissingSource at the first source that is not there."""
    m = Metrics(root)
    collect_data(m, cfg)
    collect_forecast(m, cfg)
    collect_stats(m, cfg)
    collect_pricer(m, cfg)
    collect_pinn(m, cfg)
    collect_pinn_calib_and_rest(m, cfg)
    collect_vix_svi(m, cfg)
    collect_trading(m, cfg)
    return m


def write_metrics(m: Metrics, path: Path, cfg_path: str = "configs/nifty.yaml") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m.to_json(cfg_path), indent=1, ensure_ascii=False))
    return path


# =================================================================================================================== 3. tables
@dataclass
class Table:
    name: str
    caption: str
    df: pd.DataFrame            # numeric columns stay numeric (CSV is written at full precision)
    formats: dict = field(default_factory=dict)   # column -> format string or callable, for MD / TeX
    notes: list[str] = field(default_factory=list)
    ci: list = field(default_factory=list)       # (new column name, lo column, hi column, decimals, signed): MD / TeX show "[lo, hi]" in one column; CSV keeps both

    def formatted(self) -> pd.DataFrame:
        out = pd.DataFrame(index=self.df.index)
        for c in self.df.columns:
            f = self.formats.get(c)
            col = []
            for v in self.df[c]:
                if isinstance(v, float) and math.isnan(v):
                    col.append("")
                elif isinstance(v, str):
                    col.append(v)
                elif f is not None:
                    col.append(f(v) if callable(f) else f.format(v))
                elif isinstance(v, (int, np.integer)):
                    col.append(f"{int(v):,}")
                else:
                    col.append(str(v))
            out[c] = col
        for name, lo, hi, d, signed in self.ci:
            pos = list(out.columns).index(lo)
            vals = [("" if (isinstance(a, float) and math.isnan(a)) else _ci(a, b, d, signed)) for a, b in zip(self.df[lo], self.df[hi])]
            out = out.drop(columns=[lo, hi])
            out.insert(pos, name, vals)
        return out


def pfmt(p: float) -> str:
    return "<1e-10" if p < 1e-10 else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|")


def to_markdown(tb: Table) -> str:
    f = tb.formatted()
    head = "| " + " | ".join(_md_escape(c) for c in f.columns) + " |"
    sep = "|" + "|".join("---" for _ in f.columns) + "|"
    body = ["| " + " | ".join(_md_escape(v) for v in row) + " |" for row in f.itertuples(index=False)]
    notes = [f"\n*{n}*" for n in tb.notes]
    return f"**{tb.caption}**\n\n" + "\n".join([head, sep] + body) + "\n" + "\n".join(notes) + ("\n" if notes else "")


_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex_escape(s) -> str:
    s = "".join(_TEX.get(ch, ch) for ch in str(s))
    return s.replace("<=", r"$\leq$").replace(">=", r"$\geq$").replace("<", r"$<$").replace(">", r"$>$")


def to_latex(tb: Table) -> str:
    f = tb.formatted()
    num = [(pd.api.types.is_numeric_dtype(tb.df[c]) if c in tb.df.columns else True) for c in f.columns]
    spec = "".join("r" if n else "l" for n in num)
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", rf"\caption{{{tex_escape(tb.caption)}}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in f.columns) + r" \\", r"\midrule"]
    for row in f.itertuples(index=False):
        lines.append(" & ".join(tex_escape(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if tb.notes:
        lines.append(r"\par\smallskip{\footnotesize " + tex_escape(" ".join(tb.notes)) + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_table(tb: Table, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tb.df.to_csv(out_dir / f"{tb.name}.csv", index=False)
    (out_dir / f"{tb.name}.md").write_text(to_markdown(tb))
    (out_dir / f"{tb.name}.tex").write_text(to_latex(tb))


def _ci(lo, hi, d=3, signed=False):
    fmt = f"{{:+.{d}f}}" if signed else f"{{:.{d}f}}"
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


TABLE_NAMES = ["forecast_models", "significance_tests", "pricer_validation", "pinn_accuracy", "pinn_speed", "pinn_calibration", "waterfall", "crosssection",
               "identifiability", "vix_link", "svi_reference", "data_quality_funnel", "noarb_sensitivity", "trading_summary", "robustness_memory"]


def build_tables(m: Metrics, cfg: dict) -> list[Table]:
    """Every headline table, built from collected metrics only (names: TABLE_NAMES)."""
    v = m.v
    tabs: list[Table] = []
    runs, variants = cfg["forecasts"]["headline_runs"], cfg["forecasts"]["report_variants"]

    # 1 forecast comparison
    rows = []
    for b in cfg["forecasts"]["benchmarks"]:
        rows.append({"Forecast": LABELS[b], "Kind": "model-free benchmark", "Dev": v(f"forecast.{b}.dev_mean"), "Test": v(f"forecast.{b}.test_mean"),
                     "Test lo": v(f"forecast.{b}.test_mean_ci_lo"), "Test hi": v(f"forecast.{b}.test_mean_ci_hi"),
                     "Diff vs best benchmark": np.nan, "Diff lo": np.nan, "Diff hi": np.nan})
    for r in ("heston_main", "dh_main", "dhj_main"):
        for var in variants:
            k = f"forecast.{r}.{var}"
            rows.append({"Forecast": LABELS[f"{r}:{var}"], "Kind": "model", "Dev": v(f"{k}.dev_mean"), "Test": v(f"{k}.test_mean"),
                         "Test lo": v(f"{k}.test_mean_ci_lo"), "Test hi": v(f"{k}.test_mean_ci_hi"),
                         "Diff vs best benchmark": v(f"{k}.diff_vs_best_benchmark"), "Diff lo": v(f"{k}.diff_ci_lo"), "Diff hi": v(f"{k}.diff_ci_hi")})
    nb = v("forecast.best_benchmark")
    sp = cfg["split"]
    tabs.append(Table("forecast_models", "Next-day implied-volatility forecast error (vol points, lower is better), mean over days",
                      pd.DataFrame(rows), {"Dev": "{:.3f}", "Test": "{:.3f}", "Diff vs best benchmark": "{:+.3f}"},
                      [f"Dev = {sp['development']['start'].year}-{sp['development']['end'].year}, Test = {sp['test']['start'].year} onward; test days: {v('forecast.smile.n_test_days')}, development days: "
                       f"{v('forecast.smile.n_dev_days')}. Difference is against the best benchmark on the test period ({LABELS[nb]}); negative = better. "
                       "Intervals: 95% moving-block bootstrap (dh2.report, final_table.py logic)."],
                      ci=[("95% interval of the test mean", "Test lo", "Test hi", 3, False), ("95% interval of the difference", "Diff lo", "Diff hi", 3, True)]))

    # 2 stats
    rows = []
    for a, b in STATS_PAIRS:
        base = f"stats.all.{slug(a)}__{slug(b)}"
        rows.append({"A": a, "B": b, "A minus B (vol points)": v(f"{base}.mean_diff_A_minus_B"), "Days A better": v(f"{base}.share_days_A_better"),
                     "Wilcoxon p (Holm)": v(f"{base}.wilcoxon_p_holm"), "Diebold-Mariano stat": v(f"{base}.dm_stat_hln"), "DM p (Holm)": v(f"{base}.dm_p_holm"),
                     "Verdict": v(f"{base}.verdict")})
    tabs.append(Table("significance_tests", "Paired tests on the daily next-day error, test period (all quotes)", pd.DataFrame(rows),
                      {"A minus B (vol points)": "{:+.3f}", "Days A better": "{:.0%}", "Wilcoxon p (Holm)": pfmt, "Diebold-Mariano stat": "{:+.2f}", "DM p (Holm)": pfmt},
                      [f"{v(f'stats.all.{slug(STATS_PAIRS[0][0])}__{slug(STATS_PAIRS[0][1])}.n_days')} test days; Holm adjustment over the 10 pairs; DM with Newey-West variance and the Harvey-Leybourne-Newbold correction."]))

    # 3 pricer
    rows = []
    for src, lab in (("real_chains", "real chains (calibrated parameters of sampled days)"), ("synthetic_grid", "synthetic grid (calibrated-parameter sets, maturities, strikes)")):
        for model in ("dh", "dhj"):
            for comp, cl in (("project_vs_cos", "project pricer vs COS"), ("lewis_vs_cos", "Lewis vs COS")):
                k = f"pricer.{src}.{model}.{comp}"
                rows.append({"Points": lab, "Model": "jumps" if model == "dhj" else "diffusion", "Comparison": cl, "n": v(f"{k}.n"),
                             "Max price error (Rs)": v(f"{k}.max_abs_price_err_rs"), "Max relative price error": v(f"{k}.max_rel_price_err"),
                             "Max IV error (vol pts)": v(f"{k}.max_abs_iv_err_pts"), "Mean IV error (vol pts)": v(f"{k}.mean_abs_iv_err_pts")})
    rows.append({"Points": "Monte Carlo (antithetic QE paths)", "Model": "diffusion", "Comparison": "Monte Carlo vs COS (max |z|, mean |z|)", "n": v("pricer.monte_carlo.dh.mc_vs_cos.n"),
                 "Max price error (Rs)": np.nan, "Max relative price error": np.nan, "Max IV error (vol pts)": v("pricer.monte_carlo.dh.mc_vs_cos.max_abs_z"),
                 "Mean IV error (vol pts)": v("pricer.monte_carlo.dh.mc_vs_cos.mean_abs_z")})
    tabs.append(Table("pricer_validation", "The project pricer against three independent pricers", pd.DataFrame(rows),
                      {"Max price error (Rs)": "{:.2e}", "Max relative price error": "{:.2e}", "Max IV error (vol pts)": "{:.2e}", "Mean IV error (vol pts)": "{:.2e}"},
                      ["Implied-vol errors only for options worth at least one tick (Rs 0.05). Monte Carlo row: the last two columns are standard errors (z), not vol points."]))

    # 4 pinn accuracy
    rows = []
    for vk, lab in (("data_only", "data-only (shipped for calibration)"), ("hybrid", "hybrid data + PDE (shipped for Greeks)"), ("spec_gate", "data-only, spec gate sqrt(tau)"),
                    ("physics_only", "physics-only (no labels)")):
        k = f"pinn.ablation.{vk}"
        rows.append({"Network": lab, "Real held-out mean": v(f"{k}.real.iv_err_mean"), "median": v(f"{k}.real.iv_err_median"), "95th pct": v(f"{k}.real.iv_err_p95"),
                     "max": v(f"{k}.real.iv_err_max"), "Box-set mean": v(f"{k}.box.iv_err_mean"), "PDE residual median (real)": v(f"{k}.real.pde_res_median")})
    tabs.append(Table("pinn_accuracy", "PINN surrogate implied-vol error against the exact pricer (vol points)", pd.DataFrame(rows),
                      {c: "{:.3f}" for c in ("Real held-out mean", "median", "95th pct", "max", "Box-set mean", "PDE residual median (real)")},
                      ["Real held-out = every quote of the held-out development days (the last fifth of 2016-2021, never in the training sampler); box set = unseen parameter sets drawn from the "
                       "training box. Seed averages for the 3-seed rows."]))

    # 5 pinn speed
    rows = []
    for k, (what, dev) in SPEED_ROWS.items():
        rows.append({"Task": f"{what} [{dev}]", "Median time (ms)": v(f"pinn.speed.{k}.median_ms")})
    tabs.append(Table("pinn_speed", "Speed: exact Fourier pricer against the surrogate", pd.DataFrame(rows), {"Median time (ms)": "{:.1f}"},
                      [f"Speed-up of the surrogate: one chain on CPU {v('pinn.speed.speedup_one_chain_cpu'):.1f}x, 128 parameter sets {v('pinn.speed.speedup_128_sets_cpu'):.0f}x (CPU) and "
                       f"{v('pinn.speed.speedup_128_sets_mps'):.0f}x (GPU), Greeks {v('pinn.speed.speedup_greeks'):.1f}x."]))

    # 6 pinn calibration
    rows = []
    for k, lab in PINN_CALIB_METHODS.items():
        kk = f"pinn_calib.method.{k}"
        rows.append({"Method": lab, "Same-day IV RMSE": v(f"{kk}.mean_rmse"), "RMSE lo": v(f"{kk}.ci95_lo"), "RMSE hi": v(f"{kk}.ci95_hi"),
                     "Mean log(cost / exact cost)": v(f"{kk}.mean_log_cost_ratio_vs_exact"), "Days within 1% of exact cost": v(f"{kk}.share_days_cost_within_1pct_of_exact_or_better")})
    tabs.append(Table("pinn_calibration", "PINN-accelerated calibration of the diffusion model (paired days, exact pricer scores every fit)", pd.DataFrame(rows),
                      {"Same-day IV RMSE": "{:.3f}", "Mean log(cost / exact cost)": "{:+.3f}", "Days within 1% of exact cost": "{:.0%}"},
                      [f"{v('pinn_calib.n_days')} days. Wall-time speed-up {v('pinn_calib.speedup_wall_total.est'):.1f}x {_ci(v('pinn_calib.speedup_wall_total.lo'), v('pinn_calib.speedup_wall_total.hi'), 1)}; "
                       f"exact-pricer calls {v('pinn_calib.call_ratio.est'):.1f}x fewer; paired RMSE difference accelerated minus exact {v('pinn_calib.paired_accel_vs_exact.mean_diff'):+.4f} "
                       f"{_ci(v('pinn_calib.paired_accel_vs_exact.ci95_lo'), v('pinn_calib.paired_accel_vs_exact.ci95_hi'), 4, True)}."],
                      ci=[("95% interval", "RMSE lo", "RMSE hi", 3, False)]))

    # 7 waterfall
    rows = []
    for k, lab in ATTR_STEPS:
        r = {"Step": lab, "Same-day mean": v(f"attribution.{k}.in_mean"), "Next-day mean": v(f"attribution.{k}.oos_mean"),
             "Next lo": v(f"attribution.{k}.oos_lo"), "Next hi": v(f"attribution.{k}.oos_hi"),
             "Next-day change vs previous step": v(f"attribution.{k}.oos_delta_vs_prev") if k != "step0" else np.nan}
        rows.append(r)
    for k, lab in (("yesterdays_smile", "reference: yesterday's smile"), ("time_scaled_smile", "reference: time-scaled smile")):
        rows.append({"Step": lab, "Same-day mean": np.nan, "Next-day mean": v(f"attribution.ref.{k}.oos_mean"), "Next lo": np.nan, "Next hi": np.nan,
                     "Next-day change vs previous step": np.nan})
    tabs.append(Table("waterfall", "From the 1.0 recipe to 2.0, one fix at a time (vol points; paired sample of test days)", pd.DataFrame(rows),
                      {"Same-day mean": "{:.3f}", "Next-day mean": "{:.3f}", "Next-day change vs previous step": "{:+.3f}"},
                      [f"{v('attribution.step0.n_days_in'):.0f} sampled test days (every {cfg['evaluation']['paired_sample']['every_nth']}th trading day). Step 6 changes only the next-day forecast."],
                      ci=[("Next-day 95% interval", "Next lo", "Next hi", 3, False)]))

    # 8 cross-section
    rows = []
    for setting, lab in (("random", "random hold-out of quotes (interpolation)"), ("maturity", "fit short expiries, score longer ones (extrapolation in maturity)")):
        for model, ml in (("bs", "Black-Scholes"), ("heston", "Heston"), ("dh", "Double Heston"), ("dhj", "Double Heston + jumps")):
            kk = f"crosssection.{setting}.{model}"
            rows.append({"Test": lab, "Model": ml, "Held-out IV RMSE": v(f"{kk}.mean_iv_rmse"), "lo": v(f"{kk}.ci95_lo"), "hi": v(f"{kk}.ci95_hi"), "Median": v(f"{kk}.median")})
    tabs.append(Table("crosssection", f"Cross-sectional out-of-sample test (vol points, independent daily fits, {v('crosssection.random.dh.n_days')} days)", pd.DataFrame(rows),
                      {"Held-out IV RMSE": "{:.3f}", "Median": "{:.3f}"}, [], ci=[("95% interval", "lo", "hi", 3, False)]))

    # 9 identifiability
    rows = []
    for key, lab, unit_fmt in (("n_params", "parameters", "{:.0f}"), ("median_cond_colnorm_ls", "median condition number (columns normalised)", "{:.0f}"),
                               ("median_effective_rank_ls", "median effective rank", "{:.0f}"), ("share_params_within_2pct_of_bound", "share of parameters within 2% of a bound", "{:.0%}"),
                               ("median_param_se_over_range_ls", "median standard error / bound width", "{:.3f}"), ("mean_distinct_sets_tol5", f"mean distinct near-optimal parameter sets ({v('identifiability.dh.starts'):.0f} starts)", "{:.2f}"),
                               ("share_days_2plus_distinct_sets_tol5", "days with 2+ distinct near-optimal sets", "{:.0%}")):
        rows.append({"Quantity": lab, "Double Heston": unit_fmt.format(v(f"identifiability.dh.{key}")), "Double Heston + jumps": unit_fmt.format(v(f"identifiability.dhj.{key}"))})
    tabs.append(Table("identifiability", "How well the parameters are pinned down by one day of prices", pd.DataFrame(rows), {}, [f"{v('identifiability.dh.days')} sampled test days."]))

    # 10 vix
    rows = []
    for sk, lab in (("jump", "jump model, 30-day implied vol"), ("diff", "diffusion Double Heston, 30-day implied vol"), ("atm", "surface ATM 30-day IV")):
        for pk, pl in (("all", "2016-2026"), ("test", "2022-2026 test")):
            kk = f"vix.{sk}.{pk}"
            rows.append({"Series": lab, "Period": pl, "Days": v(f"{kk}.n_levels"), "Correlation of levels": v(f"{kk}.corr_levels"), "Correlation of daily changes": v(f"{kk}.corr_changes"),
                         "Mean gap, model minus VIX (pts)": v(f"{kk}.mean_gap_model_minus_vix"), "RMSE of gap (pts)": v(f"{kk}.rmse_gap")})
    tabs.append(Table("vix_link", "Model-implied 30-day volatility against India VIX", pd.DataFrame(rows),
                      {"Correlation of levels": "{:.3f}", "Correlation of daily changes": "{:.3f}", "Mean gap, model minus VIX (pts)": "{:+.2f}", "RMSE of gap (pts)": "{:.2f}"}, []))

    # 11 svi
    rows = []
    for key, lab in (("all", "all quotes"), ("core", "near the money (|k| <= 0.04)"), ("wings", "wings (|k| > 0.10)")):
        kk = f"svi.pooled.{key}"
        rows.append({"Quotes": lab, "n": v(f"{kk}.quotes"), "SVI (5 parameters per expiry)": v(f"{kk}.svi_rmse"), "Double Heston + jumps": v(f"{kk}.dhj_rmse"), "Double Heston": v(f"{kk}.dh_rmse")})
    tabs.append(Table("svi_reference", "Same-day implied-vol RMSE (vol points) of the SVI reference and the walk-forward models", pd.DataFrame(rows),
                      {c: "{:.3f}" for c in ("SVI (5 parameters per expiry)", "Double Heston + jumps", "Double Heston")},
                      [f"{v('svi.n_days')} sampled test days, {v('svi.fits.n_expiries')} expiries fitted. {v('svi.fits.share_butterfly_violation_anywhere'):.0%} of SVI fits have a butterfly violation somewhere on the grid "
                       f"({v('svi.fits.share_butterfly_violation_in_quote_range'):.1%} inside their quote range)."]))

    # 12 data quality
    rows = []
    prev = None
    fl_ = cfg["data"]["filters"]
    for c, lab in (("nifty_options", "NIFTY option rows in the files"), ("traded", "traded (volume > 0)"),
                   ("expiry_ge3d", f"expiry at least {fl_['min_days_to_expiry']} days away"), ("has_forward", "put-call parity gives a forward"),
                   ("otm", "out of the money"), ("price_ge1", f"price at least Rs {fl_['min_price_rupees']:g}"),
                   ("moneyness_ok", f"|ln K/F| <= {fl_['max_abs_log_moneyness']}"), ("iv_ok", "implied vol solves"), ("final", "final quotes")):
        val = v(f"data.total.{c}")
        rows.append({"Filter": lab, "Quotes": val, "Share of the previous step": (val / prev) if prev else np.nan, "Share of all NIFTY rows": val / v("data.total.nifty_options")})
        prev = val
    tabs.append(Table("data_quality_funnel", "Data-quality funnel: quotes surviving each filter, 2016-2026", pd.DataFrame(rows),
                      {"Quotes": "{:,.0f}", "Share of the previous step": "{:.1%}", "Share of all NIFTY rows": "{:.1%}"},
                      [f"{v('data.total.days_with_surface')} of {v('data.total.files')} daily files give a usable surface; {v('data.quotes_per_surface_mean'):.0f} final quotes per usable day on average."]))

    # 13 no-arb sensitivity
    rows = []
    for ex, el in (("none_all_quotes_as_published", "all quotes (as published)"), ("a_monotonic", "exclude monotonicity violators"), ("b_butterfly", "exclude butterfly violators"),
                   ("c_calendar", "exclude calendar violators"), ("any_violation", "exclude any violator")):
        k1, k2 = f"noarb.sens.1tick.{ex}", f"noarb.sens.1tick_025vol.{ex}"
        rows.append({"Scored quotes": el, "Share excluded (1 tick)": v(f"{k1}.mean_share_of_quotes_excluded"), "Time-scaled smile": v(f"{k1}.smile_scaled_rmse"),
                     "Jump model + misfit": v(f"{k1}.jump_carry_res_rmse"), "Difference (1 tick)": v(f"{k1}.diff_model_minus_benchmark"),
                     "Share excluded (1 tick + 0.25 pt)": v(f"{k2}.mean_share_of_quotes_excluded"), "Difference (1 tick + 0.25 pt)": v(f"{k2}.diff_model_minus_benchmark")})
    tabs.append(Table("noarb_sensitivity", "No-arbitrage sensitivity: next-day error (vol points) scored without the target day's violating quotes, test period", pd.DataFrame(rows),
                      {"Share excluded (1 tick)": "{:.1%}", "Time-scaled smile": "{:.3f}", "Jump model + misfit": "{:.3f}", "Difference (1 tick)": "{:+.3f}",
                       "Share excluded (1 tick + 0.25 pt)": "{:.1%}", "Difference (1 tick + 0.25 pt)": "{:+.3f}"},
                      ["Predictions and calibrations are unchanged; only the scored quotes differ. Difference = jump model minus benchmark."]))

    # 14 trading
    rows = []
    for lab, key in (("signal day close", "signal_close"), ("one day later", "one_day_later")):
        for tag in ("forecast_edge / plain", "forecast_edge / hedged", "rich_cheap_edge / plain", "rich_cheap_edge / hedged"):
            base = f"trading.{key}.{slug(tag)}"
            for cn, cl in (("cost0", "0%"), ("cost1pct", "1%")):
                rows.append({"Entry": lab, "Signal / holding": tag, "Cost per side": cl, "Trades": v(f"{base}.trades"), "Wins per 100": v(f"{base}.{cn}.wins_per_100"),
                             "Blind, same mix": v(f"{base}.{cn}.blind_same_mix_wins_per_100"), "Rs per trade": v(f"{base}.{cn}.mean_rs"),
                             "Rs lo": v(f"{base}.{cn}.mean_rs_ci_lo"), "Rs hi": v(f"{base}.{cn}.mean_rs_ci_hi")})
    tabs.append(Table("trading_summary", "Following the jump model one trade at a time, test years (backtest, not advice)", pd.DataFrame(rows),
                      {"Trades": "{:,.0f}", "Wins per 100": "{:.1f}", "Blind, same mix": "{:.1f}", "Rs per trade": "{:+.2f}"},
                      ["Ranges resample whole days. Closing prints are last trades, not quotes you can deal at."],
                      ci=[("95% range (Rs per trade)", "Rs lo", "Rs hi", 2, True)]))

    # 15 robustness and memory
    rows = [{"Quantity": "jump model + misfit, test error over saved reruns (min)", "Value": v("robustness.final_forecast.test_min")},
            {"Quantity": "jump model + misfit, test error over saved reruns (max)", "Value": v("robustness.final_forecast.test_max")},
            {"Quantity": "pure jump model, test error over saved reruns (min)", "Value": v("robustness.pure_model_forecast.test_min")},
            {"Quantity": "pure jump model, test error over saved reruns (max)", "Value": v("robustness.pure_model_forecast.test_max")},
            {"Quantity": "pattern memory on the jump forecast: paired difference", "Value": v("memory.jump_misfit.paired_diff")},
            {"Quantity": "pattern memory on the time-scaled smile: paired difference", "Value": v("memory.smile_scaled.paired_diff")}]
    tabs.append(Table("robustness_memory", "Robustness of the final forecast and the pattern-memory effect (vol points)", pd.DataFrame(rows), {"Value": "{:.3f}"},
                      [f"Reruns: {v('robustness.final_forecast.n_runs')} jump-model runs (windows, refit schedules, priors, memory priors, bounds). Memory intervals: jump forecast {_ci(v('memory.jump_misfit.ci95_lo'), v('memory.jump_misfit.ci95_hi'), 3, True)}, time-scaled smile "
                       f"{_ci(v('memory.smile_scaled.ci95_lo'), v('memory.smile_scaled.ci95_hi'), 3, True)}."]))
    if [t.name for t in tabs] != TABLE_NAMES:
        raise RuntimeError(f"build_tables produced {[t.name for t in tabs]}, TABLE_NAMES says {TABLE_NAMES}")
    return tabs


# =================================================================================================================== report.md
def render_report(m: Metrics, cfg: dict, tables: list[Table], figures_dir: str = "figures") -> str:
    v = m.v
    from .report_figures import FIG
    figs = lambda keys: "Figures: " + ", ".join(f"`{figures_dir}/{FIG[k]}.png`" for k in keys) + "."
    T_ = {t.name: t for t in tables}
    nb = LABELS[v("forecast.best_benchmark")]
    mk = "forecast.dhj_main.carry_res"
    L = ["# Double Heston 2.0 on NIFTY 50: results report",
         "",
         f"*Generated by `dh2/report.py` from `results/metrics.json` ({len(m.items)} numbers, each with its source file and column or key). Nothing in this file is typed by hand. "
         f"Development period {cfg['split']['development']['start'].year}-{cfg['split']['development']['end'].year}, test period {cfg['split']['test']['start'].year} onward; "
         "errors are implied-volatility points. Not investment advice.*",
         "",
         "## 1. Headline: next-day forecast error", "",
         f"On the {v('forecast.smile.n_test_days')} test days the jump model plus yesterday's misfit has a mean next-day error of {v(mk + '.test_mean'):.3f} vol points; the best model-free "
         f"benchmark ({nb}) has {v('forecast.' + v('forecast.best_benchmark') + '.test_mean'):.3f}. The paired difference is {v(mk + '.diff_vs_best_benchmark'):+.3f} "
         f"{_ci(v(mk + '.diff_ci_lo'), v(mk + '.diff_ci_hi'), 3, True)} (95% block bootstrap). The model beats the time-scaled smile in "
         f"{v('forecast_by_year.years_jump_misfit_beats_time_scaled_smile')} of {v('forecast_by_year.years_total')} calendar years.", "",
         to_markdown(T_["forecast_models"]), figs(["bars", "by_year", "summary"]), "",
         "## 2. Is the difference real? Paired tests", "", to_markdown(T_["significance_tests"]), "",
         "## 3. From the 1.0 recipe to 2.0", "",
         f"Same-day error falls from {v('attribution.step0.in_mean'):.2f} (1.0 recipe) to {v('attribution.step5.in_mean'):.2f} vol points; the next-day forecast error from "
         f"{v('attribution.step0.oos_mean'):.2f} to {v('attribution.step6.oos_mean'):.2f}.", "", to_markdown(T_["waterfall"]), figs(["waterfall"]), "",
         "## 4. Where the models miss, the SVI yardstick, cross-section, identifiability", "",
         f"Pooled same-day error over the sampled test days: one volatility {v('buckets.pooled.bs.iv_rmse'):.2f}, Heston {v('buckets.pooled.heston.iv_rmse'):.2f}, Double Heston "
         f"{v('buckets.pooled.dh.iv_rmse'):.2f}, with jumps {v('buckets.pooled.dhj.iv_rmse'):.2f} vol points.", "",
         to_markdown(T_["svi_reference"]), to_markdown(T_["crosssection"]), to_markdown(T_["identifiability"]),
         figs(["smiles", "miss", "svi_day", "svi_bucket", "xsec", "ident"]), "",
         "## 5. Pricer validation", "",
         f"On real chains the project pricer differs from the independent COS pricer by at most {v('pricer.real_chains.dh.project_vs_cos.max_abs_iv_err_pts'):.1e} vol points "
         f"({v('pricer.real_chains.dh.project_vs_cos.n'):,} points, diffusion model). Largest synthetic-grid difference {v('pricer.synthetic_grid.dh.project_vs_cos.max_abs_iv_err_pts'):.1e} vol points.", "",
         to_markdown(T_["pricer_validation"]), figs(["pricer"]), "",
         "## 6. PINN surrogate", "",
         f"Held-out mean implied-vol error of the shipped data-only network: {v('pinn.ablation.data_only.real.iv_err_mean'):.3f} vol points; hybrid {v('pinn.ablation.hybrid.real.iv_err_mean'):.3f}; "
         f"physics-only {v('pinn.ablation.physics_only.real.iv_err_mean'):.2f}.", "",
         to_markdown(T_["pinn_accuracy"]), to_markdown(T_["pinn_speed"]), figs(["pinn_abl", "pinn_de", "pinn_err", "pinn_speed"]), "",
         "## 7. PINN-accelerated calibration", "", to_markdown(T_["pinn_calibration"]), figs(["cal1", "cal3"]), "",
         "## 8. Link to India VIX", "", to_markdown(T_["vix_link"]), figs(["vix_ts", "vix_ch"]), "",
         "## 9. Data quality and no-arbitrage", "", to_markdown(T_["data_quality_funnel"]), to_markdown(T_["noarb_sensitivity"]), "",
         "## 10. Trading test", "", to_markdown(T_["trading_summary"]), "",
         "## 11. Robustness and pattern memory", "", to_markdown(T_["robustness_memory"]), "",
         "## How to reproduce", "",
         "`python -m dh2.run_all --config configs/nifty.yaml` rebuilds every table, figure and number from the cached data (stages whose outputs exist are skipped; "
         "`--list` shows the cost of each); `--quick` is the demo-scale run in `results_quick/`. See `results/PIPELINE.md`.", ""]
    return "\n".join(L)


def build_all(cfg: dict, root: Path = ROOT, cfg_path: str = "configs/nifty.yaml") -> tuple[Metrics, list[Table]]:
    m = build_metrics(cfg, root)
    write_metrics(m, root / cfg["paths"]["results"] / "metrics.json", cfg_path)
    tabs = build_tables(m, cfg)
    for t in tabs:
        write_table(t, root / cfg["paths"]["tables"])
    return m, tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nifty.yaml")
    ap.add_argument("--forecast-tables", action="store_true", help="also (re)build results/tables/forecast_*.csv from data/results/hyb_*.pkl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.forecast_tables:
        build_forecast_tables(cfg)
    m, tabs = build_all(cfg, ROOT, a.config)
    rep = render_report(m, cfg, tabs)
    (ROOT / cfg["paths"]["results"] / "report.md").write_text(rep)
    print(f"{len(m.items)} metrics -> results/metrics.json; {len(tabs)} tables -> {cfg['paths']['tables']}; report.md written")


if __name__ == "__main__":
    main()
