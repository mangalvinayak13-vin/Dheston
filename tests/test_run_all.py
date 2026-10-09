"""Fast tests of the one-command pipeline (dh2/run_all.py), the metrics collector and the table writers (dh2/report.py).

Nothing here runs a real stage: the stage graph is checked as data, skip-if-cached and failure handling run on fake stages in a temp dir, `--list` and
`--dry-run` only print, and the collector is exercised on tiny files it must either read exactly or refuse loudly.
"""
import inspect
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dh2 import report as RP, run_all as RA  # noqa: E402

CFG_PATH = ROOT / "configs" / "nifty.yaml"


@pytest.fixture(scope="module")
def cfg():
    return RA.load_config(CFG_PATH)


# ------------------------------------------------------------------ config
def test_config_parses_and_has_every_required_key(cfg):
    for sec, keys in RA.REQUIRED.items():
        for k in keys:
            assert k in cfg[sec], f"{sec}.{k}"
    assert cfg["execution"]["workers"] <= cfg["execution"]["max_workers"]
    assert cfg["split"]["test"]["start"] == date(2022, 1, 1) and cfg["split"]["development"]["end"] == date(2021, 12, 31)


def test_config_missing_key_stops_with_a_clear_message(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths: {raw_data: x}\n")
    with pytest.raises(SystemExit) as e:
        RA.load_config(bad)
    assert "missing" in str(e.value)
    with pytest.raises(SystemExit):
        RA.load_config(tmp_path / "nope.yaml")


def test_config_equals_the_defaults_and_constants_in_the_code(cfg):
    src = (ROOT / "run_walk.py").read_text()
    w = cfg["walk_forward"]
    for flag, key in (("--window", "window"), ("--refit", "refit"), ("--burn", "burn"), ("--per-expiry", "per_expiry"), ("--nfev-cold", "nfev_cold"), ("--nfev-warm", "nfev_warm"), ("--prior", "prior")):
        m = re.search(rf'add_argument\("{flag}"[^)]*default=([0-9.]+)', src)
        assert m, flag
        assert float(m.group(1)) == float(w[key]), f"{flag}: run_walk.py default {m.group(1)} vs config {w[key]}"
    m = re.search(r"run_walk\.py --model dhj --tag main --workers \d+ --chunks (\d+) --burn (\d+)", (ROOT / "README.md").read_text())
    if m:                                                                               # the command that produced the shipped results, as documented in the README
        assert int(m.group(1)) == w["chunks"] and int(m.group(2)) == w["burn"]
    assert re.search(r"7 \+ j", src) and w["seed_base"] == 7                           # chunk j uses seed 7 + j
    import final_table, tournament
    from dh2 import data, stats, svi
    for mod in (final_table, stats, svi):
        assert mod.SPLIT == cfg["split"]["test"]["start"]
    f = cfg["data"]["filters"]
    assert (data.MIN_DAYS, data.MIN_PRICE, data.MAX_LOG_MONEYNESS, data.PARITY_STRIKES) == (f["min_days_to_expiry"], f["min_price_rupees"], f["max_abs_log_moneyness"], f["parity_strikes"])
    sig = inspect.signature(tournament.boot).parameters
    b = cfg["evaluation"]["bootstrap"]
    assert (sig["block"].default, sig["n"].default, sig["seed"].default) == (b["block"], b["n"], b["seed"])


def test_bounds_in_the_config_equal_calibrator_bounds(cfg):
    if os.environ.get("DH2_WIDE"):
        pytest.skip("DH2_WIDE changes the calibrator's bounds")
    from dh2 import calib
    for model, bnd in cfg["walk_forward"]["bounds"].items():
        lo, hi = calib.BOUNDS[model]
        assert np.allclose(bnd["lo"], lo) and np.allclose(bnd["hi"], hi), model


def test_paired_sample_rule_matches_the_code(cfg):
    from dh2 import svi
    import inspect as _i
    assert _i.signature(svi.sample_days).parameters["step"].default == cfg["evaluation"]["paired_sample"]["every_nth"]


def test_workers_are_clipped_to_the_configured_maximum(cfg, tmp_path):
    c = RA.Ctx(cfg, tmp_path, workers=64)
    assert c.workers == cfg["execution"]["max_workers"]
    assert RA.Ctx(cfg, tmp_path, workers=1).workers == 1


# ------------------------------------------------------------------ the stage graph
@pytest.mark.parametrize("quick", [False, True])
def test_stage_graph_is_acyclic_with_declared_outputs(cfg, quick):
    stages = RA.build_stages(cfg, quick)
    RA.validate_graph(stages)                       # acyclic, known dependencies, outputs declared and unique
    order = RA.topo_order(stages)
    pos = {n: i for i, n in enumerate(order)}
    for n, s in stages.items():
        assert s.outputs and s.run is not None and s.est_min > 0 and s.basis, n
        for d in s.deps:
            assert pos[d] < pos[n], f"{n} before its dependency {d}"
        if not s.optional:
            assert all(not stages[d].optional for d in s.deps), f"{n} depends on an optional stage"


def test_quick_stages_write_only_to_results_quick_and_full_stages_never_do(cfg):
    q = cfg["paths"]["results_quick"] + "/"
    for s in RA.build_stages(cfg, True).values():
        assert all(o.startswith(q) for o in s.outputs), s.name
    for s in RA.build_stages(cfg, False).values():
        assert not any(o.startswith(q) for o in s.outputs), s.name


def test_the_stages_named_in_the_brief_exist(cfg):
    st = RA.build_stages(cfg, False)
    for need in ("data_check", "tests_fast", "walk_dh", "walk_dhj", "walk_heston", "forecasts_dhj", "benchmark_table", "trading_tests", "pricer_validation", "pinn_validation",
                 "dq", "noarb", "vix", "svi", "stats", "attribution", "crosssection", "identifiability", "buckets", "pinn_calib_report", "ui_adapter", "metrics_tables", "figures", "report"):
        assert need in st, need
    assert st["walk_dh"].est_min >= cfg["execution"]["confirm_over_minutes"]       # a walk-forward is flagged expensive


def test_new_stages_exist_with_dependencies_and_outputs(cfg):
    st = RA.build_stages(cfg, False)
    assert st["wing_benchmark"].deps == ["data_check", "forecasts_dhj"]
    assert "results/benchmark_check/summary.csv" in st["wing_benchmark"].outputs
    assert set(st["forecast_variants"].deps) == {"forecasts_dh", "forecasts_dhj", "forecasts_heston"}
    assert "results/tables/forecast_variants.csv" in st["forecast_variants"].outputs
    assert {"wing_benchmark", "forecast_variants"} <= set(st["metrics_tables"].deps)
    assert len(st) == 36


def test_plain_run_needs_only_the_two_checkpoints_in_git_and_retrain_needs_all_sixteen(cfg, tmp_path):
    st = RA.build_stages(cfg, False)
    ck = cfg["paths"]["pinn_checkpoints"]
    assert st["pinn_checkpoints"].outputs == [f"{ck}/final.pt", f"{ck}/final_greeks.pt"]
    (tmp_path / ck).mkdir(parents=True)
    for n in RA.PINN_IN_GIT:                                         # a fresh clone: only the two tracked checkpoints
        (tmp_path / ck / f"{n}.pt").write_text("x")
    assert RA.is_cached(st["pinn_checkpoints"], tmp_path)
    st = RA.apply_retrain(RA.build_stages(cfg, False), True)
    assert len(st["pinn_checkpoints"].outputs) == 16 and not RA.is_cached(st["pinn_checkpoints"], tmp_path)


def test_plain_pytest_is_the_fast_suite():
    ini = (ROOT / "pytest.ini").read_text()
    assert 'addopts = -m "not slow"' in ini


def test_graph_checks_catch_a_cycle_an_unknown_dependency_and_missing_outputs():
    mk = lambda name, deps=(), outs=("x",): RA.Stage(name, name, list(outs), list(deps), lambda c: None)
    with pytest.raises(ValueError, match="cycle"):
        RA.validate_graph({"a": mk("a", ["b"], ["a.out"]), "b": mk("b", ["a"], ["b.out"])})
    with pytest.raises(ValueError, match="unknown"):
        RA.validate_graph({"a": mk("a", ["zzz"], ["a.out"])})
    with pytest.raises(ValueError, match="no outputs"):
        RA.validate_graph({"a": mk("a", [], [])})
    with pytest.raises(ValueError, match="declared by both"):
        RA.validate_graph({"a": mk("a", [], ["same"]), "b": mk("b", [], ["same"])})


# ------------------------------------------------------------------ skip-if-cached, force, failures (fake stages in a temp dir)
def fake_graph(tmp_path, calls, fail=None):
    def writer(name):
        def run(c):
            calls.append(name)
            if name == fail:
                raise RA.StageError("boom")
            for o in S[name].outputs:
                (c.root / o).parent.mkdir(parents=True, exist_ok=True)
                (c.root / o).write_text("x")
        return run
    S = {}
    for n, deps in (("a", []), ("b", ["a"]), ("c", ["b"]), ("d", ["a"])):
        S[n] = RA.Stage(n, n, [f"out/{n}.txt"], deps, writer(n), est_min=1.0, basis="test")
    return S


def test_stages_with_existing_outputs_are_skipped_unless_forced(cfg, tmp_path):
    calls = []
    S = fake_graph(tmp_path, calls)
    ctx = RA.Ctx(cfg, tmp_path, out=lambda m: None)
    plan = RA.make_plan(S, None, False, tmp_path)
    assert [i.action for i in plan] == ["run"] * 4
    res = RA.execute_plan(plan, ctx)
    assert [r.status for r in res] == ["ran"] * 4 and calls == ["a", "b", "c", "d"]
    plan = RA.make_plan(S, None, False, tmp_path)
    assert all(i.action == "skip" for i in plan)
    assert all(r.status == "skipped" for r in RA.execute_plan(plan, ctx)) and calls == ["a", "b", "c", "d"]        # nothing ran again
    plan = RA.make_plan(S, ["b"], True, tmp_path)                       # forcing b reruns b only, not its dependency a
    assert [(i.stage.name, i.action) for i in plan] == [("a", "skip"), ("b", "run")]
    RA.execute_plan(plan, ctx)
    assert calls[-1] == "b" and calls.count("a") == 1
    assert RA.stale_downstream(S, plan) == []                           # c is not in this plan
    full = RA.make_plan(S, None, True, tmp_path)
    assert all(i.action == "run" for i in full)


def test_a_missing_dependency_output_runs_the_dependency_but_a_cached_one_does_not(cfg, tmp_path):
    calls = []
    S = fake_graph(tmp_path, calls)
    ctx = RA.Ctx(cfg, tmp_path, out=lambda m: None)
    RA.execute_plan(RA.make_plan(S, None, False, tmp_path), ctx)
    calls.clear()
    (tmp_path / "out" / "b.txt").unlink()
    plan = RA.make_plan(S, ["c"], False, tmp_path)                      # c is cached but b is missing: only b reruns; c is reported as possibly stale
    assert [(i.stage.name, i.action) for i in plan] == [("a", "skip"), ("b", "run"), ("c", "skip")]
    assert RA.stale_downstream(S, plan) == ["c"]
    RA.execute_plan(plan, ctx)
    assert calls == ["b"]


def test_failure_blocks_dependents_and_is_reported(cfg, tmp_path):
    calls = []
    S = fake_graph(tmp_path, calls, fail="b")
    ctx = RA.Ctx(cfg, tmp_path, out=lambda m: None)
    res = RA.execute_plan(RA.make_plan(S, None, False, tmp_path), ctx, keep_going=True)
    st = {r.name: r.status for r in res}
    assert st == {"a": "ran", "b": "failed", "c": "blocked", "d": "ran"}
    assert "boom" in next(r.detail for r in res if r.name == "b")
    calls.clear()
    res2 = RA.execute_plan(RA.make_plan(S, None, True, tmp_path), RA.Ctx(cfg, tmp_path, out=lambda m: None), keep_going=False)
    assert [r.status for r in res2][-1] == "failed" and "d" not in calls            # stops at the first failure without --keep-going


def test_timing_record_marks_quick_runs_and_keeps_a_history(cfg, tmp_path):
    q = RA.Ctx(cfg, tmp_path, quick=True)
    res = [RA.Result("a", "ran", 1.5, 2.0), RA.Result("b", "skipped", 0.0, 1.0, "cached")]
    p = RA.write_last_run(res, q, 1.5, None)
    rec = json.loads(p.read_text())
    assert rec["notice"].startswith("DEMO-SCALE RUN") and rec["quick"] and p.parent == tmp_path / cfg["paths"]["results_quick"] / "run_logs"
    full = RA.Ctx(cfg, tmp_path, quick=False)
    p2 = RA.write_last_run(res, full, 1.5, None)
    assert "notice" not in json.loads(p2.read_text())
    RA.write_last_run(res, full, 2.5, None)
    hist = [json.loads(l) for l in (p2.parent / "run_history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and [s["name"] for s in hist[0]["stages"]] == ["a"]          # skipped stages are not part of the history


def test_a_stage_that_does_not_produce_its_outputs_is_a_failure(cfg, tmp_path):
    S = {"a": RA.Stage("a", "a", ["out/a.txt"], [], lambda c: None, 1.0, "t")}
    res = RA.execute_plan(RA.make_plan(S, None, False, tmp_path), RA.Ctx(cfg, tmp_path, out=lambda m: None))
    assert res[0].status == "failed" and "outputs are missing" in res[0].detail


def test_unknown_stage_name_is_refused(cfg, tmp_path):
    S = fake_graph(tmp_path, [])
    with pytest.raises(SystemExit):
        RA.make_plan(S, ["nope"], False, tmp_path)


def test_an_empty_output_file_does_not_count_as_cached(tmp_path):
    s = RA.Stage("a", "a", ["o.txt"], [], None, 1.0, "t")
    (tmp_path / "o.txt").write_text("")
    assert not RA.is_cached(s, tmp_path)
    (tmp_path / "o.txt").write_text("x")
    assert RA.is_cached(s, tmp_path)


# ------------------------------------------------------------------ the command line
def test_list_prints_every_stage_and_runs_nothing(cfg):
    for argv, quick in ((["--list"], False), (["--list", "--quick"], True)):
        lines = []
        rc = RA.main(["--config", str(CFG_PATH)] + argv, out=lines.append)
        text = "\n".join(lines)
        assert rc == 0
        for n in RA.build_stages(cfg, quick):
            assert n in text, n
        assert "cost basis" in text and "est. time" in text


def test_dry_run_prints_the_plan_with_costs_and_runs_nothing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(RA, "ROOT", tmp_path)
    monkeypatch.setattr(RA, "build_stages", lambda cfg, quick=False: fake_graph(tmp_path, calls))
    lines = []
    rc = RA.main(["--config", str(CFG_PATH), "--dry-run"], out=lines.append)
    assert rc == 0 and calls == [] and any("RUN" in l and "est" in l for l in lines) and not (tmp_path / "out").exists()


def test_expensive_stages_need_confirmation_and_are_not_started_silently(monkeypatch, tmp_path):
    calls = []

    def graph(cfg, quick=False):
        S = fake_graph(tmp_path, calls)
        S["a"].est_min = 600.0
        return S

    monkeypatch.setattr(RA, "ROOT", tmp_path)
    monkeypatch.setattr(RA, "build_stages", graph)
    monkeypatch.setattr(sys, "stdin", open(os.devnull))              # not a terminal: no prompt possible
    lines = []
    assert RA.main(["--config", str(CFG_PATH)], out=lines.append) == 2 and calls == [] and any("--yes" in l for l in lines)
    assert RA.main(["--config", str(CFG_PATH), "--yes"], out=lambda m: None) == 0 and calls == ["a", "b", "c", "d"]
    assert (tmp_path / "results" / "run_logs" / "last_run.json").exists()                   # honest timing record
    rec = json.loads((tmp_path / "results" / "run_logs" / "last_run.json").read_text())
    assert [s["status"] for s in rec["stages"]] == ["ran"] * 4 and rec["wall_seconds"] >= 0


def test_nothing_to_run_when_everything_is_cached(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(RA, "ROOT", tmp_path)
    monkeypatch.setattr(RA, "build_stages", lambda cfg, quick=False: fake_graph(tmp_path, calls))
    assert RA.main(["--config", str(CFG_PATH)], out=lambda m: None) == 0 and len(calls) == 4
    lines = []
    assert RA.main(["--config", str(CFG_PATH)], out=lines.append) == 0 and len(calls) == 4 and any("Nothing to run" in l for l in lines)


# ------------------------------------------------------------------ the metrics collector fails loudly
def test_collector_reads_exact_cells_and_records_the_source(tmp_path):
    pd.DataFrame({"run": ["a", "b"], "x": [1.5, 2.5]}).to_csv(tmp_path / "t.csv", index=False)
    (tmp_path / "j.json").write_text(json.dumps({"k": {"v": [3.0, 4.0]}}))
    m = RP.Metrics(tmp_path)
    assert m.cell("m1", "t.csv", "x", "pts", {"run": "b"}) == 2.5
    assert m.jcell("m2", "j.json", ["k", "v", 1], "pts") == 4.0
    r = m.items["m1"]
    assert r["source"] == "t.csv" and "x" in r["key"] and "run == 'b'" in r["key"] and r["unit"] == "pts" and r["kind"] == "read"
    d = m.derived("m3", 6.0, "pts", ["t.csv"], "sum of column 'x' over rows a, b is wrong on purpose: rule text")
    assert d == 6.0 and m.items["m3"]["kind"] == "derived"
    js = m.to_json()
    assert js["n_metrics"] == 3 and set(js["sources"]) == {"t.csv", "j.json"} and len(js["sources"]["t.csv"]["sha256"]) == 64


def test_collector_refuses_a_missing_file_column_key_or_row(tmp_path):
    pd.DataFrame({"run": ["a", "a", "b"], "x": [1.0, 2.0, np.nan]}).to_csv(tmp_path / "t.csv", index=False)
    (tmp_path / "j.json").write_text(json.dumps({"k": 1}))
    m = RP.Metrics(tmp_path)
    with pytest.raises(RP.MissingSource, match="missing source file"):
        m.cell("a", "nope.csv", "x")
    with pytest.raises(RP.MissingSource, match="no column 'zzz'"):
        m.cell("b", "t.csv", "zzz", where={"run": "b"})
    with pytest.raises(RP.MissingSource, match="no column 'qq'"):
        m.cell("c", "t.csv", "x", where={"qq": 1})
    with pytest.raises(RP.MissingSource, match="expected exactly one row"):
        m.cell("d", "t.csv", "x", where={"run": "a"})                      # two rows match: ambiguous, no silent first-row pick
    with pytest.raises(RP.MissingSource, match="expected exactly one row"):
        m.cell("e", "t.csv", "x", where={"run": "zzz"})                    # no row at all
    with pytest.raises(RP.MissingSource, match="not finite"):
        m.cell("f", "t.csv", "x", where={"run": "b"})                      # NaN is not a number
    with pytest.raises(RP.MissingSource, match="key path"):
        m.jcell("g", "j.json", ["k", "deeper"])
    with pytest.raises(RP.MissingSource, match="missing source file"):
        m.derived("h", 1.0, "", ["gone.csv"], "rule")
    with pytest.raises(RP.MissingSource, match="not collected"):
        m.v("never_added")
    assert not m.items                                                      # nothing was recorded by any failed read


def test_collector_rejects_duplicate_ids(tmp_path):
    pd.DataFrame({"x": [1.0]}).to_csv(tmp_path / "t.csv", index=False)
    m = RP.Metrics(tmp_path)
    m.cell("id", "t.csv", "x")
    with pytest.raises(ValueError, match="duplicate"):
        m.cell("id", "t.csv", "x")


def test_build_metrics_on_an_empty_tree_fails_loudly(cfg, tmp_path):
    with pytest.raises(RP.MissingSource):
        RP.build_metrics(cfg, tmp_path)


def test_a_missing_column_in_a_real_result_file_is_caught(cfg, tmp_path):
    need = ROOT / "results" / "dq" / "data_quality_by_year.csv"
    if not need.exists():
        pytest.skip("results not present")
    (tmp_path / "results" / "dq").mkdir(parents=True)
    pd.read_csv(need).drop(columns=["traded"]).to_csv(tmp_path / "results" / "dq" / "data_quality_by_year.csv", index=False)
    with pytest.raises(RP.MissingSource, match="traded"):
        RP.collect_data(RP.Metrics(tmp_path), cfg)


def test_metrics_from_the_real_results_have_sources_when_present(cfg):
    need = ROOT / "results" / "tables" / "forecast_scores.csv"
    if not need.exists() or not (ROOT / "results" / "pinn_calib" / "headline.json").exists():
        pytest.skip("results not present")
    m = RP.build_metrics(cfg, ROOT)
    assert len(m.items) > 500
    assert all(r["source"] and r["key"] and r["kind"] in ("read", "derived") and r["unit"] is not None for r in m.items.values())
    assert abs(m.v("forecast.dhj_main.carry_res.test_mean") - 1.0336) < 5e-4                      # the headline number of the README


def test_benchmark_check_and_variants_metrics_when_present(cfg):
    if not (ROOT / "results" / "benchmark_check" / "summary.csv").exists() or not (ROOT / "results" / "tables" / "forecast_variants.csv").exists():
        pytest.skip("results not present")
    m = RP.Metrics(ROOT)
    RP.collect_benchmark_check(m, cfg)
    RP.collect_variants(m, cfg)
    assert abs(m.v("benchmark_check.all.model_vs_edge.mean_difference") + 0.2021) < 5e-4
    assert abs(m.v("benchmark_check.all.model_vs_linear.mean_difference") + 0.0445) < 5e-4
    assert m.v("benchmark_check.inside.model_vs_edge.mean_difference") > 0 > m.v("benchmark_check.outside.model_vs_edge.mean_difference")
    assert abs(m.v("variants.dhj_main.carry_res.test_mean") - 1.0336) < 5e-4


def test_collector_refuses_a_missing_benchmark_check(cfg, tmp_path):
    with pytest.raises(RP.MissingSource):
        RP.collect_benchmark_check(RP.Metrics(tmp_path), cfg)
    with pytest.raises(RP.MissingSource):
        RP.collect_variants(RP.Metrics(tmp_path), cfg)


# ------------------------------------------------------------------ tables and the cross-check of final_table.py
def test_tables_are_written_as_csv_markdown_and_latex_booktabs(tmp_path):
    df = pd.DataFrame({"Forecast": ["a_b", "c%"], "Test": [1.23456, 2.0], "lo": [1.0, 1.5], "hi": [1.5, 2.5], "n": [10, 2000]})
    tb = RP.Table("demo", "A caption with 5% and an underscore_here", df, {"Test": "{:.2f}"}, ["a note"], ci=[("95% interval", "lo", "hi", 2, False)])
    RP.write_table(tb, tmp_path)
    tex = (tmp_path / "demo.tex").read_text()
    for tok in (r"\toprule", r"\midrule", r"\bottomrule", r"\begin{tabular}", r"a\_b", r"c\%", "[1.00, 1.50]"):
        assert tok in tex, tok
    md = (tmp_path / "demo.md").read_text()
    assert "| Forecast | Test | 95% interval | n |" in md and "1.23" in md and "2,000" in md
    assert "lo" in pd.read_csv(tmp_path / "demo.csv").columns            # the CSV keeps the numbers at full precision
    assert pd.read_csv(tmp_path / "demo.csv")["Test"][0] == 1.23456


def test_crosscheck_with_final_table_printout():
    text = ("1256 common days\n\nrun  variant dev test\ndhj_main                  carry      1.734   1.201   -0.035 [-0.095, +0.028]\n"
            "dhj_main                  carry_res  1.317   1.034   -0.202 [-0.227, -0.181]\n")
    parsed = RP.parse_final_table(text)
    assert parsed[("dhj_main", "carry_res")] == (1.317, 1.034, -0.202, -0.227, -0.181)
    models = pd.DataFrame([{"run": "dhj_main", "variant": "carry", "dev_mean": 1.7341, "test_mean": 1.2009, "diff_vs_best_benchmark": -0.0352, "diff_ci_lo": -0.0951, "diff_ci_hi": 0.0281},
                           {"run": "dhj_main", "variant": "carry_res", "dev_mean": 1.3172, "test_mean": 1.0336, "diff_vs_best_benchmark": -0.2021, "diff_ci_lo": -0.2271, "diff_ci_hi": -0.1809}])
    assert RP.crosscheck_final_table(models, text) == []
    models.loc[1, "test_mean"] = 1.2
    assert any("test" in b for b in RP.crosscheck_final_table(models, text))
    assert RP.crosscheck_final_table(models, "nothing here")                     # an unparseable printout is a failure, not a pass


def test_table_names_constant_matches_the_tables_the_report_builds(cfg):
    need = ROOT / "results" / "tables" / "forecast_scores.csv"
    if not need.exists() or not (ROOT / "results" / "pinn_calib" / "headline.json").exists():
        pytest.skip("results not present")
    m = RP.build_metrics(cfg, ROOT)
    assert [t.name for t in RP.build_tables(m, cfg)] == RP.TABLE_NAMES


# ------------------------------------------------------------------ quick mode and figures
def test_demo_csv_carries_the_banner_and_reads_back(tmp_path):
    from dh2 import quick
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    quick.demo_csv(df, tmp_path / "t.csv")
    first = (tmp_path / "t.csv").read_text().splitlines()[0]
    assert first.startswith("# DEMO-SCALE RUN, NOT THE HEADLINE RESULT")
    assert quick.read_demo_csv(tmp_path / "t.csv").equals(df)
    m = RP.Metrics(tmp_path, comment="#")
    assert m.cell("x", "t.csv", "b", "", {"a": 2}) == 1.5


def test_figure_registry_is_consistent():
    from dh2 import report_figures as RF
    stems = [f.stem for f in RF.REGISTRY]
    assert len(set(stems)) == len(stems) and len({s.split("_")[0] for s in stems}) == len(stems)
    assert all(f.kind in ("new", "copy") and f.caption and f.source and f.data for f in RF.REGISTRY)
    assert all((f.kind == "copy") == (f.src is not None) for f in RF.REGISTRY)
    assert {f.key for f in RF.REGISTRY if f.kind == "new"} == set(RF.NEW_BUILDERS)
    for f in RF.REGISTRY:                                                    # captions are plain sentences: no result numbers typed in
        assert not re.search(r"\d+\.\d+|\d+\s*%", f.caption.replace("1.0", "").replace("2.0", "").replace("95%", "")), f.caption     # (the 95% is the interval level, not a result)
