# Pipeline: one command from the cached data to every table, figure and number

```
python -m dh2.run_all --config configs/nifty.yaml --list      # every stage: cached or missing, estimated cost, outputs, dependencies
python -m dh2.run_all --config configs/nifty.yaml              # everything the cached data allows; a stage whose outputs exist is skipped
python -m dh2.run_all --config configs/nifty.yaml --quick      # demo-scale run, writes ONLY to results_quick/, 5.3 minutes measured on 2 cores
```

Settings are in one file, `configs/nifty.yaml` (paths, the 2016-2021 development / 2022-onward test split, the walk-forward settings of the shipped results: window 40,
refit 10, burn-in 40, bounds, seeds, the every-10th-test-day paired sample, the PINN settings pointer, the quick-mode window and budgets).
`tests/test_run_all.py` checks that the walk-forward settings, the bounds, the split, the data filters and the bootstrap settings in the file equal the defaults and constants in the code.
Environment: `requirements.txt` (pinned versions; Python 3.13, macOS/Apple silicon; nothing needs a GPU to run `--quick`).

## What the command does

| Flag | Meaning |
|---|---|
| (none) | Plan, then run every stage in dependency order. A stage is **skipped when all its declared outputs exist**. |
| `--list` | Print every stage with its state (cached / missing), estimated time and where the estimate comes from, output count and dependencies. Runs nothing. |
| `--dry-run` | Print the plan (what would run and why) and stop. |
| `--stages a,b` | Run these stages plus any dependency whose outputs are missing (cached dependencies are not rerun; a warning names cached stages that would be stale). |
| `--force` | Rerun the selected stages although their outputs exist. Forced dependencies are not implied. A forced walk-forward takes hours: see the plan's estimate. |
| `--retrain-pinn` | Train the PINN checkpoints that are **missing**. The shipped checkpoints are only validated, never retrained by default. |
| `--quick` | The demo-scale pipeline (stages `q_*`), outputs in `results_quick/`, never in `results/`. |
| `--yes` / `--keep-going` / `--workers N` | Skip the confirmation prompt / continue after a failure with independent stages / fewer workers (never more than `execution.max_workers`, which is 2). |

A plan with estimated costs is printed before anything runs. A stage estimated above `execution.confirm_over_minutes` (30), or one that needs the network, asks for confirmation; without a terminal it
refuses to start unless `--yes` is given. The downloaders run only if the raw NSE files are missing. Every stage runs in the foreground, one after the other; subprocess output goes to
`results/run_logs/<stage>.log`, the real elapsed time of each stage to `results/run_logs/last_run.json` and `run_history.jsonl`. No background process is left behind.

Existing scripts are not edited: a stage calls them as subprocesses (`run_walk.py`, `hybrids.py`, `final_table.py`, `trade_*.py`, `exp_*.py`, `dh2.dq`, `dh2.svi`, ...) or imports their functions
(`run_walk.worker`, `hybrids.chunk`, `tournament.boot`, `dh2.stats.compare`, ...).

## Stages

Estimates are the compute times recorded in this repository (README, the work packages' READMEs, file times of the cached results), scaled where the recorded run used more workers; they are
estimates, labelled as such. "Measured here" is the real elapsed time of the stage when it was run through the pipeline in this session; the stages marked "not re-run" were not recomputed
(their results are the cached ones), so their cost is the recorded one.

| # | Stage | What it does | Outputs | Estimated cost with 2 workers (basis) | Measured here |
|---|---|---|---|---|---|
| 1 | `data_check` | Check the raw NSE files and the surface cache (never downloads unless raw files are missing) | 1 | <1 min; seconds when cached; building the cache from the raw files takes minutes (dh2/data.py); the downloaders (only if raw files are missing) take hours | 0.1 s |
| 2 | `tests_fast` | Fast tests (pricer cross-checks, PINN, calibration, statistics, this pipeline) | 1 | <1 min; measured: about 17 s on this machine with warm caches (about 170 fast tests pass, 14 slow ones are deselected, 2 are expected failures); a cold cache adds the one-off surface build | 16.8 s |
| 3 | `walk_dh` | Walk-forward of the dh model (40-day window, refit every 10 days) | 1 | 2.5 h; about 1 h each with 8-9 workers on an idle machine (README, brief: whole programme about 22 h); about 2.5x that with 2 workers (extrapolated, not measured) | not re-run (cached) |
| 4 | `forecasts_dh` | Forecast variants of the dh model (hybrids.py: smile, carry, revert, lev, carry_res, ...) | 1 | 5 min; the hyb_* file was written about 1 minute after its walk file with 8 workers (file times); about 4x with 2 workers | not re-run (cached) |
| 5 | `walk_dhj` | Walk-forward of the dhj model (40-day window, refit every 10 days) | 1 | 2.5 h; about 1 h each with 8-9 workers on an idle machine (README, brief: whole programme about 22 h); about 2.5x that with 2 workers (extrapolated, not measured) | not re-run (cached) |
| 6 | `forecasts_dhj` | Forecast variants of the dhj model (hybrids.py: smile, carry, revert, lev, carry_res, ...) | 1 | 5 min; the hyb_* file was written about 1 minute after its walk file with 8 workers (file times); about 4x with 2 workers | not re-run (cached) |
| 7 | `walk_heston` | Walk-forward of the heston model (40-day window, refit every 10 days) | 1 | 2.5 h; about 1 h each with 8-9 workers on an idle machine (README, brief: whole programme about 22 h); about 2.5x that with 2 workers (extrapolated, not measured) | not re-run (cached) |
| 8 | `forecasts_heston` | Forecast variants of the heston model (hybrids.py: smile, carry, revert, lev, carry_res, ...) | 1 | 5 min; the hyb_* file was written about 1 minute after its walk file with 8 workers (file times); about 4x with 2 workers | not re-run (cached) |
| 9 | `benchmark_table` | Model and benchmark table on the test period (final_table.py logic, paired block-bootstrap intervals), per-year means, robustness over saved runs | 5 | <1 min; measured: 4.3 s (the table, the per-year means, 40 robustness bootstraps and the final_table.py cross-check) | 4.3 s |
| 10 | `trading_tests` | Trading test: one trade at a time, signal-day close and one day later, robustness, matched comparison | 5 | 3 min; measured: 2.2 min for the four commands (re-run in this session; all five JSON files and both trade pickles came out byte-identical) | 2.2 min |
| 11 | `pricer_validation` | Pricer validation: project pricer against COS, Lewis and Monte Carlo (and the mpmath characteristic-function scan) | 3 | 16 min; results/pricer_validation/README.md: make_validation.py about 10 min, cf_scan_mpmath.py about 6 min, one process | not re-run (cached) |
| 12 | `pinn_checkpoints` | PINN checkpoints (shipped; trained only with --retrain-pinn) | 16 | 48 min; results/pinn/README.md: 48 GPU minutes in all for the final runs (only when --retrain-pinn and a checkpoint is missing) | not re-run (cached) |
| 13 | `pinn_validation` | PINN validation from the shipped checkpoints: held-out accuracy, Greeks, arbitrage, speed, PDE residual, domain, figures | 17 | 4 min; results/pinn/README.md: about 3 minutes of validation | not re-run (cached) |
| 14 | `dq` | Data-quality report: quotes surviving each filter, per year and on ten sample days | 3 | 1 min; measured: 46 s with 2 workers (re-run in this session; all outputs byte-identical to the saved ones) | 46.1 s |
| 15 | `noarb` | No-arbitrage diagnostics (monotonicity, butterfly, calendar) and the scoring sensitivity | 3 | <1 min; measured: 2.2 s (re-run in this session; outputs byte-identical) | 2.2 s |
| 16 | `vix_data` | India VIX history (downloaded only if data/vix/india_vix.csv is missing) | 1 | 2 min; fetch_vix.py reads one NSE file per trading day (network) | not re-run (cached) |
| 17 | `vix` | Model-implied 30-day volatility against India VIX | 9 | <1 min; measured: 2.5 s (re-run in this session; CSVs and figures byte-identical) | 2.5 s |
| 18 | `svi` | Raw-SVI reference per expiry against the walk-forward Double Heston fits (118 sampled test days) | 8 | 1 min; measured: 55 s with 2 workers (re-run in this session; outputs identical, the .csv.gz only differs in its gzip timestamp) | 55.2 s |
| 19 | `stats` | Paired Wilcoxon and Diebold-Mariano tests on the test period | 2 | <1 min; measured: 0.7 s (re-run in this session; outputs byte-identical) | 0.7 s |
| 20 | `daily_fits` | Independent same-day fits of four models on the paired sample (prerequisite of the C2 experiments) | 1 | 20 min; cache file times: C2 queue started 12:17, daily_fits_v1.pkl written 12:30 (3 workers, machine shared with other jobs); the script uses 3 workers | not re-run (cached) |
| 21 | `attribution` | DH1 to DH2.0 step-by-step attribution waterfall | 5 | 1.7 h; cache file times: about 1.5 h for the steps 0-2 fits (3 workers, shared machine); needs the old research repository's nifty_comparison_detail.csv for the replication table | not re-run (cached) |
| 22 | `crosssection` | Cross-sectional out-of-sample tests (random 80/20 split and maturity split) | 5 | 4.7 h; cache file times: about 4.7 h (3 workers, shared machine, upper bound) | not re-run (cached) |
| 23 | `buckets` | Per-bucket same-day implied-vol RMSE for all models | 4 | 10 min; progress file times: minutes after the daily fits (reads the cached fits) | not re-run (cached) |
| 24 | `identifiability` | Identifiability: Jacobian condition numbers, standard errors, multi-start solutions | 4 | 1.5 h; cache file times: about 1.5 h (3 workers, shared machine) | not re-run (cached) |
| 25 | `pinn_calib_design` | PINN-accelerated calibration: design on development days | 4 | 30 min; results/pinn_calib/README.md: about 30 minutes, one process | not re-run (cached) |
| 26 | `pinn_calib_test` | PINN-accelerated calibration: the frozen design once on the paired test days | 1 | 15 min; results/pinn_calib/README.md: about 15 minutes | not re-run (cached) |
| 27 | `pinn_calib_report` | PINN-accelerated calibration: tables and figures | 12 | <1 min; measured: 3.9 s from the cached per-day results (re-run in this session; outputs byte-identical) | 3.9 s |
| 28 | `ui_adapter` | UI adapter: today.json with the latest calibrated parameters in the website's format | 1 | <1 min; measured: 0.5 s (re-run in this session; today.json byte-identical) | 0.5 s |
| 29 | `metrics_tables` | Metrics collector (results/metrics.json, every number with its source and column) and the headline tables as CSV, Markdown and LaTeX | 46 | <1 min; measured: 0.3 s | 0.3 s |
| 30 | `figures` | Presentation figures (300 dpi PNG and PDF): new summary / comparison / smile / pricer figures and re-exports of the work packages' figures, with INDEX.md | 68 | <1 min; measured: 5.4 s for 33 figures (png + pdf) | 5.6 s |
| 31 | `report` | Plain-numbers report (results/report.md), every number from results/metrics.json's sources | 1 | <1 min; measured: 0.2 s | 0.2 s |
| 32 | `ablation_runs` *(optional)* | Robustness reruns of the jump model (windows, refit schedules, priors, memory priors): the saved runs behind the robustness table | 11 | 27.5 h; about 1 h each with 8-9 workers (queue1/queue4.sh), 11 runs; about 2.5x with 2 workers (extrapolated) | not re-run (cached) |
| 33 | `comparison_note` *(optional)* | Regenerate COMPARISON_1.0_vs_2.0.md and outputs/comparison.json (update_comparison.py; overwrites the note) | 1 | 3 min; seconds to minutes (estimate) | not re-run (cached) |
| 34 | `legacy_figures` *(optional)* | The five original figures of make_figures.py (outputs/figures); superseded by the numbered figures | 1 | 2 min; seconds (estimate) | not re-run (cached) |

Quick-mode stages:

| # | Stage | What it does | Measured |
|---|---|---|---|
| 1 | `q_data` | Real surfaces of the window 2025-01-02 .. 2025-04-30 (cached; no download) | 0.2 s |
| 2 | `q_walk` | Mini walk-forward of heston, dh, dhj (window 20, refit 5, small budgets) | 5.1 min |
| 3 | `q_forecast` | Forecast variants and the mini forecast table against the model-free benchmarks (paired bootstrap) | 5.0 s |
| 4 | `q_pricer` | Mini pricer cross-check: project pricer against COS (and Lewis on a subset) on real chains | 0.6 s |
| 5 | `q_pinn` | PINN surrogate validation from the shipped checkpoints on real chains | 0.4 s |
| 6 | `q_svi` | Mini SVI reference on a few real days | 1.6 s |
| 7 | `q_stats` | Wilcoxon and Diebold-Mariano on the mini forecast errors | 0.0 s |
| 8 | `q_report` | Quick report (same layout as results/report.md), metrics and figure, all marked demo-scale | 1.0 s |

`ablation_runs`, `comparison_note` and `legacy_figures` are optional: they run only when named in `--stages`. `ablation_runs` regenerates the eleven robustness reruns of the jump model; the frozen-refit run
(`slowrefit`, a fixed bug), the wide-bounds run (`wb`) and the jump-rate-as-state run (`dhjs`) are not regenerated by it. `comparison_note` rewrites `COMPARISON_1.0_vs_2.0.md`.

## What is cached (and where)

| What | Where | Size |
|---|---|---|
| Raw NSE files (one csv per trading day) | `data/nse_index/` | 520 MB |
| Per-day surfaces | `data/cache/surfaces.pkl` | 45 MB |
| Walk-forward fits, forecast variants, trading results, ablation reruns | `data/results/` | 475 MB |
| PINN training data and held-out evaluation sets | `data/cache/pinn/` | 97 MB |
| Shipped PINN checkpoints | `results/pinn/checkpoints/` | 16 MB |
| Resumable caches of the C2 experiments and the PINN-accelerated calibration | `results/_cache/`, `results/pinn_calib/_cache/` | 6 MB |
| India VIX | `data/vix/india_vix.csv` | small |

The C2 scripts (`exp_*.py`) resume from their caches; `--force` on those stages reruns the script (and rewrites the README it generates), it does not delete the cache. They fix 3 worker processes in
`c2_common.py` (`execution.workers_c2`); they were not edited.

## Measured in this session

* `--quick`: **5.3 minutes wall** (315 s; 499 s of CPU on 2 worker processes; the 15-minute target is met). Stages: walk-forward 5.1 min (heston, dh, dhj on the real window 2025-01-02 to 2025-04-30,
  window 20 days, refit every 5 days, 10 burn-in days, 60/12 optimiser evaluations), forecast variants and table 5 s, pricer cross-check 0.6 s, PINN surrogate on 3 real days
  0.4 s, SVI on 4 days 1.6 s, tests 0.0 s, report 1.0 s. Every output says it is a demo-scale run and not the headline result (banner in the markdown, `#` first line
  in the CSVs, `notice` key in the JSON files, red title on the figure). On the same days the mini forecasts are close to the shipped full-run forecasts (the report shows both).
* The cheap full-pipeline stages together (`--stages tests_fast,benchmark_table,metrics_tables,figures,report --force`): 27 s (tests 17 s, table 4.3 s, metrics and tables 0.3 s,
  figures 5.3 s, report 0.2 s).
* Reproducibility check: `dq`, `noarb`, `vix`, `svi`, `stats`, `trading_tests`, `pinn_calib_report` and `ui_adapter` were rerun from the cached inputs and reproduced the saved outputs byte for byte (the SVI quote-error
  `.csv.gz` differs only in its gzip header timestamp). The walk-forwards, `pricer_validation`, `pinn_validation` (its speed table re-measures timings and would change), the C2 experiments and the PINN
  calibration design and test were not rerun.

## results/metrics.json

1159 numbers (1119 read straight from a CSV column or JSON key, 40 derived by a stated rule such as a sum, a ratio or a mean over seeds), from 36 source files (size and sha256 recorded).
Each record has `value`, `unit`, `source` (file path), `key` (the exact column with its row selector, the JSON key path, or the rule) and `kind`. Groups: data 51, noarb 92, forecast 67, forecast_by_year 90, robustness 6, memory 14, stats 140, pricer 62, pinn 91, pinn_calib 60, attribution 77, crosssection 76, identifiability 26, buckets 13, vix 75, svi 20, trading 196, ui 3.
The collector (`dh2/report.py`) raises `MissingSource` for a missing file, column, key or row, an ambiguous row selector or a non-finite value; there are no defaults. Nothing could not be sourced: every metric is
filled. Numbers that the READMEs quote but that no result file carries (compute minutes, counts printed only in prose) are not in the file. The headline tables (`results/tables/*.csv|md|tex`, LaTeX booktabs), the 33 figures
(`results/figures/`, `INDEX.md`) and `results/report.md` are written from these numbers.

## Limits

* Estimates for stages that were not rerun are recorded times, scaled for 2 workers where the recorded run used more; the C2 timings come from cache file times on a machine shared with other jobs and are upper bounds.
* `pinn_validation` and `pricer_validation` rewrite result files containing timings (`speed.csv`, `timing.json`); rerun them only if new timings are wanted.
* The old research repository's `nifty_comparison_detail.csv` is needed by `attribution` for its replication table (path in `exp_attribution.py`); the cached results do not need it.
* A forced rerun of a C2 stage rewrites that package's README from its own template.
