# Double Heston 2.0 (NIFTY 50)

A recalibration and evaluation of the Double Heston option model on ten years of NIFTY 50 option prices (Oct 2016 to Oct 2026), with a jump component, a "pattern memory" that looks up past days that resembled today,
an independent exact pricer (COS, Lewis, Monte Carlo), a neural-network (PINN) surrogate of the pricer, PINN-accelerated calibration, and an honest evaluation. Everything is scored the same way: calibrate on days up to *t*,
forecast the implied-volatility surface of day *t+1*, measure the error in volatility points (one point = one percentage point of implied volatility). Settings were chosen on 2016-2021 only; 2022-2026 is the test period.

Where to read next: `RESULTS.md` (technical report, every number with its source), `docs/EXPLAINER.md` (plain-English guide for a student, how to read every figure, the hard questions),
`PRESENTATION_NOTES.md` (slide plan), `results/report.md` (auto-generated numbers), `COMPARISON_1.0_vs_2.0.md` (living comparison with the original research project), `DECISIONS.md` (all 12 decisions), `results/VERIFICATION.md` (the independent check, with its resolution at the end), `AUDIT.md` and `PLAN.md`.

## What it found

| Next-day forecast, 2022-2026 test, 1,173 days (vol points, lower is better) | Error |
|---|---|
| Yesterday's smile carried forward | 1.468 |
| Yesterday's smile, time-scaled (the better of the two model-free benchmarks tried; repeats its edge value at strikes yesterday did not quote) | 1.236 |
| Same, with the wings continued linearly (added after the independent verification; one variant only) | 1.078 |
| Double Heston, no jumps, plus yesterday's misfit | 1.377 (worse than the benchmark) |
| Single Heston, plus yesterday's misfit | 1.416 (worse than the benchmark) |
| Jump model, today's state only | 1.201 (a tie: -0.035 [-0.095, +0.028]) |
| **Jump model + yesterday's misfit** | **1.034 (-0.202 [-0.227, -0.181] against the time-scaled smile; -0.045 [-0.061, -0.032] against the linear-wing benchmark; better in all 5 test years and all 6 development years)** |

[results/tables/forecast_scores.csv, results/tables/forecast_by_year.csv, results/benchmark_check/summary.csv]

| Other results | Number | Source |
|---|---|---|
| From the 1.0 recipe to 2.0, next-day error (118 sampled test days; every quote of the clean surface; on 1.0's own metric the rebuilt 1.0 recipe scores 2.27) | 5.03 to 1.10 (same-day 4.84 to 0.99) | results/attribution/waterfall.csv |
| Project pricer against the independent COS pricer, real chains | at most 1.5e-6 vol points on options worth at least Rs 0.05 (16,492 of 18,071 diffusion points; all 18,071 jump points); over all 18,071 diffusion points up to 4.30 vol points because near-worthless options cannot be inverted | results/pricer_validation/errors.csv |
| PINN surrogate against the exact pricer (real held-out chains; 3-seed means, the shipped files are seed 2: 0.052 and 0.064; exact price at least about Rs 1) | data-only 0.049, hybrid 0.077, physics-only 1.307 vol points | results/pinn/tables/ablation_summary.csv, heldout_metrics_all_runs.csv |
| PINN-accelerated calibration | 8.0x faster [7.4, 8.7], same-day error 2.560 against 2.555, objective cost +0.8% | results/pinn_calib/headline.json |
| Hidden-quote test (random 20% of quotes), jump model against diffusion | 0.71 against 2.49 | results/crosssection/summary.csv |
| Link to India VIX (jump model) | correlation of levels 0.989; of daily changes 0.879 (all years), 0.948 (test years) | results/vix/vix_vs_model_stats.csv |
| Same-day fit against SVI (5 parameters per expiry), 118 days | SVI 0.502, jump model 1.155, diffusion 3.069 | results/svi/README.md |

## The honest findings (read these before quoting any number)

- **Jumps are what make the model fit.** Same-day error (pooled, 118 days) falls from 2.91 to 0.77 vol points with jumps [results/buckets/README.md]. Without jumps the model loses to the benchmark.
- **What fixed the old recipe** (the attribution waterfall): the implied-vol-space objective (-1.05), the global multi-start with the kappa ordering (-1.04) and Merton jumps (-1.66); yesterday's misfit adds -0.155.
- **The shared-slow-parameter ("structure/state") design is not shown to help.** Adding it made the diffusion model slightly worse (+0.08, p = 0.003), and independent daily jump fits (1.142) forecast better than the walk-forward jump model (1.254) on the same days. The earlier claim is withdrawn
  [results/attribution/waterfall.csv, DECISIONS.md decision 11]. It is kept only because the finished results were produced with it.
- **DH1 (the 1.0 recipe) is a documented rebuild** of a lost script: close in aggregate (2.53 against 2.37) but not an exact reproduction [results/attribution/README.md].
- **The headline lead depends on the benchmark and concerns strikes yesterday did not quote.** The time-scaled smile repeats its edge value outside the strike range yesterday quoted. The model's lead over it is -0.202, but against a benchmark that continues the wings linearly it is -0.045 [-0.061, -0.032] (ahead on 71% of days). **Inside yesterday's quoted range (93.5% of quotes) the benchmark is slightly better** (+0.012 [+0.009, +0.016]); outside it (6.5%) the model wins (1.100 against 1.511 linear-wing and 2.288 edge-value). The linear-wing benchmark was **added after the independent verification**, only **one variant** was tried, and stronger ones (an SVI continuation) are **untested** [results/benchmark_check/README.md, DECISIONS.md decision 12].
- **The model's own dynamics add nothing.** For the jump model, one-day mean reversion -0.002 [-0.006, +0.001], leverage +0.090, leverage plus misfit +0.102, smile plus model change +0.084 (the last three are worse); the forecasting value is the calibrated state plus yesterday's misfit [results/tables/forecast_variants.md].
- **Near the money the time-scaled smile beats the jump model + misfit** (1.071 against 1.088, significant); the model's advantage is at strikes yesterday did not quote and at short expiries [results/stats/tests_test_period.md].
- **The diffusion Double Heston loses to the time-scaled smile** (and, when extrapolating in maturity, to plain Heston: 2.02 against 1.37). The jump model is the strong one.
- **The PINN:** the plain data-trained network beats the physics-informed one on price accuracy; the hybrid is better on gamma, PDE residual and strike convexity; physics-only does not work as a pricer. It is valid only inside its training box and for the diffusion model only [results/pinn/README.md].
- **PINN-accelerated calibration** is about 8x faster than the four-start exact fit (5.85 s) with the same implied-vol error but about 0.8% higher objective cost (worst day 10.9%); one exact start with the same polish takes 0.40 s but is 0.28 vol points worse, and the global stage leaves out the 9.6% of fit quotes beyond 0.6 years on 92% of days; the surrogate alone is not enough (the exact polish is essential); it is weaker on days whose exact optimum is outside the box [results/pinn_calib/README.md].
- **The individual parameters are not identified** (every sampled day has at least one on a bound): the fitted surfaces are what is validated, not the parameters [results/identifiability/README.md].
- **Trading test:** 41.1 to 45.0 of 100 trades profit in `trading_summary.md` (40.6 to 48.2 across the matched table); the relevant control is the blind same-mix column, not a coin flip, and always selling wins 61 to 65 of 100 [results/VERIFICATION.md L7]; wins are larger than losses, so the average is positive at the signal day's close (all four versions, range above zero with no cost); the edge is **not shown to survive a one-day delay** (none of the matched one-day-later averages differs from zero; the 4,525-trade hedged rows of `trading_summary.md` at 1% cost are significantly negative). It is a backtest on closing prints, not investment advice [results/tables/trading_summary.md].
- **The pricer is accurate** (for options worth at least Rs 0.05) with one documented corner case (1-year options with extreme parameters, up to 1.0e-4 vol points) [results/pricer_validation/README.md].
- The final forecast is insensitive to the model's slow settings: 15 jump-model runs (the headline run plus 14 reruns, one of them the known-buggy frozen-refit run) with different windows, refit schedules, priors and bounds score 1.033 to 1.036 on test, while the pure model ranges from 1.185 to 1.436; this fits decision 11 (yesterday's misfit supplies most of the forecast) and is not an overfitting test [results/tables/robustness_runs.csv]. Pattern memory changes nothing as a prior and
  at most -0.004 (not significant) as a forecast correction [results/tables/robustness_memory.md].

## How it works

1. `dh2/data.py` reads NSE's daily F&O files, keeps out-of-the-money quotes that traded, and builds a surface per day: forwards from put-call parity, an implied financing rate, implied volatilities.
2. `dh2/models.py` prices with the project's own pricer (`pricer/models.py`, **unchanged**); jumps are added by passing a different log characteristic function to it. `dh2/cos.py`, `dh2/lewis.py` and `dh2/mc.py` are three independent pricers used to check it.
3. `dh2/calib.py` fits a day: implied-vol-space residuals (price error divided by vega), a robust loss, bounds, multi-start, and the kappa2 = kappa1 + dk ordering. `dh2/struct.py` fits the slow parameters over a 40-day window (refit every 10 days) and the daily state;
   it is how the finished walk-forward results were produced, **not** shown to be the source of the gain (see above).
4. `run_walk.py` runs the walk-forward; `hybrids.py` builds the forecasts ("carry" and "carry + yesterday's misfit"); `dh2/memory.py` is the pattern memory.
5. `dh2/pinn/` is the surrogate (training, PDE check, validation); `dh2/calib_pinn.py` is the accelerated calibration; `exp_attribution.py`, `exp_crosssection.py`, `exp_identifiability.py`, `exp_buckets.py` and `exp_pinn_calib.py` are the heavy experiments; `dh2/dq.py`, `dh2/noarb.py`, `dh2/svi.py`, `dh2/vixcompare.py`, `dh2/stats.py` produce the data-quality, no-arbitrage, SVI, VIX and test results.
6. `dh2/report.py`, `dh2/report_figures.py` and `dh2/run_all.py` build `results/metrics.json` (every number with its source file and column), the tables, 34 figures and `results/report.md`. `exp_wing_benchmark.py` (benchmark check) and `exp_variants_table.py` (forecast variants) are two small stages added after the independent verification.

## Run it

```
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt     # Python 3.13, pinned versions
python -m dh2.run_all --config configs/nifty.yaml --list     # every stage: cached or missing, estimated cost
python -m dh2.run_all --config configs/nifty.yaml            # one command: everything the cached data allows (stages with outputs are skipped)
python -m dh2.run_all --config configs/nifty.yaml --quick    # demo-scale run (about 5.3 minutes on 2 cores), writes only to results_quick/
python3 -m pytest -q                                         # the fast suite: 172 passed, 2 expected failures, 14 slow ones deselected (pytest.ini); the slow ones: python3 -m pytest -m slow
```

On a bare clone (no data files) five PINN tests are skipped because they need `data/results/walk_dh_main.pkl`. Only `final.pt` and `final_greeks.pt` of the PINN checkpoints are in git; a plain `run_all` needs only those two (the other 14 are rebuilt with `--retrain-pinn`, about 48 GPU minutes). Call `run_walk.py` directly with `--chunks 12` (its default is 18; the shipped results used 12, which `run_all` passes).

What is cached (see `results/PIPELINE.md`): the raw NSE files `data/nse_index/` (about 0.5 GB, downloaded only if missing: `dh2.fetch_nse_old`, `dh2.fetch_nse`), the per-day surfaces, the walk-forward fits and forecast variants in `data/results/`, the two shipped PINN checkpoints (`final.pt`, `final_greeks.pt`) in `results/pinn/checkpoints/`,
and the caches of the heavy experiments. A stage whose outputs exist is skipped; the walk-forwards take hours when forced (the whole programme took about 22 hours of computing). Every setting is in `configs/nifty.yaml` and `configs/pinn.yaml`. Heavy jobs run one at a time.

## What is in this repository

- `dh2/` the package: data, models, calibration, walk-forward structure, pattern memory, evaluation; the exact-pricer checks (`cos.py`, `lewis.py`, `mc.py`); `pinn/` (surrogate) and `calib_pinn.py`; `dq.py`, `noarb.py`, `svi.py`, `vixcompare.py`, `stats.py`; `report.py`, `report_figures.py`, `run_all.py`, `quick.py`, `ui_adapter.py`.
- `pricer/models.py` the option pricer, copied unchanged from the earlier Streamlit site of the research project.
- `run_walk.py`, `hybrids.py`, `final_table.py`, `tournament.py`, `trade_*.py`, `exp_*.py`, `c2_*.py` and the scripts beside them drive the studies.
- `results/` all outputs: `metrics.json`, `report.md`, `PIPELINE.md`, `tables/`, `figures/` (34 figures, `INDEX.md`), and one folder per study with its README (`pricer_validation`, `pinn`, `pinn_calib`, `dq`, `vix`, `svi`, `stats`, `attribution`, `crosssection`, `identifiability`, `buckets`, `benchmark_check`), and `VERIFICATION.md`.
- `tests/` the test suite (pricer cross-checks, PINN, calibration, statistics, pipeline).
- `configs/` the settings. `docs/research_1.0/` plain-English reports on the original 1.0 research project. `v1_results/` the few version 1.0 result files that `update_comparison.py` reads.
- `COMPARISON_1.0_vs_2.0.md` the living 1.0-vs-2.0 note; `outputs/` earlier figures and parameter tables; `data/results/*.json` small result summaries; `today.json` the calibrated parameters in the website's format (UI adapter).
- Not stored (too large, rebuilt by the commands above): the raw NSE files `data/nse_index/` (about 0.5 GB), `data/cache/` the walk-forward pickles `data/results/*.pkl` (about 0.5 GB) and the 14 PINN seed / ablation checkpoints.

## Limits

NSE data are end-of-day prints, not quotes; illiquid options can be stale (this flatters persistence in the thin early years). The data are not filtered for no-arbitrage violations (64.1% of quotes take part in one at a tolerance of one tick); the headline difference against the time-scaled smile is robust to excluding them
[results/dq/NOARB.md]. The headline benchmark repeats its edge value outside yesterday's quoted strikes; see the benchmark finding above. Expiry time of day is not modelled (matters for 1-3 day options). The individual parameters are not identified. The shortest expiries are the model's weakest region. The PINN covers the diffusion model only and only inside its box. The expensive experiments use 118 sampled days.
1.0 and 2.0 differ in data, instrument and metric, so rows marked "not identical" in the comparison are not like-for-like. Not investment advice.

## Acknowledgements

- Builds on the Double Heston inverse-calibration research project (version 1.0): forked from Viraj Choudhary's repository
  (github.com/virajchoudhary/double-heston-inverse-calibration), with the neural-network and PINN studies by Dhruva Ambhaikar. 2.0 uses none of their code;
  it compares against and reads some of their published result files (see the comparison note and `docs/research_1.0/`).
- The option pricer `pricer/models.py` comes from the project's earlier Streamlit site (original author to be confirmed here).
- The files in `v1_results/` are outputs of the 1.0 research project (by Dhruva Ambhaikar and the repository owner).
- Data: NSE India daily F&O bhavcopy files; India VIX history from NSE's index-close archive (`data/vix/SOURCE.txt`).
- Implemented with AI assistance (Claude), directed and reviewed by the project owner.
