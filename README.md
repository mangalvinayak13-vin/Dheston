# Double Heston 2.0 (NIFTY 50)

A recalibration of the Double Heston option model on ten years of NIFTY 50 option prices (Oct 2016 to Oct 2026), with a jump component and a
"pattern memory" that looks up past days that resembled today. Everything is scored the same way: calibrate on days up to *t*, forecast the
implied-volatility surface of day *t+1*, measure the error in volatility points. Settings were chosen on 2016-2021 only; 2022-2026 was scored once.

The full, living comparison with version 1.0 (all old tests plus the new ones) is `COMPARISON_1.0_vs_2.0.md`.

## What it found

| Next-day forecast, 2022-2026 test (vol points, lower is better) | Error |
|---|---|
| Yesterday's smile carried forward | 1.468 |
| Yesterday's smile, time-scaled (best model-free benchmark) | 1.236 |
| Double Heston, no jumps, plus yesterday's misfit | 1.377 |
| Single Heston, plus yesterday's misfit | 1.416 |
| Jump model, today's state only | 1.201 (a tie with the benchmark: -0.035 [-0.095, +0.028]) |
| **Jump model + yesterday's misfit** | **1.034 (-0.202 [-0.227, -0.181], better in every year 2016-2026)** |

- **Jumps are what make the model fit.** Same-day error drops from about 1.7 to about 0.6 vol points (2025-26). Without them the model loses to the benchmark.
- **The result does not depend on tuning.** Windows of 20-120 days, refits every 5-20 days, prior strengths, a memory prior, wide bounds and a daily jump rate all give 1.033-1.036 on test.
- **Pattern memory:** as a forecast correction it improves a weak forecast (yesterday's smile: 1.468 to 1.326) and the best one not at all (-0.004, not significant). As a prior for the parameters it changes nothing (three memory runs and a plain-shrinkage control score alike).
- **Individual parameters are not identified** (many sit at bounds); the surfaces they produce are what is validated.
- **Trading test** (`trade_test.py`, `trade_robust.py`, `trade_matched.py`): following the model one NIFTY option at a time, 41-45 of 100 trades profit (a coin flip gives about 50), but wins are larger than losses, so the average is positive at the signal day's close. Whether it survives entering one day later is not established. This is a backtest, not investment advice.

## How it works

1. `dh2/data.py` reads NSE's daily F&O files, keeps out-of-the-money quotes that traded, and builds a surface per day: forwards from put-call parity, an implied financing rate, implied volatilities.
2. `dh2/models.py` prices with the project's own pricer (`pricer/models.py`, **unchanged**); jumps are added by passing a different log characteristic function to it.
3. `dh2/struct.py` splits the parameters into slow structural ones (fitted jointly over a trailing 40-day window, refit every 10 days) and the daily state (two variance levels), so the daily fit has two free numbers instead of ten.
4. `run_walk.py` runs the walk-forward; `hybrids.py` builds the forecasts; `dh2/memory.py` is the pattern memory.

## Run it

```
python3 -m dh2.fetch_nse_old ; python3 -m dh2.fetch_nse           # data (resumable, never overwrites)
python3 run_walk.py --model dhj --tag main --workers 8 --chunks 12 --burn 40
python3 hybrids.py --model dhj --tag main
python3 final_table.py dhj_main dhj_w20 dh_main heston_main        # error table
python3 update_comparison.py ; python3 make_figures.py             # note and figures
python3 -m pytest -q                                              # 12 tests (one skips unless the NSE files are downloaded)
python3 progress.py --serve                                       # live progress page on :8780
```

Heavy jobs: one at a time (the machine had 4 fast and 6 slow cores). The whole programme took about 22 hours of computing.

## What is in this repository

- `dh2/` the package (data, models, calibration, walk-forward structure, pattern memory, evaluation); `run_walk.py`, `hybrids.py` and the scripts beside them drive it.
- `pricer/models.py` the option pricer, copied unchanged from the earlier Streamlit site of the research project.
- `v1_results/` the few version 1.0 result files that `update_comparison.py` reads (`v1_results/outputs/...`); the comparison note cites them by their original `outputs/...` paths.
- `COMPARISON_1.0_vs_2.0.md` the living 1.0-vs-2.0 note; `outputs/` figures and parameter tables; `data/results/*.json` small result summaries.
- Not stored (too large, rebuilt by the commands above): the raw NSE files `data/nse_index/` (about 0.5 GB), `data/cache/` and the walk-forward pickles `data/results/*.pkl` (about 0.5 GB).

## Limits

NSE data are end-of-day prints, not quotes; illiquid options can be stale (this flatters persistence in the thin early years). Near the money, yesterday's smile
is already excellent and 2.0 adds nothing; the gain is in the wings and at the shortest expiries. 1.0 and 2.0 differ in data, instrument and metric, so rows
marked "not identical" in the comparison are not like-for-like. Not investment advice.

## Acknowledgements

- Builds on the Double Heston inverse-calibration research project (version 1.0): forked from Viraj Choudhary's repository
  (github.com/virajchoudhary/double-heston-inverse-calibration), with the neural-network and PINN studies by Dhruva Ambhaikar. 2.0 uses none of their code;
  it compares against and reads some of their published result files (see the comparison note).
- The option pricer `pricer/models.py` comes from the project's earlier Streamlit site (original author to be confirmed here).
- The files in `v1_results/` are outputs of the 1.0 research project (by Dhruva Ambhaikar and the repository owner).
- Data: NSE India daily F&O bhavcopy files.
- Implemented with AI assistance (Claude), directed and reviewed by the project owner.
