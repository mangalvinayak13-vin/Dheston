# VERIFICATION (Phase 8): independent check of the Double Heston 2.0 package

Verifier: an independent reviewer who did not build the package. Date of the check: 2026-10-09. Repository: `Dheston`, branch `dh2-upgrade`, commit `18548b7` (the clone and the working repository were at the same commit).
Nothing in the working repository was edited except this new file. Scratch scripts are outside the repository (`/Users/vinayakmangal/.claude/jobs/5f0f09be/tmp`).

**What "independent" means here.** `results/metrics.json` and `dh2/report.py` were not used as evidence. Every number below was recomputed with my own code from the raw artifacts, or, where a number could not be recomputed, read from the cited CSV/JSON and said so.
What I did trust, and so did not re-derive: the raw NSE files (re-filtered for 2023 only), the cached surfaces `data/cache/surfaces.pkl` (checked against the raw 2023 files), and the stored walk-forward predictions `data/results/hyb_*.pkl` / `walk_*.pkl` (their multi-hour fits were not repeated; I checked on 6 random test days that the stored forecasts equal "day t parameters priced on day t+1's surface", maximum difference 1.5e-8, and I read `run_walk.py` and `hybrids.py` for look-ahead and found none).
Scoring used `dh2/evalpairs.py` (`score`, `pred_last_smile`, `pred_smile_scaled`) as the brief allowed; the bootstrap, the Diebold-Mariano and Wilcoxon tests, and everything else are my own code.

---

## 1. Environment test (clean clone)

**Verdict: PASS for what the README promises on a machine that has the data; FAIL for "one command rebuilds everything" on a bare clone** (documentation gaps, section 1.3).

### 1.1 What was done, with times

| Step | Command (run inside `Dheston_verify`) | Result | Time | Status |
|---|---|---|---|---|
| clone | `git clone -b dh2-upgrade https://github.com/mangalvinayak13-vin/Dheston.git Dheston_verify` | succeeded; HEAD `18548b7`, same as the working repository; 86 MB | seconds | PASS |
| venv | `python3 -m venv .venv` (python.org Python 3.13.14, the version `requirements.txt` names) | created inside the clone | included in the 18 s below | PASS |
| install | `.venv/bin/python -m pip install -r requirements.txt` | exit 0, 38 packages (the 12 pinned ones plus dependencies), no resolver conflict, no error lines. torch 2.14.0, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1 as pinned. `jinja2` is installed as a torch dependency although the requirements header says it is not needed (harmless) | 18 s for venv creation plus install, with a warm pip cache (the log has 66 "Using cached" lines against 10 "Downloading" lines); a cold first install would take longer, so 18 s is not representative | PASS |
| tests, bare clone (before any data was copied) | `.venv/bin/python -m pytest -q -m "not slow"` | 162 passed, 14 deselected, 2 xfailed, **5 errors**: all five in `tests/test_pinn.py`, `FileNotFoundError: data/results/walk_dh_main.pkl` (the PINN domain is built from the walk-forward pickle, which is not in git) | 17.8 s | expected, but not documented (1.3 item 4) |
| data copy | see 1.2 | | seconds (APFS clones) | |
| fast tests | `.venv/bin/python -m pytest -q -m "not slow"` | **167 passed, 14 deselected, 2 xfailed**, 0 errors. README/PIPELINE say about 170 fast tests, 14 slow deselected, 2 expected failures; PIPELINE measured 16.8 s | 16.7 s | PASS |
| list | `.venv/bin/python -m dh2.run_all --config configs/nifty.yaml --list` | 34 stages printed with state, estimated cost, outputs, dependencies. All stages "cached" except `pinn_checkpoints` **MISSING 14/16**; "estimated time of the stages that would run now: 48 min" | 0.4 s | PASS, with the discrepancy in 1.3 item 1 |
| default plan | `... run_all --config configs/nifty.yaml --dry-run` and `... --stages pinn_checkpoints --yes` | the default plan would run `pinn_checkpoints` (EXPENSIVE, 48 min). Running that stage alone fails after 0.0 s with a clear message (14 checkpoints missing, pass `--retrain-pinn`). Nothing was retrained | 0.0 s | see 1.3 item 1 |
| quick | `... run_all --config configs/nifty.yaml --quick --yes` (2 workers) | exit 0, 8 stages ran, 0 failed. **315.5 s wall = 5.26 min** (README/PIPELINE: 5.3 min). Writes only `results_quick/` (gitignored). Stage times: q_walk 305.6 s, q_forecast 5.1 s, q_pricer 0.6 s, q_pinn 0.6 s (PIPELINE says 0.4), q_svi 1.6 s, q_stats 0.0 s, q_report 1.7 s (PIPELINE says 1.0) | 5.26 min | PASS |
| reproduction of tracked outputs (extra) | `... run_all --stages benchmark_table,trading_tests,dq,noarb,vix,stats --force --yes` | benchmark_table 4.4 s, trading_tests 2.3 min (trades rebuilt from the raw files), dq 49.1 s, noarb 2.4 s, vix 2.7 s, stats 0.8 s. `results/tables`, `results/dq`, `results/vix`, `results/stats`, `trade_test_dhj_main.json` and `trade_matched_dhj_main.json` are **byte-identical** to the working repository; `git status` in the clone shows only the tracked `results/run_logs/*` files changed | 3.4 min | PASS |

### 1.2 Exactly what was copied from the working repository (`cp -c -R`, APFS clones, no extra physical disk)

`data/nse_index/` (2,470 csv files, 520 MB); `data/cache/` (`surfaces.pkl` 45 MB and `pinn/` with `box_12000_2001.npz`, `eval_sets.pkl`, `train_1000000_1000.npz`, `val_24000_2000.npz`, 97 MB; 140 MB in all); `data/results/*.pkl` (42 files, 475 MB); `results/_cache/` (5 pickles, 6 MB); `results/pinn_calib/_cache/` (1 MB).
**Not copied:** the 14 seed/ablation PINN checkpoints (`results/pinn/checkpoints/*.pt` other than `final.pt` and `final_greeks.pt`), `logs/`, `data/vix/_nse_*` partial files.
A genuinely from-scratch run would instead download the raw files with `dh2.fetch_nse_old` and `dh2.fetch_nse` (hours) and rebuild the walk-forwards (about 22 hours of computing by the project's own figure). **That was not done.** Not rerun either: the walk-forwards, `pricer_validation` (16 min), `pinn_validation`, the C2 experiments, the PINN calibration, and `metrics_tables`/`figures`/`report` (blocked by the missing checkpoints, 1.3 item 1).
Size of `Dheston_verify`: 2.2 GB as reported by `du` (clone 86 MB, `.venv` 1.0 GB, copied data about 1.1 GB logically, `results_quick` 3.7 MB). I left it in place.

### 1.3 Things that fail, are undocumented, or behave differently from the docs

1. **A fresh clone cannot run the default one-command pipeline.** Only 2 of the 16 PINN checkpoints are in git (`.gitignore` keeps `final.pt` and `final_greeks.pt`); `pinn_checkpoints` declares all 16 as outputs. README ("What is cached", "Not stored") and `results/PIPELINE.md` ("Shipped PINN checkpoints, 16 MB") present the checkpoints as cached and list only the raw files, `data/cache` and the pickles as "not stored". Result: `run_all` with no flags stops at stage `pinn_checkpoints` with `StageError`, and every stage downstream of it (`pinn_validation`, `pinn_calib_*`, `metrics_tables`, `figures`, `report`) is blocked; even `--stages metrics_tables,figures,report` pulls `pinn_checkpoints` in as a missing dependency. The workaround (copy the 14 `.pt` files from the working repository, or pass `--retrain-pinn`, about 48 GPU minutes by the project's own estimate; I did not test whether retrained networks reproduce the shipped tables) is not documented.
2. `python3 -m pytest -q` (README line 67, EXPLAINER Part D) does **not** deselect the slow tests: `pytest.ini` has no `addopts`. The "about 170 fast tests (14 slow ones deselected)" result needs `-m "not slow"`.
3. `configs/nifty.yaml` header still says `--quick` takes "about 10 minutes" (README, PIPELINE and the measured run say 5.3).
4. In a clone without the pickles, 5 tests error (section 1.1). README says the pickles are "not stored" but not that the fast test suite then reports errors.
5. `run_walk.py` defaults to `--chunks 18` whereas the shipped results used 12 (`configs/nifty.yaml`, `queue*.sh`); `run_all` passes 12, so the one-command route is right, but calling `run_walk.py` directly as the docstring shows gives different numbers (results depend on the chunking, as the config comment says).
6. Every `run_all` run rewrites tracked files (`results/run_logs/*`), so the working tree is dirty after any run (cosmetic).
7. `--quick`, mini window: jump model + misfit **1.0706** (documented about 1.071; shipped full-run forecasts on the same days 1.0707); time-scaled smile 1.2517; difference -0.181 [-0.322, -0.088]. Heston + misfit 1.367 (shipped 1.369), diffusion + misfit 1.334 (shipped 1.330). The jump model without misfit differs (mini 1.182 against shipped 1.255) because of the small window/budget; the report shows both. PINN check: on 1 of the 3 demo days (2025-01-30) no quote of the mini calibration lies inside the surrogate box, so no PINN error is produced for that day (the report footnotes this). No other difference.

---

## 2. Claim table

Status words: **MATCH** (recomputed value equals the claim within rounding or within the stated bootstrap noise), **MISMATCH** (the claim is false or mis-scoped as written; corrected value given), **NOT CHECKABLE** (no artifact in the repository allows a check, or the check would be a re-run I did not do). "read" means I read the cited CSV/JSON myself and did not recompute it from deeper artifacts.
Method abbreviations: **F** = my script on `surfaces.pkl` and `hyb_*.pkl`, daily RMSE from `evalpairs.score`; **MBB** = my moving-block bootstrap (overlapping 20-day blocks, ceil(n/20) blocks, 2,000 resamples, seed 12345; other seeds change endpoints by under 0.003); **PB** = the project's own `tournament.boot` (reproduces every printed interval exactly, so differences from MBB are bootstrap-method noise).

### A. Next-day forecast, test period (item 1)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| A1 | test days | README, RESULTS 2 and 6.1, PRES s1/s2, EXPLAINER A1 | 1,173 (first 2022-01-03) | 1,173; first calibration day 2022-01-03 (target 2022-01-04) | F: count of t with surface day on or after 2022-01-01 | MATCH |
| A2 | development days | same | 1,256 | 1,256 | F | MATCH |
| A3 | scored pairs / calibrated days | RESULTS 2, EXPLAINER A1 | 2,429 / 2,430 | 2,429 (hyb keys 40..2468) / 2,430 (walk records) | F | MATCH |
| A4 | yesterday's smile test / dev mean | README, 6.1 | 1.468 / 1.876 | 1.4675 / 1.8755 | F (benchmark recomputed with `pred_last_smile`; stored copy differs by at most 1.5e-8) | MATCH |
| A5 | time-scaled smile test / dev | same | 1.236 / 1.624 | 1.2356 / 1.6239 | F | MATCH |
| A6 | Heston test / dev | 6.1 | 3.532 / 4.028 | 3.5315 / 4.0280 | F | MATCH |
| A7 | Heston + misfit | 6.1 | 1.416 / 1.776 | 1.4163 / 1.7757 | F | MATCH |
| A8 | diffusion DH | 6.1 | 2.792 / 3.409 | 2.7925 / 3.4087 | F | MATCH |
| A9 | diffusion DH + misfit | 6.1 | 1.377 / 1.736 | 1.3766 / 1.7363 | F | MATCH |
| A10 | jump model | 6.1 | 1.201 / 1.734 | 1.2009 / 1.7338 | F | MATCH |
| A11 | jump model + misfit | README, 1, 6.1, PRES s1/s6, EXPLAINER | 1.034 / 1.317 | 1.0336 / 1.3167 | F | MATCH |
| A12 | paired difference, best model vs time-scaled smile | README, 1, 6.1 | -0.202 [-0.227, -0.181] | -0.2021 [-0.226, -0.181] (seeds 1, 2: [-0.224, -0.181], [-0.226, -0.180]; non-overlapping blocks [-0.226, -0.179]); PB gives [-0.227, -0.181] | F, MBB, PB | MATCH |
| A13 | jump alone vs benchmark | README, 6.1, PRES s6 | -0.035 [-0.095, +0.028] | -0.0348 [-0.091, +0.030] (PB: [-0.0948, +0.0283]) | F, MBB, PB | MATCH |
| A14 | diffusion + misfit / Heston + misfit vs benchmark | 6.1 | +0.141 [+0.119, +0.164] / +0.181 [+0.160, +0.203] | +0.1409 [+0.118, +0.164] / +0.1807 [+0.160, +0.203] | F, MBB | MATCH |
| A15 | diffusion alone / Heston alone vs benchmark | 6.1 | +1.557 [+1.419, +1.694] / +2.296 [+2.127, +2.480] | +1.5568 [+1.427, +1.692] / +2.2959 [+2.141, +2.479] | F, MBB | MATCH |
| A16 | 95% intervals of the eight test means | 6.1 | e.g. smile [1.335, 1.614], jump + misfit [0.915, 1.171] | MBB: [1.345, 1.614], [0.920, 1.170]; largest endpoint gap 0.026 (Heston alone, lower end 3.306 against 3.332); PB reproduces all printed endpoints exactly | MBB, PB | MATCH (bootstrap-method noise: the project draws floor(n/20) = 58 blocks, MBB draws 59) |
| A17 | calendar years in which jump + misfit beats the time-scaled smile | README, 1, 6.1, PRES s7 | 11 of 11 | 11 of 11 (by the year of the calibration day and by the year of the target day) | F | MATCH (the count is right; its placement in the test-period table is row S1, see also L3) |
| A18 | by-year table, 22 numbers | 6.1 | 0.802 ... 1.163; 0.707 ... 1.050 | 22 of 22 equal to 3 decimals | F | MATCH |
| A19 | jump alone is better than the benchmark in some years, worse in others | EXPLAINER fig 02 | mixed | better in 5 (2017, 2021, 2022, 2023, 2025), worse in 6 | F | MATCH |
| A20 | share of days jump + misfit beats the benchmark | 6.2, PRES s6 | 90% | 89.6% | F | MATCH |
| A21 | six paired rows: A-B, share of days A better, Diebold-Mariano | 6.2 | -0.202/90%/-19.03; -0.434/78%/-25.59; -0.343/77%/-21.96; -0.383/78%/-24.86; +0.141/41%/+14.40; +0.181/34%/+18.87 | all 18 equal; the other four rows of `tests_test_period.md` also equal (-0.040/77%/-20.27; -0.091/79%/-32.79; -0.051/78%/-22.78; -0.232/70%/-22.34) | own Diebold-Mariano (Bartlett, floor(4(T/100)^(2/9)) = 6 lags, Harvey-Leybourne-Newbold factor) | MATCH |
| A22 | p < 1e-10 for both tests | 6.2, PRES s6 | p < 1e-10 | Wilcoxon p between 1e-23 and 1e-158 (jump + misfit vs time-scaled: 1.9e-158) | scipy `wilcoxon` | MATCH |
| A23 | near the money (abs ln(K/F) <= 0.04): jump + misfit vs time-scaled | README, 1(e), 6.2, PRES s8, EXPLAINER Q13 | 1.088 vs 1.071, +0.017, 47% of days, Wilcoxon (Holm) 2.85e-07, DM +9.41 | 1.0877 vs 1.0708, +0.0169, 46.5%, raw Wilcoxon 1.43e-07 (x 2 Holm = 2.85e-07), DM +9.41 | F with `score()['core']` | MATCH |
| A24 | test years are calmer than development years for every method | 2 | true | all 8 series lower on test | F | MATCH |
| A25 | fifteen reruns: final forecast 1.033 to 1.036; pure model 1.185 to 1.436 | README, 6.1, PRES s7, EXPLAINER Q3 | 1.033-1.036; 1.185-1.436 | `robustness_runs.csv`: 1.0331-1.0355; 1.1850-1.4365. The 15 runs are 14 reruns plus the headline run `dhj_main`, and include the known-buggy `slowrefit` run | read | MATCH (the count of runs is row S2; see also L4) |
| A26 | pattern memory: -0.004 [-0.012, +0.003]; on the benchmark -0.029 [-0.043, -0.018] | README, 6.1 | as written | `data/results/memory_*.json`: -0.0043 [-0.0118, +0.0030]; -0.0289 [-0.0427, -0.0178] | read | MATCH (not recomputed; needs the memory pipeline) |
| A27 | no-arbitrage shares: 64.1% (14.81% / 62.68% / 1.64%), 12.0% with slack | README, 2, 13, PRES B13 | as written | quote-weighted pooling of `noarb_by_year.csv`: 64.08 / 14.81 / 62.68 / 1.64; 12.00 | read, pooled by me | MATCH |
| A28 | advantage stays between -0.166 and -0.224 when violators are excluded | 2, PRES B13 | -0.166 .. -0.224 | `noarb_sensitivity.csv`: min -0.166, max -0.224 (slack variants -0.202 .. -0.215) | read | MATCH (not recomputed; the `dq` and `noarb` stages reproduce byte for byte in the clean clone) |
| A29 | the model's advantage is in the wings and at short expiries | README, 1(e), 6.2, PRES s8, EXPLAINER Q13 | qualitative | pooled test quotes, jump + misfit minus time-scaled RMSE: <=7 days -1.13; 8-14 days -0.15; 15-30 days -0.05; 31-120 days -0.01; >120 days -0.05; deep wings (<-12% / >+12%) -1.24 / -1.86; abs k < 2% +0.008 | own bucket script on the same predictions | MATCH, but no cited file contains this; see L1 for the sharper statement |
| A30 | walk-forward uses only days up to t | EXPLAINER A10, Q5; README | causal | stored carry and carry + misfit forecasts equal "walk record of day t priced on day t+1" on 6 random test days (max difference 1.5e-8); `run_walk.py`: window = t-39..t, refit every 10 days, states fitted on day t only; `data.py` rate = trailing 10-day median | recomputation with `dh2.models`/`calib`/`struct` plus code reading | MATCH |

### B. Waterfall (item 2): 118 sampled test days, `results/attribution/per_day.csv`

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| B1 | sample | RESULTS 5, EXPLAINER | 118 days (2022-01-03 .. 2026-09-30), 53,046 quotes | 118; same dates; 53,046 | column counts | MATCH |
| B2 | same-day means, steps 0-6 | 5 | 4.839, 4.705, 3.589, 2.555, 2.683, 0.994, 0.994 | 4.8391, 4.7047, 3.5892, 2.5547, 2.6832, 0.9943, 0.9943 | column means | MATCH |
| B3 | next-day means, steps 0-6 | 5, README | 5.027, 4.929, 3.881, 2.838, 2.917, 1.254, 1.100 | 5.0272, 4.9292, 3.8813, 2.8381, 2.9166, 1.2544, 1.0997 | column means | MATCH |
| B4 | next-day step differences | 5, README, PRES s9 | -0.098, -1.048, -1.043, +0.078, -1.662, -0.155 | -0.0980, -1.0479, -1.0432, +0.0785, -1.6622, -0.1548 | paired differences | MATCH |
| B5 | Wilcoxon p of the steps | 5 | 0.26, 4e-21, 7e-21, 0.003, 8e-21, 4e-08 | 0.265, 4.3e-21, 7.4e-21, 0.0026, 8.2e-21, 3.9e-08 | scipy | MATCH |
| B6 | same-day step differences (step 4 is +0.13) | EXPLAINER fig 04 | +0.13 at step 4 | -0.134, -1.116, -1.035, +0.1285, -1.689, 0 | paired differences | MATCH |
| B7 | next-day intervals of steps 0, 5, 6 | 5 | [4.680, 5.393], [1.106, 1.415], [0.965, 1.246] | [4.685, 5.384], [1.107, 1.414], [0.967, 1.244] | day bootstrap, 5,000 resamples | MATCH |
| B8 | reference rows | 5 | 1.585, 1.333, 5.970, 3.585, 1.142 | 1.5850, 1.3326, 5.9702, 3.5854, 1.1417 | column means | MATCH |
| B9 | 1.0 recipe to 2.0 | README, 1, PRES s9 | 5.03 to 1.10 (same-day 4.84 to 0.99) | 5.027 to 1.0997 (4.839 to 0.994) | column means | MATCH (scope, L6) |
| B10 | DH1 on 1.0's own metric on the 118 days | 5 | 2.27 | 2.273 (`dh1_holdout_iv_rmse_pts` mean; median 2.285) | column mean | MATCH |
| B11 | DH1 rebuild vs 1.0 | README, 1(b), 5, PRES s9 | median held-out 2.53 vs 2.37 (+7%), correlation 0.74, held-out 170-193 vs 133-142, sigma 10% higher, quote and expiry counts equal on all ten dates | 2.527 vs 2.366 (+6.8%), 0.744, 170-193 vs 133-142, ratio 1.099, counts equal on 10 of 10 | `replication_check.csv` medians | MATCH |
| B12 | the model reaches the benchmarks only after the jump step | EXPLAINER fig 04 | after step 5 | step 4: 2.917 > 1.333; step 5: 1.254 < 1.333 | per_day | MATCH |

### C. Cross-section (item 3): `results/crosssection/per_day.csv`, 118 days

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| C1 | random 20% hidden, BS / Heston / DH / jump | README, 6.4 | 5.787 [5.469, 6.123], 3.245 [2.980, 3.523], 2.486 [2.242, 2.743], 0.705 [0.639, 0.780] | 5.787 [5.467, 6.128], 3.245 [2.984, 3.528], 2.486 [2.242, 2.754], 0.705 [0.638, 0.777] | column means, day bootstrap | MATCH |
| C2 | maturity extrapolation | 6.4 | 2.674, 1.368, 2.022, 1.410 (intervals as printed) | 2.674 [2.526, 2.834], 1.368 [1.213, 1.575], 2.022 [1.803, 2.268], 1.410 [1.239, 1.622] | same | MATCH |
| C3 | held-out error within about 0.1 of training-quote error; jump 0.683 train | 6.4, EXPLAINER Q3 | within 0.1; 0.683 | gaps -0.058, -0.094, -0.071, +0.023; jump train 0.683 | column means | MATCH |
| C4 | DH worse than Heston in the maturity test | README, 6.4, PRES s8 | +0.65, better on 33 of 118 days, p 4.1e-08 | +0.653 [0.442, 0.872], 33, 4.07e-08 | paired | MATCH |
| C5 | jump vs Heston, maturity test | 6.4 | +0.04 [-0.08, +0.16], p 0.73 | +0.042 [-0.079, +0.163], p 0.728 | paired | MATCH |

### D. PINN-accelerated calibration (item 4): `results/pinn_calib/per_day.csv`

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| D1 | same-day RMSE exact / accelerated / surrogate-only / control | README, 1, 9 | 2.555, 2.560, 2.554, 2.837 | 2.5547, 2.5601, 2.5541, 2.8370 (intervals e.g. [2.311, 2.805] against [2.312, 2.812]) | column means | MATCH |
| D2 | paired RMSE difference | 9 | +0.0054 [-0.0091, +0.0205], p 0.17 | +0.0054 [-0.0092, +0.0211], p 0.174 | paired | MATCH |
| D3 | mean log cost ratio | 9 | +0.008 (0.8%), +0.166 (18%), control +0.085 | 0.0079 (0.79%), 0.1664 (18.1%), 0.0851 | cost columns | MATCH |
| D4 | days within 1% of exact cost | 9, EXPLAINER Q9 | 100%, 82%, 7%, 25% | 100%, 82.2%, 6.8%, 25.4% | ratio <= 1.01 | MATCH |
| D5 | wall-time speedup | README, 1, PRES s13 | 8.0x [7.4, 8.7] | 8.025 [7.43, 8.66] | sum(exact s)/sum(accel s), day bootstrap | MATCH |
| D6 | exact-pricer call ratio and per-day calls | 9 | 11.6x [10.7, 12.6]; 1,380 against 119 | 11.64 [10.72, 12.62]; 1,379.8 against 118.6 | calls columns | MATCH |
| D7 | seconds per day | 9, PRES s13 | 5.85 s against 0.73 s | 5.848 / 0.729 | column means | MATCH |
| D8 | faster on every day; smallest and largest speedup | 9, EXPLAINER fig 30 | 118 of 118; 4.0x; 23.5x | 118; 4.04; 23.48 | per-day ratio | MATCH |
| D9 | worst-day cost | README, 9, PRES s13 | +10.9% (2024-02-09) | ratio 1.1090 on 2024-02-09 | cost ratio | MATCH |
| D10 | accelerated cost higher: Wilcoxon p; median | 9 | 7e-18; 0.08% | 6.6e-18; +0.075% | scipy | MATCH |
| D11 | polish budget exhausted | 9, 13 | 115 of 118 | 115 | column | MATCH |
| D12 | exact optimum outside the box; accelerated cost >1% above exact | 9, PRES s13 | 19 of 118 (16%); 47% outside, 12% inside | 19 (16.1%); 47.4%, 12.1% | columns | MATCH |
| D13 | worst RMSE day | 9, EXPLAINER fig 32 | 2023-03-17, +0.52, outside the box | 2023-03-17, +0.520, outside | columns | MATCH |
| D14 | same parameter vector | 9 | 37% (accelerated), 2.5% (surrogate-only) | 37.3%, 2.5% (gap <= 5% of range) | column | MATCH |
| D15 | cost within 1% yet parameters differ by >5% of range; surface gap | 7, 9, PRES s10, EXPLAINER A9 | 51 of 118; median 0.04, 90th percentile 0.14 | 51 (with abs(ratio-1) <= 1%; 53 if only "not worse than 1%"); 0.040; 0.141 | columns | MATCH |
| D16 | time split of the accelerated fit | EXPLAINER fig 30 | 0.12 / 0.18 / 0.43 s | 0.121 / 0.180 / 0.428 | columns | MATCH |
| D17 | control (one exact start, same polish) is 0.28 RMSE worse | EXPLAINER Q9, 9 | +0.28 | +0.277 | paired | MATCH |
| D18 | share of fit quotes beyond tau 0.6 left out of the global stage | `pinn_calib/README.md` (not in RESULTS) | 9.6% of fit quotes on 92% of days | `n_quotes_outside_surrogate_box` / quotes = 2.4% of all quotes (the 9.6% is of the thinned fit set) | read | NOT CHECKABLE (thinned fit sets are not stored per day); see L8 |

### E. Pricer validation (item 5)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| E1 | real chains, diffusion, project vs COS: the maximum | README, 1(3), 4, PRES s3 | n 18,071; max 1.45e-06 vol pts | max 1.4527e-06 (rows with price >= Rs 0.05: n 16,492 of 18,071) | max over `errors.csv` joint-bucket rows with domain "all" | MATCH (value as defined; its scope is row S7) |
| E2 | real chains, jumps | 4 | 18,071; 6.03e-07 | 6.029e-07 (all 18,071 priced >= Rs 0.05) | same | MATCH |
| E3 | synthetic grid, diffusion / jumps | 4 | 8,736; 1.04e-04 / 1.01e-04 | 1.0427e-04 (n priced >= Rs 0.05: 5,717) / 1.0098e-04 | same | MATCH |
| E4 | Lewis vs COS | 4 | 7.57e-10 / 1.51e-09 (diffusion); 1.3e-9 relative | 7.566e-10 / 1.512e-09; max relative price error 1.259e-9 | same | MATCH |
| E5 | Monte Carlo vs COS | 4, PRES s3 | 70 points; max abs z 1.98; mean abs z 0.60 | 70; 1.977; 0.596 | same | MATCH |
| E6 | one corner case | README, 4, PRES s3 | up to Rs 0.0076, 1.0e-4 vol pts; 40 worst points are two 2020 parameter sets at 365 days | max Rs 0.0076073, 1.043e-4; 40 rows: sets 2020-03-18 (33) and 2020-04-21 (7), all 365 d | `worst_cases.csv` | MATCH |
| E7 | speed of one chain (659 quotes, 9 expiries) | 4 | 26 / 156 / 52 / about 6,000 microseconds per option | 25.65 / 156.3 / 51.8 / 6,001 (diffusion); jumps 29.4 / 230 / 73 / 8,807 | `timing.json` | MATCH (read) |
| E8 | fresh spot check, 3 options priced three ways | (new) | agreement | 2024-03-05 call K 22500, 30 d: project 337.51943067, COS 337.51943067, own quadrature 337.51942037; 2022-06-20 put K 14300, 31 d: 98.25288657 / 98.25288657 / 98.25288657; 2026-03-12 call K 25350, 47 d: 143.62585734 / 143.62585734 / 143.62585733. Project minus COS at most 1.6e-10 Rs; own minus COS at most 1.0e-5 Rs (my quadrature stops at u = 400) | calibrated walk_dh_main parameters of those days, `pricer/models.py`, `dh2/cos.py`, and my own characteristic function plus `scipy.quad` Gil-Pelaez | MATCH |
| E9 | characteristic function agrees with a 30-digit mpmath solution at 540 points to 5e-14 | 4 | 5e-14 | the scan script (`cf_scan_mpmath.py`, about 6 min) was not re-run; no output file stores the result | n/a | NOT CHECKABLE |

### F. PINN (item 6)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| F1 | held-out IV error, real chains: data-only / hybrid / gate sqrt(tau) / physics-only (mean, median, 95th, max, box mean, PDE residual) | README, 1(4), 8, PRES s12 | 0.049, 0.039, 0.130, 0.444, 0.082, 0.733 / 0.077, 0.064, 0.196, 0.651, 0.141, 0.362 / 0.172, 0.129, 0.478, 2.163, 0.324, 1.801 / 1.307, 0.969, 3.207, 23.786, 4.511, 0.260 | equal to `ablation_summary.csv`; the 3-seed rows equal the average of the per-seed rows of `heldout_metrics_all_runs.csv` (data-only 0.0447, 0.0517, 0.0519 give 0.0494; hybrid 0.0828, 0.0852, 0.0642 give 0.0774) | read and averaged | MATCH |
| F2 | held-out set | 8 | 65,465 quotes of 251 days (2020-12-28 .. 2021-12-31) | `quote_counts.csv`: 74,811 quotes, 1,741 beyond 0.6 y, 7,605 below the price floor, 65,465 scored; 251 days | read | MATCH |
| F3 | **fresh re-measurement of the shipped `final.pt`** | task item 6 | 0.049 mean, 0.039 median, 0.130 95th, 0.444 max (3-seed averages) | 8 real held-out days (2021-02-15, 03-24, 04-12, 06-15, 08-17, 10-14, 11-15, 12-15), 2,307 quotes inside the box. Exact model price >= Rs 1 (1,956 quotes): mean **0.0485**, median 0.038, 95th 0.128, max 0.237. Rs 0.01 to 1 (266): mean 0.104, 95th 0.277, max 0.461. **Below Rs 0.01 (85, reported separately, ill-conditioned): mean 4.67, median 2.68, 95th 12.6, max 14.15.** `final.pt` itself is seed 2 (project-measured 0.0519) | `Surrogate.load(final.pt)`, parameters of `walk_dh_main.pkl`, exact price from `pricer/models.py`, Black IV of both prices by my own bisection | MATCH |
| F4 | architecture and training | 8 | width 256, depth 5, SiLU, 267,521 weights, 30,000 steps, batch 2,048, 1,000,000 labels (125,000 x 8) | checkpoint `info`: params 267,521; net_cfg width 256, depth 5, silu; train_cfg batch 2,048, 30,000 steps; `configs/pinn.yaml` n_train 1e6, 8 strikes | read | MATCH |
| F5 | speed ratios | 8, PRES s13, EXPLAINER A12 | 10.0x, 21x (CPU), 79x (MPS), 5.5x (Greeks); 38 microseconds per quote | 9.98x, 20.7x, 79.4x, 5.48x; 38.4 | ratios of `speed.csv` medians | MATCH |
| F6 | data efficiency | 8, PRES B5 | 0.38 and 0.40 at 1,000; 0.049 and 0.077 at 1,000,000 | 0.3802 and 0.3993; 0.0494 and 0.0774 | read | MATCH |
| F7 | Greeks | 8, EXPLAINER fig 25 | delta median 0.0008-0.0012 (data-only), 0.0011-0.0015 (hybrid); gamma 1.8-2.4% (hybrid), 2.5-6.0% (data-only) | 0.00076-0.00121; 0.00105-0.00146; 1.80-2.44%; 2.47-6.04% | `greeks_errors.csv`, three seeds each | MATCH |
| F8 | butterfly-convexity violations at tolerance 1e-6 | 8, EXPLAINER Q1 | hybrid 0.027 / 0.008 / 0.009%, data-only 0.24 / 1.04 / 0.24%, exact 0 | 0.0269 / 0.0081 / 0.0094%; 0.246 / 1.04 / 0.237%; 0 | `arbitrage_checks.csv` | MATCH |
| F9 | PDE verification: residual medians and the sign-flip check | 8, EXPLAINER A11 | median residual 5.7e-5, 1e-6, below 1e-6; sign flipped 0.199 | project pricer 5.74e-5, 9.95e-7, 1.7e-8; COS 5.74e-5, 1.34e-6, 4.4e-8; flipped 0.1992 | `pde_verification_summary.csv` | MATCH (the "64 times per halving" rate is row S6) |
| F10 | gate change | 8, EXPLAINER A11, DECISIONS 8 | 0.053 against 0.242 (monitoring set); "3.5x worse" | spec gate 0.2421 (`mix` set); data-only validation mean at the end of training 0.0530 (seed 0; seeds 1, 2: 0.0651, 0.0524); on real chains 0.1723 / 0.0494 = 3.5x | read (`train_dat_s*.csv`, `ablation_summary.csv`) | MATCH |
| F11 | quotes beyond 0.6 y; test days outside the box | DECISIONS 9, 8 | 2.3%; 22 of 1,174 days (1.9%), rho2 | 1,741 / 74,811 = 2.33%; 22 (1.87%) | read | MATCH |
| F12 | figure 26 text (2021-07-01; data-only error smaller at 7, 28, 182 days; hybrid +0.43 at the edge of the 182-day smile) | EXPLAINER fig 26 | as written | with the seed-0 networks the figure script uses: data-only mean abs 0.037 / 0.034 / 0.061, hybrid 0.055 / 0.115 / 0.172; hybrid maximum +0.43 at 182 d | re-ran the figure's comparison | MATCH |

### G. India VIX (item 7): `data/vix/india_vix.csv`, `walk_dh_main.pkl`, `walk_dhj_main.pkl`; formula coded by me from RESULTS 10 (sigma30^2 = sum over factors of theta + (v - theta)(1 - exp(-kappa T))/(kappa T), T = 30/365, plus lam (mu_j^2 + delta_j^2) for jumps)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| G1 | jump model: correlation of levels / daily changes, days | README, 1(7), 10, PRES s11 | 0.989 / 0.879 over 2,430 days; test 0.984 / 0.948 over 1,174 | 0.9887 / 0.8788 over 2,430; 0.9835 / 0.9480 over 1,174 | own formula, pandas | MATCH |
| G2 | diffusion model; mean gap; RMSE of gap | 10 | 0.989 / 0.874 / 0.949 test; gap 0.44 / 0.63 (jump), 0.00 / 0.13 (diffusion); RMSE 1.07 / 0.93, 0.97 / 0.72 | 0.9888 / 0.8738 / 0.9486; +0.437 / +0.628, -0.001 / +0.128; 1.067 / 0.929, 0.970 / 0.716 | same | MATCH |
| G3 | regression of VIX on the jump model | 10, EXPLAINER fig 13/14 | levels slope 1.01, R2 0.98; changes slope 0.86, R2 0.77 | 1.010, 0.9776; 0.855, 0.7723 | OLS | MATCH |
| G4 | lead-lag | 10, PRES s11, EXPLAINER fig 15 | peak at lag 0 (about 0.88), at most about 0.11 elsewhere | 0.879 at lag 0; largest other 0.104 | shifted correlations | MATCH |
| G5 | 16 days with a variance level on the upper bound; jump 52.7-68.6%; VIX 51.5-83.6; VIX peak about 84 | 10, PRES s11 | as written | union of days with a state at 0.30 in the diffusion (14 days) and jump (15 days) walk-forwards = 16 (2020-03-13 .. 2020-04-08); jump 52.7-68.6; VIX 51.5-83.6, peak 83.61 on 2020-03-24 | state arrays | MATCH |
| G6 | surface at-the-money 30-day IV row (2,415 days, 0.985, 0.831, -1.24, 1.63) | 10 | as written | not recomputed (needs `dh2.vixcompare.atm30`) | read `vix_vs_model_stats.csv` | MATCH (read) |
| G7 | rolling 120-day correlation of changes (EXPLAINER fig 16) | EXPLAINER fig 16 | jump mostly above 0.8, dips in 2019 and 2020; ATM about 0.47 in late 2023 | jump: 93% of days above 0.8, minima 0.68 (2019-05-14) and 0.69 (2020); ATM minimum 0.46 on 2023-09-26 | rolling pandas | MATCH |

### H. Data funnel (item 8): all 245 raw files of 2023 re-filtered by my own code (traded volume > 0; expiry at least 3 days; put-call-parity forward from the six strikes nearest the money with both sides traded; out of the money; price >= Rs 1; abs ln(K/F) <= 0.22; implied vol by my own bisection; the day's causal rate taken from the cached surface)

| # | Claim | Where | Claimed (`data_quality_by_year.csv`, 2023) | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| H1 | NIFTY option rows / traded | RESULTS 2, README | 339,069 / 177,522 | 339,069 / 177,522 | own filters on raw csv | MATCH |
| H2 | expiry >= 3 d / forward exists | 2 | 156,113 / 153,034 | 156,113 / 153,034 | same | MATCH |
| H3 | out of the money / price >= Rs 1 | 2 | 95,381 / 93,421 | 95,381 / 93,421 | same | MATCH |
| H4 | within 22% / IV solves = final; usable days | 2 | 90,902 / 90,902; 245 of 245 | 90,902 / 90,902; 245; equals the quote count in `surfaces.pkl` for 2023 | same | MATCH |
| H5 | all-year funnel and shares | RESULTS 2, README, PRES s2, EXPLAINER A1 | 5,254,005; 1,517,128; 1,342,577; 1,299,272; 794,565; 766,943; 736,313; shares 28.9 / 25.6 / 24.7 / 15.1 / 14.6 / 14.0%; 48.5% of traded | the ALL row of the CSV; yearly rows sum to 5,254,005 / 1,517,128 / 736,313; shares recomputed | read and summed (other 9 years not re-filtered; the `dq` stage reproduced byte-identically in the clean clone) | MATCH |
| H6 | quotes per day | 2, EXPLAINER A1 | 298 (median 311), 6.9 expiries (2 to 11), 94.5 (2016), 550.9 (2026) | 298.10, 311, 6.894, 2..11, 94.51, 550.95 | read | MATCH |

### I. Identifiability and SVI (item 9)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| I1 | median condition number (columns normalised); effective rank; largest | 7, EXPLAINER A9 | 76 / 529; 9 / 11; 1,338 / 6,681 | 76.15 / 529.1; 9 / 11; 1,337.8 / 6,680.7 | `per_day.csv` and `_cache/ident_v1.pkl` | MATCH |
| I2 | parameters within 2% of a bound; days with at least one on a bound (0.3%) | README, 1(8), 7, PRES s10 | 41.0% / 38.7%; 100% / 100% | 41.02% / 38.66%; 118 of 118 both | cache | MATCH |
| I3 | median standard error / bound width | 7, EXPLAINER A9/Q2, PRES s10 | 0.213 / 0.124 ("median over parameters and days" in the study README) | diffusion 0.212 pooled, 0.213 as median of daily medians; **jumps 0.153 pooled, 0.124 only as median of daily medians** | cache `se_share_ls` | MISMATCH for the jump column under the stated definition; corrected 0.153 (15%), see M4 |
| I4 | distinct near-optimal sets; days with two or more; starts within 1% | 7 | 1.27 / 1.50; 16.1% / 29.7%; 12.2 | 1.271 / 1.500; 16.1% / 29.7%; 12.17 (jumps 11.39) | `per_day.csv` | MATCH |
| I5 | standard error exceeds the interval on at least a quarter of days (diffusion) | 7, EXPLAINER A9 | kappa1, theta1, rho1, dk | 29%, 30%, 31%, 27% (all others at most 8%) | cache | MATCH |
| I6 | variance levels are tight, standard error about 1.4% | 7, EXPLAINER A9 | 1.4% | diffusion 1.41% / 1.44%; jumps 2.77% / 2.95% | cache | MATCH for diffusion only (L10) |
| I7 | bound hits: xi2 91%, rho1 64% (diffusion) | EXPLAINER fig 19 | 91%, 64% | 90.7%, 63.6% | cache | MATCH |
| I8 | SVI pooled RMSE all / near money / wings (SVI, jump, diffusion) | README, 11 | 0.502, 1.155, 3.069; 0.533, 1.042, 1.107; 0.513, 1.421, 6.235; n 52,980 / 22,168 / 7,608 | same RMSEs; n 52,980 / 22,173 / 7,608 (5 quotes at the edge, abs k = 0.04) | `svi_vs_dh_quote_errors.csv.gz` | MATCH |
| I9 | mean daily RMSE; SVI lowest on nearly every day | EXPLAINER fig 09 | 0.460 / 0.994 / 2.685, peaks above 8 | 0.460 / 0.994 / 2.685; maximum 8.45; SVI lowest on 99.2% of days | same | MATCH |
| I10 | SVI expiries and arbitrage | 11 | 1,031 expiries; 48.8% butterfly anywhere; 3.8% in range | 1,031; 48.8%; 3.8% | `svi_fits.csv` | MATCH |
| I11 | short-expiry gap to SVI | 11, EXPLAINER Q11 | 7 days or less: jump 1.65-3.24 against SVI 0.35-0.54; diffusion 16.76 / 17.68 | 1.65-3.24; 0.35-0.54; 16.76 (k < -0.10), 17.68 (k > +0.10) | same | MATCH |
| I12 | pooled same-day RMSE 6.13 / 3.67 / 2.91 / 0.77; walk-forward 3.07 / 1.15 | README, 6.3, PRES s5 | as written | 6.1306 / 3.6690 / 2.9115 / 0.7740; 3.0675 / 1.1544 | quote-weighted pooling of `bucket_rmse.csv` | MATCH |
| I13 | bucket cells | 6.3, PRES s5, EXPLAINER fig 06/07 | 3-7 d deep put 29.2 / 22.1 / 19.1 / 1.48, walk-forward 3.65, diffusion walk 19.4; beyond 120 d 0.43 to 1.00; 8-14 d deep put 9.2 and 0.96; 3-7 d call wing 12.1 / 10.1; walk-vs-fresh gap 0.04 to 0.35; counts 4,128 (central) and 29 (corner) | all equal (29.23, 22.06, 19.10, 1.477, 3.65, 19.42; 0.434-0.998; 9.16, 0.96; 12.13, 10.10; 0.04-0.35; 4,128; 29) | `bucket_rmse.csv` | MATCH |

### J. Trading test (item 10): `trades_dhj_main.pkl`, `trades_dhj_main_lag1.pkl`, thresholds from `trade_test_dhj_main*.json`; trades recomputed as sign(signal) x (p1 - p0 - hedge) - cost x (p0 + p1); ranges resample whole days (2,000)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| J1 | signal-day-close table, 5 rows (trades, wins per 100, blind same mix, rupees per trade [range]) | 12, PRES s14 | 1,370 / 43.9 / 42.1 / +6.38 [+1.35, +12.95]; 1,370 / 43.5 / 41.4 / +5.78 [+1.83, +10.03]; 4,168 / 44.9 / 43.4 / +2.69 [+0.22, +5.45]; 1,313 / 44.5 / 41.9 / +6.48 [+2.33, +11.21]; 1,313 / 42.0 / 39.0 / +4.34 [+0.19, +8.98] | all equal, ranges to the cent (the other 3 rows at 1% cost: 43.0 / 41.0 / +4.30 [-0.69, +10.85]; 41.1 / 38.5 / +3.69 [-0.24, +7.93]; 43.7 / 42.2 / +0.71 [-1.75, +3.44] also equal) | trade pickles | MATCH |
| J2 | one-day-later rows of `trading_summary.md` (8 rows) | 12 | e.g. 845 / 44.6 / +6.04 [-2.26, +14.34]; 4,525 / 44.5 / +0.02 [-1.25, +1.32]; 1% cost 4,525 / 41.2 / -2.04 [-3.31, -0.76] | all 8 equal | lag-1 pickle | MATCH |
| J3 | matched one-day-later table | 12, PRES B17, EXPLAINER Q8 | trades 845 / 845 / 3,159 / 838; wins 45.8/44.6, 47.8/40.9, 45.8/43.3, 48.2/40.6; rupees 5.84/6.04, 6.54/1.55, 2.59/0.95, 7.23/1.21; lost -0.20 [-10.25, +9.42], +4.99 [-2.14, +11.92], +1.64 [-2.87, +6.07], +6.03 [-0.99, +13.45] | trades and all point values equal; ranges [-10.25, +9.42], [-1.88, +12.08], [-2.54, +6.01], [-1.06, +13.75] (differences up to 0.4: bootstrap seed) | merge of the two pickles on (day, contract) | MATCH |
| J4 | median trade negative in every row | 12 | true | medians -2.1 to -6.5 in all 8 signal-day rows | pickles | MATCH |
| J5 | the model mostly says buy; wins are larger than losses | 12, PRES s14 | true | buy share 74-79%; average win 32-65 against average loss -14.5 to -40; profit factor 1.03-1.81 | `trade_test_dhj_main.json` | MATCH |
| J6 | "41 to 45 of 100 trades profit in every row of `trading_summary.md`" | 1(g), 12, README, EXPLAINER A15/Q8 | 41-45 | the file's rows run 41.1 to 45.0. In the matched table of the same section the one-day-later wins are 40.6 to 44.6 (37.7 to 43.6 at 1% cost) and the signal-close wins 45.8 to 48.2 | pickles | MISMATCH for the general statement; correct: 41.1-45.0 for `trading_summary.md`; 40.6-48.2 over the matched table |
| J7 | "three of the four versions have a positive average whose range stays above zero with no cost" | 12, COMPARISON 123 | three | **all four**: [+1.35, +12.95], [+1.83, +10.03], [+0.22, +5.45], [+2.33, +11.21] | J1 | MISMATCH; corrected "all four" |
| J8 | "none of the one-day-later averages is distinguishable from zero" | 12, COMPARISON 123, PRES s14 | none | true for the matched trades (all 8 of my intervals contain 0, e.g. hedged forecast edge [-3.14, +6.39]); **false for `trading_summary.md`**: the 4,525-trade hedged rows at 1% cost are -2.04 [-3.31, -0.76] and -2.02 [-3.25, -0.70], which exclude zero (negative) | J2, J3 | MISMATCH (scope); corrected: true only for the matched trades |
| J9 | hedged versions turn negative one day later at 1% cost | 12 | negative | matched: -0.67 and -1.04 rupees per trade; their intervals contain 0 | J3 | MATCH |
| J10 | an independent audit found no look-ahead | 12, COMPARISON | no look-ahead | the audit report is not in the repository; I read `trade_test.py`: the signal uses the day-t calibration and day-t prices; entry price p0 is the same close | code reading | NOT CHECKABLE (audit); my own reading found none |

### K. Other numbers in the slides and documents (item 11)

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| K1 | jump rate through time | EXPLAINER fig 08 | mostly 0.05-0.4 a year, above 1 in 2020, lower and steady after 2022 | 92% of days in 0.05-0.4; above 1 on 10 days in 2020 (maximum 1.28 on 2020-05-06); 2022 onward 0.042-0.130 | walk_dhj_main eta[8] | MATCH |
| K2 | SVI example smile (2024-05-28, 30 d) | EXPLAINER fig 11 | walk-forward jump about 30% where the market is 36%; stray quotes near 18% | 30.3% against 36.0% at k = -0.19; market minimum 17.7% | surface + walk record | MATCH |
| K3 | smile-figure days | PRES s4, EXPLAINER fig 05 | percentiles 10 / 50 / 90 (2023-11-28, 2022-02-01, 2022-05-02); 9-day error 2.16 on 2022-02-01 | `smile_figure_days.csv`: ranks 10 / 50 / 90; 2.165 | read | MATCH |
| K4 | cost-ratio figure, parameter figure | EXPLAINER fig 29, 31 | ends about 1.11, surrogate-only up to about 2.2; kappa1, theta1 differ by about 0.4 and 0.2 (log units) | 1.109; 2.222; 0.402 and 0.217 | pinn_calib columns | MATCH |
| K5 | `metrics.json` composition | PIPELINE | 1,159 numbers (1,119 read, 40 derived), 36 sources, 18 groups with stated counts | identical; 33 figures as png and pdf | JSON counts | MATCH |
| K6 | tests | README, PIPELINE | about 170 fast, 14 slow deselected, 2 expected failures | 167 + 2 xfailed, 14 deselected | pytest | MATCH |
| K7 | quick run time and numbers | README, PIPELINE | 5.3 min; jump + misfit about 1.071 | 5.26 min; 1.0706 | section 1 | MATCH |
| K8 | the whole programme took about 22 hours of computing | README | 22 h | no run log adds up to this; the per-stage estimates are "file times" | n/a | NOT CHECKABLE |
| K10 | training-curve figure | EXPLAINER fig 27 | 30,000 steps; data-only runs end at about 0.05 to 0.07, hybrid at about 0.09 to 0.11 | final validation means: data-only 0.0530 / 0.0651 / 0.0524, hybrid 0.1006 / 0.1069 / 0.0935 (step 30,000) | read `train_*.csv` | MATCH |
| K9 | "Settings were chosen on 2016-2021 only" | README, DECISIONS 1, EXPLAINER Q3 | true | no log shows what was looked at when; `dhj_main` is not the minimum on the development years among the 15 runs (carry + misfit: w20 1.3149 and dhjs 1.3160 against main 1.3167), within 0.002 | `robustness_runs.csv` | NOT CHECKABLE (process claim) |

### S. Scope, count and documentation statements checked against the evidence

| # | Claim | Where | Claimed | Recomputed | Method | Status |
|---|---|---|---|---|---|---|
| S1 | "better in all 11 calendar years" in the row of the table headed "2022-2026 test, 1,173 days" | README line 19 | 11 calendar years | the test period has 5 calendar years (2022-2026), all won; the other 6 years are development years (A17) | calendar | MISMATCH (scope); corrected "5 of 5 test years (11 of 11 including development years)" |
| S2 | "fifteen reruns of the jump model" | RESULTS 6.1, README, EXPLAINER Q3, PRES s7/B14 | 15 reruns | `robustness_runs.csv` has 15 jump-model runs = 14 reruns plus the headline run `dhj_main`; one of them (`slowrefit`) is the known-buggy frozen-refit run | count | MISMATCH; corrected "14 reruns plus the headline run" |
| S3 | `python3 -m pytest -q` gives "about 170 fast tests (14 slow ones deselected, 2 expected failures)" | README line 67, EXPLAINER Part D | as written | `pytest.ini` has no `addopts`; the plain command runs the slow tests as well; `-m "not slow"` gives 167 passed, 14 deselected, 2 xfailed | run | MISMATCH; corrected command |
| S4 | PINN checkpoints are cached, 16 MB, "stages with outputs are skipped" | README ("What is cached"), PIPELINE "What is cached" | 16 checkpoints | only `final.pt` and `final_greeks.pt` are tracked (`.gitignore`); `--list` in a clone: `pinn_checkpoints` MISSING 14/16; the default run stops there | clone | MISMATCH; see section 1.3 item 1 |
| S5 | `--quick` takes "about 10 minutes" | `configs/nifty.yaml` header | 10 min | 5.26 min measured; README/PIPELINE say 5.3 | run | MISMATCH (stale comment) |
| S6 | PDE finite-difference residual "falls about 64 times per halving of the step" | RESULTS 8, EXPLAINER A11 | 64x | project pricer 58x and 60x; COS check 43x and 31x | `pde_verification_summary.csv` | MISMATCH (approximate; 64 is the theoretical rate) |
| S7 | "at most 1.5e-6 vol points on 18,071 real-chain points" | README, RESULTS 1(3)/4, PRES s3 | all 18,071 points | the maximum is over the points priced at Rs 0.05 or more: 16,492 of 18,071 (diffusion), 18,071 of 18,071 (jumps); over all 18,071 diffusion points the implied-vol difference reaches 4.30 vol points (price difference at most Rs 4.1e-5; near-worthless options cannot be inverted) | `errors.csv` (`n_tradeable`, `max_abs_iv_err_pts_any_price`) | MISMATCH (scope) |
| S8 | the old text that needs correcting (AUDIT.md section 4, `dh2/struct.py` docstring, COMPARISON) | EXPLAINER Part F | still wrong | all three were corrected on 2026-10-09 (AUDIT: "A large cause"; `struct.py`: decision-11 note; COMPARISON header: "not shown to help") | read | MISMATCH (Part F outdated) |
| S9 | "data-only network 0.049 (shipped for calibration), hybrid 0.077 (shipped for Greeks)" | RESULTS 8 table, README, PRES s12 | the shipped networks | 0.049 and 0.077 are 3-seed means; the shipped files are seed 2: 0.0519 and 0.0642 (project-measured on the same held-out set) | `heldout_metrics_all_runs.csv`, checkpoint `info` | MISMATCH (scope) |
| S10 | DH1 baseline "on the same clean data" | DECISIONS 6 | clean data | the waterfall's step 0 uses 1.0's data; "clean data" is step 1 (RESULTS 5, `waterfall.csv`) | read | MISMATCH (internal inconsistency) |
| S11 | no-arbitrage sensitivity "run on 2025-26" | DECISIONS 4 | 2025-26 | `noarb_sensitivity.csv` covers the full test period 2022-2026 | read | MISMATCH (stale) |

**Counts of the rows of this section 2 (A1 to K9 and S1 to S11): see section 6.**

---

## 3. Numeric-token coverage (the integers and percentages the E-b check skipped)

A script extracted every numeric token from the four documents (dates, file paths and identifiers removed): RESULTS.md 940, README.md 114, PRESENTATION_NOTES.md 237, docs/EXPLAINER.md 828 (2,119 tokens). Each token was looked up, in this order, (1) in the files cited on its own line, then in the same paragraph and the same section (literal string, or equal after rounding to the token's own precision; percentages also as fractions, for example 64.1% as 0.641); (2) literally anywhere in `results/`, COMPARISON, DECISIONS, AUDIT, `configs/`, `dh2/`, `tests/`, `pricer/`, the root scripts, `v1_results/`, `outputs/`, `docs/research_1.0/`.

| Document | Tokens | Found in a cited file | Found only by literal string elsewhere | Found only by rounding in an unrelated file | Not found |
|---|---|---|---|---|---|
| RESULTS.md | 940 | 896 | 30 | 13 | 1 |
| README.md | 114 | 102 | 12 | 0 | 0 |
| PRESENTATION_NOTES.md | 237 | 193 | 42 | 2 | 0 |
| docs/EXPLAINER.md | 828 | 559 | 264 | 5 | 0 |

Every token in the last three columns was read by me:
- **Not found (1):** `2009` in RESULTS.md, the year of the Christoffersen-Heston-Jacobs reference. Benign.
- **Found only by rounding (20):** the by-year table values (3.015, 2.695, 1.049, 1.634 and others), which are in `forecast_by_year.csv` with more decimals (all 22 verified in A18); `38.4` and `42.9` microseconds (`speed.csv` stores milliseconds per quote, 0.0384 and 0.0429); `8,800` (8,807 microseconds in `timing.json`); `8e-21` (8.17e-21). All benign.
- **Found only by literal string elsewhere (348):** 115 are small integers, years or list numbers; 161 are decimals that repeat, at coarser precision, numbers verified in section 2 (for example 1.47 for 1.4675, 2.16 for 2.165, +0.13 for +0.1285) or section numbers such as "6.2"; the remaining 72 are integers of three digits or more and percentages: 1,173 / 1,256 / 2,429 / 18,071 / 65,465 / 4,128 / 1,380 / 119 / 283 / 128 / 120 / 6,000 / 30,000 / 1e3-1e6 (counts or settings, each verified in the table above or in `configs/`); 64.1%, 10.9%, 0.8%, 82%, 7%, 1%, 5%, 91%, 64%, 0.3%, 2% (verified in A27, D3-D12, I7); 95% (the interval level in `configs/nifty.yaml`); "20% versus 21%" (an illustration of what one vol point means, not a result); 30%, 36%, 18% (figure 11 readings, verified in K2); +22% (bucket label); reference years 1993, 1999, 2001, 2008, 2014, 2015 (citation years). No token was unexplained.
What this check cannot do: a number that is wrong but happens to appear literally somewhere else in the repository would pass it. That is why section 2 recomputes the headline numbers from the artifacts instead of relying on this scan.

---

## 4. Mismatches and corrected values

Numerically the package is very close to its artifacts: none of the headline figures is wrong. The 15 mismatches are about definitions, scope, counts and documentation; each is a row of the claim table (column "Row").

| # | Row | Where | Written | Corrected |
|---|---|---|---|---|
| M1 | J7 | RESULTS 12 "Reading it (g)", COMPARISON line 123 | "three of the four versions have a positive average whose range stays above zero with no cost" | all four versions (lower ends +1.35, +1.83, +0.22, +2.33 rupees); the 1%-cost sentence is right (only the hedged rich/cheap version stays above zero: +0.19) |
| M2 | J6 | RESULTS 12 and 1(g), README, EXPLAINER A15/Q8, COMPARISON | "41 to 45 of 100 trades profit" | true for the rows of `results/tables/trading_summary.md` (41.1 to 45.0); the matched table printed in the same section has 40.6 to 48.2 (37.7 at 1% cost) |
| M3 | J8 | RESULTS 12, COMPARISON line 123, PRES s14 | "none of the one-day-later averages is distinguishable from zero" | true for the matched trades; false for the 4,525-trade hedged rows of `trading_summary.md` at 1% cost: -2.04 [-3.31, -0.76] and -2.02 [-3.25, -0.70] |
| M4 | I3 | RESULTS 7 table, EXPLAINER A9/Q2, PRES s10 | median standard error / bound width 0.124 for the jump model, defined as median over parameters and days | pooled median 0.153 (15%); 0.124 is the median over days of each day's median. Diffusion 0.212 against 0.213: no change |
| M5 | S1 | README line 19 (table headed "2022-2026 test, 1,173 days") | "better in all 11 calendar years" | 5 of 5 calendar years inside the test period; the other 6 are development years |
| M6 | S2 | RESULTS 6.1, README, EXPLAINER Q3, PRES s7/B14 | "fifteen reruns" | 14 reruns plus the headline run itself (`dhj_main`), including the known-buggy frozen-refit run |
| M7 | S3 | README line 67, EXPLAINER Part D | `python3 -m pytest -q` gives "about 170 fast tests (14 slow deselected)" | needs `-m "not slow"`; plain `pytest -q` runs the slow tests too |
| M8 | S4 | README, PIPELINE "What is cached" | PINN checkpoints cached, 16 MB | only `final.pt` and `final_greeks.pt` are in git; a fresh clone's default run stops at `pinn_checkpoints` and blocks metrics, figures and report |
| M9 | S5 | `configs/nifty.yaml` line 4 | `--quick` "about 10 minutes" | 5.3 minutes |
| M10 | S6 | RESULTS 8, EXPLAINER A11 | PDE residual "falls about 64 times per halving" | 58x and 60x for the project pricer; 43x and 31x for the COS check in the same table |
| M11 | S7 | RESULTS 1(3) and 4, README, PRES s3 | "at most 1.5e-6 vol points on 18,071 real-chain points" | at most 1.5e-6 on the 16,492 diffusion points priced at Rs 0.05 or more (18,071 for jumps); over all 18,071 diffusion points the implied-vol difference reaches 4.30 vol points (price difference at most Rs 4.1e-5) because near-worthless options cannot be inverted |
| M12 | S8 | EXPLAINER Part F | says, in the present tense, that AUDIT.md section 4, the `dh2/struct.py` docstring and COMPARISON still describe the old design | all three were corrected on 2026-10-09 (AUDIT section 4 now says "a large cause"; `struct.py` carries the decision-11 note; COMPARISON header says "not shown to help"); Part F's present-tense statements are outdated |
| M13 | S9 | RESULTS 8 table, README, PRES s12 | "data-only 0.049 (shipped for calibration), hybrid 0.077 (shipped for Greeks)" | these are 3-seed means; the shipped checkpoints are seed 2 of each: data-only 0.0519, hybrid 0.0642 (project-measured, same held-out set) |
| M14 | S10 | DECISIONS 6 | DH1 baseline "on the same clean data" | the waterfall's step 0 is on 1.0's data; clean data is step 1 |
| M15 | S11 | DECISIONS 4 | sensitivity "on 2025-26" | the delivered sensitivity covers the full test period |

## 5. Logic and wording problems (most important first)

**L1. The headline gain comes entirely from quotes that yesterday's smile does not cover. (New finding; not in any document.)** The benchmark reads yesterday's smile by interpolation and, outside yesterday's quoted strikes, repeats the edge value. I split the 524,959 test quotes by whether the strike (scaled by sqrt(T0/T1)) lies inside the range yesterday's matched expiry quoted. Inside (93.5% of quotes, about 418 a day) the time-scaled smile is **better**: daily RMSE 1.006 against 1.018 for jump model + misfit, difference +0.012 [+0.009, +0.016], model better on 52% of days (Wilcoxon p 0.048); the diffusion model + misfit is far worse there (1.202). Outside (6.5%, about 29 quotes a day) the benchmark scores 2.29 and the model 1.10: -1.19 [-1.28, -1.11]; 104% of the pooled squared-error reduction comes from these quotes. So "better than the best no-model benchmark in 11 of 11 years" is true, but it is a statement about extrapolating to strikes that yesterday did not quote; where yesterday's smile exists, it ties or is slightly better. RESULTS ("the model's advantage is in the wings and at short expiries") is correct but not sharp enough, cites a file (`tests_test_period.md`) that only splits all quotes against abs k <= 0.04, and calls the clamped-interpolation smile "the best no-model benchmark" although only two persistence rules were tried. A benchmark with a sensible wing extrapolation (for example linear or SVI-style continuation, or an at-the-money level carried flat) may close part of the gap; this was not tested. Suggested wording: "The jump model + misfit is better than the benchmark for strikes beyond yesterday's quoted range (6.5% of quotes); inside that range the benchmark is slightly better (+0.012 vol points [+0.009, +0.016], with the model ahead on 52% of days)." Method: my script `t_extrap.py`/`t2_extrap_daily.py` on the same stored predictions; the in-range test is `scaled log-moneyness within [min, max] of yesterday's quotes of the matched expiry`.

**L2. Forecast variants computed and not reported.** `hybrids.py` builds seven variants; the documents show two (carry, carry + misfit). The others test the model's own dynamics. Test-period means for the jump model (my recomputation from `hyb_dhj_main.pkl` with `evalpairs.score`; no document or results file reports them): model's one-day mean reversion ("revert") 1.199 against carry 1.201 (no gain); leverage update ("lev") 1.291 (worse); leverage plus misfit 1.136 against 1.034 (worse); yesterday's smile plus the model-implied change 1.552 against the plain smile 1.468 (worse). So the forecasting value of the model is carry plus yesterday's misfit; its dynamic content (mean reversion, correlation with the index) adds nothing or hurts. This is an unreported null result and bears on the sentence "forecasts tomorrow's implied-volatility surface". Also, for single Heston the development-only selection rule would pick leverage plus misfit (dev 1.774 against 1.776) and it is better on test (1.382 against 1.416); the documents use carry + misfit for all models. It does not change any conclusion.

**L3. "11 of 11 calendar years" mixes development and test years.** Six of the eleven years (2016-2021) are where settings were chosen. README puts it in the row of the table headed "2022-2026 test", PRES slide 7 and EXPLAINER fig 02 present it as consistency evidence. For the untouched period it is 5 of 5. Also EXPLAINER Q3 lists it among five "overfit checks"; the hidden-quote check (item 3 there) is a within-day interpolation test, and the robustness reruns (item 4) score the test period, so they show insensitivity to settings, not absence of overfitting.

**L4. "Fifteen reruns ... score 1.033 to 1.036" understates how little the model matters once the misfit is added.** The same fifteen runs span 1.185 to 1.436 without the misfit (including the frozen-refit bug run, 1.436). With the misfit the spread collapses to 0.003, which says the final forecast is almost insensitive to the model's slow parameters, in line with decision 11, but the text presents the collapse as robustness of the model. Run 15 is the headline run itself.

**L5. Pricer headline scope.** The 1.5e-6 vol points holds for options worth at least Rs 0.05 (`errors.csv` and the study README say so; RESULTS, README and the slides do not). Three different "ill-conditioned" thresholds appear: Rs 0.01 (DECISIONS 10), Rs 0.05 (pricer study), about Rs 1 (PINN scoring floor). One rule should be stated.

**L6. "5.03 to 1.10" (README table, PRES s9) is not "1.0's error to 2.0's error" on 1.0's own terms.** The waterfall scores every quote of the clean surface, including 3-19 day expiries and far wings; on 1.0's own metric (held-out third, 20-800 days) the rebuilt DH1 scores 2.27 on the same 118 days (RESULTS 5 says so; README and the slides do not). "Close in aggregate (2.53 against 2.37)" is a different sample (ten April-2026 dates). DECISIONS 6 says the DH1 baseline runs "on the same clean data", while waterfall step 0 uses 1.0's data and "clean data" is step 1.

**L7. Trading wording.** (a) M1-M3 above. (b) "a coin flip would give 50": the relevant blind yardstick is the "blind, same mix" column (38.5 to 43.4), which the table shows; always selling the same options wins 61 to 65 of 100 trades (its average is negative or about zero, except +2.06 rupees for the plain rich/cheap rule), so the win count alone is a weak signal. The model beats its blind same-mix control by 1.5 to 3.0 wins per 100, a fact worth saying. (c) The signal is computed from the closing print and traded at the same print; the lag-1 test is the right control and the docs say so, but the rich/cheap signal is the kind that a one-print bounce can produce, so "possible edge" is generous (my reading; not tested).

**L8. PINN-accelerated calibration.** (a) The 8x is against the four-start exact fit (5.85 s). The control "one exact start, same polish" takes **0.40 s** per day, 1.8 times faster than the accelerated fit (0.73 s), at 0.28 vol points worse RMSE; the documents give the control's accuracy but never its time, and no cheaper exact configuration of equal accuracy was tested, so the "speed at the same error" claim is relative to an expensive baseline. (b) RESULTS 9 does not mention that the global stage leaves out about 9.6% of the fit quotes (those beyond 0.6 years) on 92% of days; the `pinn_calib` README does.

**L9. PINN accuracy headline.** It is a 3-seed mean, the shipped files are seed 2 (0.052 and 0.064), and it excludes the 7,605 held-out quotes (10%) whose exact price is below about Rs 1 and the 1,741 beyond 0.6 y; RESULTS 8 gives the count of scored quotes but not the exclusions. My re-measurement of `final.pt`: Rs 0.01 to 1 mean 0.104; below Rs 0.01 mean 4.7 vol points (3.7% of in-box quotes), which matters if an optimiser visits such parameters.

**L10. Identifiability.** Median standard error 0.124 for jumps is a median of daily medians (pooled 0.153); "today's variance levels are tight (about 1.4%)" is the diffusion model (jumps 2.8-3.0%).

**L11. PDE verification.** "falls about 64 times" is the theoretical rate for a sixth-order stencil; the table shows 31x to 60x.

**L12. Documentation drift.** EXPLAINER Part F is stale (M12). AUDIT.md still reads as a pre-upgrade snapshot (PINN, VIX, SVI, cross-section "missing"; "existing tests (12)") although README lists it under "where to read next"; it is dated, but a line saying "superseded" would avoid confusion. DECISIONS 4 mentions "a sensitivity run on 2025-26" whereas the delivered sensitivity covers the full test period. The README "Run it" block and PIPELINE do not mention that 5 PINN tests need the pickle (section 1.3).

**L13. Things checked and found consistent with DECISIONS 8-11 (no contradiction found).** Decision 8 (data-only shipped for calibration, hybrid for Greeks, 0.049 against 0.077, vega-plus-tail gate) is repeated correctly in RESULTS 8, README, EXPLAINER A13/Q1, PRES s12. Decision 9 (box, 2.3%) matches (F11); RESULTS 8 also lists the restricted parameter ranges. Decision 10 is used consistently in the calibration description (vega floor 0.05, soft-L1 0.8, both confirmed in `dh2/calib.py`). Decision 11 is carried through every document and its numbers (+0.08, p 0.003, 1.142 against 1.254, -0.155) match B3-B5, B8; the structure/state design is not presented as the source of the gain anywhere I looked (README, RESULTS 1(a)/5/13, EXPLAINER A10/E1, PRES s9, COMPARISON header).

**Negative findings hidden or softened?** None is hidden: near the money, diffusion loses, structure/state, PINN-versus-plain, 0.8% higher cost, identifiability, trading are all stated, mostly up front. Softened or incomplete: L1 (no benefit inside yesterday's range), L2 (dynamics add nothing), L8(a) (speed baseline), M2/M3 (trading ranges), L3 (11/11).

---

## 6. Counts

Rows of the claim table in section 2 (A1 to K9 and S1 to S11; each row covers one claim and often several values, about 600 individual values in all):

| Rows checked | MATCH | MISMATCH | NOT CHECKABLE |
|---|---|---|---|
| 143 | 123 | 15 (M1-M15, section 4) | 5 (D18, E9, J10, K8, K9) |

Of the 123 MATCH rows, 12 carry a qualifier in the status cell (read from the cited file and not recomputed, bootstrap-method noise, or a scope remark pointing to a mismatch row). MATCH means the claim is true as written to the stated precision; where it is true but incomplete, the incompleteness is in section 5 (L1 to L13).
Environment test: PASS for the fast tests, `--list` and `--quick`; FAIL for the default one-command run on a bare clone (section 1.3 item 1). Clone left in place: `/Users/vinayakmangal/Everything related to my projects/Dheston_verify`, 2.2 GB by `du` (1.0 GB of it the venv). No process of mine is left running (`ps` checked; the pre-existing `progress.py --serve` belongs to the project and was left alone).

## 7. Overall verdict

The package is reproducible and its numbers are accurate: in a clean clone with the cached data the fast tests (167 passed), `--list` and `--quick` (5.26 minutes, jump model + misfit 1.0706 against the documented 1.071) all work, and forced reruns of the tables, data-quality, VIX, statistics and trading stages reproduce the tracked outputs byte for byte; I re-derived the headline forecast scores, the paired difference and its interval, the 11-of-11 years, the waterfall, the cross-section, the PINN-accelerated calibration, the pricer maxima (plus a fresh three-option spot check), the PINN accuracy (re-measured on the shipped `final.pt`), the VIX correlations, the 2023 data funnel, the identifiability and SVI figures and the trading numbers, and found no wrong headline number. What needs fixing is wording and scope, not arithmetic: the trading sentences (M1-M3), the identifiability median (M4), the "11 calendar years" claim in the test-period table (M5), "fifteen reruns" (M6), the pricer scope (M11), two documentation gaps that make the one-command run fail on a bare clone (M7, M8), and a stale Part F. The most important interpretive point, which no document makes, is that the forecast advantage over the benchmark exists only for strikes outside yesterday's quoted range (6.5% of quotes); inside it the time-scaled smile is slightly better, and the model's own dynamics (mean reversion, leverage) add nothing, so the defensible claim is "the jump model plus yesterday's misfit forecasts the wings that yesterday's smile cannot reach better than a smile carried flat, and is within about 0.01 vol points of it (slightly worse) elsewhere", not a general forecasting superiority; the unreported variants (L2) and the speed baseline (L8) should also be stated.


---

## Resolution (2026-10-09)

Written after the verification above (that text is unchanged). Status of each finding; "checked" means verified by search or by running the code after the edit.

**Mismatches M1-M15.** Fixed in RESULTS.md, README.md, PRESENTATION_NOTES.md and docs/EXPLAINER.md: M1 (all four versions are above zero with no cost), M2/M3 (the 41-45 range is quoted for the first table, the matched table is quoted for the one-day-later claim), M4 (pooled 0.153 vs daily-median 0.124 stated), M5 and L3 ("11 of 11" replaced by "all 5 test years, and the 6 development years"; no occurrence left, checked), M6 ("the headline run plus fourteen reruns"; "fifteen reruns" no longer appears, checked), M7 (plain `pytest -q` now deselects slow tests: pytest.ini addopts; checked), M8 (the pinn_checkpoints stage now needs only final.pt and final_greeks.pt unless --retrain-pinn; checked in dh2/run_all.py), M9 (config header says about 5 minutes; checked), M10 (PDE refinement factors stated as 31-60), M11 (1.5e-6 is the maximum over options priced at 0.05 rupee or more), M12 (EXPLAINER Part F rewritten: the old text was corrected on 9 October), M13 (3-seed means vs the shipped seed-2 files), M14 (DECISIONS wording: step 0 uses 1.0's data), M15 (the sensitivity run covers the full test period).

**Logic and wording L1-L13.** L1 (lead comes from strikes outside yesterday's quoted range): confirmed independently (inside 1.006 vs 1.018, outside 2.288 vs 1.100) and extended with a second, linear-wing benchmark (results/benchmark_check/, DECISIONS decision 12): the lead is -0.202 against the edge-value benchmark and -0.045 [-0.061, -0.032] against the linear-wing benchmark; stated in README, RESULTS, PRESENTATION_NOTES and EXPLAINER (new question E0). The linear-wing benchmark was added after the verification, one variant was tried, stronger ones (SVI continuation) are untested. L2 (unreported forecast variants): scored by exp_variants_table.py into results/tables/forecast_variants.*; the model's own dynamics add nothing. L4 (fifteen runs collapse once the misfit is added): stated as insensitivity to settings, not evidence of overfitting resistance. L5-L9, L13: wording corrected in the documents. L8 (exact-only polish 0.40 s but 0.28 vol points worse; 9.6% of fit quotes outside the box on 92% of days) stated in RESULTS section 9. Not independently re-checked after the edits: the exact sentences for L10-L12 in each document.

**Pipeline.** New stages wing_benchmark and forecast_variants (36 stages in total); metrics.json and the tables include the benchmark check and the variants table; a 34th figure shows the inside/outside split; the fast test suite passes (172 passed, 14 slow deselected, 2 expected failures).

**Not fixed.** The independent audit report cited in a document (J10) is not in the repo; the 22-hour compute figure (K8) and the "settings chosen on development data only" process claim (K9) remain statements that cannot be re-derived from files.
