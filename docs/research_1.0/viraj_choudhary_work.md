# What Viraj Choudhary built: a plain-English dissection

Repository: `double-heston-inverse-calibration` (origin `virajchoudhary/double-heston-inverse-calibration`).
Author names searched: `virajchoudhary` and `Viraj Choudhary` (`git log --all -i --author=viraj`, 186 commits, 6 Aug to 25 Sep 2026).
Method: I read the newest copy of every file across all branches with `git show`, without checking anything out. Each function description comes from its docstring or its first lines of code, which a script pulled out for every function. Where a description is missing, the table says "not read". "Used by" was found by searching every branch tip that contains the file for imports of that module. It can miss dynamic calls.

Who is who:
- **Viraj** wrote most of the research pipeline.
- **Dhruva** (DhruvaBamb / Dhruva Ambhaikar) added the PINN code (`src/mentor_dh_pinn/regular_pinn*`, `models/pinn_model.py` and others).
- **The user** (mangalvinayak13-vin) forked the repo later.

One important trap: many of Viraj's files show "last touched by DhruvaBamb, 21 Aug, commit 8e281ba". That commit is Dhruva's bulk re-import of Viraj's work, called "Initial commit of downloaded files". I checked it: for `src/`, `scripts/`, `models/` and `tests/`, it is byte-identical to Viraj's `main` at his merge `8a6b009` on 13 Aug. So those files are Viraj's.

---

## 1. In one paragraph

Viraj tried to answer a simple question. If you only see a stock's option prices, can you work out the ten hidden settings ("parameters") of the Double Heston model that produced them? And can a fast neural network learn to do that instead of a slow optimizer? He first wrote and checked his own Double Heston price calculator. It matched an independent slow calculator on 36 out of 36 test cases (`docs/DOUBLE_HESTON_VALIDATION_RESULTS.md`). He then looked at which real NSE option prices actually trade and froze a 20-price "R2" input format. He built a clean 10,000-surface synthetic dataset and compared three methods on 1,250 untouched test surfaces:
- Model 1: a plain neural network.
- Model 2: a neural network with built-in rules plus a "re-price and compare" loss.
- Traditional calibration: a slow optimizer.

The honest conclusion is mostly negative. The slow optimizer was the most accurate, with range-scaled parameter error 0.1151 against about 0.166 for both networks. But it took about 376 seconds per surface, while the networks needed only milliseconds. Model 2's extra "physics-style" loss gave no measurable advantage over the plain network (`docs/R2_PRIMARY_COMPARISON_RESULTS.md`). The central finding is that the ten parameters are not reliably recoverable: very different parameter sets produce almost identical prices. The traditional fits re-priced within 1e-4 on 100% of surfaces, yet only 71.9% had parameter error of 0.10 or less (same file). Earlier diagnostics had already shown this. One found 40 near-perfect fits split into 39 separate parameter clusters (`docs/RESULTS_TO_DATE.md`). Another found Double Heston gave "NO_CLEAR_WINNER" against plain Heston on real NTPC options (`docs/NTPC_MENTOR_CHECKPOINT.md`). Viraj then built heavily locked machinery for several next steps. None of these steps produced any committed result:
- a genuine physics-equation network ("Model 3");
- out-of-distribution (OOD) tests;
- noise tests at higher noise;
- a final real-market test ("G8");
- a mentor-requested forward PINN.

His last code (11 Sep) tried to improve Dhruva's PINN, and the decision note says it did not help: 0 out of 4 recoveries for every variant (`outputs/regular_pinn_recovery/PHASE1_PHASE2_DECISION.md`).

---

## 2. Timeline (from commit dates and messages)

- **6 Aug 2026** — Set up the first neural-network ("ANN") project (`f988a0e`) and wrote his own Double Heston price calculator, `src/double_heston.py` (`dea6a19`).
- **7 Aug** — An independent slow reference calculator agreed on 36/36 cases (`47773b8`). Two audits of the parameter limits ended with the verdicts `NEEDS_BOUNDS_REVIEW` and `NEEDS_SAMPLER_CORRECTION` (`bdb5aa2`).
- **9–10 Aug** — "Stage A" screen of official NSE option data for 1, 15 and 22 July 2026. NTPC, CIPLA, INFY and HDFCBANK were chosen as the four stocks (`e6d2056`, `e3c1fd4`).
- **11–12 Aug** — "G2" identifiability studies. Global ambiguity was established: many parameter sets fit equally well. Adding extra information was judged "insufficient" (`383186a`, `8226dc0`, `20215f8`).
- **12–15 Aug** — NTPC real-market pilot comparing Black-Scholes, Heston and Double Heston, with verdict "NO_CLEAR_WINNER". Three follow-up attempts to stabilize the Double Heston fit (new coordinates, a bigger optimizer budget, three dates at once) all failed (`dd53915`, `f9e1155`, `acb0b9f`, `e301438`).
- **22 Aug** — An overnight run with Viraj as coordinator ("Node A"; Dhruva ran Nodes B and C). It found a bug in the imported "Archive-2" physics loss. That loss silently set derivatives to zero (`3cba33d`). The same day, a sealed R2-versus-R3 study froze the two-expiry "R2" format (`09aa1d7`, merged `d81d543`).
- **23 Aug** — Generated and validated the final clean 10,000-surface dataset (`ad6d497`). Locked the imported code so it can never train on real market data (`d29a437`).
- **24 Aug** — Main comparison finished: traditional calibration was best, and the two networks were about equal (`a42566e`, merged to main as `72ad8e1`).
- **24–27 Aug** — Noise study. Networks were tested at 0–1% noise. Traditional calibration was done only at 0% and 0.10% (`4ffddc2`, `83c19ee`).
- **25–27 Aug** — Built and sealed the machinery for Model 3, OOD, G8, the paper draft and a "research control plane". None of it was run. The Model 3 cloud launch was blocked because no Kaggle login was found (`f34a4d3`, `84d70e1`, `7eecc71`, `1fe18e6`, `f5ffd01`, `ff2d7d1`).
- **28–29 Aug** — Mentor forward-PINN baseline plus Phases 2A–2D, 3A and the 3B stock audit (`5fdef34` … `bbe8be6`). No training results are committed.
- **11 Sep** — Fixed a not-a-number (NaN) bug in the PyTorch pricer and made a byte-reproducible "V2" dataset. Tried L-BFGS and Jacobian fine-tuning of Dhruva's regular PINN, which gave no improvement (`2ceca16`, now in `origin/main`).
- **25 Sep** — Merged Dhruva's PR #35 "Double/Single Heston: market tests, improved DH-PINN, and presentation figures" into main (`434a5d0`). This is Viraj's last commit.

Branch status, checked with `git merge-base --is-ancestor`:
- **On `origin/main`:** the engine, Stage A, G2, the R2 format, the 10k dataset, the main comparison, the quarantine lock and the Sep 11 work.
- **Not on `origin/main`:** the noise study, OOD, G8, Model 3, the mentor PINN, the paper and research control. These live only on their own `origin/research/*` and `origin/codex/*` branches.

---

## 3. Folder by folder

Status words:
- **Live**: still used by other code on `main`.
- **Branch-only**: exists only on an unmerged branch.
- **Superseded**: replaced by later work.
- **Record**: a failed or finished attempt kept for history.

### src/ (top-level files)
- **What:** the core engine and the early pipeline:
  - `double_heston.py`, the price calculator;
  - `double_heston_reference.py`, the slow checker;
  - `calibrate_double_heston.py`, the traditional optimizer;
  - `constraints.py`, the rulebook;
  - `pricing_interface.py`, the doorway to the pricer;
  - `synthetic_dataset.py`, `surface_grid.py` and `train.py`, the original 108-price network pipeline;
  - `evaluate_*.py`, audits and runners.
- **Why:** you need a trusted price calculator before you can test anything else.
- **Result:**
  - The 64-node pricer differed from the reference by RMSE `5.458369984817452e-13` on 36/36 cases (`docs/RESULTS_TO_DATE.md`).
  - On a clean test surface, parameter error was `2.2178553561322553e-11`. With 1% noise it rose to `0.43113598206350984` (same file).
- **Status:** Live. `double_heston.py`, `constraints.py` and `calibrate_double_heston.py` are used by almost everything on `main`. The 108-grid files (`surface_grid.py`, `synthetic_dataset.py`, `train.py`) are superseded: Stage A rejected the 108 grid.

### src/g2_r2r3/
- **What:** the harness for the sealed "R2 vs R3" study: two expiries (20 prices) versus three (30 prices).
- **Why:** to pick the input format with a rule written before seeing any results.
- **Result:**
  - R3 was better on clean data.
  - At 0.5% noise it was "NO_MATERIAL_IMPROVEMENT", so R2 was frozen.
  - The verdict was `G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY`.
  - At 0.5% noise the median parameter error was 0.383 while the price fit was 0.88% (`docs/G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md`).
- **Status:** Record. The study is finished and on main, and nothing imports this package now.

### src/r2_representation/
- **What:** the official R2 surface object: 20 price slots plus a "usable" mask, with saving and loading.
- **Why:** so every later step reads the same format and nobody fills in missing real quotes.
- **Result:** an interface, not a number.
- **Status:** Live on main. It is used by `r2_primary`, `r2_synthetic_generation`, OOD and G8.

### src/r2_synthetic_generation.py and src/r2_final_generation.py
- **What:** the frozen recipe for building the training data, and the separately authorized 10,000-surface run.
- **Result:** `data/final_r2_clean_10000/`, with splits 7,500 / 1,250 / 1,250 and SHA-256 starting `148b579a` (`docs/R2_PRIMARY_COMPARISON_RESULTS.md`).
- **Status:** Live on main. The dataset is the input to every later experiment.

### src/r2_primary/
- **What:** dataset loading, training of Model 1 and Model 2, traditional calibration on the test split, the frozen scorecard and the one-time final evaluation.
- **Why:** this is the project's main experiment.
- **Result** (all from `docs/R2_PRIMARY_COMPARISON_RESULTS.md`):

  | Measure | Model 1 | Model 2 | Traditional |
  |---|---|---|---|
  | Range-scaled parameter RMSE | 0.1659 | 0.1664 | 0.1151 |
  | Repricing within 1e-3 | 77.84% | 78.08% | 100% |
  | Time per surface | about 0.004 ms | about 0.008 ms | about 376 s |

  Only 7.6% of the networks' good-repricing surfaces also had parameter error of 0.10 or less.
- **Status:** Live on main. OOD, noise, G8 and Model 3 all import it.

### src/r2_noise/
- **What:** the observation-noise study at 0%, 0.10%, 0.25%, 0.50% and 1.00%.
- **Result:**
  - The networks' parameter error barely moved. Model 1 seed 11 went from `0.16599` at 0% to `0.16603` at 1%.
  - Their price fit to noisy data got worse: repricing within 1e-3 went from `0.8008` to `0.5632` (`evidence/r2_noise_robustness/neural/all_neural_seed_headline.csv`).
  - Traditional calibration was run only at 0% and 0.10%. At 0.10%, 750 starts were recorded and 103 were flagged as successful (`evidence/r2_noise_robustness/traditional/level_0_10pct/run_summary.json`).
  - The 0.25–1.00% traditional levels were never run. The draft paper says the three-way noise comparison is incomplete (`paper/sections/00_abstract.tex` on `research/paper-synthesis`).
- **Status:** Branch-only (`research/r2-noise-traditional-overnight`). Unfinished.

### src/model3_pde/ and src/model3_evaluation/
- **What:** Model 3, a "real PINN" whose pricing network must obey the Double Heston equation, plus its locked evaluation layer.
- **Why:** to test whether genuine physics helps where Model 2 did not.
- **Result:** none. The cloud check recorded `MODEL3_STAGE_A_BLOCKED_CLOUD_AUTH_PREPARATION_READY` and `stage_a_executed: false` (`handoff/model3_stage_a_cloud_discovery.json`).
- **Status:** Branch-only. Never trained.

### src/g8_readiness/ (and scripts/validate_g8_protocol.py)
- **What:** a locked plan for a final test on fresh real NSE data. The dates must fall between 30 Sep and 31 Dec 2026 and must not have been used in development.
- **Result:**
  - The checkpoint check passed: "six of six checkpoints loaded after hash approval".
  - But "No real NSE/RBI acquisition or availability scan was performed" (`outputs/g8_checkpoint_forensics/review_package.md`).
- **Status:** Branch-only (`research/g8-acquisition-readiness`). Never executed.

### src/ood_boundary_protocol.py and src/ood_boundary_evaluation.py
- **What:** 420 frozen "unusual" test surfaces (near parameter edges, outside the training range, odd maturities, and with missing slots), plus the evaluation harness.
- **Result:**
  - The cohorts were built with zero pricing failures (`evidence/ood_boundary_protocol_v1/integrity_report.json`).
  - The evaluation status is `EVALUATION_INFRASTRUCTURE_SEALED_NO_RESULTS` (`docs/OOD_BOUNDARY_EVALUATION_READY.md`).
- **Status:** Branch-only. Never run on the research cohorts.

### src/research_control.py (and configs/research_*.yaml, scripts/verify_research_state.py)
- **What:** a machine-checked register of experiments, what each one depends on, and what may run next.
- **Why:** to stop anyone running steps out of order or quoting stale status.
- **Result:** status documents only.
- **Status:** Branch-only (`research/repro-orchestration`).

### src/dheston/real_market_policy.py
- **What:** a safety lock on Dhruva's imported "Archive-2" code so it cannot train network weights on real market data.
- **Result:** 20 regression tests (`tests/test_real_market_weight_update_quarantine.py`).
- **Status:** Live on main. `train_double_heston.py` uses it.

### src/nse_stage_a.py, src/market_data_audit.py, src/ntpc_*.py
- **What:** official NSE data download and parsing, plus NTPC helper modules.
- **Result:**
  - Stage A found 24 candidate surfaces.
  - 180-day maturity was "unsupported on all 24 candidate surfaces" (README on `origin/main`).
  - NTPC holdout price RMSE: Black-Scholes 1.05334, Heston 0.910569, Double Heston 0.926825, so `NO_CLEAR_WINNER` (`docs/NTPC_MENTOR_CHECKPOINT.md`).
- **Status:**
  - `nse_stage_a.py`: live. It is reused by G2 and NTPC scripts.
  - `market_data_audit.py`: superseded. It was written for spreadsheet exports, and the NSE screen replaced it.
  - NTPC modules: record.

### src/mentor_dh_pinn/ (Viraj's files only)
- **What:** a mentor-requested forward PINN. It learns call prices for one fixed parameter set while obeying four losses: data, physics equation, expiry payoff and stock-price boundaries. On top of that sit Phase 2A–2D tuning of the loss weights (lambdas), the Phase 3A final evaluation and the Phase 3B volatile-stock audit. Also Viraj's Sep 11 `reproducible_regular_pinn_data.py` and a fix in Dhruva's `torch_pricer.py`.
- **Result:**
  - The mentor PINN outputs were written to an ignored folder, so no training numbers are in git (`docs/mentor_dh_pinn/README.md`).
  - The stock audit chose POWERGRID on 15 July 2026, the only stock and date that passed. Its three-month realized volatility was `0.1768540817014945`, at only the `8.977%` percentile of its own five-year history (`docs/mentor_dh_pinn/VOLATILE_STOCK_SELECTION_AUDIT.md`).
  - The V2 data has 131,072 training rows and "zero rejected proposals" (`outputs/regular_pinn_recovery/ORIGINAL_DATA_PROVENANCE_RECOVERY.md`).
- **Status:**
  - Mentor PINN: branch-only (`research/mentor-dh-pinn-basics`).
  - The Sep 11 files: on main.

### models/
- **What:**
  - `ann_model.py`: Model 1, the plain network.
  - `parameter_transform.py`: keeps outputs in range and standardizes targets.
  - Viraj did not write `pinn_model.py`, which holds Model 2's class. Dhruva added it on 22 Aug.
- **Status:** Live on main. `r2_primary/training.py` builds Model 1 from `ann_model.py`.

### scripts/ (top level)
- **What:** command-line runners:
  - Stage A screen and Power tie-break;
  - the seven G2 analyses;
  - the six R2/R3 study steps;
  - five NTPC experiments;
  - noise, OOD and G8 runners;
  - the Model 3 pilot driver.
- **Result:**
  - NTPC follow-ups: `INSUFFICIENT` for new coordinates (`docs/NTPC_DH_STABILITY_REPARAMETERIZATION.md`).
  - `OPTIMIZER_CAP_UNRESOLVED`: 10 of 12 starts still hit the budget at 320 steps (`docs/NTPC_DH_OPTIMIZER_CAP_SENSITIVITY.md`).
  - `MULTI_DATE_INSUFFICIENT` (`docs/NTPC_DH_MULTI_DATE_CALIBRATION.md`).
- **Status:**
  - G2 and NTPC scripts: record. They are on main but finished.
  - Noise, OOD, G8 and Model 3 scripts: branch-only.

### scripts/cloud/ and scripts/model3_cloud/
- **What:** cloud (Kaggle/GPU) identity checks, benchmarks and launch packaging.
- **Result:** a P100 GPU benchmark "PASSED" before Model 2 training (`docs/R2_PRIMARY_COMPARISON_RESULTS.md`). Model 3 Stage A was blocked.
- **Status:**
  - `cloud/`: on main.
  - `model3_cloud/`: branch-only.

### scripts/mentor_dh_pinn/
- **What:** thin runners for the mentor PINN phases, plus Viraj's Sep 11 V2 scripts: L-BFGS fine-tuning, the Mac precheck, development evaluation and V2 data preparation.
- **Result:**
  - Scaled recovery RMSE: Baseline `0.124257`, Phase 1 `0.150639`, Phase 2 `0.132324`.
  - All passes: 0/4 for every variant. Decision: "Stop here" (`outputs/regular_pinn_recovery/PHASE1_PHASE2_DECISION.md`).
- **Status:**
  - Sep 11 scripts: on main.
  - Phase runners: branch-only.

### tests/ and tests/mentor_dh_pinn/
- **What:** about 65 test files that check every contract (listed one per file in Section 4).
- **Result:** "219 passed at the global-ambiguity milestone" (README on `origin/main`).
- **Status:** Live, alongside their code.

### configs/
- **What:** frozen settings files. Each one fixes the seeds, thresholds and file fingerprints for one experiment before it runs. Examples: `parameter_bounds_PROVISIONAL.yaml`, `r2_primary_comparison_FINAL.yaml`, `r2_noise_robustness_FINAL.yaml`, `g8_final_real_market.yaml`, `model3_pde_protocol.yaml`, `ood_boundary_*.yaml`, `mentor_dh_pinn/*.yaml` and `double_data_v2_reproducible.json`.
- **Status:** Each one follows its experiment's status.

### docs/
- **What:** more than 60 Markdown documents. Protocols were written before each experiment, and results docs after. Also status and handoff notes. The key results docs are:
  - `R2_PRIMARY_COMPARISON_RESULTS.md`
  - `G2_R2_R3_REPRESENTATION_SELECTION_RESULTS.md`
  - `RESULTS_TO_DATE.md`
  - `NTPC_*.md`
  - `OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md`
  - `DOUBLE_HESTON_V2_EXECUTION.md`
- **Status:** Accurate for its own date. On main, the README itself says its status table is "historical".

### paper/sections, paper/generated, paper/scripts
- **What:** a LaTeX paper draft. Its tables and figures are generated straight from the evidence files.
- **Result:** the draft conclusion: "low repricing error can coexist with material parameter displacement" (`paper/sections/14_conclusion.tex`). No PDF was compiled: "pdflatex ... absent" (`paper/README.md`).
- **Status:** Branch-only (`research/paper-synthesis`).

### evidence/ (folder level)
- `g2_r2_r3_20260822/`: every one of the 1,920 R2/R3 fit runs, the truth panel and the final decision. Record.
- `final_r2_synthetic_pilot_20260822*/`, `final_r2_candidate_pool_readiness_20260822/`: the pilot dataset and its replay, and the check that 10,000 parameter sets were available. Record.
- `attic_superseded_pilot_runs/`: two earlier pilot runs, one with an integrity-hash defect. A failed attempt kept for the record.
- `r2_primary_comparison_20260823/`: all predictions, metrics, manifests and cloud provenance behind the headline table. Live (main).
- `r2_noise_robustness/`: noise-study outputs, including a folder of a failed run (`zero_percent_neural_failed_serialization_python313`) and a "containment record". The containment record admits that positive-noise results were first committed before the 0% checks passed, so those first results are kept but classed as non-canonical (`RECOVERY_CONTAINMENT_RECORD.json`). Branch-only.
- `ood_boundary_protocol_v1*/`, `ood_boundary_protocol_failed_attempts/`, `ood_boundary_evaluation_ready/`, `ood_boundary_development_smoke_v1/`: the OOD cohorts, six failed generation attempts kept for the record, and readiness files. Branch-only.
- `R2_CONTRACT_FROZEN_BEFORE_PILOT.txt`, `R2_FINAL_10K_GENERATION_AUTHORIZED.txt`, `OOD_BOUNDARY_PROTOCOL_FROZEN.txt`: "I froze this before looking" markers.

### outputs/ (folder level)
- `double_heston_validation/`: the 6 Aug pricer validation (clean and 1% noise recovery, 12-surface pilot). Record.
- `double_heston_benchmark/`: the 36-case reference benchmark. Record.
- `parameter_bounds_audit/`: the 5,000-candidate limits audit. Verdict `NEEDS_BOUNDS_REVIEW` (`docs/DOUBLE_HESTON_VALIDATION_RESULTS.md`). Record.
- `reviewed_sampling_audit/`: the 19,000-candidate sampling audit. Record.
- `engine_freeze/`: fingerprints of the frozen engine, with `"status": "NEEDS_SAMPLER_CORRECTION"` (`outputs/engine_freeze/decision.json`). Record.
- `core_dataset_readiness/`, `metrics/smoke_test/`: the 108-grid readiness record and fake smoke-test outputs. Superseded.
- `g8_checkpoint_forensics/`: the G8 review package. Branch-only.
- `regular_pinn_recovery/`: manifests and decision notes for the Sep 11 V2 work. On main.

### market_data_audit/stage_a/derived/
- **What:** evidence from the NTPC reparameterization experiment: 12-start results, pairwise distances, an equivalence round-trip table and 8 figures. The raw NSE files themselves are git-ignored.
- **Status:** Record.

### data/
- `final_r2_clean_10000/` (+ `_replay`): the 10k dataset, its integrity and sanity reports, and the replay proving it regenerates identically. Live.
- `r2_noise_robustness/levels/`: four noisy copies of the 1,250 test surfaces. Branch-only.

### .ai-research/overnight/2026-08-22/node-A/
- **What:** Viraj's coordinator notes and five small diagnostic scripts from the 22 Aug overnight run.
- **Result:** it reproduced the Archive-2 bug: "all variance-factor derivative terms are dropped" (`docs/OVERNIGHT_SWARM_2026-08-22_DECISION_RECORD.md`). It also set the naming used after that: Model 1, Model 2, and a future Model 3.
- **Status:** Record (archive branch `archive/overnight-20260822-node-a`).

### handoff/, notebooks/mentor_dh_pinn/, checkpoints/
- `handoff/`: team context notes, a handoff PDF, and Model 3 cloud discovery and package files.
- `notebooks/`: a Kaggle notebook to run the mentor PINN on a GPU. It is not read here.
- `checkpoints/`: holds only `.gitkeep`. The real trained network files were never committed. The OOD doc says they were "untracked and absent".

---

## 4. Function by function

How to read the tables:
- "in-file" lists the functions in the same file that call this one.
- Backticked paths are other files that import and use it.
- "tests" means a test file exercises it.
- "no caller found by search" usually means a command-line entry point or an internal step reached some other way.

Files that are only thin command-line wrappers get one line instead of a table. Files Viraj only edited, but did not write, get one line describing his edit.

### src/ (top-level engine, data, calibration and study files)

#### `src/__init__.py`

*1 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

One-line package marker so src/ can be imported. (No functions in this file.)

#### `src/audit_parameter_bounds.py`

*346 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Early audit (Aug 7): draws 5,000 random parameter sets inside the provisional limits, checks which are valid and prices 250 of them, to see whether the limits are sensible; verdict NEEDS_BOUNDS_REVIEW.

| Function / class | What it does | Used by |
|---|---|---|
| `_load_bounds` | Reads the provisional limits file. | in-file: `_freeze`, `run_audit`; tests |
| `_margin_row` | For one parameter set, records validity checks and how close it sits to each limit. | in-file: `run_audit`; tests |
| `_select_priced` | Picks a spread-out subset of accepted sets to price. | in-file: `run_audit` |
| `_surface_metrics` | Prices one full grid and checks prices are finite and arbitrage-free. | in-file: `run_audit` |
| `_write_reviewed_config` | Writes a "reviewed" limits config marked provisional. | in-file: `run_audit` |
| `_sha` | Fingerprints a file. | in-file: `_freeze` |
| `_freeze` | Writes the engine-freeze evidence and decision files. | in-file: `run_audit` |
| `run_audit` | Runs the full bounds audit and writes outputs. | in-file: `main` |
| `main` | Command-line entry point. | command line |

#### `src/audit_reviewed_sampling.py`

*334 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Second sampling audit (Aug 7): draws 19,000 candidates in four separate groups (normal interior, wide, boundary-challenge, out-of-distribution) and checks them; 81.16% interior accepted, decision NEEDS_SAMPLER_CORRECTION.

| Function / class | What it does | Used by |
|---|---|---|
| `write_json` | Writes JSON, converting numpy numbers. | in-file: `_freeze`, `run_audit` |
| `_range` | Checks a range entry has all required fields. | in-file: `load_reviewed_config` |
| `load_reviewed_config` | Reads and validates the reviewed sampling config. | in-file: `run_audit`, `sample_challenge`, `sample_distribution`; tests |
| `canonical_feller_margin` | A 0-to-1 style measure of how safely the Feller condition holds. | in-file: `_row`; tests |
| `_scale` | Maps a 0-1 number into a range. | in-file: `_polar`, `_row` |
| `_lhs` | Latin-hypercube random draws (evenly spread random points). | in-file: `sample_challenge`, `sample_distribution` |
| `_split` | Assigns a candidate to train, validation, test, challenge or OOD. | in-file: `_row`, `sample_distribution` |
| `_boundary_predicates` | Decides which "near the edge" flags apply. | in-file: `_row` |
| `_polar` | Picks the two correlations by radius and angle inside the allowed disk. | in-file: `_row` |
| `_row` | Builds one candidate parameter set and its checks. | in-file: `sample_challenge`, `sample_distribution` |
| `sample_distribution` | Draws all candidates for one group. | in-file: `run_audit`; tests |
| `validate_challenge_labels` | Checks challenge candidates really have the edge property they are labelled with. | in-file: `sample_challenge` |
| `sample_challenge` | Draws the boundary-challenge candidates. | in-file: `run_audit`; tests |
| `validate_ood_support` | Proves OOD candidates lie outside the normal range and never reach training. | in-file: `run_audit`; tests |
| `validate_declared_range_containment` | Proves every drawn value is inside its declared range. | in-file: `run_audit`; tests |
| `_priced` | Prices a spread-out subset and records validity. | in-file: `run_audit`; tests |
| `_noise_evidence` | Checks how percentage noise affects prices, without clipping or dropping rows. | in-file: `run_audit`; tests |
| `_diversity_by_distribution` | Diversity statistics per group. | in-file: `run_audit` |
| `_freeze` | Updates the freeze fingerprints and decision. | in-file: `run_audit`; tests |
| `run_audit` | Runs the whole sampling audit. | in-file: `main` |
| `main` | Command-line entry point. | command line |

#### `src/calibrate_double_heston.py`

*359 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The "traditional calibration" method: an optimizer adjusts the ten parameters until model prices match observed prices, from three fixed starting points; this is the benchmark both networks were compared against.

| Function / class | What it does | Used by |
|---|---|---|
| `_logit` | Inverse of the S-curve squeeze, with safety clipping. | in-file: `parameters_to_unconstrained` |
| `load_hard_safety_bounds` | Reads the hard numerical safety limits from YAML. | in-file: `calibrate_double_heston`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` and more |
| `unconstrained_to_parameters` | Turns any ten real numbers into a parameter set that obeys every rule (positivity, slow-slower-than-fast, Feller, correlation disk), like a machine that can only output legal moves. | in-file: `calibrate_double_heston`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` and more |
| `parameters_to_unconstrained` | The reverse: finds the ten raw numbers that produce a given valid parameter set. | in-file: `deterministic_initial_starts`; `scripts/run_g2_complementary_observable_analysis.py` |
| `deterministic_initial_starts` | Gives three named starting guesses, one of them informed by the truth (disclosed as favouring this method). | in-file: `calibrate_double_heston` |
| `boundary_diagnostics` | Flags when an answer sits near a limit, the ordering rule, Feller or the correlation disk. | in-file: `calibrate_double_heston`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` and more |
| `calibrate_double_heston` | Runs the least-squares fit from each start and records price error, parameter error and edge flags for every start. | `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` and more |

#### `src/constants.py`

*51 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Fixed shared definitions: the ten parameter names in their exact order and other contract constants (no functions). (No functions in this file.)

#### `src/constraints.py`

*125 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The rulebook for a legal Double Heston parameter set, with a function per rule.

| Function / class | What it does | Used by |
|---|---|---|
| `feller_gap` | Computes 2*kappa*theta - sigma^2, which must be positive so variance never hits zero. | in-file: `validate_parameters` |
| `positivity_is_valid` | Checks every kappa, theta, sigma and v0 is positive. | in-file: `validate_parameters` |
| `correlations_are_bounded` | Checks both correlations are strictly between -1 and 1. | in-file: `validate_parameters` |
| `correlation_disk_value` | Computes rho_slow^2 + rho_fast^2. | in-file: `joint_correlation_disk_is_valid`, `validate_parameters`; `src/audit_reviewed_sampling.py`, `src/evaluate_parameters.py`; tests |
| `joint_correlation_disk_is_valid` | Checks that sum is below 1. | no caller found by search |
| `slow_fast_ordering_is_valid` | Checks the slow factor really is slower (kappa_slow < kappa_fast). | in-file: `validate_parameters` |
| `dictionary_to_vector` | Turns a named parameter dictionary into the fixed-order list of ten. | in-file: `validate_parameters`; `src/double_heston.py`, `src/double_heston_reference.py`, `src/pricing_interface.py`; tests |
| `vector_to_dictionary` | Turns the list of ten back into a named dictionary. | in-file: `validate_parameters`; `src/synthetic_dataset.py`; tests |
| `validate_parameters` | Runs every rule and returns a full report of which pass and which fail. | `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` and more |
| `_coerce_vector` | Checks the input is ten finite numbers. | in-file: `dictionary_to_vector`, `positivity_is_valid`, `validate_parameters`, `vector_to_dictionary` |

#### `src/dataset.py`

*151 lines; Viraj's file; newest version 642702e (Aug 22) is a small edit by Dhruva; on origin/main.*

PyTorch dataset that holds one row per option surface (inputs) with its ten true parameters (answers), never letting one surface cross train/validation/test.

| Function / class | What it does | Used by |
|---|---|---|
| `SurfaceParameterDataset` | The dataset container: features, targets, ids, optional masks and metadata. | `src/run_pinn_improved_benchmark.py`, `src/run_pinn_synthetic_baseline.py`, `src/run_pinn_two_stage_baseline.py`, `src/run_smoke_test.py` and more; tests |
| `SurfaceParameterDataset.__init__` | Stores the arrays and checks ids are unique and values finite. | Python (automatic) |
| `SurfaceParameterDataset.__len__` | Number of surfaces. | Python (automatic) |
| `SurfaceParameterDataset.__getitem__` | Returns one surface's features, targets, id and metadata. | Python (automatic) |
| `SurfaceParameterDataset.from_surface_frame` | Builds the dataset from a long table of quotes, grouping complete surfaces. | `src/run_pinn_improved_benchmark.py`, `src/run_pinn_synthetic_baseline.py`, `src/run_pinn_two_stage_baseline.py`, `src/run_smoke_test.py` and more; tests |
| `SurfaceParameterDataset.indices_for_split` | Lists the surfaces in one split. | `scripts/run_model3_pde_pilot.py`, `src/model3_evaluation/harness.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/calibration.py` and more |

#### `src/double_heston.py`

*417 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The heart of the project (Aug 6): Viraj's own Double Heston option pricer, which turns ten parameters plus strike and maturity into an option price using a Fourier integral with 64 Gauss-Laguerre points.

| Function / class | What it does | Used by |
|---|---|---|
| `validate_double_heston_inputs` | Checks spot, strikes, maturities, rates, option types and parameters before pricing. | in-file: `double_heston_characteristic_function`, `price_double_heston_call`, `price_double_heston_surface`; tests |
| `heston_log_characteristic_exponent` | Computes one variance factor's contribution to the price "fingerprint" using the numerically stable "Little Heston Trap" formula. | in-file: `double_heston_characteristic_function`; `scripts/run_ntpc_single_stock_pilot.py` |
| `double_heston_characteristic_function` | Adds the slow and fast contributions and the drift to get the full fingerprint (characteristic function) of the future log price. | in-file: `price_double_heston_call`; tests |
| `_gauss_laguerre_rule` | Gets the fixed sample points and weights used to approximate the integral. | in-file: `price_double_heston_call`; `scripts/run_ntpc_single_stock_pilot.py` |
| `price_double_heston_call` | Prices one European call by integrating the fingerprint. | in-file: `price_double_heston_option`, `price_double_heston_put`; `src/run_double_heston_validation.py`; tests |
| `price_double_heston_put` | Prices one put from the call using put-call parity. | in-file: `price_double_heston_option`; `src/run_double_heston_validation.py`; tests |
| `price_double_heston_option` | Prices one call or put. | in-file: `price_double_heston_surface`; `scripts/run_ntpc_single_stock_pilot.py`, `src/run_independent_pricing_benchmark.py`; tests |
| `price_double_heston_surface` | Prices a whole list of quotes (a surface) at once. | `scripts/run_g2_identifiability_analysis.py`, `src/audit_parameter_bounds.py`, `src/audit_reviewed_sampling.py`, `src/calibrate_double_heston.py` and more; tests |
| `propagate_variance_state` | Expected future variance of one factor after some days (it drifts back toward theta). | tests |

#### `src/double_heston_reference.py`

*176 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

A second, independently coded slow pricer (adaptive integration) used only to check that double_heston.py is right; they agreed to about 1e-12 on 36/36 cases.

| Function / class | What it does | Used by |
|---|---|---|
| `_vector` | Turns input into a validated parameter vector. | in-file: `_validate_quote` |
| `_validate_quote` | Checks one quote's inputs. | in-file: `reference_double_heston_call`, `reference_double_heston_characteristic_function` |
| `_factor_exponent` | Independent version of one factor's fingerprint contribution. | in-file: `reference_double_heston_characteristic_function` |
| `reference_double_heston_characteristic_function` | Independent full fingerprint. | in-file: `reference_double_heston_call`; tests |
| `_quad_diagnostic` | Runs the adaptive integral and records any warnings. | in-file: `reference_double_heston_call` |
| `reference_double_heston_call` | Reference call price with integration diagnostics. | in-file: `reference_double_heston_option` |
| `reference_double_heston_option` | Reference call or put. | in-file: `reference_double_heston_surface`; `src/run_independent_pricing_benchmark.py`; tests |
| `reference_double_heston_surface` | Reference prices for a list of quotes. | no caller found by search |

#### `src/evaluate_parameters.py`

*146 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Scores predicted parameters against true ones (errors per parameter) and checks they obey the rules.

| Function / class | What it does | Used by |
|---|---|---|
| `evaluate_parameter_recovery` | Computes per-parameter errors, validity rates and saves every row. | in-file: `main`; `src/run_smoke_test.py` |
| `_safe_diagnostics` | Runs the rulebook, treating non-finite predictions as invalid instead of crashing. | in-file: `evaluate_parameter_recovery` |
| `main` | Command-line entry point. | command line |

#### `src/evaluate_repricing.py`

*142 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Takes predicted parameters, re-prices the original surfaces with the real engine, and reports how close the prices come back (repricing error).

| Function / class | What it does | Used by |
|---|---|---|
| `evaluate_repricing` | Re-prices every surface, keeping both successes and failures. | in-file: `main` |
| `main` | Command-line entry point (by default evaluates the controlled calibration output, not a network). | command line |

#### `src/market_data_audit.py`

*385 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Stage A helper (Aug 9) for loading spreadsheet exports of option, futures and spot data and measuring coverage; written for Bloomberg-style files, later superseded by the NSE screen.

| Function / class | What it does | Used by |
|---|---|---|
| `StageASurfaceRaw` | Holds the raw tables and manifest for one stock/date. | in-file: `load_stage_a_surface` |
| `load_audit_config` | Reads the Stage A config. | in-file: `load_stage_a_surface`, `required_columns`; tests |
| `required_columns` | Lists required columns for each table type. | in-file: `validate_required_columns` |
| `validate_required_columns` | Errors clearly if a table is missing a column. | in-file: `_attach_spot`, `derive_option_metrics`, `read_raw_table`, `summarize_futures_coverage`; tests |
| `read_raw_table` | Reads one Excel export and checks columns. | in-file: `load_stage_a_surface` |
| `load_stage_a_surface` | Loads the three tables and manifest for one surface. | tests |
| `derive_option_metrics` | Adds moneyness, maturity bucket and quality flags to option rows. | tests |
| `summarize_option_coverage` | Summarizes which strikes and expiries are covered. | tests |
| `summarize_futures_coverage` | Summarizes futures availability. | no caller found by search |
| `futures_implied_carry` | Computes implied carry from futures as a diagnostic. | tests |
| `_attach_spot` | Adds the spot price to each option row. | in-file: `derive_option_metrics` |
| `_expiry_buckets` | Labels expiries near/mid/far. | in-file: `derive_option_metrics` |
| `_parse_dates` | Parses dates, failing on bad ones. | in-file: `_attach_spot`, `_validate_loaded_identity`, `derive_option_metrics`, `summarize_futures_coverage` |
| `_validate_manifest` | Checks the collection manifest fields. | in-file: `load_stage_a_surface` |
| `_validate_loaded_identity` | Checks the tables match the manifest's stock and date. | in-file: `load_stage_a_surface` |

#### `src/nse_stage_a.py`

*837 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Stage A market screen (Aug 9): downloads official NSE daily files for 1, 15 and 22 July 2026 and measures, for eight candidate stocks, which strikes and expiries actually trade.

| Function / class | What it does | Used by |
|---|---|---|
| `NSEStageAError` | Base error for Stage A failures. | in-file: `ArchiveIntegrityError`, `_acquire_output_lock`, `_cleanup_output_staging`, `_cm_spot_by_underlying` ... |
| `ArchiveIntegrityError` | Error when a stored archive is invalid or conflicts with its record. | in-file: `_atomic_write_new`, `_validate_prior_evidence`, `_validated_zip_member`, `acquire_udiff_archive`; `scripts/run_g2_common_support_analysis.py`, `scripts/run_power_candidate_tiebreak_extension.py`; tests |
| `AcquisitionRecord` | Receipt for one downloaded archive. | in-file: `_acquisition_dict`, `acquire_udiff_archive`, `analyze_stage_a`; `scripts/run_power_candidate_tiebreak_extension.py` |
| `nse_archive_filename` | Official file name for a market and date. | in-file: `acquire_udiff_archive`, `nse_archive_url`; `scripts/run_g2_common_support_analysis.py`, `scripts/run_power_candidate_tiebreak_extension.py`; tests |
| `nse_archive_url` | Official download URL. | in-file: `acquire_udiff_archive`; `scripts/run_g2_common_support_analysis.py`, `scripts/run_power_candidate_tiebreak_extension.py`; tests |
| `validate_udiff_schema` | Checks the CSV has the exact 34 NSE columns. | in-file: `_read_udiff_bytes`, `_require_raw_frame`; tests |
| `read_udiff_csv` | Reads a raw NSE CSV unchanged. | `scripts/run_g2_common_support_analysis.py`, `scripts/run_nse_stage_a_screen.py`, `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_single_stock_pilot.py` and more |
| `acquire_udiff_archive` | Reuses a valid stored archive or downloads it safely from NSE only. | `scripts/run_nse_stage_a_screen.py`, `scripts/run_ntpc_single_stock_pilot.py`, `scripts/run_power_candidate_tiebreak_extension.py`; tests |
| `derive_option_observations` | Adds moneyness, maturity and activity flags to option rows without changing raw fields. | in-file: `analyze_stage_a`; `scripts/run_g2_common_support_analysis.py`, `scripts/run_power_candidate_tiebreak_extension.py`; tests |
| `analyze_stage_a` | Builds the eight Stage A audit tables. | `scripts/run_nse_stage_a_screen.py`; tests |
| `write_stage_a_outputs` | Writes the eight tables as one locked, all-or-nothing update. | `scripts/run_nse_stage_a_screen.py`; tests |
| `read_known_hashes` | Reads earlier output fingerprints. | no caller found by search |
| `read_prior_acquisition_evidence` | Reads earlier download records. | in-file: `read_known_hashes`; `scripts/run_nse_stage_a_screen.py`, `scripts/run_power_candidate_tiebreak_extension.py`; tests |
| `_surface_summary` | Per stock/date summary of expiries and futures. | in-file: `analyze_stage_a` |
| `_expiry_coverage` | Per expiry: how many strikes are quoted and active. | in-file: `analyze_stage_a` |
| `_moneyness_coverage` | Which target strike distances are observed. | in-file: `analyze_stage_a` |
| `_candidate_grid_support` | Which maturities and strike distances of the old 108-price grid are supported. | in-file: `analyze_stage_a` |
| `_derive_futures` | Extracts the candidate stocks' futures rows. | in-file: `analyze_stage_a` |
| `_spot_consistency` | Checks cash-market close equals the F&O underlying price. | in-file: `analyze_stage_a` |
| `_universe_presence` | Checks each stock has options and futures on each date. | in-file: `analyze_stage_a` |
| `_cm_spot_by_underlying` | Each stock's closing price. | in-file: `_spot_consistency`, `derive_option_observations` |
| `_attach_activity_flags` | Adds "traded", "open interest" and "transactions" flags. | in-file: `_derive_futures`, `derive_option_observations` |
| `_expiry_slots` | Labels expiries near/mid/far. | in-file: `derive_option_observations` |
| `_moneyness_classification` | Direct, bracketed or outside for a strike distance. | in-file: `_candidate_grid_support`, `_moneyness_coverage` |
| `_maturity_classification` | Direct, near, bracketed or outside for a maturity. | in-file: `_candidate_grid_support` |
| `_atm_counts` | Counts strikes below, at and above spot. | in-file: `_expiry_coverage` |
| `_empty_option_frame` | An empty table with the right columns. | in-file: `analyze_stage_a`, `derive_option_observations` |
| `_require_raw_frame` | Checks a raw table's schema and identity. | in-file: `_derive_futures`, `_spot_consistency`, `derive_option_observations` |
| `_acquisition_dict` | Receipt as a dictionary. | in-file: `analyze_stage_a` |
| `_validate_prior_evidence` | Checks a reused archive matches its earlier record. | in-file: `acquire_udiff_archive` |
| `_prior_first_acquisition_status` | How an archive was first obtained. | in-file: `acquire_udiff_archive` |
| `_read_udiff_bytes` | Validates raw CSV bytes and records encoding. | in-file: `acquire_udiff_archive`, `read_udiff_csv` |
| `_validate_udiff_identity` | Checks dates, source and segment in the file. | in-file: `_read_udiff_bytes`, `_require_raw_frame` |
| `_download_official` | Downloads only from the official NSE archive site. | in-file: `acquire_udiff_archive`; tests |
| `_validated_zip_member` | Checks the zip and reads its file. | in-file: `acquire_udiff_archive` |
| `_atomic_write_new` | Writes a new raw file safely, never overwriting. | in-file: `acquire_udiff_archive`; tests |
| `_remove_created_raw_path` | Rolls back a file only if this run created it. | in-file: `acquire_udiff_archive` |
| `_write_complete_file` | Writes and flushes a file to disk. | in-file: `write_stage_a_outputs` |
| `_acquire_output_lock` | Takes a lock so two writers cannot clash. | in-file: `write_stage_a_outputs`; tests |
| `_release_output_lock` | Releases the lock. | in-file: `write_stage_a_outputs` |
| `_commit_output_generation` | Swaps in the new outputs with backups. | in-file: `write_stage_a_outputs` |
| `_cleanup_output_staging` | Removes only this run's temporary files. | in-file: `write_stage_a_outputs`; tests |
| `_filesystem_timestamp` | File modification time as a fallback timestamp. | in-file: `acquire_udiff_archive` |
| `_sha256` | Fingerprints bytes. | in-file: `_remove_created_raw_path`, `acquire_udiff_archive` |
| `_numeric` | Converts text to numbers, blanks for bad values. | in-file: `_atm_counts`, `_attach_activity_flags`, `_candidate_grid_support`, `_cm_spot_by_underlying` ... |
| `_market_code` | Checks market is CM or FO. | in-file: `_require_raw_frame`, `acquire_udiff_archive`, `nse_archive_filename`, `nse_archive_url` ... |
| `_as_date` | Converts to a date. | in-file: `_surface_summary`, `acquire_udiff_archive`, `derive_option_observations`, `nse_archive_filename` ... |
| `_optional_nse_date` | Converts to a date or None. | in-file: `_derive_futures`, `derive_option_observations` |
| `_parse_nse_date` | Parses NSE date formats. | in-file: `_as_date`, `_optional_nse_date`, `_validate_udiff_identity` |
| `_ranking_status` | Marks NIFTY as reference-only. | in-file: `_derive_futures`, `derive_option_observations` |

#### `src/ntpc_dh_reparameterization.py`

*269 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The alternative "total plus share" coordinates used in the NTPC reparameterization experiment, proven to cover exactly the same parameter space.

| Function / class | What it does | Used by |
|---|---|---|
| `_unit` | S-curve into 0-1. | in-file: `_bounded` |
| `_logit_unit` | Reverse of that. | in-file: `_inverse_bounded` |
| `_bounded` | Maps a number into a range. | in-file: `_total_pair_from_coordinates`, `structured_to_canonical` |
| `_inverse_bounded` | Reverse of that. | in-file: `_total_pair_to_coordinates`, `canonical_to_structured` |
| `_total_pair_from_coordinates` | Turns a "total" and "share" into slow and fast values. | in-file: `structured_to_canonical` |
| `_total_pair_to_coordinates` | Turns slow and fast values into total and share. | in-file: `canonical_to_structured` |
| `_sigma_upper` | Largest sigma that still satisfies Feller. | in-file: `canonical_to_structured`, `structured_to_canonical` |
| `structured_to_canonical` | New coordinates to the normal ten parameters. | `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |
| `canonical_to_structured` | Normal ten parameters to new coordinates. | `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |
| `canonical_diagnostics` | Validity checks plus the pilot's numeric limits. | in-file: `canonical_to_structured`, `structured_to_canonical`; `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |
| `derived_coordinates` | Readable totals and shares for reporting. | `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |

#### `src/ntpc_pricing_input_contract.py`

*22 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Tiny fix-up module: rebuilds maturity T exactly from days-to-expiry after loading a CSV (the rounding bug).

| Function / class | What it does | Used by |
|---|---|---|
| `canonicalize_pricing_inputs` | Recomputes derived pricing inputs from the stable raw columns. | `scripts/investigate_ntpc_canonical160_reproduction.py`, `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |

#### `src/ood_boundary_evaluation.py`

*1997 lines; newest version 84d70e1 (2026-08-26, virajchoudhary); on research/ood-boundary-protocol.*

The locked evaluation machinery for the 420 "unusual" (OOD/boundary) test surfaces; built and sealed on Aug 26 but never run on the research cohort (Model 1/2 checkpoints absent locally, Model 3 never trained).

| Function / class | What it does | Used by |
|---|---|---|
| `EvaluationSealError` | Error when an action would break the sealed-cohort rules. | in-file: `ResearchEvaluationLocked`, `_record_to_ood_item`, `_safe_output_directory`, `_safe_readiness_root` ... |
| `ResearchEvaluationLocked` | Error when the research evaluation is not explicitly authorized. | in-file: `check_research_authorization`, `load_frozen_research_cohort`; tests |
| `sha256_file` | Fingerprints a file. | in-file: `_artifact_hashes`, `_write_metric_artifact`, `checkpoint_catalog`, `compare_replay` ... |
| `sha256_bytes` | Fingerprints bytes. | in-file: `_settings_fingerprint`, `_surface_order_hash`, `build_development_fixture`, `check_research_authorization` ... |
| `_json_default` | Converts numpy values and paths for JSON. | in-file: `deterministic_json_bytes` |
| `deterministic_json_bytes` | JSON in a fixed, byte-stable format. | in-file: `_settings_fingerprint`, `build_development_fixture`, `write_json` |
| `_jsonable_metrics` | Converts nested metric values for JSON. | in-file: `_write_metric_artifact` |
| `write_json` | Writes deterministic JSON. | in-file: `_write_metric_artifact`, `build_development_fixture`, `checkpoint_readiness`, `execute_traditional_calibration` ... |
| `read_json` | Reads JSON. | in-file: `checkpoint_catalog`, `freeze_identities`, `load_frozen_research_cohort`, `load_id_baseline_summary` ... |
| `write_csv` | Writes a CSV and returns its fingerprint. | in-file: `_write_metric_artifact`, `execute_traditional_calibration` |
| `_git_head` | Current commit id. | in-file: `prepare_freeze_identity`, `run_evaluation` |
| `_hardware` | Machine and library details. | in-file: `run_evaluation` |
| `load_ready_config` | Reads and checks the sealed evaluation config. | in-file: `checkpoint_catalog`, `load_frozen_research_cohort`; tests |
| `check_research_authorization` | Allows research evaluation only after three explicit command-line choices including a confirmation phrase. | in-file: `run_evaluation`; tests |
| `freeze_identities` | Reports the cohort fingerprints and structure without computing any metric. | in-file: `load_ready_config`, `main`, `prepare_freeze_identity`; tests |
| `assert_no_frozen_result_artifacts` | Fails if any prediction file sneaked into the frozen cohort folders. | in-file: `freeze_identities` |
| `OODCohort` | A loaded cohort with its surfaces, records and fingerprints. | in-file: `_prediction_frame`, `_write_metric_artifact`, `evaluate_prediction_matrix`, `execute_traditional_calibration` ... |
| `OODCohort.truths` | The true parameters of all surfaces. | in-file: `run_evaluation`; tests |
| `_record_to_ood_item` | Turns a stored record into a surface item. | in-file: `build_development_fixture`, `load_development_fixture`, `load_frozen_research_cohort` |
| `_surface_order_hash` | Fingerprint of the surface order. | in-file: `build_development_fixture`, `load_development_fixture`, `load_frozen_research_cohort` |
| `load_development_fixture` | Loads the small non-research test fixture. | in-file: `run_evaluation`; tests |
| `load_frozen_research_cohort` | Loads the real 420 surfaces only after authorization and identity checks. | in-file: `run_evaluation`; tests |
| `_safe_readiness_root` | Refuses to write inside the frozen folders. | in-file: `checkpoint_readiness`, `materialize_traditional_subset`, `model3_readiness`, `prepare_freeze_identity` ... |
| `prepare_freeze_identity` | Writes the cohort identity record. | in-file: `main`; tests |
| `prepare_reference_scaling` | Writes scaling numbers taken from the training split only. | in-file: `main` |
| `load_reference_scaling` | Reads and checks those scaling numbers. | in-file: `run_evaluation`; tests |
| `checkpoint_catalog` | Lists the expected Model 1/2 checkpoints and fingerprints. | in-file: `checkpoint_readiness`, `run_neural_adapter` |
| `checkpoint_readiness` | Reports which checkpoints are present (locally they were absent). | in-file: `main`; tests |
| `_validate_neural_payload` | Checks a checkpoint's architecture and scaler. | in-file: `run_neural_adapter`; tests |
| `run_neural_adapter` | Runs every frozen seed on a cohort, or fails without partial results. | in-file: `run_evaluation` |
| `model3_readiness` | Reports Model 3 as waiting for frozen checkpoints. | in-file: `main`, `run_evaluation`; tests |
| `evenly_spaced_indices` | Picks evenly spaced row numbers. | in-file: `expected_traditional_selections` |
| `materialize_traditional_subset` | Fixes the 60 surfaces (15 per cohort) for traditional calibration. | in-file: `main`; tests |
| `load_traditional_subset` | Reads and checks that subset. | in-file: `execute_traditional_calibration`; tests |
| `expected_traditional_selections` | Computes which 60 surfaces the rule selects. | in-file: `load_traditional_subset`, `materialize_traditional_subset` |
| `_settings_fingerprint` | Fingerprint of the calibration settings. | in-file: `execute_traditional_calibration` |
| `_traditional_worker` | Calibrates one surface in a worker process. | in-file: `execute_traditional_calibration` |
| `execute_traditional_calibration` | Runs calibration on the subset with resume and retained failures. | in-file: `run_evaluation` |
| `_prediction_frame` | Table of predictions per surface. | in-file: `_write_metric_artifact` |
| `evaluate_prediction_matrix` | Computes recovery, validity and repricing metrics using the primary-comparison formulas. | in-file: `_write_metric_artifact`; tests |
| `load_id_baseline_summary` | Loads the normal (in-distribution) results to compare against. | in-file: `_write_metric_artifact` |
| `bootstrap_materiality` | Resampling test of whether an OOD change is meaningful. | in-file: `_write_metric_artifact`; tests |
| `degradation_against_id_baseline` | Applies the frozen "how much worse than normal" rule. | in-file: `_write_metric_artifact`; tests |
| `aggregate_neural_seeds` | Combines seeds and reports their spread. | tests |
| `_write_metric_artifact` | Writes predictions and metrics for one method/seed. | in-file: `run_evaluation` |
| `_safe_output_directory` | Refuses to write into protected or non-empty folders. | in-file: `run_evaluation` |
| `build_development_fixture` | Builds three made-up surfaces unrelated to the real truths, for testing. | in-file: `run_evaluation`; tests |
| `_artifact_hashes` | Fingerprints all result files. | in-file: `verify_result_intake` |
| `verify_result_intake` | Checks a result bundle is complete and untampered. | in-file: `run_evaluation`; tests |
| `run_evaluation` | Runs the development fixture now, or the sealed research evaluation later. | in-file: `main`; tests |
| `compare_replay` | Compares two result folders, ignoring timing files. | tests |
| `main` | Command-line entry with subcommands. | command line |

#### `src/ood_boundary_protocol.py`

*1226 lines; newest version e675d0e (2026-08-25, virajchoudhary); on research/ood-boundary-protocol.*

Builds the frozen OOD/boundary test cohorts (Aug 25): 360 complete surfaces (120 near the parameter edges, 120 outside the training range, 120 with unusual maturities) plus 60 deliberately incomplete ones, all priced once and replay-verified.

| Function / class | What it does | Used by |
|---|---|---|
| `OODProtocolError` | Error when the protocol cannot be honoured. | in-file: `RemoteCheckpointRequired`, `_mask_pattern`, `_mixed_radix_conditioning`, `_standard_conditioning` ... |
| `RemoteCheckpointRequired` | Error when generation is attempted before the protocol was pushed. | in-file: `require_remote_checkpoint` |
| `sha256_file` | Fingerprints a file. | in-file: `_write_csv`, `_write_json`, `load_protocol_config`, `run_generation` ...; `src/ood_boundary_evaluation.py` |
| `sha256_bytes` | Fingerprints bytes. | in-file: `run_generation`; `src/ood_boundary_evaluation.py` |
| `deterministic_json_bytes` | Byte-stable JSON. | in-file: `_jsonl_bytes`, `run_generation`; `src/ood_boundary_evaluation.py` |
| `_json_default` | Converts numpy values for JSON. | in-file: `_write_json`, `deterministic_json_bytes`; `src/ood_boundary_evaluation.py` |
| `_jsonl_bytes` | Many JSON lines as bytes. | in-file: `run_generation` |
| `_write_bytes` | Writes bytes to a file. | in-file: `run_generation` |
| `_write_csv` | Writes a CSV and returns its fingerprint. | in-file: `run_generation` |
| `_write_json` | Writes JSON and returns its fingerprint. | in-file: `run_generation`, `run_replay` |
| `load_protocol_config` | Loads and fully validates the frozen protocol config. | in-file: `build_parameter_pools`, `main`, `run_generation`; tests |
| `_validate_hard_bounds_against_reviewed` | Checks the limits match the reviewed ones. | in-file: `load_protocol_config` |
| `require_remote_checkpoint` | Refuses unless the protocol is committed and confirmed. | in-file: `run_generation`, `run_replay`; tests |
| `_load_frozen_parameter_hashes` | Fingerprints of the 10k dataset's parameters, to avoid overlap. | in-file: `build_parameter_pools`, `run_generation` |
| `_add_eligibility` | Marks which candidates are valid and not already in the 10k dataset. | in-file: `build_parameter_pools` |
| `build_parameter_pools` | Builds candidate pools and picks the fixed quotas per cohort. | in-file: `run_generation`; tests |
| `build_development_panel` | Builds a 12-row non-research sanity panel. | in-file: `run_generation`; tests |
| `_standard_conditioning` | Typical maturities and rates for a surface. | in-file: `_conditioning_for` |
| `_shifted_conditioning` | Unusual maturities and rates for the maturity-shift cohort. | in-file: `_conditioning_for` |
| `_mixed_radix_conditioning` | Deterministically walks through all maturity/rate combinations. | in-file: `_shifted_conditioning`, `_standard_conditioning` |
| `_conditioning_for` | Chooses standard or shifted conditioning. | in-file: `generate_clean_surfaces` |
| `_surface_provenance` | Records where a surface came from. | in-file: `generate_clean_surfaces` |
| `_arbitrage_and_shape_checks` | Checks prices are arbitrage-free and well shaped. | in-file: `generate_clean_surfaces` |
| `generate_clean_surfaces` | Prices every selected truth with the production pricer. | in-file: `run_generation` |
| `_mask_pattern` | Which slots are hidden for each incomplete pattern. | in-file: `make_incomplete_surface` |
| `make_incomplete_surface` | Makes a surface with some slots masked as missing. | in-file: `derive_incomplete_surfaces`; `src/ood_boundary_evaluation.py`; tests |
| `derive_incomplete_surfaces` | Makes the 60 incomplete surfaces. | in-file: `run_generation` |
| `_environment` | Library versions. | in-file: `run_generation` |
| `_git_head` | Current commit id. | in-file: `run_generation`; `src/ood_boundary_evaluation.py` |
| `_payload_records` | Surfaces as storable records. | in-file: `run_generation` |
| `run_generation` | Generates all cohorts after the gate passes. | in-file: `main`, `run_replay` |
| `validate_generated_output` | Checks counts, fingerprints, masks and identities. | in-file: `run_generation`, `run_replay`; `src/ood_boundary_evaluation.py` |
| `verify_replay` | Compares scientific fingerprints of two runs. | in-file: `run_replay` |
| `run_replay` | Regenerates everything and checks it is byte-identical. | in-file: `main` |
| `main` | Command-line entry with subcommands. | command line |

#### `src/pricing_interface.py`

*160 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The doorway between the network/dataset code and the real pricer, plus a clearly labelled fake pricer used only in smoke tests.

| Function / class | What it does | Used by |
|---|---|---|
| `MissingPricingEngineError` | Old error kept so older callers still work. | no caller found by search |
| `price_double_heston_surface` | Prices a surface with the real engine and checks the output is sensible. | `src/evaluate_repricing.py`, `src/synthetic_dataset.py`; tests |
| `dummy_surface_generator_for_smoke_test` | Produces fake but well-behaved prices labelled NOT_RESEARCH_DATA, only for wiring tests. | `src/synthetic_dataset.py` |
| `_validate_surface_prices` | Checks prices are finite, non-negative and within basic option bounds. | in-file: `dummy_surface_generator_for_smoke_test`, `price_double_heston_surface` |

#### `src/r2_final_generation.py`

*1276 lines; newest version f67e349 (2026-08-23, virajchoudhary); on origin/main.*

Generates the final clean 10,000-surface R2 dataset (Aug 23) behind several authorization gates, then re-reads and replays it to prove it is exactly reproducible.

| Function / class | What it does | Used by |
|---|---|---|
| `FinalGenerationAuthorizationError` | Error when generation is not explicitly authorized. | in-file: `parse_authorization_marker`, `run_final_generation`, `run_final_replay`, `verify_authorization` ...; tests |
| `FinalDatasetValidationError` | Error when the produced dataset breaks the contract. | in-file: `_generate_selected_final_surfaces`, `load_final_panel`, `load_readiness_manifest`, `run_final_generation` ...; tests |
| `_final_module_path` | This file's path. | in-file: `scientific_dependency_hashes` |
| `scientific_dependency_hashes` | Fingerprints of every frozen code and config file the dataset depends on. | in-file: `run_final_generation`, `run_final_replay`, `run_preflight`, `verify_authorization`; tests |
| `_git` | Runs a git command. | in-file: `_base_is_ancestor`, `_current_head`, `_head_on_remote`, `_marker_committed_at_head` |
| `_current_head` | Current commit id. | in-file: `run_final_generation`, `run_preflight`, `verify_authorization`, `write_authorization_marker`; tests |
| `_marker_committed_at_head` | Checks the authorization marker is committed. | in-file: `verify_authorization`; tests |
| `_base_is_ancestor` | Checks a base commit is in history. | in-file: `verify_authorization`; tests |
| `_head_on_remote` | Checks the current commit was pushed. | in-file: `verify_authorization`; tests |
| `load_final_panel` | Loads the frozen 10,000 parameter sets exactly. | in-file: `run_final_generation`, `run_final_replay`, `run_preflight`, `validate_final_dataset`; tests |
| `load_readiness_manifest` | Loads the verified readiness record. | in-file: `run_preflight`; tests |
| `_quadratic_gcd_ok` | Checks the conditioning walk visits every combination. | in-file: `run_preflight` |
| `run_preflight` | Runs every check that does not need pricing. | in-file: `main`, `run_final_generation`, `run_final_replay`, `write_authorization_marker`; tests |
| `parse_authorization_marker` | Reads the authorization file and its required statements. | in-file: `verify_authorization`; tests |
| `verify_authorization` | Refuses unless authorization is complete, committed and pushed. | in-file: `run_final_generation`, `run_final_replay`; tests |
| `write_authorization_marker` | Runs preflight and writes the authorization file. | in-file: `main`; tests |
| `_final_surface_metadata` | Metadata stored with each surface. | in-file: `_generate_selected_final_surfaces` |
| `_generate_selected_final_surfaces` | Prices selected rows in order, keeping any failures. | in-file: `run_final_generation`, `run_final_replay` |
| `run_final_generation` | Generates all 10,000 surfaces under the gates. | in-file: `main`; tests |
| `validate_final_dataset` | Re-reads the dataset from disk and checks the whole contract. | in-file: `run_final_generation`; tests |
| `predeclared_replay_indices` | Rows to replay, chosen by rule only. | in-file: `run_final_replay`; tests |
| `run_final_replay` | Re-prices the dataset (or subset) and compares. | in-file: `main`; tests |
| `main` | Command-line entry with subcommands. | command line |

#### `src/r2_synthetic_generation.py`

*1252 lines; newest version 65aab62 (2026-08-23, virajchoudhary); on origin/main.*

The frozen recipe for building the synthetic R2 training data (Aug 23): candidate pools, quotas per split, deterministic maturity/rate conditioning, a pilot and its replay, then readiness for the full 10k.

| Function / class | What it does | Used by |
|---|---|---|
| `GenerationContractError` | Error when the generation recipe cannot be honoured. | in-file: `CandidatePoolInsufficientError`, `_build_generation_cohort`, `_require_contract_freeze`, `_surface_payloads` ...; tests |
| `CandidatePoolInsufficientError` | Error when a pool has too few valid candidates for its quota. | in-file: `_build_generation_cohort`, `run_final_readiness`, `select_accepted_candidates`; tests |
| `load_generation_config` | Loads and checks the frozen generation config. | in-file: `_build_generation_cohort`, `_retain_insufficient_pool_evidence`, `config_quotas`, `generate_candidate_pools` ...; tests |
| `sha256_file` | Fingerprints a file. | in-file: `_build_generation_cohort`, `_verified_pilot_evidence`, `run_final_readiness`; `src/ood_boundary_protocol.py`; tests |
| `sha256_bytes` | Fingerprints bytes. | in-file: `_build_generation_cohort`, `_write_csv`, `_write_jsonl`, `parameter_vector_hash`; `src/ood_boundary_protocol.py`; tests |
| `_json_default` | Converts numpy values for JSON. | in-file: `deterministic_json_bytes`, `write_json`; `src/ood_boundary_protocol.py` |
| `deterministic_json_bytes` | Byte-stable JSON. | in-file: `_build_generation_cohort`, `_write_jsonl`, `parameter_vector_hash`; `src/ood_boundary_protocol.py`; tests |
| `write_json` | Writes JSON. | in-file: `_build_generation_cohort`, `_retain_insufficient_pool_evidence`, `run_final_readiness`, `run_pilot`; tests |
| `parameter_vector_hash` | Fingerprint of one parameter set. | in-file: `_build_generation_cohort`, `_surface_metadata`, `generate_selected_surfaces`, `run_final_readiness`; `src/ood_boundary_protocol.py`; tests |
| `generate_candidate_pools` | Draws the whole fixed candidate pool for each distribution. | in-file: `_build_generation_cohort`, `run_final_readiness`; tests |
| `select_accepted_candidates` | Picks exact quotas by candidate id, never redrawing. | in-file: `_build_generation_cohort`, `run_final_readiness`; tests |
| `build_conditioning` | Assigns each surface a deterministic two-expiry maturity, rate and carry setup. | in-file: `generate_selected_surfaces`; tests |
| `validate_conditioning_support` | Checks every setup is valid and within support. | in-file: `generate_selected_surfaces`; tests |
| `_surface_metadata` | Metadata stored with each surface. | in-file: `generate_selected_surfaces` |
| `generate_selected_surfaces` | Prices every selected surface, keeping failures. | in-file: `_build_generation_cohort`; tests |
| `_numerical_sanity` | Checks calls fall and puts rise with strike. | in-file: `generate_selected_surfaces` |
| `_dataframe_bytes` | Table as fixed-format CSV bytes. | in-file: `_write_csv` |
| `_write_csv` | Writes a CSV and returns its fingerprint. | in-file: `_build_generation_cohort`, `_retain_insufficient_pool_evidence`, `run_final_readiness`; `src/ood_boundary_protocol.py`; tests |
| `_write_jsonl` | Writes JSON lines and returns the fingerprint. | in-file: `_build_generation_cohort`, `_retain_insufficient_pool_evidence`, `run_final_readiness`; tests |
| `_surface_payloads` | Serializes surfaces and checks the round trip. | in-file: `_build_generation_cohort` |
| `_git_head` | Current commit id. | in-file: `_milestone_git_head`; `src/ood_boundary_protocol.py` |
| `_milestone_git_head` | Resolves the milestone branch commit even if detached. | in-file: `_build_generation_cohort`, `run_final_readiness` |
| `environment_metadata` | Library versions. | in-file: `_build_generation_cohort`, `run_final_readiness` |
| `rejection_summary` | Counts rejected candidates by reason. | in-file: `run_final_readiness` |
| `split_summary` | Counts per split and checks quotas. | in-file: `_build_generation_cohort`, `run_final_readiness` |
| `config_quotas` | Quotas for the pilot or final cohort. | in-file: `_build_generation_cohort`, `_verified_pilot_evidence`, `split_summary`; tests |
| `_retain_insufficient_pool_evidence` | Saves the whole pool when a quota cannot be met. | in-file: `_build_generation_cohort`; tests |
| `_build_generation_cohort` | Generates one cohort after ordering gates pass. | in-file: `run_generation_cohort`, `run_pilot`; tests |
| `verify_replay` | Compares two runs' fingerprints. | in-file: `run_pilot`; `src/ood_boundary_protocol.py` |
| `run_generation_cohort` | Generates a controlled cohort. | in-file: `run_pilot`; tests |
| `_require_contract_freeze` | Refuses the pilot unless the contract was frozen first. | in-file: `run_generation_cohort`, `run_pilot` |
| `_verified_pilot_evidence` | Checks the pilot and its replay are complete and matching. | in-file: `run_final_readiness`; tests |
| `run_pilot` | Runs the pilot and an independent replay. | in-file: `main`; tests |
| `run_final_readiness` | Checks the final candidate pools without pricing. | in-file: `main`; tests |
| `main` | Command-line entry with subcommands. | command line |

#### `src/research_control.py`

*923 lines; newest version 823b472 (2026-08-26, virajchoudhary); on research/repro-orchestration.*

A machine-checkable "control plane" (Aug 25-26): a registry of experiments and their dependencies, checks that recorded git identities and evidence fingerprints are real, flags stale documents, and writes a status matrix and next-day handoff.

| Function / class | What it does | Used by |
|---|---|---|
| `ResearchControlError` | Error when the control plane fails a check. | in-file: `detect_stale_documents`, `git_text`, `load_yaml`, `main` ... |
| `load_yaml` | Reads a YAML object. | in-file: `load_control_inputs`, `main` |
| `sha256_bytes` | Fingerprints bytes. | in-file: `verify_git_identities` |
| `is_sha256` | True if a value looks like a SHA-256 fingerprint. | in-file: `_validate_artifact_list`, `validate_registry`, `validate_result_intake`, `verify_git_identities` |
| `is_git_sha` | True if a value looks like a git commit id. | in-file: `validate_registry`, `validate_result_intake`, `verify_git_identities` |
| `is_hash_placeholder` | True if a value is a placeholder label rather than a hash. | in-file: `_validate_artifact_list`, `validate_registry` |
| `_require` | Adds an error message if a condition is false. | in-file: `_validate_artifact_list`, `validate_graph`, `validate_registry`, `validate_result_intake` |
| `_validate_artifact_list` | Checks a list of evidence entries. | in-file: `validate_registry` |
| `validate_registry` | Checks the experiment registry. | in-file: `validate_control_inputs`; tests |
| `validate_graph` | Checks the dependency graph. | in-file: `validate_control_inputs`; tests |
| `validate_control_inputs` | Checks registry and graph together and that they match. | in-file: `build_report`, `preflight_lane`; tests |
| `validate_result_intake` | Checks a result-intake record. | in-file: `main`; tests |
| `detect_stale_documents` | Finds documents that still describe outdated status. | in-file: `build_report`, `main`; tests |
| `git_text` | Runs a git command. | in-file: `verify_git_identities`, `verify_protected_scientific_state` |
| `read_git_blob` | Reads a file at a commit. | in-file: `detect_stale_documents`, `verify_git_identities`; tests |
| `verify_git_identities` | Checks every recorded branch and commit really exists and matches. | in-file: `build_report`, `preflight_lane`; tests |
| `verify_protected_scientific_state` | Checks no protected evidence has uncommitted changes. | in-file: `build_report`; tests |
| `build_execution_matrix` | Writes the Markdown research execution matrix. | in-file: `main`; tests |
| `build_tomorrow_handoff` | Writes the next-step handoff note. | in-file: `main`; tests |
| `build_report` | Builds the full verification report. | in-file: `main`; tests |
| `write_report` | Saves the report. | in-file: `main` |
| `load_control_inputs` | Loads registry and graph. | in-file: `main` |
| `preflight_lane` | Readiness checklist for launching one experiment, without running it. | in-file: `main`; tests |
| `build_parser` | Command-line options. | in-file: `main` |
| `main` | Command-line entry point. | command line |

#### `src/run_double_heston_validation.py`

*328 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The Aug 6 validation run of the new pricer: pricing checks, convergence, recovering known parameters from one clean surface (exact) and from a 1% noisy one (unstable), plus a 12-surface pilot.

| Function / class | What it does | Used by |
|---|---|---|
| `_load_fixture` | Loads the fixed test surface. | in-file: `main` |
| `_best_row` | Picks the calibration start with the lowest error. | in-file: `main` |
| `_parameter_vector` | Reads predicted parameters from a row. | no caller found by search |
| `_surface_frame` | Builds the table of the test surface. | in-file: `main` |
| `main` | Runs all validation steps and writes outputs/double_heston_validation. | command line |

#### `src/run_independent_pricing_benchmark.py`

*245 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Compares the production pricer against the independent slow reference on 36 fixed cases (passed 36/36).

| Function / class | What it does | Used by |
|---|---|---|
| `_load_fixture` | Loads the benchmark cases and checks tolerances. | in-file: `run_benchmark` |
| `_maturity_bucket` | Short, medium or long maturity. | in-file: `run_benchmark` |
| `_moneyness_bucket` | In, at or out of the money. | in-file: `run_benchmark`; tests |
| `_bounds` | Checks a price is inside no-arbitrage bounds. | in-file: `run_benchmark` |
| `_summary_for_node` | Error and runtime summary for 64 or 96 nodes. | in-file: `run_benchmark` |
| `run_benchmark` | Prices every case both ways and writes seven result files. | in-file: `main`; tests |
| `main` | Command-line entry point. | command line |

#### `src/run_smoke_test.py`

*118 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Quick end-to-end wiring check of the ordinary ANN using fake data only.

| Function / class | What it does | Used by |
|---|---|---|
| `run_smoke_test` | Makes fake data, trains briefly, checks outputs exist. | in-file: `main` |
| `main` | Command-line entry point. | command line |

#### `src/surface_grid.py`

*131 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Builds the original fixed 108-price grid (9 strikes x 6 maturities x calls/puts), later rejected by Stage A.

| Function / class | What it does | Used by |
|---|---|---|
| `construct_strikes` | Strikes from spot and log-moneyness. | in-file: `build_surface_grid` |
| `maturity_days_to_years` | Days to years (divide by 365). | in-file: `build_surface_grid` |
| `build_surface_grid` | Builds the grid rows in fixed order. | `src/synthetic_dataset.py`; tests |
| `flatten_surface_prices` | Lays calls then puts into one list. | tests |
| `expected_input_size` | Length of that list. | `src/dataset.py`, `src/run_smoke_test.py`; tests |
| `normalize_prices_by_spot` | Divides prices by spot, keeping masked spots zero. | `src/synthetic_dataset.py` |
| `_finite_vector` | Checks a list is finite. | in-file: `flatten_surface_prices`, `normalize_prices_by_spot` |

#### `src/synthetic_dataset.py`

*397 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Original synthetic-data generator for the 108-grid ANN (Aug 6-8), with strict separation between real-engine data and fake smoke-test data.

| Function / class | What it does | Used by |
|---|---|---|
| `assign_surface_splits` | Assigns whole surfaces to train/validation/test. | in-file: `_generate_and_save`, `assign_reviewed_distribution_splits` |
| `generate_research_dataset` | Generates surfaces with the real engine, never a fake fallback. | in-file: `generate_canonical_pilot_dataset` |
| `generate_canonical_pilot_dataset` | Generates at most 100 real-engine surfaces (the 12-surface pilot). | in-file: `main`; `src/run_double_heston_validation.py` |
| `generate_smoke_test_dataset` | Generates fake smoke-test data in a clearly labelled folder. | `src/run_smoke_test.py`; tests |
| `load_confirmed_parameter_bounds` | Loads only fully confirmed bounds. | no caller found by search |
| `load_parameter_bounds` | Loads confirmed or explicitly allowed provisional bounds. | in-file: `generate_research_dataset`, `load_confirmed_parameter_bounds` |
| `_generate_and_save` | Prices surfaces, adds optional noise, saves with labels. | in-file: `generate_research_dataset`, `generate_smoke_test_dataset` |
| `_sample_valid_parameter_vector` | Draws random parameters until valid. | in-file: `generate_research_dataset` |
| `_sample_smoke_parameter_vector` | Draws simple valid parameters for smoke tests. | in-file: `generate_smoke_test_dataset` |
| `main` | Command-line entry for the small pilot. | command line |
| `assign_reviewed_distribution_splits` | Assigns splits so challenge/OOD rows never leak into training. | tests |

#### `src/torch_double_heston.py`

*682 lines; newest version ab11b09 (2026-08-23, virajchoudhary); on origin/main.*

A PyTorch copy of the pricer that can be differentiated (needed for Model 2's repricing loss); created by Dhruva on Aug 22, and Viraj added the fast vectorized version on Aug 23 to make CPU training feasible.

| Function / class | What it does | Used by |
|---|---|---|
| `price_double_heston_surface_batch_vectorized` | (Viraj) Prices every quote of every surface in one vectorized pass, numerically identical to the loop version. | `scripts/cloud/benchmark_cloud.py`, `src/r2_primary/training.py`; tests |
| `_price_quotes_vectorized` | (Viraj) Vectorized call prices for many quotes at once. | in-file: `price_double_heston_surface_batch_vectorized` |
| `_shifted_exponent_vectorized` | (Viraj) Fingerprint exponent at shifted frequencies needed by the price formula. | in-file: `_price_quotes_vectorized` |
| `_heston_log_characteristic_exponent_vectorized` | (Viraj) Vectorized one-factor fingerprint using the stable Little Heston Trap. | in-file: `_price_quotes_vectorized`, `_shifted_exponent_vectorized` |
| `price_double_heston_surface_batch` | (Dhruva) Prices a batch of surfaces with gradients, one quote at a time. | `scripts/cloud/benchmark_cloud.py`, `src/train_pinn.py`; tests |
| `price_double_heston_surface_tensor` | (Dhruva) Prices one surface. | in-file: `price_double_heston_surface_batch` |
| `price_double_heston_option_tensor` | (Dhruva) Prices one call or put. | in-file: `price_double_heston_surface_tensor` |
| `price_double_heston_call_tensor` | (Dhruva) Prices one call. | in-file: `price_double_heston_option_tensor`, `price_double_heston_put_tensor` |
| `price_double_heston_put_tensor` | (Dhruva) Prices one put via parity. | in-file: `price_double_heston_option_tensor` |
| `double_heston_characteristic_function_tensor` | (Dhruva) Full fingerprint function. | in-file: `price_double_heston_call_tensor` |
| `heston_log_characteristic_exponent_tensor` | (Dhruva) One factor's fingerprint exponent. | in-file: `double_heston_characteristic_function_tensor` |
| `_cached_gauss_laguerre` | (Dhruva) Cached integration points. | in-file: `_gauss_laguerre_rule` |
| `_gauss_laguerre_rule` | (Dhruva) Integration points as tensors. | in-file: `price_double_heston_call_tensor`, `price_double_heston_surface_batch_vectorized` |
| `_validate_node_count` | (Dhruva) Checks the number of integration points. | in-file: `_gauss_laguerre_rule`, `price_double_heston_surface_tensor` |
| `_validate_batch_vector` | (Dhruva) Checks a per-surface input list. | in-file: `price_double_heston_surface_batch`, `price_double_heston_surface_batch_vectorized` |
| `_real_dtype` | (Dhruva) Picks float32 or float64. | in-file: `double_heston_characteristic_function_tensor`, `heston_log_characteristic_exponent_tensor`, `price_double_heston_call_tensor`, `price_double_heston_put_tensor` |
| `_complex_dtype` | (Dhruva) Matching complex type. | in-file: `double_heston_characteristic_function_tensor`, `heston_log_characteristic_exponent_tensor`, `price_double_heston_call_tensor` |
| `_scalar_tensor` | (Dhruva) Makes a checked single-number tensor. | in-file: `double_heston_characteristic_function_tensor`, `heston_log_characteristic_exponent_tensor`, `price_double_heston_call_tensor`, `price_double_heston_put_tensor` |

#### `src/train.py`

*216 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Trains the original ordinary ANN (Model 1 prototype) with early stopping on validation data.

| Function / class | What it does | Used by |
|---|---|---|
| `train_ann` | Training loop with validation-only checkpointing. | in-file: `main`; `src/run_smoke_test.py` |
| `predict_parameters` | Predicts parameters in real units. | `src/run_smoke_test.py` |
| `_evaluate_loss` | Average loss on a data loader. | in-file: `train_ann` |
| `main` | Command-line entry point. | command line |

#### `src/utils.py`

*44 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Small helpers: set all random seeds; write stable JSON.

| Function / class | What it does | Used by |
|---|---|---|
| `set_deterministic_seed` | Seeds Python, NumPy and PyTorch and asks for repeatable algorithms. | `src/run_smoke_test.py`, `src/train.py` |
| `write_json` | Writes readable, stable JSON. | `src/audit_parameter_bounds.py`, `src/evaluate_parameters.py`, `src/evaluate_repricing.py`, `src/run_double_heston_validation.py` and more |
| `_json_default` | Converts paths, numpy values and tensors for JSON. | in-file: `write_json` |

### src/dheston/

#### `src/dheston/real_market_policy.py`

*82 lines; newest version d29a437 (2026-08-23, virajchoudhary); on origin/main.*

A safety lock (Aug 23) on the imported "Archive-2" code so it can never secretly train network weights on real market data.

| Function / class | What it does | Used by |
|---|---|---|
| `RealMarketWeightUpdateQuarantineError` | The error raised when someone tries to train on real market data. | in-file: `resolve_real_market_epochs`; `src/r2_final_generation.py`; tests |
| `quarantine_error_message` | Builds the explanation of which setting triggered the lock. | in-file: `resolve_real_market_epochs` |
| `resolve_real_market_epochs` | Returns zero real-market training epochs unless explicitly approved, otherwise raises the error. | `src/r2_final_generation.py`, `train_double_heston.py`; tests |

### src/g2_r2r3/

#### `src/g2_r2r3/__init__.py`

*8 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

Package marker for the R2-vs-R3 study harness. (No functions in this file.)

#### `src/g2_r2r3/calibration.py`

*268 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

Runs the 12-start fits for one cell of the R2-vs-R3 study and saves results line by line so an interrupted run can resume.

| Function / class | What it does | Used by |
|---|---|---|
| `scaled_coordinates` | Rescales parameters to 0-1 within their allowed ranges. | in-file: `fit_cell` |
| `clean_observables` | Noise-free stock-normalized prices for a truth on a layout. | in-file: `observed_observables`; `src/g2_r2r3/jacobian.py`; tests |
| `observed_observables` | The same prices with the study's fixed, keyed noise added. | in-file: `fit_cell`; tests |
| `fit_cell` | Fits one truth/layout/noise cell from 12 starts and returns one record per start. | `scripts/run_g2_r2r3_matrix.py`, `scripts/run_g2_r2r3_smoke.py`; tests |
| `ResultLog` | An append-only results diary that remembers which runs are already done. | `scripts/run_g2_r2r3_matrix.py`; tests |
| `ResultLog.__init__` | Opens the diary and reads which runs are already complete. | Python (automatic) |
| `ResultLog.cell_complete` | Says whether all 12 starts of a cell are already recorded. | `scripts/run_g2_r2r3_matrix.py`; tests |
| `ResultLog.append` | Adds run records to the diary. | in-file: `fit_cell`; `scripts/run_g2_r2r3_matrix.py`, `scripts/run_g2_r2r3_smoke.py`, `src/dheston/data/synthetic.py`, `src/g2_r2r3/jacobian.py` and more; tests |
| `ResultLog.append_event` | Adds a non-run event (like a start or stop) to the diary. | `scripts/run_g2_r2r3_matrix.py` |

#### `src/g2_r2r3/clusters.py`

*190 lines; newest version 5d74ead (2026-08-22, virajchoudhary); on origin/main.*

Groups the near-perfect fits of each cell into clusters to measure how many genuinely different answers fit equally well.

| Function / class | What it does | Used by |
|---|---|---|
| `complete_linkage_clusters` | Groups solutions so every member of a group is within a cutoff of every other. | in-file: `dispersion_record` |
| `_scaled_matrix` | Pulls the scaled parameter errors out of a table. | in-file: `classify_basin`, `dispersion_record` |
| `classify_basin` | Describes the near-equal solution set (count, clusters, main direction of spread). | in-file: `dispersion_record` |
| `dispersion_record` | Spread and cluster statistics for the 12 runs of one cell. | `scripts/run_g2_r2r3_diagnostics.py`; tests |
| `aggregate_dispersion` | Combines cell statistics into R2-level and R3-level summaries. | `scripts/run_g2_r2r3_diagnostics.py`; tests |

#### `src/g2_r2r3/decision.py`

*167 lines; newest version 48248bf (2026-08-22, virajchoudhary); on origin/main.*

Applies the decision rule written before the study, in order and unchanged, to choose R2 or R3.

| Function / class | What it does | Used by |
|---|---|---|
| `_improvement` | Fractional improvement of a candidate over a baseline. | in-file: `comparative_assessment` |
| `comparative_assessment` | Labels R3 versus R2 as strong, partial or no material improvement at each noise level. | `scripts/run_g2_r2r3_diagnostics.py`; tests |
| `practical_non_identifiability` | Decides whether parameters stay far off even when prices fit to noise level. | `scripts/run_g2_r2r3_diagnostics.py` |
| `apply_frozen_decision` | Applies the freeze rules once and returns the final label (R2 frozen, non-identifiability retained). | `scripts/run_g2_r2r3_diagnostics.py`; tests |

#### `src/g2_r2r3/frozen.py`

*202 lines; newest version 48248bf (2026-08-22, virajchoudhary); on origin/main.*

All the study's fixed numbers (seeds, thresholds, noise levels, slot counts) in one place, written before results (no functions). (No functions in this file.)

#### `src/g2_r2r3/geometry.py`

*134 lines; newest version 48248bf (2026-08-22, virajchoudhary); on origin/main.*

Defines which price "slots" (expiry rank x strike distance x call/put) make up R2 (20) and R3 (30).

| Function / class | What it does | Used by |
|---|---|---|
| `DateProfile` | A real date's market setup: spot, expiries, days to expiry, rates and carries. | in-file: `profile_for_truth`, `representation_slots`; `scripts/run_g2_r2r3_diagnostics.py`, `scripts/run_g2_r2r3_matrix.py`, `src/g2_r2r3/calibration.py`, `src/g2_r2r3/market.py`; tests |
| `DateProfile.summary` | Turns the profile into a dictionary. | `scripts/run_g2_r2r3_diagnostics.py` |
| `SlotSpec` | One price slot: expiry rank, strike distance, option type, maturity, rate, carry. | in-file: `build_geometry`, `representation_slots` |
| `SlotSpec.key` | A short identifier for the slot. | in-file: `build_geometry`; `scripts/run_g2_r2r3_diagnostics.py`, `scripts/run_g2_r2r3_smoke.py`, `src/g2_r2r3/calibration.py`, `src/g2_r2r3/market.py`; tests |
| `representation_slot_count` | 20 for R2, 30 for R3. | in-file: `representation_slots` |
| `representation_expiry_ranks` | Expiry ranks 1-2 for R2, 1-3 for R3. | in-file: `representation_slots` |
| `representation_slots` | Lists all slots of a layout for a date in a fixed order. | `scripts/run_g2_r2r3_diagnostics.py`, `src/g2_r2r3/calibration.py`; tests |
| `build_geometry` | Turns slots into strike, maturity, type, rate and carry arrays for pricing. | `src/g2_r2r3/calibration.py`, `src/g2_r2r3/jacobian.py`; tests |
| `profile_for_truth` | Assigns each truth a date profile (truth number mod 5). | `scripts/run_g2_r2r3_diagnostics.py`, `scripts/run_g2_r2r3_matrix.py`, `scripts/run_g2_r2r3_smoke.py` |

#### `src/g2_r2r3/jacobian.py`

*110 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

Measures local information: how strongly prices react to each parameter and which combinations barely move prices.

| Function / class | What it does | Used by |
|---|---|---|
| `scaled_parameter_jacobian` | Sensitivity table of prices to each parameter, by small nudges. | in-file: `full_jacobian_record` |
| `jacobian_summary` | Condition number, smallest singular value and practical rank of that table. | in-file: `full_jacobian_record`; `scripts/run_g2_r2r3_diagnostics.py` |
| `weakest_directions` | The parameter combinations that change prices least. | in-file: `full_jacobian_record` |
| `parameter_sensitivities` | Overall sensitivity size per parameter. | in-file: `full_jacobian_record` |
| `full_jacobian_record` | All of the above for one truth and layout. | `scripts/run_g2_r2r3_diagnostics.py`; tests |

#### `src/g2_r2r3/market.py`

*352 lines; newest version 5d74ead (2026-08-22, virajchoudhary); on origin/main.*

Audits the five real NTPC dates: which R2/R3 slots have an active, correctly matched real quote; nothing missing is filled in.

| Function / class | What it does | Used by |
|---|---|---|
| `_assert_rate_fidelity` | Checks the hard-coded RBI rates equal the sealed official ones. | in-file: `audit_date` |
| `_raw_paths` | Paths to a date's NSE cash and F&O files. | in-file: `audit_date` |
| `_implied_volatility` | Black implied volatility from the futures price. | in-file: `audit_date` |
| `_assign_targets` | Matches real strikes to target strike distances (Hungarian assignment, 0.05 limit). | in-file: `audit_date` |
| `audit_date` | Full audit of one date, recording every failure reason. | in-file: `audit_all_dates`; `scripts/run_g2_r2r3_smoke.py`, `src/r2_representation/real.py`; tests |
| `audit_all_dates` | Audits all five dates and joins the slot tables. | `scripts/run_g2_r2r3_market_support.py` |
| `date_profiles` | Turns audit results into date profiles for the synthetic study. | `scripts/run_g2_r2r3_diagnostics.py`, `scripts/run_g2_r2r3_market_support.py`, `scripts/run_g2_r2r3_smoke.py` |

#### `src/g2_r2r3/noise.py`

*77 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

Deterministic noise: each slot's noise is derived from a hash of its identity, so R2 and R3 share identical noise on shared slots.

| Function / class | What it does | Used by |
|---|---|---|
| `slot_seed` | Makes a repeatable seed from the slot's identity. | in-file: `slot_noise_factor` |
| `slot_noise_factor` | The multiplicative factor (1 + level x random) for one slot. | in-file: `perturb_slot`; tests |
| `perturb_slot` | Applies that factor to a clean price, failing if it goes negative. | no caller found by search |

#### `src/g2_r2r3/pricer.py`

*187 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

A faster vectorized copy of the pricer used only for this study, checked to match the main engine exactly (max difference 0.0).

| Function / class | What it does | Used by |
|---|---|---|
| `_gauss_laguerre` | Integration points and weights. | in-file: `price_surface_fast` |
| `price_surface_fast` | Prices a whole surface at once. | in-file: `price_surface_per_carry` |
| `price_surface_per_carry` | Prices a surface where each expiry has its own rate and carry. | in-file: `normalized_observables` |
| `normalized_observables` | Prices divided by spot, in the study's convention. | `src/g2_r2r3/calibration.py`; tests |

#### `src/g2_r2r3/starts.py`

*42 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

The fixed list of 12 starting guesses per cell.

| Function / class | What it does | Used by |
|---|---|---|
| `start_seed` | Repeatable seed per truth and noise level. | in-file: `start_schedule_for_cell`; tests |
| `deterministic_starts` | The 12 named starting points. | in-file: `start_schedule_for_cell` |
| `start_schedule_for_cell` | Starting points for one cell. | `scripts/run_g2_r2r3_matrix.py`, `scripts/run_g2_r2r3_smoke.py`; tests |

#### `src/g2_r2r3/truths.py`

*110 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

Builds the frozen 20 true parameter sets: the four standing G2 cases plus 16 seeded draws.

| Function / class | What it does | Used by |
|---|---|---|
| `additional_truths` | The 16 seeded extra truths. | in-file: `truth_panel`; tests |
| `truth_panel` | The full 20-truth table with ids and provenance. | `scripts/run_g2_r2r3_diagnostics.py`, `scripts/run_g2_r2r3_matrix.py`, `scripts/run_g2_r2r3_merge.py`, `scripts/run_g2_r2r3_smoke.py` and more; tests |
| `panel_vectors` | Truths as a dictionary of parameter vectors. | no caller found by search |
| `load_bounds` | Loads the hard safety limits. | `src/g2_r2r3/calibration.py` |

### src/r2_representation/

#### `src/r2_representation/__init__.py`

*101 lines; newest version b9fb0c7 (2026-08-22, virajchoudhary); on origin/main.*

Package front door for the frozen R2 surface format (20 price slots plus a missing-data mask). (No functions in this file.)

#### `src/r2_representation/contract.py`

*203 lines; newest version b9fb0c7 (2026-08-22, virajchoudhary); on origin/main.*

Defines the exact identity and order of the 20 R2 slots and rejects anything else, including the old 108-number grid.

| Function / class | What it does | Used by |
|---|---|---|
| `RepresentationContractError` | Error when data breaks the R2 format. | in-file: `canonical_slot_keys`, `slot_index`, `validate_slot_keys`, `validate_vector_length`; `src/r2_representation/__init__.py`, `src/r2_representation/real.py`, `src/r2_representation/serialization.py`, `src/r2_representation/surface.py` and more |
| `SlotKey` | Identity of one slot: expiry rank, strike distance, call or put. | in-file: `canonical_slot_keys`, `slot_index`, `validate_slot_keys`; `src/r2_representation/__init__.py`, `src/r2_representation/real.py`, `src/r2_representation/serialization.py`, `src/r2_representation/surface.py` and more; tests |
| `canonical_slot_keys` | The 20 slot identities in fixed order. | `src/r2_representation/__init__.py` |
| `slot_index` | Position of a slot in that order. | `src/r2_representation/__init__.py`; tests |
| `_length_diagnostic` | Explains a wrong length, naming the rejected 108-grid if relevant. | in-file: `validate_slot_keys`, `validate_vector_length` |
| `validate_slot_keys` | Checks a slot list is exactly the canonical one. | `src/r2_representation/__init__.py`, `src/r2_representation/serialization.py`, `src/r2_representation/surface.py` |
| `validate_vector_length` | Checks a per-slot list has exactly 20 values. | `src/r2_representation/__init__.py`, `src/r2_representation/serialization.py`, `src/r2_representation/surface.py` |

#### `src/r2_representation/real.py`

*176 lines; newest version b9fb0c7 (2026-08-22, virajchoudhary); on origin/main.*

Builds a real NTPC R2 surface from official NSE data for a development date, masking anything missing.

| Function / class | What it does | Used by |
|---|---|---|
| `RealSurfaceNotConstructibleError` | Error when a date cannot produce a real surface. | in-file: `build_real_surface`; `src/r2_representation/__init__.py` |
| `build_real_surface` | Builds one masked real surface for an NTPC date. | `src/r2_representation/__init__.py` |
| `canonical_key_for_slots` | The canonical slot identities, for parity checks. | no caller found by search |

#### `src/r2_representation/serialization.py`

*307 lines; newest version ccba453 (2026-08-22, virajchoudhary); on origin/main.*

Saves and loads R2 surfaces as versioned JSON, rejecting any drift or non-finite numbers.

| Function / class | What it does | Used by |
|---|---|---|
| `surface_to_payload` | Turns a surface into a JSON-safe dictionary. | in-file: `dataset_manifest`, `write_surface_json`; `src/g8_readiness/harness.py`, `src/g8_readiness/manifests.py`, `src/ood_boundary_evaluation.py`, `src/r2_representation/__init__.py` |
| `validate_payload` | Checks a stored dictionary against the schema. | in-file: `payload_to_surface`; `src/ood_boundary_evaluation.py`, `src/r2_representation/__init__.py` |
| `payload_to_surface` | Rebuilds a surface from a checked dictionary. | in-file: `manifest_from_payload`, `read_surface_json`; `src/g8_readiness/harness.py`, `src/ood_boundary_evaluation.py`, `src/r2_primary/dataset.py`, `src/r2_representation/__init__.py` |
| `write_surface_json` | Saves one surface. | `src/r2_representation/__init__.py` |
| `read_surface_json` | Loads one surface. | `src/r2_representation/__init__.py` |
| `dataset_manifest` | Builds a manifest over many surfaces. | in-file: `write_manifest_json`; `src/r2_representation/__init__.py` |
| `manifest_from_payload` | Rebuilds surfaces from a manifest. | in-file: `read_manifest_json`; `src/r2_representation/__init__.py` |
| `write_manifest_json` | Saves a manifest. | `src/r2_representation/__init__.py` |
| `read_manifest_json` | Loads a manifest, checking every surface. | `src/r2_representation/__init__.py` |
| `_load_json` | Reads JSON, refusing NaN/Infinity. | in-file: `read_manifest_json`, `read_surface_json` |
| `_payload_slot_keys` | Reads slot identities from stored data. | in-file: `manifest_from_payload`, `payload_to_surface`, `validate_payload` |
| `_require_serializable` | Checks a dictionary is finite-number JSON. | in-file: `dataset_manifest`, `surface_to_payload` |
| `_write_json` | Writes deterministic JSON. | in-file: `write_manifest_json`, `write_surface_json` |

#### `src/r2_representation/surface.py`

*406 lines; newest version ccba453 (2026-08-22, virajchoudhary); on origin/main.*

The R2 surface object itself: 20 normalized prices, a true/false "usable" mask, and per-slot maturity, rate and carry.

| Function / class | What it does | Used by |
|---|---|---|
| `R2Conditioning` | The market setup of a surface: spot, two expiries, days to expiry, rates, carries. | `src/r2_representation/__init__.py`, `src/r2_representation/synthetic.py` |
| `R2Conditioning.__post_init__` | Checks the setup values are valid. | Python (automatic) |
| `R2Conditioning.maturities` | Time to maturity in years per expiry. | in-file: `R2Surface`, `R2Surface.__post_init__`, `R2Surface.maturities_array`, `surface_from_vectors`; `src/g8_readiness/harness.py`, `src/g8_readiness/surfaces.py`, `src/r2_representation/real.py`, `src/r2_representation/serialization.py` and more; tests |
| `R2Conditioning.from_date_profile` | Builds the setup from a G2 date profile. | no caller found by search |
| `R2Surface` | One R2 surface with mask, never filling missing quotes. | in-file: `surface_from_vectors`; `src/g8_readiness/harness.py`, `src/g8_readiness/manifests.py`, `src/g8_readiness/pipeline.py`, `src/g8_readiness/surfaces.py` and more |
| `R2Surface.__post_init__` | Checks slots, lengths, mask, and that masked prices are exactly zero. | Python (automatic) |
| `R2Surface._per_rank_rank_slices` | Slot positions per expiry rank. | in-file: `R2Surface`, `R2Surface.__post_init__` |
| `R2Surface.prices_array` | Prices as an array. | in-file: `R2Surface`, `R2Surface.denormalized_prices_array`, `R2Surface.valid_prices_array` |
| `R2Surface.mask_array` | Mask as an array. | in-file: `R2Surface`, `R2Surface.denormalized_prices_array`, `R2Surface.valid_prices_array`; `src/g8_readiness/pipeline.py`; tests |
| `R2Surface.maturities_array` | Maturities as an array. | no caller found by search |
| `R2Surface.rates_array` | Rates as an array. | no caller found by search |
| `R2Surface.carries_array` | Carries as an array. | no caller found by search |
| `R2Surface.valid_prices_array` | Only the usable prices. | no caller found by search |
| `R2Surface.denormalized_prices_array` | Prices times spot, masked slots stay zero. | no caller found by search |
| `R2Surface.valid_slot_keys` | Identities of usable slots. | no caller found by search |
| `R2Surface.masked_slot_keys` | Identities of missing slots. | no caller found by search |
| `R2Surface.usable_slot_count` | How many slots are usable. | `src/r2_representation/real.py` |
| `R2Surface.slot_count` | Always 20. | no caller found by search |
| `_require_finite_positive` | Checks a number is finite and positive. | in-file: `R2Conditioning`, `R2Conditioning.__post_init__`, `R2Surface`, `R2Surface.__post_init__` |
| `_require_mask_boolean` | Accepts only real true/false values in the mask. | in-file: `R2Surface`, `R2Surface.__post_init__` |
| `normalize_metadata_mapping` | Checks metadata is JSON-safe and returns a copy. | in-file: `_prepared_stored_metadata`; `src/r2_representation/serialization.py`, `src/r2_representation/synthetic.py` |
| `_immutable_metadata` | Freezes metadata so it cannot change. | in-file: `_prepared_stored_metadata` |
| `_prepared_stored_metadata` | Checks then freezes metadata. | in-file: `R2Surface`, `R2Surface.__post_init__` |
| `surface_from_vectors` | Builds a surface from plain lists with full checks. | `src/g8_readiness/surfaces.py`, `src/r2_representation/__init__.py`; tests |

#### `src/r2_representation/synthetic.py`

*159 lines; newest version ccba453 (2026-08-22, virajchoudhary); on origin/main.*

Builds a complete synthetic R2 surface from a parameter set using the production pricer.

| Function / class | What it does | Used by |
|---|---|---|
| `build_synthetic_surface` | Prices all 20 slots for given parameters and market setup. | `src/r2_representation/__init__.py` |

### src/r2_primary/

#### `src/r2_primary/__init__.py`

*1 lines; newest version ab11b09 (2026-08-23, virajchoudhary); on origin/main.*

Package marker for the main R2 comparison. (No functions in this file.)

#### `src/r2_primary/calibration.py`

*229 lines; newest version 3486ca2 (2026-08-23, virajchoudhary); on origin/main.*

Runs traditional calibration (3 starts, 300 evaluations each) on all 1,250 test surfaces in parallel, saving after each surface.

| Function / class | What it does | Used by |
|---|---|---|
| `calibrate_surface` | Calibrates one surface in a worker process. | in-file: `run_traditional_calibration` |
| `run_traditional_calibration` | Calibrates every surface with resume support. | in-file: `main`; `src/r2_noise/traditional_runner.py` |
| `select_representatives` | Picks each surface's answer: the start with the lowest final error (ties to the lowest start number). | in-file: `main`; `src/ood_boundary_evaluation.py`, `src/r2_noise/aggregation.py`, `src/r2_noise/traditional_runner.py`, `src/r2_primary/final_evaluation.py`; tests |
| `main` | Command-line entry point. | command line |

#### `src/r2_primary/dataset.py`

*265 lines; newest version ab11b09 (2026-08-23, virajchoudhary); on origin/main.*

Loads the frozen 10k R2 dataset and builds the 100-number input each network sees.

| Function / class | What it does | Used by |
|---|---|---|
| `R2DatasetError` | Error for any record that breaks the rules. | in-file: `R2PrimaryDataset`, `R2PrimaryDataset.__init__`, `_record_to_item`, `_surface_split` ...; tests |
| `build_r2_features` | Builds the 100-number input: 20 normalized prices, 20 masks, and maturities, rates and carries per slot. | in-file: `_record_to_item`; `src/g8_readiness/harness.py`, `src/ood_boundary_evaluation.py`; tests |
| `_surface_split` | Reads a surface's split (train/validation/test). | in-file: `R2PrimaryDataset`, `R2PrimaryDataset.from_jsonl`, `_record_to_item` |
| `_surface_targets` | Reads its ten true parameters. | in-file: `_record_to_item` |
| `R2SurfaceItem` | One surface with everything the comparison needs. | in-file: `R2PrimaryDataset`, `R2PrimaryDataset.__getitem__`, `R2PrimaryDataset.__init__`, `_record_to_item` ...; `src/ood_boundary_evaluation.py`, `src/r2_primary/calibration.py`; tests |
| `_record_to_item` | Turns a stored record into an item, rejecting non-R2 or non-synthetic data. | in-file: `R2PrimaryDataset`, `R2PrimaryDataset.from_jsonl`; `scripts/cloud/benchmark_cloud.py`; tests |
| `iter_r2_jsonl` | Reads records one by one. | in-file: `R2PrimaryDataset`, `R2PrimaryDataset.from_jsonl`; `scripts/cloud/benchmark_cloud.py` |
| `R2PrimaryDataset` | PyTorch dataset of R2 surfaces. | in-file: `assert_split_isolation`; `scripts/run_model3_pde_pilot.py`, `src/model3_evaluation/adapter.py`, `src/model3_evaluation/harness.py`, `src/ood_boundary_evaluation.py` and more |
| `R2PrimaryDataset.__init__` | Stacks features and targets, checking ids are unique. | Python (automatic) |
| `R2PrimaryDataset.from_jsonl` | Loads from file, optionally only some splits. | `scripts/run_model3_pde_pilot.py`, `src/model3_evaluation/harness.py`, `src/ood_boundary_evaluation.py`, `src/r2_noise/aggregation.py` and more |
| `R2PrimaryDataset.__len__` | Number of surfaces. | Python (automatic) |
| `R2PrimaryDataset.__getitem__` | One surface's features, targets, id and item. | Python (automatic) |
| `R2PrimaryDataset.indices_for_split` | Row numbers in a split. | `scripts/run_model3_pde_pilot.py`, `src/model3_evaluation/harness.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/calibration.py` and more |
| `R2PrimaryDataset.split_counts` | How many surfaces per split. | `src/model3_evaluation/harness.py`, `src/r2_primary/final_evaluation.py`, `src/run_pinn_improved_benchmark.py`, `src/run_pinn_synthetic_baseline.py` and more; tests |
| `R2PrimaryDataset.subset` | A dataset with only chosen rows. | `scripts/run_model3_pde_pilot.py`, `src/ood_boundary_evaluation.py`, `src/r2_noise/traditional_runner.py`, `src/r2_primary/calibration.py` |
| `assert_split_isolation` | Fails if any surface or parameter set appears in two splits. | `src/model3_evaluation/harness.py`, `src/r2_primary/final_evaluation.py`; tests |

#### `src/r2_primary/evaluation.py`

*363 lines; newest version 9bf1d49 (2026-08-23, virajchoudhary); on origin/main.*

The frozen scorecard used for every method: parameter recovery, rule validity, repricing, "fit-versus-truth", stability and runtime.

| Function / class | What it does | Used by |
|---|---|---|
| `reprice_normalized` | Re-prices predicted parameters with the production engine. | in-file: `summarize_run`; `src/ood_boundary_evaluation.py`, `src/r2_noise/aggregation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py`; tests |
| `train_split_scaling` | Ranges, means and spreads of each parameter on training data. | `src/model3_evaluation/harness.py`, `src/ood_boundary_evaluation.py`, `src/r2_noise/aggregation.py`, `src/r2_noise/neural_evaluation.py` and more; tests |
| `parameter_recovery_metrics` | How close predicted parameters are to the truth (RMSE, per-parameter errors, success rates). | in-file: `summarize_run`; `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py`; tests |
| `_factor_swap_confusion_rate` | How often a prediction is closer to the slow/fast-swapped truth. | in-file: `parameter_recovery_metrics` |
| `constraint_validity_metrics` | How often predictions obey every rule. | in-file: `summarize_run`; `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py`; tests |
| `repricing_metrics` | How close re-priced prices are to observed ones. | in-file: `identifiability_aware_metrics`, `summarize_run`; `src/model3_evaluation/harness.py`, `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py`; tests |
| `identifiability_aware_metrics` | Of the surfaces that re-price well, how many also have correct parameters (the key "fit is not recovery" test). | in-file: `summarize_run`; `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py` |
| `_range_scaled_error_matrix` | Errors divided by each parameter's range. | in-file: `identifiability_aware_metrics`; `src/ood_boundary_evaluation.py`, `src/r2_noise/aggregation.py`, `src/r2_noise/neural_evaluation.py` |
| `stability_metrics` | How much predictions vary between seeds. | `src/model3_evaluation/harness.py`, `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py`; tests |
| `measure_inference_runtime` | Milliseconds per surface for a network. | `src/model3_evaluation/harness.py`, `src/r2_primary/final_evaluation.py` |
| `summarize_run` | All metric families for one method. | `src/model3_evaluation/harness.py`, `src/r2_primary/final_evaluation.py` |

#### `src/r2_primary/final_evaluation.py`

*381 lines; newest version 285a724 (2026-08-24, virajchoudhary); on origin/main.*

The one-time final test (Aug 24) that opened the untouched 1,250 test surfaces and produced the headline Model 1 vs Model 2 vs traditional numbers.

| Function / class | What it does | Used by |
|---|---|---|
| `_sha256` | Fingerprints a file. | in-file: `_protocol_identity`, `main` |
| `_predicted_matrix_from_traditional` | Traditional answers as a matrix. | in-file: `main` |
| `_traditional_multi_start_dispersion` | How much the three traditional starts disagree per surface. | in-file: `main`; `src/r2_noise/aggregation.py` |
| `_protocol_identity` | Fingerprint of the frozen protocol config. | in-file: `main`; tests |
| `main` | Loads everything, checks isolation, scores all methods, writes the evidence bundle. | command line |

#### `src/r2_primary/training.py`

*671 lines; newest version f0d2878 (2026-08-23, virajchoudhary); on origin/main.*

Trains Model 1 (plain ANN) and Model 2 (ANN with built-in rules plus a "re-price and compare" loss) on the 7,500 training surfaces, picking checkpoints by validation only.

| Function / class | What it does | Used by |
|---|---|---|
| `_git_sha` | Current commit id. | in-file: `train_model1`, `train_model2` |
| `_environment_summary` | Library and GPU details. | in-file: `train_model1`, `train_model2` |
| `build_model1` | Creates the plain ANN with 100 inputs. | in-file: `load_run`, `train_model1`; `src/g8_readiness/harness.py` |
| `build_model2` | Creates the constraint-plus-repricing network with 100 inputs. | in-file: `load_run`, `train_model2`; `scripts/cloud/benchmark_cloud.py`, `src/g8_readiness/harness.py` |
| `_repricing_loss` | Re-prices the network's predicted parameters with the differentiable pricer and measures the price gap. | in-file: `_model2_validation_metrics`, `train_model2`; `scripts/cloud/benchmark_cloud.py`; tests |
| `train_model1` | Trains Model 1 with early stopping on validation. | in-file: `main`; tests |
| `_model1_validation_loss` | Model 1's validation loss. | in-file: `train_model1` |
| `train_model2` | Trains Model 2 with parameter and repricing losses. | in-file: `main`; tests |
| `_model2_validation_metrics` | Model 2's validation losses. | in-file: `train_model2` |
| `predict_parameters` | Predicts real-unit parameters for chosen surfaces. | `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py` |
| `load_run` | Loads a best checkpoint with its provenance. | `src/ood_boundary_evaluation.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_primary/final_evaluation.py` |
| `_build_argument_parser` | Command-line options. | in-file: `main`; tests |
| `main` | Command-line entry: trains one model and seed, refusing to overwrite. | command line |

### src/r2_noise/

#### `src/r2_noise/__init__.py`

*8 lines; newest version 91b66b6 (2026-08-24, virajchoudhary); on codex/r2-noise-recovery.*

Package marker for the noise-robustness study. (No functions in this file.)

#### `src/r2_noise/aggregation.py`

*234 lines; newest version 46327f2 (2026-08-25, virajchoudhary); on codex/r2-noise-recovery.*

Combines the network and traditional results across noise levels into headline and "how much worse with noise" tables.

| Function / class | What it does | Used by |
|---|---|---|
| `_observed` | Observed prices with masked slots set to blank. | in-file: `evaluate_traditional_levels` |
| `evaluate_traditional_levels` | Scores the 250-surface traditional calibration runs at each noise level. | `scripts/run_r2_noise_robustness.py` |
| `paired_degradation` | Compares each method at each noise level against its own 0% result. | `scripts/run_r2_noise_robustness.py` |

#### `src/r2_noise/execution.py`

*156 lines; newest version b54bc6d (2026-08-24, virajchoudhary); on codex/r2-noise-recovery.*

Shared plumbing for the noise study: protocol loading, fingerprint checks and turning a clean surface into a noisy one.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256_path` | Fingerprints a file. | in-file: `assert_clean_dataset_identity`, `derivation_provenance`, `load_frozen_protocol`; `src/r2_noise/generator.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_noise/traditional_runner.py`, `src/r2_noise/validation.py` |
| `load_frozen_protocol` | Loads the frozen noise protocol after checking its fingerprint. | in-file: `_levels`; `scripts/run_r2_noise_robustness.py`, `src/r2_noise/aggregation.py`, `src/r2_noise/generator.py`, `src/r2_noise/neural_evaluation.py` and more; tests |
| `assert_clean_dataset_identity` | Fails if the clean 10k dataset is not byte-identical to the frozen one. | `src/r2_noise/aggregation.py`, `src/r2_noise/generator.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_noise/traditional_runner.py` and more; tests |
| `level_label` | Turns a noise number into its frozen label (e.g. 0.001 to "0.10%"). | `src/r2_noise/generator.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_noise/traditional_runner.py` |
| `_levels` | The frozen noise levels. | in-file: `level_label` |
| `iter_test_records` | Reads test-split records one by one. | `src/r2_noise/generator.py`, `src/r2_noise/neural_evaluation.py`, `src/r2_noise/validation.py`; tests |
| `derivation_provenance` | Records how a noisy surface was derived. | in-file: `derive_noisy_record` |
| `derive_noisy_record` | Makes the noisy version of one surface without changing the clean one. | `src/r2_noise/generator.py`, `src/r2_noise/validation.py`; tests |
| `serialize_record` | Byte-stable JSON line for a record. | `src/r2_noise/generator.py`, `src/r2_noise/validation.py`; tests |

#### `src/r2_noise/generator.py`

*99 lines; newest version b54bc6d (2026-08-24, virajchoudhary); on codex/r2-noise-recovery.*

Generates the four noisy copies (0.10%, 0.25%, 0.50%, 1.00%) of the 1,250 test surfaces.

| Function / class | What it does | Used by |
|---|---|---|
| `generate_cohorts` | Builds and saves the four noisy cohorts and their manifest. | `scripts/run_r2_noise_robustness.py`; tests |

#### `src/r2_noise/neural_evaluation.py`

*547 lines; newest version 9ec1f33 (2026-08-25, virajchoudhary); on codex/r2-noise-recovery.*

Runs the six frozen network checkpoints (Model 1 and 2, three seeds each) on every noise level and scores them.

| Function / class | What it does | Used by |
|---|---|---|
| `safe_level_label` | Turns "0.10%" into a folder-safe name. | in-file: `evaluate_neural_levels`; `src/r2_noise/aggregation.py`, `src/r2_noise/traditional_runner.py` |
| `_observed_matrix` | Observed prices with masked slots blank. | in-file: `evaluate_neural_levels` |
| `static_arbitrage_diagnostics` | Counts noisy prices that break no-arbitrage rules, without fixing them. | in-file: `evaluate_neural_levels`; tests |
| `_run_metrics` | All metric families for one prediction set. | in-file: `_seed_mean_metrics`, `evaluate_neural_levels`; `src/r2_noise/aggregation.py` |
| `_headline_row` | One headline row for a method, seed and level. | in-file: `evaluate_neural_levels`; `src/r2_noise/aggregation.py` |
| `_load_checkpoint_run` | Loads a research checkpoint and checks its provenance. | in-file: `evaluate_neural_levels` |
| `json_safe_metrics` | Converts metrics for JSON. | in-file: `evaluate_neural_levels`; `src/r2_noise/aggregation.py` |
| `_seed_mean_metrics` | Metrics for the average prediction over seeds. | in-file: `evaluate_neural_levels` |
| `_compare_zero_percent_gate` | Checks 0% results exactly reproduce the earlier primary comparison. | in-file: `evaluate_neural_levels`, `recheck_zero_percent_gate` |
| `evaluate_neural_levels` | Evaluates all six checkpoints at the requested levels and writes evidence. | `scripts/run_r2_noise_robustness.py` |
| `recheck_zero_percent_gate` | Re-checks saved 0% evidence without rerunning models. | `scripts/run_r2_noise_robustness.py` |

#### `src/r2_noise/path_compatibility.py`

*46 lines; newest version aeaea26 (2026-08-27, virajchoudhary); on research/r2-noise-traditional-overnight.*

Rule for when a stored file path may differ only because the repo was moved to another folder.

| Function / class | What it does | Used by |
|---|---|---|
| `documented_repo_relative_path` | Converts an absolute path to a repo-relative one only under approved roots. | in-file: `is_documented_worktree_relocation`; `src/r2_noise/provenance_intake.py`, `src/r2_noise/validation.py`; tests |
| `is_documented_worktree_relocation` | True only if two paths are the same file in a moved repo. | `src/r2_noise/traditional_runner.py`; tests |

#### `src/r2_noise/perturbation.py`

*123 lines; newest version 91b66b6 (2026-08-24, virajchoudhary); on codex/r2-noise-recovery.*

The frozen recipe for adding percentage noise to each quote, keyed by the quote's identity so it is always the same.

| Function / class | What it does | Used by |
|---|---|---|
| `slot_seed` | Repeatable seed from a quote's identity. | in-file: `perturb_slot`, `slot_noise_factor`; tests |
| `slot_noise_factor` | The multiplier (1 + level x random) for one quote. | tests |
| `perturb_slot` | Applies noise to one quote, redrawing if it would go negative. | in-file: `perturb_surface_prices`; tests |
| `perturb_surface_prices` | Applies noise to every quote of a surface. | `src/r2_noise/execution.py`; tests |

#### `src/r2_noise/provenance_intake.py`

*349 lines; newest version aeaea26 (2026-08-27, virajchoudhary); on research/r2-noise-traditional-overnight.*

Repairs a provenance hiccup (Aug 27): the 250-surface subset file existed with Windows vs Unix line endings, so this accepts only the two known exact forms and records the fix.

| Function / class | What it does | Used by |
|---|---|---|
| `_sha256` | Fingerprints bytes. | in-file: `_byte_identity`, `canonical_record_digest`, `ensure_canonical_subset`; tests |
| `_byte_identity` | Size, fingerprint and line-ending counts of a file. | in-file: `canonicalize_known_subset_representation` |
| `_load_json_object` | Reads JSON, rejecting byte-order marks. | in-file: `canonicalize_known_subset_representation` |
| `canonical_record_digest` | Fingerprint of the content regardless of line endings. | in-file: `canonicalize_known_subset_representation`; tests |
| `assert_regenerated_selection` | Checks the subset equals what the selection rule produces. | in-file: `canonicalize_known_subset_representation` |
| `canonicalize_known_subset_representation` | Accepts only the canonical or known historical byte form. | in-file: `ensure_canonical_subset`; tests |
| `_validate_existing_intake_evidence` | Checks an existing repair record matches. | in-file: `ensure_canonical_subset` |
| `_candidate_timestamp_utc` | The file's timestamp, checked against the historical one. | in-file: `ensure_canonical_subset` |
| `ensure_canonical_subset` | Makes sure the canonical subset file exists, keeping the old one. | `src/r2_noise/traditional_runner.py`; tests |

#### `src/r2_noise/subset.py`

*94 lines; newest version 91b66b6 (2026-08-24, virajchoudhary); on codex/r2-noise-recovery.*

Chooses the 250 test surfaces for slow traditional calibration by a fixed rule (spread evenly over v0_total and kappa_slow terciles), no randomness.

| Function / class | What it does | Used by |
|---|---|---|
| `_tercile_edges` | Cut points splitting values into thirds. | in-file: `select_traditional_subset` |
| `_cell_index` | Which third a value falls in. | in-file: `select_traditional_subset` |
| `select_traditional_subset` | Picks the 250 surfaces proportionally from the nine cells. | `src/r2_noise/provenance_intake.py`; tests |

#### `src/r2_noise/traditional_runner.py`

*219 lines; newest version aeaea26 (2026-08-27, virajchoudhary); on research/r2-noise-traditional-overnight.*

Runs traditional calibration on the 250-surface subset at one noise level, saving after each surface so a crash can resume.

| Function / class | What it does | Used by |
|---|---|---|
| `_subset_dataset` | Loads the 250 subset surfaces from a noisy cohort. | in-file: `run_traditional_subset` |
| `_provenance` | Records settings and fingerprints for the journal. | in-file: `run_traditional_subset` |
| `_is_documented_worktree_relocation` | Allows only a moved-folder path difference. | in-file: `_assert_resume_provenance_compatible` |
| `_assert_resume_provenance_compatible` | Refuses to resume if anything but the folder changed. | in-file: `run_traditional_subset`; tests |
| `run_traditional_subset` | Runs or resumes one level, never overwriting a finished bundle. | `scripts/run_r2_noise_robustness.py` |
| `compare_zero_percent_traditional_gate` | Checks 0% results match the earlier primary run for the same 250 surfaces. | `scripts/run_r2_noise_robustness.py` |

#### `src/r2_noise/validation.py`

*156 lines; newest version aeaea26 (2026-08-27, virajchoudhary); on research/r2-noise-traditional-overnight.*

Final checks that the whole noise-study bundle is complete and consistent.

| Function / class | What it does | Used by |
|---|---|---|
| `_read_json` | Reads JSON. | in-file: `validate_cohorts`, `validate_completed_bundle` |
| `_serialize_expected_with_documented_relocation` | Rebuilds sealed bytes after allowing only documented folder moves. | in-file: `validate_cohorts`; tests |
| `validate_cohorts` | Checks the four noisy cohorts (1,250 records each) match the manifest. | in-file: `validate_completed_bundle`; `scripts/run_r2_noise_robustness.py` |
| `validate_completed_bundle` | Checks cohorts, gates and aggregates all exist and agree. | `scripts/run_r2_noise_robustness.py` |

### src/model3_pde/

#### `src/model3_pde/__init__.py`

*31 lines; newest version a01ddc1 (2026-08-25, virajchoudhary); on research/mentor-dh-pinn-basics.*

Package front door for the Model 3 physics foundation. (No functions in this file.)

#### `src/model3_pde/collocation.py`

*173 lines; newest version a01ddc1 (2026-08-25, virajchoudhary); on research/mentor-dh-pinn-basics.*

Picks the points (stock price, two variances, maturity) where Model 3's physics equation is checked.

| Function / class | What it does | Used by |
|---|---|---|
| `CollocationDomain` | A box of allowed stock, variance and maturity values. | in-file: `sample_collocation_states`; `scripts/run_model3_pde_smoke.py`, `src/model3_pde/__init__.py`; tests |
| `CollocationDomain.__post_init__` | Checks every range is positive and increasing. | Python (automatic) |
| `CollocationDomain.sample` | Draws random points in the box. | in-file: `sample_collocation_states`; `src/mentor_dh_pinn/final_synthetic_eval.py` |
| `sample_collocation_states` | Draws points with a seeded generator. | `scripts/run_model3_pde_smoke.py`, `src/model3_pde/__init__.py`; tests |
| `sample_eligible_contract_slot_indices` | For each point, picks one real (unmasked) option slot of its surface. | in-file: `sample_conditioned_collocation_states`; `scripts/run_model3_pde_pilot.py`, `src/model3_pde/__init__.py`; tests |
| `sample_conditioned_collocation_states` | Draws points in a box around each observed surface's own spot and theta. | `scripts/run_model3_pde_pilot.py`, `src/model3_pde/__init__.py`; tests |

#### `src/model3_pde/losses.py`

*101 lines; newest version 02c2a2c (2026-08-25, virajchoudhary); on research/mentor-dh-pinn-basics.*

Model 3's loss pieces: physics residual, expiry payoff, no-arbitrage bounds and masked price fit.

| Function / class | What it does | Used by |
|---|---|---|
| `pde_residual_loss` | Average squared physics-equation error, scaled by price size. | `scripts/run_model3_pde_pilot.py`, `src/model3_pde/__init__.py`; tests |
| `discounted_arbitrage_bounds` | Lowest and highest sensible European option prices. | in-file: `arbitrage_boundary_loss`; `src/model3_pde/model.py` |
| `terminal_payoff_loss` | Error versus the payoff at expiry. | `src/model3_pde/__init__.py`; tests |
| `arbitrage_boundary_loss` | Penalty for prices outside the sensible bounds. | `src/model3_pde/__init__.py`; tests |
| `masked_normalized_price_loss` | Price-fit error that ignores masked (missing) slots. | `scripts/run_model3_pde_pilot.py`, `src/model3_pde/__init__.py` |

#### `src/model3_pde/model.py`

*156 lines; newest version 02c2a2c (2026-08-25, virajchoudhary); on research/mentor-dh-pinn-basics.*

Model 3's two networks: an inverse network (prices to parameters) plus a forward pricing network that must obey the Double Heston equation.

| Function / class | What it does | Used by |
|---|---|---|
| `structural_parameter_conditioning` | Keeps the eight structural parameters and drops v0 so the network must use the state variances. | in-file: `ConditionalDoubleHestonPriceNetwork`, `ConditionalDoubleHestonPriceNetwork.forward` |
| `ConditionalDoubleHestonPriceNetwork` | Forward network that outputs a price kept inside no-arbitrage bounds and exactly equal to the payoff at expiry. | in-file: `Model3PDESystem`, `Model3PDESystem.__init__`; `src/model3_pde/__init__.py` |
| `ConditionalDoubleHestonPriceNetwork.__init__` | Builds the smooth tanh layers. | Python (automatic) |
| `ConditionalDoubleHestonPriceNetwork._column` | Reshapes inputs to one value per point. | in-file: `ConditionalDoubleHestonPriceNetwork`, `ConditionalDoubleHestonPriceNetwork.forward` |
| `ConditionalDoubleHestonPriceNetwork.forward` | Computes the bounded price for each point. | no caller found by search |
| `Model3PDESystem` | The joint system: inverse network, constraint map and forward pricing network. | `scripts/run_model3_pde_pilot.py`, `src/model3_evaluation/adapter.py`, `src/model3_pde/__init__.py`; tests |
| `Model3PDESystem.__init__` | Builds the three parts in float64. | Python (automatic) |
| `Model3PDESystem.predict_parameters` | Turns a 100-number R2 surface into ten valid parameters. | `scripts/run_model3_pde_pilot.py`, `src/model3_evaluation/adapter.py`; tests |
| `Model3PDESystem.predict_prices` | Prices points with the forward network. | `scripts/run_model3_pde_pilot.py`; tests |

#### `src/model3_pde/operator.py`

*198 lines; newest version 02c2a2c (2026-08-25, virajchoudhary); on research/mentor-dh-pinn-basics.*

The exact Double Heston pricing equation (PDE) written as code, refusing to silently drop any derivative (the bug found in Archive-2).

| Function / class | What it does | Used by |
|---|---|---|
| `PDEState` | The points (stock, two variances, maturity) as differentiable tensors. | in-file: `double_heston_pde_residual`; `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/losses.py`, `src/model3_pde/__init__.py`, `src/model3_pde/collocation.py` and more; tests |
| `PDEState.__post_init__` | Checks the points are valid and differentiable. | Python (automatic) |
| `_require_same_points` | Checks all inputs have the same shape. | in-file: `double_heston_pde_residual` |
| `validate_canonical_parameters` | Checks parameters obey all rules. | in-file: `double_heston_pde_residual`; `src/model3_pde/model.py` |
| `_derivative` | Takes a derivative and raises an error instead of returning zero if it is missing. | in-file: `double_heston_pde_residual` |
| `double_heston_pde_residual` | How much a price surface violates the Double Heston equation at each point. | `scripts/run_model3_pde_smoke.py`, `src/mentor_dh_pinn/losses.py`, `src/model3_pde/__init__.py`, `src/model3_pde/losses.py`; tests |

### src/model3_evaluation/

#### `src/model3_evaluation/__init__.py`

*17 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Package front door for the sealed Model 3 evaluation layer. (No functions in this file.)

#### `src/model3_evaluation/adapter.py`

*96 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Loads a frozen Model 3 checkpoint for prediction only.

| Function / class | What it does | Used by |
|---|---|---|
| `Model3CheckpointError` | Error when a checkpoint cannot be loaded safely. | in-file: `Model3EvaluationAdapter`, `Model3EvaluationAdapter.__init__`, `Model3EvaluationAdapter.predict_parameters`; tests |
| `Model3EvaluationAdapter` | Wraps one frozen Model 3 seed so it can predict parameters. | `src/model3_evaluation/__init__.py`, `src/model3_evaluation/harness.py`; tests |
| `Model3EvaluationAdapter.__init__` | Loads and checks the checkpoint, its seed and its target scaler. | Python (automatic) |
| `Model3EvaluationAdapter.seed` | The seed this checkpoint belongs to. | `src/model3_evaluation/harness.py`; tests |
| `Model3EvaluationAdapter.predict_parameters` | Predicts the ten parameters for given surfaces. | in-file: `Model3EvaluationAdapter`; `src/model3_evaluation/harness.py`; tests |
| `Model3EvaluationAdapter.checkpoint_metadata` | The checkpoint's saved metadata. | no caller found by search |

#### `src/model3_evaluation/contracts.py`

*532 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Strict checks that a finished Model 3 Stage B run is complete and untampered, producing a three-seed "freeze manifest".

| Function / class | What it does | Used by |
|---|---|---|
| `CheckpointContractError` | Error when a run or manifest breaks the contract. | in-file: `_population_signature`, `_read_epoch_history`, `_read_frame`, `_validate_completion` ...; `scripts/verify_model3_research_checkpoints.py` |
| `sha256_file` | Fingerprints a file. | in-file: `_verify_artifact_manifest`, `build_seed_contract`, `verify_freeze_manifest`; `src/model3_evaluation/harness.py`, `src/model3_evaluation/paper_export.py` |
| `read_json` | Reads JSON, rejecting NaN/Infinity. | in-file: `_verify_artifact_manifest`, `build_seed_contract`; `src/model3_evaluation/harness.py`, `src/model3_evaluation/paper_export.py` |
| `deterministic_digest` | Fingerprints a dictionary in a fixed order. | in-file: `_population_signature`; `src/model3_evaluation/harness.py` |
| `_population_signature` | Fingerprint of the exact surfaces used in a split. | in-file: `build_seed_contract` |
| `_read_frame` | Reads a history CSV with exact columns. | in-file: `_read_batch_history`, `_read_epoch_history` |
| `_read_epoch_history` | Reads an epoch history and checks it is complete. | in-file: `build_seed_contract` |
| `_read_batch_history` | Reads a batch history. | in-file: `build_seed_contract` |
| `_tensor_equal` | Exact equality for nested values and tensors. | in-file: `build_seed_contract` |
| `_verify_artifact_manifest` | Checks every output file matches its recorded fingerprint. | in-file: `build_seed_contract` |
| `_validate_completion` | Checks the run finished properly (or stopped early by the rule). | in-file: `build_seed_contract` |
| `build_seed_contract` | Validates one seed's run and writes its identity record. | in-file: `build_freeze_manifest`; `src/model3_evaluation/__init__.py`, `src/model3_evaluation/harness.py`; tests |
| `verify_freeze_manifest` | Re-checks a saved three-seed freeze manifest. | in-file: `build_freeze_manifest`; `scripts/verify_model3_research_checkpoints.py`, `src/model3_evaluation/__init__.py`, `src/model3_evaluation/harness.py`, `src/model3_evaluation/locking.py`; tests |
| `build_freeze_manifest` | Builds the three-seed freeze manifest. | `scripts/verify_model3_research_checkpoints.py`; tests |
| `render_report` | Text report of a passed freeze. | `scripts/verify_model3_research_checkpoints.py` |

#### `src/model3_evaluation/g8_adapter.py`

*56 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Placeholder check for a future Model 3 G8 request; never runs anything.

| Function / class | What it does | Used by |
|---|---|---|
| `G8AdapterError` | Error for an invalid future G8 request. | in-file: `validate_g8_request`; tests |
| `validate_g8_request` | Checks a proposed G8 request keeps Model 3 evaluation-only. | tests |

#### `src/model3_evaluation/harness.py`

*456 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

The future, locked clean-test evaluation of frozen Model 3 against the existing Model 1, Model 2 and traditional results.

| Function / class | What it does | Used by |
|---|---|---|
| `SealedEvaluationError` | Error when results fail the intake contract. | in-file: `load_frozen_baseline_evidence`, `run_clean_evaluation` |
| `canonical_train_validation_signatures` | Fingerprints of the exact non-test surfaces. | in-file: `run_clean_evaluation` |
| `_current_git_sha` | Current commit id. | in-file: `run_clean_evaluation` |
| `_git_blob_sha` | Git's own fingerprint of a file. | in-file: `load_frozen_baseline_evidence` |
| `load_frozen_baseline_evidence` | Loads and pins the existing Model 1/2/traditional results for comparison. | in-file: `run_clean_evaluation`; tests |
| `_headline_from_summary` | Pulls headline metrics from a summary. | in-file: `run_clean_evaluation` |
| `_write_predictions` | Saves predictions with surface ids. | in-file: `run_clean_evaluation` |
| `_partial_failure` | Records a failed, partial run honestly. | in-file: `run_clean_evaluation` |
| `run_clean_evaluation` | Runs the clean-test evaluation only after a valid freeze and explicit authorization. | `scripts/run_model3_clean_evaluation.py`; tests |

#### `src/model3_evaluation/locking.py`

*47 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

The lock on the clean test set: refuses access without a valid freeze manifest and an explicit flag.

| Function / class | What it does | Used by |
|---|---|---|
| `FrozenTestLockedError` | Error raised when test access is not authorized. | in-file: `_read_json`, `require_frozen_test_authorization`; tests |
| `_read_json` | Reads the manifest safely. | in-file: `require_frozen_test_authorization` |
| `require_frozen_test_authorization` | Checks the manifest and the flag before anything opens the test set. | `src/model3_evaluation/harness.py`; tests |

#### `src/model3_evaluation/noise_adapter.py`

*64 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Placeholder check for a future Model 3 noise-robustness request.

| Function / class | What it does | Used by |
|---|---|---|
| `NoiseAdapterError` | Error for an invalid request. | in-file: `validate_noise_request`; tests |
| `validate_noise_request` | Checks the request without reading any noise results. | tests |

#### `src/model3_evaluation/ood_adapter.py`

*65 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Placeholder check for a future Model 3 OOD request.

| Function / class | What it does | Used by |
|---|---|---|
| `OODAdapterError` | Error for an invalid request. | in-file: `validate_ood_request`; tests |
| `validate_ood_request` | Checks the request without computing outcomes. | tests |

#### `src/model3_evaluation/paper_export.py`

*109 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Copies verified, complete Model 3 results into paper tables; refuses partial or tampered evidence.

| Function / class | What it does | Used by |
|---|---|---|
| `PaperExportError` | Error when export is refused. | in-file: `export_publication_tables`; tests |
| `export_publication_tables` | Exports only verified COMPLETE results, never placeholders. | in-file: `main`; `scripts/export_model3_paper_results.py`; tests |
| `main` | Command-line entry point. | command line |

### src/g8_readiness/

#### `src/g8_readiness/__init__.py`

*43 lines; newest version ee7be24 (2026-08-27, virajchoudhary); on research/g8-acquisition-readiness.*

Package front door that re-exports the G8 readiness pieces. (No functions in this file.)

#### `src/g8_readiness/acquisition.py`

*594 lines; newest version ee7be24 (2026-08-27, virajchoudhary); on research/g8-acquisition-readiness.*

Future-only code to download official NSE and RBI data for the final real-market test, locked so it cannot run before 30 Sep 2026 or without explicit authorization.

| Function / class | What it does | Used by |
|---|---|---|
| `G8AcquisitionLocked` | Error raised when downloading is attempted without authorization. | in-file: `CurrentDateAcquisitionLocked`, `_atomic_new`, `intake_official_nse`, `intake_official_rbi` ...; `src/g8_readiness/__init__.py`; tests |
| `CurrentDateAcquisitionLocked` | Error raised when today is before the allowed date window. | in-file: `verify_acquisition_gate`; `src/g8_readiness/__init__.py`, `src/g8_readiness/pipeline.py` |
| `verify_acquisition_gate` | The single gatekeeper that checks every precondition (date, authorization, protocol commit, reviews) before any download. | in-file: `intake_official_nse`, `intake_official_rbi`; `src/g8_readiness/__init__.py`, `src/g8_readiness/pipeline.py` |
| `NSEArchiveRecord` | Receipt for one downloaded NSE file: URL, time, size, fingerprints, paths. | in-file: `intake_official_nse`; `src/g8_readiness/__init__.py`, `src/g8_readiness/manifests.py`, `src/g8_readiness/pipeline.py`; tests |
| `RbiRateRecord` | Receipt for one RBI T-bill rate observation. | in-file: `intake_official_rbi`, `normalize_rbi_auction`; `src/g8_readiness/__init__.py`, `src/g8_readiness/manifests.py`, `src/g8_readiness/pipeline.py`, `src/g8_readiness/surfaces.py`; tests |
| `_sha256_bytes` | Fingerprints bytes. | in-file: `_retain_failed_archive`, `intake_official_nse`, `intake_official_rbi`, `normalize_rbi_auction` |
| `_atomic_new` | Writes a new file safely and refuses to overwrite an existing one. | in-file: `_retain_failed_archive`, `intake_official_nse`, `intake_official_rbi` |
| `nse_archive_identity` | Builds the official file name and URL for a market and date. | in-file: `intake_official_nse` |
| `_validate_udiff_bytes` | Checks the downloaded CSV is the right market, date and format. | in-file: `intake_official_nse` |
| `_validated_zip_member` | Checks the zip file is intact and contains the expected file. | in-file: `intake_official_nse` |
| `_retain_failed_archive` | Keeps a record of any rejected download instead of silently dropping it. | in-file: `intake_official_nse` |
| `_read_json` | Reads a provenance JSON object safely. | in-file: `intake_official_nse` |
| `intake_official_nse` | Downloads, validates and stores one NSE archive, but only after the gate passes. | `src/g8_readiness/__init__.py`; tests |
| `normalize_rbi_auction` | Extracts date, price and yield from an RBI auction page. | in-file: `intake_official_rbi`; `src/g8_readiness/__init__.py`, `src/g8_readiness/pipeline.py`; tests |
| `intake_official_rbi` | Stores an RBI rate observation after the gate passes, refusing conflicts. | tests |
| `_html_attribute` | Reads one data attribute from the RBI page. | in-file: `normalize_rbi_auction` |
| `_parse_iso_or_nse` | Reads a date in either ISO or NSE format. | in-file: `_validate_udiff_bytes` |

#### `src/g8_readiness/checkpoints.py`

*206 lines; newest version 746060b (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Prepares the six frozen network checkpoints (Model 1 and 2, seeds 11/22/33) for G8 without retraining or replacing them.

| Function / class | What it does | Used by |
|---|---|---|
| `_sha256_file` | Fingerprints a file. | in-file: `checkpoint_readiness_manifest`, `stage_canonical_checkpoint` |
| `_default_loader` | Loads a PyTorch checkpoint on CPU. | in-file: `checkpoint_readiness_manifest` |
| `_canonical_parameter_order` | The official ten-parameter order. | in-file: `checkpoint_readiness_manifest` |
| `_payload_model` | Maps a method label to its stored model name. | in-file: `checkpoint_readiness_manifest` |
| `checkpoint_readiness_manifest` | Reports which checkpoints are present, correct and loadable, without running them. | `src/g8_readiness/acquisition.py`, `src/g8_readiness/pipeline.py`; tests |
| `stage_canonical_checkpoint` | Copies a supplied checkpoint into place only if its fingerprint matches exactly. | tests |

#### `src/g8_readiness/contracts.py`

*162 lines; newest version 746060b (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Shared G8 rules and finance helpers (date window, discounting, carry, Black price, implied vol, slot roles).

| Function / class | What it does | Used by |
|---|---|---|
| `G8ReadinessError` | Base error for any G8 rule violation. | in-file: `discount_factor`, `forward_black_price`, `futures_implied_carry`, `implied_volatility` ...; `src/g8_readiness/__init__.py`, `src/g8_readiness/acquisition.py`, `src/g8_readiness/checkpoints.py`, `src/g8_readiness/scanner.py` and more |
| `validate_g8_valuation_date` | Rejects dates outside 30 Sep to 31 Dec 2026. | `src/g8_readiness/__init__.py`, `src/g8_readiness/acquisition.py`, `src/g8_readiness/scanner.py`, `src/g8_readiness/surfaces.py`; tests |
| `discount_factor` | Discount factor from a simple yield. | in-file: `continuous_rate`; `src/g8_readiness/__init__.py`, `src/g8_readiness/pipeline.py`, `src/g8_readiness/surfaces.py`; tests |
| `continuous_rate` | Continuous interest rate from a simple yield. | in-file: `futures_implied_carry`; `src/g8_readiness/__init__.py`, `src/g8_readiness/harness.py`, `src/g8_readiness/pipeline.py`; tests |
| `futures_implied_carry` | Rate and carry implied by the futures price. | `src/g8_readiness/__init__.py`, `src/g8_readiness/harness.py`, `src/g8_readiness/surfaces.py`; tests |
| `forward_black_price` | Black's formula from the futures price. | in-file: `implied_volatility`; `src/g8_readiness/__init__.py`, `src/g8_readiness/harness.py`, `src/g8_readiness/pipeline.py`; tests |
| `no_arbitrage_bounds` | Lowest and highest sensible option price. | in-file: `implied_volatility` |
| `implied_volatility` | Volatility that matches a market price. | `src/g8_readiness/__init__.py`, `src/g8_readiness/harness.py`, `src/g8_readiness/surfaces.py`; tests |
| `canonical_slot_roles` | Marks each of the 20 R2 slots as calibration (inner) or holdout (wings). | `src/g8_readiness/__init__.py`, `src/g8_readiness/harness.py`, `src/g8_readiness/pipeline.py`, `src/g8_readiness/surfaces.py`; tests |

#### `src/g8_readiness/harness.py`

*536 lines; newest version ebde508 (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

The future G8 evaluation machinery: fits Black-Scholes, Heston and Double Heston on inner slots, tests on wings, and runs the frozen networks in read-only mode.

| Function / class | What it does | Used by |
|---|---|---|
| `pricing_rows_from_surface` | Turns a masked real surface into rows for pricing, skipping masked slots. | `src/g8_readiness/pipeline.py`; tests |
| `_price_metrics` | Price error metrics. | in-file: `fit_pricing_family_surface` |
| `_iv_metrics` | Implied-volatility error metrics, counting failures. | in-file: `fit_pricing_family_surface` |
| `_mark_holdout_iv_failure` | Marks a fit failed if a required holdout volatility could not be computed. | in-file: `fit_pricing_family_surface` |
| `_mark_nonfinite_required_price_failure` | Marks a fit failed if a required price is not a finite number. | in-file: `fit_pricing_family_surface` |
| `_standard_heston_parameters` | Turns optimizer numbers into valid single-Heston parameters. | in-file: `fit_pricing_family_surface` |
| `fit_pricing_family_surface` | Fits all three models on inner slots and scores them on held-out wings. | `src/g8_readiness/pipeline.py` |
| `aggregate_pricing_family` | Applies the frozen winner rule across surfaces. | `src/g8_readiness/pipeline.py`; tests |
| `real_g8_traditional_starts` | Two starting points that never use any truth. | in-file: `calibrate_real_g8_traditional`; tests |
| `TraditionalRunResult` | Result bundle of a traditional calibration. | in-file: `calibrate_real_g8_traditional` |
| `calibrate_real_g8_traditional` | Two-start traditional calibration on a real surface. | `src/g8_readiness/pipeline.py` |
| `NeuralCheckpointAdapter` | Loads a frozen network only if its fingerprint matches, and only for prediction. | no caller found by search |
| `NeuralCheckpointAdapter.__init__` | Verifies fingerprint and method, then loads the network. | Python (automatic) |
| `NeuralCheckpointAdapter.predict` | Predicts ten parameters for a surface with no learning. | no caller found by search |
| `build_g8_neural_features` | Builds the 100-number network input from a real surface. | in-file: `NeuralCheckpointAdapter`, `NeuralCheckpointAdapter.predict` |

#### `src/g8_readiness/manifests.py`

*415 lines; newest version ee7be24 (2026-08-27, virajchoudhary); on research/g8-acquisition-readiness.*

Defines the "seals" (fingerprinted records) that must exist before data download and before evaluation.

| Function / class | What it does | Used by |
|---|---|---|
| `_json_bytes` | Turns a dictionary into canonical bytes. | in-file: `sha256_payload` |
| `sha256_payload` | Fingerprints a dictionary. | in-file: `build_pre_acquisition_freeze`, `build_selected_data_freeze`; `src/g8_readiness/acquisition.py`; tests |
| `artifact_identity` | Path, size and fingerprint of a file. | in-file: `build_pre_acquisition_freeze`; `src/g8_readiness/acquisition.py`, `src/g8_readiness/pipeline.py` |
| `build_pre_acquisition_freeze` | Builds the first seal, refusing if any prerequisite is missing. | `src/g8_readiness/pipeline.py` |
| `_archive_identity` | Identity summary of a downloaded archive. | in-file: `build_selected_data_freeze` |
| `_rate_identity` | Identity summary of a rate record. | in-file: `build_selected_data_freeze` |
| `build_selected_data_freeze` | Builds the second seal after the dates and stocks are chosen, before any model runs. | `src/g8_readiness/pipeline.py`; tests |
| `validate_evidence_schema` | Checks an evidence file has all required fields. | no caller found by search |

#### `src/g8_readiness/model3.py`

*106 lines; newest version ebde508 (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Decides, before any download, whether Model 3 may join G8 (only with complete frozen proof).

| Function / class | What it does | Used by |
|---|---|---|
| `Model3FreezeEvidence` | The proof bundle Model 3 would need. | in-file: `evaluate_model3_inclusion` |
| `evaluate_model3_inclusion` | Includes Model 3 only if the proof is complete and committed before acquisition. | `src/g8_readiness/acquisition.py`, `src/g8_readiness/pipeline.py`; tests |

#### `src/g8_readiness/pipeline.py`

*337 lines; newest version ee7be24 (2026-08-27, virajchoudhary); on research/g8-acquisition-readiness.*

Wires the G8 steps together, locked by default; can only replay a fully fake synthetic example end to end.

| Function / class | What it does | Used by |
|---|---|---|
| `FinalEvaluationLocked` | Error raised when the final evaluation is attempted without authorization. | in-file: `assert_final_evaluation_gate`; `scripts/run_g8_readiness_pipeline.py` |
| `assert_future_acquisition_gate` | Calls the download gatekeeper. | `scripts/run_g8_readiness_pipeline.py`; tests |
| `assert_final_evaluation_gate` | Refuses final evaluation unless authorized, sealed and on a permitted date. | `scripts/run_g8_readiness_pipeline.py`; tests |
| `_udiff_frame` | Builds a fake NSE-format table. | in-file: `_fixture_market_frames` |
| `_fixture_market_frames` | Makes fake cash and F&O data for the replay. | in-file: `run_synthetic_end_to_end_replay` |
| `_fixture_rate` | Makes a fake RBI rate record. | in-file: `run_synthetic_end_to_end_replay` |
| `_mock_pricing_functions` | Fake pricers that just return observed prices. | in-file: `run_synthetic_end_to_end_replay` |
| `_mock_inverse_pricer` | Placeholder that must never be called. | no caller found by search |
| `run_synthetic_end_to_end_replay` | Runs the whole pipeline on fake data, labelled not real and not a result. | `scripts/run_g8_readiness_pipeline.py`; tests |

#### `src/g8_readiness/scanner.py`

*139 lines; newest version ee7be24 (2026-08-27, virajchoudhary); on research/g8-acquisition-readiness.*

Chooses the two evaluation dates by scanning forward for dates where all four stocks have usable surfaces, and decides when a backup stock replaces a primary.

| Function / class | What it does | Used by |
|---|---|---|
| `ScanFailure` | Records why a stock/date failed. | in-file: `ScanResult`, `scan_common_dates` |
| `ScanResult` | Result of the scan: chosen dates, failures, flags. | in-file: `full_window_backup_replacements`, `scan_common_dates`; `src/g8_readiness/manifests.py` |
| `scan_common_dates` | Walks dates in order and stops at the first two that work for all stocks. | `src/g8_readiness/pipeline.py`; tests |
| `BackupDecision` | Records a primary-to-backup swap and why. | in-file: `full_window_backup_replacements`; `src/g8_readiness/manifests.py` |
| `full_window_backup_replacements` | Swaps a primary for its backup only if it had zero usable surfaces in the whole window. | `src/g8_readiness/pipeline.py`; tests |

#### `src/g8_readiness/state_machine.py`

*67 lines; newest version ebde508 (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Reports the earliest unmet prerequisite in the G8 workflow; never grants permission.

| Function / class | What it does | Used by |
|---|---|---|
| `G8StateTransitionError` | Error for an invalid workflow step. | no caller found by search |
| `assess_pre_acquisition_freeze` | Returns the first missing prerequisite. | tests |

#### `src/g8_readiness/surfaces.py`

*423 lines; newest version ebde508 (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Builds a masked 20-slot R2 surface from real NSE and RBI data under the frozen rules, refusing anything used in development.

| Function / class | What it does | Used by |
|---|---|---|
| `development_contract_key` | Makes an identifier for a contract used in development. | in-file: `build_g8_r2_surface`, `validate_development_contract_keys` |
| `validate_development_contract_keys` | Checks the development-contract registry format. | in-file: `build_g8_r2_surface` |
| `G8SurfaceConstructionError` | Error for surface-building failures. | in-file: `_latest_rate`, `build_g8_r2_surface`, `development_contract_key`, `validate_development_contract_keys` |
| `_positive_numeric` | Numbers in a column that are finite and positive. | in-file: `_activity_mask`, `build_g8_r2_surface` |
| `_activity_mask` | Rows with positive close, volume, open interest and trades. | in-file: `build_g8_r2_surface` |
| `_latest_rate` | Most recent official rate on or before the valuation date, never a future one. | in-file: `build_g8_r2_surface` |
| `build_g8_r2_surface` | Builds one masked real surface, recording why each slot was kept or rejected. | `src/g8_readiness/pipeline.py`; tests |

### src/mentor_dh_pinn/

#### `src/mentor_dh_pinn/__init__.py`

*0 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Empty package marker. (No functions in this file.)

#### `src/mentor_dh_pinn/collocation.py`

*366 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Draws the "check points" for the mentor forward-PINN: interior points where the physics equation is tested, expiry points where the payoff is tested, and very-low and very-high stock-price boundary points.

| Function / class | What it does | Used by |
|---|---|---|
| `_uniform` | Draws evenly spread random numbers in a range with a fixed seed. | in-file: `_sample_terminal_or_boundary_fields`, `_state_variances`, `sample_pde_points` |
| `_state_variances` | Draws slow and fast current variances inside ranges tied to each factor's theta. | in-file: `_sample_terminal_or_boundary_fields`, `sample_pde_points` |
| `_leaf` | Copies values so the network's derivatives with respect to them can be taken. | in-file: `_sample_terminal_or_boundary_fields`, `sample_pde_points` |
| `PDEPoints` | A batch of interior points plus their parameters, checked for shape and precision. | in-file: `sample_pde_points`; `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/losses.py`; tests |
| `PDEPoints.__post_init__` | Checks the batch is consistent and float64. | Python (automatic) |
| `PDEPoints.features` | Stacks the seven network inputs for the batch. | `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/losses.py` and more; tests |
| `TerminalPoints` | A batch of expiry-time points (tau exactly 0) for the payoff check. | in-file: `sample_terminal_points`; `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/losses.py` |
| `TerminalPoints.__post_init__` | Checks shapes, precision and that tau is exactly zero. | Python (automatic) |
| `TerminalPoints.fields` | The seven input columns. | in-file: `BoundaryPoints`, `BoundaryPoints.__post_init__`, `BoundaryPoints.features`, `PDEPoints` ...; `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`; tests |
| `TerminalPoints.features` | Stacks them into network input. | `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/losses.py` and more; tests |
| `BoundaryPoints` | A batch of points at a stock-price boundary with their target price. | in-file: `HighSBoundaryPoints`, `LowSBoundaryPoints`; `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/losses.py`; tests |
| `BoundaryPoints.__post_init__` | Checks shapes, precision and boundary name. | Python (automatic) |
| `BoundaryPoints.fields` | The seven input columns. | in-file: `BoundaryPoints`, `BoundaryPoints.__post_init__`, `BoundaryPoints.features`, `PDEPoints` ...; `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`; tests |
| `BoundaryPoints.features` | Stacks them into network input. | `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/losses.py` and more; tests |
| `LowSBoundaryPoints` | Boundary points where the stock is near zero (call worth zero). | in-file: `sample_low_s_boundary_points` |
| `HighSBoundaryPoints` | Boundary points where the stock is very high (call worth stock minus discounted strike). | in-file: `sample_high_s_boundary_points` |
| `_make_generator` | Creates a seeded random generator. | in-file: `_sample_terminal_or_boundary_fields`, `sample_pde_points` |
| `sample_pde_points` | Draws the interior physics check points. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/plotting.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `_sample_terminal_or_boundary_fields` | Shared drawing code for expiry and boundary points. | in-file: `sample_high_s_boundary_points`, `sample_low_s_boundary_points`, `sample_terminal_points` |
| `sample_terminal_points` | Draws expiry points with the call payoff as target. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `sample_low_s_boundary_points` | Draws low-stock boundary points. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `sample_high_s_boundary_points` | Draws high-stock boundary points. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/trainer.py`; tests |

#### `src/mentor_dh_pinn/config.py`

*338 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

The strict settings file reader for the mentor forward-PINN: rejects any unknown, missing or changed setting.

| Function / class | What it does | Used by |
|---|---|---|
| `DomainConfig` | Allowed ranges for spot, strike ratio, maturity, rate, carry and variances. | in-file: `BaselineConfig`, `baseline_config_from_mapping`, `variance_state_bounds`; `src/mentor_dh_pinn/collocation.py`, `src/mentor_dh_pinn/model.py`, `src/mentor_dh_pinn/synthetic_data.py` |
| `DomainConfig.feature_min` | Lowest value of each of the seven inputs (for scaling). | `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py` and more |
| `DomainConfig.feature_max` | Highest value of each input. | `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py` and more |
| `DomainConfig.validate` | Checks ranges make sense and boundaries lie outside the interior. | in-file: `BaselineConfig`, `BaselineConfig.validate`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `variance_state_bounds` | Allowed current-variance range for a factor given its theta. | `src/mentor_dh_pinn/collocation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/synthetic_data.py` |
| `DataConfig` | Dataset settings (counts, file names, source fingerprint). | in-file: `BaselineConfig`, `baseline_config_from_mapping` |
| `DataConfig.total_count` | Train + validation + test count. | no caller found by search |
| `DataConfig.validate` | Checks counts and sampling method. | in-file: `BaselineConfig`, `BaselineConfig.validate`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `NetworkConfig` | Network settings (7 inputs, five 128-wide tanh layers, float64). | in-file: `BaselineConfig`, `baseline_config_from_mapping` |
| `NetworkConfig.validate` | Rejects any other network shape. | in-file: `BaselineConfig`, `BaselineConfig.validate`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `LossConfig` | The four loss weights (lambdas) and formulas. | in-file: `BaselineConfig`, `baseline_config_from_mapping`; `src/mentor_dh_pinn/lambda_sweep.py`, `src/mentor_dh_pinn/multiseed_confirmation.py` |
| `LossConfig.weights` | The four weights as a dictionary. | in-file: `LossConfig`, `LossConfig.validate`; `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py` and more; tests |
| `LossConfig.validate` | In the V1 baseline, insists all four weights equal 1. | in-file: `BaselineConfig`, `BaselineConfig.validate`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `TrainingConfig` | Optimizer, batch sizes, epochs, patience and checkpoint rule. | in-file: `BaselineConfig`, `baseline_config_from_mapping`; `src/mentor_dh_pinn/lambda_sweep.py`, `src/mentor_dh_pinn/multiseed_confirmation.py` |
| `TrainingConfig.validate` | Checks those training settings. | in-file: `BaselineConfig`, `BaselineConfig.validate`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `EvaluationConfig` | Settings for the one-shot test and the figures. | in-file: `BaselineConfig`, `baseline_config_from_mapping` |
| `EvaluationConfig.validate` | Checks test is one-shot and figures are 300 dpi grayscale. | in-file: `BaselineConfig`, `BaselineConfig.validate`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `BaselineConfig` | The full experiment settings bundle. | in-file: `baseline_config_from_mapping`, `load_baseline_config`; `src/mentor_dh_pinn/__init__.py`, `src/mentor_dh_pinn/collocation.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py` and more |
| `BaselineConfig.validate` | Checks identity, seed 3407, call-only, 64 nodes and all sections. | in-file: `BaselineConfig`, `BaselineConfig.with_overrides`, `baseline_config_from_mapping` |
| `BaselineConfig.to_dict` | Settings as a plain dictionary. | no caller found by search |
| `BaselineConfig.with_overrides` | Copy with only allowed fields changed (counts, epochs, patience). | `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/synthetic_data.py`; tests |
| `_strict_values` | Rejects unknown or missing fields in a section. | in-file: `baseline_config_from_mapping` |
| `baseline_config_from_mapping` | Builds the settings from a dictionary, strictly. | in-file: `load_baseline_config`; `src/mentor_dh_pinn/trainer.py`; tests |
| `load_baseline_config` | Reads the YAML settings file. | `scripts/mentor_dh_pinn/evaluate_baseline.py`, `scripts/mentor_dh_pinn/generate_dataset.py`, `scripts/mentor_dh_pinn/make_figures.py`, `scripts/mentor_dh_pinn/run_baseline.py` and more |

#### `src/mentor_dh_pinn/evaluation.py`

*299 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Evaluates the trained mentor PINN on the sealed test split exactly once, writing a claim file first so it can never be repeated.

| Function / class | What it does | Used by |
|---|---|---|
| `EvaluationResult` | The saved result paths and metrics. | in-file: `evaluate_test_once` |
| `_rmse` | Root mean squared error. | in-file: `evaluate_test_once` |
| `_mae` | Mean absolute error. | in-file: `evaluate_test_once` |
| `_json_safe` | Converts values for JSON. | in-file: `evaluate_test_once` |
| `write_mentor_summary` | Writes the human-readable result summary. | in-file: `evaluate_test_once` |
| `evaluate_test_once` | Claims the test, computes price errors and physics diagnostics, saves results. | `scripts/mentor_dh_pinn/evaluate_baseline.py`; tests |

#### `src/mentor_dh_pinn/final_synthetic_eval.py`

*1213 lines; newest version 6a6ed0f (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Phase 3A (Aug 29): builds a brand-new synthetic holdout set and evaluates the Phase 1 and Phase 2D checkpoints on it once; no committed result.

| Function / class | What it does | Used by |
|---|---|---|
| `DiagnosticConfig` | Counts and seeds of the physics check points used in the final evaluation. | in-file: `FinalSyntheticEvalSpec`, `load_final_synthetic_eval_spec` |
| `CheckpointSetting` | One checkpoint to evaluate: run id, seed, path and loss weights. | in-file: `CheckpointIdentity`, `FinalSyntheticEvalSpec`, `_checkpoint_tuple`, `_expected_checkpoint_config` ... |
| `CheckpointSetting.weights` | Its four loss weights. | in-file: `_validate_completed_metric`, `run_final_synthetic_eval`, `validate_frozen_checkpoint_provenance`; tests |
| `FinalSyntheticEvalSpec` | The frozen Phase 3A protocol settings. | in-file: `build_common_diagnostics`, `generate_final_synthetic_dataset`, `load_final_synthetic_eval_spec`, `validate_frozen_checkpoint_provenance` |
| `FinalSyntheticDataset` | The final holdout set (inputs, reference prices, ids) with no train/test API. | in-file: `_evaluate_checkpoint`, `generate_final_synthetic_dataset`; tests |
| `FinalSyntheticDataset.__post_init__` | Checks shapes, finiteness and unique ids. | Python (automatic) |
| `FinalSyntheticDataset.size` | Number of samples. | in-file: `_evaluate_checkpoint`; tests |
| `CheckpointIdentity` | A checkpoint with its fingerprint and contents. | in-file: `_evaluate_checkpoint`, `_load_inference_model`, `validate_frozen_checkpoint_provenance`; tests |
| `CommonDiagnostics` | One shared set of physics check points for all checkpoints. | in-file: `_evaluate_checkpoint`, `build_common_diagnostics` |
| `_strict_keys` | Rejects unknown or missing fields. | in-file: `_parse_checkpoint`, `load_final_synthetic_eval_spec` |
| `_parse_checkpoint` | Reads one checkpoint entry from the config. | in-file: `load_final_synthetic_eval_spec` |
| `_checkpoint_tuple` | A checkpoint setting as a comparable tuple. | in-file: `load_final_synthetic_eval_spec`; tests |
| `load_final_synthetic_eval_spec` | Loads and strictly validates the Phase 3A config. | in-file: `run_final_synthetic_eval`, `summarize_final_synthetic_eval`; tests |
| `_resolve` | Makes a path absolute. | in-file: `run_final_synthetic_eval`, `summarize_final_synthetic_eval` |
| `_sha256_file` | Fingerprints a file. | in-file: `generate_final_synthetic_dataset`, `summarize_final_synthetic_eval`, `validate_frozen_checkpoint_provenance` |
| `_json_sha256` | Fingerprints a JSON value. | in-file: `generate_final_synthetic_dataset` |
| `_stable_hash_strings` | Fingerprints a list of strings. | in-file: `generate_final_synthetic_dataset`, `summarize_final_synthetic_eval` |
| `_git_sha` | Current commit id. | in-file: `generate_final_synthetic_dataset`, `run_final_synthetic_eval`; tests |
| `_load_json` | Reads a JSON object. | in-file: `summarize_final_synthetic_eval`, `validate_frozen_checkpoint_provenance` |
| `_load_checkpoint` | Loads a checkpoint dictionary. | in-file: `validate_frozen_checkpoint_provenance` |
| `_complete_identity` | Fails if an identity record is missing. | in-file: `validate_frozen_checkpoint_provenance` |
| `_scientific_source_identity` | Identity without machine-specific paths. | in-file: `validate_frozen_checkpoint_provenance` |
| `_expected_checkpoint_config` | The config each checkpoint should have been trained with. | in-file: `validate_frozen_checkpoint_provenance` |
| `_validate_phase2d_config` | Checks Phase 2D used seeds 11/22/33 and the frozen cohort. | in-file: `validate_frozen_checkpoint_provenance` |
| `validate_frozen_checkpoint_provenance` | Checks every checkpoint is exactly the frozen one before evaluation. | in-file: `run_final_synthetic_eval`; tests |
| `_price_call` | Prices one call with the main engine. | in-file: `generate_final_synthetic_dataset`; tests |
| `_domain_identity` | Summary of the input ranges. | in-file: `generate_final_synthetic_dataset` |
| `generate_final_synthetic_dataset` | Generates and saves the final holdout set. | in-file: `run_final_synthetic_eval`; tests |
| `_tensor_hash` | Fingerprints tensors. | in-file: `build_common_diagnostics` |
| `build_common_diagnostics` | Builds the shared physics check points. | in-file: `run_final_synthetic_eval`; tests |
| `_rmse` | Root mean squared error. | in-file: `_evaluate_checkpoint` |
| `_mae` | Mean absolute error. | in-file: `_evaluate_checkpoint` |
| `_sync` | Waits for the GPU to finish (for timing). | in-file: `_evaluate_checkpoint` |
| `_load_inference_model` | Loads a network for prediction only. | in-file: `_evaluate_checkpoint` |
| `_evaluate_checkpoint` | Computes price and physics metrics for one checkpoint. | in-file: `run_final_synthetic_eval`; tests |
| `_preflight_output_root` | Refuses if results already exist. | in-file: `run_final_synthetic_eval` |
| `_write_atomic_claim` | Writes the one-time claim file, failing if it exists. | in-file: `run_final_synthetic_eval` |
| `run_final_synthetic_eval` | Generates the holdout, claims it once, evaluates every checkpoint. | `scripts/mentor_dh_pinn/run_final_synthetic_eval.py`; tests |
| `_format_number` | Formats a number with full precision. | in-file: `_summary_row` |
| `_format_bool` | Formats true/false. | in-file: `_summary_row` |
| `_validate_completed_metric` | Checks a saved metric file is complete and matches its checkpoint. | in-file: `summarize_final_synthetic_eval` |
| `_summary_row` | One summary row. | in-file: `summarize_final_synthetic_eval` |
| `_aggregate_metric` | Mean and spread of a metric across seeds. | in-file: `summarize_final_synthetic_eval` |
| `summarize_final_synthetic_eval` | Writes the Phase 3A summary tables. | in-file: `run_final_synthetic_eval`; `scripts/mentor_dh_pinn/summarize_final_synthetic_eval.py`; tests |

#### `src/mentor_dh_pinn/full_horizon_confirmation.py`

*408 lines; newest version 9dbe592 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Phase 2C (Aug 29): retrains the best Phase 2B loss-weight setting for the full training length, judged on validation only, and compares with Phase 1.

| Function / class | What it does | Used by |
|---|---|---|
| `Phase1Reference` | The Phase 1 baseline's weights and best validation score. | in-file: `FullHorizonConfirmationSpec`, `_parse_reference`, `_reference_tuple`, `_validate_phase1_anchor` |
| `Phase1Reference.weights` | Its four loss weights. | in-file: `_validate_phase1_anchor`; tests |
| `FullHorizonConfirmationSpec` | Frozen Phase 2C settings. | in-file: `load_full_horizon_confirmation_spec` |
| `_strict_keys` | Rejects unknown or missing fields. | in-file: `_parse_reference`, `_parse_setting`, `load_full_horizon_confirmation_spec` |
| `_parse_setting` | Reads the candidate setting. | in-file: `load_full_horizon_confirmation_spec` |
| `_parse_reference` | Reads the Phase 1 reference. | in-file: `load_full_horizon_confirmation_spec` |
| `_candidate_tuple` | Candidate as a comparable tuple. | in-file: `load_full_horizon_confirmation_spec`; tests |
| `_reference_tuple` | Reference as a comparable tuple. | in-file: `load_full_horizon_confirmation_spec` |
| `load_full_horizon_confirmation_spec` | Loads Phase 2C config and rejects drift. | in-file: `run_full_horizon_confirmation`, `summarize_full_horizon_confirmation`; tests |
| `_resolve` | Makes a path absolute. | in-file: `run_full_horizon_confirmation`, `summarize_full_horizon_confirmation` |
| `_load_phase1_checkpoint` | Loads the Phase 1 checkpoint. | in-file: `run_full_horizon_confirmation`; tests |
| `_cohort_identity` | Fingerprints of the train/validation cohort. | in-file: `_validate_phase1_anchor` |
| `_validate_phase1_anchor` | Checks the run uses exactly the Phase 1 data and settings. | in-file: `run_full_horizon_confirmation` |
| `_git_sha` | Current commit id. | in-file: `run_full_horizon_confirmation`; tests |
| `_selected_epoch_metrics` | Reads metrics of the chosen best epoch. | in-file: `run_full_horizon_confirmation` |
| `run_full_horizon_confirmation` | Trains the candidate with validation only. | `scripts/mentor_dh_pinn/run_full_horizon_confirmation.py`; tests |
| `_format_number` | Formats a number. | in-file: `summarize_full_horizon_confirmation` |
| `summarize_full_horizon_confirmation` | Writes the candidate-vs-Phase-1 comparison. | `scripts/mentor_dh_pinn/summarize_full_horizon_confirmation.py`; tests |

#### `src/mentor_dh_pinn/lambda_refinement.py`

*267 lines; newest version 7c2ee8a (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Phase 2B (Aug 28): a finer search of loss weights around the best Phase 2A setting (A9), validation only.

| Function / class | What it does | Used by |
|---|---|---|
| `LambdaRefinementSpec` | Frozen Phase 2B settings. | in-file: `load_lambda_refinement_spec` |
| `_strict_keys` | Rejects unknown or missing fields. | in-file: `_parse_setting`, `load_lambda_refinement_spec` |
| `_parse_setting` | Reads one weight setting. | in-file: `load_lambda_refinement_spec` |
| `_setting_tuple` | Setting as a comparable tuple. | in-file: `load_lambda_refinement_spec`; tests |
| `load_lambda_refinement_spec` | Loads Phase 2B config and rejects drift. | in-file: `run_lambda_refinement`, `summarize_lambda_refinement`; tests |
| `_resolve_config` | Makes a path absolute. | in-file: `run_lambda_refinement`, `summarize_lambda_refinement` |
| `run_lambda_refinement` | Trains the selected B runs. | `scripts/mentor_dh_pinn/run_lambda_refinement.py`; tests |
| `_format_number` | Formats a number. | in-file: `summarize_lambda_refinement` |
| `summarize_lambda_refinement` | Writes the B0-B8 summary table. | `scripts/mentor_dh_pinn/summarize_lambda_refinement.py`; tests |

#### `src/mentor_dh_pinn/lambda_sweep.py`

*321 lines; newest version 7e4b4cb (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Phase 2A (Aug 28): tries nine predeclared combinations of the four loss weights (A1-A9), judged on validation data only.

| Function / class | What it does | Used by |
|---|---|---|
| `LambdaSetting` | One combination of the four loss weights. | in-file: `LambdaSweepSpec`, `build_run_config`, `load_lambda_sweep_spec`; `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py` |
| `LambdaSetting.as_dict` | As a dictionary. | in-file: `run_lambda_sweep`, `summarize_lambda_sweep`; `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py`; tests |
| `LambdaSweepSpec` | Frozen Phase 2A settings. | in-file: `build_run_config`, `load_lambda_sweep_spec` |
| `LambdaRunConfig` | The baseline settings with one run's weights swapped in. | in-file: `build_run_config` |
| `LambdaRunConfig.__getattr__` | Falls back to baseline settings for anything not overridden. | Python (automatic) |
| `LambdaRunConfig.validate` | Checks weights are positive and other rules hold. | in-file: `LambdaRunConfig`, `build_run_config` |
| `LambdaRunConfig.to_dict` | As a dictionary. | in-file: `LambdaRunConfig` |
| `_strict_keys` | Rejects unknown or missing fields. | in-file: `load_lambda_sweep_spec`; `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py` |
| `load_lambda_sweep_spec` | Loads the nine-run matrix and rejects drift. | in-file: `run_lambda_sweep`, `summarize_lambda_sweep`; tests |
| `build_run_config` | Builds one run's settings. | in-file: `run_lambda_sweep`; `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py`; tests |
| `run_lambda_sweep` | Trains the selected runs, never touching test. | `scripts/mentor_dh_pinn/run_lambda_sweep.py`; tests |
| `_format_number` | Formats a number. | in-file: `summarize_lambda_sweep`; `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py` |
| `summarize_lambda_sweep` | Writes the A1-A9 summary table. | `scripts/mentor_dh_pinn/summarize_lambda_sweep.py`; tests |

#### `src/mentor_dh_pinn/losses.py`

*210 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

The four "marking schemes" for the mentor forward-PINN: match known prices (data), obey the Double Heston equation (PDE), equal the payoff at expiry (terminal), and behave correctly at very low and very high stock prices (boundary).

| Function / class | What it does | Used by |
|---|---|---|
| `_finite_scalar` | Checks a loss is a single finite number. | in-file: `high_s_boundary_loss`, `low_s_boundary_loss`, `pde_loss`, `terminal_loss` ... |
| `data_loss` | Average squared price error, divided by stock price. | in-file: `normalized_data_loss`; `src/mentor_dh_pinn/trainer.py`; tests |
| `normalized_data_loss` | Runs the network and returns the data loss, predictions and raw error. | no caller found by search |
| `pde_residual` | How badly the network's price surface breaks the Double Heston pricing equation at each interior point (uses automatic derivatives). | in-file: `pde_loss`; `scripts/run_model3_pde_pilot.py`; tests |
| `pde_loss` | Average squared, scaled equation error. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/plotting.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `terminal_payoff` | Call payoff max(S - K, 0) at expiry. | in-file: `terminal_loss`; `scripts/run_model3_pde_pilot.py` |
| `terminal_loss` | Average squared gap between the network and the payoff at expiry. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `low_s_boundary_loss` | Error at near-zero stock price, where a call is worth zero. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `high_s_boundary_loss` | Error at very high stock price, where a call is worth stock minus discounted strike. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `LossComponents` | Holds the five loss parts. | in-file: `weighted_total_loss`; `src/mentor_dh_pinn/trainer.py`; tests |
| `LossComponents.total` | Weighted total of the parts. | in-file: `weighted_total_loss`; `scripts/run_model3_pde_pilot.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/trainer.py`, `train_double_heston.py`; tests |
| `LossComponents.as_dict` | The parts as a dictionary. | in-file: `weighted_total_loss`; `src/mentor_dh_pinn/trainer.py` |
| `weighted_total_loss` | Adds the parts with their lambda weights (all 1 by default). | in-file: `LossComponents`, `LossComponents.total`; `src/mentor_dh_pinn/trainer.py`; tests |

#### `src/mentor_dh_pinn/model.py`

*102 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

The mentor forward-PINN network itself: 7 inputs (S, two variances, tau, K, r, q) through five 128-wide tanh layers to one call price.

| Function / class | What it does | Used by |
|---|---|---|
| `InputNormalizer` | Rescales each raw input into the range -1 to 1 before the network sees it. | in-file: `DoubleHestonForwardPINN`, `DoubleHestonForwardPINN.__init__` |
| `InputNormalizer.__init__` | Stores and checks the input ranges. | Python (automatic) |
| `InputNormalizer.forward` | Applies the rescaling, checking shape and finiteness. | in-file: `DoubleHestonForwardPINN`, `DoubleHestonForwardPINN.predict_price` |
| `DoubleHestonForwardPINN` | The network: given a contract and current state, it guesses the call price. | `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/__init__.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py` and more |
| `DoubleHestonForwardPINN.__init__` | Builds the fixed five-layer architecture. | Python (automatic) |
| `DoubleHestonForwardPINN.parameter_count` | Counts trainable weights (67,201 per the protocol doc). | `scripts/mentor_dh_pinn/run_baseline.py`; tests |
| `DoubleHestonForwardPINN.normalized_features` | Rescaled inputs. | in-file: `DoubleHestonForwardPINN`, `DoubleHestonForwardPINN.forward` |
| `DoubleHestonForwardPINN.forward` | Runs the network. | in-file: `DoubleHestonForwardPINN`, `DoubleHestonForwardPINN.predict_price` |
| `DoubleHestonForwardPINN.predict_price` | Returns one price per row. | no caller found by search |

#### `src/mentor_dh_pinn/multiseed_confirmation.py`

*635 lines; newest version c7a7f32 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Phase 2D (Aug 29): re-trains the best-weight variant and the equal-weight variant with three fresh seeds (11, 22, 33) to check the improvement is not luck; validation only.

| Function / class | What it does | Used by |
|---|---|---|
| `MultiseedRunSetting` | One run: variant, seed and four weights. | in-file: `MultiseedConfirmationSpec`, `_load_complete_result`, `_parse_run`, `_run_tuple` ... |
| `MultiseedRunSetting.weights` | Its weights. | in-file: `MultiseedRunConfig`, `MultiseedRunConfig.validate`; tests |
| `MultiseedRunSetting.as_dict` | As a dictionary. | in-file: `_load_complete_result`, `run_multiseed_confirmation`; tests |
| `Phase1Reference` | Phase 1 best epoch and validation score. | in-file: `MultiseedConfirmationSpec`, `load_multiseed_confirmation_spec` |
| `MultiseedConfirmationSpec` | Frozen Phase 2D settings. | in-file: `_validate_phase1_anchor`, `build_multiseed_run_config`, `load_multiseed_confirmation_spec` |
| `MultiseedRunConfig` | Baseline settings with only seed and weights changed. | in-file: `build_multiseed_run_config` |
| `MultiseedRunConfig.__getattr__` | Falls back to baseline settings. | Python (automatic) |
| `MultiseedRunConfig.validate` | Checks seed is allowed and controls unchanged. | in-file: `MultiseedRunConfig`, `build_multiseed_run_config` |
| `MultiseedRunConfig.to_dict` | As a dictionary. | in-file: `MultiseedRunConfig`, `_validate_phase1_anchor` |
| `_strict_keys` | Rejects unknown or missing fields. | in-file: `_parse_run`, `load_multiseed_confirmation_spec` |
| `_parse_run` | Reads one run entry. | in-file: `load_multiseed_confirmation_spec` |
| `_run_tuple` | Run as a comparable tuple. | in-file: `load_multiseed_confirmation_spec`; tests |
| `load_multiseed_confirmation_spec` | Loads Phase 2D config and rejects drift. | in-file: `run_multiseed_confirmation`, `summarize_multiseed_confirmation`; tests |
| `build_multiseed_run_config` | Builds one run's settings. | in-file: `run_multiseed_confirmation`; tests |
| `_resolve` | Makes a path absolute. | in-file: `run_multiseed_confirmation`, `summarize_multiseed_confirmation` |
| `_load_phase1_checkpoint` | Loads Phase 1 checkpoint. | in-file: `run_multiseed_confirmation`; tests |
| `_cohort_identity` | Fingerprints of the data cohort. | in-file: `_validate_phase1_anchor` |
| `_validate_phase1_anchor` | Checks the runs use exactly the Phase 1 data and settings. | in-file: `run_multiseed_confirmation` |
| `_git_sha` | Current commit id. | in-file: `run_multiseed_confirmation`; tests |
| `_selected_epoch_metrics` | Metrics of the chosen best epoch. | in-file: `run_multiseed_confirmation` |
| `run_multiseed_confirmation` | Trains the selected runs. | `scripts/mentor_dh_pinn/run_multiseed_confirmation.py`; tests |
| `_format_number` | Formats a number. | in-file: `_run_summary_row`, `summarize_multiseed_confirmation` |
| `_format_bool` | Formats true/false. | in-file: `_run_summary_row`, `summarize_multiseed_confirmation` |
| `_load_complete_result` | Loads one finished run, failing if incomplete. | in-file: `summarize_multiseed_confirmation` |
| `_run_summary_row` | One summary row. | in-file: `summarize_multiseed_confirmation` |
| `_pde_ratio` | Ratio of optimized to equal-weight physics error. | in-file: `summarize_multiseed_confirmation` |
| `summarize_multiseed_confirmation` | Writes per-run, paired and overall summaries. | `scripts/mentor_dh_pinn/summarize_multiseed_confirmation.py`; tests |

#### `src/mentor_dh_pinn/parameter_source.py`

*238 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Picks the single fixed "true" parameter set for the mentor experiment: the first valid training record of the frozen 10k dataset.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256_file` | Fingerprints a file. | in-file: `select_first_eligible_train_record` |
| `iter_surface_records` | Reads dataset lines one by one, rejecting bad ones. | in-file: `select_first_eligible_train_record` |
| `_metadata` | Gets a record's metadata. | in-file: `_user_metadata`, `parameter_vector_from_record`, `select_first_eligible_train_record` |
| `_user_metadata` | Gets the user metadata inside it. | in-file: `select_first_eligible_train_record` |
| `parameter_vector_from_record` | Extracts the ten parameters from a record. | in-file: `select_first_eligible_train_record` |
| `ParameterSource` | The chosen parameter set plus where it came from. | in-file: `select_first_eligible_train_record`, `select_first_train_record`; `src/mentor_dh_pinn/__init__.py`, `src/mentor_dh_pinn/collocation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/plotting.py` and more; tests |
| `ParameterSource.__post_init__` | Checks it is ten valid numbers from the training split. | Python (automatic) |
| `ParameterSource.mapping` | As a named dictionary. | `src/mentor_dh_pinn/final_synthetic_eval.py`; tests |
| `ParameterSource.structural_vector` | The eight non-v0 parameters. | tests |
| `ParameterSource.parameters_for_state` | Same structure but with chosen current variances put in place of v0. | `src/mentor_dh_pinn/collocation.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/plotting.py`, `src/mentor_dh_pinn/synthetic_data.py`; tests |
| `ParameterSource.provenance` | Where it came from (file, fingerprint, id, split). | `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/synthetic_data.py`; tests |
| `select_first_eligible_train_record` | Finds and validates the first training record. | in-file: `select_first_train_record`; `src/mentor_dh_pinn/__init__.py`, `src/mentor_dh_pinn/final_synthetic_eval.py`, `src/mentor_dh_pinn/synthetic_data.py`; tests |
| `select_first_train_record` | Short alias for the above. | no caller found by search |

#### `src/mentor_dh_pinn/plotting.py`

*353 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Draws the six grayscale result figures for the mentor PINN.

| Function / class | What it does | Used by |
|---|---|---|
| `_read_csv` | Reads a training-history CSV into numbers. | in-file: `make_figures` |
| `_save` | Saves a figure. | in-file: `make_figures` |
| `_features` | Builds network inputs for a figure slice. | in-file: `make_figures` |
| `_classical_prices` | Exact engine prices for the same slice. | in-file: `make_figures` |
| `_predict` | Network prices for a slice. | in-file: `make_figures` |
| `make_figures` | Draws price-vs-strike, price-vs-maturity, error, losses, validation and physics-residual figures. | `scripts/mentor_dh_pinn/make_figures.py` |

#### `src/mentor_dh_pinn/reproducible_regular_pinn_data.py`

*313 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Viraj's Sep 11 "V2" data generator for Dhruva's regular PINN, made byte-for-byte reproducible on any machine because the original training files were lost and could not be regenerated exactly.

| Function / class | What it does | Used by |
|---|---|---|
| `runtime_provenance` | Records library versions and checks they match the frozen ones. | in-file: `generate_dataset` |
| `sha256` | Fingerprints a file. | in-file: `array_sha256`, `generate_dataset`; `scripts/mentor_dh_pinn/prepare_regular_pinn_v2.py`; tests |
| `_little_endian` | Forces a standard byte order so files are identical across machines. | in-file: `canonicalize` |
| `canonicalize` | Rounds floats to a fixed number of decimals and fixes byte order. | in-file: `accepted_split`, `collocation_set`, `generate_dataset`, `npy_bytes` ... |
| `npy_bytes` | Serializes an array in a fixed format. | in-file: `array_sha256`, `write_deterministic_npz` |
| `array_sha256` | Fingerprints an array. | in-file: `generate_dataset` |
| `write_deterministic_npz` | Writes a zip of arrays with fixed metadata so it is identical everywhere. | in-file: `generate_dataset`; tests |
| `lhs_unit_block` | One block of evenly spread random points from a named random stream. | in-file: `proposal_block`; tests |
| `proposal_block` | Turns a block into candidate inputs (moneyness, log maturity, parameters). | in-file: `accepted_split`, `collocation_set` |
| `structural_validity` | Checks candidate parameters obey all Double Heston rules. | in-file: `rejection_codes` |
| `rejection_codes` | Labels why a candidate was rejected (invalid parameters, bad price, etc.). | in-file: `label_proposals` |
| `label_proposals` | Prices candidates and computes rejection codes. | in-file: `accepted_split` |
| `accepted_split` | Keeps drawing until enough valid rows exist, recording rejections. | in-file: `generate_dataset` |
| `collocation_set` | Draws physics check points. | in-file: `generate_dataset` |
| `generate_dataset` | Writes train, validation and collocation files plus a manifest. | `scripts/mentor_dh_pinn/prepare_regular_pinn_v2.py`; tests |

#### `src/mentor_dh_pinn/synthetic_data.py`

*411 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Generates the mentor PINN's synthetic call-price dataset (4096 train / 1024 validation / 1024 test intended) with the fixed parameters and varied contracts and states.

| Function / class | What it does | Used by |
|---|---|---|
| `_stable_hash_strings` | Fingerprints a list of strings. | in-file: `SyntheticDataset`, `SyntheticDataset.split_id_hashes`, `generate_synthetic_dataset` |
| `_resolve_path` | Makes a path absolute. | in-file: `generate_synthetic_dataset` |
| `SyntheticDataset` | Read-only container of inputs, reference prices and split labels. | in-file: `_persist_dataset`, `generate_synthetic_dataset`, `load_synthetic_dataset`, `validate_dataset_identity`; `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/plotting.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `SyntheticDataset.__post_init__` | Checks shapes, positivity and split names. | Python (automatic) |
| `SyntheticDataset.size` | Number of rows. | in-file: `load_synthetic_dataset`; `scripts/mentor_dh_pinn/generate_dataset.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `SyntheticDataset.spot` | Stock price column. | in-file: `_sample_split`; `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/plotting.py` |
| `SyntheticDataset.variance_slow` | Slow variance column. | in-file: `_sample_split`; `src/mentor_dh_pinn/plotting.py`; tests |
| `SyntheticDataset.variance_fast` | Fast variance column. | in-file: `_sample_split`; `src/mentor_dh_pinn/plotting.py`; tests |
| `SyntheticDataset.tau` | Time-to-expiry column. | in-file: `_sample_split`; `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/plotting.py` |
| `SyntheticDataset.strike` | Strike column. | in-file: `_sample_split`; `src/mentor_dh_pinn/plotting.py` |
| `SyntheticDataset.rate` | Interest-rate column. | in-file: `_sample_split`; `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/plotting.py` |
| `SyntheticDataset.carry` | Carry (dividend yield) column. | in-file: `_sample_split`; `src/mentor_dh_pinn/plotting.py` |
| `SyntheticDataset.indices` | Row numbers in one split. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/trainer.py`; tests |
| `SyntheticDataset.split_id_hashes` | Fingerprint of each split's ids. | in-file: `load_synthetic_dataset`, `validate_dataset_identity`; `scripts/mentor_dh_pinn/generate_dataset.py`, `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py` and more; tests |
| `SyntheticDataset.to_tensors` | Converts to PyTorch tensors. | no caller found by search |
| `_sample_split` | Draws one split's contracts and states and prices them with the engine. | in-file: `generate_synthetic_dataset` |
| `generate_synthetic_dataset` | Generates, saves and returns the dataset. | `scripts/mentor_dh_pinn/generate_dataset.py`, `scripts/mentor_dh_pinn/run_baseline.py`; tests |
| `_persist_dataset` | Saves arrays, manifest and provenance files. | in-file: `generate_synthetic_dataset` |
| `load_synthetic_dataset` | Loads a saved dataset and verifies it. | `scripts/mentor_dh_pinn/evaluate_baseline.py`, `scripts/mentor_dh_pinn/make_figures.py`, `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py` and more; tests |
| `_json_sha256` | Fingerprints a JSON value. | in-file: `generate_synthetic_dataset`, `validate_dataset_identity` |
| `validate_dataset_identity` | Fails unless dataset, config and splits all agree. | in-file: `load_synthetic_dataset`; `src/mentor_dh_pinn/trainer.py` |
| `_sha256_file` | Fingerprints a file. | in-file: `_persist_dataset`, `load_synthetic_dataset`; `src/mentor_dh_pinn/trainer.py` |

#### `src/mentor_dh_pinn/torch_pricer.py`

*143 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Dhruva's file; Viraj's 11-line edit (Sep 11) added a safe complex log(1+z) so the PyTorch pricer stops producing bad numbers for tiny inputs on CPU. No per-function table, because Viraj did not write this file.

#### `src/mentor_dh_pinn/trainer.py`

*506 lines; newest version 7e4b4cb (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Trains the mentor forward-PINN with AdamW, keeps the checkpoint with the best validation score, and never touches test data.

| Function / class | What it does | Used by |
|---|---|---|
| `TrainingResult` | Paths and best-epoch information returned after training. | in-file: `train_baseline`; tests |
| `TrainingResult.as_dict` | As a dictionary. | `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py`, `src/mentor_dh_pinn/lambda_sweep.py` and more; tests |
| `seed_everything` | Sets every random generator's seed. | in-file: `train_baseline`; `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py`, `src/mentor_dh_pinn/lambda_sweep.py` and more; tests |
| `_git_sha` | Current commit id or "unknown". | in-file: `_checkpoint_payload`, `validate_checkpoint_identities`; `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/multiseed_confirmation.py`; tests |
| `runtime_identity` | Device and library versions. | in-file: `_checkpoint_payload` |
| `_sha256_file` | Fingerprints a file. | no caller found by search |
| `_batch_indices` | Seeded shuffle of rows into batches. | in-file: `train_baseline` |
| `_as_float64_tensor` | Converts to a float64 tensor on the device. | in-file: `train_baseline`, `validation_metrics` |
| `validation_metrics` | Price errors on validation rows only. | in-file: `train_baseline` |
| `_gradient_summary` | Size of the gradients and whether they are finite. | in-file: `train_baseline` |
| `_write_csv` | Writes a history table. | in-file: `train_baseline` |
| `_dataset_identity` | Dataset identity summary. | in-file: `_checkpoint_payload`, `validate_checkpoint_identities` |
| `validate_checkpoint_identities` | Checks a checkpoint matches its config, data and code. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/plotting.py`; tests |
| `_checkpoint_payload` | Everything saved in a checkpoint. | in-file: `train_baseline` |
| `train_baseline` | The training loop: batches, four losses, validation, early stopping, saving. | `scripts/mentor_dh_pinn/run_baseline.py`, `src/mentor_dh_pinn/full_horizon_confirmation.py`, `src/mentor_dh_pinn/lambda_refinement.py`, `src/mentor_dh_pinn/lambda_sweep.py` and more; tests |
| `load_checkpoint_model` | Restores a trained network. | `src/mentor_dh_pinn/evaluation.py`, `src/mentor_dh_pinn/plotting.py` |
| `load_checkpoint_config` | Reads the settings saved in a checkpoint. | `scripts/mentor_dh_pinn/evaluate_baseline.py`, `scripts/mentor_dh_pinn/make_figures.py` |

#### `src/mentor_dh_pinn/volatile_stock_selection.py`

*488 lines; newest version bbe8be6 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Phase 3B prep (Aug 29): downloads five years of official NSE prices for ten stocks, measures 3-month realized volatility, and checks which stock/date has a usable call surface; only POWERGRID on 15 July passed.

| Function / class | What it does | Used by |
|---|---|---|
| `VolatileStockAuditError` | Error for any audit rule violation. | in-file: `_exact_keys`, `_get_with_retry`, `_write_new`, `acquire_official_histories` ...; tests |
| `AuditSpec` | The frozen audit settings. | in-file: `_write_symbol_evidence`, `acquire_official_histories`, `audit_candidates`, `candidate_volatility` ... |
| `load_spec` | Reads and strictly checks the settings file. | `scripts/mentor_dh_pinn/run_volatile_stock_selection_audit.py`; tests |
| `parse_security_report` | Reads one daily NSE price report (CSV or Excel) and checks its date. | in-file: `acquire_official_histories`; tests |
| `acquire_official_histories` | Downloads and stores every daily report, building the price history. | `scripts/mentor_dh_pinn/run_volatile_stock_selection_audit.py` |
| `validate_history` | Checks for duplicates and bad fields. | in-file: `acquire_official_histories`; tests |
| `volatility_windows` | Rolling 3-month realized volatility over five years. | in-file: `candidate_volatility` |
| `candidate_volatility` | One stock/date's 3-month volatility and its five-year percentile. | in-file: `audit_candidates`; tests |
| `option_surface` | Builds the active-call surface for a stock/date and checks it fits the model's domain. | in-file: `audit_candidates` |
| `propose_split` | Proposes calibration vs holdout calls by strike quantiles (not executed). | in-file: `audit_candidates`; `scripts/mentor_dh_pinn/run_volatile_stock_selection_audit.py`; tests |
| `audit_candidates` | Runs the audit for every stock and date. | `scripts/mentor_dh_pinn/run_volatile_stock_selection_audit.py` |
| `_write_symbol_evidence` | Saves each stock's history extract. | in-file: `acquire_official_histories` |
| `_get_with_retry` | Downloads a URL with retries. | in-file: `acquire_official_histories` |
| `_read_manifest` | Reads the download manifest. | in-file: `acquire_official_histories` |
| `_atomic_csv` | Writes a CSV safely. | in-file: `_write_symbol_evidence`, `acquire_official_histories` |
| `_write_new` | Writes a new file, failing if a different one exists. | in-file: `acquire_official_histories` |
| `_weekdays` | Lists weekdays between two dates. | in-file: `acquire_official_histories` |
| `_git_sha` | Current commit id. | in-file: `_write_symbol_evidence`, `acquire_official_histories` |
| `_sha256` | Fingerprints bytes. | in-file: `_write_symbol_evidence`, `acquire_official_histories` |
| `_as_date` | Converts to a date. | in-file: `load_spec`, `option_surface` |
| `_exact_keys` | Rejects drifted config fields. | in-file: `load_spec` |

### models/

#### `models/__init__.py`

*13 lines; Viraj's file; newest version 642702e (Aug 22) is a small edit by Dhruva; on origin/main.*

Package front door that re-exports the model classes so other code can import them from one place. (No functions in this file.)

#### `models/ann_model.py`

*68 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The "Model 1" ordinary neural network: takes a list of option prices in and gives ten Double Heston numbers out, with no physics inside.

| Function / class | What it does | Used by |
|---|---|---|
| `ANNInverseCalibrator` | A plain stack of neural-network layers that turns a surface of prices into the ten model parameters, like a student who learns answers purely from worked examples. | `models/__init__.py`, `src/run_smoke_test.py`, `src/train.py`; tests |
| `ANNInverseCalibrator.__init__` | Builds the layers after checking the sizes and dropout setting make sense; refuses anything that is not exactly ten outputs. | Python (automatic) |
| `ANNInverseCalibrator.forward` | Pushes a batch of price surfaces through the layers and returns ten numbers per surface. | no caller found by search |
| `_activation_factory` | Looks up the chosen "squashing" function (relu, gelu or tanh) by name and errors on anything else. | in-file: `ANNInverseCalibrator`, `ANNInverseCalibrator.__init__` |

#### `models/parameter_transform.py`

*108 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Helpers that keep network outputs inside allowed ranges and put targets on a common scale for training.

| Function / class | What it does | Used by |
|---|---|---|
| `BoundedParameterTransform` | Squeezes any raw number into a fixed lower-upper range using an S-shaped curve, like a dimmer switch that can never go past fully on or fully off. | `models/__init__.py`; tests |
| `BoundedParameterTransform.__init__` | Stores the checked lower and upper limits for the ten parameters. | Python (automatic) |
| `BoundedParameterTransform.forward` | Converts raw network outputs into values guaranteed to sit between the limits. | no caller found by search |
| `BoundedParameterTransform.inverse` | Runs the squeeze backwards, but only for values strictly inside the limits. | tests |
| `TargetStandardizer` | Rescales the ten training answers to mean zero and spread one using training data only, so no single parameter dominates the learning. | `models/__init__.py`, `src/train.py`; tests |
| `TargetStandardizer.__init__` | Starts empty; it has no mean or spread until fitted. | Python (automatic) |
| `TargetStandardizer.fit` | Measures the mean and spread of each parameter on the training set and refuses if any parameter never varies. | `src/train.py`; tests |
| `TargetStandardizer.transform` | Converts real parameter values into the standardized scale. | `src/train.py`; tests |
| `TargetStandardizer.inverse_transform` | Converts standardized values back into real parameter values. | `src/train.py`; tests |
| `TargetStandardizer.state_dict` | Packages the fitted mean and spread so they can be saved with a checkpoint. | `src/train.py` |
| `TargetStandardizer._fitted_values` | Returns the stored mean and spread, erroring if fit was never called. | in-file: `TargetStandardizer`, `TargetStandardizer.inverse_transform`, `TargetStandardizer.state_dict`, `TargetStandardizer.transform` |
| `_validated_bounds` | Checks that there are ten lower and ten upper limits and each lower is below its upper. | in-file: `BoundedParameterTransform`, `BoundedParameterTransform.__init__` |
| `_validate_last_dimension` | Checks that the last dimension of an array is exactly ten. | in-file: `BoundedParameterTransform`, `BoundedParameterTransform.forward`, `BoundedParameterTransform.inverse` |
| `_validate_target_matrix` | Checks a table of targets is rows-by-ten, real-valued and finite. | in-file: `TargetStandardizer`, `TargetStandardizer.fit`, `TargetStandardizer.inverse_transform`, `TargetStandardizer.transform` |

### scripts/ (top level)

#### `scripts/create_market_data_audit_stage_a_structure.py`

*107 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Creates empty, git-ignored folders and "not collected" placeholder files for each stock and date in the Stage A market-data audit, never fake data.

| Function / class | What it does | Used by |
|---|---|---|
| `create_structure` | Makes every Stage A folder and writes placeholder manifests only where none exist. | in-file: `main`; tests |
| `_manifest` | Builds the placeholder manifest content saying "NOT_COLLECTED". | in-file: `create_structure` |
| `_write_missing_manifest` | Writes a placeholder manifest only if one is not already there. | in-file: `create_structure` |
| `main` | Command-line entry point. | command line |

#### `scripts/export_model3_paper_results.py`

*29 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Thin wrapper: exports paper tables from a sealed, complete Model 3 result (none exists).

#### `scripts/investigate_ntpc_canonical160_reproduction.py`

*302 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Forensic script that explained why a re-run of the NTPC 12-start calibration did not reproduce the original numbers (tiny maturity rounding from a CSV file).

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Computes an upper-case file fingerprint. | in-file: `run`; tests |
| `_validated_population` | Checks the 12-start results table has every start 0 to 11 exactly once. | in-file: `compare_starts`, `reference_stability` |
| `reference_stability` | Recomputes the spread-of-solutions statistics from scratch, independent of project helpers. | in-file: `run`; tests |
| `_margin_values` | Measures how close one solution sits to the parameter limits. | in-file: `compare_starts` |
| `compare_starts` | Compares the original and replayed answer for each start side by side. | in-file: `run`; tests |
| `write_json` | Writes a JSON file in a stable, sorted format. | in-file: `run` |
| `validate_causal_probe` | Refuses to conclude unless the start-6 experiments prove the rounding was the cause. | in-file: `run`; tests |
| `run` | Checks input fingerprints, runs the comparisons and writes the forensic summary. | no caller found by search |

#### `scripts/run_g2_common_support_analysis.py`

*1314 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

G2 step 1 (Aug 11): looks at real NSE option data for NTPC, CIPLA, INFY and HDFCBANK on three dates and asks "which strikes and expiries actually trade on all of them?"; answer: near + middle expiry, five central strikes, calls and puts (20 prices).

| Function / class | What it does | Used by |
|---|---|---|
| `CandidateSpec` | A description of one candidate price-grid layout (which maturities, which strike distances, calls/puts). | in-file: `_candidate_expiry_slots`, `select_maximal_passing_geometry` |
| `_sha256_bytes` | Fingerprints a block of bytes. | in-file: `_sha256_file`, `validate_canonical_stage_a_provenance` |
| `_sha256_file` | Fingerprints a file. | in-file: `snapshot_canonical_outputs` |
| `snapshot_canonical_outputs` | Fingerprints the eight Stage A output files, failing if any is missing. | in-file: `assert_canonical_outputs_preserved`, `run_analysis`; `scripts/run_g2_information_remediation.py`; tests |
| `assert_canonical_outputs_preserved` | Fails if any Stage A output changed while this analysis ran. | in-file: `run_analysis`; `scripts/run_g2_information_remediation.py`; tests |
| `validate_canonical_stage_a_provenance` | Checks every downloaded NSE archive and derived CSV matches its recorded fingerprint and path. | in-file: `run_analysis`; `scripts/run_g2_information_remediation.py`; tests |
| `load_balanced_panel` | Reads option rows for exactly the four chosen stocks and three dates. | in-file: `run_analysis`; `scripts/run_g2_information_remediation.py` |
| `validate_balanced_panel` | Fails unless the panel has exactly those four stocks, three dates and three expiries each. | in-file: `load_balanced_panel`; tests |
| `build_surface_support` | Counts, for each stock/date/expiry/type, how many strikes were quoted and actively traded. | in-file: `run_analysis`; tests |
| `_node_support` | Decides whether a target strike distance was directly observed, bracketed, or unsupported. | in-file: `build_moneyness_support`; tests |
| `build_moneyness_support` | Applies that test to every target strike distance on every slice. | in-file: `run_analysis`; `scripts/run_g2_information_remediation.py`; tests |
| `_fixed_maturity_support` | Decides whether a fixed maturity (like 30 days) is directly listed, bracketed, or outside the listed expiries. | in-file: `build_maturity_support`; tests |
| `build_maturity_support` | Applies that maturity test to every stock and date. | in-file: `run_analysis`; tests |
| `_minimum_group_support` | Finds the worst-case support percentage across groups. | in-file: `build_representation_candidates`, `render_report` |
| `_candidate_expiry_slots` | Works out which expiry slots (near/mid/far) a candidate layout needs. | in-file: `build_representation_candidates` |
| `select_maximal_passing_geometry` | Picks the largest strike set that passes on all stocks, without favouring more features. | in-file: `build_representation_candidates`; tests |
| `build_representation_candidates` | Scores every candidate layout and marks which pass the support rules. | in-file: `run_analysis`; tests |
| `_serialize_csv` | Turns a table into CSV bytes in a fixed format. | in-file: `publish_evidence_atomically` |
| `publish_evidence_atomically` | Writes the four result CSVs all at once so a crash cannot leave half-written files. | in-file: `run_analysis`; tests |
| `_atomic_write_bytes` | Writes a file via a temporary file then renames it, so it is never half-written. | in-file: `run_analysis`; `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` |
| `write_figures` | Draws the maturity and strike-support heat-map figures. | in-file: `run_analysis`; `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` |
| `_pct` | Formats a percentage to one decimal. | in-file: `render_report` |
| `render_report` | Writes the Markdown report with all the support tables. | in-file: `run_analysis`; `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` |
| `run_analysis` | Runs the whole pipeline: check provenance, load, measure support, choose layout, write evidence. | in-file: `main`; `scripts/run_g2_identifiability_analysis.py`, `scripts/run_g2_information_remediation.py` |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_complementary_observable_analysis.py`

*1887 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

G2 experiment (Aug 12): would adding non-option information (total variance, realized volatility from past returns, persistence) make the ten parameters recoverable? Result: INSUFFICIENT.

| Function / class | What it does | Used by |
|---|---|---|
| `Design` | One of the four tested information designs A/B/C/D and which extra features it adds. | in-file: `_model_feature_vector`, `_record_solution`, `_scaled_model_vector`, `observed_features` ... |
| `_sha256` | Fingerprints a file. | in-file: `_protected_snapshot`, `_validate_replay_evidence`, `render_existing_outputs`, `run_analysis`; tests |
| `_atomic_write_bytes` | Writes a file safely via a temporary file. | in-file: `_write_csv`, `_write_json`, `render_existing_outputs`, `run_analysis` |
| `_write_csv` | Safely writes a table as CSV. | in-file: `run_analysis` |
| `_write_json` | Safely writes JSON. | in-file: `run_analysis` |
| `_parameter_widths` | Width of the allowed range for each parameter. | in-file: `_record_solution`, `_scaled_coordinates`, `scaled_jacobian` |
| `_scaled_coordinates` | Rescales parameters to 0-1 within their allowed ranges. | in-file: `_record_solution` |
| `experiment_matrix` | Lists the designs and their feature sets as a table. | in-file: `predeclared_contract`, `run_analysis`; tests |
| `predeclared_contract` | Returns the rules written down before any fitting, so they cannot be changed after seeing results. | in-file: `_predeclared_contract_sha256`, `_validate_replay_evidence`, `classify_decision`, `run_analysis`; tests |
| `_predeclared_contract_sha256` | Fingerprints those rules. | in-file: `run_analysis` |
| `_case_seed` | Makes a unique, repeatable random seed per case and purpose. | in-file: `observed_features`, `simulate_causal_return_history` |
| `_factor_indices` | Gives the positions of the slow or fast factor's five parameters. | in-file: `expected_persistence`, `simulate_causal_return_history` |
| `simulate_causal_return_history` | Simulates one year of daily stock returns from the model that ends at the valuation state. | in-file: `run_analysis`; tests |
| `path_observables` | Computes 21-day and 126-day realized variance and a persistence number from that history. | in-file: `run_analysis`; tests |
| `truth_fit_diagnostics` | Checks how far the true parameters' expected features are from the simulated ones. | in-file: `run_analysis`; tests |
| `_expected_trailing_variance` | Model-predicted average variance over a past window. | in-file: `model_features`; tests |
| `expected_persistence` | Approximate model-predicted persistence of variance across 5-day blocks. | in-file: `model_features`; tests |
| `model_features` | All model-predicted extra features for a parameter vector. | in-file: `_model_feature_vector`, `observed_features`, `run_analysis`; tests |
| `observed_features` | The "observed" extra features for one design, with optional noise. | in-file: `run_recovery` |
| `_model_feature_vector` | Model features in the order a design expects. | in-file: `_scaled_model_vector`, `run_recovery` |
| `_scaled_model_vector` | Prices plus extra features, each scaled so they weigh fairly in the fit. | in-file: `scaled_jacobian` |
| `scaled_jacobian` | Sensitivity of all observables to each parameter, by small nudges. | in-file: `run_identifiability` |
| `run_identifiability` | Computes rank and condition number (how "flat" the problem is) for every case and design. | in-file: `run_analysis` |
| `_prior_solutions` | Loads the earlier options-only (Design A) solutions. | in-file: `run_recovery` |
| `_case_truth` | Reads a case's true parameters. | in-file: `run_recovery` |
| `_noise_observed_prices` | Makes noisy observed prices for a case. | in-file: `run_recovery` |
| `_record_solution` | Records one fitted solution with its price error, feature error and distance from truth. | in-file: `run_recovery` |
| `run_recovery` | Runs the multi-start fits for every design, case and noise level. | in-file: `run_analysis` |
| `summarize_recovery` | Summarizes recovery accuracy and how many separated near-equal solutions remain. | in-file: `run_analysis` |
| `add_matched_noise_comparison` | Compares clean and noisy results only on cases tested at every noise level. | in-file: `_validate_replay_evidence`, `run_analysis`; tests |
| `market_feasibility` | Checks, without downloading anything, whether real NSE data could supply these extra features. | in-file: `run_analysis`; tests |
| `classify_decision` | Applies the pre-written rules to label the experiment (it ended INSUFFICIENT). | in-file: `_validate_replay_evidence`, `run_analysis`; tests |
| `_save_figure` | Saves and closes a figure. | in-file: `write_figures` |
| `_validate_output_paths` | Refuses output paths that would overwrite protected evidence. | in-file: `render_existing_outputs`, `run_analysis`; tests |
| `_validate_run_mode` | Refuses non-canonical settings unless clearly marked as a test. | in-file: `run_analysis`; tests |
| `_validate_replay_evidence` | Checks that re-rendered results exactly match the frozen run. | in-file: `render_existing_outputs`; tests |
| `write_figures` | Draws the conditioning, recovery and ambiguity figures. | in-file: `render_existing_outputs`, `run_analysis` |
| `_protected_snapshot` | Fingerprints all earlier Stage A and G2 evidence. | in-file: `_assert_protected_unchanged`, `render_existing_outputs`, `run_analysis`; tests |
| `_snapshot_aggregate` | One overall fingerprint for that snapshot. | in-file: `run_analysis` |
| `_assert_protected_unchanged` | Fails if earlier evidence changed during the run. | in-file: `render_existing_outputs`, `run_analysis` |
| `render_report` | Writes the Markdown report. | in-file: `render_existing_outputs`, `run_analysis` |
| `run_analysis` | Runs the whole experiment end to end. | in-file: `main`; tests |
| `render_existing_outputs` | Rebuilds the report and figures from saved results only. | in-file: `main`; tests |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_global_ambiguity_analysis.py`

*917 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

G2 experiment (Aug 11): fits each of four representative clean surfaces from many starting points and shows 40 near-perfect fits split into 39 different parameter clusters: GLOBAL_AMBIGUITY = ESTABLISHED.

| Function / class | What it does | Used by |
|---|---|---|
| `_sha256` | Fingerprints a file. | in-file: `_protected_snapshot`, `render_existing_outputs`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py` |
| `_atomic_write_bytes` | Writes a file safely via a temporary file. | in-file: `_write_csv`, `_write_json`, `render_existing_outputs`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py` |
| `_write_csv` | Safely writes CSV. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py` |
| `_write_json` | Safely writes JSON. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py` |
| `_parameter_widths` | Allowed range width per parameter. | in-file: `_recover_case`, `_scaled_coordinates`; `scripts/run_g2_complementary_observable_analysis.py` |
| `_scaled_coordinates` | Rescales parameters to 0-1. | in-file: `_recover_case`, `cluster_summary`, `constraint_margins`; `scripts/run_g2_complementary_observable_analysis.py` |
| `constraint_margins` | How far a solution is from each constraint edge. | in-file: `_recover_case`; tests |
| `_protected_snapshot` | Fingerprints earlier evidence. | in-file: `_assert_protected_unchanged`, `render_existing_outputs`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`; tests |
| `_snapshot_aggregate` | One overall fingerprint for the snapshot. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py` |
| `_assert_protected_unchanged` | Fails if earlier evidence changed. | in-file: `render_existing_outputs`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py` |
| `select_cases` | Picks the fixed four representative truth cases. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`; tests |
| `deterministic_starts` | Makes a repeatable list of starting guesses (one neutral plus random ones). | in-file: `_recover_case`, `start_schedule`; tests |
| `_case_seed` | Unique repeatable seed per case, noise level and purpose. | in-file: `_recover_case`, `multiplicative_noise`, `start_schedule`; `scripts/run_g2_complementary_observable_analysis.py` |
| `start_schedule` | Lists the exact starting points for all 120 fits. | in-file: `start_schedule_sha256`; tests |
| `start_schedule_sha256` | Fingerprints that start list. | in-file: `render_report`, `run_analysis`; tests |
| `runtime_provenance` | Records Python and library versions. | in-file: `run_analysis`; tests |
| `multiplicative_noise` | Adds percentage noise to prices with a fixed seed. | in-file: `_recover_case`; `scripts/run_g2_complementary_observable_analysis.py` |
| `complete_linkage_clusters` | Groups solutions into clusters where every member is within a cutoff distance of every other. | in-file: `noise_summary`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`; tests |
| `classify_basin` | Describes the shape of the set of near-equal solutions (how many clusters, how spread). | in-file: `noise_summary`, `run_analysis`; tests |
| `classify_global_ambiguity` | Labels ambiguity ESTABLISHED if at least three of four cases are ambiguous. | in-file: `run_analysis`; tests |
| `classify_alignment` | Checks whether the global wrong answers point along the locally flattest direction. | in-file: `run_analysis`; tests |
| `_recover_case` | Runs all fits for one case and noise level. | in-file: `run_analysis` |
| `compensation_pairs` | Finds parameter pairs that move together to keep prices the same (trade-offs). | in-file: `_write_figures`, `run_analysis` |
| `cluster_summary` | Summarizes each cluster's size, spread and distance from the truth. | in-file: `run_analysis`; tests |
| `noise_summary` | Summarizes results at each noise level. | in-file: `run_analysis`; tests |
| `noise_compensation_pairs` | Trade-off pairs under noise. | in-file: `run_analysis` |
| `matched_noise_comparison` | Restricts clean-vs-noisy comparison to cases run at both. | in-file: `_write_figures`, `render_report`; tests |
| `_write_figures` | Draws the six figures. | in-file: `render_existing_outputs`, `run_analysis` |
| `render_report` | Writes the Markdown report. | in-file: `render_existing_outputs`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`; tests |
| `_read_json` | Reads a JSON object file. | in-file: `render_existing_outputs` |
| `render_existing_outputs` | Rebuilds report and figures from saved results. | in-file: `main`; `scripts/run_g2_complementary_observable_analysis.py`; tests |
| `run_analysis` | Runs the whole diagnostic end to end (64 pricing nodes only). | in-file: `main`; `scripts/run_g2_complementary_observable_analysis.py` |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_identifiability_analysis.py`

*1156 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

G2 experiment (Aug 11): with only the 20 market-supported prices, can the ten parameters be pinned down? Measures local "flatness" (Jacobian rank) and tries fits; result: 0/12 recoveries, gate not passed.

| Function / class | What it does | Used by |
|---|---|---|
| `Representation` | Describes one price layout (which strike distances and option types). | in-file: `build_grid`, `normalized_observables`, `scaled_parameter_jacobian`; `scripts/run_g2_information_remediation.py` |
| `_sha256` | Fingerprints a file. | in-file: `_snapshot_existing_g2`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py` |
| `_snapshot_existing_g2` | Fingerprints the earlier G2 evidence files. | in-file: `_assert_snapshot_unchanged`, `run_analysis`; `scripts/run_g2_information_remediation.py` |
| `_assert_snapshot_unchanged` | Fails if earlier G2 evidence changed. | in-file: `run_analysis`; `scripts/run_g2_information_remediation.py` |
| `inspect_carry_contract_sources` | Reads the config files to see whether interest-rate and dividend ("carry") inputs are settled. | in-file: `run_analysis`; tests |
| `build_grid` | Builds the list of strikes, maturities and call/put labels for a layout. | tests |
| `normalized_observables` | Prices the layout with the real engine and divides by the stock price. | in-file: `_one_recovery`, `run_recovery_experiment`, `scaled_parameter_jacobian`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py`; tests |
| `_parameter_widths` | Allowed range width per parameter. | in-file: `_maximin_select`, `_one_recovery`, `run_recovery_experiment`, `scaled_parameter_jacobian`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py` |
| `scaled_parameter_jacobian` | Measures how much each price moves when each parameter is nudged (the sensitivity table). | in-file: `run_jacobian_experiment`; `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`; tests |
| `_read_accepted` | Reads previously accepted parameter samples. | in-file: `select_representative_parameters` |
| `_maximin_select` | Picks samples spread as far apart as possible. | in-file: `select_representative_parameters` |
| `select_representative_parameters` | Chooses the representative truth cases used across G2. | in-file: `run_analysis`; `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py`; tests |
| `run_jacobian_experiment` | Computes rank, condition number and weakest directions for every case and layout. | in-file: `run_analysis` |
| `_deterministic_starts` | Repeatable list of starting guesses. | in-file: `_one_recovery`; `scripts/run_g2_information_remediation.py` |
| `_one_recovery` | Fits one surface from several starts and records each answer. | in-file: `run_recovery_experiment` |
| `run_recovery_experiment` | Runs recovery fits for all cases at 0, 0.5 and 1 percent noise. | in-file: `run_analysis` |
| `decide_gate` | Applies the pass/fail rule (rank, conditioning, recovery rate); it failed. | in-file: `run_analysis`; tests |
| `_atomic_write_bytes` | Writes a file safely via a temporary file. | in-file: `_write_csv`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py` |
| `_write_csv` | Safely writes CSV. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py` |
| `write_figures` | Draws the identifiability figures. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py` |
| `render_report` | Writes the Markdown report. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py` |
| `run_analysis` | Runs the whole analysis. | in-file: `main`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_global_ambiguity_analysis.py`, `scripts/run_g2_information_remediation.py`, `scripts/run_g2_multi_date_identifiability.py`; tests |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_information_remediation.py`

*1491 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

G2 experiment (Aug 11): does adding a third (far) expiry fix recovery? Conditioning got much better but clean recovery was only 2/12 and noisy 0/12, and the far expiry barely trades, so it was not admitted.

| Function / class | What it does | Used by |
|---|---|---|
| `ExperimentSpec` | Describes the two-expiry or three-expiry layout being compared. | in-file: `_recovery_rows` |
| `ExperimentSpec.candidate_input_dimension` | Counts how many numbers the layout would feed a network. | in-file: `experiment_matrix`, `render_report`, `run_jacobian_comparison`; tests |
| `_sha256` | Fingerprints a file. | in-file: `_snapshot_paths`, `run_analysis` |
| `_snapshot_paths` | Fingerprints a list of earlier files, failing if any is missing. | in-file: `run_analysis` |
| `_atomic_write_bytes` | Writes a file safely. | in-file: `_write_csv`, `run_analysis` |
| `_write_csv` | Safely writes CSV. | in-file: `run_analysis` |
| `experiment_matrix` | Table of the layouts being compared. | in-file: `run_analysis`; tests |
| `_central5_rows` | Picks far-expiry rows at the five central strike distances. | in-file: `build_far_expiry_support` |
| `build_far_expiry_support` | Measures whether the far expiry is listed, actively traded, and usable. | in-file: `run_analysis`; tests |
| `_scaled_jacobian_with_step` | Independent sensitivity calculation with an explicit nudge size, as a cross-check. | in-file: `independent_baseline_spot_check` |
| `independent_baseline_spot_check` | Confirms the earlier sensitivity numbers do not depend on nudge size. | in-file: `run_analysis` |
| `run_jacobian_comparison` | Compares rank and conditioning for two versus three expiries. | in-file: `run_analysis` |
| `material_improvement` | Decides whether the third expiry improved conditioning by a meaningful amount. | in-file: `run_analysis`; tests |
| `_recovery_rows` | Fits one case from several starts for one layout. | in-file: `run_recovery_comparison`; tests |
| `coupled_recovery_noise` | Makes noise where the first 20 shocks match the two-expiry run, so comparisons are fair. | in-file: `run_recovery_comparison`; tests |
| `run_recovery_comparison` | Runs recovery fits for both layouts at each noise level. | in-file: `run_analysis` |
| `_recovery_frequency` | Fraction of successful recoveries for one layout and noise level. | in-file: `classify_g2`, `write_figures` |
| `classify_g2` | Applies the gate rules to the three-expiry option; it did not pass. | in-file: `run_analysis`; tests |
| `_discount_source_section` | Writes the report section on where interest-rate data came from (or that it is unresolved). | in-file: `run_analysis`; tests |
| `write_figures` | Draws comparison figures. | in-file: `run_analysis` |
| `render_report` | Writes the Markdown report. | in-file: `run_analysis` |
| `run_analysis` | Runs the whole comparison. | in-file: `main` |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_multi_date_identifiability.py`

*1628 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

G2 experiment (Aug 11): fit several dates at once, sharing the eight structural parameters (designs A-D, some with exact CIR variance paths); conditioning improved but recovery stayed poor: MULTI_DATE_DIAGNOSTIC = INSUFFICIENT.

| Function / class | What it does | Used by |
|---|---|---|
| `Design` | Describes one multi-date design (how many dates, whether later variance states are known or estimated). | in-file: `_decode_latent`, `deterministic_starts`, `joint_normalized_prices`, `scaled_price_jacobians`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `Design.unknown_count` | How many unknowns the design has (10, or 14 with estimated later states). | in-file: `deterministic_starts`, `experiment_designs`, `run_recovery` |
| `_sha256` | Fingerprints a file. | in-file: `_protected_snapshot`, `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `_protected_snapshot` | Fingerprints earlier evidence. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py`; tests |
| `_atomic_write_bytes` | Writes a file safely. | in-file: `_write_csv`, `_write_json`, `render_report`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `_write_csv` | Safely writes CSV. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `_write_json` | Safely writes JSON. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `experiment_designs` | Table of the four designs. | in-file: `run_analysis`; `scripts/run_g2_multi_date_replication.py`; tests |
| `_target_widths` | Allowed range widths of the ten main parameters. | in-file: `_joint_widths`, `run_recovery`, `scaled_price_jacobians` |
| `_nuisance_widths` | Range widths of the extra later-date variance states. | in-file: `_joint_widths`, `run_recovery` |
| `_joint_widths` | Both sets of widths together. | in-file: `scaled_price_jacobians`, `scaled_transition_scores` |
| `cir_transition_parameters` | Gives the exact probability-distribution settings for how a variance factor moves over a time step. | in-file: `exact_cir_transition_from_uniform`, `exact_cir_transition_logpdf` |
| `exact_cir_transition_from_uniform` | Turns a random number into an exact simulated next variance value. | in-file: `simulate_state_paths`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py`; tests |
| `exact_cir_transition_logpdf` | How likely an observed variance move is under the model. | in-file: `transition_logpdf_vector`; tests |
| `_factor_parameters` | Picks out kappa, theta, sigma for the slow or fast factor. | in-file: `simulate_state_paths`, `transition_logpdf_vector`; `scripts/run_g2_multi_date_replication.py` |
| `simulate_state_paths` | Simulates the true variance states on the later dates for each case. | in-file: `run_analysis`; tests |
| `state_vector` | Reads the four later-date variance states from a row. | in-file: `run_identifiability`, `run_recovery` |
| `_parameters_at_date` | Builds the parameter vector for a given date (same structure, date-specific variances). | in-file: `joint_normalized_prices` |
| `joint_normalized_prices` | Prices all dates and joins them into one long vector. | in-file: `run_identifiability`, `run_recovery`, `scaled_price_jacobians`; tests |
| `transition_logpdf_vector` | Likelihood of the variance moves between dates, factor by factor. | in-file: `run_recovery`, `scaled_transition_scores` |
| `_valid_joint` | Checks parameters and states are all valid. | in-file: `scaled_price_jacobians`, `scaled_transition_scores` |
| `scaled_price_jacobians` | Sensitivity of joint prices to parameters and to the extra states. | in-file: `run_identifiability` |
| `nuisance_projected_jacobian` | Removes the part of the sensitivity that the extra states could explain, to see what is left for the real parameters. | in-file: `run_identifiability`; tests |
| `scaled_transition_scores` | Sensitivity of the variance-move likelihood to each unknown. | in-file: `run_identifiability` |
| `_metrics_from_singular_values` | Turns singular values into rank and condition-number metrics. | in-file: `run_identifiability` |
| `_profile_physics_information` | Combines price and variance-path information to judge identifiability. | in-file: `run_identifiability` |
| `run_identifiability` | Computes those metrics for every case and design. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `_decode_latent` | Turns optimizer numbers back into parameters and states. | in-file: `run_recovery`; tests |
| `_state_to_latent` | Turns states into optimizer numbers (the reverse). | no caller found by search |
| `deterministic_starts` | Repeatable starting guesses sized for each design. | in-file: `run_recovery` |
| `coupled_noise` | Makes noise where the first date's shocks match the single-date study. | in-file: `run_recovery`; tests |
| `_state_bound_hits` | Lists extra states that ended at their limits. | in-file: `run_recovery` |
| `run_recovery` | Runs recovery fits for each design at each noise level. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `classify_diagnostic` | Applies the pass/fail rules; verdict INSUFFICIENT. | in-file: `run_analysis`; `scripts/run_g2_multi_date_replication.py` |
| `_save_figure` | Saves a figure safely. | in-file: `write_figures`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `write_figures` | Draws the design comparison figures. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `_summary_table` | Builds the summary Markdown table. | in-file: `render_report` |
| `render_report` | Writes the Markdown report. | in-file: `run_analysis`; `scripts/run_g2_complementary_observable_analysis.py`, `scripts/run_g2_multi_date_replication.py` |
| `run_analysis` | Runs the whole diagnostic. | in-file: `main`; `scripts/run_g2_complementary_observable_analysis.py`; tests |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_multi_date_replication.py`

*799 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Re-runs the multi-date study with a different random seed only, to see if the conclusion holds; result REPLICATION = MIXED but the core recovery failure replicated.

| Function / class | What it does | Used by |
|---|---|---|
| `_protected_snapshot` | Fingerprints earlier evidence. | in-file: `run_replication` |
| `replication_contract` | States which seed changes and checks everything else is frozen. | in-file: `run_replication`; tests |
| `simulate_replication_state_paths` | Simulates new variance paths with the new seed. | in-file: `run_replication`; tests |
| `_load_original_evidence` | Loads the original study's results. | in-file: `run_replication` |
| `_ident_metrics` | Pulls identifiability metrics for one design. | in-file: `build_comparison` |
| `_recovery_metrics` | Pulls recovery metrics for one design and noise level. | in-file: `build_comparison` |
| `weakest_direction_stability` | Checks whether the flattest direction is the same in both runs. | in-file: `run_replication`; tests |
| `build_comparison` | Puts original and replication results side by side. | in-file: `run_replication` |
| `classify_hypotheses` | Marks each of four hypotheses as replicated or not. | in-file: `run_replication`; tests |
| `decide_replication` | Gives the overall replication verdict. | in-file: `run_replication`; tests |
| `_save_figure` | Saves a figure. | in-file: `write_figures` |
| `write_figures` | Draws comparison figures. | in-file: `run_replication` |
| `_format_pass` | Formats a pass rate as "n/6". | in-file: `render_report` |
| `render_report` | Writes the Markdown report. | in-file: `run_replication` |
| `run_replication` | Runs the whole replication. | in-file: `main` |
| `main` | Command-line entry point. | command line |

#### `scripts/run_g2_r2r3_diagnostics.py`

*319 lines; newest version 5d74ead (2026-08-22, virajchoudhary); on origin/main.*

Final step of the R2-vs-R3 study (Aug 22): reads all 1,920 fit results, computes the summary statistics and mechanically applies the pre-written decision rule, which froze R2.

| Function / class | What it does | Used by |
|---|---|---|
| `load_profiles` | Loads the five NTPC date profiles (spot, rates, maturities). | in-file: `main` |
| `factor_swap_check` | Confirms that swapping the slow and fast blocks gives the same prices (to about 1e-14) and that the "slow must be slower" rule rejects the swapped twin. | in-file: `main` |
| `hard_requirements` | Checks the must-have market-construction rules for each candidate layout. | in-file: `main` |
| `main` | Loads every run, computes Jacobian, dispersion and noise summaries, applies the decision rule and writes the final decision files. | command line |

#### `scripts/run_g2_r2r3_market_support.py`

*118 lines; newest version 0920f87 (2026-08-22, virajchoudhary); on origin/main.*

Checkpoint C (Aug 22): counts how many of the 20 (R2) or 30 (R3) price slots really trade on each of five NTPC dates; R2 got 78/100, R3's third expiry got zero.

| Function / class | What it does | Used by |
|---|---|---|
| `run_audit` | Audits all five dates, writes the slot table, date profiles and summary. | no caller found by search |

#### `scripts/run_g2_r2r3_matrix.py`

*111 lines; newest version 28ccc46 (2026-08-22, virajchoudhary); on origin/main.*

Runs the big R2/R3 fitting matrix (20 truths x 2 layouts x 4 noise levels x 12 starts = 1,920 fits), split into shards for parallel machines.

| Function / class | What it does | Used by |
|---|---|---|
| `load_profiles` | Loads the date profiles. | in-file: `run_matrix` |
| `run_matrix` | Runs this shard's share of the cells and appends each result to a log. | no caller found by search |

#### `scripts/run_g2_r2r3_merge.py`

*102 lines; newest version 28ccc46 (2026-08-22, virajchoudhary); on origin/main.*

Joins the six shard logs into one canonical results file and checks no run is missing or duplicated with different content.

| Function / class | What it does | Used by |
|---|---|---|
| `_canonical_form` | Strips timing fields so two copies of the same run can be compared. | in-file: `merge` |
| `merge` | Merges all shard records, refusing conflicts, and writes the combined file. | no caller found by search |

#### `scripts/run_g2_r2r3_smoke.py`

*96 lines; newest version 1aad179 (2026-08-22, virajchoudhary); on origin/main.*

A tiny trial run of the R2/R3 harness to prove it works before the full matrix.

| Function / class | What it does | Used by |
|---|---|---|
| `run_smoke` | Runs one date and a few fits and reports timings and checks. | no caller found by search |

#### `scripts/run_g2_r2r3_truth_panel.py`

*121 lines; newest version 6986281 (2026-08-22, virajchoudhary); on origin/main.*

Checkpoint A: fixes ("freezes") the 20 true parameter vectors before any R2/R3 result is computed, and saves their fingerprint.

| Function / class | What it does | Used by |
|---|---|---|
| `environment_metadata` | Records Python, library and platform versions. | in-file: `freeze_panel` |
| `_git_version` | Asks git for its version. | in-file: `environment_metadata` |
| `freeze_panel` | Builds the 20-truth panel, writes it, and writes a manifest with fingerprints. | no caller found by search |

#### `scripts/run_g8_readiness_pipeline.py`

*70 lines; newest version 746060b (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Thin wrapper for G8 readiness: by default only replays a synthetic fixture; real acquisition or final evaluation need explicit authorization flags.

#### `scripts/run_model3_clean_evaluation.py`

*58 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Thin wrapper for a future, authorized Model 3 clean-test evaluation; stays sealed without explicit flags.

#### `scripts/run_model3_pde_pilot.py`

*1237 lines; newest version ff2d7d1 (2026-08-26, virajchoudhary); on research/mentor-dh-pinn-basics.*

The Model 3 (genuine physics-equation network) training driver for the Stage A pilot and Stage B research runs, with crash-safe resume and strict identity checks; never actually run for research.

| Function / class | What it does | Used by |
|---|---|---|
| `PilotSettings` | All run settings in one record; checks they match the frozen protocol. | in-file: `checkpoint_metadata`, `environment_provenance`, `evaluate_batch`, `main` ...; tests |
| `PilotSettings.__post_init__` | Validates every setting (positive counts, allowed seeds 11/22/33, run kind) as soon as settings are created. | Python (automatic) |
| `RunIdentity` | The run's identity card: git commit, config and dataset fingerprints, dirty-tree status. | in-file: `build_run_identity`, `checkpoint_metadata`, `environment_provenance`; tests |
| `RunIdentity.payload` | Turns the identity card into a dictionary for saving. | in-file: `_atomic_json`, `_atomic_torch_save`, `checkpoint_metadata`, `environment_provenance`; tests |
| `sha256_file` | Fingerprints a file. | in-file: `build_run_identity`; tests |
| `current_git_sha` | Gets the current commit id, failing if unavailable. | in-file: `build_run_identity`; tests |
| `current_git_dirty_state` | Checks whether tracked files have uncommitted changes. | in-file: `build_run_identity`; tests |
| `build_run_identity` | Builds the identity card, refusing a wrong dataset or (for real runs) a dirty tree. | in-file: `run_pilot`; tests |
| `load_pilot_dataset` | Loads only train and validation surfaces in file order, never test. | in-file: `run_pilot`; tests |
| `fit_train_only_standardizer` | Fits target scaling on training rows only. | in-file: `run_pilot`; tests |
| `build_system` | Creates the Model 3 network pair in training mode. | in-file: `run_pilot`; tests |
| `build_optimizer` | Creates the AdamW optimizer with the frozen settings. | in-file: `run_pilot`; tests |
| `make_batches` | Shuffles row numbers with a seeded generator and cuts them into batches. | in-file: `run_pilot`, `run_validation_epoch` |
| `subset_signature` | Records exactly which surfaces were used, for later proof. | in-file: `run_pilot` |
| `checkpoint_metadata` | Builds the label and settings saved with each checkpoint. | in-file: `run_pilot` |
| `validate_resume_identity` | Refuses to resume if anything about the run changed. | in-file: `run_pilot`; tests |
| `capture_rng_states` | Saves all random-number generator states so a resumed run is identical. | in-file: `run_pilot`; tests |
| `restore_rng_states` | Restores those states, refusing if any is missing. | in-file: `run_pilot`; tests |
| `validate_checkpoint_pair` | Checks the model checkpoint and optimizer file belong to the same epoch. | in-file: `run_pilot`; tests |
| `_values_exactly_equal` | Deep exact-equality test for nested values and tensors. | in-file: `validate_checkpoint_pair` |
| `_atomic_json` | Writes JSON safely via a temporary file. | in-file: `run_pilot` |
| `_json_default` | Converts paths, numpy numbers and tensors into JSON-friendly values. | in-file: `_atomic_json`, `main` |
| `_atomic_torch_save` | Saves a PyTorch file safely via a temporary file. | in-file: `run_pilot` |
| `_atomic_csv` | Writes a CSV safely via a temporary file. | in-file: `run_pilot` |
| `read_history` | Reads an existing training-history CSV and checks its columns. | in-file: `run_pilot` |
| `validate_history_consistency` | Checks histories have exactly one row per finished epoch and correct batch counts. | in-file: `run_pilot`; tests |
| `environment_provenance` | Records library versions and hardware. | in-file: `run_pilot`; tests |
| `_device_leaf` | Moves numbers to the device so the physics derivatives can still be taken. | in-file: `_terminal_state`, `evaluate_batch`; tests |
| `_terminal_state` | Samples expiry-time points (tau = 0) used to check the payoff condition. | in-file: `evaluate_batch` |
| `evaluate_batch` | Computes the three-part loss (parameter error, price reconstruction, physics residual) for one batch. | in-file: `run_pilot`, `run_validation_epoch`; tests |
| `run_validation_epoch` | Computes average validation losses without training. | in-file: `run_pilot`; tests |
| `run_pilot` | The main training loop with checkpointing and resume; never opens the test split. | in-file: `main`; tests |
| `smoke_settings` | Tiny settings (2 train, 2 validation surfaces) for a wiring test. | tests |
| `build_argument_parser` | Defines the command-line options. | in-file: `main`; tests |
| `main` | Command-line entry point. | command line |

#### `scripts/run_model3_pde_smoke.py`

*73 lines; newest version 02c2a2c (2026-08-25, virajchoudhary); on research/mentor-dh-pinn-basics.*

Development-only check that the Model 3 physics machinery produces finite, sensible numbers on four fixed parameter sets.

| Function / class | What it does | Used by |
|---|---|---|
| `main` | Builds a small batch, samples physics points, evaluates the loss and reports PASSED or FAILED. | command line |

#### `scripts/run_nse_stage_a_screen.py`

*58 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Thin wrapper: runs the official-NSE Stage A three-date coverage screen (download or offline).

#### `scripts/run_ntpc_canonical160_causal_probe.py`

*156 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Re-fits NTPC start number 6 under controlled one-field changes to prove the reproduction mismatch was caused by rounding of maturity T in a CSV.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Fingerprints a file. | in-file: `_fit`, `run` |
| `_fit` | Fits the NTPC calibration rows from one start. | in-file: `run` |
| `_vector` | Reads the ten parameters from a row. | in-file: `run` |
| `_decorate` | Adds "distance from original and from replay" to a result. | in-file: `run` |
| `run` | Runs every probe and writes the evidence file. | no caller found by search |

#### `scripts/run_ntpc_dh_multi_date_calibration.py`

*553 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

NTPC real-market experiment (Aug 12): fits Double Heston to three NTPC dates at once (shared structure, date-specific variances) to see if more data stabilizes the answer; per docs it did not settle the instability.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Fingerprints a file. | in-file: `_date_panel`, `_publish`, `joint_start_population`, `validated_rate_contract` ...; tests |
| `write_json` | Writes sorted JSON. | in-file: `_publish` |
| `validated_rate_contract` | Checks the saved RBI T-bill pages match their fingerprints and that no future rate is used for a past date. | in-file: `_date_panel`, `_publish`; tests |
| `_raw_paths` | Builds the file names of the NSE cash and F&O bhavcopies for a date. | in-file: `_date_panel`, `_publish` |
| `_assign_targets` | Matches real strikes to the five target strike distances with an optimal one-to-one assignment. | in-file: `_date_panel` |
| `_date_panel` | Builds the cleaned NTPC option table for one date (spot, expiries, strikes, rates, carry, implied vol). | in-file: `build_three_date_panel` |
| `build_three_date_panel` | Joins the three dates' tables. | in-file: `run`; tests |
| `support_inventory` | Counts which of the 60 target cells were found or missing. | in-file: `_publish`, `run`; tests |
| `map_joint_coordinate` | Turns 14 optimizer numbers into eight shared parameters plus each date's two variances. | in-file: `_fit`; tests |
| `canonical_vector` | Builds the normal ten-parameter vector for one date. | in-file: `_figures`, `_fit`; tests |
| `date_balanced_residual` | Joins the dates' price errors so each date counts equally. | in-file: `_fit`; tests |
| `reported_date_balanced_objective` | One overall error score with dates weighted equally. | in-file: `_fit`; tests |
| `joint_start_population` | Builds the 12 fixed starting points. | in-file: `_fit`, `run`; tests |
| `classify_multi_date` | Labels the result as strong, partial or insufficient improvement. | in-file: `_publish`; tests |
| `_price` | Prices a set of option rows for one parameter vector. | in-file: `_figures`, `_fit` |
| `_metrics` | Price and implied-volatility error metrics for priced rows. | in-file: `_fit` |
| `_fit` | Runs the 12-start joint fit. | in-file: `run` |
| `_stability` | Measures how spread out the near-best solutions are. | in-file: `_figures`, `_publish`, `single_date_shared_baseline`; tests |
| `single_date_shared_baseline` | Computes the same spread for the earlier single-date fit, for comparison. | in-file: `_publish`; tests |
| `_figures` | Draws the support, fit and spread figures. | in-file: `_publish` |
| `_overall_rmse` | Combines per-date errors into one overall error. | in-file: `_publish` |
| `_publish` | Computes all summaries, classifies, writes report, figures and manifest. | in-file: `render_existing_outputs`, `run` |
| `run` | Builds the panel, runs the fits, publishes. | no caller found by search |
| `render_existing_outputs` | Re-publishes from saved fit results only. | no caller found by search |
| `verify_manifest_artifacts` | Lists any file whose fingerprint no longer matches the manifest. | tests |
| `render_report` | Writes the Markdown report. | in-file: `_publish` |

#### `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`

*910 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

NTPC experiment (Aug 12-15): doubles the optimizer's step budget from 160 to 320 to test whether instability was just the optimizer running out of steps; result OPTIMIZER_CAP_UNRESOLVED (10/12 still hit the cap, spread persisted).

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Fingerprints a file. | in-file: `_corrected_control_artifacts`, `_publish`, `_role_hash`, `frozen_starts` ...; tests |
| `write_json` | Writes JSON safely. | in-file: `_publish` |
| `write_csv` | Writes CSV. | in-file: `_publish`, `resume_320_from_verified_160`, `run` |
| `protected_evidence_hashes` | Fingerprints the earlier reviewed NTPC evidence. | in-file: `_publish`, `render_existing_outputs`, `resume_320_from_verified_160`, `run`; tests |
| `_role_hash` | Fingerprints the calibration or holdout rows. | in-file: `verify_frozen_contract` |
| `load_frozen_selected_options` | Loads frozen NTPC rows, rebuilding maturity T exactly (the rounding fix). | in-file: `resume_320_from_verified_160`, `run`, `verify_frozen_contract`; tests |
| `verify_frozen_contract` | Checks rows, prices and inputs are exactly the frozen ones. | in-file: `render_existing_outputs`, `resume_320_from_verified_160`, `run`; tests |
| `optimizer_contract` | Returns the optimizer settings for a 160 or 320 budget. | in-file: `_fit_cell`, `_publish`, `validate_160_control_frames`, `validate_experiment_cells`; tests |
| `optimizer_contracts_differ_only_by_budget` | Checks the two settings differ only in the step budget. | in-file: `validate_160_control_frames`, `validate_experiment_cells`; tests |
| `reached_cap` | True if a fit used its whole step budget. | in-file: `_fit_cell`, `paired_analysis`, `stability_metrics`; tests |
| `canonical_from_coordinate` | Turns optimizer numbers into the ten parameters. | in-file: `_fit_cell`, `frozen_starts`; tests |
| `frozen_starts` | Builds the 12 fixed starts in both coordinate systems. | in-file: `resume_320_from_verified_160`, `run`; tests |
| `price_rows` | Prices option rows with given parameters. | in-file: `_fit_cell` |
| `_fit_cell` | Runs all 12 fits for one coordinate system and budget. | in-file: `resume_320_from_verified_160`, `run` |
| `stability_metrics` | Measures cap rate and spread of near-best solutions. | in-file: `_publish`, `summarize_160_frames`; tests |
| `cell_pricing` | Best fit's calibration and holdout errors. | in-file: `_publish`, `summarize_160_frames` |
| `classify_cap` | Labels how much the cap rate fell. | in-file: `_publish`, `classify_final`; tests |
| `_reduction` | Fractional reduction from old to new. | in-file: `classify_dispersion` |
| `classify_dispersion` | Labels whether the spread collapsed, improved or persisted. | in-file: `_publish`; tests |
| `classify_final` | Gives the overall verdict. | in-file: `_publish`; tests |
| `historical_160_comparison` | Checks the 160-step control reproduces the earlier reviewed numbers. | in-file: `_publish`, `summarize_160_frames`; tests |
| `_validate_cell_frame` | Checks a results table has the right columns and starts. | in-file: `validate_160_control_frames`, `validate_experiment_cells` |
| `validate_160_control_frames` | Validates both 160-step control tables. | in-file: `_publish`, `resume_320_from_verified_160`, `run`; tests |
| `require_valid_160_controls` | Stops if the controls failed. | in-file: `resume_320_from_verified_160`, `run`; tests |
| `validate_experiment_cells` | Validates all four cells. | in-file: `_publish`; tests |
| `summarize_160_frames` | Summarizes the 160 controls and compares them to history. | in-file: `resume_320_from_verified_160`, `run` |
| `paired_analysis` | Compares each start's 160 and 320 answers. | in-file: `_publish` |
| `percentage_changes` | Percentage change in errors and runtime from 160 to 320. | in-file: `_publish` |
| `_plots` | Draws comparison bar charts. | in-file: `_publish` |
| `_fmt` | Formats a number to nine significant digits. | in-file: `render_report` |
| `render_report` | Writes the Markdown report. | in-file: `_publish` |
| `_corrected_control_artifacts` | Lists the control files with their fingerprints. | in-file: `_publish`, `validate_corrected_control_artifacts` |
| `_expected_control_hashes_from_manifest` | Reads expected control fingerprints from the manifest. | in-file: `render_existing_outputs`, `resume_320_from_verified_160` |
| `_expected_start_hashes_from_manifest` | Reads expected start-file fingerprints from the manifest. | in-file: `render_existing_outputs`; tests |
| `_require_prior_protected_seal` | Stops if protected evidence differs from the sealed manifest. | in-file: `render_existing_outputs`, `resume_320_from_verified_160`, `run`; tests |
| `validate_corrected_control_artifacts` | Checks control files match expected fingerprints. | in-file: `_publish`, `resume_320_from_verified_160` |
| `validate_persisted_start_artifacts` | Checks saved start files match expected fingerprints. | in-file: `render_existing_outputs`; tests |
| `_publish` | Computes all metrics and verdicts and writes outputs. | in-file: `render_existing_outputs`, `resume_320_from_verified_160`, `run` |
| `run` | Runs the full experiment. | no caller found by search |
| `resume_320_from_verified_160` | Continues with the 320 runs after an interruption, reusing verified 160 results. | no caller found by search |
| `render_existing_outputs` | Re-publishes from saved results. | tests |
| `verify_manifest_artifacts` | Lists files whose fingerprints no longer match. | tests |

#### `scripts/run_ntpc_dh_stability_reparameterization.py`

*982 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

NTPC experiment (Aug 12): re-expresses the same ten parameters in "total plus share" coordinates to see if the optimizer becomes more stable; pricing was preserved but instability was not removed.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Fingerprints a file. | in-file: `paired_start_population`, `protected_hashes`, `render_existing_outputs`, `run` ...; `scripts/run_ntpc_canonical160_causal_probe.py`, `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`; tests |
| `write_json` | Writes JSON safely. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py` |
| `write_csv` | Writes CSV safely. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py` |
| `protected_hashes` | Fingerprints the protected baseline files. | in-file: `render_existing_outputs`, `run`, `verify_baseline_contract` |
| `verify_baseline_contract` | Checks the baseline files are unchanged. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_canonical160_causal_probe.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py` |
| `paired_start_population` | Makes the same 12 starts in both coordinate systems and checks the round trip. | in-file: `run`; `scripts/run_ntpc_canonical160_causal_probe.py`, `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`; tests |
| `equivalence_audit` | Tests 2,029 vectors to prove the new coordinates cover exactly the same space (0 failures). | in-file: `run` |
| `price_rows` | Prices rows with given parameters. | in-file: `run_transformed_calibrations`; `scripts/run_ntpc_canonical160_causal_probe.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`; tests |
| `boundary_margins` | How close a solution is to the limits. | in-file: `run_transformed_calibrations`; `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py` |
| `run_transformed_calibrations` | Runs the 12 fits in the new coordinates. | in-file: `run` |
| `stability_metrics` | Measures the spread of near-best solutions. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`; tests |
| `classify_experiment` | Applies the pre-written rules to label the outcome. | in-file: `render_existing_outputs`, `run`; tests |
| `_plot_outputs` | Draws the comparison figures. | in-file: `render_existing_outputs`, `run` |
| `_summary_table` | Builds the baseline-vs-new metric table. | in-file: `render_report` |
| `render_report` | Writes the Markdown report. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py` |
| `load_frozen_selected_options` | Loads frozen rows, restoring exact maturity values. | in-file: `run`; `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`; tests |
| `run` | Runs the experiment end to end. | no caller found by search |
| `render_existing_outputs` | Re-publishes from saved results. | `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`; tests |

#### `scripts/run_ntpc_single_stock_pilot.py`

*1374 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

The first real-market test (Aug 12): calibrates Black-Scholes, Heston and Double Heston to NTPC options on 15 July 2026 and compares them on held-out options; verdict NO_CLEAR_WINNER and Double Heston multi-start answers were unstable.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Fingerprints a file. | in-file: `build_option_dataset`, `render_existing_outputs`, `render_reports`, `run`; `scripts/run_ntpc_canonical160_causal_probe.py`, `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |
| `atomic_write` | Writes bytes via a temporary file. | in-file: `_acquire_corporate_actions`, `render_reports`, `write_csv`, `write_json` |
| `write_json` | Writes JSON safely. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py` |
| `write_csv` | Writes CSV safely. | in-file: `run`; `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py` |
| `dte_and_time` | Days to expiry and the same in years. | in-file: `_figures`, `_futures_contract`, `volatility_comparison`; tests |
| `log_moneyness` | log(strike / spot), how far a strike is from the current price. | `scripts/run_ntpc_canonical160_causal_probe.py`, `scripts/run_ntpc_dh_multi_date_calibration.py`; tests |
| `discount_from_simple_yield` | Turns a T-bill simple yield into a discount factor. | in-file: `_futures_contract` |
| `carry_from_forward` | Backs out the interest rate and the futures-implied carry (dividend-like yield). | in-file: `_futures_contract` |
| `forward_black_price` | Black's formula price of a call or put from the futures price. | in-file: `fit_black_scholes`, `implied_volatility`; tests |
| `no_arbitrage_bounds` | Lowest and highest prices an option can sensibly have. | in-file: `implied_volatility` |
| `implied_volatility` | Finds the volatility that makes Black's formula match a market price. | in-file: `_metrics`, `_prediction_frame`, `build_option_dataset`; `scripts/run_ntpc_dh_multi_date_calibration.py`; tests |
| `realized_volatility` | Annualized volatility from a series of daily closing prices. | in-file: `build_realized_history`, `render_existing_outputs`; tests |
| `expected_average_variance` | Heston's expected average variance up to a maturity. | in-file: `volatility_comparison`; tests |
| `half_life` | How long a variance shock takes to halve, from kappa. | in-file: `_figures`, `render_reports`; tests |
| `_option_type` | Turns NSE codes CE/PE into call/put. | in-file: `build_option_dataset` |
| `_fo_paths` | Paths to the 15 July F&O bhavcopy file and its zip. | in-file: `build_option_dataset`, `render_reports`, `run`; tests |
| `_acquire_corporate_actions` | Loads or downloads NTPC corporate actions to check no split or dividend distorted prices. | in-file: `build_realized_history` |
| `_futures_contract` | Reads the matching NTPC futures for each expiry and derives rate and carry. | in-file: `build_option_dataset` |
| `_unique_target_assignment` | Matches real strikes to the five target strike distances one to one. | in-file: `build_option_dataset` |
| `build_option_dataset` | Builds the cleaned NTPC option table and splits it into 12 calibration and 7 holdout options. | in-file: `run`; `scripts/run_ntpc_canonical160_causal_probe.py`; tests |
| `_rows` | Turns a table into a list of row dictionaries. | in-file: `run` |
| `_price_rows` | Prices every row with a given price function. | in-file: `fit_black_scholes`, `fit_stochastic_model`; `scripts/run_ntpc_dh_stability_reparameterization.py` |
| `_metrics` | Price and implied-volatility error metrics. | in-file: `fit_black_scholes`, `fit_stochastic_model`; `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py` |
| `fit_black_scholes` | Fits one flat volatility (Black-Scholes) to the calibration options. | in-file: `run`; tests |
| `_heston_parameters` | Turns optimizer numbers into valid single-Heston parameters. | in-file: `fit_stochastic_model` |
| `price_heston_option` | Prices one option under single-factor Heston. | in-file: `fit_stochastic_model`; tests |
| `_double_heston_row_price` | Prices one option row under Double Heston using the main engine. | in-file: `fit_stochastic_model`; `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_stability_reparameterization.py` |
| `_prediction_frame` | Table of observed vs predicted prices for one model. | in-file: `fit_black_scholes`, `fit_stochastic_model` |
| `fit_stochastic_model` | Multi-start fit of Heston or Double Heston, keeping every start. | in-file: `run`; tests |
| `_heston_boundary_reasons` | Notes when a Heston fit sits near a constraint edge. | in-file: `fit_stochastic_model` |
| `model_winner` | Declares a winner only if it beats the runner-up on holdout by at least the pre-set margin. | in-file: `run` |
| `parameter_stability` | Measures how different the near-best Double Heston starts are. | in-file: `run`; `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py` |
| `build_realized_history` | Downloads NTPC closing prices to compute later realized volatility. | in-file: `run` |
| `volatility_comparison` | Compares each model's implied average volatility with what NTPC actually did. | in-file: `run` |
| `_figures` | Draws the pilot figures. | in-file: `render_existing_outputs`, `run`; `scripts/run_ntpc_dh_multi_date_calibration.py` |
| `_markdown_table` | Turns a table into Markdown. | in-file: `render_reports` |
| `render_reports` | Writes the pilot report and mentor checkpoint. | in-file: `render_existing_outputs`, `run` |
| `run` | Runs the whole pilot. | no caller found by search |
| `render_existing_outputs` | Re-renders reports from saved results and checks nothing changed. | `scripts/run_ntpc_dh_multi_date_calibration.py`, `scripts/run_ntpc_dh_optimizer_cap_sensitivity.py`, `scripts/run_ntpc_dh_stability_reparameterization.py`; tests |

#### `scripts/run_ood_boundary_evaluation.py`

*7 lines; newest version 84d70e1 (2026-08-26, virajchoudhary); on research/ood-boundary-protocol.*

Thin wrapper: command-line entry for the sealed OOD evaluation harness.

#### `scripts/run_power_candidate_tiebreak_extension.py`

*769 lines; written by Viraj (Aug 6-13); newest copy is Dhruva's Aug 21 bulk re-import 8e281ba, byte-identical to Viraj's main at merge 8a6b009; on origin/main.*

Stage A tie-break (Aug 10): NTPC and POWERGRID were tied for the Power-sector slot, so five July Wednesdays of official NSE data were compared; NTPC was chosen at moderate confidence.

| Function / class | What it does | Used by |
|---|---|---|
| `_date_text` | Joins dates into one text. | in-file: `_evidence_rows`, `main` |
| `_pct` | Formats a percentage. | in-file: `_evidence_rows` |
| `_number` | Formats a number to six decimals. | in-file: `_evidence_rows` |
| `_summary` | Builds an "overall and by date" summary string. | in-file: `_evidence_rows` |
| `_expiry_summary` | Summarizes listed expiries per date. | in-file: `_evidence_rows` |
| `_dominance` | Counts on how many dates each candidate was better. | in-file: `_evidence_rows` |
| `_overall_better` | Says which candidate is better overall, or tie. | in-file: `_evidence_rows` |
| `_support` | Which central strike distances fall inside the quoted strike range. | in-file: `_slice_support_metrics` |
| `_slice_support_metrics` | How often all five central strikes were covered, per date. | in-file: `_metric_payloads` |
| `_row_coverage` | Percentage of rows with a given property. | in-file: `_metric_payloads` |
| `_reported_coverage` | Percentage of rows with close, settlement, last and volume reported. | in-file: `_metric_payloads` |
| `_minimum_symmetric_reach` | How far out on both sides the strikes reach, worst case. | in-file: `_metric_payloads`; tests |
| `_metric_payloads` | Computes all comparison metrics for one candidate. | in-file: `_evidence_rows`; tests |
| `_expiry_lists` | Lists expiries per date. | in-file: `_evidence_rows`; tests |
| `_evidence_rows` | Builds the ranked evidence table for both candidates. | in-file: `main`; tests |
| `_stage_csv` | Writes a CSV to a temporary file first. | in-file: `_publish_csv_outputs_atomically`, `_write_csv_atomic` |
| `_write_csv_atomic` | Writes a CSV safely. | tests |
| `_publish_csv_outputs_atomically` | Writes the manifest and evidence together, rolling back on failure. | in-file: `main`; tests |
| `_manifest_rows` | Turns acquisition records into manifest rows. | in-file: `_acquire_extension_records`, `main`; tests |
| `_validate_original_manifest_row` | Checks one earlier archive record is intact. | in-file: `_validate_original_stage_a_provenance` |
| `_validate_original_stage_a_provenance` | Checks all original three-date sources before starting. | in-file: `main`; tests |
| `_acquire_extension_records` | Downloads (or loads offline) the extra dates' NSE archives. | in-file: `main`; tests |
| `_read_panel` | Reads NTPC and POWERGRID option data for all five dates. | in-file: `main` |
| `main` | Runs the tie-break and writes evidence. | command line |

#### `scripts/run_r2_noise_robustness.py`

*101 lines; newest version 46327f2 (2026-08-25, virajchoudhary); on codex/r2-noise-recovery.*

Operator entry point for the R2 noise study: make noisy cohorts, evaluate the networks, run traditional calibration per noise level, aggregate.

| Function / class | What it does | Used by |
|---|---|---|
| `_parse_levels` | Turns "gate" or "all" into a list of noise levels. | in-file: `main` |
| `main` | Defines the subcommands and runs the chosen one. | command line |

#### `scripts/validate_g8_protocol.py`

*278 lines; newest version 7eecc71 (2026-08-26, virajchoudhary); on research/g8-acquisition-readiness.*

Checks the frozen G8 final real-market protocol: config rules, that a stock/date was not used in development, and that the six saved network checkpoints match their fingerprints.

| Function / class | What it does | Used by |
|---|---|---|
| `G8ProtocolValidationError` | The error raised when a frozen G8 rule is broken. | in-file: `_load_mapping`, `_require`, `is_development_observation` |
| `_load_mapping` | Reads the YAML config and insists it is a mapping. | in-file: `validate_config` |
| `sha256_file` | Fingerprints a file. | in-file: `main`, `validate_config`, `verify_checkpoint_registry`; tests |
| `_require` | Raises the G8 error if a condition is false. | in-file: `validate_config` |
| `validate_config` | Checks all frozen config rules without touching market data. | in-file: `main`; tests |
| `is_development_observation` | Says whether a stock/date was already used during development (and so is banned from G8). | in-file: `check_candidate`; tests |
| `check_candidate` | Checks one proposed stock/date identity. | in-file: `main`; tests |
| `verify_checkpoint_registry` | Checks the frozen network files exist and match fingerprints, without running them. | in-file: `main`; tests |
| `main` | Command-line entry with subcommands. | command line |

#### `scripts/verify_model3_research_checkpoints.py`

*87 lines; newest version a2a9883 (2026-08-26, virajchoudhary); on research/model3-evaluation-readiness.*

Checks the three Model 3 Stage B seed runs are complete and frozen; never authorizes evaluation.

| Function / class | What it does | Used by |
|---|---|---|
| `_seed_roots` | Parses "seed=path" arguments, refusing duplicates. | in-file: `main` |
| `main` | Runs the freeze-gate check and reports pass or fail. | command line |

#### `scripts/verify_research_state.py`

*18 lines; newest version f5ffd01 (2026-08-25, virajchoudhary); on research/repro-orchestration.*

Thin wrapper: command-line check of the research "control plane" (which experiments are allowed to run next).

### scripts/cloud/

#### `scripts/cloud/benchmark_cloud.py`

*246 lines; newest version d7863ed (2026-08-23, virajchoudhary); on origin/main.*

Times the training step and the pricer on a cloud GPU to decide whether moving to the cloud is worth it; explicitly not a research result.

| Function / class | What it does | Used by |
|---|---|---|
| `_train_items` | Loads the first few training surfaces from the frozen 10k dataset. | in-file: `main` |
| `_batch_tensors` | Stacks those surfaces into tensors on the chosen device. | in-file: `_model2_step_time`, `_pricer_equivalence` |
| `_model2_step_time` | Times a few Model 2 training steps on the device. | in-file: `main` |
| `_pricer_equivalence` | Checks the fast vectorized pricer gives the same prices as the slow loop pricer on the device. | in-file: `main` |
| `_calibration_timing` | Times traditional calibration on two surfaces. | in-file: `main` |
| `main` | Runs the benchmark and writes a report, refusing to overwrite an old one. | command line |

#### `scripts/cloud/verify_environment.py`

*129 lines; newest version d7863ed (2026-08-23, virajchoudhary); on origin/main.*

Before a cloud run, checks the dataset, config and code fingerprints match the frozen ones and records the machine's details.

| Function / class | What it does | Used by |
|---|---|---|
| `_sha256` | Computes a file's fingerprint (SHA-256 hash). | in-file: `main` |
| `_git_sha` | Asks git for the current commit id, or says it is unavailable. | in-file: `main` |
| `_provenance_output` | Picks the output file name for this host and refuses to overwrite. | in-file: `main` |
| `main` | Does all the identity checks and writes the provenance record; exits with failure on any mismatch. | command line |

### scripts/model3_cloud/

#### `scripts/model3_cloud/launch_stage_a.py`

*113 lines; newest version ff2d7d1 (2026-08-26, virajchoudhary); on research/mentor-dh-pinn-basics.*

Starts the frozen Model 3 "Stage A" pilot on a cloud GPU, but only if the preflight check passes (never actually run: cloud login was missing).

| Function / class | What it does | Used by |
|---|---|---|
| `main` | Runs preflight, refuses if it fails, then launches the exact frozen pilot command. | command line |

#### `scripts/model3_cloud/package_outputs.py`

*107 lines; newest version ff2d7d1 (2026-08-26, virajchoudhary); on research/mentor-dh-pinn-basics.*

After a cloud run, fingerprints every output file (or writes a failure record) so results can be proven untouched.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256_file` | Computes a file fingerprint in chunks. | in-file: `manifest`, `verify` |
| `collect_files` | Lists every output file except excluded ones. | in-file: `manifest`, `verify` |
| `manifest` | Builds a list of every file with its size and fingerprint. | in-file: `main` |
| `verify` | Re-checks files against a saved manifest and lists any mismatch. | in-file: `main` |
| `main` | Command-line entry: write or verify the manifest, refusing to overwrite. | command line |

#### `scripts/model3_cloud/preflight.py`

*161 lines; newest version ff2d7d1 (2026-08-26, virajchoudhary); on research/mentor-dh-pinn-basics.*

Checks the git commit, clean working tree, file fingerprints and GPU before any Model 3 cloud run.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256_file` | Computes a file fingerprint. | in-file: `build_report` |
| `run_git` | Runs a git command in the repo and returns its output. | in-file: `build_report` |
| `build_report` | Collects all identity and GPU checks into a PASS/FAIL report. | in-file: `main`; `scripts/model3_cloud/launch_stage_a.py` |
| `main` | Command-line entry: prints the report and exits non-zero on failure. | command line |

#### `scripts/model3_cloud/prepare_stage_b.py`

*122 lines; newest version ff2d7d1 (2026-08-26, virajchoudhary); on research/mentor-dh-pinn-basics.*

Writes out (but never runs) the three frozen Model 3 Stage B launch commands for seeds 11, 22, 33.

| Function / class | What it does | Used by |
|---|---|---|
| `load_driver_module` | Loads the Model 3 pilot driver script as a module. | in-file: `package` |
| `package` | Builds the settings and exact command for one seed. | in-file: `main` |
| `main` | Builds all three packages and saves them as JSON. | command line |

### scripts/mentor_dh_pinn/

#### `scripts/mentor_dh_pinn/assess_regular_pinn.py`

*438 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Dhruva's file; Viraj's 24-line edit (Sep 11) let it load the new PyTorch float64 checkpoints and made the Apple MLX library optional. No per-function table, because Viraj did not write this file.

#### `scripts/mentor_dh_pinn/check_regular_pinn_physics.py`

*251 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Dhruva's file; Viraj's 55-line edit (Sep 11) adapted it to the PyTorch checkpoints (diff not read line by line). No per-function table, because Viraj did not write this file.

#### `scripts/mentor_dh_pinn/diagnose_regular_pinn_validation.py`

*121 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Dhruva's file; Viraj's 25-line edit (Sep 11) moved the fixed validation-surface builder into this file and added a price-error column next to the volatility error. No per-function table, because Viraj did not write this file.

#### `scripts/mentor_dh_pinn/evaluate_baseline.py`

*51 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: evaluates the trained mentor forward-PINN on the sealed test split exactly once.

#### `scripts/mentor_dh_pinn/evaluate_regular_pinn_development.py`

*111 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Scores one regular-PINN checkpoint on four fixed "development" truth cases by trying to recover their parameters (added by Viraj on Sep 11).

| Function / class | What it does | Used by |
|---|---|---|
| `development_surfaces` | Builds the four fixed test surfaces from a fixed random seed and checks their labels are valid. | in-file: `evaluate`; tests |
| `evaluate` | For each case, fits parameters with the network as the pricer and counts how many cases pass the 5 percent recovery gate. | in-file: `main` |
| `main` | Command-line entry: loads a checkpoint, runs evaluate, writes JSON, refuses to overwrite. | command line |

#### `scripts/mentor_dh_pinn/finetune_regular_pinn_lbfgs.py`

*467 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Viraj's "Phase 1/Phase 2" attempt (Sep 11) to polish Dhruva's regular PINN with a high-precision L-BFGS optimizer, optionally matching price sensitivities (Jacobians); the decision doc says it did not help.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Computes a file fingerprint. | in-file: `expected_data_hashes`, `index_sha256`, `main`; `scripts/mentor_dh_pinn/evaluate_regular_pinn_development.py` |
| `index_sha256` | Fingerprints a list of chosen row numbers so the batch choice is provable later. | in-file: `select_batches`, `select_jacobian_surfaces` |
| `expected_data_hashes` | Reads the checkpoint's manifest to learn which data fingerprints it was trained on. | in-file: `main`; tests |
| `load_torch_checkpoint` | Loads a two-factor PyTorch checkpoint and insists every weight is float64. | in-file: `main`; `scripts/mentor_dh_pinn/evaluate_regular_pinn_development.py`; tests |
| `select_batches` | Picks a fixed, seeded set of training rows and physics points for fine-tuning. | in-file: `main`; tests |
| `select_jacobian_surfaces` | Picks a few teacher surfaces that carry exact sensitivities, for the Jacobian-matching variant. | in-file: `main` |
| `surface_jacobian_loss` | Measures how far the network's price sensitivities are from the exact ones, with extra weight on the weakest directions. | in-file: `loss_components`; tests |
| `loss_components` | Adds up price-fit, sensitivity, physics-equation and optional Jacobian losses into one score. | in-file: `fine_tune`; tests |
| `fine_tune` | Runs the L-BFGS optimizer on that score and records the loss before and after. | in-file: `main`; tests |
| `save_checkpoint` | Saves weights, refusing anything that is not float64. | in-file: `main`; tests |
| `create_output_directory` | Makes the output folder, refusing if it already exists. | in-file: `main`; tests |
| `main` | Command-line entry: verifies hashes, fine-tunes, checks weights actually changed, saves everything. | command line |

#### `scripts/mentor_dh_pinn/generate_dataset.py`

*62 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: generates the deterministic synthetic call-price dataset for the mentor forward-PINN.

#### `scripts/mentor_dh_pinn/make_figures.py`

*51 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: draws the six grayscale diagnostic figures for a trained mentor forward-PINN.

#### `scripts/mentor_dh_pinn/precheck_double_v2_mac.py`

*187 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Safety check run on an Apple-Silicon Mac before the frozen "V2" regular-PINN run: verifies data fingerprints, row counts and that the MLX library trains and saves correctly.

| Function / class | What it does | Used by |
|---|---|---|
| `sha256` | Computes a file fingerprint. | in-file: `verify_artifacts` |
| `command` | Runs a shell command and returns its output. | in-file: `main`, `memory_status` |
| `memory_status` | Reads the Mac's total and free memory. | in-file: `main` |
| `verify_artifacts` | Hard-stops if any frozen V2 input file is missing or has the wrong fingerprint. | in-file: `main` |
| `verify_dataset_structure` | Hard-stops if the train/validation/collocation row counts are not the frozen ones. | in-file: `main` |
| `mlx_smoke` | Does one tiny MLX training step and a save/load round trip to prove the setup works. | in-file: `main` |
| `main` | Runs all checks, insists on macOS arm64, writes a report. | command line |

#### `scripts/mentor_dh_pinn/prepare_regular_pinn_surfaces.py`

*87 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Dhruva's file; Viraj's 1-line edit (Sep 11) also saves the raw Jacobian so the Phase 2 Jacobian pilot can use it. No per-function table, because Viraj did not write this file.

#### `scripts/mentor_dh_pinn/prepare_regular_pinn_v2.py`

*44 lines; newest version f4bc492 (Sep 21) is a later edit by Dhruva; on origin/main.*

Thin wrapper (Viraj, Sep 11): generates the frozen, reproducible "DOUBLE_DATA_V2" training data contract.

#### `scripts/mentor_dh_pinn/run_baseline.py`

*95 lines; newest version 5fdef34 (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: trains the mentor forward-PINN (with a tiny "smoke" mode), never touching test data.

#### `scripts/mentor_dh_pinn/run_final_synthetic_eval.py`

*39 lines; newest version 6a6ed0f (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: runs the one-shot Phase 3A final synthetic evaluation.

#### `scripts/mentor_dh_pinn/run_full_horizon_confirmation.py`

*39 lines; newest version 9dbe592 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: runs the Phase 2C full-length training confirmation.

#### `scripts/mentor_dh_pinn/run_lambda_refinement.py`

*41 lines; newest version 7c2ee8a (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: runs the Phase 2B finer loss-weight refinement.

#### `scripts/mentor_dh_pinn/run_lambda_sweep.py`

*38 lines; newest version 7e4b4cb (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: runs the Phase 2A loss-weight (lambda) sweep on validation data only.

#### `scripts/mentor_dh_pinn/run_multiseed_confirmation.py`

*41 lines; newest version c7a7f32 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: runs the Phase 2D multi-seed confirmation.

#### `scripts/mentor_dh_pinn/run_volatile_stock_selection_audit.py`

*54 lines; newest version bbe8be6 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: downloads official NSE price history and runs the volatile-stock selection audit.

#### `scripts/mentor_dh_pinn/summarize_final_synthetic_eval.py`

*34 lines; newest version 6a6ed0f (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: summarizes Phase 3A metrics.

#### `scripts/mentor_dh_pinn/summarize_full_horizon_confirmation.py`

*39 lines; newest version 9dbe592 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: summarizes Phase 2C against the Phase 1 reference.

#### `scripts/mentor_dh_pinn/summarize_lambda_refinement.py`

*39 lines; newest version 7c2ee8a (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: summarizes Phase 2B runs.

#### `scripts/mentor_dh_pinn/summarize_lambda_sweep.py`

*36 lines; newest version 7e4b4cb (2026-08-28, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: summarizes Phase 2A sweep runs.

#### `scripts/mentor_dh_pinn/summarize_multiseed_confirmation.py`

*46 lines; newest version c7a7f32 (2026-08-29, virajchoudhary); on research/mentor-dh-pinn-basics.*

Thin wrapper: summarizes Phase 2D runs.

### paper/scripts/

#### `paper/scripts/generate_results_assets.py`

*472 lines; newest version 1fe18e6 (2026-08-25, virajchoudhary); on research/paper-synthesis.*

Builds the LaTeX tables and PDF figures for the paper directly from committed evidence files, so no number is typed by hand.

| Function / class | What it does | Used by |
|---|---|---|
| `write_text` | Writes a text file, refusing any non-ASCII character so LaTeX will not choke. | in-file: `latex_table`, `main` |
| `copy_figure` | Copies a generated figure into the paper's figures folder. | in-file: `main` |
| `sci` | Formats a number nicely for LaTeX, switching to "x 10^n" form for very big or small values. | in-file: `canonical_engine_table`, `neural_noise_tables_and_figure`, `primary_comparison_table`, `r2_selection_table` |
| `pct` | Formats a fraction as a LaTeX percentage string. | in-file: `neural_noise_tables_and_figure`, `primary_comparison_table` |
| `latex_table` | Assembles a complete LaTeX table from a caption, header and rows. | in-file: `canonical_engine_table`, `evidence_classification_table`, `neural_noise_tables_and_figure`, `primary_comparison_table` ... |
| `git_text` | Reads a file as it existed at a given past commit, so results can be pulled from another branch. | in-file: `neural_noise_tables_and_figure` |
| `canonical_engine_table` | Makes the table showing the pricing engine passed 36 of 36 benchmark cases. | in-file: `main` |
| `primary_comparison_table` | Makes the main table comparing Model 1, Model 2 and traditional calibration. | in-file: `main` |
| `real_market_development_table` | Pulls the NTPC Black-Scholes vs Heston vs Double Heston rows out of the NTPC doc into a table. | in-file: `main` |
| `r2_selection_table` | Makes the R2-versus-R3 comparison table at each noise level. | in-file: `main` |
| `evidence_classification_table` | Makes a table labelling each piece of evidence as complete, development-only, or pending. | in-file: `main` |
| `neural_noise_tables_and_figure` | Builds the table and chart of how the two networks behave as price noise grows. | in-file: `main` |
| `speed_accuracy_figure` | Draws the speed-versus-accuracy chart for the three methods. | in-file: `main` |
| `fit_recovery_figure` | Draws the chart showing that a good price fit does not mean the parameters are right. | in-file: `main` |
| `main` | Runs every table and figure builder in order. | command line |

#### `paper/scripts/validate_paper.py`

*271 lines; newest version 1fe18e6 (2026-08-25, virajchoudhary); on research/paper-synthesis.*

A spell-checker-style script for the paper: checks sections exist, references resolve, citations exist and nothing is unbalanced.

| Function / class | What it does | Used by |
|---|---|---|
| `parse_bibtex` | Reads the bibliography file into a dictionary of entries and their fields. | in-file: `main` |
| `citation_keys` | Finds every citation key used in a LaTeX text. | in-file: `main` |
| `fail` | Stops the script with a "VALIDATION FAILED" message. | in-file: `check_ascii`, `check_balanced_environments`, `main`, `parse_bibtex` ... |
| `read` | Reads a file, failing if it does not exist. | in-file: `main`, `parse_bibtex` |
| `check_ascii` | Fails if a file has any non-ASCII character and reports the line. | in-file: `main`, `parse_bibtex` |
| `check_balanced_environments` | Fails if any LaTeX begin/end pair is unmatched. | in-file: `main` |
| `referenced_inputs` | Lists every file pulled in with \input. | in-file: `main` |
| `referenced_figures` | Lists every image file the paper includes. | in-file: `main` |
| `referenced_labels` | Lists every \ref target used. | in-file: `main` |
| `defined_labels` | Lists every \label defined. | in-file: `main` |
| `main` | Runs all the checks over main.tex, the sections, figures and bibliography. | command line |

### tests/ (one line per test file)

- `tests/mentor_dh_pinn/test_baseline_contract.py` — Checks the mentor forward-PINN's fixed architecture, its settings file, and that a one-epoch run saves a valid checkpoint (4 tests).
- `tests/mentor_dh_pinn/test_final_synthetic_eval.py` — Checks the Phase 3A final evaluation is one-shot, uses a brand-new deterministic holdout, and refuses any checkpoint or seed drift (7 tests).
- `tests/mentor_dh_pinn/test_full_horizon_confirmation.py` — Checks Phase 2C reuses the Phase 1 data, never touches test data, and refuses protocol drift (4 tests).
- `tests/mentor_dh_pinn/test_lambda_refinement.py` — Checks Phase 2B parses the exact weight matrix around the A9 winner and never touches test data (4 tests).
- `tests/mentor_dh_pinn/test_lambda_sweep.py` — Checks the Phase 2A nine-run weight sweep uses only train/validation data and summarizes deterministically (3 tests).
- `tests/mentor_dh_pinn/test_losses.py` — Checks the data, boundary and payoff losses behave correctly (e.g. exact payoff gives zero loss) (4 tests).
- `tests/mentor_dh_pinn/test_multiseed_confirmation.py` — Checks Phase 2D's seed matrix is frozen, never touches test data, and summaries are correct (5 tests).
- `tests/mentor_dh_pinn/test_parameter_source.py` — Checks the fixed parameter set is the first valid training record and only v0 is swapped for state variances (3 tests).
- `tests/mentor_dh_pinn/test_pde.py` — Checks the PINN's physics equation matches a known exact solution, uses float64 and has no spurious cross term (5 tests).
- `tests/mentor_dh_pinn/test_split_isolation.py` — Checks train/validation/test ids never overlap and the trainer never asks for test rows (2 tests).
- `tests/mentor_dh_pinn/test_synthetic_data.py` — Checks the synthetic dataset is deterministic and variance states follow the theta rule (3 tests).
- `tests/mentor_dh_pinn/test_volatile_stock_selection.py` — Checks the volatile-stock audit reads official reports strictly, computes 3-month volatility correctly, and rejects bad data (8 tests).
- `tests/test_ann_forward.py` — Checks the plain ANN outputs ten numbers per surface and rejects bad input (2 tests).
- `tests/test_constraints.py` — Checks the rulebook accepts a valid parameter set and rejects each kind of violation (6 tests).
- `tests/test_core_dataset_readiness.py` — Checks the (now historical) 108-grid dataset readiness record is consistent (2 tests).
- `tests/test_dataset_shapes.py` — Checks the old grid is built deterministically, surfaces stay within one split, and smoke data is labelled fake (3 tests).
- `tests/test_double_heston_engine.py` — Mathematical checks of the pricer: fingerprint equals 1 at zero, prices finite and arbitrage-free, put-call parity, node convergence, fixture reproducible (18 tests).
- `tests/test_double_heston_reference_benchmark.py` — Checks the independent reference pricer and that it agrees with production within frozen tolerances (14 tests).
- `tests/test_g2_common_support_analysis.py` — Checks the G2 support analysis uses exactly the four stocks and three dates and never extrapolates silently (9 tests).
- `tests/test_g2_complementary_observable_analysis.py` — Checks the complementary-observable experiment's designs, seeds and decision rule were fixed in advance (19 tests).
- `tests/test_g2_global_ambiguity_analysis.py` — Checks the global-ambiguity study's cases, starts and runtime are frozen and comparisons are fair (13 tests).
- `tests/test_g2_identifiability_analysis.py` — Checks the carry contract, grid order and that calls/puts are compared fairly in the identifiability study (8 tests).
- `tests/test_g2_information_remediation.py` — Checks the two-vs-three-expiry study's matrix, shapes and coupled noise (9 tests).
- `tests/test_g2_multi_date_identifiability.py` — Checks the multi-date design, exact CIR simulation and date-specific states (9 tests).
- `tests/test_g2_multi_date_replication.py` — Checks the replication changes only the CIR path seed and its hypothesis rules (7 tests).
- `tests/test_g2_r2r3_harness.py` — Checks the R2/R3 harness: 20 and 30 slots, explicit masking, no made-up prices, no interpolation, and the decision rule (23 tests).
- `tests/test_g8_final_protocol.py` — Checks the G8 config is frozen with execution disabled and development stock/dates are banned (14 tests).
- `tests/test_g8_readiness_contracts.py` — Checks the G8 date window, rate/carry maths, Black-pricing round trip and authorization locks (15 tests).
- `tests/test_g8_readiness_harness.py` — Checks G8 traditional calibration uses two uninformed starts and checkpoint staging refuses wrong fingerprints (3 tests).
- `tests/test_g8_readiness_selection_surfaces.py` — Checks the G8 date scanner, backup rule and real-surface builder behave as frozen (13 tests).
- `tests/test_g8_readiness_state_machine.py` — Checks the G8 workflow never grants acquisition on its own and prerequisites are ordered (4 tests).
- `tests/test_market_data_audit.py` — Checks the Stage A spreadsheet loader's surface grouping, maturity buckets, moneyness and carry helper (10 tests).
- `tests/test_model3_cloud_packaging.py` — Checks the Model 3 cloud preflight, output manifests and Stage B packages, and that no secrets are written (5 tests).
- `tests/test_model3_pde_foundation.py` — Checks the Model 3 physics operator against hand-computed and exact solutions and that it refuses missing derivatives (14 tests).
- `tests/test_model3_pde_pilot_driver.py` — Checks the Model 3 training driver's frozen settings, test-split refusal, deterministic subsets and resume safety (21 tests).
- `tests/test_model3_sealed_evaluation.py` — Checks the Model 3 freeze gate rejects mixed, drifted, missing or tampered seeds and that evaluation stays locked (15 tests).
- `tests/test_nse_stage_a.py` — Checks NSE file names, the exact 34-column schema, raw identity checks and Stage A coverage tables (28 tests).
- `tests/test_ntpc_canonical160_reproduction.py` — Checks the NTPC reproduction forensics and that maturity T is rebuilt exactly from days (8 tests).
- `tests/test_ntpc_dh_multi_date_calibration.py` — Checks the three-date NTPC panel, sealed RBI rates with no future leakage, and valid shared/date-specific parameters (15 tests).
- `tests/test_ntpc_dh_optimizer_cap_sensitivity.py` — Checks the 160-vs-320 experiment changes only the step budget and detects caps exactly (17 tests).
- `tests/test_ntpc_dh_stability_reparameterization.py` — Checks the "total plus share" coordinates round-trip exactly and cover the whole valid space (11 tests).
- `tests/test_ntpc_single_stock_pilot.py` — Checks NTPC pilot maths: days to years, log-moneyness, deterministic row selection, implied-vol inversion (12 tests).
- `tests/test_ood_boundary_evaluation_seal.py` — Checks the OOD evaluation stays locked without all three explicit choices and never writes into frozen folders (21 tests).
- `tests/test_ood_boundary_protocol.py` — Checks the OOD protocol config, hashes, remote-commit gate and fixed quotas (9 tests).
- `tests/test_parameter_order.py` — Checks the exact ten-parameter order and dictionary round trip (2 tests).
- `tests/test_parameter_transform.py` — Checks the bounded transform stays inside limits and the standardizer round-trips (3 tests).
- `tests/test_power_candidate_tiebreak_extension.py` — Checks the NTPC-vs-POWERGRID tie-break refuses any tampered original data (13 tests).
- `tests/test_r2_final_generation_gate.py` — Checks the 10k generation refuses to run without a committed, pushed authorization and validates output fully (39 tests).
- `tests/test_r2_noise_contract.py` — Checks the noise recipe is deterministic, replays bit-for-bit and is shared across methods (12 tests).
- `tests/test_r2_noise_execution.py` — Checks noisy records keep the clean data intact and arbitrage violations are flagged, not repaired (3 tests).
- `tests/test_r2_noise_provenance_intake.py` — Checks the subset-file line-ending repair accepts only the two known exact forms (8 tests).
- `tests/test_r2_primary_comparison_protocol.py` — Checks before training that the dataset fingerprint, counts, splits and R2-only format are exact (10 tests).
- `tests/test_r2_primary_device_plumbing.py` — Checks the CPU/GPU option did not change any frozen setting and test data stays excluded (10 tests).
- `tests/test_r2_primary_implementation.py` — Checks the 100-number feature builder, target order, data loader and split isolation (15 tests).
- `tests/test_r2_representation.py` — Checks every requirement of the R2 format: 20 slots, fixed order, five strike distances, two expiries, masks, serialization (44 tests).
- `tests/test_r2_synthetic_generation.py` — Checks the frozen generation recipe: identity, quotas, seeds, splits, conditioning, pilot and replay (27 tests).
- `tests/test_r2_synthetic_generation_freeze.py` — Checks the generation contract before it was frozen (10 tests).
- `tests/test_real_market_weight_update_quarantine.py` — Checks the Archive-2 lock: no real-market training unless explicitly opted in (20 tests).
- `tests/test_regular_pinn_lbfgs.py` — Checks Viraj's L-BFGS fine-tuning keeps float64, is deterministic, changes weights and lowers loss (8 tests).
- `tests/test_reproducible_regular_pinn_data.py` — Checks the V2 data files are byte-identical on repeat and the random blocks are repeatable (3 tests).
- `tests/test_research_control_plane.py` — Checks the research registry and dependency graph reject duplicates, cycles, bad hashes and stale status (14 tests).
- `tests/test_reviewed_sampling.py` — Checks the reviewed sampling config, Feller transform, correlation disk and OOD isolation (6 tests).
- `tests/test_reviewed_sampling_matrix.py` — Checks sampling determinism, constraints, split isolation and the priced subset (9 tests).
- `tests/test_torch_pricer.py` — Checks the safe complex log fix makes the PyTorch pricer match independent pricers on tiny inputs (1 test).

---

## 5. Glossary (one plain sentence each)

- **Option / call / put**: a contract giving the right to buy (call) or sell (put) a stock at a fixed price (the strike) on a future date (the expiry).
- **Option price surface**: the grid of option prices across different strikes and expiries for one stock on one day.
- **Heston model**: a model where a stock's "nervousness" (variance) wanders randomly but is pulled back toward a long-run level, described by five parameters (kappa, theta, sigma, rho, v0).
- **Double Heston**: the same idea with two independent nervousness factors, a slow one and a fast one, so it has ten parameters in total.
- **The ten parameters**: for each factor, kappa is pull-back speed, theta is long-run level, sigma is how jumpy the variance is, rho is how the stock and its variance move together, and v0 is today's variance.
- **Feller condition**: a rule (2·kappa·theta > sigma²) that keeps the variance from touching zero; the project insists on it for both factors.
- **Factor swap**: swapping the slow and fast blocks gives identical prices, so the rule "slow must be slower than fast" (kappa_slow < kappa_fast) is used to pick one.
- **Characteristic function / Fourier pricing / Gauss–Laguerre / Little Heston Trap**: the maths trick the pricer uses (a "fingerprint" of future prices integrated at 64 fixed points), in a numerically stable form.
- **Calibration**: turning the dials (parameters) of a model until its prices match the prices seen in the market.
- **Traditional calibration**: doing that with a numerical optimizer (here SciPy least squares, 3 starting points, up to 300 tries each).
- **Inverse problem / inverse calibration**: going backwards from prices to parameters, which is harder than going from parameters to prices.
- **Identifiability**: whether the prices contain enough information to pin down one unique set of parameters; "practical non-identifiability" means many very different parameter sets fit the prices almost equally well.
- **Repricing error**: feed the recovered parameters back into the pricer and measure how far the prices are from the observed ones.
- **Range-scaled parameter RMSE**: the average parameter error after dividing each parameter's error by the width of its allowed range, so all ten count equally.
- **Multi-start / near-equivalent solutions / clusters**: running the optimizer from several starting points; fits whose error is within a tiny tolerance of the best are "near-equivalent", and groups of similar ones are "clusters".
- **Jacobian / condition number / practical rank**: a table of how much each price moves when each parameter is nudged; a huge condition number or low rank means some parameter combinations barely move prices.
- **ANN / Model 1**: an ordinary artificial neural network trained to map a price surface straight to ten parameters.
- **Model 2**: Model 1 plus built-in rules (outputs always valid) and a "re-price and compare" loss; the project explicitly says it is not a true PINN.
- **PINN (physics-informed neural network)**: a network trained not only on examples but also to obey the model's governing equation.
- **PDE (partial differential equation)**: the Double Heston pricing equation that any correct option price must satisfy.
- **Model 3**: the planned genuine PINN where a pricing network must satisfy the PDE; built but never trained.
- **Mentor forward PINN**: a separate, simpler exercise where a network learns call prices for one fixed parameter set using four losses (data, PDE, expiry payoff, boundaries).
- **Lambda / lambda sweep**: the weights given to each of those losses, and the experiments (Phases 2A–2D) that tried different weights.
- **Collocation points**: made-up input points where the network is checked against the PDE, expiry or boundary rule.
- **R2 / R3**: input formats; R2 = first two listed expiries × five strikes near the money × calls and puts = 20 prices; R3 adds a third expiry = 30 prices.
- **Mask**: a true/false flag per price slot saying whether a real quote exists; missing quotes are never filled in.
- **G2 / G8**: numbered gates in the research plan — G2 was the "which input format, and are parameters identifiable?" gate, G8 the final test on fresh, never-seen real market data; no G1 or G3–G7 definitions were found in the files read.
- **Stage A**: the first real-data screen of which NSE stocks and option strikes/expiries actually trade.
- **OOD (out-of-distribution)**: test cases deliberately outside what the model was trained on (edge parameters, unusual maturities, missing slots).
- **Noise robustness**: whether results hold up when small random errors (0.10%–1%) are added to the prices.
- **Train / validation / test split**: data for learning, for choosing the best version, and a locked set opened only once for the final score.
- **Seed**: a starting number for randomness; seeds 11, 22, 33 were used to check results are not luck.
- **Frozen / sealed / predeclared / fail-closed**: rules and data fixed and fingerprinted before results are seen; code stops with an error rather than guessing when anything differs.
- **SHA-256 / hash / provenance**: a file fingerprint used to prove a file is exactly the one recorded, and the record of where data came from.
- **NSE UDiFF bhavcopy**: the official daily end-of-day data files of the National Stock Exchange of India.
- **RBI 91-day T-bill**: the government bill whose auction yield is used as the risk-free interest rate.
- **Carry (q)**: a dividend-like yield implied from the futures price.
- **Implied volatility**: the single volatility that makes the simple Black–Scholes formula match a market price.
- **Moneyness / log-moneyness**: how far a strike is from the current stock price, here log(K/S).
- **DTE**: days to expiry.
- **CIR process**: the specific random process each variance factor follows.
- **Archive-2**: an imported older code base (by Dhruva) that Viraj found had a broken PDE loss and a path that trained on real data; it was quarantined.
- **Node A/B/C (overnight swarm)**: three parallel overnight investigation lanes on 22 Aug; Viraj ran Node A (coordinator/architecture), Dhruva's commits are on Nodes B and C.
- **L-BFGS / Jacobian fine-tuning**: a precise optimizer and a training add-on that matches the network's price sensitivities to the exact ones; tried on Sep 11 without benefit.
- **Smoke test**: a tiny run to check the plumbing works, never a scientific result.

---

## 6. What the user's Double Heston 2.0 uses from this

Short answer: **nothing**. The belief was correct.

What I checked: I searched every `.py` file under
`/Users/vinayakmangal/Everything related to my projects/double-heston-inverse-calibration/.claude/worktrees/complete-training/wild/.shots/dh2/double_heston_2.0`
for three kinds of reference:
- imports of `src`, `models` or `legacy_streamlit_site`;
- names of Viraj's modules: `double_heston`, `pricing_interface`, `calibrate_double_heston`, `synthetic_dataset`, `parameter_transform`, `ann_model`, `r2_primary`, `r2_noise`, `g2_r2r3`, `g8_readiness`, `model3_`, `mentor_dh_pinn`, `r2_representation`, `torch_double`;
- any `sys.path` changes.

What I found:
- About 20 scripts add their own folder to `sys.path`. This only lets them import the local `dh2` package.
- `dh2/models.py` loads the pricer from `legacy_streamlit_site/models.py` by file path (`importlib.util.spec_from_file_location`). Its docstring says: "imported unchanged". `tests/test_dh2.py` points at the same file.
- `dh2/calib.py` and `dh2/evalpairs.py` import only `dh2.models`.
- There are no imports of `src/`, of `models/`, or of any Viraj module.

Who wrote `legacy_streamlit_site/models.py`? It is not Viraj's file. Its only git commit is `cc3487e` by mangalvinayak13-vin (30 Sep 2026), "Archive the Streamlit site's source before retiring its repo". It is a separate, self-contained pricer: Black-Scholes, Monte Carlo paths, and Heston and Double Heston with the "little trap" form, using parameter names like `v0_1, kappa1, ..., xi1`. It does not import anything from `src/`. It uses the same textbook maths as Viraj's `src/double_heston.py`, but it is a different implementation with a different parameter convention.

What 2.0 indirectly inherits is lessons, not code. Two of Viraj's findings carry over:
- Double Heston parameters are practically non-identifiable from option prices alone.
- A good price fit does not prove correct parameters.
