# What Dhruva built: a plain-English dissection

Scope: every commit by "Dhruva Ambhaikar" and "DhruvaBamb" (60 commits, 21 Aug to 25 Sep 2026) in double-heston-inverse-calibration. Everything below was read from the files with `git show` and `git ls-tree`. Nothing was run. Numbers are quoted only from files, and the file path is given.

Where his work lives (it is spread over several branches):

| Branch | What of Dhruva's is only there |
|---|---|
| origin/main (and the user's main, night-build, simple-text) | almost everything: experiments/ (except v5 multiscale and v6 architecture), src/mentor_dh_pinn, scripts/mentor_dh_pinn, single_heston_pinn, outputs/*, docs/mentor_dh_pinn, docs/figures, tests |
| origin/codex/improved-double-heston-pinn | experiments/nifty_multiscale_v5, experiments/pinn_architecture_v6, src/mentor_dh_pinn/multiscale_pinn.py, research_modified_pinn.py, PINN_ARCHITECTURE_IMPROVEMENT_REPORT.md, CURRENT_PINN_BASELINE.md (one big commit f4bc492, 21 Sep) |
| origin/Double/Single-heston | outputs/affine_factor_pinn and the "factor-structured" and "composition" PINN code (commit 5b6ae15, 10 Sep) |
| origin/black-scholes-with-pinn | black_scholes_pinn/ (1 Sep) |
| origin/archive/overnight-20260822-node-b and -node-c | .ai-research/overnight/2026-08-22/node-B and node-C, scripts/node_b_toolkit.py and run_node_b_*.py |

Two of his commits are not really his own research. Commit 8e281ba (21 Aug, "Initial commit of downloaded files") is a separate starting point that re-uploads the team's existing code (Viraj's pricing engine, docs, G2/NTPC scripts, configs). Commit 642702e (22 Aug, "Add PINN implementation from Desktop") adds an early neural calibrator that Dhruva's own audit later called "Archive-2". Both are described briefly, not dissected.

---

## 1. In one paragraph

Dhruva was the team's neural-network person. His job was to see whether a physics-informed neural network (a network trained to obey the Double Heston pricing equation) could do two things: price options as accurately as the slow exact formula but much faster, and read the model's ten hidden settings back out of real option prices ("inverse calibration"). The first part clearly worked. His locked network copies the exact Double Heston price to a price error of 1.10e-5, about 0.10 volatility points (experiments/nifty_multifactor_v4/FINAL_REPORT.md), and later versions got to 6.6e-6 (experiments/dh_pinn_v5/IMPROVED_DH_PINN_REPORT.md). The second part mostly failed. In test after test, no network recovered all ten settings reliably: "0/24" (outputs/calibration_repair/RESULTS.md), "Zero of 18 recovered all ten parameters" (docs/PROPOSED_PARAMETER_SELECTION_RESULTS.md), at best "2/24" (outputs/deeper_pinn/recovery_report/REPORT.md). His overnight audit explained why: many different setting combinations give almost the same prices once you allow for normal market noise (.ai-research/overnight/2026-08-22/node-B/FINAL_REPORT.md). On real markets the honest picture is mixed. With fixed textbook settings, Double Heston and its PINN lost badly to a simple Black-Scholes on untouched NIFTY options (107.7 against 22.4 index points). With settings fitted fresh each day on option chains that have many expiries (BTC, ETH, IWM), Double Heston clearly beat Single Heston and Black-Scholes, but that edge vanished on thin chains (XRP, SOL, HYPE) and on Indian power stocks, where Single Heston won. His one neural success on real data was a "projection fine-tuned" calibrator that beat classical Double Heston on 10 NIFTY days (2.478 against 2.523 volatility points) while running 21.7 times faster. It had no effect on single-expiry stock surfaces. He also showed that a better network barely moves market error: 0.003 volatility points out of 5.8 on SPX. The live website shows that NIFTY result and the 1.1e-5 accuracy figure.

---

## 2. Timeline

- **21–22 Aug: start and overnight audits.** He uploaded a copy of the team's code (8e281ba) and an early neural calibrator, "Archive-2" (642702e). Overnight on 22 Aug he ran two AI-assisted audits. Node B asked why the ten settings cannot be recovered and answered: "MIXED mechanism dominated by practical non-identifiability" (node-B/FINAL_REPORT.md). Node C checked the PINN physics and found that Archive-2's equation term was broken: "Every variance-state derivative evaluates to exactly ZERO" (node-C/FINAL_REPORT.md).
- **1 Sep: Black-Scholes PINN.** This was a small warm-up network that learns one volatility from 180 synthetic quotes. It got 0.1999991 against a true 0.20 (black_scholes_pinn/RESULTS.md).
- **4 Sep: big library drop (src/mentor_dh_pinn).** It contained:
  - forward pricing PINNs called model_v3, v4 and v5;
  - a two-network "dual PINN" that split the ten settings between short and long expiries, plus a "ridge" polish step;
  - the "unified" set-transformer calibrator trained on dataset v6 (this is what "unified_v6" means);
  - a single Heston PINN for Indian power stocks.

  Verdict: it beat classical fitting on synthetic data but "does not beat classical calibration on real market data" (docs/mentor_dh_pinn/DELIVERABLE_F_FINAL_REPORT.md).
- **5 Sep: projection fine-tuning.** He retrained the unified calibrator to fit real quotes instead of copying known answers. NIFTY error went from 0.02902 to 0.02478, beating classical 0.02523 at "21.7x faster". There was "no effect" on single-expiry surfaces (docs/mentor_dh_pinn/RESEARCH_BRIEF_REAL_DATA_GAP.md §5.1). This is the number the website shows.
- **7–10 Sep: many recovery attempts, all honestly failed.** He tried:
  - calibration repair (outputs/calibration_repair);
  - the "regular" PDE PINN for single and double Heston (outputs/regular_pinn_recovery);
  - 11- and 17-layer deeper PINNs (outputs/deeper_pinn);
  - factor-structured "affine" PINNs on the Double/Single-heston branch.

  Pricing improved, but full ten-setting recovery stayed at 0 to 2 cases out of 24. On 8 Sep he also added classical single Heston research on NSE power stocks, where Single Heston beat Double Heston.
- **15–17 Sep: fixed published truths and NIFTY crisis windows.** The literature-truth pilot recovered 0 of 6 cases, and the parameter sweep recovered 0 of 18. The NIFTY crisis forward test found no "meaningful or consistent Double Heston edge" (docs/NIFTY_FIXED_CRISIS_RESULTS_20260917.md).
- **17–18 Sep: nifty_multifactor v1 to v4.** Each version was frozen in advance:
  - v1 failed an exact-price short-time test;
  - v2 fixed that integration;
  - v3 widened the network's input range, but its PINN failed the accuracy gates because of loss scaling;
  - v4 rescaled the loss and climbed a ladder of three candidates (C1, C2, C3), locking C3.

  Result: controlled synthetic test "DH-PINN < SH < BS ... on 40 of 40" surfaces, but untouched NIFTY market test "no" (experiments/nifty_multifactor_v4/FINAL_REPORT.md).
- **18 Sep: crypto replications.** He ran per-date calibrated DH vs SH vs BS on Deribit options for BTC, then ETH, XRP (in place of ENJ, which has no options), SOL and HYPE. Double Heston beat Single Heston significantly only on BTC and ETH. It beat Black-Scholes everywhere (experiments/cross_asset_figures/cross_asset_primary.csv).
- **20–21 Sep: v5 multiscale and v6 architecture (codex branch).**
  - v5 added three small maturity "expert" branches. That gained only 0.396% and still lost to Single Heston in all three NIFTY windows (experiments/nifty_multiscale_v5/REPORT.md).
  - v6 tested published PINN tricks. A "modified MLP" correction cut fresh price error by 9.24%. Gradient balancing and adaptive sampling (RAD) did not help (PINN_ARCHITECTURE_IMPROVEMENT_REPORT.md).
  - On main he also archived the portable dual PINN (it failed its own criterion) and a BTC data-leak caveat.
- **24 Sep: hard-coded parameters and figures.** He tested published parameters against today's SPX and crypto shock days (experiments/hardcoded_v1), drew price curves and drew the C3 architecture diagram.
- **25 Sep: dh_pinn_v5.** This is a second, separate "v5" built on v4 C3, not on nifty_multiscale_v5. Its gated network was "1.6x more accurate", but that moved SPX error by "0.003 vol points out of 5.8". A small market-residual head cut SPX error by about two thirds, but the Double Heston prior "does not beat the Single Heston prior" (experiments/dh_pinn_v5/IMPROVED_DH_PINN_REPORT.md).
- **25 Sep (last day): surface selection, figures and reference PDFs.**
  - He picked a surface in advance by a model-free score (IWM). Fully calibrated Double Heston reached 0.583 vol points on held-out quotes, against Single Heston 1.893 and Black-Scholes 5.388 (experiments/dh_relevance_selection_v1/RESULTS.md).
  - He produced the separation figures (clear_separation_figures_v1, financial_curve_figures_v2).
  - He wrote the equations reference and a plain-language explainer as PDFs (model_reference_v1).

---

## 3. Folder by folder

Status words used below:
- **LIVE (website)**: the live website reads it or quotes its numbers.
- **USED (DH 2.0)**: the user's Double Heston 2.0 comparison script reads it.
- **REUSED**: another of Dhruva's later experiments loads its code or weights.
- **SUPERSEDED**: replaced by a later version.
- **RECORD**: a failed or finished attempt kept for honesty.

### 3.1 Starting material (not his own research)

**Repository bootstrap (commit 8e281ba, 21 Aug).**
- What: a re-upload of the team's existing code. It includes Viraj's exact Double Heston pricer (src/double_heston.py), constraints, G2/NTPC analysis scripts, configs, and outputs such as double_heston_benchmark, double_heston_validation, engine_freeze, parameter_bounds_audit, reviewed_sampling_audit and core_dataset_readiness.
- Why: to have the code on his own branch.
- Result: nothing new.
- Status: not Dhruva's research. main already had this code from Viraj's commits starting 6 Aug.

**Archive-2 neural calibrator (commit 642702e, 22 Aug).** The files are src/dheston/, models/pinn_model.py, src/train_pinn.py, src/torch_double_heston.py, train_double_heston.py, src/run_pinn_*.py, outputs/metrics/pinn_*, experiments/import_smoke_20260821 and four tests.
- What: a network that looks at an option price surface and outputs the ten Double Heston settings. It was trained with "parameter supervision plus repricing consistency" and a PDE term.
- Why: the first attempt at a PINN calibrator.
- Result: the two-stage version lowered the mean repricing RMSE from 1.73 to 0.70 (in that file's own units) in its own synthetic benchmark (outputs/metrics/pinn_improved_benchmark_run2/summary.json). However, Dhruva's own Node C audit later proved that its physics term was broken, so the "PINN" part did nothing.
- Status: RECORD. Viraj later quarantined its real-market weight updates (commit d29a437).

### 3.2 .ai-research/overnight (Node B and Node C, 22 Aug, archive branches)

**Node B (identifiability).**
- What: an overnight AI-agent investigation, with scripts/node_b_toolkit.py and run_node_b_*.py.
- Why: to find out why classical fitting cannot recover the ten settings.
- Result: on the full 108-quote grid the problem is not mathematically singular ("practical rank 10/10"). But "the weakest sensitivities sit below the noise floor". So at realistic 0.5–2% noise many setting combinations fit equally well. Giving the optimiser more budget did not help.
- Status: RECORD. This is the key diagnosis that every later failure kept confirming.

**Node C (PDE physics audit).**
- What: a mathematical audit of the equations, with tests/test_node_c_pde_physics_audit.py (27 tests) and probe scripts.
- Why: to check that the PINN's physics is right.
- Results:
  - The canonical pricer "satisfied to machine precision" its own pricing equation.
  - The canonical calibrator is "NOT PDE-informed". Its "physics" is repricing plus hard rules, so the audit's label for it is "constraint + differentiable-repricing informed inverse network".
  - Archive-2's PDE term is broken.
- Status: RECORD. This matters for how the website labels things (see section 6).

### 3.3 black_scholes_pinn/ (1 Sep, branch black-scholes-with-pinn)

- What: a small standalone PINN for plain Black-Scholes. It learns a price surface and one unknown volatility at the same time. There is a vendored copy in experiments/clear_separation_figures_v1/vendor_bs_pinn.
- Why: a warm-up to prove the inverse-PINN idea on the simplest model.
- Result: "Final robust PDE-calibrated volatility 0.1999991" against a true 0.2000000, relative error 0.000468% (black_scholes_pinn/RESULTS.md).
- Status: RECORD. It is reused only as a validation figure in clear_separation_figures_v1.

### 3.4 src/mentor_dh_pinn/ (the main library, 4 Sep onward)

- What: all his reusable neural code. It has five families.
  - (a) Early forward PINNs and the dual PINN: model_v3, model_v4, model_v5, data_v4, collocation_domain, dual_pinn, dual_pinn_data, calibrate, polish, nifty_panel.
  - (b) The unified set calibrator: unified, params_v2, dataset_v6, collate, ood, baselines, torch_pricer, finetune_projection, precise_calibration.
  - (c) The "regular" PDE PINN: regular_pinn (Apple MLX), regular_pinn_torch (PyTorch copy), regular_pinn_data, deep_regular_pinn, surrogate_error, identifiability, recovery_diagnostics, parameter_transforms. Viraj added reproducible_regular_pinn_data on 11 Sep.
  - (d) The experimental maturity_dual_pinn.
  - (e) Branch-only extras: the affine/factor PINNs (Double/Single-heston branch), and multiscale_pinn plus research_modified_pinn (codex branch).
- Why: the deliverables a mentor asked for (docs/mentor_dh_pinn/DELIVERABLE_A, B, F).
- Results: the per-family numbers are in the outputs/ and experiments/ entries below.
- Status:
  - Family (b) produced the website's NIFTY number, so it is LIVE indirectly through outputs/unified_v6.
  - In family (c), regular_pinn_torch.py is the most important file in his whole body of work. Its `TorchRegularVariancePINN` class is the locked v4 C3 network, REUSED by nifty_multifactor_v1–v4, dh_pinn_v5, hardcoded_v1, price_curve_figures, portable_dual_pinn_v1, financial_curve_figures_v2 and clear_separation_figures_v1.
  - Family (a) is SUPERSEDED. Its training scripts (for example train_dual_pinn.py) are not in the repository. Only the modules and outputs/dual_pinn_legacy weights remain.

### 3.5 scripts/mentor_dh_pinn/ (4 Sep onward)

- What: the command-line drivers that build data, train, fine-tune, evaluate and write reports for the library above. There are 47 scripts on main and about 20 more on the Double/Single-heston branch.
- Why: each run is reproducible and writes into outputs/.
- Results: see the outputs entries.
- Status: build_v6_data, train_unified, evaluate_unified, evaluate_real_markets, run_ablations, run_finetune, evaluate_finetune and evaluate_panel_test produced outputs/unified_v6. The rest are RECORD for failed recovery studies. Five were written by Viraj (prepare_regular_pinn_v2, finetune_regular_pinn_lbfgs, evaluate_regular_pinn_development, precheck_double_v2_mac, plus edits).

### 3.6 docs/mentor_dh_pinn/ (4–5 Sep)

- What: his written deliverables.
  - DELIVERABLE_A is a repository audit. It found that the old `PARAM_BOX` "does silently clip".
  - DELIVERABLE_B is the design.
  - DELIVERABLE_F is the final report.
  - DUAL_PINN_RESULTS and TWO_STAGE_RESULTS cover the dual PINN.
  - NIFTY_THREE_MODEL_COMPARISON compares models on NIFTY.
  - RESEARCH_BRIEF_REAL_DATA_GAP diagnoses why the model loses on real data.
- Why: hand-in reports for the mentor.
- Results:
  - TWO_STAGE: the network plus ridge polish gives "2.1089e-04" held-out repricing and "beats single Heston" on 78% of surfaces (synthetic).
  - NIFTY_THREE_MODEL_COMPARISON: "dual PINN + ridge 3 ... 2.293 vol pts" against single Heston 2.361 and Black-Scholes 3.039, on 1,358 held-out NIFTY quotes.
  - DELIVERABLE_F: see section 1.
- Status: RECORD. The NIFTY comparison numbers are USED (DH 2.0) through outputs/real_markets.

### 3.7 single_heston_pinn/ (4 Sep and 8 Sep)

- What, in two parts:
  - src/ and scripts/ hold a single Heston PINN for 13 NSE power-sector stocks. It predicts implied total variance, and the price follows from the Black formula.
  - research/ holds classical single Heston studies: an overfitting audit, a next-day forecast, a generalisation tweak, and a single-vs-double comparison.
- Why: to build and measure a one-factor baseline on Indian stock options.
- Results:
  - PINN: "0.02541 implied-volatility RMSE against 0.02710 for exact Fourier Heston", with the whole-panel optimiser at "67 s against 1722 s" (single_heston_pinn/outputs/PINN_SINGLE_HESTON_REPORT.md).
  - Classical comparison: "Single Heston has IV RMSE 0.044200 and Double Heston has IV RMSE 0.045510 ... Double Heston beats Single Heston for 2 of 11 stocks", with the recommendation "retain Single Heston" (research/outputs/single_double_heston_comparison/HONEST_SINGLE_DOUBLE_HESTON_REPORT.md).
  - Next-day forecast: Heston 0.044469 against a naive baseline 0.068079 (research/outputs/single_heston_next_day_forecast/NEXT_DAY_FORECAST_REPORT.md).
  - Overfitting audit: a quadratic smile fitted on the same day beats Heston on held-out strikes, 0.014712 against 0.030028.
- Status: RECORD. The single Heston PINN code was not reused elsewhere.

### 3.8 outputs/ folders

**outputs/unified_v6/** (4–5 Sep). **Status: LIVE (website) and USED (DH 2.0).**
- What: the trained unified calibrators: unified.pt, the retrained unified_v2.pt, and the fine-tuned unified_ft.pt. It also holds the test sets and every result JSON.
- Why: the final deliverable of the unified redesign.
- Results:
  - finetune_real_summary.json, NIFTY (10 dates, 1,750 held-out quotes), median IV RMSE: fine-tuned with 3 refinement steps 0.02478, classical Double Heston 0.02523, single Heston 0.02625, Black-Scholes 0.02982. The fine-tuned model was best on 4 of 10 dates, with median time 1.16 s against 25.26 s.
  - finetune_real_summary.json, ADANIPOWER: fine-tuned 0.01819 against classical 0.00398. "improvement_ft_vs_base_pct" is −3.7, meaning fine-tuning made it slightly worse.
  - panel_test_summary.json (140 single- or two-expiry stock surfaces): fine-tuned 0.0176 against classical Double Heston 0.0069, Wilcoxon p for fine-tuned vs base 0.549.

**outputs/real_markets/** (4 Sep). **Status: USED (DH 2.0).**
- What: the NIFTY 10-date selection (nifty_selection.json) and the dual-PINN NIFTY comparison (nifty_comparison/*).
- Why: the first real-market test of the dual PINN.
- Result: nifty_comparison_summary.json gives a median held-out IV RMSE of 0.02293 for "dual PINN + ridge 3", 0.02361 for single Heston and 0.03039 for Black-Scholes with one sigma.

**outputs/real_corpus/** (5 Sep).
- What: real_validation.npz, the real-quote validation corpus for fine-tuning.
- Status: RECORD.

**outputs/dual_pinn_legacy/** (4 Sep).
- What: the dual PINN weights (dual_aware.pt, dual_independent.pt) and dual_evaluation_summary_v2.json.
- Result: the networks alone beat single Heston on only "0.0333" of surfaces.
- Status: SUPERSEDED.

**outputs/calibration_repair/** (7 Sep).
- What: a repaired loss and refinement for the unified calibrator, three re-trained seeds, a synthetic recovery assessment, and a notebook.
- Why: the old loss "did not provide the claimed barrier against covariance collapse" (docs/CALIBRATION_REPAIR_HANDOFF.md).
- Results: actual IV improved only "about 1.20%". Network-only recovery was 0/24 in every condition. The network plus an exact optimiser reached 21/24 on clean rich surfaces and 0/24 with 1% noise (RESULTS.md).
- Status: RECORD.

**outputs/regular_pinn_recovery/** (7–18 Sep).
- What: the "regular" PDE PINN (5 layers × 160 tanh), trained for single and double Heston, then used to recover parameters. It has many variants (standard, Sobolev, continuation, grouped, parameter-aware, float64, deep17) and a residual-connection test.
- Why: to try the cleanest PINN route to recovery.
- Results:
  - "Double Heston parameter recovery is **not achieved**". Every double variant scored 0/12 (locked_assessment_report/REPORT.md).
  - The residual-connection test found "Residual connections do not improve parameter recovery", and recovery tracks pricing accuracy (Pearson r = +0.86) (residual_test/RESULTS.md).
- Status: RECORD. The same network class became the locked v4 C3.

**outputs/deeper_pinn/** (9 Sep).
- What: 11- and 17-layer versions of the regular PINN, error-covariance weighting ("GLS"), and a midpoint "ridge" penalty.
- Results:
  - Pricing: the 17-layer model "reduces clean heldout price RMSE by 98.06% versus the Single Heston PINN".
  - Recovery: "Parameter recovery is not better than Single Heston". The best was "2/24" (recovery_report/REPORT.md).
  - The ridge penalty cut noisy parameter error by 34.6% (regularized_report/REPORT.md).
- Status: RECORD.

**outputs/affine_factor_pinn/** (10 Sep, Double/Single-heston branch only).
- What: "factor-structured" PINNs that learn the building blocks of the model's characteristic function (affine/Riccati coefficients) instead of prices.
- Result: the seven variants passed 19 to 32 of 120 individual parameter checks and "0/12" complete cases (seven_variant_exposed12_report909101/REPORT.md).
- Status: RECORD (failed).

**outputs/literature_fixed_truth_pilot_20260915/** (15 Sep).
- What: one published Double Heston parameter set (DH1, from Kyriakou, Brignone and Fusai) plus two variants of it. A PINN tried to recover the settings from prices.
- Result: "Zero of six passed all ten parameter gates" (docs/LITERATURE_FIXED_TRUTH_RESULTS_20260915.md).
- Status: RECORD.

**outputs/literature_parameter_sweep_20260915/** (15 Sep).
- What: seven candidate "true" settings were screened, and the least-bad one was confirmed on fresh seeds.
- Result: "Zero of 18 recovered all ten parameters; only two of 180 individual parameter checks passed" (docs/PROPOSED_PARAMETER_SELECTION_RESULTS.md).
- Status: RECORD.

**outputs/nifty_fixed_crisis_20260917/** and **outputs/crisis_option_coverage_20260917/** (17 Sep).
- What: Single and Double Heston PINNs with fixed settings, tested on 3 NIFTY crisis windows (61 dates, 3,903 held-out quotes). The coverage audit was a pre-check showing that ADANIPOWER lacked options in those windows.
- Result: "This trial does not justify a meaningful or consistent Double Heston edge" (docs/NIFTY_FIXED_CRISIS_RESULTS_20260917.md).
- Status: RECORD. Its prepare/quote filters were frozen into nifty_multifactor v1–v4.

**outputs/metrics/pinn_*** (Archive-2, 14 Aug data, committed 22 Aug).
- Status: RECORD (see 3.1).

### 3.9 experiments/ — the NIFTY line, v1 to v6

Shared design of nifty_multifactor_v1–v4:
- Teacher and test surfaces. An exact Double Heston "teacher" (experiments/*/literature_exact.py) prices 100,000 training points. Fifty controlled synthetic surfaces come from five "families" built around one published parameter set (Chang, Wang & Zhang 2021), with fast and slow speeds 10.75 and 0.95.
- Models trained or fitted:
  - a PINN (`TorchRegularVariancePINN`, two seeds, 17 and 43), trained on exact prices plus the pricing equation at 18,000 points;
  - single Heston, fully refitted to each surface;
  - Black-Scholes, either flat or with a term structure.
- Market test. NIFTY options are tested on a frozen validation window, then on a separate final window opened once.
- Fairness rules. Every step is hash-frozen before results are seen.

**nifty_multifactor_v1** (17–18 Sep).
- Status: RECORD.
- The exact-price short-time test failed, because adaptive integration hit its limit at T = 1e-7. Nothing was trained.

**nifty_multifactor_v2.**
- Status: RECORD.
- Only the short-time reference integration changed (AMENDMENT.md).
- It passed, but the PINN's state range missed some scenario extremes.

**nifty_multifactor_v3.**
- Status: RECORD.
- It widened the slow-variance range to [.00015, .18] and the fast range to [.001, .22], then generated the teacher and baselines.
- Its PINN "failed, all 4" gates, because the price term was only "1.5–2% of the objective".

**nifty_multifactor_v4.**
- Status: LIVE (website quotes its numbers as text) and REUSED (its C3 weights are loaded by dh_pinn_v5, hardcoded_v1, price_curve_figures, financial_curve_figures_v2, clear_separation_figures_v1 and docs/figures).
- It rescaled each loss term to its accuracy target and climbed the ladder C1 → C2 → C3, locking C3 (width 256, depth 5).
- Results (FINAL_REPORT.md):
  - Accuracy: held-out price RMSE "1.10e-5 (IV RMSE 0.10 vol pts)" on 8,192 untouched points.
  - Controlled test: "DH-PINN < SH < BS ... on 40 of 40" surfaces.
  - Untouched NIFTY market test (967 quotes, 17 dates): exact Double Heston 107.6, PINN 107.7 and term Black-Scholes 22.4 index points. The cause was the parameter level (14.9% implied against the market's 10.4%), not the network ("PINN approximation error is 0.19 index points").

**nifty_multiscale_v5** (20–21 Sep, codex branch only).
- Status: SUPERSEDED by v6.
- What: three small width-32 "expert" branches added on top of frozen C3. The MULTISCALE version gates them by maturity; the SHARED version weights them equally.
- Results (REPORT.md):
  - SHARED was selected, with a "0.396%" fresh-test gain.
  - In three NIFTY windows (March 2020, June 2025, March 2026), calibrated Single Heston beat the restricted PINN every time: 48.09 against 82.86, 5.16 against 15.19, and 20.68 against 34.26 index points.

**pinn_architecture_v6** (20–21 Sep, codex branch only).
- Status: RECORD. The selected network is "not replaced as the default".
- What: a predeclared ablation of published PINN tricks on top of v5 SHARED: a plain correction branch, Wang's "modified MLP", gradient balancing, and residual-adaptive sampling (RAD). Copies of the official source code sit in research/official_*.py; those are not Dhruva's code.
- Result: the modified MLP was selected, with fresh price RMSE "1.0588034e-05 to 9.6101261e-06 (9.24%)" but "+50.06%" latency. Gradient balancing and RAD were rejected (PINN_ARCHITECTURE_IMPROVEMENT_REPORT.md).

**dh_pinn_v5** (25 Sep, main). Not the same as nifty_multiscale_v5.
- Status: REUSED (its final weights are loaded by financial_curve_figures_v2 and the IWM comparison). The market-residual "hybrid" is RECORD.
- What, three parts:
  - a fresh six-stage architecture ablation (A0–A5) starting from the C3 recipe;
  - a "Track B" market-residual head trained on SPX;
  - a multi-surface scan of 62 US option chains using a model-free "two timescale score" (multi_surface/RULE.md).
- Results (IMPROVED_DH_PINN_REPORT.md):
  - The gated A5 network gives price RMSE "1.062e-5 → 6.610e-6", but SPX error moves "5.850 → 5.847".
  - With an identical residual head, Double Heston 1.990 against Single Heston 1.917 vol points: "did **not** beat Single Heston".

### 3.10 experiments/ — crypto line

**Shared design** (btc_multifactor_v1 code, reused by the four others).
- Data: Deribit public trade archive, trades in a fixed 06:00–08:00 UTC window.
- Fitting: per date, fit Black-Scholes (one vol per expiry), single Heston (5 settings) and Double Heston (10 settings). Both Heston models use the same optimiser: differential evolution, then 12 local starts.
- Scoring: two held-out designs. Design B holds out whole alternate expiries and is the primary test. Design A holds out alternate strikes.
- Date choice: shock days chosen by a fixed rule (DVOL at least 1.2× its 60-day median), plus calm control days.
- Statistics: Wilcoxon tests and cluster bootstrap.
- Freezing: all rules frozen before data were fetched.

**btc_multifactor_v1.** Status: REUSED (its engine.py is the exact pricer used by most later experiments; its data are reused in portable_dual_pinn_v1).
- Result: shock B median held-out IV RMSE was Black-Scholes 8.13, single Heston 2.50 and Double Heston 1.86. "DH beats SH ✔ and DH beats BS ✔" (REPORT.md).
- A Black-Scholes bug (one vol per quote) was fixed in AMENDMENT_01.
- Later audit caveat: "NOT certified leakage-free", because forwards are rebuilt from option prices before the split (experiments/portable_dual_pinn_v1/deliverables/REPORT.md).

**eth_multifactor_v1.**
- Difference: the same inverse-option data code as BTC, with ETH DVOL shock dates.
- Result: Double Heston 2.32 against single Heston 3.23. It "replicates in full".

**xrp_multifactor_v1.**
- Difference: ENJ has no listed options, so XRP was used. The data code was rewritten for linear USDC options (data_xrp.py with `parse` and `solve_f`). Shock dates come from realised volatility because there is no DVOL.
- Result: single Heston 5.24 against Double Heston 5.72 on held-out expiries. Double Heston beating Black-Scholes was met; beating Single Heston was "not met".

**sol_multifactor_v1.**
- Difference: the same as XRP, applied to SOL.
- Result: Double Heston 3.67 against single Heston 3.82, with p = 0.23, so "not met".

**hype_multifactor_v1.**
- Difference: the same as XRP. Options only listed from 22 Jun 2026, so every day is a test day and the statistical unit is the ISO week.
- Result: Double Heston 3.26 against single Heston 3.61, but the bootstrap lower bound was −0.06, so "not met" (narrowly).

**cross_asset_summary.py** and **cross_asset_figures/**.
- Status: RECORD.
- One summary figure across all five coins.
- Its title states the result: "DH always beats BS; it beats SH significantly only where many liquid expiries exist" (cross_asset_primary.csv: BTC and ETH significant; SOL, HYPE and XRP not).

### 3.11 experiments/ — the rest

**portable_dual_pinn_v1** (20 Sep).
- Status: RECORD (failed).
- What: two parts.
  - It re-audited the BTC result.
  - It trained a small "dual-maturity" PINN with short and long experts mixed by a smooth gate. The aim was that it would work on any asset.
- Result: "Predeclared development criterion passed: **False**". The physics error got 148% worse (deliverables/REPORT.md).

**hardcoded_v1** (24 Sep).
- Status: RECORD. Its SPX surface is REUSED by dh_pinn_v5.
- What: the published Christoffersen–Heston–Jacobs parameters, with nothing fitted, tested against SPX today and against BTC and ETH shock days.
- Result: on SPX, Double Heston 5.34 against single Heston 5.36 against Black-Scholes 6.86 vol points, which is "nothing" between the two Heston models. On crypto, every model was off by 30–60 vol points (REPORT.md).

**price_curve_figures** (24 Sep).
- Status: RECORD (figures).
- Plots of C against S and C against maturity for fixed parameters, plus a diagram of the C3 network.

**financial_curve_figures_v2** (25 Sep).
- Status: RECORD (figures).
- Presentation figures on linear price axes with residual panels. All 30 curves passed the shape checks (no delta, gamma or price-bound violations).

**dh_relevance_selection_v1** (25 Sep).
- Status: RECORD. Its IWM surface is REUSED by clear_separation_figures_v1.
- What: a fixed rule scored 62 symbols to find the surface where a second timescale should matter most, and picked IWM. All models were then compared on held-out quotes.
- Result: Double Heston 0.583, single Heston 1.893, Black-Scholes 5.388 vol points.
- The report itself discloses that the selection "is **not blind**" (RESULTS.md).

**clear_separation_figures_v1** (25 Sep).
- Status: RECORD (figures).
- Figures of time value, IV smile and term structure that make the model differences visible.
- On the controlled benchmark, the PINN is "100–200× closer to exact Double Heston than either one-factor model". The last commit fixed a mislabelled Black-Scholes curve.

**model_reference_v1** (25 Sep).
- Status: reference documents, the best entry point for a newcomer.
- What: MODEL_EQUATIONS_REFERENCE.md and .pdf (a formula sheet for each model plus the PINN loss) and HOW_IT_WORKS.pdf (a plain-language explainer).

**import_smoke_20260821.**
- Status: RECORD.
- A smoke-test run of Archive-2.

### 3.12 docs/ (top level) and docs/figures/

- What, top level: protocol and handoff notes for each study: REGULAR_PINN_*, CALIBRATION_REPAIR_*, DEEPER_PINN_COMPARISON_PROTOCOL, LITERATURE_FIXED_TRUTH_*, PROPOSED_PARAMETER_SELECTION_*, NIFTY_FIXED_CRISIS_RESULTS, CRISIS_PRICE_RECOVERY_FEASIBILITY, RESEARCH_UPDATE_20260920, AFFINE_FACTOR/COMPOSITION/DEEP_REGULAR protocols, and PINN_RESEARCH_GITHUB_HANDOFF.
- What, docs/figures/bs_pinn_curve_validation/: 21 figures that check the v4 C3 PINN's price curves for sensible shape.
- Result: on a 121 × 121 grid, "zero detected stock-price monotonicity/convexity violations", with RMSE 0.00244455 against exact Double Heston (docs/RESEARCH_UPDATE_20260920.md).
- Status: RECORD.

### 3.13 tests/

- What: about 50 test files across branches, listed one per line in section 4.16. They check the physics, block leaks between training and test data, and catch numerical failures.
- Status: the ones on main still exist. Nothing in the website or DH 2.0 runs them.

---

## 4. Function by function

How to read these tables:
- Each table has three columns: name, one plain sentence on what it does, and "used by".
- In "used by", "this file" means it is a helper called only inside its own file. "tests" means only a test file calls it.
- "Archived" means the scripts that used to call it are no longer in the repo.
- Files that only contain a `main()` entry point get one row.
- `__init__` and `forward` rows are kept only where they say something useful. In PyTorch, `forward` is "what happens when the network is run on an input".

### 4.1 src/mentor_dh_pinn/ on main: the unified calibrator family (used for the website number)

**unified.py**: the "unified physics-informed set calibrator" that reads any set of option quotes and outputs ten settings.

| Name | What it does | Used by |
|---|---|---|
| `fourier_features` | Turns a number like maturity into a bundle of sine and cosine waves so the network can "see" it at many scales, like describing a time by both the hour and the minute hand. | this file |
| `quote_features` | Turns each option quote (spot, strike, maturity, rate, price) into a scale-free token, without assuming any fixed grid of strikes. | this file |
| `SetBlock` | One layer of self-attention in which every quote looks at every other quote, like students in a room comparing notes. | this file |
| `ParameterBlock` | One round in which ten "parameter tokens" first read the quotes and then read each other. | this file |
| `UnifiedCalibrator` | The whole model: it embeds the quotes, runs the parameter tokens, outputs a best guess plus an uncertainty, and then polishes the guess with a few exact-pricer steps. | evaluate_unified, evaluate_real_markets, evaluate_finetune, evaluate_panel_test, run_ablations, run_finetune, train_unified, assess_calibration_repair, evaluate_repair_validation |
| `UnifiedCalibrator.encode` | Converts a padded batch of quotes into token vectors and marks which slots are padding. | this file, finetune_projection |
| `UnifiedCalibrator.tokens_forward` | Runs the ten parameter tokens through the communication rounds and keeps the attention weights. | this file, run_ablations, diag_multistart, diag_real |
| `UnifiedCalibrator.gaussian_head` | Turns the tokens into a centre estimate (mu_z) and a full 10 × 10 uncertainty matrix that is guaranteed to be valid. | this file |
| `UnifiedCalibrator._prices` | Prices every quote with the exact Fourier pricer at the given latent settings. | this file |
| `UnifiedCalibrator.refinement_objective` | Scores a candidate: squared price misfit divided by quote noise, plus a pull back toward the network's own guess, so one bad quote invalidates the whole surface. | this file, evaluate_repair_validation |
| `UnifiedCalibrator.refine` | A few damped Gauss-Newton steps (a fast "nudge toward a better fit" method) that are accepted only if they really lower the score. | this file and most evaluation scripts |
| `UnifiedCalibrator.local_identifiability` | Measures, through a singular-value decomposition (SVD) of the sensitivities, which directions in setting-space the quotes actually pin down and which are flat. | assess_calibration_repair |
| `UnifiedCalibrator.forward` | Runs encode, tokens, head and refinement in one call and returns the refined settings. | all callers |
| `quote_scale` | Picks how big a "normal" quote error is: an explicit per-quote noise if given, otherwise a relative noise level. | this file |
| `_batched_jacobian` | Computes how every quote's price changes with each of the ten settings, for a whole batch at once. | this file |

**params_v2.py**: the one-to-one map between ten free numbers and ten valid Double Heston settings.

| Name | What it does | Used by |
|---|---|---|
| `_softplus`, `_sig` | Smooth functions that keep numbers positive or between 0 and 1, working in both NumPy and PyTorch. | this file |
| `_share` | A 0-to-1 split that never touches exactly 0 or 1, so neither variance factor can become zero. | this file |
| `decode` | Turns ten unrestricted numbers into ten settings that always obey the engine's rules (Feller, slow slower than fast, correlation limits). | unified, dataset_v6, baselines, train_unified, run_finetune, many evaluation scripts |
| `encode` | The exact reverse of `decode`; it refuses settings outside the model's valid set instead of quietly clipping them. | dataset_v6, baselines, assess_calibration_repair |
| `to_array` | Converts the decoded list into a plain NumPy array. | dataset_v6 |
| `encode_batch` | The same as `encode` but for many rows at once. | train_unified, run_finetune, evaluate_unified, run_ablations |

**dataset_v6.py**: the synthetic training data for the unified calibrator.

| Name | What it does | Used by |
|---|---|---|
| `sample_parameters` | Draws random setting vectors evenly across four volatility "regimes" (low-vol index, ordinary equity, high-vol stock, and so on). | build, assess_calibration_repair |
| `sample_geometry` | Invents one realistic layout of quotes: how many expiries, which strikes, which maturities. | build |
| `expected_total_variance` | Computes the average variance expected over an option's life, the "baseline volatility" of a surface. | this file |
| `admissible` | Keeps only quotes where the exact pricer is numerically trustworthy. | build, regular_pinn_data |
| `apply_noise` | Adds realistic noise to clean prices: correlated, uneven, heavy-tailed, with occasional outliers. | build |
| `price_batch` | Prices many variable-length surfaces exactly in one batched call. | build |
| `build` | Generates n complete noisy surfaces with masks and saves them. | build_v6_data, prepare_repair_synthetic |

**collate.py**

| Name | What it does | Used by |
|---|---|---|
| `collate` | Packs selected surfaces into one padded batch trimmed to the longest surface, like putting different-length rows into one table. | train_unified, evaluate_unified, run_ablations, run_finetune |

**ood.py**: out-of-distribution warnings, a "the model is guessing here" alarm.

| Name | What it does | Used by |
|---|---|---|
| `build_reference` | Records, once and from training data only, the typical ranges of quote counts, expiries and settings. | evaluate_unified |
| `_pct` | Estimates where a value falls within a stored list of percentiles. | this file |
| `assess` | Labels one calibration as in-distribution, mild extrapolation or severe extrapolation. | evaluate_unified, evaluate_real_markets |

**baselines.py**: classical comparison models, rebuilt after the originals were lost.

| Name | What it does | Used by |
|---|---|---|
| `black76` | The Black formula for a call on a forward. | this file, evaluation scripts |
| `bs_surface` | Prices a whole surface with one Black-Scholes volatility. | evaluation scripts, diag scripts |
| `fit_black_scholes` | Finds the single best volatility by a grid scan followed by a precise one-dimensional search. | evaluation scripts, diag scripts |
| `_sh_decode`, `_sh_encode` | Map between free numbers and valid single Heston settings (Feller condition and correlation limits enforced). | this file |
| `sh_surface` | Prices a surface with single Heston through the exact PyTorch pricer. | evaluation scripts |
| `fit_single_heston` | Least-squares fit of the 5 single Heston settings to one surface. | evaluation scripts |
| `dh_surface` | Prices a surface with Double Heston through the exact pricer. | evaluation scripts, run_ablations |
| `fit_double_heston` | Classical multi-start Double Heston fit, optionally pulled toward a prior, so the same routine also serves as the "scalar ridge" baseline. | evaluation scripts, diag scripts |

**torch_pricer.py**: the exact Double Heston pricer, written so that gradients flow through it. Viraj later modified it on 11 Sep.

| Name | What it does | Used by |
|---|---|---|
| `_complex_log1p` | A numerically safe log(1+z) for complex numbers, including tiny "subnormal" values. | this file |
| `laguerre_rule` | Returns the Gauss-Laguerre nodes and weights (the fixed sampling points of the integral). | this file |
| `_heston_log_cf` | The "little Heston trap" formula for one variance factor's characteristic function, the mathematical fingerprint of the price distribution. | this file |
| `_dh_cf` | Adds the two factors' fingerprints to get the Double Heston one. | this file |
| `price_call` | Exact European call prices for any batch of quotes, differentiable with respect to all ten settings. | almost everything: baselines, dataset_v6, unified, regular_pinn_data, nifty_fixed_crisis, nifty v4 teacher_quality, and others |
| `price_call_single` | The same for single Heston (5 settings). | baselines, nifty_fixed_crisis, regular_pinn_data |

**finetune_projection.py**: the 5 Sep "projection fine-tuning" that produced the website's NIFTY number.

| Name | What it does | Used by |
|---|---|---|
| `gaussian_nll_latent` | Scores how likely the true settings are under the network's predicted bell curve, without ever inverting a matrix. | run_finetune |
| `log_gauss_latent` | The same score with the opposite sign. | this file |
| `ProjectionFineTuner` | The training wrapper that keeps the network honest on synthetic data while teaching it to fit real quotes. | run_finetune |
| `ProjectionFineTuner._std` | Standardises latent coordinates with fixed means and spreads. | this file |
| `ProjectionFineTuner._encode` | Runs only the network part (no refinement) to get the centre and uncertainty. | this file |
| `ProjectionFineTuner._price` | Prices the active quotes at a latent point, filling padding slots with harmless values. | this file |
| `ProjectionFineTuner.recovery_loss` | The "anchor": on synthetic surfaces with known answers, it keeps predictions close to the truth so the model does not forget. | this file |
| `ProjectionFineTuner.projection_loss` | The new self-supervised goal: make exact model prices match real market quotes, with each error divided by the quote's vega (its sensitivity to volatility). | this file |
| `ProjectionFineTuner.self_consistency_loss` | An "ELBO" (evidence lower bound) term that rewards fitting the quotes while keeping the uncertainty sensible. | this file |
| `ProjectionFineTuner._balance` | Divides each loss by a running average of its own size, so no single term drowns the others. | this file |
| `ProjectionFineTuner.compute_losses` | Combines the three balanced losses for one synthetic batch plus one real batch. | run_finetune |
| `training_step`, `validation_step` | One training update, and one validation measurement of coverage and errors (PyTorch Lightning style). | Lightning or run_finetune |
| `on_before_optimizer_step`, `configure_gradient_clipping` | Skip updates whose gradients are not finite, and cap gradient size. | Lightning |
| `lr_lambda`, `configure_optimizers` | A gentle warm-up then cosine-decay learning-rate schedule with AdamW. | run_finetune |

**precise_calibration.py**

| Name | What it does | Used by |
|---|---|---|
| `exact_polish` | Starting from the network's guess, runs a SciPy optimiser with the exact pricer on calibration quotes only, from three starts, and returns all of them. | assess_calibration_repair |

**nifty_panel.py**: builds NIFTY option surfaces from NSE daily files.

| Name | What it does | Used by |
|---|---|---|
| `black76` | The Black call price on a forward. | evaluate_real_markets |
| `implied_vol` | Finds the volatility that reproduces a given price (root finding), or NaN if impossible. | evaluate_real_markets, evaluate_panel_test, evaluate_repair_validation |
| `bhav_path` | Builds the file path of an NSE F&O "bhavcopy" zip for a date. | build_real_corpus |
| `read_quotes` | Reads NIFTY call and put rows (settlement price, open interest) from one bhavcopy. | this file |
| `surface` | Cleans one day's liquid out-of-the-money quotes and reserves every third strike as held-out before estimating carry, so held-out quotes cannot leak into the fit. | build_real_corpus, evaluate_real_markets |
| `fit_smile` | Fits a quadratic total-variance smile per expiry, weighted by open interest. | this file |
| `canonical_grid` | Builds the dual PINN's fixed 45-number input (5 expiries × 9 strikes) by interpolating the smile. | archived (dual PINN) |

### 4.2 src/mentor_dh_pinn/ on main: the "regular" PDE PINN family (the locked v4 C3 network)

**regular_pinn_torch.py**: the most-reused file. Its class is the network behind the website's "1.1 × 10⁻⁵".

| Name | What it does | Used by |
|---|---|---|
| `mean_reversion_ratio` | Computes (1 − e^−u)/u stably, the factor that says how much of today's variance survives over time. | this file |
| `expected_variance` | Expected average variance over the option's life, summed over the factors: the "no-smile" baseline. | this file, maturity_dual_pinn |
| `unit_log` | Rescales a positive quantity to roughly [−1, 1] on a log scale. | this file |
| `black_call` | Normalised Black call price from total variance, with the exact payoff when variance is zero. | this file |
| `TorchRegularVariancePINN` | The PINN itself. It takes moneyness, variance states, maturity and the structural settings, builds engineered features, runs a tanh network, and outputs a bounded correction to the baseline volatility. | nifty_multifactor v1–v4, dh_pinn_v5, hardcoded_v1, price_curve_figures, portable_dual_pinn_v1, many scripts |
| `.from_mlx` | Copies weights trained on Apple's MLX library into this float64 PyTorch version, switching to the deep variant if needed. | assess_regular_pinn, check_regular_pinn_physics, others |
| `.features` | Builds the input features: moneyness, standardised moneyness, log maturity, per-factor terms and skew terms. | this file, subclasses |
| `.correction` | Runs the hidden layers and returns the bounded log-volatility correction (at most ±1.8). | this file, subclasses |
| `.iv` | Implied volatility = baseline volatility × e^correction. | callers |
| `.log_total_variance` | Log of implied total variance, for positive maturity. | this file |
| `.price` | Normalised call price via the Black formula, exactly equal to the payoff at expiry. | callers |
| `residual` | The Double Heston pricing-equation error at a point, computed with automatic differentiation and scaled so that it is comparable everywhere. This is the "physics" in the PINN. | training code in experiments and scripts |

**regular_pinn.py**: the Apple-MLX original of the same network, used for training.

| Name | What it does | Used by |
|---|---|---|
| `mean_reversion_ratio`, `_unit_log`, `expected_variance`, `black_call` | The same helpers as in the PyTorch file, written for MLX. | this file |
| `RegularVariancePINN` | The trainable 1- or 2-factor PINN (13 or 24 features), with `features`, `correction`, `log_total_variance`, `iv` and `price` as in the PyTorch copy. | train_regular_pinn, assess_regular_pinn, precheck_double_v2_mac, deep_regular_pinn |
| `residual` | The scaled pricing-equation residual, plus a convexity factor (Durrleman) that flags butterfly arbitrage. | train_regular_pinn, check_regular_pinn_physics |

**regular_pinn_data.py**: synthetic training labels for the regular PINN.

| Name | What it does | Used by |
|---|---|---|
| `decode_unit` | Maps numbers in [0, 1] to physical settings inside a fixed, Feller-feasible box, with slow kept slower than fast. | many scripts |
| `coordinates` | Splits a query row into state coordinates and structural settings. | many scripts |
| `expected_variance` | The baseline average variance for query rows. | prepare scripts |
| `draw_points` | Latin-hypercube sampling of training points, with half the maturities on fixed 30/60/90/180/365/730-day slices. | prepare_regular_pinn, train_regular_pinn, check scripts |
| `black_call` | Black call price (NumPy). | this file |
| `invert_total_variance` | Recovers total variance from a price by bisection (75 halvings), or NaN if invalid. | about 15 scripts and experiments |
| `exact_prices` | Exact Fourier prices for query rows (single or double). | assess_regular_pinn, train_unified |
| `teacher_labels` | Exact prices plus how the IV changes with each setting, computed in float64 by implicit differentiation. | prepare scripts, train_regular_pinn |

**deep_regular_pinn.py**

| Name | What it does | Used by |
|---|---|---|
| `ResidualBlock` (MLX) and `TorchResidualBlock` | An extra layer that starts as "do nothing" (zero output weights) and adds 0.1 × a learned tweak. | this file |
| `DeepRegularVariancePINN` (MLX) and `TorchDeepRegularVariancePINN` | The regular PINN with extra residual blocks on top (the 11- and 17-layer models). | train_regular_pinn, assess_regular_pinn |
| `.correction` | Runs the base layers, then the residual blocks, then the bounded head. | this file |
| `.from_mlx` | Copies trained MLX weights into the PyTorch version. | regular_pinn_torch |

**surrogate_error.py**

| Name | What it does | Used by |
|---|---|---|
| `SurrogateError` | Stores the network's typical training errors (mean and covariance) for one fixed quote layout. | calibrate_recovery_pinn, evaluate_recovery_improvements, tune_recovery_prior |
| `SurrogateError.transform` | "Whitens" residuals, so that errors the network is known to make count less (generalised least squares). | same |

**identifiability.py**

| Name | What it does | Used by |
|---|---|---|
| `canonicalize_parameters` | Puts the two factors in a fixed order (by speed), because swapping them gives identical prices. | assess_regular_pinn, diagnose_identifiability_v2 |
| `normalized_errors` | Relative error for positive settings and absolute error for correlations. | this file |
| `match_factors` | Checks whether a prediction matches the truth better with the factors swapped. | this file |
| `recovery_metrics` | Pass/fail per setting with a 5% tolerance (0.05 for correlations), plus counts. | diagnose_identifiability_v2 |
| `parameter_error_summary` | Per-setting error table for a batch. | diagnose_identifiability_v2 |

**recovery_diagnostics.py**

| Name | What it does | Used by |
|---|---|---|
| `price_and_jacobian` | Exact price over spot and its sensitivity to each physical setting. | diagnose_identifiability_v2 |
| `physical_to_unit` | Reverses the training box map with no clipping. | diagnose_identifiability_v2 |
| `information_diagnostics` | Measures how much information the quotes carry about each setting (an SVD of the sensitivities). | diagnose_identifiability_v2 |

**parameter_transforms.py**

| Name | What it does | Used by |
|---|---|---|
| `to_optimizer` | Converts unit settings into the optimiser's working coordinates (plain unit or logit). | assess_regular_pinn |
| `to_unit_and_jacobian` | Converts back and also returns the slope of that conversion. | assess_regular_pinn |

**reproducible_regular_pinn_data.py** (written by Viraj on 11 Sep; listed because Dhruva re-committed it on the codex branch)

| Name | What it does | Used by |
|---|---|---|
| `runtime_provenance`, `sha256`, `_little_endian`, `canonicalize`, `npy_bytes`, `array_sha256`, `write_deterministic_npz` | Make data files byte-identical on any machine and fingerprint them. | this file |
| `lhs_unit_block`, `proposal_block`, `structural_validity`, `rejection_codes`, `label_proposals`, `accepted_split`, `collocation_set` | Generate, check and label synthetic training points in fixed blocks, and record why any point was rejected. | this file |
| `generate_dataset` | Writes the frozen "DOUBLE_DATA_V2_REPRODUCIBLE" training, validation and collocation set. | prepare_regular_pinn_v2 |

**maturity_dual_pinn.py** (20 Sep)

| Name | What it does | Used by |
|---|---|---|
| `PortableVariancePINN` | A smaller single-branch control PINN with scale-free features, so it should work for any asset. | portable_dual_pinn_v1 |
| `PortableVariancePINN.features` | The scale-free input features. | this file |
| `MaturityDualPINN` | Two expert networks (short and long maturity) blended by a smooth gate at 90 days. | portable_dual_pinn_v1, tests |
| `MaturityDualPINN.correction` | Runs both experts and mixes their outputs by the gate. | this file |
| `european_price` | Converts the network's normalised output into a real-currency price, given an independent forward and discount factor. | portable_dual_pinn_v1, tests |

### 4.3 src/mentor_dh_pinn/ on main: the early (4 Sep) forward PINNs and dual PINN, now archived

No current script imports these; their training scripts are not in the repo.

**model_v3.py**

| Name | What it does | Used by |
|---|---|---|
| `mean_reversion_ratio` | Stable (1 − e^−u)/u with a Taylor-series branch near zero. | model_v4, model_v5 |
| `expected_integrated_variance` | Exact expected average variance for Double Heston. | this file |
| `_normal_cdf` | The bell-curve cumulative probability. | this file |
| `black76_normalised` | Black call over strike from total variance. | model_v4, model_v5 |
| `HardConstrainedDoubleHestonPINN` | A PINN for one fixed parameter set that predicts implied total variance, so price bounds hold automatically. | archived |
| `.features`, `.log_total_variance`, `.price`, `.forward`, `.predict_price`, `._unit`, `.parameter_count` | Build inputs, compute log total variance, turn it into a price, and count weights. | this file |

**model_v4.py**

| Name | What it does | Used by |
|---|---|---|
| `_unit_log`, `_unit` | Rescale values to [−1, 1] on a log or linear scale. | this file |
| `ConditionedDoubleHestonPINN` | Like v3, but takes the structural settings as inputs, so one network covers many parameter sets. | archived |
| `.vbar`, `.features`, `.log_total_variance`, `.price` | Baseline variance, inputs, log total variance, price. | this file |
| `broadcast_params` | Expands one ten-vector to every row. | archived |
| `params_from_columns` | Builds the per-row parameter dict from table columns. | archived |

**model_v5.py**

| Name | What it does | Used by |
|---|---|---|
| `ConditionedDoubleHestonPINNV5` | v4 with richer inputs. | archived |
| `_d` | One automatic derivative helper. | this file |
| `residual_and_structure` | The pricing-equation residual plus four shape quantities (delta, convexity, two vegas) in one pass. | archived |
| `structural_penalty` | Penalises delta outside [0, 1], negative convexity and negative vega. | archived |

**data_v4.py**

| Name | What it does | Used by |
|---|---|---|
| `load_panel` | Reads the sealed 10,000-vector parameter panel. | dual_pinn_data |
| `_valid` | Checks the structural rules row by row. | dual_pinn_data |
| `draw_parameters` | Mixes panel vectors with fresh in-box draws. | this file |
| `stationary_bounds` | The range a variance factor spends most of its time in (gamma distribution quantiles). | this file |
| `draw_states` | Draws variance states from each vector's own long-run distribution. | this file |
| `mean_reversion_ratio` | Stable (1 − e^−u)/u. | this file |
| `build` | Makes the v4 training set of priced points. | archived |

**collocation_domain.py**

| Name | What it does | Used by |
|---|---|---|
| `FactorLaw` | The long-run (gamma) distribution of one variance factor, with mean, spread, half-life, quantiles and sampling. | this file |
| `CollocationDomain` | The four-axis box (spot, slow variance, fast variance, maturity) where the equation is enforced, derived from one parameter vector. | archived |
| `.admissible_for_reference` | Says where the 64-node exact pricer is accurate enough to supervise. | archived |
| `.standardised_moneyness` | log(F/K) divided by the total standard deviation. | this file |
| `sample_variance_states`, `sample_maturities`, `sample_spots` | Draw training points concentrated where prices are sensitive. | archived |

**dual_pinn.py**

| Name | What it does | Used by |
|---|---|---|
| `_logbox` | Squeezes a number into [lo, hi] on a log scale. | this file |
| `_mlp` | Builds a simple stack of linear layers with SiLU activations. | this file |
| `encode_surface` | Turns a surface into scale-free features (log price, per-expiry level and slope). | this file |
| `ShortEndNet` | Reads short expiries and outputs today's variances, vol-of-vol ratios and correlations. | DualPINN |
| `LongEndNet` | Reads long expiries and outputs long-run levels and speeds. | DualPINN |
| `combine` | Joins both outputs into a valid ten-vector, with Feller guaranteed by construction. | this file |
| `stack_short`, `stack_long`, `stack_canonical` | Pack outputs into tensors. | this file |
| `DualPINN` | Runs the two nets once, or iterates them to agreement ("mutually aware"). | archived |

**dual_pinn_data.py**

| Name | What it does | Used by |
|---|---|---|
| `draw_truths` | Draws true settings from the sealed panel plus fresh ones. | this file |
| `price_surface` | Prices the fixed 45-quote surface (5 expiries × 9 strikes). | this file |
| `build` | Makes clean surfaces, a noise level and noisy surfaces for training. | archived |
| `save` | Saves arrays to a compressed file. | archived |

**calibrate.py**

| Name | What it does | Used by |
|---|---|---|
| `_sig` | Logistic squashing function. | this file |
| `decode`, `encode` | The old (v1) map between free numbers and settings, using the `PARAM_BOX` that Deliverable A found silently clips. | polish |
| `Surface` | A simple container for one surface's quotes. | this file |
| `FourierCalibrator` | Calibration engine using the exact NumPy pricer (`prices`, `residual`). | archived |
| `PinnCalibrator` | Calibration engine using a frozen PINN, with `jacobian` computed by autograd. | archived |
| `calibrate` | Bounded multi-start least squares that keeps every start. | archived |

**polish.py**

| Name | What it does | Used by |
|---|---|---|
| `exact_surface` | Prices a geometry with the production engine. | this file |
| `prior_scale` | The typical per-quote noise size for this surface. | this file |
| `polish` | "Stage two": a warm-started least-squares fit pulled toward the network's answer by a ridge, the method behind "dual PINN + ridge 3". | archived |
| `blend` | Averages two networks' answers in free coordinates, so the result stays valid. | archived |

### 4.4 src/mentor_dh_pinn/ branch-only modules

**Double/Single-heston branch (10 Sep): factor-structured and composition PINNs (all failed recovery)**

| File · Name | What it does | Used by |
|---|---|---|
| affine_factor_pinn · `ratios`, `scales` | Stable helper ratios and the natural size scales of a factor's coefficients. | factor PINN files |
| affine_factor_pinn · `AffineFactorPINN` | Learns the two building-block functions (D and A) of one factor's characteristic function instead of learning prices directly. | train_affine_factor_pinn, other factor files |
| affine_factor_pinn · `.raw_corrections`, `.normalized_coefficients`, `.coefficients`, `.log_cf` | Network outputs, turned into complex coefficients and then into the log characteristic function. | this family |
| affine_factor_pinn · `riccati_residual` | The error in the Riccati equations those coefficients must satisfy (this family's "physics"). | training |
| affine_factor_pricing · `quadrature`, `neural_call_prices` | Integrate the learned characteristic function into call prices. | affine_factor_calibration |
| affine_factor_reference · `reference_coefficients`, `draw_factor_points`, `correction_targets` | Exact coefficient labels and training points, used only for training and checking. | training |
| affine_factor_calibration · `fit_factor_pinn` | Fits the ten settings through the frozen factor network, using IV error only. | assess_affine_factor_pinn |
| component_pinn_data · `decode_component_unit`, `component_coordinates` | A wider one-factor training box that covers both factors of Double Heston. | convolution experiments |
| conjugate_factor_pinn · `unshifted_coordinates`, `ConjugateFactorPINN`, `build_factor_pinn` | Enforces an exact identity between shifted and unshifted coefficients, and picks the architecture from the config. | factor scripts |
| convolution_pinn · `ImplicitBlackIV` | Black IV inversion with an exact first derivative for backpropagation. | this file |
| convolution_pinn · `ConvolutionPINN` (`price`, `iv`, `price_and_diagnostics`) | Builds Double Heston prices by combining one single-factor PINN used twice. | composition scripts |
| integrated_factor_pinn · `IntegratedFactorPINN` | Learns only A and gets D as its exact time derivative. | factor scripts |
| integrated_factor_pinn · `linear_coefficient_targets` | Re-expresses labels in linear units. | training |
| moment_factor_pinn · `MomentFactorPINN.primitive` | Forces the first moment (mean variance) to be exact. | factor scripts |
| two_moment_factor_pinn · `primitive_imaginary_slope`, `TwoMomentFactorPINN.primitive` | Forces the first two moments to be exact. | factor scripts |

**codex/improved-double-heston-pinn branch (21 Sep)**

| File · Name | What it does | Used by |
|---|---|---|
| multiscale_pinn · `TorchMultiscaleVariancePINN` | Adds three small expert branches on top of a frozen C3 network, starting as "no change". | nifty_multiscale_v5/run.py |
| multiscale_pinn · `.maturity_weights` | Smooth weights that hand control between experts at 30 and 90 days, or equal weights for "SHARED". | this file |
| multiscale_pinn · `.correction` | Parent correction plus the bounded weighted expert sum. | this file |
| research_modified_pinn · `ModifiedMLP` | Wang–Teng–Perdikaris "modified MLP": every layer mixes two input encodings through a gate, optionally with adaptive tanh. | this file |
| research_modified_pinn · `ResearchCorrectionPINN` | Wraps the frozen v5 network and adds 0.05 × tanh(modified-MLP output) as a correction. | pinn_architecture_v6/train.py |
| research_modified_pinn · `.branch_features`, `.correction` | Standardised inputs for the new branch, and parent plus correction. | this file |

### 4.5 scripts/mentor_dh_pinn/ on main

Most scripts are one-shot programs: they read inputs, do one job and write a folder. Helper functions inside a script are listed only when they do something meaningful.

| Script · Name | What it does | Used by |
|---|---|---|
| build_v6_data (no functions) | Generates the unified calibrator's synthetic train, validation and test sets. | run by hand |
| train_unified · `masked_mse` | Mean squared error over real quotes only, skipping prices the engine could not compute. | this file |
| train_unified · `gaussian_nll` | How unlikely the true settings are under the predicted bell curve. | this file |
| train_unified · `exact_prices` | Exact prices for a batch. | this file |
| train_unified · `Balance` | Scales each loss by its own running size, then applies phase weights. | this file |
| train_unified · `main` | Trains the unified calibrator through a staged curriculum (14 epochs, 72 minutes according to Deliverable F). | run by hand |
| evaluate_unified · `load_model`, `geo_of`, `reprice_rmse`, `main` | Synthetic benchmark: repricing, parameter error, buckets, uncertainty coverage, OOD and speed against baselines. | run by hand |
| evaluate_real_markets · `load_model`, `as_batch`, `iv_rmse`, `nifty_cases`, `adanipower_cases`, `run`, `main` | Scores Black-Scholes, single Heston, classical Double Heston and the unified model on real NIFTY and ADANIPOWER holdouts. | run by hand |
| run_ablations · `load`, `metrics`, `main` | Switches pieces off at run time (refinement steps, spring type, rounds) to see what matters. | run by hand |
| write_final_report (no functions) | Writes Deliverable F from the stored results. | run by hand |
| diag_real (no functions) | Tests whether the real-data gap is an optimisation problem or a prior problem. | run by hand |
| diag_multistart (no functions) | Tests whether starting from the model's own uncertainty finds better answers. | run by hand |
| build_real_corpus · `_vega`, `_stratified_indices`, `_pack`, `stock_surfaces`, `nifty_surfaces`, `save`, `source_hash`, `corpus_summary`, `evenly`, `main` | Builds the real-quote fine-tuning corpus from the stock panel and NIFTY, keeping test dates out by calendar. | run by hand |
| run_finetune · `load_real`, `real_batch`, `validation_batches`, `main` | Runs projection fine-tuning, alternating synthetic and real batches. | run by hand |
| evaluate_finetune · `load_enc`, `main` | Base against fine-tuned on the frozen NIFTY and ADANIPOWER test sets; writes finetune_real_summary.json, the file the website reads. | run by hand |
| evaluate_panel_test · `load_enc`, `cases`, `score`, `main` | Base against fine-tuned on 140 single-expiry stock surfaces; writes panel_test_summary.json. | run by hand |
| prepare_repair_synthetic · `main` | Rebuilds the synthetic corpus for the repair study. | run by hand |
| assess_calibration_repair · `prices`, `iv_values`, `input_batch`, `main` | Synthetic recovery test of repaired checkpoints; truth is used only to generate and score. | run by hand |
| evaluate_repair_validation · `digest`, `score_quote`, `summarize`, `write_csv`, `evaluate_checkpoint`, `main` | Scores checkpoints on every real validation quote and marks a metric "unavailable" if any quote fails. | run by hand |
| diagnose_tail_precision · `price_and_iv`, `main` | Re-prices failed tiny-price quotes at 40- and 60-digit precision to check they are real. | run by hand |
| report_calibration_repair · `main` | Builds the repair report, figures and notebook. | run by hand |
| prepare_regular_pinn · `main` | Makes regular-PINN reference data (single or double). | run by hand |
| prepare_regular_pinn_surfaces · `preconditioner`, `main` | Makes training surfaces plus a "parameter-aware" error weighting. | run by hand |
| train_regular_pinn · `array`, `recovery_validation`, `make_validation_surfaces`, `main` | Trains the regular PDE PINN (MLX) with price, IV, equation and optional sensitivity losses, checking recovery on 4 fixed surfaces. | run by hand |
| check_regular_pinn_physics · `array_hash`, `distribution`, `evaluate_points`, `dataset_provenance`, `main` | Independent physics checks of a frozen PINN (equation residual, payoff, numerics). | run by hand |
| assess_regular_pinn · `fit_network` | Fits settings using only the network's IV and analytic slopes, never the true answer. | several scripts |
| assess_regular_pinn · `sha256`, `load_checkpoint`, `_query`, `_exact`, `_network_iv`, `_rmse`, `_score_price_iv`, `_write_csv`, `main` | Loads checkpoints, scores prices and IV, and writes the frozen assessment. | run by hand |
| report_regular_pinn · `finite`, `show`, `clean_json`, `read_json`, `collect_run`, `csv_write`, `plot_history`, `build_report`, `main` | Turns training runs into tables and plots without retraining. | run by hand |
| report_regular_pinn_assessment · `rows`, `yes`, `unique`, `sha`, `byte_rows`, `main` | Audits and reports the frozen assessment. | run by hand |
| diagnose_regular_pinn_generalization · `evaluate`, `row_bytes`, `recover_stage1_snapshot`, `main` | Compares training and validation errors, with no new fitting. | run by hand |
| diagnose_regular_pinn_validation · `exact_iv_jacobian`, `make_validation_surfaces`, `main` | Compares the network's sensitivities with the exact ones on 4 surfaces. | run by hand |
| compare_deep_pinn · `main` | Same-surface price comparison of the deep PINNs and their own-model recovery. | run by hand |
| diagnose_deep_gls · `fit_gls`, `main` | Development test of fitting with the network's known error covariance. | run by hand |
| diagnose_identifiability_v2 · `read`, `write_csv`, `historical_records`, `main` | Factor-swap symmetry audit, rescoring of old results, and an information study. | run by hand |
| calibrate_recovery_pinn · `calibrate_quotes`, `main` | Calibrates one frozen PINN to supplied quotes with the chosen "clean" or "noise" profile. | run by hand |
| evaluate_recovery_improvements · `main` | Raw neural fits against GLS fits on frozen synthetic cases. | run by hand |
| tune_recovery_prior · `main` | Picks the ridge strength (0, 0.1, 1, 10, 100) on development surfaces only. | run by hand |
| report_deep_comparison, report_recovery_improvements, report_regularized_recovery · `js`/`read`, `rows`, `main` | Write the three deeper-PINN reports while keeping pricing and recovery claims separate. | run by hand |
| literature_fixed_truth_pilot · `write_json`, `digest`, `feller`, `decode`, `initial_raw`, `data_coords`, `teacher` | Helpers: save JSON, fingerprint, Feller gap, settings map, blind start, coordinates, exact teacher prices. | this file, run_literature_parameter_sweep |
| literature_fixed_truth_pilot · `prepare` | Generates and audits teacher quotes. | same |
| literature_fixed_truth_pilot · `train` | Trains a PINN and its ten unknown settings together from quotes and the equation, with no access to the truth. | same |
| literature_fixed_truth_pilot · `score` | Opens the truth and grades recovery. | same |
| literature_fixed_truth_pilot · `main` | Dispatches prepare, train or score. | run by hand |
| audit_literature_pilot · `main` | Post-training audit that never changes a checkpoint. | run by hand |
| run_literature_parameter_sweep · `ranking`, `run_stage`, `confirmation_quotes`, `promotion_allowed`, `main` | Screens 7 candidate truths, freezes the least-bad one, confirms it on fresh seeds, and allows promotion only if every check passes. | run by hand |
| audit_crisis_option_coverage · `sha`, `parity_gate`, `main` | Counts which instruments have usable options in the crisis windows, with no model fitting. | run by hand |
| nifty_fixed_crisis · `sha`, `write_json`, `tensor` | Fingerprint, save JSON, make tensors. | this file |
| nifty_fixed_crisis · `single_match`, `parameters` | Builds a single Heston set matching the double set's total variance. | this file |
| nifty_fixed_crisis · `state`, `reference`, `lhs_xt`, `collocation` | Build network inputs, exact prices and training points. | this file |
| nifty_fixed_crisis · `normalized_call`, `raw_option` | Convert between quoted and normalised prices. | this file |
| nifty_fixed_crisis · `prepare`, `verify`, `train`, `score`, `plot` | The fixed-parameter NIFTY crisis test: clean quotes, freeze code, train the PINNs, score held-out quotes, plot. | run by hand; prepare is frozen into nifty v1–v4 |
| finetune_regular_pinn_lbfgs (Viraj) · `sha256`, `index_sha256`, `expected_data_hashes`, `load_torch_checkpoint`, `select_batches`, `select_jacobian_surfaces`, `surface_jacobian_loss`, `loss_components`, `fine_tune`, `save_checkpoint`, `create_output_directory`, `main` | Deterministic float64 L-BFGS polishing of the regular PINN's weights. | run by hand |
| prepare_regular_pinn_v2, evaluate_regular_pinn_development, precheck_double_v2_mac (Viraj) | Frozen v2 data, development scoring, and a Mac pre-flight check. | run by hand |

### 4.6 scripts/mentor_dh_pinn/ on the Double/Single-heston branch only (10 Sep)

| Script · Name | What it does | Used by |
|---|---|---|
| train_affine_factor_pinn · `main` | Trains the factor-structured Riccati PINN in float64. | run by hand |
| finetune_affine_factor_lbfgs · `FrozenFactorObjective` (`vector`, `assign`), `gradient_check`, `main` | L-BFGS weight refinement with a finite-difference check of the gradient. | run by hand |
| finetune_factor_prices · `load_surfaces`, `surface_loss`, `main` | "Price-aware" refinement using vega-weighted price error on separate training surfaces. | run by hand |
| assess_affine_factor_pinn · `evaluate_case`, `evaluate_cases`, `main` | Recovery test of factor PINNs on 4 or 12 exposed cases, optionally in parallel. | run by hand |
| audit_factor_fit_isolation · `main` | Replays one fit with corrupted held-out quotes and the exact pricer blocked, to prove isolation. | run by hand |
| audit_factor_price_validation · `main` | Actual-IV audit of reserved validation quotes. | run by hand |
| audit_factor_repricing · `main` | Recomputes price scores and per-expiry errors after fitting. | run by hand |
| diagnose_factor_pinn_bias · `main` | Compares network bias with local sensitivity after fitting. | run by hand |
| report_affine_factor_pinn · `audit_regular_comparison`, `audit_training`, `audit_assessment`, `main` | Audits and writes the factor-PINN reports. | run by hand |
| build_convolution_checkpoint · `main` | Packages a frozen single-factor network for the composition PINN. | run by hand |
| check_composition_physics · `raw_price_residual`, `main` | Checks the composed network against the raw pricing equation. | run by hand |
| compare_composition_pinn · `neural_only`, `without_timing`, `geometry`, `case_worker`, `main` | Same-quote comparison on 12 cases while forbidding exact-pricer calls inside fitting. | run by hand |
| diagnose_composition_bias · `main` | Training-only bias proxy. | run by hand |
| report_composition_pinn · `main` | Composition report. | run by hand |
| finetune_regular_lbfgs · `FrozenObjective` (`vector`, `assign`), `directional_check`, `main` | L-BFGS refinement of the regular PINN's weights (MLX). | run by hand |
| finetune_regular_torch · `tensor`, `anchor_loss`, `relevance_weights`, `physics_loss`, `validation_rmse`, `main` | Float64 PyTorch refinement using anchor labels and the equation only. | run by hand |
| regular_lbfgs_surface_loss · `stable_iv_error`, `FrozenSurfaceObjective` | Adds a grouped-surface error term as a parameter-bias proxy. | finetune_regular_lbfgs |
| regular_lbfgs_jacobian_loss · `training_geometry`, `neural_iv_unit_jacobian`, `FrozenJacobianObjective` | Adds a term making the network's sensitivities match the teacher's. | finetune_regular_lbfgs |
| audit_regular_continuation · `read_json`, `read_csv`, `without_seconds`, `collect`, `audit_training_and_isolation`, `main` | Audits continued training runs and replays the holdout isolation. | run by hand |
| diagnose_regular_pinn_roundtrip · `main` | Generates IVs with the network and checks it can recover its own settings (a self-consistency test). | run by hand |

### 4.7 single_heston_pinn/

**src/pinn_heston_core.py**: the single Heston PINN.

| Name | What it does | Used by |
|---|---|---|
| `_load_single_heston` | Loads the classical single_heston.py module by file path. | this file |
| `_panelled_grid` | Integration points concentrated where the price integral changes fastest. | this file |
| `log_return_cf` | The single Heston characteristic function ("little trap" form). | this file |
| `normalised_forward_call` | Exact Heston call price by Lewis' single-integral formula. | this file |
| `normalised_forward_call_batch` | The same, vectorised with a different parameter set per row. | this file |
| `implied_vol_batch`, `atm_implied_vol` | Exact implied vols, used as a realism screen. | this file |
| `black76_normalised`, `implied_total_variance` | Black price from total variance, and the inverse. | this file |
| `expected_integrated_variance_rate` | Expected average variance over the option's life. | this file |
| `Box` | The training range, with vol-of-vol drawn through the Feller ratio. | all PINN files |
| `draw_parameters` | Random Heston settings screened for economic realism. | this file |
| `_loguniform` | Random draws evenly on a log scale. | this file |
| `sample_collocation` | Physics points spanning spot from 0 to 1.5 × the ten-year price maximum. | pinn_run_training |
| `sample_anchor` | Points where the exact price supervises the smile. | pinn_run_training |
| `anchor_targets` | Exact price, total variance and the network's correction target. | pinn_run_training |
| `_mean_reversion_ratio`, `integrated_variance_rate_mx` | Stable baseline-variance helpers in MLX. | this file |
| `ImpliedVarianceNet` | The network: it predicts the log ratio of implied vol to the "no vol-of-vol" vol. | pinn_train |
| `Normaliser`, `_unit`, `build_features` | Scale the 13 inputs. | this file |
| `log_total_variance` | Network output turned into log total variance. | this file |
| `state_derivatives` | All derivatives needed by the equation, by nested autodiff. | this file |
| `heston_residual` | The Heston pricing-equation error in total-variance form. | pinn_train |
| `price_relevance_weight` | Weights physics points by how much they move the price. | pinn_train |
| `calendar_penalty`, `butterfly_penalty` | Penalise total variance falling with maturity, and butterfly arbitrage. | pinn_train |
| `_norm_cdf_mx`, `black76_normalised_mx` | MLX Black helpers. | this file |
| `ParameterTransform` (`decode`, `encode`) | Bounded map between free numbers and Heston settings. | pinn_calibrate |
| `PinnPricer` (`normalised_call`, `price`, `price_arrays`) | A pricer backed by the trained network. | pinn_calibrate |
| `audit_reference_pricer` | Cross-checks the Lewis reference against the repository engine. | pinn_build_report |

**src/pinn_train.py**

| Name | What it does | Used by |
|---|---|---|
| `TrainConfig` | Training settings. | pinn_run_training |
| `_mx`, `make_collocation_tensors` | Convert data to MLX arrays. | this file |
| `build_loss` | Huber-loss mix of equation error, smile anchor and arbitrage penalties. | this file |
| `train` | Trains the network. | pinn_run_training |
| `evaluate` | IV and price error against exact Heston. | pinn_run_training |
| `pde_residual_score` | Equation error on the collocation set. | pinn_run_training |

**src/pinn_calibrate.py**

| Name | What it does | Used by |
|---|---|---|
| `_assemble` | Turns calls into puts by put-call parity where needed. | this file |
| `FourierEngine` (`price`, `price_rowwise`) | Exact Heston pricer used for classical calibration. | pinn_run_calibration |
| `PinnEngine` (`_normalised`, `price`, `price_rowwise`) | The PINN as a pricer, evaluating inside its training box only. | pinn_run_calibration |
| `quote_arrays` | Pulls arrays out of a quote table. | this file |
| `surface_residual` | Vega-scaled price misfit. | this file |
| `fit_joint_structural` | Fits four shared settings across dates plus one today-variance (v0) per date. | pinn_run_calibration |
| `soft_l1_cost` | The robust cost being minimised. | this file |
| `fit_v0` | One-dimensional search for a date's v0 on calibration quotes only. | pinn_run_calibration |
| `predict` | Model prices and IVs for quotes. | pinn_run_calibration |
| `metrics` | IV and price error summary. | pinn_run_calibration |

**src/pinn_data_prep.py**

| Name | What it does | Used by |
|---|---|---|
| `_load_single_heston` | Loads single_heston.py. | this file |
| `ten_year_price_domain` | Per-stock price ceiling from ten years of NSE files. | main |
| `build_quote_panel` | All leakage-safe quotes the benchmark would use. | main |
| `variance_domain`, `maturity_domain`, `carry_domain` | Ranges of variance, maturity, rate and dividend for the PINN. | main |
| `main` | Writes the domain spec and quote cache. | run by hand |

**src/pinn_report.py**

| Name | What it does | Used by |
|---|---|---|
| `plot_training`, `plot_accuracy_vs_exact`, `plot_requested_domain`, `plot_market_fit`, `plot_collocation`, `plot_state_paths` | Figures for the PINN report. | pinn_build_report |
| `reconstruct_surface` | Prices a spot grid with an engine. | this file |

**src/single_heston.py**: classical single Heston calibration on NSE power stocks.

| Name | What it does | Used by |
|---|---|---|
| `sha256` | File fingerprint. | this file |
| `heston_characteristic`, `heston_call_prices`, `heston_prices` | Exact single Heston pricing (64-node Gauss-Laguerre integration). | research scripts |
| `black_scholes_price`, `black_scholes_vega`, `implied_volatility` | Black-Scholes price, vega and IV inversion. | research scripts |
| `encode_params`, `decode_params` | Bounded map for the 5 settings. | this file |
| `residuals` | Vega-scaled price errors. | this file |
| `fit_all_params` | Multi-start fit of all 5 settings. | this file |
| `encode_joint`, `decode_joint`, `joint_residuals`, `fit_joint_structural` | Fit 4 shared settings plus a v0 per date. | this file |
| `fit_v0` | v0 for one date. | this file |
| `robust_carry` | Rate and dividend from a put-call parity regression that downweights outliers. | this file |
| `prepare_date` | Cleans one trading day and splits calibration from held-out strikes. | this file |
| `metrics`, `predict` | Score and predict. | this file |
| `choose_training_dates`, `make_split_map` | Chronological 70/15/15 split per stock. | this file, research |
| `add_realized_volatility` | Adds 20-day realised vol. | this file |
| `plot_actual_vs_model`, `plot_realized_vs_state`, `make_surface_artifacts` | Figures and surface files. | this file |
| `load_ready_data` | Reads the authentic NSE model-input file. | this file |
| `build_checks` | Integrity checks. | this file |
| `main` | Runs the whole single Heston study. | run by hand |

**scripts/**

| Script · Name | What it does | Used by |
|---|---|---|
| pinn_run_training · `check_gates`, `load_domain`, `save_model`, `load_model`, `main` | Trains rounds "until it works" against fixed accuracy gates. | run by hand |
| pinn_run_calibration · `choose_training_dates`, `run_protocol`, `synthetic_recovery`, `SH_iv`, `main` | Runs the frozen calibration protocol with the PINN engine and with the exact engine. | run by hand |
| pinn_build_report · `build_checks`, `arbitrage_audit`, `engine_attribution`, `main` | Strict checks, arbitrage audit, and a split of error into "network" and "model" parts. | run by hand |
| pinn_write_report · `table`, `load`, `main` | Writes PINN_SINGLE_HESTON_REPORT.md. | run by hand |

**research/** (classical single Heston studies)

| Script · Name | What it does | Used by |
|---|---|---|
| audit_single_heston_overfitting · `iv_from_params`, `baseline_iv`, `score`, `fit_training_halves`, `cluster_bootstrap_difference`, `plot_comparison`, `main` | Three-fold cross-fit, simple baselines (flat, quadratic smile), and stability across training halves. | run by hand |
| forecast_single_heston_next_day · `sha256`, `propagate_variance`, `score`, `load_candidates`, `prepare_target`, `choose_candidates`, `make_grid`, `add_groups`, `bootstrap_daily_rmse_difference`, `plot_outputs`, `main` | Next-session IV-surface forecast: today's v0 rolled forward with the model's mean reversion, chosen on validation only. | run by hand |
| improve_heston_generalization · `design`, `fit_calibrator`, `apply_calibrator`, `rmse`, `daily_rmse`, `select_models`, `score`, `plot_results`, `main` | A guarded correction layer on top of the forecast, accepted only if validation improves by at least 5%. | run by hand |
| compare_single_double_heston · `score`, `parameter_row`, `propagate_state`, `predict_iv`, `cluster_bootstrap`, `plot_comparison`, `plot_parameter_tables`, `plot_stability`, `main` | Single against Double Heston on an identical chronological test. | run by hand |
| audit_heston_handoff · `sha256`, `score`, `make_split_map`, `compare_metric`, `price_bounds`, `main` | Recomputes 60 handoff claims from row-level files. | run by hand |
| build_heston_handoff_report · about 20 Word-formatting helpers (`set_font`, `add_table`, `add_figure`, `fmt`, and so on), `build_markdown`, `main` | Builds the shareable handoff .docx and .pdf. | run by hand |
| build_heston_results_notebook · `markdown`, `code` | Builds the results notebook cell by cell. | run by hand |
| checks/test_*.py · `main` (4 files) | Small script-style numerical checks (pytest finds no tests in them). | run by hand |

### 4.8 experiments/nifty_multifactor_v4 (and how v1 to v3 differ)

v4 is given in full. Every v1–v4 folder also contains artifacts/frozen_sources/, which holds byte-for-byte copies of the source files at freeze time. Those are evidence copies, not separate code.

**literature_exact.py**: the exact pricer for the "four-shock" convention used in this experiment.

| Name | What it does | Used by |
|---|---|---|
| `admissible` | Rejects invalid settings (positivity, correlation in range, Feller). | run, test |
| `rule` | Gauss-Laguerre weights for the price integral. | this file, other scripts |
| `Grid` | A fast pricer for one fixed set of (moneyness, maturity) points. | run, test |
| `adaptive` | An independent slow adaptive-integration price, used as a cross-check. | teacher_quality, report, supplement, test |
| `exact` | The evaluation price: 128-node and 96-node results must agree, otherwise it falls back to `adaptive`. | almost every v4 script; also the btc tests |
| `batch_teacher` | Exact prices for many independent training points. | run |
| `iv`, `black` | Implied vol from price, and the Black price. | run, report, controlled_analysis |
| `fit_bs`, `bs_predict` | Fits Black-Scholes (flat or one vol per maturity knot) and predicts with it. | run, market_tables |
| `decode_sh`, `fit_sh` | Fits all 5 single Heston settings: global search, then local starts. | run |

**run.py**: the frozen experiment driver.

| Name | What it does | Used by |
|---|---|---|
| `sha`, `save`, `read`, `stamp`, `tensor` | Fingerprint a file, write JSON, read JSON, time-stamp, and make a tensor. | all v4 scripts |
| `scaled` | Scales a parameter set's variance level up or down by one factor. | report |
| `freeze` | Writes the scenario bank and hashes all source files before anything runs. | run by hand |
| `inherit` | Copies v3 artifacts that do not depend on the PINN loss, checking each hash. | run by hand |
| `verify` | Refuses to continue if any frozen file has changed. | every v4 script |
| `validate` | Runs the exact-pricer and physics test suites and records pass or fail. | run by hand |
| `require_exact` | Allows the next step only if the exact pricer passed validation. | this file |
| `fetch_market`, `market_panel` | Downloads and cleans NIFTY quotes for the validation or final window. | this file |
| `fit_market_bs`, `market_select` | Fits Black-Scholes on validation and picks the frozen Heston candidates (a 14% level was chosen). | this file |
| `sample` | Latin-hypercube sampling of PINN training points inside the declared ranges. | this file, market_tables, supplement |
| `teacher_data` | Prices training and development points with the exact teacher. | run by hand |
| `model` | Builds the `TorchRegularVariancePINN` of the chosen width (C3: 256 wide, 5 deep). | this file, other scripts |
| `structural` | Shapes the ten settings into the network's two-factor structural input. | other scripts |
| `train` | Trains one candidate PINN for one seed (price, IV, equation and convexity loss, then L-BFGS polish). | run by hand |
| `cases` | Builds the 50 controlled scenarios from the published anchor. | controlled_analysis, report, test |
| `surfaces` | Prices the 61 strikes × 33 maturities controlled grids. | run by hand |
| `baselines` | Fits single Heston and Black-Scholes to each controlled surface's calibration cells. | run by hand |
| `nets`, `candidate_nets` | Loads trained seeds after checking their fingerprints. | other scripts |
| `select_candidate` | Locks the first candidate on the C1 → C2 → C3 ladder that passes all four accuracy gates. | run by hand |
| `neural` | The two-seed average PINN price for given settings and points. | other scripts |
| `metrics` | Price RMSE, MAE, P95, max and IV RMSE. | other scripts |
| `lock`, `evaluate` | Freezes everything, then scores fidelity and the controlled benchmark once. | run by hand |
| `market_final` | Scores the untouched final NIFTY window with fixed parameters. | run by hand |

**Other v4 files**

| File · Name | What it does | Used by |
|---|---|---|
| teacher_quality · `_adaptive_job`, `main` | Re-checks teacher prices against independent integration and reports where fallbacks were needed. | run by hand |
| pinn_development_check · `bucket_masks`, `main` | Checks the four accuracy gates on the development split by maturity and moneyness bucket. | run by hand |
| candidate_development_report · `main` | A descriptive development report for one candidate. | run by hand |
| controlled_analysis · `err_stats`, `grid`, `expected_avg_vol`, `main` | Computes the controlled analyses: regime tables, the fixed-total-variance twist, smirk slope and advantage maps. | run by hand |
| market_tables · `masks`, `score`, `decomposition`, `daily`, `validation_frame`, `main` | Market tables by bucket, and splits the error into "network" and "model/parameter" parts. | run by hand |
| final_freeze · `components`, `create`, `verify_freeze`, `record_opening` | Hashes every input of the final test, then records when it was opened. | run by hand |
| supplement · `freeze`, `audit`, `report` | A pre-declared reporting addendum (error decomposition, state-only refit). | run by hand |
| report · `markdown`, `main` | Writes the report from frozen outputs. | run by hand |
| test_experiment (5 tests) | Checks the shock convention, pricer bounds and parity, factor reduction, the Black-Scholes fit not reading held-out data, and range coverage. | validate |

**Differences in earlier versions**
- **v1, v2.** Same files (literature_exact, run, report, test_experiment), each about 439 lines. v2 changed only the short-maturity reference integration in literature_exact.
- **v3.** Added teacher_quality, pinn_development_check, controlled_analysis, market_tables, final_freeze and supplement, and widened the state ranges.
- **v4.** run.py grew to 533 lines, with the loss rescaled to the gate tolerances, the candidate ladder (`select_candidate`, `candidate_nets`) and `inherit`. It also added candidate_development_report.

### 4.9 experiments/nifty_multiscale_v5 and pinn_architecture_v6 (codex branch)

| File · Name | What it does | Used by |
|---|---|---|
| v5 run · `save`, `freeze`, `verify` | Save JSON; hash the protocol and sources; refuse to run if anything changed. | v5 and v6 scripts |
| v5 run · `model` | Loads a C3 seed and wraps it as FROZEN, SHARED or MULTISCALE. | v5 and v6 |
| v5 run · `train` | Trains only the new expert branches (800 Adam + 60 L-BFGS steps). | run by hand |
| v5 run · `prediction_metrics`, `pde_metrics` | Error by maturity bucket, and the equation residual. | this file |
| v5 run · `select`, `evaluate` | Chooses on development data, then scores once on fresh data. | run by hand |
| v5 market · `panel`, `split_day`, `arrays`, `choose_bs` | Loads NIFTY crisis quotes, keeps anchor and test strikes apart, and picks a flat or term Black-Scholes. | this file |
| v5 market · `baseline` | Fits Black-Scholes and single Heston per day. | run by hand |
| v5 market · `decode`, `predictor`, `fit_pinn`, `pinn` | Fits the PINN's three free numbers (variance scale, slow state, fast state) per day. | run by hand |
| v5 market · `block_interval`, `report`, `verify` | Block-bootstrap confidence intervals and the market summary. | run by hand |
| v5 controlled · `main` | Descriptive maturity table on reused v4 surfaces. | run by hand |
| v5 report · `markdown`, `main` | Writes REPORT.md. | run by hand |
| v6 common · `data`, `tensors`, `losses`, `gradients`, `physics`, `metrics`, `speed` | Shared loaders, the loss, per-loss gradient sizes, physics checks, accuracy and timing. | all v6 files |
| v6 diagnose · `main` | Reproduces the baseline and logs a 200-step diagnostic run. | run by hand |
| v6 train · `initialize`, `verify` | Freezes the inputs and the input-scaling statistics. | run by hand |
| v6 train · `network` | Builds one arm (continue, plain, modified MLP, plus balancing, plus RAD). | evaluate, report |
| v6 train · `stats_args`, `strata`, `sample_indices` | Stratified mini-batches that always include every maturity, moneyness and variance bucket. | this file |
| v6 train · `rad` | Residual-adaptive resampling: replaces 4,096 physics points with ones where the equation error is large. | this file |
| v6 train · `run` | Trains one arm for one seed. | run by hand |
| v6 evaluate · `nets`, `ensemble_metrics`, `passed`, `development`, `select`, `fidelity` | Development table, gate check, selection, then a one-time fresh test. | run by hand |
| v6 curves · `compute`, `plot` | Fixed-grid price curves, derivative checks and 3D surfaces. | run by hand |
| v6 figures · `training_plots`, `metrics_plots`, `diagram` | Training-loss, metric and architecture figures. | run by hand |
| v6 report, deliver, audit · `md`, `regional`, `main`, `diagram` | Assemble the report, the common-IV and regional checks, and integrity audits. | run by hand |
| v6 research/official_*.py | Copies of published third-party code (JaxPI, GradientPathologies, RAD), kept for reference only; not Dhruva's. | none |
| outputs/regular_pinn_recovery/residual_test/scripts · `forgetting_eval` (`truths`, `main`), `forgetting_analysis` (`metrics`, `main`), `final_report` | Calibrates every saved checkpoint on the same 24 truths, then measures "forgetting" and churn per arm. | run by hand |

### 4.10 experiments/btc_multifactor_v1 (shared by all crypto folders)

| File · Name | What it does | Used by |
|---|---|---|
| engine · `admissible` | Validity check, with the Feller condition optional. | this family |
| engine · `rule`, `Grid` | Fast fixed-point integration pricer. | this family |
| engine · `adaptive`, `exact` | Independent check price; then the evaluation price (128 and 96 nodes must agree, else fallback). | this family, dh_pinn_v5, dh_relevance, clear_separation, financial_curve_v2 |
| engine · `black`, `vega`, `iv` | Black price, vega and IV inversion. | this family |
| engine · `box`, `decode` | Search ranges and the map to valid single or double Heston settings. | this family |
| engine · `fit_heston` | Differential evolution plus 12 local least-squares starts, keeping all starts. | run, run_eth/sol/xrp/hype |
| engine · `fit_bs_term`, `bs_predict` | The original Black-Scholes term fit, later found buggy (one vol per quote). | run |
| data · `expiry_of` | Parses a Deribit expiry code into a date. | this family |
| data · `_get`, `fetch` | Downloads every option trade in the time window from Deribit's public archive and caches it. | run, run_eth |
| data · `forward_price`, `solve_k` | Converts Deribit's coin-denominated prices to forward-normalised form. | this file |
| data · `clean` | Filters trades and builds the surface plus the held-out designs A and B. | run, run_eth |
| run · `sha`, `save`, `read`, `stamp` | File helpers. | amend01, other crypto runs |
| run · `freeze`, `verify`, `validate` | Freeze rules, check hashes, run the tests. | run by hand |
| run · `select_dates`, `all_dates` | Picks shock dates by the DVOL rule, plus calm dates. | run by hand |
| run · `fetch` | Builds the surfaces. | run by hand |
| run · `fit_task`, `fit` | Fits all models per date and design in parallel. | run by hand |
| run · `metrics`, `score` | Held-out IV and price errors. | this file |
| run · `select_variant` | Chooses Feller-strict or Feller-free on validation dates only. | run by hand |
| run · `cluster_boot`, `evaluate` | Episode-cluster bootstrap and the primary endpoint test. | run by hand |
| amend01 · `fit_bs_expiry`, `predict_bs_expiry` | The corrected Black-Scholes: one vol per expiry. | amend01, dh_relevance, clear_separation |
| amend01 · `freeze`, `check`, `fit`, `preds`, `scores`, `cells`, `evaluate`, `secondary` | Refit only the Black-Scholes arm and re-score, including mark-price and bucket secondaries. | run by hand |
| plot_btc · `piv` | Pivot helper for the figures. | this file |
| plot_price_curves · `c_model`, `atm_market` | C against S and C against maturity curves for one BTC date. | run by hand |
| test_btc (7 tests), test_amend01 (2 tests) | Engine agrees with nifty v4's pricer, Feller-free prices are finite, factor reduction works, decode stays valid, forward round-trip works, Black-Scholes uses only calibration expiries, designs stay disjoint, and one vol per expiry. | validate |

**Differences in the other coins**
- **eth (run_eth.py).** Same steps, but imports BTC's data.py (also inverse options). `select_dates` applies "BTC date rule verbatim on ETH DVOL".
- **xrp, sol, hype.** New data_xrp/data_sol/data_hype files for linear USDC options: `fetch` with a symbol prefix, `parse` (decimal strikes such as "52d5"), `solve_f` (finds the forward from a USDC price) and `clean`.
- **Shock dates for xrp, sol, hype.** Chosen by `rv_ratio` (realised volatility) because there is no DVOL index.
- **hype.** No shock/calm split: `cells` groups as "all", "shock" and "other", with the week as the statistical unit.
- **Plot files.** plot_eth, plot_sol, plot_xrp and plot_hype draw one summary figure each.

### 4.11 Other experiments on main

| File · Name | What it does | Used by |
|---|---|---|
| cross_asset_summary.py (no functions) | Reads the five coins' primary results and draws one comparison figure plus a CSV. | run by hand |
| portable_dual_pinn_v1/run_pilot · `save`, `sha`, `structural`, `logmap`, `parameters`, `geometry`, `dataset`, `tensors`, `evaluate`, `train`, `main` | Builds a synthetic data set where all ten settings vary, and trains the single-branch and dual-branch PINNs on an equal budget. | run by hand |
| portable_dual_pinn_v1/run_extended · `save`, `stratified_label_indices`, `fit`, `main` | The larger matched-budget run (256 training cases). | run by hand |
| portable_dual_pinn_v1/model_training_snapshot | An exact copy of maturity_dual_pinn.py as trained, before a broadcast fix. | verify_saved_compatibility |
| portable_dual_pinn_v1/audit_btc · `main` | Reproduces the 400 BTC score rows and shows that held-out prices influence the forwards. | run by hand |
| portable_dual_pinn_v1/summarize_pilot, build_report, verify_saved_compatibility · `main` | Pilot figures, the handoff report, and proof that the fix did not change saved predictions. | run by hand |
| hardcoded_v1/run_hardcoded · `sha`, `save`, `stamp` | File helpers. | this file |
| hardcoded_v1/run_hardcoded · `nets` | Loads the two C3 seeds. | this file |
| hardcoded_v1/run_hardcoded · `price` | Price with published settings, optionally scaled by one level factor. | this file |
| hardcoded_v1/run_hardcoded · `freeze`, `verify`, `fetch_spx` | Freeze; download today's SPX chain from CBOE and clean it. | run by hand |
| hardcoded_v1/run_hardcoded · `score`, `best_scale`, `surfaces`, `evaluate` | Score each model with nothing fitted, then with one fitted level scale. | run by hand |
| hardcoded_v1/plot_hardcoded_results (no functions) | Draws the results figure. | run by hand |
| price_curve_figures/plot_hardcoded_curves · `c_pinn`, `c_model` | C against S and C against maturity for fixed published settings plus the C3 PINN. | run by hand |
| price_curve_figures/plot_pinn_architecture · `box`, `arrow` | Draws the C3 network diagram. | run by hand |
| dh_pinn_v5/v5_models · `AdaptiveTanh` | tanh with one learned steepness per layer. | this file |
| dh_pinn_v5/v5_models · `ResidualBody` | Layers that add a small learned tweak to the previous layer. | this file |
| dh_pinn_v5/v5_models · `GatedBody` | Wang–Teng–Perdikaris gated layers; this was the winning A5 design. | this file |
| dh_pinn_v5/v5_models · `PlainBody` | The locked v4 layer stack. | this file |
| dh_pinn_v5/v5_models · `V5Pinn`, `.correction`, `count` | The v4 PINN with a swappable body, and a weight counter. | train, evaluate, track_b |
| dh_pinn_v5/train · `tensor`, `structural`, `load`, `grad_norm`, `train` | Trains one ablation stage on the identical v4 data and budget, with optional gradient balancing and RAD. | run by hand |
| dh_pinn_v5/data · `sample`, `build` | New untouched evaluation sets priced by the exact teacher. | run by hand |
| dh_pinn_v5/evaluate · `locked`, `load_stage`, `predict`, `fidelity`, `pde`, `report` | Loads locked C3 or a v5 stage and scores price, IV, equation error and speed. | several v5 scripts |
| dh_pinn_v5/track_b · `sha`, `scaled`, `prior_c`, `split` | Helpers; `prior_c` gives the "prior" price from BS, SH, DH or the PINN; `split` is a checkerboard SPX split. | this folder, dh_relevance, financial_curve_v2 |
| dh_pinn_v5/track_b · `Head` | A 4,545-weight network that adds a bounded IV correction on top of any prior model. | this file |
| dh_pinn_v5/track_b · `features`, `fit_scale`, `train_head`, `evaluate` | Builds the head's inputs, fits the level scale, trains the head, and scores the hybrid. | run by hand |
| dh_pinn_v5/track_b_analysis · `cells`, `boot`, `rmse`, `run` | Maturity and moneyness buckets, paired statistics and arbitrage counts for the hybrids. | run by hand |
| dh_pinn_v5/diagnostics · `run` | How large and how smooth the market residual is that each prior leaves behind. | run by hand |
| dh_pinn_v5/spx_benchmark · `run` | The fixed-parameter SPX benchmark, re-run with the improved PINN. | run by hand |
| dh_pinn_v5/plots, extra_plots, diagrams, two_graphs, smile_graph | Figures: ablation, old against new, market, Track B, curves, 3D surface, architecture diagrams. | run by hand |
| dh_pinn_v5/multi_surface/scan · `chain`, `surface`, `atm_curve`, `score` | Downloads CBOE chains, builds surfaces, and scores how badly one timescale fits the ATM term structure (market data only). | evaluate_surfaces, select_surface |
| dh_pinn_v5/multi_surface/evaluate_surfaces (no functions), plot_two · `draw` | Applies the frozen rule and scores all models; draws the two requested graphs. | run by hand |
| dh_relevance_selection_v1/select_surface · `polite`, `sha`, `skew_by_expiry`, `components`, `domain_ok`, `main` | Scores 62 symbols on five model-free structure measures and picks the top one (IWM). | run by hand |
| dh_relevance_selection_v1/compare_models · `bucket_rmse`, `matched_capacity`, `strongest`, `_score`, `heldout` | Compares models with fixed parameters (PINN included) and with full calibration on a held-out half. | run by hand |
| dh_relevance_selection_v1/make_selected_figures · `split`, `calibrated`, `price_curve`, `fig_C_vs_S`, `fig_calendar`, `fig_matched` | The IWM figures. | run by hand |
| financial_curve_figures_v2/common · `nets` | Loads the dh_pinn_v5 FINAL network. | figure scripts |
| financial_curve_figures_v2/common · `dh_params`, `baseline` | Scenario settings, and the frozen single Heston and Black-Scholes fits. | figure scripts |
| financial_curve_figures_v2/common · `_c_forward`, `price_S`, `payoff`, `shape_checks`, `in_pinn_domain` | Price as a function of S, plus shape-sanity checks. | figure scripts |
| financial_curve_figures_v2/make_figures · `grid_S`, `curves_S`, `record`, `legend`, `fig_C_vs_S`, `fig_C_vs_S_residual`, `fig_C_vs_tau`, `fig_calendar_time`, `fig_fast_vs_slow`, `_implied_var`, `surfaces` | Controlled-benchmark presentation figures. | run by hand |
| financial_curve_figures_v2/make_market_figures · `run` | Market figures for a selected surface. | run by hand |
| clear_separation_figures_v1/pricers · `price`, `time_value`, `implied_vol`, `atm_iv` | Price, time value and IV for each model on the controlled benchmark. | figure scripts |
| clear_separation_figures_v1/pricers · `bs_pinn`, `bs_pinn_meta`, `bs_pinn_price`, `bs_analytic`, `_n`, `european_lower_bound`, `shape_checks` | Load and check the Black-Scholes PINN against analytic Black-Scholes; shape checks. | figure scripts |
| clear_separation_figures_v1/controlled_figures · `two_panel`, `finish`, `save`, `time_value_panel`, `multipanel`, `smile_panel`, `smile_triple`, `atm_term_structure`, `atm_time_value`, `decay`, `fast_vs_slow` | Controlled time-value, smile, term-structure and decay figures, each with residual panels. | run by hand |
| clear_separation_figures_v1/market_and_validation_figures · `surface`, `fitted`, `model_c`, `fig_market_*`, `fig_dh_pinn_validation`, `fig_bs_pinn_validation` | IWM held-out market figures and the two PINN validation figures. | run by hand |
| clear_separation_figures_v1/atm_iv_corrected · `validate_flat`, `curves`, `fig_term_structure`, `fig_fast_vs_slow`, `final_checks` | Fixes the mislabelled Black-Scholes curve and proves a fixed-vol Black-Scholes IV line is flat to 1e-13. | run by hand |
| clear_separation_figures_v1/vendor_bs_pinn/* | A copy of black_scholes_pinn (see 4.12). | pricers |
| model_reference_v1/make_pdf, make_explainer_pdf · `Page` (`title`, `h`/`sub`/`part`, `p`/`text`, `bullet`, `eq`, `table`, `box`/`note`, `close`) | A tiny page-layout helper that writes the formula sheet and the explainer as PDFs with matplotlib. | run by hand |
| model_reference_v1/make_diagram · `box`, `arrow`, `panel` | Side-by-side diagram of the locked v4 and improved v5 PINNs. | run by hand |

### 4.12 black_scholes_pinn/ (branch black-scholes-with-pinn)

| Name | What it does | Used by |
|---|---|---|
| market · `black_scholes_call` | Analytic Black-Scholes call, used only to make data and to grade. | train, tests |
| market · `generate_synthetic_market` | 180 fake quotes from a true vol of 0.20, which the model never sees. | train |
| market · `load_market_csv`, `market_to_normalized`, `dense_evaluation_grid` | Read quotes, normalise them, and build an out-of-sample grid. | train |
| model · `Domain`, `LossWeights` | The training range and the loss weights (`validate`, `to_dict`). | train |
| model · `BlackScholesPINN` | A smooth network for price plus one learned, bounded volatility. | train |
| model · `.sigma` | The current learned volatility. | train |
| model · `.normalized_inputs` | Scales the inputs. | this file |
| model · `.forward` | Price over strike, computed without the analytic formula. | train |
| model · `.pde_residual` | The Black-Scholes equation error by autodiff. | train |
| model · `.condition_errors` | Expiry-payoff and boundary errors. | train |
| model · `.metadata` | Description of the network. | train |
| train · `TrainConfig`, `set_seed`, `sobol_points`, `tensor_column` | Settings, reproducibility, and evenly spread physics points. | this file |
| train · `compute_losses`, `detached_row`, `train` | The joint loss (equation, quotes, payoff, boundaries) and the training loop. | this file |
| train · `estimate_sigma_from_price_field`, `estimate_sigma_robust_from_price_field`, `set_model_sigma` | Reads the volatility back out of the learned price surface through the equation itself, robustly. | this file |
| train · `evaluate_and_save`, `save_loss_plot`, `save_calibration_plot`, `save_surface_plot`, `parse_args`, `main` | Grade, plot and save. | run by hand |

### 4.13 docs/figures/bs_pinn_curve_validation/code/generate_plots.py

| Name | What it does | Used by |
|---|---|---|
| `digest`, `dump`, `read`, `table` | File helpers. | this file |
| `savefig`, `axis`, `lines`, `one_curve`, `error_curve`, `surface_axis` | Plotting helpers for the 21 curve figures. | this file |
| `shape_checks` | Counts places where price falls when S rises, or bends the wrong way. | this file |
| `main` | Loads frozen C3 weights and draws Black-Scholes, PINN and exact-DH curves and errors. | run by hand |

### 4.14 Overnight audits (archive branches, 22 Aug)

| File · Name | What it does | Used by |
|---|---|---|
| node_b_toolkit · `_gauss_laguerre`, `price_surface_fast`, `characteristic_full_scalar` | A fast vectorised copy of the production pricer for diagnostics. | Node B scripts |
| node_b_toolkit · `Geometry`, `full_108_geometry`, `central5_market_geometry` | The quote layouts studied: the full 108-quote grid and the 5-quote market grid. | Node B scripts |
| node_b_toolkit · `normalized_observables_fast` | Prices divided by spot for a layout. | Node B scripts |
| node_b_toolkit · `parameter_widths`, `scaled_coordinates`, `scaled_parameter_jacobian_fast`, `jacobian_summary` | Sensitivity matrix and its condition number and rank. | Node B scripts |
| node_b_toolkit · `representative_cases`, `case_vector`, `log_experiment`, `multiplicative_noise` | Test cases, logging and noise. | Node B scripts |
| run_node_b_validate_fast_pricer · `main` | Checks the fast pricer against production. | run by hand |
| run_node_b_phase_a_baseline · `main` | Reproduces the classical recovery baseline. | run by hand |
| run_node_b_phase_b_multistart · `case_seed`, `deterministic_starts`, `recover_one_case`, `latent_factor_table`, `main` | Multi-start recovery spread, with and without noise. | run by hand |
| run_node_b_phase_d_jacobian · `build_geometries`, `analyse_jacobian`, `factor_swap_check`, `main` | Conditioning across layouts, and proof that swapping factors leaves prices identical. | run by hand |
| run_node_b_phase_e_profiles · `objective_against_clean`, `valley_scan`, `compensated_profiles`, `main` | Maps the flat "valleys" of the fit landscape. | run by hand |
| run_node_b_phase_h_improvements, run_node_b_noise_propagation · `evaluate_arm`, `main` | Tests small calibration fixes, and predicts the noise collapse from the sensitivity matrix. | run by hand |
| Node C probe_residuals · `canonical_pde_terms`, `canonical_to_archive2`, `archive2_residual`, `archive2_derivative_instrumentation`, `archive2_effective_residual`, `bs_call`, and others | Assembles the true pricing equation, and shows that Archive-2's version silently drops all variance terms. | evidence |
| Node C probe_extensions · `closure_experiment`, `correctly_wired_cos_residual`, `defect_invariance`, `bs_limit_error`, and others | Shows bit-for-bit that the defect matches a Black-Scholes-only equation, and quantifies the gap. | evidence |
| Node C probe_variance_propagation_gap | Measures error from rolling variance forward deterministically (up to −14.5% at 90 days, per the commit message). | evidence |
| Node C run_node_c_tests, run_repo_engine_tests · `_Approx`, `_Mark`, `_PytestShim`, `load_patched_src_module` | A small pytest stand-in so tests ran under an older Python. | evidence |

### 4.15 Archive-2 neural calibrator (commit 642702e, 22 Aug)

| File · Name | What it does | Used by |
|---|---|---|
| src/dheston/pricing/heston · about 25 functions (`*_characteristic_function_*`, `_cos_*`, `_heston_factor_cumulants_*`, `price_*_heston_numpy/torch`) | Single and Double Heston pricing by the COS method, in NumPy and PyTorch. | the Archive-2 package |
| src/dheston/models/losses · `predict_surface_prices`, `parameter_supervision_loss`, `price_reconstruction_loss`, `ordering_penalty`, `boundary_penalty`, `_safe_grad`, `pde_residual_loss`, `build_loss_components` | The training losses; `pde_residual_loss` is the one Node C proved has zero variance derivatives. | train_double_heston |
| src/dheston/models/networks · `DeepSurfaceInverseModel` | Network from surface features to settings. | train_double_heston |
| src/dheston/data/surfaces · `SurfaceTemplate`, `SurfaceRecord`, `read_option_rows`, `build_surface_records`, `verify_zero_leakage`, `split_records_chronologically`, `cap_records`, `SurfaceDataset`, `pad_surface_batch` | Builds and splits real surfaces by date, checking there is no overlap. | train_double_heston |
| src/dheston/data/synthetic · `build_synthetic_records` | Noisy synthetic surfaces. | train_double_heston |
| src/dheston/calibration · `calibrate_surface_multistart`, transforms (`sample_parameter_vector`, `constrain_parameter_tensor`, and others) | Classical fit and parameter maps. | Archive-2 |
| src/dheston/config, evaluation/metrics | Config merging; MAE and RMSE summaries. | Archive-2 |
| models/pinn_model · `DoubleHestonConstraintMap`, `PhysicsInformedInverseCalibrator`, `_FeedForwardBlock`, `_positive`, `_sigma_with_feller_margin`, `_correlation_disk_map`, `_activation_factory` | Surface → ten settings, with outputs forced into the valid region. | src/train_pinn |
| src/torch_double_heston · `price_double_heston_*_tensor`, characteristic-function and quadrature helpers | Differentiable PyTorch copy of the canonical pricer. | src/train_pinn |
| src/train_pinn · `train_pinn`, `predict_parameters`, `_evaluate_validation`, `_prepare_batch`, `_masked_mse`, `main` | Trains with parameter supervision plus repricing. | run_pinn_* |
| src/run_pinn_synthetic_baseline, run_pinn_two_stage_baseline, run_pinn_improved_benchmark | Synthetic baselines and a curriculum two-stage benchmark. | run by hand |
| train_double_heston · about 23 functions (`train_stage`, `evaluate_loader`, `build_dataloaders`, checkpoint and resume helpers, `main`) | Staged, resumable training driver for the Archive-2 model. | run by hand |

### 4.16 tests/ (one line per file)

On main, written by Dhruva:
- **conftest.py**: puts the single Heston sources on the import path so the test suite can be collected.
- **test_unified_calibrator.py** (22 tests): the unified calibrator works with any number or order of quotes, single or missing expiries, low and high volatility; decode is always valid; the exact pricer matches NumPy; gradients match finite differences; refinement is stable.
- **test_projection_finetune.py** (8): the fine-tuning losses behave correctly, and bad or padded quotes cannot corrupt training.
- **test_refinement_objective.py** (6): refinement uses quote noise correctly, only accepts improving steps, freezes a surface with one bad quote, and never reads held-out quotes.
- **test_precise_calibration.py** (1): exact polish lowers quote error without ever seeing the truth.
- **test_real_corpus_integrity.py** (6): held-out NIFTY quotes cannot leak into carry, weights or noise; corpus packing keeps masks and drops duplicate contracts.
- **test_repair_selection.py** (1): held-out prices cannot reach the encoder through validation.
- **test_tail_precision.py** (2): high-precision inversion agrees on normal prices and resolves tiny tails.
- **test_torch_pricer.py** (1): the differentiable pricer does not turn into NaN on a tricky tiny number.
- **test_regular_pinn.py** (5): the regular PINN's equation, derivatives and payoff are correct (MLX).
- **test_regular_pinn_torch_physics.py** (4): the PyTorch copy matches the raw equation, finite differences and the MLX version.
- **test_regular_pinn_calibration.py** (8): calibration uses only calibration quotes and the frozen network, keeps failed starts, and scores failures as failures.
- **test_regular_pinn_data.py** (2): IV sensitivity labels and the float64 copy are correct.
- **test_regular_pinn_lbfgs.py** (8): the L-BFGS fine-tune is float64, deterministic, lowers the loss and refuses existing folders.
- **test_regular_pinn_physics_diagnostic.py** (1): the physics diagnostic does not depend on chunk size and keeps failures.
- **test_regular_pinn_surfaces.py** (1): tolerance coordinates and sensitivity targets invert correctly.
- **test_deep_regular_pinn.py** (4): added layers start as "no change", train correctly, and calibration stays neural-only.
- **test_identifiability_v2.py** (3): factor ordering, exact swap symmetry and the information scaling are correct.
- **test_recovery_prior.py** (1): the ridge-regularised fit's objective and holdout isolation are correct.
- **test_surrogate_error.py** (2): GLS whitening is correct, neural-only and holdout-blind.
- **test_pinn_single_heston.py** (12): the single Heston PINN's reference pricer, smile direction, Black inversion, equation and derivatives are correct.
- **test_literature_fixed_truth_pilot.py** (3): pilot training cannot see the truth, held-out data or the exact pricer.
- **test_literature_parameter_selection.py** (2): good prices or a lucky seed cannot approve a failed recovery.
- **test_nifty_fixed_crisis.py** (5): the single match, put/call normalisation, frozen settings, equation and published price check are correct.
- **test_crisis_option_coverage.py** (2): held-out quotes stay out of carry estimation, and impossible discounts are rejected.
- **test_maturity_dual_pinn.py** (4): the dual-maturity PINN's physics, adapter, gate derivatives and broadcast fix are correct.
- **test_node_c_pde_physics_audit.py** (27, Node C): the canonical pricer satisfies its equation, while Archive-2's variance derivatives are exactly zero.

Only on the Double/Single-heston branch: test_affine_factor_assessment, _calibration, _lbfgs, _pinn, _pricing, _report, test_composition_comparison, test_conjugate_factor_pinn, test_convolution_pinn, test_factor_price_training, test_integrated_factor_pinn, test_moment_factor_pinn, test_two_moment_factor_pinn, test_regular_pinn_assessment_filters, test_regular_pinn_lbfgs_jacobian, _lbfgs_report, _lbfgs_surfaces and test_regular_pinn_torch_training. These are 1 to 6 tests each. They check that the factor and composition PINNs match their reference equations, keep gradients and never read held-out quotes.

Only on the codex branch: test_multiscale_pinn (6: the extension starts as "no change", the gate is smooth, and prices match the raw equation) and test_research_modified_pinn (6: the published modified MLP is reproduced faithfully and the gates reject failures). test_reproducible_regular_pinn_data was re-committed there but is Viraj's.

From Archive-2 (642702e): test_archive2_pricing_smoke, test_pinn_forward, test_pinn_training and test_torch_double_heston. These are basic smoke checks for put-call parity, output shape and constraints, one training step, and torch matching NumPy.

Inside experiments/: test_experiment.py (nifty v1–v4), test_btc, test_amend01, test_hype/sol/xrp, test_extended (portable dual), and single_heston_pinn/research/checks (script-style, not pytest).

---

## 5. Glossary

- **Option, call, strike, expiry/maturity**: a contract to buy the asset (a call) at a fixed price (the strike) on a fixed date (the expiry). Maturity is the time left until then.
- **Implied volatility (IV), "vol points"**: the volatility you would have to plug into Black-Scholes to reproduce a market price. One vol point is 1 percentage point of volatility, and most errors in this report are measured this way.
- **Volatility smile/smirk, term structure**: how IV changes across strikes (the smile) and across expiries (the term structure).
- **Black-Scholes (BS)**: the classic formula that assumes volatility never changes. It has one number, so it cannot produce a smile.
- **Single Heston (SH)**: volatility itself moves randomly and drifts back toward a long-run level at one speed. It has 5 settings (κ speed, θ long-run level, σ vol-of-vol, ρ correlation, v0 today's variance).
- **Double Heston (DH)**: two independent random-variance factors, one fast and one slow, that add together. It has 10 settings.
- **Calibration**: choosing a model's settings so its prices match today's market prices.
- **Inverse calibration / parameter recovery**: reading the true settings back out of prices that were generated with known settings. It is the test of whether the settings are "identifiable".
- **Identifiability**: whether prices pin down the settings uniquely. Here they usually do not, because many setting combinations give almost the same prices.
- **PINN (physics-informed neural network)**: a neural network trained to match example answers and also to obey the model's pricing equation at thousands of random "collocation" points.
- **Forward / surrogate PINN**: a PINN that takes settings and returns prices very fast, as a stand-in for the slow exact formula. The C3 network is one.
- **PDE residual**: how badly a network's prices break the pricing equation at a point. Zero means it obeys it exactly.
- **Teacher**: the exact (slow) pricer whose prices the network learns from.
- **Collocation points**: random points where the equation is enforced in training. These are not real quotes.
- **Multifactor**: a model with more than one variance factor, here two (fast and slow). The "nifty/btc_multifactor" folders test whether the second factor helps.
- **Multiscale**: the v5 idea of giving the network separate small "experts" for short, medium and long maturities.
- **Fine-tuning**: continuing to train an already-trained network on a new goal or new data.
- **Self-supervised projection fine-tuning**: retraining the calibrator to make its prices fit real market quotes directly. No "true settings" labels are needed, because the market quotes themselves are the target.
- **Projection vs recovery**: on real data there is no true setting to recover, only the best-fitting setting (the "projection"). The network had been trained on recovery, which is the wrong problem for real data.
- **Unified calibrator / set transformer**: Dhruva's network that reads any number of quotes in any order, outputs ten settings with an uncertainty, then polishes them with a few exact-pricer steps.
- **Affine factor (PINN)**: Heston models are "affine", meaning their characteristic function has a known structure built from two coefficient functions. The affine-factor PINN tried to learn those coefficients instead of prices.
- **Dual PINN**: a pair of networks, one reading short expiries and one reading long expiries, each outputting part of the ten settings. "Portable dual PINN" was a later pricing version with short and long experts.
- **Ridge (penalty)**: an extra cost that pulls a fit toward a reference answer (the network's guess, or the middle of the range). It trades a little fit for much more stability.
- **GLS (generalised least squares)**: fitting that weights errors by how noisy they are known to be.
- **OOD (out of distribution)**: an input unlike anything seen in training. The ood.py alarm says when the model is extrapolating.
- **Literature parameter sweep**: trying several "true" setting sets based on published papers, to see whether any of them can be recovered.
- **Fixed / hard-coded parameters**: using settings published in a paper with no fitting to today's market.
- **Held-out quotes**: quotes hidden from fitting and used only for scoring, so the score is honest.
- **Frozen / predeclared / locked**: rules and code fixed and fingerprinted (hashed) before results are seen, so nobody can tune toward a desired answer.
- **Wilcoxon test, bootstrap**: standard ways to check whether a difference between two models is real or luck.
- **Feller condition**: a rule (2κθ > σ²) that keeps the variance from touching zero.
- **Gated / modified MLP, RAD, gradient balancing**: published PINN-training tricks. They are, respectively: a layer that mixes two encodings; resampling physics points where the error is largest; and re-weighting loss parts so none dominates.

---

## 6. What the user's Double Heston 2.0 and the live website use from this

### 6.1 Double Heston 2.0 (folder wild/.shots/dh2/double_heston_2.0)

How this was checked: a grep for every Dhruva folder and module name, and a read of every import line in the dh2/ package and the top-level scripts.

- **No code is imported.**
  - The dh2/ package imports only its own modules, NumPy, SciPy and pandas.
  - The one external file it loads is legacy_streamlit_site/models.py (through `importlib`, in dh2/models.py). That is the user's own archived Streamlit pricer, not Dhruva's.
  - Nothing from src/mentor_dh_pinn, single_heston_pinn or experiments/ is imported anywhere.
- **Three of Dhruva's result files are read,** all by update_comparison.py, the script that writes COMPARISON_1.0_vs_2.0.md:
  - outputs/real_markets/nifty_comparison/nifty_comparison_summary.json (from the 4 Sep dual PINN study). Row 5 of COMPARISON_1.0_vs_2.0.md prints it as "Black-Scholes 3.04 pts, single Heston 2.36, Double Heston 2.37, PINN + ridge 2.29". The "PINN + ridge" there is Dhruva's dual PINN plus ridge polish, not a PDE PINN.
  - outputs/real_markets/nifty_selection.json. It supplies the same 10 April-2026 NIFTY dates so that 2.0 is compared on "the old test's own 10 high-volatility days".
  - outputs/unified_v6/finetune_real_summary.json ("NIFTY" block). It is loaded into the comparison JSON as `nifty_ft`, but it is not printed in the markdown table.
- **read_v1.py is a diagnostic printer.** It also opens finetune_real_summary.json. It looks for nifty_selection.json under outputs/real_markets/nifty_comparison/, which is the wrong path (the file is one folder up), so it prints "MISSING" for that one.
- **Not Dhruva's.** The other files 2.0 reads are the user's or Viraj's: outputs/ambiguity/*, outputs/consolidated_results.json, outputs/g8_fixed/* and outputs/option_backtest/option_backtest_pairs.csv (committed by mangalvinayak13-vin on 30 Sep).

### 6.2 The live website (`git show night-build:website/tools/content.py`)

**One of Dhruva's files is read at build time.** That file is outputs/unified_v6/finetune_real_summary.json, NIFTY block, and the code is at content.py lines 196–213. It produces these displayed values (with the values in the file):
- "median error of 2.48 volatility points" = `ft_3` 0.024780800 × 100. This is the projection-fine-tuned unified calibrator with 3 exact-pricer refinement steps.
- "against 2.98 for Black–Scholes" = `bs` 0.029824952.
- "2.52 for a classical Double Heston fit" = `dh_cold` 0.025231206.
- "in 1.2 s instead of 25 s" = `median_seconds` 1.1622567 and 25.25797.
- "best model on 4 of the 10 days" = `dates_best.ft_3` 4.
- `sh` 0.02625 is loaded but is not in the displayed sentence.

Where these appear on the site:
- the PINN_MARKET paragraph;
- the RESULTS row "PINN calibrator on NIFTY";
- the key-number tile "2.48 vs 2.98";
- the "pinn" explainer and its glossary card.

**Some numbers are typed in as text, not read from a file.** They come from experiments/nifty_multifactor_v4/FINAL_REPORT.md (the locked C3 forward PINN). Line 196 credits them to "experiments/nifty_multifactor_v4 and outputs/unified_v6".

| Text on the site | Value in FINAL_REPORT.md |
|---|---|
| "reproduces the exact pricer to 1.1 × 10⁻⁵ (about 0.1 volatility points) on 8,192 points it never saw" (PINN_TEXT; RESULTS "PINN fidelity"; DONE "Trained PINNs") | Table A, two-seed mean price RMSE 1.10e-5, IV 0.103, on 8,192 points |
| "the pricing equation at 18,000 points" and "PINN training 100,000 + 18,000" | "100,000 train ... and 18,000 collocation points" |
| "On 40 controlled test surfaces it beat a refitted one-factor Heston, which beat Black–Scholes, every time" (PINN_TEXT; RESULTS "PINN, controlled test 40 of 40") | "DH-PINN < SH < BS ... on 40 of 40" |

**Also from Dhruva's work:**
- The data row "NIFTY, high volatility: 10 days, 1,750 quotes ... chosen by a rule fixed before any fit". This is his date selection (docs/mentor_dh_pinn/NIFTY_THREE_MODEL_COMPARISON.md) and quote count (finetune_real_summary.json `holdout_quotes` 1750).

**Not Dhruva's:**
- outputs/option_backtest/option_backtest_pairs.csv, which is the user's.
- website/assets/data/pinn_vs_ann*.json. These come from website/tools/pinn_vs_ann.py, the user's own 3 × 64 demonstration networks built on legacy_streamlit_site/models.py (commits c7ecda4 and 1acc5e8 by mangalvinayak13-vin).
- All ANN, ambiguity and backtest results.

### 6.3 Precision notes for the site

The user's rule is verified, precisely worded content. These points come from Dhruva's own files.

1. **"PINN" is used for two different networks.**
   - The 1.1e-5 / 40-of-40 network is the C3 forward pricer. It was trained on the pricing equation, but with fixed published settings.
   - The 2.48-point NIFTY number comes from the unified set-transformer calibrator. Dhruva's own audit says this calibrator is "NOT PDE-informed". Its "physics" is an exact differentiable pricer inside the model (node-C/FINAL_REPORT.md; docs/CALIBRATION_REPAIR_HANDOFF.md, "This iteration does not train the old four-coordinate forward PDE PINN").
   - PINN_TEXT then says "Our PINNs ... are also trained on the model's own pricing equation" next to the NIFTY number. That is accurate for C3 but not for the calibrator that produced 2.48. A safer label is "physics-based calibrator (exact pricer in the loop)".
2. **The same files contain results the site does not show.**
   - ADANIPOWER: fine-tuned 0.0182 against classical 0.0040 (finetune_real_summary.json).
   - 140 single-expiry stock surfaces: no effect, about 2.6× behind classical (panel_test_summary.json).
   - Untouched NIFTY with the C3 PINN at fixed settings: 107.7 against Black-Scholes 22.4 index points (nifty_multifactor_v4/FINAL_REPORT.md).
3. **The "40 of 40" surfaces are synthetic surfaces generated by Double Heston itself.** The report warns: "The synthetic truth is Double Heston by design ... not market superiority".
4. **"Best on 4 of 10 days" is correct.** The research brief adds that the fine-tuned model beat classical Double Heston on 7 of 10 dates with p = 0.064, which is not significant at the usual 5% level (RESEARCH_BRIEF_REAL_DATA_GAP.md §5.1).

---

## 7. Limits of this dissection

- **Where descriptions come from.** Function descriptions were written from each function's docstring and its first lines of code, read through a small read-only AST script. Every listed file was opened that way, but not every function body was read line by line. Very long scripts, such as the 600–900-line report builders, are described at the level of their helper groups.
- **Which version was read.**
  - For files on main, the main version was read.
  - Branch-only files were read on their branch: Double/Single-heston, codex/improved-double-heston-pinn, black-scholes-with-pinn and the node-B/node-C archives.
  - Archive-2 was read at commit 642702e.
- **What was not read.** Binary weights (.pt, .safetensors), .npz data, PDFs and images were not opened. The outputs/ folders are described from their REPORT.md and summary JSON files only.
- **Other people's work in the same folders.** Viraj wrote parts of src/mentor_dh_pinn and scripts/mentor_dh_pinn (reproducible_regular_pinn_data, finetune_regular_pinn_lbfgs, prepare_regular_pinn_v2, evaluate_regular_pinn_development, precheck_double_v2_mac, and later torch_pricer edits). These are marked where they appear.

