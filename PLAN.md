# PLAN: the upgrade brief integrated with the finished DH2.0

Branch `dh2-upgrade` (from tag `dh2.0-pre-upgrade`). Only what AUDIT.md marks missing is built; finished work is reused as is.
Rules from the brief hold throughout: real data only for market results, no look-ahead, design on 2016-2021 and report on 2022-2026,
failures reported, DH1 and the pre-upgrade DH2.0 kept runnable, `PROGRESS.md` and `DECISIONS.md` kept current, commit after each phase.

## Work packages (file-disjoint, so they can run in parallel)

| # | Brief phase | Deliverable | New files |
|---|---|---|---|
| A | P2 exact pricer | COS pricer (cumulant truncation), Lewis cross-check, Monte Carlo (QE) check, parity/convexity/limit tests, timing of one chain | `dh2/cos.py`, `dh2/mc.py`, `tests/test_pricer_cross.py`, `results/pricer_validation/` |
| B | P4 PINN | normalised-PDE hybrid PINN in torch, validation vs exact pricer, ablation (data-only / physics-only / hybrid), data-efficiency, speed, Greeks, arbitrage checks | `dh2/pinn/`, `configs/pinn.yaml`, `results/pinn/` |
| C1 | P1/P6 gaps (light) | data-quality report (quotes per filter, no-arbitrage violation share), India VIX download, model-implied 30-day vol vs VIX, per-expiry SVI reference, Wilcoxon + Diebold-Mariano | `dh2/dq.py`, `dh2/svi.py`, `dh2/stats.py`, `fetch_vix.py`, `results/dq/`, `results/vix/` |
| C2 | P3/P6 gaps (CPU heavy) | DH1 -> DH2.0 step-by-step attribution on a test-period sample, cross-sectional out-of-sample tests, Jacobian condition numbers / standard errors, per-bucket in-sample IV RMSE for all models | `exp_attribution.py`, `exp_crosssection.py`, `exp_identifiability.py`, `results/attribution/` etc. |
| D | P5 | PINN-accelerated calibration vs exact-pricer calibration (after A and B) | `dh2/calib_pinn.py`, `results/pinn_calib/` |
| E | P7/P8 | `dh2.run_all` + `--quick`, figures (300 dpi), tables (CSV/MD/LaTeX), `results/metrics.json`, `RESULTS.md`, `docs/EXPLAINER.md`, `PRESENTATION_NOTES.md`, UI adapter (`today.json`), fresh-venv verification and independent re-derivation of every headline number | (last) |

## Order and resources

1. A, B, C1 start now (light CPU; B trains on the Apple GPU with <= 2 CPU threads). C2 starts at the same time with <= 4 workers.
2. D starts when A and B report. E starts when D reports; E also pulls in all results already produced (walk-forward, ablations, trading test).
3. Heavy CPU jobs never overlap beyond 4 workers in total (the machine has 4 fast cores).
4. I review every agent's output against its files before anything is committed; agents do not commit.

## Decisions taken at the start (details in DECISIONS.md)

- The model-of-record stays the one already validated: diffusion Double Heston (the brief's spec) as DH2.0, and the jump extension `dhj` as the stretch result, which is the strongest one. The PINN targets the diffusion model (the brief's PDE); jumps are priced by the exact Fourier pricer.
- Development 2016-2021 / test 2022-2026 split is kept (fixed before any result was seen).
- No-arbitrage filters are diagnostics, not filters, to avoid invalidating ~22 hours of finished runs; a sensitivity run quantifies the effect.
- New results go to `results/` as the brief asks; the finished DH2.0 outputs stay in `outputs/` and `data/results/`.
