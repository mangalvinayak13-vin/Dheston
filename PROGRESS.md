# PROGRESS

Updated 2026-10-08.

## Done (before the upgrade brief)
- Ten-year NIFTY 50 pipeline, DH2.0 walk-forward (diffusion and jump variants), pattern memory, trading test, ablations, comparison note, 12 tests. Tagged `dh2.0-pre-upgrade`.
- Headline (test 2022-2026, next-day IV error in vol points): model-free best benchmark 1.236; jump model + yesterday's misfit 1.034 (-0.202 [-0.227, -0.181]); diffusion Double Heston + misfit 1.377; single Heston + misfit 1.416.

## Current
- Phase 0 done: AUDIT.md, PLAN.md, DECISIONS.md written; branch `dh2-upgrade`.
- Work packages A (COS pricer), B (PINN), C1 (data quality, VIX, SVI, stats), C2 (attribution, cross-section, identifiability) launching.

## Next
- D (PINN-accelerated calibration) after A and B; E (one command, figures, docs, UI adapter, verification).

## Open issues
- Pricer's original author unknown (README placeholder).
- Whether the trading edge survives a one-day delay is not established.
