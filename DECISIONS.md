# DECISIONS

Every non-obvious choice with its reason. Newest last.

1. **Keep development 2016-2021 / test 2022-2026** instead of 70/30. It was fixed before any result existed and is more conservative on length of history; changing it now would be a tuning choice made after seeing results.
2. **Jump extension stays as the headline model.** The brief lists jumps as a stretch; the finished work shows jumps are what fix the fit (same-day error ~1.7 -> ~0.6 vol points, 2025-26). The brief's diffusion Double Heston is reported alongside as "DH2.0 diffusion".
3. **PINN targets the diffusion model.** The brief's PDE has no jump integral term. Jumps are priced by the exact Fourier pricer; a PINN with an integro-differential term is out of scope.
4. **No-arbitrage quote filters are diagnostics.** Removing violators changes every surface and would invalidate the finished walk-forward runs (~22 h of compute). The violation share is reported per year and a sensitivity run on 2025-26 shows the effect.
5. **New results go to `results/`; finished results stay where they are.** Avoids breaking `update_comparison.py` and the committed notes.
6. **DH1 baseline = 1.0's calibration style on the same clean data** (independent daily multi-start fit with the 1.0 pricer), measured on a stratified sample of test days (every 10th trading day) because a full independent refit of 1,173 days is ~3 hours; the same sampled days are used for every model in the attribution so comparisons are paired.
7. **Agents do not commit.** One reviewer (the main session) checks each output against its files, then commits per phase.
