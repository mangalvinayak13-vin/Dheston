**PINN surrogate implied-vol error against the exact pricer (vol points)**

| Network | Real held-out mean | median | 95th pct | max | Box-set mean | PDE residual median (real) |
|---|---|---|---|---|---|---|
| data-only (shipped for calibration) | 0.049 | 0.039 | 0.130 | 0.444 | 0.082 | 0.733 |
| hybrid data + PDE (shipped for Greeks) | 0.077 | 0.064 | 0.196 | 0.651 | 0.141 | 0.362 |
| data-only, spec gate sqrt(tau) | 0.172 | 0.129 | 0.478 | 2.163 | 0.324 | 1.801 |
| physics-only (no labels) | 1.307 | 0.969 | 3.207 | 23.786 | 4.511 | 0.260 |

*Real held-out = every quote of the held-out development days (the last fifth of 2016-2021, never in the training sampler); box set = unseen parameter sets drawn from the training box. Seed averages for the 3-seed rows.*
