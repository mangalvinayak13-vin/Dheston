**No-arbitrage sensitivity: next-day error (vol points) scored without the target day's violating quotes, test period**

| Scored quotes | Share excluded (1 tick) | Time-scaled smile | Jump model + misfit | Difference (1 tick) | Share excluded (1 tick + 0.25 pt) | Difference (1 tick + 0.25 pt) |
|---|---|---|---|---|---|---|
| all quotes (as published) | 0.0% | 1.236 | 1.034 | -0.202 | 0.0% | -0.202 |
| exclude monotonicity violators | 15.7% | 1.132 | 0.967 | -0.166 | 4.3% | -0.208 |
| exclude butterfly violators | 66.2% | 1.262 | 1.038 | -0.224 | 9.5% | -0.215 |
| exclude calendar violators | 1.9% | 1.208 | 1.000 | -0.208 | 1.2% | -0.207 |
| exclude any violator | 67.6% | 1.210 | 1.021 | -0.189 | 10.7% | -0.212 |

*Predictions and calibrations are unchanged; only the scored quotes differ. Difference = jump model minus benchmark.*
