**The forecast variants of hybrids.py on the test period (mean next-day error, vol points); the jump model is the headline model**

| Model | Forecast | Dev | Test | Compared with | Test minus that forecast | 95% interval of the difference | Days better |
|---|---|---|---|---|---|---|---|
| jump model | yesterday's smile (no model) | 1.876 | 1.468 |  |  |  |  |
| jump model | model, today's state | 1.734 | 1.201 |  |  |  |  |
| jump model | model + its one-day mean reversion | 1.738 | 1.199 | model, today's state | -0.002 | [-0.006, +0.001] | 51% |
| jump model | model + mean reversion + leverage update | 1.803 | 1.291 | model, today's state | +0.090 | [+0.037, +0.145] | 40% |
| jump model | model + yesterday's misfit | 1.317 | 1.034 | model, today's state | -0.167 | [-0.226, -0.117] | 73% |
| jump model | model + leverage update + yesterday's misfit | 1.382 | 1.136 | model + yesterday's misfit | +0.102 | [+0.043, +0.158] | 39% |
| jump model | yesterday's smile + model-implied change | 1.928 | 1.552 | yesterday's smile (no model) | +0.084 | [+0.032, +0.132] | 38% |
| diffusion Double Heston | model, today's state | 3.409 | 2.792 |  |  |  |  |
| diffusion Double Heston | model + its one-day mean reversion | 3.402 | 2.786 | model, today's state | -0.006 | [-0.011, -0.001] | 62% |
| diffusion Double Heston | model + mean reversion + leverage update | 3.660 | 2.965 | model, today's state | +0.173 | [+0.114, +0.234] | 39% |
| diffusion Double Heston | model + yesterday's misfit | 1.736 | 1.377 | model, today's state | -1.416 | [-1.534, -1.298] | 99% |
| diffusion Double Heston | model + leverage update + yesterday's misfit | 2.086 | 1.679 | model + yesterday's misfit | +0.302 | [+0.224, +0.380] | 30% |
| diffusion Double Heston | yesterday's smile + model-implied change | 2.216 | 1.758 | yesterday's smile (no model) | +0.290 | [+0.214, +0.368] | 31% |
| single Heston | model, today's state | 4.028 | 3.532 |  |  |  |  |
| single Heston | model + its one-day mean reversion | 4.013 | 3.515 | model, today's state | -0.016 | [-0.022, -0.010] | 77% |
| single Heston | model + mean reversion + leverage update | 4.024 | 3.493 | model, today's state | -0.038 | [-0.064, -0.012] | 55% |
| single Heston | model + yesterday's misfit | 1.776 | 1.416 | model, today's state | -2.115 | [-2.285, -1.962] | 99% |
| single Heston | model + leverage update + yesterday's misfit | 1.774 | 1.382 | model + yesterday's misfit | -0.034 | [-0.072, +0.003] | 48% |
| single Heston | yesterday's smile + model-implied change | 1.877 | 1.435 | yesterday's smile (no model) | -0.032 | [-0.068, +0.005] | 49% |

*Negative difference = the forecast in the row is better than the one it is compared with. Source: results/tables/forecast_variants.csv (exp_variants_table.py); the full table with the difference to the time-scaled smile is results/tables/forecast_variants.md.*
