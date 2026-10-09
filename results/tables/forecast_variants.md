**Forecast variants of hybrids.py on the test period (mean next-day implied-volatility error, vol points, lower is better)**

| Model | Forecast | Dev | Test | Compared with | Test minus that forecast [95% interval] | Days better | Test minus time-scaled smile [95% interval] |
|---|---|---|---|---|---|---|---|
| jump model | yesterday's smile (no model) | 1.876 | 1.468 |  |   |  | +0.232 [+0.210, +0.255] |
| jump model | model, today's state | 1.734 | 1.201 |  |   |  | -0.035 [-0.095, +0.028] |
| jump model | model + its one-day mean reversion | 1.738 | 1.199 | model, today's state | -0.002 [-0.006, +0.001] | 51% | -0.037 [-0.098, +0.027] |
| jump model | model + mean reversion + leverage update | 1.803 | 1.291 | model, today's state | +0.090 [+0.037, +0.145] | 40% | +0.055 [-0.018, +0.133] |
| jump model | model + yesterday's misfit (headline forecast) | 1.317 | 1.034 | model, today's state | -0.167 [-0.226, -0.117] | 73% | -0.202 [-0.227, -0.181] |
| jump model | model + leverage update + yesterday's misfit | 1.382 | 1.136 | model + yesterday's misfit (headline forecast) | +0.102 [+0.043, +0.158] | 39% | -0.100 [-0.162, -0.040] |
| jump model | yesterday's smile + model-implied change | 1.928 | 1.552 | yesterday's smile (no model) | +0.084 [+0.032, +0.132] | 38% | +0.316 [+0.258, +0.371] |
| diffusion Double Heston | yesterday's smile (no model) | 1.876 | 1.468 |  |   |  | +0.232 [+0.210, +0.255] |
| diffusion Double Heston | model, today's state | 3.409 | 2.792 |  |   |  | +1.557 [+1.419, +1.694] |
| diffusion Double Heston | model + its one-day mean reversion | 3.402 | 2.786 | model, today's state | -0.006 [-0.011, -0.001] | 62% | +1.551 [+1.415, +1.687] |
| diffusion Double Heston | model + mean reversion + leverage update | 3.660 | 2.965 | model, today's state | +0.173 [+0.114, +0.234] | 39% | +1.730 [+1.588, +1.880] |
| diffusion Double Heston | model + yesterday's misfit (headline forecast) | 1.736 | 1.377 | model, today's state | -1.416 [-1.534, -1.298] | 99% | +0.141 [+0.119, +0.164] |
| diffusion Double Heston | model + leverage update + yesterday's misfit | 2.086 | 1.679 | model + yesterday's misfit (headline forecast) | +0.302 [+0.224, +0.380] | 30% | +0.443 [+0.359, +0.527] |
| diffusion Double Heston | yesterday's smile + model-implied change | 2.216 | 1.758 | yesterday's smile (no model) | +0.290 [+0.214, +0.368] | 31% | +0.522 [+0.438, +0.607] |
| single Heston | yesterday's smile (no model) | 1.876 | 1.468 |  |   |  | +0.232 [+0.210, +0.255] |
| single Heston | model, today's state | 4.028 | 3.532 |  |   |  | +2.296 [+2.127, +2.480] |
| single Heston | model + its one-day mean reversion | 4.013 | 3.515 | model, today's state | -0.016 [-0.022, -0.010] | 77% | +2.279 [+2.111, +2.463] |
| single Heston | model + mean reversion + leverage update | 4.024 | 3.493 | model, today's state | -0.038 [-0.064, -0.012] | 55% | +2.258 [+2.082, +2.447] |
| single Heston | model + yesterday's misfit (headline forecast) | 1.776 | 1.416 | model, today's state | -2.115 [-2.285, -1.962] | 99% | +0.181 [+0.160, +0.203] |
| single Heston | model + leverage update + yesterday's misfit | 1.774 | 1.382 | model + yesterday's misfit (headline forecast) | -0.034 [-0.072, +0.003] | 48% | +0.146 [+0.100, +0.194] |
| single Heston | yesterday's smile + model-implied change | 1.877 | 1.435 | yesterday's smile (no model) | -0.032 [-0.068, +0.005] | 49% | +0.200 [+0.154, +0.246] |

*Test = origin days from 2022-01-01 (1173 days), Dev = before (1256 days); days are the intersection of the three headline runs, as in final_table.py. The time-scaled smile (no model) scores 1.236 on the test days. Intervals: 95% moving-block bootstrap (blocks of 20 days, 2,000 resamples). A negative difference means the forecast in that row is better than the one it is compared with. Nothing is refitted: the numbers are read from data/results/hyb_<run>.pkl and scored with dh2/evalpairs.py. Written by exp_variants_table.py.*
