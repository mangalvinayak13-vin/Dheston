**Next-day implied-volatility forecast error (vol points, lower is better), mean over days**

| Forecast | Kind | Dev | Test | 95% interval of the test mean | Diff vs best benchmark | 95% interval of the difference |
|---|---|---|---|---|---|---|
| yesterday's smile | model-free benchmark | 1.876 | 1.468 | [1.335, 1.614] |  |  |
| time-scaled smile | model-free benchmark | 1.624 | 1.236 | [1.112, 1.380] |  |  |
| Heston | model | 4.028 | 3.532 | [3.306, 3.762] | +2.296 | [+2.127, +2.480] |
| Heston + misfit | model | 1.776 | 1.416 | [1.286, 1.560] | +0.181 | [+0.160, +0.203] |
| diffusion DH | model | 3.409 | 2.792 | [2.574, 3.018] | +1.557 | [+1.419, +1.694] |
| diffusion DH + misfit | model | 1.736 | 1.377 | [1.245, 1.524] | +0.141 | [+0.119, +0.164] |
| jump model | model | 1.734 | 1.201 | [1.053, 1.372] | -0.035 | [-0.095, +0.028] |
| jump model + misfit | model | 1.317 | 1.034 | [0.915, 1.171] | -0.202 | [-0.227, -0.181] |

*Dev = 2016-2021, Test = 2022 onward; test days: 1173, development days: 1256. Difference is against the best benchmark on the test period (time-scaled smile); negative = better. Intervals: 95% moving-block bootstrap (dh2.report, final_table.py logic).*
