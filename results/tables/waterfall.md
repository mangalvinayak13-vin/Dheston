**From the 1.0 recipe to 2.0, one fix at a time (vol points; paired sample of test days)**

| Step | Same-day mean | Next-day mean | Next-day 95% interval | Next-day change vs previous step |
|---|---|---|---|---|
| DH1 (1.0 recipe) | 4.839 | 5.027 | [4.680, 5.393] |  |
| + clean data | 4.705 | 4.929 | [4.565, 5.317] | -0.098 |
| + IV-space vega-weighted objective | 3.589 | 3.881 | [3.550, 4.244] | -1.048 |
| + global multi-start, kappa2 > kappa1 | 2.555 | 2.838 | [2.531, 3.181] | -1.043 |
| + structure/state split, walk-forward | 2.683 | 2.917 | [2.596, 3.277] | +0.078 |
| + Merton jumps | 0.994 | 1.254 | [1.106, 1.415] | -1.662 |
| + yesterday's misfit carried | 0.994 | 1.100 | [0.965, 1.246] | -0.155 |
| reference: yesterday's smile |  | 1.585 |  |  |
| reference: time-scaled smile |  | 1.333 |  |  |

*118 sampled test days (every 10th trading day). Step 6 changes only the next-day forecast.*
