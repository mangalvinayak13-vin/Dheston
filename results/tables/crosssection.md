**Cross-sectional out-of-sample test (vol points, independent daily fits, 118 days)**

| Test | Model | Held-out IV RMSE | 95% interval | Median |
|---|---|---|---|---|
| random hold-out of quotes (interpolation) | Black-Scholes | 5.787 | [5.469, 6.123] | 5.714 |
| random hold-out of quotes (interpolation) | Heston | 3.245 | [2.980, 3.523] | 2.840 |
| random hold-out of quotes (interpolation) | Double Heston | 2.486 | [2.242, 2.743] | 2.075 |
| random hold-out of quotes (interpolation) | Double Heston + jumps | 0.705 | [0.639, 0.780] | 0.597 |
| fit short expiries, score longer ones (extrapolation in maturity) | Black-Scholes | 2.674 | [2.526, 2.843] | 2.480 |
| fit short expiries, score longer ones (extrapolation in maturity) | Heston | 1.368 | [1.210, 1.572] | 1.139 |
| fit short expiries, score longer ones (extrapolation in maturity) | Double Heston | 2.022 | [1.804, 2.268] | 1.699 |
| fit short expiries, score longer ones (extrapolation in maturity) | Double Heston + jumps | 1.410 | [1.239, 1.619] | 1.106 |
