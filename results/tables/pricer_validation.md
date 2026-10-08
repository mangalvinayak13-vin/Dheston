**The project pricer against three independent pricers**

| Points | Model | Comparison | n | Max price error (Rs) | Max relative price error | Max IV error (vol pts) | Mean IV error (vol pts) |
|---|---|---|---|---|---|---|---|
| real chains (calibrated parameters of sampled days) | diffusion | project pricer vs COS | 18,071 | 4.13e-05 | 9.64e-07 | 1.45e-06 | 3.90e-09 |
| real chains (calibrated parameters of sampled days) | diffusion | Lewis vs COS | 18,071 | 2.17e-08 | 1.03e-09 | 7.57e-10 | 2.08e-12 |
| real chains (calibrated parameters of sampled days) | jumps | project pricer vs COS | 18,071 | 4.18e-05 | 6.79e-08 | 6.03e-07 | 1.44e-09 |
| real chains (calibrated parameters of sampled days) | jumps | Lewis vs COS | 18,071 | 4.01e-09 | 9.71e-11 | 3.15e-10 | 4.66e-12 |
| synthetic grid (calibrated-parameter sets, maturities, strikes) | diffusion | project pricer vs COS | 8,736 | 7.56e-03 | 2.02e-05 | 1.04e-04 | 3.29e-07 |
| synthetic grid (calibrated-parameter sets, maturities, strikes) | diffusion | Lewis vs COS | 8,736 | 7.76e-09 | 1.26e-09 | 1.51e-09 | 1.05e-11 |
| synthetic grid (calibrated-parameter sets, maturities, strikes) | jumps | project pricer vs COS | 8,736 | 7.61e-03 | 1.05e-05 | 1.01e-04 | 2.20e-07 |
| synthetic grid (calibrated-parameter sets, maturities, strikes) | jumps | Lewis vs COS | 8,736 | 3.51e-09 | 1.11e-09 | 8.86e-09 | 4.57e-11 |
| Monte Carlo (antithetic QE paths) | diffusion | Monte Carlo vs COS (max \|z\|, mean \|z\|) | 70 |  |  | 1.98e+00 | 5.96e-01 |

*Implied-vol errors only for options worth at least one tick (Rs 0.05). Monte Carlo row: the last two columns are standard errors (z), not vol points.*
