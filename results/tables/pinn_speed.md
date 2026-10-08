**Speed: exact Fourier pricer against the surrogate**

| Task | Median time (ms) |
|---|---|
| exact pricer, one chain [cpu (1 thread, numpy)] | 10.9 |
| surrogate, one chain, prices [cpu] | 1.1 |
| surrogate, one chain, prices [mps] | 3.3 |
| exact pricer, 128 parameter sets x chain [cpu (1 thread, numpy)] | 1555.5 |
| surrogate, 128 parameter sets x chain [cpu] | 75.1 |
| surrogate, 128 parameter sets x chain [mps] | 19.6 |
| exact Greeks by finite differences (9 chain evaluations), one chain [cpu (1 thread, numpy)] | 119.6 |
| surrogate Greeks (delta, gamma, dC/dv1, dC/dv2), one chain [cpu] | 21.8 |

*Speed-up of the surrogate: one chain on CPU 10.0x, 128 parameter sets 21x (CPU) and 79x (GPU), Greeks 5.5x.*
