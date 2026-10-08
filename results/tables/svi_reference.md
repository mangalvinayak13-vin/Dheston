**Same-day implied-vol RMSE (vol points) of the SVI reference and the walk-forward models**

| Quotes | n | SVI (5 parameters per expiry) | Double Heston + jumps | Double Heston |
|---|---|---|---|---|
| all quotes | 52,980 | 0.502 | 1.155 | 3.069 |
| near the money (\|k\| <= 0.04) | 22,168 | 0.533 | 1.042 | 1.107 |
| wings (\|k\| > 0.10) | 7,608 | 0.513 | 1.421 | 6.235 |

*118 sampled test days, 1031 expiries fitted. 49% of SVI fits have a butterfly violation somewhere on the grid (3.8% inside their quote range).*
