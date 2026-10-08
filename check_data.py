# Sanity-check the surfaces built from NSE's files: counts, implied rate, forwards, smile shape, coverage, by year.
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2.data import load_all, load_day

s = load_day(date(2025, 3, 20))
print(f"{s.day}: spot {s.spot:.2f} ({s.spot_source}), implied rate {100 * s.r:.2f}%, {s.n} quotes, {len(s.T)} expiries")
for i in range(min(len(s.T), 6)):
    m = s.e == i
    print(f"  expiry {s.expiry[i]} T={s.T[i] * 365:.0f}d forward {s.F[i]:.1f} (carry {100 * s.q[i]:+.2f}%) {m.sum()} quotes, "
          f"atm iv {100 * s.atm_iv(i):.2f}%, iv range {100 * s.iv[m].min():.1f}..{100 * s.iv[m].max():.1f}%")
t = time.time()
S = load_all()
print(f"\n{len(S)} days {S[0].day}..{S[-1].day} loaded in {time.time() - t:.0f}s")
print(f"{'year':>5} {'days':>5} {'quotes/day':>11} {'expiries':>9} {'atm iv (front) %':>17} {'implied rate %':>15} {'proxy spot':>11}")
for y in sorted({x.day.year for x in S}):
    Y = [x for x in S if x.day.year == y]
    iv = [100 * x.atm_iv(0) for x in Y]
    print(f"{y:>5} {len(Y):>5} {np.median([x.n for x in Y]):>11.0f} {np.median([len(x.T) for x in Y]):>9.0f} "
          f"{np.nanmedian(iv):>8.1f} ({np.nanmin(iv):.0f}..{np.nanmax(iv):.0f}) {100 * np.median([x.r for x in Y]):>9.2f} "
          f"{sum(x.spot_source == 'proxy' for x in Y):>11}")
