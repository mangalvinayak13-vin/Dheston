# The model layer: speed, and the limits that must hold (Double Heston -> Black-Scholes as vol-of-vol -> 0; jumps off -> Double Heston).
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2.data import load_day
from dh2 import models as M

S = load_day(date(2025, 3, 20))
print(f"{S.n} quotes, {len(S.T)} expiries, {len(M.groups(S))} pricer calls per surface")

sig = 0.12
bs = M.price("bs", [sig], S)
# vol-of-vol ~ 0 and a flat variance: both factors reduce to Black-Scholes with total variance v1 + v2 = sig^2
dh0 = M.price("dh", [sig**2 / 2, 1.0, sig**2 / 2, 1e-3, 0.0, sig**2 / 2, 5.0, sig**2 / 2, 1e-3, 0.0], S)
print("DH(xi->0) vs BS: max abs price gap %.4f rupees (median price %.1f), max IV gap %.4f vol points"
      % (np.abs(dh0 - bs).max(), np.median(bs), np.abs(100 * (M.model_iv(dh0, S) - sig)).max()))
x = [0.01, 1.0, 0.015, 0.4, -0.7, 0.01, 6.0, 0.012, 0.8, -0.5]
dh = M.price("dh", x, S)
dhj0 = M.price("dhj", x + [1e-9, -0.05, 0.05], S)
print("DHJ(lam->0) vs DH: max abs price gap %.2e" % np.abs(dhj0 - dh).max())
dhj = M.price("dhj", x + [2.0, -0.04, 0.05], S)
print("jumps (2/yr, -4%%) change prices by up to %.2f rupees, IV by up to %.2f points" %
      (np.abs(dhj - dh).max(), np.abs(100 * (M.model_iv(dhj, S) - M.model_iv(dh, S))).max()))
for name, xx in (("bs", [sig]), ("heston", x[:5]), ("dh", x), ("dhj", x + [2.0, -0.04, 0.05])):
    t = time.perf_counter(); n = 30
    for _ in range(n):
        M.price(name, xx, S)
    print(f"  {name:7s} full surface: {(time.perf_counter() - t) / n * 1000:.1f} ms")
th = M.thin(S, 14)
t = time.perf_counter()
for _ in range(n):
    M.price("dh", x, S, th)
print(f"  dh thinned ({th.size} quotes): {(time.perf_counter() - t) / n * 1000:.1f} ms")
