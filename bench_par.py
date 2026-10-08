# How long does one day's Double Heston fit take alone, and with N processes at once? (cores: performance vs efficiency)
import sys
import time
from datetime import date
from pathlib import Path

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))


def one(seed):
    from dh2 import calib
    from dh2.data import load_day
    S = load_day(date(2025, 3, 20))
    t = time.time()
    f = calib.fit_day("dh", S, n_starts=3, seed=seed, max_nfev=60)
    return time.time() - t, f["rmse"]


if __name__ == "__main__":
    t, r = one(0)
    print(f"alone: {t:.1f}s (rmse {r:.3f})")
    for n in (4, 6, 9):
        w = time.time()
        res = Parallel(n_jobs=n)(delayed(one)(s) for s in range(n))
        print(f"{n} at once: wall {time.time() - w:.1f}s, each {min(x[0] for x in res):.1f}..{max(x[0] for x in res):.1f}s")
