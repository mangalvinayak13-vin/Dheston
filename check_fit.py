# One day, per-day fits of each model: time, in-sample IV error, where the parameters land.
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2.data import load_day
from dh2 import calib, models as M

S = load_day("NIFTY 50", date(2025, 3, 20))
th = M.thin(S)
print(f"{S.n} quotes ({th.size} used in the fit), atm front/back iv {100 * calib.guess(S)['front']:.1f}/{100 * calib.guess(S)['back']:.1f}%")
for model in ("bs", "heston", "dh", "dhj"):
    t = time.time()
    f = calib.fit_day(model, S, n_starts=3, max_nfev=80)
    near = np.abs(np.log(S.K / S.F[S.e])) <= 0.05
    e = M.iv_error(model, f["x"], S)
    print(f"{model:7s} {time.time() - t:5.1f}s  IV rmse {f['rmse']:.3f} pts (near-ATM {np.sqrt(np.mean(e[near] ** 2)):.3f}); "
          f"at bounds: {f['at_bound']}\n        x = {np.round(f['x'], 4).tolist()}")
