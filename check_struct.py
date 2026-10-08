# The window fit: do shared structural parameters + daily states fit nearly as well as independent daily fits, and how long does it take?
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, struct as ST, models as M
from dh2.data import load_all

S = load_all("2025-02-03", "2025-03-14")
print(f"{len(S)} days {S[0].day}..{S[-1].day}")
for model in ("heston", "dh"):
    t = time.time()
    # start: each day's own guess for the state, flat-ish structural values
    g = [calib.guess(s) for s in S]
    ei, si = ST.IDX[model]
    u0 = calib.starts(model, S[0], np.random.default_rng(0), 1)[0]
    eta0 = u0[ei]
    state0 = np.array([[u0[j] * (gg["front"] / g[0]["front"]) ** 2 for j in si] for gg in g])
    f = ST.fit_window(model, S, eta0, state0, max_nfev=30)
    print(f"\n{model}: window fit {time.time() - t:.0f}s, nfev {f['nfev']}, data rmse {f['rmse']:.3f} (weighted residual units)")
    print("  eta =", np.round(f["eta"], 4).tolist())
    print("  states first/last:", np.round(f["states"][0], 5).tolist(), np.round(f["states"][-1], 5).tolist())
    # in-sample IV error of each day using the shared eta and that day's refit state
    errs = []
    for d, s in enumerate(S):
        u = ST.assemble(model, f["eta"], f["states"][d])
        errs.append(np.sqrt(np.mean(M.iv_error(model, calib.to_phys(model, u), s) ** 2)))
    print(f"  in-sample IV rmse per day: median {np.median(errs):.3f} pts (mean {np.mean(errs):.3f})")
    t = time.time()
    sf = ST.fit_state(model, S[-1], f["eta"], f["states"][-1])
    print(f"  fit_state on the last day: {time.time() - t:.1f}s, rmse {sf['rmse']:.3f}")
    ind = calib.fit_day(model, S[-1], n_starts=3, max_nfev=80)
    print(f"  independent fit of the last day: rmse {ind['rmse']:.3f} (x = {np.round(ind['x'], 3).tolist()})")
