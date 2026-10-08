# dhjs smoke test: one day fitted independently, pricing equals dhj with the same numbers, structural split round-trips.
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, models as M, struct as ST
from dh2.data import load_day

S = load_day(date(2025, 3, 20))
f = calib.fit_day("dhjs", S, n_starts=2, max_nfev=60)
x = f["x"]
print("dhjs fit: rmse %.3f pts, phys vector length %d, lam %.3f, mu_j %.3f, delta_j %.3f" % (f["rmse"], len(x), x[10], x[11], x[12]))
print("same prices as dhj with those numbers:", np.allclose(M.price("dhjs", x, S), M.price("dhj", x, S)))
eta, st = ST.split("dhjs", f["u"])
print("eta", np.round(eta, 3).tolist(), "state", np.round(st, 4).tolist(), "round trip:", np.allclose(ST.assemble("dhjs", eta, st), f["u"]))
print("expected state:", np.round(ST.expected_state("dhjs", eta, st, 1 / 252), 5).tolist())
