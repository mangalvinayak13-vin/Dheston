# What parameters did the 2.0 calibrator find, and are they plausible and stable? python3 inspect_walk.py dhj y2526
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, struct as ST

model, tag = sys.argv[1], sys.argv[2]
W = pickle.load(open(Path(__file__).resolve().parent / "data" / "results" / f"walk_{model}_{tag}.pkl", "rb"))
ei, si = ST.IDX[model]
names = calib.__dict__  # noqa
label = {1: "kappa1 (slow speed)", 2: "theta1 (slow long-run var)", 3: "xi1 (slow vol-of-vol)", 4: "rho1", 6: "dk (fast - slow speed)",
         7: "theta2 (fast long-run var)", 8: "xi2 (fast vol-of-vol)", 9: "rho2", 10: "lam (jumps per year)", 11: "mu_j (mean log jump)",
         12: "delta_j (jump size sd)"}
E = np.array([r["eta"] for r in W])
lo, hi = (np.array(b) for b in calib.BOUNDS[model])
print(f"{len(W)} days; structural parameters (median, 10th-90th percentile, share of days within 2% of a bound):")
for k, j in enumerate(ei):
    v = E[:, k]
    tol = 0.02 * (hi[j] - lo[j])
    print(f"  {label[j]:28s} {np.median(v):9.4f}   [{np.percentile(v, 10):8.4f}, {np.percentile(v, 90):8.4f}]   at bound {100 * np.mean((v <= lo[j] + tol) | (v >= hi[j] - tol)):3.0f}%")
ST_ = np.array([r["state"] for r in W])
print(f"  daily state v0_1 (slow) median {np.median(ST_[:, 0]):.4f}, v0_2 (fast) median {np.median(ST_[:, 1]):.4f}; "
      f"total vol sqrt(v1+v2) median {100 * np.median(np.sqrt(ST_.sum(1))):.1f}%")
# how much does the structural estimate move between refits (10 days apart)?
ch = np.abs(np.diff(np.log(np.maximum(np.abs(E[::10]), 1e-6)), axis=0))
print("  median change per refit in log|parameter|:", {label[j].split(' ')[0]: round(float(np.median(ch[:, k])), 2) for k, j in enumerate(ei)})
r = np.array([x["rmse_in"] for x in W])
print(f"  in-sample (same-day) IV rmse: median {np.median(r):.3f}, mean {r.mean():.3f}")
