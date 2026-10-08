"""Fix the jump SIZE for the 'dhjs' model from the development years (before 2022) only: the median calibrated mean and standard
deviation of the log jump in the full jump model's walk-forward run. The test period (2022 onward) plays no part.

    python3 fit_jump_shape.py --tag main    ->  config/jump_shape.json
"""
import argparse
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    W = pickle.load(open(HERE / "data" / "results" / f"walk_dhj_{a.tag}.pkl", "rb"))
    dev = [r for r in W if r["day"] < date(2022, 1, 1)]
    E = np.array([r["eta"] for r in dev])          # structural order for dhj: k1 th1 xi1 rho1 dk th2 xi2 rho2 lam mu delta
    out = {"mu_j": float(np.median(E[:, 9])), "delta_j": float(np.median(E[:, 10])), "lam_median": float(np.median(E[:, 8])),
           "from": f"walk_dhj_{a.tag}, {len(dev)} development days before 2022-01-01"}
    (HERE / "config").mkdir(exist_ok=True)
    (HERE / "config" / "jump_shape.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
