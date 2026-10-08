"""Adapter between the DH2.0 calibration and the finished website (the UI is not changed).

    python3 -m dh2.ui_adapter [--tag main]        # writes today.json

The website's `POST /api/price` takes `params` (ten settings in the pricer's names) and a `market`; it prices with the project's own pricer,
which has no jumps, and clamps every setting into the pricer's HESTON_BOUNDS. This writes the latest walk-forward calibration of the
diffusion Double Heston in exactly that format, records which values the website's bounds would change (so nothing is altered silently),
and carries the jump-model parameters for information (the website's pricer cannot use them without a backend change).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import calib, models as M, struct as ST

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["v0_1", "kappa1", "theta1", "xi1", "rho1", "v0_2", "kappa2", "theta2", "xi2", "rho2"]


def latest(model: str, tag: str):
    walk = pickle.load(open(ROOT / "data" / "results" / f"walk_{model}_{tag}.pkl", "rb"))
    rec = max(walk, key=lambda r: r["day"])
    return rec, calib.to_phys(model, ST.assemble(model, rec["eta"], rec["state"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    rec, x = latest("dh", a.tag)
    ten = {k: float(v) for k, v in zip(KEYS, x[:10])}
    bounds = M.P.HESTON_BOUNDS
    clamped = {}
    for k, v in ten.items():
        lo, hi = bounds[k[:-1].rstrip("_")]
        if not lo <= v <= hi:
            clamped[k] = {"calibrated": v, "website_would_use": min(max(v, lo), hi)}
    recj, xj = latest("dhj", a.tag)
    out = {"asof": rec["day"].isoformat(), "model": "Double Heston 2.0, diffusion (no jumps), walk-forward calibration on NIFTY 50",
           "params": ten, "same_day_iv_rmse_vol_points": float(rec["rmse_in"]), "website_bounds_would_change": clamped,
           "jump_model_for_information": {"asof": recj["day"].isoformat(), "params": {k: float(v) for k, v in zip(KEYS, xj[:10])},
                                          "lam_per_year": float(xj[10]), "mu_j": float(xj[11]), "delta_j": float(xj[12]),
                                          "same_day_iv_rmse_vol_points": float(recj["rmse_in"])},
           "note": "The website prices with the pricer without jumps; the jump model needs a backend change to be priced there. "
                   "Individual parameters are weakly identified: the prices they produce are what is validated, not each number."}
    (ROOT / "today.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("asof", "same_day_iv_rmse_vol_points", "website_bounds_would_change")}, indent=1))


if __name__ == "__main__":
    main()
