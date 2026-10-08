"""Export the calibrated parameters of the 2.0 walk-forward run: one row per trading day, in the pricer's own parameter names,
plus a summary by year. Every row uses only data up to that day.

    python3 export_params.py --model dhj --tag main      ->  outputs/parameters_<model>_<tag>.csv and parameters_by_year_<model>_<tag>.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dh2 import calib, struct as ST

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dhj"); ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    W = pickle.load(open(HERE / "data" / "results" / f"walk_{a.model}_{a.tag}.pkl", "rb"))
    names = {"dhj": ["v0_slow", "kappa_slow", "theta_slow", "xi_slow", "rho_slow", "v0_fast", "kappa_fast", "theta_fast", "xi_fast",
                     "rho_fast", "jump_rate", "jump_mean", "jump_sd"],
             "dh": ["v0_slow", "kappa_slow", "theta_slow", "xi_slow", "rho_slow", "v0_fast", "kappa_fast", "theta_fast", "xi_fast", "rho_fast"],
             "heston": ["v0", "kappa", "theta", "xi", "rho"]}[a.model]
    rows = []
    for r in W:
        u = ST.assemble(a.model, r["eta"], r["state"])
        rows.append([r["day"], *calib.to_phys(a.model, u), r["rmse_in"], r["n"]])
    df = pd.DataFrame(rows, columns=["date", *names, "same_day_iv_rmse_pts", "n_quotes"]).sort_values("date")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"parameters_{a.model}_{a.tag}.csv", index=False, float_format="%.6g")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    by = df.drop(columns=["date"]).groupby("year").median().round(4)
    by.to_csv(OUT / f"parameters_by_year_{a.model}_{a.tag}.csv")
    print(f"{len(df)} days -> {OUT}")
    print("median calibrated parameters by year:")
    print(by.T.to_string())


if __name__ == "__main__":
    main()
