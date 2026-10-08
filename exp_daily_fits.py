"""C2 shared job: independent same-day fits (dh2.calib.fit_day objective, 4 starts) of bs, heston, dh and dhj on the sample days.

    python3 exp_daily_fits.py

Output: results/_cache/daily_fits_<version>.pkl, {(model, day index): {x, u, cost, rmse_in, rmse_next, at_bound, n, sec}}; read by
exp_attribution.py (step 3), exp_buckets.py (same-day fits) and exp_identifiability.py (cross-check). 4 workers, resumable.
Progress: logs/progress/c2_fits_00.json
"""
import c2_common as C


def main():
    S, samp = C.load_sample()
    print(f"{len(samp)} sample days {S[samp[0]].day} .. {S[samp[-1]].day}; fitting {C.MODELS} on {C.N_WORKERS} workers", flush=True)
    res = C.daily_fits(S, samp)
    bad = [k for k, v in res.items() if "error" in v]
    print("failed:", bad)
    for m in C.MODELS:
        v = [r["rmse_in"] for (mm, i), r in res.items() if mm == m and "error" not in r]
        o = [r["rmse_next"] for (mm, i), r in res.items() if mm == m and "error" not in r]
        print(f"{m:7s} same-day IV rmse mean {sum(v) / len(v):.3f}  next-day (params carried) mean {sum(o) / len(o):.3f}  over {len(v)} days")


if __name__ == "__main__":
    main()
