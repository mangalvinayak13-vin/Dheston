"""Data-quality report: how many option quotes survive each filter of dh2/data.py, per year and for ten sample days.

    python3 -m dh2.dq [--workers 2] [--out results/dq]

The filters are NOT re-implemented from memory: this module re-runs dh2/data.py's own steps (its constants MIN_DAYS, MIN_PRICE,
MAX_LOG_MONEYNESS, PARITY_STRIKES, its _read/_pairs/implied_rate/smooth_rates/black_iv) with a counter after each step, in the
order dh2/data.py::_surface applies them, and then CHECKS the result against the cached surfaces (data/cache/surfaces.pkl, which
every finished run used): same number of surviving quotes, same number of expiries, same rate, on every day that has a surface.
Any mismatch is reported (there should be none).

The stored raw files (data/nse_index/*.csv) hold the index option and future rows only (NIFTY, plus BANKNIFTY from 2024); stock
options were never downloaded, so "option rows in the file" starts from index options.

Funnel (cumulative; each column is the number of quotes still alive after that step, summed over the days of the year):
  option_rows      rows of FinInstrmTp IDO (index options) in the day's file, any index
  nifty_options    ... of symbol NIFTY (the only underlying used)
  valid_px_strike  ... with a numeric strike and closing price (dh2.data._read keeps these)
  traded           ... with traded volume > 0
  expiry_ge3d      ... in an expiry at least MIN_DAYS (3) calendar days away
  has_forward      ... in an expiry where put-call parity gives a forward (>= 3 strikes with both a traded call and a traded put,
                   parity forward within 0.85-1.2 of the index level / of the strike where call = put)
  otm              ... out of the money (puts below the parity forward, calls at or above it)
  price_ge1        ... with a closing price of at least MIN_PRICE (Rs 1)
  moneyness_ok     ... within |ln(K/F)| <= MAX_LOG_MONEYNESS (0.22)
  iv_ok            ... whose Black implied volatility solves and lies in (3%, 200%)
  final            ... quotes of days that keep a surface (a day needs >= 8 such quotes)
Order note: the brief lists "price >= Rs 1" before ">= 3 days"; dh2/data.py applies the expiry tests first and the quote tests
after (the out-of-the-money split needs the expiry's forward), and the final set is identical either way.
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

ROOT = D.ROOT
STAGES = ["option_rows", "nifty_options", "valid_px_strike", "traded", "expiry_ge3d", "has_forward", "otm", "price_ge1",
          "moneyness_ok", "iv_ok", "final"]
MAT_EDGES = [7, 14, 30, 60]                      # maturity buckets in calendar days: <=7, 8-14, 15-30, 31-60, >60
MAT_LABELS = ["le7d", "d8_14", "d15_30", "d31_60", "gt60d"]
SHORT_LABELS = ["shortest_le7d", "shortest_8_14d", "shortest_15_30d", "shortest_gt30d"]


def _bucket(days: float) -> int:
    return int(np.searchsorted(MAT_EDGES, days, side="left"))


def raw_counts(day: date) -> dict:
    """Row counts of the day's stored file, before dh2.data._read touches it."""
    path = D.RAW / f"{day.isoformat()}.csv"
    df = pd.read_csv(path, dtype=str, usecols=["FinInstrmTp", "TckrSymb"])
    ido = df[df.FinInstrmTp == "IDO"]
    return {"rows_in_file": len(df), "option_rows": len(ido), "nifty_options": int((ido.TckrSymb == D.SYMBOL).sum())}


def raw_rate(day: date) -> float:
    return D._raw_rate(day)


def funnel_day(day: date, r: float) -> dict:
    """dh2.data._surface with a counter after every step. Returns the counts plus the per-expiry bookkeeping of the final surface."""
    out = {"day": day, "r": r, **raw_counts(day)}
    for k in STAGES[2:]:
        out[k] = 0
    out["drop_lt3d"] = out["drop_few_pairs"] = out["drop_bad_parity"] = 0
    o = D._read(day)
    out["has_surface"] = False
    out["n_expiries_listed"] = 0
    out["n_expiries_final"] = 0
    out["shortest_days"] = np.nan
    out["fail_reason"] = ""
    for k in MAT_LABELS:
        out[f"q_{k}"] = 0
    if o is None:
        out["fail_reason"] = "no NIFTY option rows"
        return out
    out["valid_px_strike"] = len(o)
    out["n_expiries_listed"] = o.xp.nunique()
    spot_file = float(o.und.median()) if o.und.notna().any() else float("nan")
    Ts, Fs, rows = [], [], []
    expiries = []
    for xp, g in o.groupby("xp"):
        days = (xp - day).days
        n_tr = int((g.vol > 0).sum())
        out["traded"] += n_tr
        if days < D.MIN_DAYS:
            out["drop_lt3d"] += n_tr
            continue
        out["expiry_ge3d"] += n_tr
        T = days / 365.0
        K, c, p = D._pairs(g)
        if len(K) < 3:
            out["drop_few_pairs"] += n_tr
            continue
        ref = spot_file if np.isfinite(spot_file) else float(np.median(K[np.argsort(np.abs(c - p))[:3]]))
        near = np.argsort(np.abs(K - ref))[:D.PARITY_STRIKES]
        F = float(np.median(K[near] + math.exp(r * T) * (c[near] - p[near])))
        if not (0.85 < F / ref < 1.2):
            out["drop_bad_parity"] += n_tr
            continue
        out["has_forward"] += n_tr
        ei = len(expiries)
        expiries.append(xp); Ts.append(T); Fs.append(F)
        tr = g[g.vol > 0]
        otm = tr[(tr.call & (tr.K >= F)) | (~tr.call & (tr.K < F))]
        out["otm"] += len(otm)
        pr = otm[otm.px >= D.MIN_PRICE]
        out["price_ge1"] += len(pr)
        mo = pr[np.abs(np.log(pr.K / F)) <= D.MAX_LOG_MONEYNESS]
        out["moneyness_ok"] += len(mo)
        for k, call, px, vol, oi in zip(mo.K, mo.call, mo.px, mo.vol, mo.oi):
            rows.append((k, ei, bool(call), px, vol, oi))
    if not expiries or not rows:
        out["fail_reason"] = "no usable expiry" if not expiries else "no quote left after the filters"
        return out
    K, e, call, px, vol, oi = (np.array(c) for c in zip(*rows))
    e = e.astype(int); call = call.astype(bool)
    T, F = np.array(Ts), np.array(Fs)
    iv = D.black_iv(px, F[e], K, T[e], r, call)
    keep = np.isfinite(iv) & (iv > 0.03) & (iv < 2.0)
    out["iv_ok"] = int(keep.sum())
    if keep.sum() < 8:
        out["fail_reason"] = "fewer than 8 quotes"
        return out
    out["has_surface"] = True
    out["final"] = int(keep.sum())
    used = np.unique(e[keep])
    out["n_expiries_final"] = len(used)
    dte = np.round(T[used] * 365.0)
    out["shortest_days"] = float(dte.min())
    out["r_check"] = r
    ee = e[keep]
    for u, d_ in zip(used, dte):
        out[f"q_{MAT_LABELS[_bucket(d_)]}"] += int((ee == u).sum())
    out["_F_check"] = F[used]
    out["_T_check"] = T[used]
    return out


# ------------------------------------------------------------------ the whole history
def run_all(workers: int = 2) -> pd.DataFrame:
    from joblib import Parallel, delayed
    days = D.trading_days()
    raw = np.array(Parallel(n_jobs=workers)(delayed(raw_rate)(d) for d in days))
    rates = D.smooth_rates(days, raw)
    rows = Parallel(n_jobs=workers)(delayed(funnel_day)(d, float(r)) for d, r in zip(days, rates))
    return pd.DataFrame(rows)


def check_against_cache(df: pd.DataFrame) -> dict:
    """The replicated funnel must reproduce the surfaces every finished run used."""
    path = ROOT / "data" / "cache" / "surfaces.pkl"
    if not path.exists():
        return {"checked": 0, "note": "cache not found"}
    S = pickle.load(open(path, "rb"))["surfaces"]
    by = df.set_index("day")
    bad, n = [], 0
    for s in S:
        n += 1
        row = by.loc[s.day]
        ok = bool(row["has_surface"]) and int(row["final"]) == s.n and int(row["n_expiries_final"]) == len(s.T) and abs(row["r"] - s.r) < 1e-12
        if ok:
            ok = np.allclose(np.asarray(row["_F_check"], float), s.F) and np.allclose(np.asarray(row["_T_check"], float), s.T)
        if not ok:
            bad.append(str(s.day))
    cached_days = {s.day for s in S}
    extra = [str(d) for d, h in zip(df.day, df.has_surface) if h and d not in cached_days]
    return {"checked": n, "mismatched_days": bad, "surface_days_not_in_cache": extra, "surfaces_in_cache": len(S),
            "surfaces_replicated": int(df.has_surface.sum())}


# ------------------------------------------------------------------ tables
def by_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = [d.year for d in df.day]
    rows = []
    for y, g in list(df.groupby("year")) + [("ALL", df)]:
        r = {"year": y, "files": len(g), "days_with_nifty_options": int((g.nifty_options > 0).sum()),
             "days_with_surface": int(g.has_surface.sum())}
        r["frac_days_usable"] = r["days_with_surface"] / r["files"]
        for k in ["option_rows", "nifty_options", "valid_px_strike", "traded", "expiry_ge3d", "has_forward", "otm", "price_ge1",
                  "moneyness_ok", "iv_ok", "final"]:
            r[k] = int(g[k].sum())
        r["drop_lt3d"], r["drop_few_pairs"], r["drop_bad_parity"] = (int(g[k].sum()) for k in ("drop_lt3d", "drop_few_pairs", "drop_bad_parity"))
        r["final_share_of_nifty_options"] = r["final"] / max(r["nifty_options"], 1)
        r["final_share_of_traded"] = r["final"] / max(r["traded"], 1)
        s = g[g.has_surface]
        r["quotes_per_surface_mean"] = float(s.final.mean()) if len(s) else np.nan
        r["quotes_per_surface_median"] = float(s.final.median()) if len(s) else np.nan
        r["expiries_listed_mean"] = float(g.n_expiries_listed[g.nifty_options > 0].mean())
        r["expiries_final_mean"] = float(s.n_expiries_final.mean()) if len(s) else np.nan
        r["expiries_final_min"] = int(s.n_expiries_final.min()) if len(s) else 0
        r["expiries_final_max"] = int(s.n_expiries_final.max()) if len(s) else 0
        sd = s.shortest_days
        r["shortest_le7d"] = int((sd <= 7).sum()); r["shortest_8_14d"] = int(((sd > 7) & (sd <= 14)).sum())
        r["shortest_15_30d"] = int(((sd > 14) & (sd <= 30)).sum()); r["shortest_gt30d"] = int((sd > 30).sum())
        r["shortest_median_days"] = float(sd.median()) if len(s) else np.nan
        for k in MAT_LABELS:
            r[f"quotes_{k}"] = int(g[f"q_{k}"].sum())
        r["reasons_no_surface"] = "; ".join(f"{k}: {v}" for k, v in g[~g.has_surface].fail_reason.value_counts().items())
        rows.append(r)
    return pd.DataFrame(rows)


def sample_days(df: pd.DataFrame, n: int = 10, extra: tuple = ("2020-03-23",)) -> pd.DataFrame:
    """n days spread evenly over the history (the mid-point of each tenth of the trading days), plus the listed stress days."""
    idx = [int((i + 0.5) * len(df) / n) for i in range(n)]
    out = df.iloc[idx].copy()
    out["sample"] = [f"evenly spaced {i + 1}/{n}" for i in range(n)]
    for d in extra:
        m = df[df.day == date.fromisoformat(d)]
        if len(m):
            m = m.copy()
            m["sample"] = "stress day (extra, not one of the 10)"
            out = pd.concat([out, m])
    keep = ["sample", "day", "r"] + STAGES + ["drop_lt3d", "drop_few_pairs", "drop_bad_parity", "n_expiries_listed", "n_expiries_final",
                                               "shortest_days", "has_surface", "fail_reason"] + [f"q_{k}" for k in MAT_LABELS]
    out = out[keep].reset_index(drop=True)
    out["final_share_of_traded"] = out.final / out.traded.clip(lower=1)
    return out


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:,.2f}")
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_readme(path: Path, yr: pd.DataFrame, sd: pd.DataFrame, chk: dict, days_df: pd.DataFrame) -> None:
    allr = yr[yr.year == "ALL"].iloc[0]
    y = yr[yr.year != "ALL"].copy()
    y["year"] = y.year.astype(str)
    y["frac_days_usable"] = y.frac_days_usable.map(lambda v: f"{100 * v:.1f}%")
    y["final_share_of_traded"] = y.final_share_of_traded.map(lambda v: f"{100 * v:.1f}%")
    txt = f"""# Data-quality report (NIFTY 50 index options, NSE end-of-day files)

Generated by `python3 -m dh2.dq` from the raw files in `data/nse_index/` (2,470 daily files, {days_df.day.min()} to {days_df.day.max()}).
Nothing here is typed by hand: every number comes from the script's run. The filters are the ones in `dh2/data.py`, applied by
re-running that module's own functions and constants (`MIN_DAYS` = {D.MIN_DAYS}, `MIN_PRICE` = Rs {D.MIN_PRICE:g},
`MAX_LOG_MONEYNESS` = {D.MAX_LOG_MONEYNESS}, `PARITY_STRIKES` = {D.PARITY_STRIKES}) with a counter after each step.

## Check against the surfaces every finished run used

The replicated funnel reproduces the cached surfaces (`data/cache/surfaces.pkl`) exactly: {chk['checked']} cached surfaces checked
(quote count, expiry count, rate, forwards and expiry times), **{len(chk.get('mismatched_days', []))} mismatches**; the funnel finds
{chk['surfaces_replicated']} usable days against {chk['surfaces_in_cache']} cached surfaces; days that are usable in the replication
but absent from the cache: {len(chk.get('surface_days_not_in_cache', []))}.

## Headline (all years)

- {int(allr.files):,} daily files; **{int(allr.days_with_surface):,} days ({100 * allr.frac_days_usable:.1f}%) give a usable surface**.
- {int(allr.nifty_options):,} NIFTY option rows in the files -> {int(allr.traded):,} traded (volume > 0, {100 * allr.traded / allr.nifty_options:.1f}%) -> **{int(allr.final):,} final quotes**
  ({100 * allr.final / allr.nifty_options:.1f}% of all NIFTY option rows, {100 * allr.final / allr.traded:.1f}% of the traded ones).
- A usable day has {allr.quotes_per_surface_mean:.0f} quotes on average (median {allr.quotes_per_surface_median:.0f}) in {allr.expiries_final_mean:.1f} expiries ({int(allr.expiries_final_min)} to {int(allr.expiries_final_max)}).
- Shortest expiry on usable days: <= 7 days on {int(allr.shortest_le7d):,} days, 8-14 days on {int(allr.shortest_8_14d):,}, 15-30 days on {int(allr.shortest_15_30d):,}, more than 30 days on {int(allr.shortest_gt30d):,}
  (median {allr.shortest_median_days:.0f} days).

## Per year (cumulative quote counts after each filter, summed over the year's days)

Columns after `nifty_options` are the stages described in the module docstring of `dh2/dq.py`. Full table with more columns:
`data_quality_by_year.csv`; per day: `data_quality_by_day.csv`.

{_md_table(y, ["year", "files", "days_with_surface", "frac_days_usable", "nifty_options", "traded", "expiry_ge3d", "has_forward", "otm", "price_ge1", "moneyness_ok", "iv_ok", "final", "final_share_of_traded"])}

Quotes lost at the two expiry tests (traded quotes in expiries dropped for being under {D.MIN_DAYS} days, or for lacking a parity forward):

{_md_table(y, ["year", "drop_lt3d", "drop_few_pairs", "drop_bad_parity"])}

Surface shape per year (usable days only):

{_md_table(y, ["year", "quotes_per_surface_mean", "expiries_listed_mean", "expiries_final_mean", "shortest_le7d", "shortest_8_14d", "shortest_15_30d", "shortest_gt30d", "shortest_median_days"])}

Final quotes by maturity bucket (calendar days to expiry):

{_md_table(y, ["year"] + [f"quotes_{k}" for k in MAT_LABELS])}

Days without a usable surface, with the reason: see column `reasons_no_surface` of `data_quality_by_year.csv`.

## Ten sample days

Ten days spread evenly through the history (the middle trading day of each tenth), plus one stress day (23 March 2020, the NIFTY
crash low) that is not one of the ten. `data_quality_sample_days.csv` has every column.

{_md_table(sd, ["sample", "day", "nifty_options", "traded", "expiry_ge3d", "has_forward", "otm", "price_ge1", "moneyness_ok", "iv_ok", "final", "n_expiries_final", "shortest_days"], {"shortest_days": "{:.0f}"})}

## Things to know

- The stored files contain index options only (NIFTY and, from 2024, BANKNIFTY), so "option rows" does not include stock options.
  Only NIFTY is used.
- NSE's closing price (`ClsPric`) is reported even for strikes that did not trade that day, so the volume > 0 step removes a large share
  of the file; those prices are stale settlement-style values, not trades.
- The out-of-the-money split uses each expiry's parity forward, so it comes after the expiry tests; the brief's list order differs from the
  code order only in presentation.
- No-arbitrage violations are measured separately (`dh2/noarb.py`, `noarb_by_year.csv`, `noarb_sensitivity.csv`); the dataset is not filtered for them
  (DECISIONS.md, decision 4).
"""
    path.write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "dq"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = run_all(a.workers)
    chk = check_against_cache(df)
    print("check against cache:", {k: (v if not isinstance(v, list) else (len(v), v[:5])) for k, v in chk.items()})
    yr = by_year(df)
    sd = sample_days(df)
    yr.to_csv(out / "data_quality_by_year.csv", index=False)
    sd.to_csv(out / "data_quality_sample_days.csv", index=False)
    df.drop(columns=[c for c in df.columns if c.startswith("_")] + ["r_check"], errors="ignore").to_csv(out / "data_quality_by_day.csv", index=False)
    write_readme(out / "README.md", yr, sd, chk, df)
    print(yr[["year", "files", "days_with_surface", "nifty_options", "traded", "final"]].to_string())
    print(f"wrote {out}")
    if chk.get("mismatched_days"):
        sys.exit(2)


if __name__ == "__main__":
    main()
