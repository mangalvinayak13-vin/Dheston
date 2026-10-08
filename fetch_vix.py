"""Download the India VIX daily history to data/vix/india_vix.csv (idempotent: re-running refreshes it, never appends duplicates).

    python3 fetch_vix.py [--source auto|yahoo|nse]

Sources, tried in this order until one gives a full history:
  1. Yahoo Finance's public chart endpoint for ^INDIAVIX (plain HTTPS + JSON, no third-party finance package). Yahoo answers
     429 (Too Many Requests) to some networks for hours; the script retries twice with a pause and moves on.
  2. niftyindices.com's historical-data endpoint (often refuses scripts).
  3. NSE's own daily index-close archive, nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv: one small official
     file per trading day that lists every NSE index, India VIX included (open, high, low, close). This is the same archive host the
     project's option files come from (dh2/fetch_nse.py). Only days not already in data/vix/india_vix.csv are requested.
  4. A hand-downloaded file at data/vix/india_vix_manual.csv (niftyindices.com / nseindia.com "India VIX" historical CSV).
If all fail, data/vix/MANUAL_DOWNLOAD.txt says exactly what to download by hand.

The CSV has: date, open, high, low, close, volume, nifty50_close (India VIX is in percent, annualised: 15.3 means 15.3 %;
volume and nifty50_close are only filled by source 3). The record of which source filled the file is data/vix/SOURCE.txt.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "vix"
OUT = OUT_DIR / "india_vix.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX?range=max&interval=1d"
NSE_DAILY = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d}.csv"
NSE_HEADERS = {"User-Agent": UA, "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
COLS = ["date", "open", "high", "low", "close", "volume", "nifty50_close"]


def _get(url: str, timeout: int = 60) -> bytes:
    # requests (it ships its own CA bundle via certifi; this Python's urllib has no system certificates)
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"}, timeout=timeout)
    r.raise_for_status()
    return r.content


def _get_retry(urls: list[str], waits=(15, 45)) -> bytes:
    """Yahoo answers 429 when a client hits it too often: retry both hosts with growing waits, then give up."""
    last = None
    for w in (0,) + tuple(waits):
        time.sleep(w)
        for u in urls:
            try:
                return _get(u)
            except requests.RequestException as e:
                last = e
    raise last


def from_yahoo() -> pd.DataFrame:
    js = json.loads(_get_retry([YAHOO, YAHOO.replace("query1", "query2")]))
    res = js["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    tz_off = res["meta"].get("gmtoffset", 19800)     # Yahoo stamps the session open in exchange time; shift to the local calendar date
    dates = [datetime.fromtimestamp(t + tz_off, tz=timezone.utc).date().isoformat() for t in ts]
    df = pd.DataFrame({"date": dates, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                       "volume": q["volume"], "nifty50_close": float("nan")})
    return df.dropna(subset=["close"])


def from_niftyindices_web() -> pd.DataFrame:
    """niftyindices.com's historical-data endpoint (a POST that returns JSON rows); a best effort, it often refuses scripts."""
    rows = []
    for y in range(2016, datetime.now().year + 1):
        body = json.dumps({"cinfo": json.dumps({"name": "INDIA VIX", "startDate": f"01-Jan-{y}", "endDate": f"31-Dec-{y}",
                                                "indexName": "INDIA VIX"})})
        r = requests.post("https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString", data=body, timeout=60,
                          headers={"User-Agent": UA, "Content-Type": "application/json; charset=UTF-8",
                                   "Origin": "https://www.niftyindices.com", "Referer": "https://www.niftyindices.com/reports/historical-data"})
        r.raise_for_status()
        rows += json.loads(r.json()["d"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"date": pd.to_datetime(df["HistoricalDate"], format="mixed").dt.date.astype("string"),
                        "open": pd.to_numeric(df["OPEN"], errors="coerce"), "high": pd.to_numeric(df["HIGH"], errors="coerce"),
                        "low": pd.to_numeric(df["LOW"], errors="coerce"), "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
                        "volume": 0.0, "nifty50_close": float("nan")})
    return out.dropna(subset=["close"])


def _nse_day(d: date, tries: int = 6):
    """One day's India VIX row from NSE's index-close archive: a dict, None (no file that day) or the string 'FAIL' after repeated
    network failures (the caller retries those later instead of aborting the whole download)."""
    url = NSE_DAILY.format(d=d.strftime("%d%m%Y"))
    for k in range(tries):
        try:
            r = requests.get(url, headers=NSE_HEADERS, timeout=30)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            names = df["Index Name"].str.strip().str.lower()
            v, n = df[names == "india vix"], df[names == "nifty 50"]
            if v.empty:
                return None
            v = v.iloc[0]
            num = lambda x: pd.to_numeric(x, errors="coerce")
            return {"date": d.isoformat(), "open": num(v["Open Index Value"]), "high": num(v["High Index Value"]),
                    "low": num(v["Low Index Value"]), "close": num(v["Closing Index Value"]), "volume": 0.0,
                    "nifty50_close": num(n.iloc[0]["Closing Index Value"]) if not n.empty else float("nan")}
        except (requests.RequestException, pd.errors.ParserError, KeyError, ValueError):
            time.sleep(min(2.0 * (k + 1), 20))
    return "FAIL"


def from_nse_archive(workers: int = 2) -> pd.DataFrame:
    """India VIX from NSE's daily index files, for every trading day of the option data. Progress is saved to
    data/vix/_nse_partial.csv, so an interrupted run (network drop) resumes where it stopped; single failed days are retried at the end."""
    sys.path.insert(0, str(ROOT))
    from dh2.data import trading_days
    days = trading_days()
    part = OUT_DIR / "_nse_partial.csv"
    have = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=COLS)
    done = set(have["date"].astype(str))
    # days NSE has no VIX row for are remembered too, so a re-run does not ask again
    none_file = OUT_DIR / "_nse_no_file.txt"
    nofile = set(none_file.read_text().split()) if none_file.exists() else set()
    rows = have.to_dict("records")
    for rnd in range(3):
        todo = [d for d in days if d.isoformat() not in done and d.isoformat() not in nofile]
        print(f"NSE index archive, round {rnd + 1}: {len(todo)} of {len(days)} trading days to fetch", flush=True)
        if not todo:
            break
        with ThreadPoolExecutor(workers) as ex:
            for i, (d, res) in enumerate(zip(todo, ex.map(_nse_day, todo))):
                if res is None:
                    nofile.add(d.isoformat())
                elif isinstance(res, dict):
                    rows.append(res); done.add(d.isoformat())
                if (i + 1) % 250 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
                    pd.DataFrame(rows, columns=COLS).to_csv(part, index=False)
                    none_file.write_text("\n".join(sorted(nofile)))
        pd.DataFrame(rows, columns=COLS).to_csv(part, index=False)
        none_file.write_text("\n".join(sorted(nofile)))
    missing = [d for d in days if d.isoformat() not in done and d.isoformat() not in nofile]
    if missing:
        raise RuntimeError(f"NSE index archive: {len(missing)} days still failing, first {missing[0]} (network?)")
    print(f"NSE index archive: {len(done)} days with a VIX row, {len(nofile)} days without a file", flush=True)
    return pd.DataFrame(rows, columns=COLS).dropna(subset=["close"])


def from_manual() -> pd.DataFrame | None:
    """A hand-downloaded 'India VIX' history file, if put at data/vix/india_vix_manual.csv (niftyindices.com or nseindia.com layout)."""
    p = OUT_DIR / "india_vix_manual.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: "close" for c in df.columns if c.startswith("close")})
    d = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", format="mixed")
    out = pd.DataFrame({"date": d.dt.date.astype("string"), "open": pd.to_numeric(df.get("open"), errors="coerce"),
                        "high": pd.to_numeric(df.get("high"), errors="coerce"), "low": pd.to_numeric(df.get("low"), errors="coerce"),
                        "close": pd.to_numeric(df["close"], errors="coerce"), "volume": 0.0, "nifty50_close": float("nan")})
    return out.dropna(subset=["date", "close"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "yahoo", "nse"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = [("yahoo ^INDIAVIX", from_yahoo), ("niftyindices.com web", from_niftyindices_web),
            ("NSE daily index-close archive", lambda: from_nse_archive(a.workers)), ("manual file", from_manual)]
    if a.source == "yahoo":
        plan = plan[:1]
    elif a.source == "nse":
        plan = plan[2:3]
    df, src, errors = None, None, []
    for name, fn in plan:
        try:
            df = fn()
            if df is not None and len(df) > 500:
                src = name
                break
            errors.append(f"{name}: no data" if df is None else f"{name}: only {len(df)} rows")
        except (requests.RequestException, KeyError, ValueError, TimeoutError, OSError, IndexError, TypeError, RuntimeError) as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:150]}")
        df = None
    if df is None:
        msg = ("India VIX download failed automatically.\n" + "\n".join(errors) + "\n\nManual step: open https://www.niftyindices.com/reports/historical-data , "
               "choose 'India VIX', set the dates 01-Oct-2016 to today, download the CSV, save it as data/vix/india_vix_manual.csv and re-run "
               "python3 fetch_vix.py (it converts it to data/vix/india_vix.csv). NSE's alternative: https://www.nseindia.com/reports-indices-historical-vix "
               "(same columns).\n")
        (OUT_DIR / "MANUAL_DOWNLOAD.txt").write_text(msg)
        print(msg)
        return 1
    df = df[COLS].drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    skipped = "; ".join(errors)
    (OUT_DIR / "SOURCE.txt").write_text(f"source: {src}\nrows: {len(df)}\nfirst: {df.date.iloc[0]}\nlast: {df.date.iloc[-1]}\n"
                                        f"fetched: {datetime.now().isoformat(timespec='seconds')}\nearlier sources that failed: {skipped or 'none'}\n")
    (OUT_DIR / "MANUAL_DOWNLOAD.txt").unlink(missing_ok=True)
    print(f"{src}: {len(df)} rows {df.date.iloc[0]} .. {df.date.iloc[-1]} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
