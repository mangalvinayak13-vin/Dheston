"""Download NSE's official end-of-day F&O file (UDiFF bhavcopy) for a range of days and keep only the NIFTY and
NIFTY BANK index options and futures: one small CSV per trading day in data/nse_index/.

    python3 -m dh2.fetch_nse --start 2024-07-08 --end 2026-10-06

Every file is the exchange's own record (no vendor in between): trade date, expiry, strike, call/put, open/high/low/close,
settlement price, the index level at the close (UndrlygPric), volume and open interest. Days NSE has no file for (weekends,
holidays) are skipped. Re-running only fetches days that are missing, so it resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nse_index"
URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
           "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
KEEP = ("NIFTY", "BANKNIFTY")
COLS = ["TradDt", "FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "OpnPric", "HghPric", "LwPric", "ClsPric",
        "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst", "TtlTradgVol", "TtlNbOfTxsExctd"]


def business_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def fetch_day(d: date, tries: int = 4) -> str:
    """'ok', 'have', 'skip' (no file that day) or 'fail'. Writes data/nse_index/YYYY-MM-DD.csv."""
    path = OUT / f"{d.isoformat()}.csv"
    if path.exists():
        return "have"
    for k in range(tries):
        try:
            r = requests.get(URL.format(d=d.strftime("%Y%m%d")), headers=HEADERS, timeout=30)
            if r.status_code in (403, 404):
                return "skip"
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            rows = [x for x in csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), "utf-8"))
                    if x["TckrSymb"] in KEEP and x["FinInstrmTp"] in ("IDO", "IDF")]
            if not rows:
                return "skip"
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(COLS)
                w.writerows([[x.get(c, "") for c in COLS] for x in rows])
            tmp.replace(path)
            return "ok"
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(1.5 * (k + 1))
    return "fail"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2024-07-08")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = list(business_days(date.fromisoformat(a.start), date.fromisoformat(a.end)))
    tally: dict[str, int] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for d, res in zip(days, ex.map(fetch_day, days)):
            tally[res] = tally.get(res, 0) + 1
            if res == "fail":
                print("failed:", d)
    print(f"{len(days)} weekdays: {tally}; files in {OUT}: {len(list(OUT.glob('*.csv')))}")


if __name__ == "__main__":
    main()
