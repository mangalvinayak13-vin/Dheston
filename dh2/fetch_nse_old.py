"""Download NSE's OLD-format F&O bhavcopy (pre-UDiFF, available back to 2016 and earlier) for a range of days, keep only
NIFTY index options and index futures, and write each day in the SAME schema as dh2/fetch_nse.py (UDiFF) so one loader
reads both:  data/nse_index/YYYY-MM-DD.csv.

    python3 -m dh2.fetch_nse_old --start 2016-10-03 --end 2024-05-31 --workers 6

Column mapping (old -> new):
    INSTRUMENT OPTIDX/FUTIDX -> FinInstrmTp IDO/IDF      SYMBOL -> TckrSymb       EXPIRY_DT -> XpryDt (ISO)
    STRIKE_PR -> StrkPric (blank for futures, like UDiFF) OPTION_TYP CE/PE -> OptnTp (blank for futures, old 'XX')
    OPEN/HIGH/LOW/CLOSE -> OpnPric/HghPric/LwPric/ClsPric   SETTLE_PR -> SttlmPric   OPEN_INT -> OpnIntrst
    CONTRACTS -> TtlTradgVol.  NOTE: the old file's CONTRACTS is the number of CONTRACTS (lots) traded, not units
    (units = contracts x lot size); check the unit convention of the UDiFF TtlTradgVol before mixing volumes across the
    2024-06 boundary. The trade date comes from the requested date (cross-checked with the TIMESTAMP column).
Not available in the old format and written as empty strings: LastPric, PrvsClsgPric, UndrlygPric, TtlNbOfTxsExctd.

Never overwrites an existing file (so the UDiFF-derived files stay untouched); re-running only fetches missing days.
Days NSE has no file for (holidays) are skipped. 429/5xx responses trigger a backoff.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import requests

from dh2.fetch_nse import COLS, HEADERS, OUT, business_days

URL = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{y}/{mon}/fo{dd}{mon}{y}bhav.csv.zip"
KEEP_SYMBOL = "NIFTY"
TYPE_MAP = {"OPTIDX": "IDO", "FUTIDX": "IDF"}


def _num(s: str, nd: int = 2) -> str:
    s = (s or "").strip()
    if s == "":
        return ""
    return f"{float(s):.{nd}f}"


def _int(s: str) -> str:
    s = (s or "").strip()
    return "" if s == "" else str(int(round(float(s))))


def _iso(s: str) -> str:
    return datetime.strptime(s.strip().title(), "%d-%b-%Y").date().isoformat()


def convert(rows, d: date) -> list[list[str]]:
    out = []
    for x in rows:
        x = {(k or "").strip().upper(): (v or "").strip() for k, v in x.items()}
        if x.get("SYMBOL") != KEEP_SYMBOL or x.get("INSTRUMENT") not in TYPE_MAP:
            continue
        ts = x.get("TIMESTAMP", "")
        if ts:
            try:
                if _iso(ts) != d.isoformat():
                    print(f"warning: {d} TIMESTAMP {ts} != requested date", file=sys.stderr)
            except ValueError:
                pass
        fut = TYPE_MAP[x["INSTRUMENT"]] == "IDF"
        rec = {
            "TradDt": d.isoformat(), "FinInstrmTp": TYPE_MAP[x["INSTRUMENT"]], "TckrSymb": x["SYMBOL"],
            "XpryDt": _iso(x["EXPIRY_DT"]),
            "StrkPric": "" if fut else _num(x["STRIKE_PR"]),
            "OptnTp": "" if fut else x["OPTION_TYP"].upper(),
            "OpnPric": _num(x["OPEN"]), "HghPric": _num(x["HIGH"]), "LwPric": _num(x["LOW"]), "ClsPric": _num(x["CLOSE"]),
            "LastPric": "", "PrvsClsgPric": "", "UndrlygPric": "",
            "SttlmPric": _num(x["SETTLE_PR"]), "OpnIntrst": _int(x["OPEN_INT"]), "TtlTradgVol": _int(x["CONTRACTS"]),
            "TtlNbOfTxsExctd": "",
        }
        out.append([rec[c] for c in COLS])
    return out


def fetch_day(d: date, tries: int = 5) -> str:
    """'ok', 'have', 'skip' (no file / no NIFTY rows that day) or 'fail'. Writes data/nse_index/YYYY-MM-DD.csv."""
    path = OUT / f"{d.isoformat()}.csv"
    if path.exists():
        return "have"
    url = URL.format(y=d.year, mon=d.strftime("%b").upper(), dd=f"{d.day:02d}")
    gone = 0
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            if r.status_code in (403, 404):
                gone += 1
                if gone >= 2:          # confirm once: a 403 can also be throttling, a holiday stays missing
                    return "skip"
                time.sleep(2.0)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(8.0 * (k + 1))
                continue
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            rows = convert(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), "utf-8", newline="")), d)
            if not rows:
                return "skip"
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(COLS)
                w.writerows(rows)
            if path.exists():          # never overwrite
                tmp.unlink()
                return "have"
            tmp.replace(path)
            return "ok"
        except (requests.RequestException, zipfile.BadZipFile, ValueError, KeyError):
            time.sleep(1.5 * (k + 1))
    return "fail"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2016-10-03")
    ap.add_argument("--end", default="2024-05-31")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = list(business_days(date.fromisoformat(a.start), date.fromisoformat(a.end)))
    tally: dict[str, int] = {}
    skipped = []
    with ThreadPoolExecutor(min(a.workers, 6)) as ex:
        for d, res in zip(days, ex.map(fetch_day, days)):
            tally[res] = tally.get(res, 0) + 1
            if res == "fail":
                print("failed:", d, flush=True)
            elif res == "skip":
                skipped.append(d.isoformat())
    print(f"{len(days)} weekdays: {tally}; files in {OUT}: {len(list(OUT.glob('*.csv')))}")
    print("skipped:", " ".join(skipped))


if __name__ == "__main__":
    main()
