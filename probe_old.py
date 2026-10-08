# Is NSE's older (pre-UDiFF) F&O bhavcopy reachable, and what does it contain? One request per sampled date.
import io
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import requests
from dh2.fetch_nse import HEADERS

URL = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{y}/{m}/fo{d:02d}{m}{y}bhav.csv.zip"
for d in [date(2016, 10, 3), date(2017, 6, 15), date(2018, 6, 14), date(2019, 7, 10), date(2020, 3, 23), date(2021, 5, 11),
          date(2022, 5, 10), date(2023, 6, 15), date(2024, 5, 30), date(2024, 6, 3), date(2024, 7, 5)]:
    u = URL.format(y=d.year, m=d.strftime("%b").upper(), d=d.day)
    r = requests.get(u, headers=HEADERS, timeout=30)
    info = ""
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
        n = df[(df.SYMBOL == "NIFTY") & (df.INSTRUMENT == "OPTIDX")]
        info = f"{len(df)} rows; NIFTY OPTIDX {len(n)} rows, expiries {n.EXPIRY_DT.nunique()}, cols {list(df.columns)[:8]}"
    print(d, r.status_code, len(r.content), info)
