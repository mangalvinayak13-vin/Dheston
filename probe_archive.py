# How far back does NSE's UDiFF F&O archive go? (one request per sampled date)
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests
from dh2.fetch_nse import HEADERS, URL

for d in ["2026-10-06", "2026-01-15", "2025-06-16", "2025-01-15", "2024-10-15", "2024-08-01", "2024-07-08", "2024-07-05", "2024-06-03", "2023-06-15"]:
    r = requests.get(URL.format(d=d.replace("-", "")), headers=HEADERS, timeout=30)
    print(d, date.fromisoformat(d).strftime("%a"), r.status_code, len(r.content))
