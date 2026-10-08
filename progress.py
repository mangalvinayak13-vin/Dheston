"""Live progress for the Double Heston 2.0 program.

    python3 progress.py --serve [--joblib <log of a run started before progress files existed>]   # page at http://localhost:8780/progress.html
    python3 progress.py --watch                                                                       # a bar in the terminal
    python3 progress.py --once                                                                        # print once and exit

Every 2 seconds it reads the job's own state (per-chunk progress files from run_walk.py, result files, a joblib log for a run that
predates progress files) and writes logs/progress.json, which progress.html polls. Nothing is estimated except time remaining, and that
is only shown for a stage that has actually started: elapsed time divided by the share done.
"""
import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES, PROG, OUT = HERE / "data" / "results", HERE / "logs" / "progress", HERE / "outputs"

# (title, weight in expected minutes, kind, key)
STAGES = [
    ("Ten years of NIFTY option files (2016 to 2026)", 1, "files", None),
    ("Diagnosis: why the model fails (2025-26)", 1, "file", "daily_dh_test2025.pkl"),
    ("Jump model walk-forward, ten years", 70, "walk", "dhj_main"),
    ("Forecast variants and benchmarks", 3, "file", "hyb_dhj_main.pkl"),
    ("Trading test: how many of 100 trades profit", 10, "tracked", ("trade_dhj_main", "trade_test_dhj_main.json")),
    ("Trading test, entering one day late (bounce check)", 10, "tracked", ("trade_dhj_main_lag1", "trade_test_dhj_main_lag1.json")),
    ("Jump-rate-as-state walk-forward", 80, "walk", "dhjs_main"),
    ("Wide parameter bounds (for crash days like Mar 2020)", 70, "walk", "dhj_wb"),
    ("Double Heston walk-forward (no jumps)", 50, "walk", "dh_main"),
    ("Pattern memory, part 2: past look-alike days guide the parameters (4 runs incl. control)", 280, "multi", ["dhj_mp10", "dhj_p10", "dhj_mp30", "dhj_mq05"]),
    ("Heston walk-forward", 45, "walk", "heston_main"),
    ("Ablations: window 80 days, refit every 20 days", 160, "multi", ["dhj_w80", "dhj_r20"]),
    ("Ablations: window 20 days, refit every 5 days", 170, "multi", ["dhj_w20", "dhj_r5"]),
    ("Ablations: prior strength 5 / 20, window 120 days", 260, "multi", ["dhj_p5", "dhj_p20", "dhj_w120"]),
    ("Pattern memory, part 1: past look-alike days correct the forecast", 90, "file", "memory_final.json"),
    ("Robustness, figures and the written report", 40, "file", "report_done.flag"),
]


def born(f: Path) -> float:
    """When a progress file was created (st_ctime changes on every write, so it is not a start time)."""
    st = f.stat()
    return getattr(st, "st_birthtime", st.st_mtime)


def walk_pct(key: str, joblib_log: str | None = None):
    """(percent, detail, started_at) for a walk-forward run keyed 'model_tag'."""
    files = sorted(PROG.glob(f"{key}_*.json"))
    final = RES / f"walk_{key}.pkl"
    hyb = RES / f"hyb_{key}.pkl"
    if final.exists() and hyb.exists():
        return 100.0, "done", None
    if files:
        done = total = 0
        for f in files:
            try:
                d = json.loads(f.read_text())
                done += d["done"]; total += d["total"]
            except Exception:
                pass
        start = min(born(f) for f in files)
        if total:
            pct = 100.0 * done / total
            if final.exists():
                return 99.0, "computing forecast variants", start
            return min(pct, 99.0), f"{done:,} of {total:,} days fitted", start
    if joblib_log and Path(joblib_log).exists() and key == "dhj_main":
        m = re.findall(r"Done\s+(\d+) out of\s+(\d+) \| elapsed:\s+([\d.]+)min remaining:\s+([\d.]+)min", Path(joblib_log).read_text())
        if m:
            k, n, el, rem = m[-1]
            return 100.0 * int(k) / int(n), f"{k} of {n} blocks finished ({rem} min remaining by joblib's estimate)", time.time() - 60 * float(el)
        return 1.0, "running (no block finished yet)", None
    if final.exists():
        return 99.0, "computing forecast variants", None
    return 0.0, "waiting", None


def snapshot(joblib_log=None) -> dict:
    t0 = (HERE / "dh2" / "fetch_nse.py").stat().st_ctime
    stages, total_w, done_w, running = [], 0.0, 0.0, None
    for title, w, kind, key in STAGES:
        started = None
        if kind == "files":
            n = len(list((HERE / "data" / "nse_index").glob("*.csv")))
            pct, detail = min(100.0, 100.0 * n / 2450), f"{n:,} daily files"
        elif kind == "file":
            ok = (RES / key).exists() or (OUT / key).exists()
            pct, detail = (100.0, "done") if ok else (0.0, "waiting")
        elif kind == "walk":
            pct, detail, started = walk_pct(key, joblib_log)
        elif kind == "tracked":                 # a script that writes progress files, finished when its result file exists
            prefix, final = key
            if (RES / final).exists():
                pct, detail = 100.0, "done"
            else:
                files = sorted(PROG.glob(f"{prefix}_*.json"))
                done = total = 0
                for f in files:
                    try:
                        d_ = json.loads(f.read_text()); done += d_["done"]; total += d_["total"]
                    except Exception:
                        pass
                pct = min(99.0, 100.0 * done / total) if total else 0.0
                detail = f"{done:,} of {total:,} days" if total else "waiting"
                started = min((born(f) for f in files), default=None)
        else:
            parts = [walk_pct(k) for k in key]
            pct = sum(p[0] for p in parts) / len(parts)
            detail = f"{sum(p[0] >= 100 for p in parts)} of {len(parts)} runs done"
            started = None
        state = "done" if pct >= 100 else ("running" if pct > 0 else "waiting")
        if state == "running" and running is None:
            running = title
        eta = None
        if kind == "multi" and state == "running":
            # the run in progress: its own elapsed / share done; each run not started yet is assumed to take as long as that one
            live = [p for p in parts if 1 < p[0] < 100 and p[2]]
            if live:
                p_, _, st_ = live[0]
                el = time.time() - st_
                full = el / (p_ / 100.0)
                eta = (full - el) + full * sum(p[0] <= 1 for p in parts)
        elif state == "running" and started and pct > 1:
            el = time.time() - started
            eta = max(0.0, el / (pct / 100.0) - el)
        stages.append({"title": title, "pct": round(pct, 1), "state": state, "detail": detail, "eta_min": None if eta is None else round(eta / 60, 1)})
        total_w += w
        done_w += w * min(pct, 100.0) / 100.0
    rem_min = sum(s["eta_min"] or 0 for s in stages if s["state"] == "running")
    pending = sum(w for (t, w, _, _), s in zip(STAGES, stages) if s["state"] == "waiting")
    return {"updated": datetime.now().strftime("%d %b %H:%M:%S"), "elapsed_h": round((time.time() - t0) / 3600, 1),
            "overall_pct": round(100.0 * done_w / total_w, 1), "running": running, "eta_current_min": rem_min,
            "pending_expected_h": round(pending / 60.0, 1), "stages": stages}


def bar(p, width=28):
    n = int(round(width * p / 100))
    return "█" * n + "░" * (width - n)


def show(s):
    print(f"\x1b[2J\x1b[H Double Heston 2.0   {s['updated']}   elapsed {s['elapsed_h']} h")
    print(f"\n OVERALL  {bar(s['overall_pct'], 40)} {s['overall_pct']:5.1f}%\n")
    for st in s["stages"]:
        mark = {"done": "\x1b[32m✓\x1b[0m", "running": "\x1b[33m▶\x1b[0m", "waiting": "·"}[st["state"]]
        eta = f"  ~{st['eta_min']:.0f} min left" if st["eta_min"] else ""
        print(f" {mark} {st['title'][:46]:46s} {bar(st['pct'], 20)} {st['pct']:5.1f}%  {st['detail']}{eta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true"); ap.add_argument("--watch", action="store_true"); ap.add_argument("--once", action="store_true")
    ap.add_argument("--joblib"); ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)

    def write():
        s = snapshot(a.joblib)
        (HERE / "logs" / "progress.json.tmp").write_text(json.dumps(s))
        (HERE / "logs" / "progress.json.tmp").replace(HERE / "logs" / "progress.json")
        return s

    if a.once:
        show(write()); return
    if a.watch:
        while True:
            show(write()); time.sleep(2)
    if a.serve:
        threading.Thread(target=lambda: [(write(), time.sleep(2)) for _ in iter(int, 1)], daemon=True).start()

        class H(SimpleHTTPRequestHandler):
            def __init__(s, *x, **k):
                super().__init__(*x, directory=str(HERE), **k)

            def end_headers(s):
                s.send_header("Cache-Control", "no-store")
                super().end_headers()

            def log_message(s, *x):
                pass
        print(f"live progress: http://localhost:{a.port}/progress.html", flush=True)
        ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
