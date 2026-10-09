"""Live progress for the Double Heston 2.0 upgrade program.

    python3 progress.py --serve      # page at http://localhost:8780/progress.html
    python3 progress.py --watch      # a bar in the terminal
    python3 progress.py --once       # print once and exit

Every 2 seconds it reads the work's own state (deliverable files that exist, progress files that long jobs write to logs/progress/) and
writes logs/progress.json, which progress.html polls. Nothing is estimated except time remaining, and that is only shown for a stage
that is running: elapsed time divided by the share done.
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROG = HERE / "logs" / "progress"

# (title, weight in expected minutes, kind, key)
#   "paths"  = share of the listed deliverable files (relative to this folder) that exist
#   "ptrack" = (prefix of progress files {"done": n, "total": m} written by a long job, final file whose existence means finished)
STAGES = [
    ("Earlier: DH2.0 walk-forwards, ablations, trading test (finished 8 Oct)", 30, "paths", ["data/results/walk_dhj_main.pkl", "COMPARISON_1.0_vs_2.0.md"]),
    ("Audit, plan and decisions (brief phase 0)", 5, "paths", ["AUDIT.md", "PLAN.md", "DECISIONS.md"]),
    ("A. Exact pricer cross-checks: COS, Lewis, Monte Carlo (phase 2)", 30, "paths", ["dh2/cos.py", "dh2/lewis.py", "dh2/mc.py", "tests/test_pricer_cross.py", "results/pricer_validation/README.md"]),
    ("B. PINN surrogate: build, train, validate, ablate (phase 4)", 120, "ptrack", ("pinn", "results/pinn/README.md")),
    ("C1. Data quality, India VIX, SVI reference, statistical tests", 40, "paths", ["results/dq/README.md", "results/vix/README.md", "results/svi/README.md", "results/stats/tests_test_period.md"]),
    ("C2. DH1 to DH2.0 waterfall, cross-section, identifiability, buckets", 120, "paths", ["results/attribution/README.md", "results/crosssection/README.md", "results/buckets/README.md", "results/identifiability/README.md"]),
    ("D. PINN-accelerated calibration vs exact (phase 5)", 60, "paths", ["dh2/calib_pinn.py", "results/pinn_calib/README.md"]),
    ("E. One-command pipeline, figures, report, explainer, UI adapter (phase 7)", 90, "paths", ["dh2/run_all.py", "results/metrics.json", "RESULTS.md", "docs/EXPLAINER.md", "PRESENTATION_NOTES.md", "today.json"]),
    ("F. Clean-environment test and independent re-derivation of every headline number (phase 8)", 40, "paths", ["results/VERIFICATION.md"]),
]


def born(f: Path) -> float:
    """When a file was created (st_ctime changes on every write, so it is not a start time)."""
    st = f.stat()
    return getattr(st, "st_birthtime", st.st_mtime)


def tracked(prefix: str):
    """(done, total, started_at) summed over the progress files of one long job, or None if it has not started."""
    files = sorted(PROG.glob(f"{prefix}_*.json"))
    done = total = 0
    for f in files:
        try:
            d = json.loads(f.read_text())
            done += d["done"]; total += d["total"]
        except Exception:
            pass
    return (done, total, min(born(f) for f in files)) if files and total else None


def snapshot() -> dict:
    t0 = born(HERE / "PLAN.md")
    stages, total_w, done_w, running = [], 0.0, 0.0, None
    for title, w, kind, key in STAGES:
        started = None
        if kind == "paths":
            have = [p for p in key if (HERE / p).exists()]
            pct, detail = 100.0 * len(have) / len(key), f"{len(have)} of {len(key)} deliverables"
        else:
            prefix, final = key
            tr = tracked(prefix)
            if (HERE / final).exists():
                pct, detail = 100.0, "done"
            elif tr:
                done, total, started = tr
                pct, detail = min(99.0, 100.0 * done / total), f"{done:,} of {total:,} steps"
            else:
                pct, detail = 0.0, "waiting"
        state = "done" if pct >= 100 else ("running" if pct > 0 else "waiting")
        if state == "running" and running is None:
            running = title
        eta = None
        if state == "running" and started and pct > 1:
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
    print(f"\x1b[2J\x1b[H Double Heston 2.0 upgrade   {s['updated']}   elapsed {s['elapsed_h']} h")
    print(f"\n OVERALL  {bar(s['overall_pct'], 40)} {s['overall_pct']:5.1f}%\n")
    for st in s["stages"]:
        mark = {"done": "\x1b[32m✓\x1b[0m", "running": "\x1b[33m▶\x1b[0m", "waiting": "·"}[st["state"]]
        eta = f"  ~{st['eta_min']:.0f} min left" if st["eta_min"] else ""
        print(f" {mark} {st['title'][:62]:62s} {bar(st['pct'], 20)} {st['pct']:5.1f}%  {st['detail']}{eta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true"); ap.add_argument("--watch", action="store_true"); ap.add_argument("--once", action="store_true")
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)

    def write():
        s = snapshot()
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
