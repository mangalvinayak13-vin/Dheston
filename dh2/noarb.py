"""No-arbitrage DIAGNOSTICS on the quote set (decision 4 in DECISIONS.md: the dataset is NOT filtered, the violation share is reported).

    python3 -m dh2.noarb [--out results/dq]

All checks use the day's final quotes (dh2.data.Surface: out-of-the-money options, one quote per strike and expiry). Two tolerances:
  "1 tick"                the brief's: one price tick (Rs 0.05, NSE's option tick) on every comparison (two rounded prices differ by up
                          to one tick; a butterfly of three by up to one tick too);
  "1 tick + 0.25 vol pt"  an added, noise-aware variant: on top of the tick, every quote may be off by 0.25 volatility points (0.25 x vega
                          rupees). End-of-day NSE prints are closing prints, not simultaneous quotes (illiquid strikes may last have
                          traded hours earlier), so noise of a few tenths of a vol point is the normal state of the data, far bigger
                          than a tick for expensive options.

(a) Monotonicity in strike, per expiry, on the forward-price scale. Every quote is turned into the call price that put-call parity
    implies at the same forward: C = price for a call, C = price + exp(-rT)(F - K) for a put. Then, for neighbouring strikes K1 < K2,
      call prices must not rise with the strike:                  C2 - C1 <= tol
      put prices must not fall with the strike (P = C + DF(K-F)): (C2 - C1) + DF (K2 - K1) >= -tol
(b) Butterfly (convexity in strike): for three neighbouring strikes K1 < K2 < K3, C(K2) <= w C(K1) + (1-w) C(K3) + tol,
    w = (K3-K2)/(K3-K1).
(c) Calendar monotonicity of total variance w = IV^2 T at fixed log-moneyness k = ln(K/F): for neighbouring expiries, each quote of
    the longer expiry is compared with the shorter expiry's w linearly interpolated in k (no extrapolation outside the shorter expiry's
    quotes). Violation if w_long < w_short(k) - tol_w, tol_w = the total-variance change that the price tolerance causes on each of the
    two legs (2 IV T tol / (100 vega) per leg, vega in rupees per volatility point).

A quote counts as "violating" if it takes part in a violating pair/triple: both quotes of a violating pair, all three of a violating
butterfly, and for (c) the longer-expiry quote and the two shorter-expiry quotes that bracket it. This is conservative (it flags the
innocent neighbour as well as the culprit; the data cannot say which one is wrong). The call/put junction (the first call, K >= F) joins
two price series through the estimated parity forward, so a forward error of a few paise can create a violation there; the number of
violations that straddle the junction is reported separately.

Sensitivity: the test-period (2022-2026) next-day IV error of the time-scaled-smile benchmark and of the jump model + misfit
(hyb_dhj_main 'carry_res') is re-scored with the violating quotes of the TARGET day dropped from the scoring only: predictions,
calibrations and data are untouched.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import evalpairs as EP
from .data import Surface, load_all

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.05
SPLIT = date(2022, 1, 1)
TOLERANCES = {"1 tick": 0.0, "1 tick + 0.25 vol pt": 0.25}     # name -> extra volatility-point noise allowed per quote


def call_equivalent(S: Surface) -> np.ndarray:
    """Per quote: the call price at the same expiry and strike that put-call parity implies at the expiry's forward."""
    DF = np.exp(-S.r * S.T[S.e])
    return np.where(S.is_call, S.price, S.price + DF * (S.F[S.e] - S.K))


def diagnose(S: Surface, tick: float = TICK, vol_tol: float = 0.0) -> dict:
    """Violation flags per quote and the counts of what was tested, for one surface. Tolerance: one `tick` of price on every quote, plus
    `vol_tol` volatility points of noise on every quote (vol_tol * vega rupees; vega = rupees per volatility point)."""
    n = S.n
    fa, fb, fc = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    C = call_equivalent(S)
    c = dict(pairs=0, a1=0, a2=0, a_strict=0, triples=0, b=0, b_strict=0, junction=0, cal_tested=0, cal=0,
             a_excess=[], b_excess=[], c_excess=[])
    slices_ab = 0
    kw = []                                     # per expiry: sorted quote indices, k, total variance, tolerance in total variance
    for e in range(len(S.T)):
        q = np.flatnonzero(S.e == e)
        if q.size == 0:
            kw.append(None)
            continue
        q = q[np.argsort(S.K[q])]
        K, Ce, DF, vg = S.K[q], C[q], np.exp(-S.r * S.T[e]), S.vega[q]
        eps = vol_tol * vg                        # price noise allowed on each quote, on top of the one tick per comparison
        bad_ab = False
        if q.size >= 2:
            dK, dC = np.diff(K), np.diff(Ce)
            tol = tick + eps[:-1] + eps[1:]
            v1 = dC > tol
            dP = dC + DF * dK
            v2 = dP < -tol
            c["pairs"] += dK.size
            c["a1"] += int(v1.sum()); c["a2"] += int(v2.sum())
            c["a_strict"] += int(((dC > 1e-9) | (dP < -1e-9)).sum())
            v = v1 | v2
            c["a_excess"] += list(((dC - tol)[v1]) / tick) + list(((-dP - tol)[v2]) / tick)
            iv_ = np.flatnonzero(v)
            fa[q[iv_]] = True; fa[q[iv_ + 1]] = True
            c["junction"] += int((S.is_call[q][iv_] != S.is_call[q][iv_ + 1]).sum())
            bad_ab |= bool(v.any())
        if q.size >= 3:
            w = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
            ex = Ce[1:-1] - (w * Ce[:-2] + (1 - w) * Ce[2:])
            tol = tick + eps[1:-1] + w * eps[:-2] + (1 - w) * eps[2:]
            vb = ex > tol
            c["triples"] += ex.size; c["b"] += int(vb.sum()); c["b_strict"] += int((ex > 1e-9).sum())
            c["b_excess"] += list(((ex - tol)[vb]) / tick)
            ib = np.flatnonzero(vb)
            fb[q[ib]] = True; fb[q[ib + 1]] = True; fb[q[ib + 2]] = True
            c["junction"] += int((S.is_call[q][ib] != S.is_call[q][ib + 2]).sum())
            bad_ab |= bool(vb.any())
        slices_ab += int(bad_ab)
        k = np.log(K / S.F[e])
        iv = S.iv[q]
        dw = 2.0 * iv * S.T[e] * (tick / (100.0 * np.maximum(vg, 1e-6)) + vol_tol / 100.0)
        kw.append((q, k, iv**2 * S.T[e], dw))
    for e in range(1, len(S.T)):
        if kw[e] is None:
            continue
        s = next((kw[j] for j in range(e - 1, -1, -1) if kw[j] is not None and len(kw[j][0]) >= 2), None)
        if s is None:
            continue
        qs, ks, ws, dws = s
        ql, kl, wl, dwl = kw[e]
        m = (kl >= ks[0]) & (kl <= ks[-1])
        if not m.any():
            continue
        w_int = np.interp(kl[m], ks, ws)
        tol = dwl[m] + np.interp(kl[m], ks, dws)
        viol = wl[m] < w_int - tol
        c["cal_tested"] += int(m.sum()); c["cal"] += int(viol.sum())
        sig = np.maximum(np.sqrt(np.maximum(wl[m] / S.T[e], 1e-12)), 1e-6)
        c["c_excess"] += list(((w_int - tol - wl[m])[viol]) / (2.0 * sig[viol] * S.T[e]) * 100.0)       # vol points, beyond the tolerance
        fc[ql[m][viol]] = True
        hi = np.searchsorted(ks, kl[m][viol])
        fc[qs[np.clip(hi - 1, 0, len(qs) - 1)]] = True
        fc[qs[np.clip(hi, 0, len(qs) - 1)]] = True
    return {"a": fa, "b": fb, "c": fc, "any": fa | fb | fc, "counts": c, "slices_ab": slices_ab, "slices": len(S.T)}


def by_year(S: list[Surface], vol_tol: float = 0.0, label: str = "1 tick"):
    acc: dict = {}
    flags = {}
    for s in S:
        d = diagnose(s, vol_tol=vol_tol)
        flags[s.day] = d
        y = s.day.year
        for key in (y, "ALL"):
            a = acc.setdefault(key, dict(days=0, quotes=0, slices=0, slices_ab=0, qa=0, qb=0, qc=0, qany=0, da=0, db=0, dc=0, dany=0,
                                         pairs=0, a1=0, a2=0, a_strict=0, triples=0, b=0, b_strict=0, junction=0, cal_tested=0, cal=0,
                                         a_ex=[], b_ex=[], c_ex=[]))
            c = d["counts"]
            a["days"] += 1; a["quotes"] += s.n; a["slices"] += d["slices"]; a["slices_ab"] += d["slices_ab"]
            a["qa"] += int(d["a"].sum()); a["qb"] += int(d["b"].sum()); a["qc"] += int(d["c"].sum()); a["qany"] += int(d["any"].sum())
            a["da"] += int(d["a"].any()); a["db"] += int(d["b"].any()); a["dc"] += int(d["c"].any()); a["dany"] += int(d["any"].any())
            for k in ("pairs", "a1", "a2", "a_strict", "triples", "b", "b_strict", "junction", "cal_tested", "cal"):
                a[k] += c[k]
            a["a_ex"] += c["a_excess"]; a["b_ex"] += c["b_excess"]; a["c_ex"] += c["c_excess"]
    rows = []
    for key in sorted(k for k in acc if k != "ALL") + ["ALL"]:
        a = acc[key]
        rows.append({
            "tolerance": label, "year": key, "days": a["days"], "quotes": a["quotes"],
            "share_quotes_a_monotonic": a["qa"] / a["quotes"], "share_quotes_b_butterfly": a["qb"] / a["quotes"],
            "share_quotes_c_calendar": a["qc"] / a["quotes"], "share_quotes_any": a["qany"] / a["quotes"],
            "share_surfaces_a": a["da"] / a["days"], "share_surfaces_b": a["db"] / a["days"], "share_surfaces_c": a["dc"] / a["days"],
            "share_surfaces_any": a["dany"] / a["days"], "share_expiry_slices_ab": a["slices_ab"] / max(a["slices"], 1),
            "pairs_tested": a["pairs"], "pairs_violating_call_decreasing": a["a1"], "pairs_violating_put_increasing": a["a2"],
            "share_pairs_a": (a["a1"] + a["a2"]) / max(a["pairs"], 1), "share_pairs_a_strict_zero_tolerance": a["a_strict"] / max(a["pairs"], 1),
            "triples_tested": a["triples"], "triples_violating": a["b"], "share_triples_b": a["b"] / max(a["triples"], 1),
            "share_triples_b_strict_zero_tolerance": a["b_strict"] / max(a["triples"], 1),
            "calendar_comparisons": a["cal_tested"], "calendar_violations": a["cal"], "share_calendar_c": a["cal"] / max(a["cal_tested"], 1),
            "ab_violations_at_call_put_junction": a["junction"],
            "median_excess_a_ticks": float(np.median(a["a_ex"])) if a["a_ex"] else 0.0,
            "median_excess_b_ticks": float(np.median(a["b_ex"])) if a["b_ex"] else 0.0,
            "median_excess_c_volpts": float(np.median(a["c_ex"])) if a["c_ex"] else 0.0,
            "p95_excess_c_volpts": float(np.percentile(a["c_ex"], 95)) if a["c_ex"] else 0.0})
    return pd.DataFrame(rows), flags


def score_masked(pred: np.ndarray, S1: Surface, keep: np.ndarray | None = None) -> float:
    """EP.score's RMSE (vol points over all quotes with a finite prediction), over the kept quotes only."""
    ok = np.isfinite(pred) if keep is None else (np.isfinite(pred) & keep)
    if ok.sum() < 5:
        return float("nan")
    e = 100.0 * (pred[ok] - S1.iv[ok])
    return float(np.sqrt(np.mean(e**2)))


KINDS = {"none (all quotes, as published)": None, "a monotonic": "a", "b butterfly": "b", "c calendar": "c", "any violation": "any"}


def sensitivity(S: list[Surface], flags: dict, res_dir: Path, tol_label: str) -> pd.DataFrame:
    H = pickle.load(open(res_dir / "hyb_dhj_main.pkl", "rb"))
    test = sorted(t for t in H["carry_res"] if S[t].day >= SPLIT)
    rec = []
    for t in test:
        S1 = S[t + 1]
        preds = {"time-scaled smile": EP.pred_smile_scaled(S[t], S1), "jump model + misfit": H["carry_res"][t].astype(float)}
        f = flags[S1.day]
        row = {"t": t, "year": S1.day.year}
        for name, p in preds.items():
            for kn, key in KINDS.items():
                row[(name, kn)] = score_masked(p, S1, None if key is None else ~f[key])
        for kn, key in KINDS.items():
            row[("drop", kn)] = 0.0 if key is None else float(f[key].mean())
        rec.append(row)
    df = pd.DataFrame(rec)
    base = "none (all quotes, as published)"
    out = []
    groups = [(str(y), df[df.year == y]) for y in sorted(df.year.unique())] + [("2022-2026", df)]
    for gname, g in groups:
        for kn in KINDS:
            sm, jm = g[("time-scaled smile", kn)], g[("jump model + misfit", kn)]
            ok = sm.notna() & jm.notna()
            b_sm, b_jm = g[("time-scaled smile", base)][ok], g[("jump model + misfit", base)][ok]
            out.append({"tolerance": tol_label, "period": gname, "excluded": kn, "days_scored": int(ok.sum()),
                        "mean_share_of_quotes_excluded": float(g[("drop", kn)].mean()),
                        "smile_scaled_rmse": float(sm[ok].mean()), "jump_carry_res_rmse": float(jm[ok].mean()),
                        "diff_model_minus_benchmark": float((jm[ok] - sm[ok]).mean()),
                        "smile_scaled_rmse_all_quotes_same_days": float(b_sm.mean()),
                        "jump_carry_res_rmse_all_quotes_same_days": float(b_jm.mean()),
                        "change_smile_scaled": float((sm[ok] - b_sm).mean()), "change_jump_carry_res": float((jm[ok] - b_jm).mean()),
                        "change_in_diff": float(((jm[ok] - sm[ok]) - (b_jm - b_sm)).mean()),
                        "model_better_days_share": float((jm[ok] < sm[ok]).mean())})
    return pd.DataFrame(out)


def _md(df: pd.DataFrame, cols: list[str], pct: tuple = (), nd: int = 3) -> str:
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct:
                cells.append(f"{100 * v:.1f}%")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.{nd}f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        s += "| " + " | ".join(cells) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "dq"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = load_all()
    yrs, sens = [], []
    for label, vt in TOLERANCES.items():
        yr, flags = by_year(S, vt, label)
        yrs.append(yr)
        sens.append(sensitivity(S, flags, ROOT / "data" / "results", label))
    yr, sens = pd.concat(yrs, ignore_index=True), pd.concat(sens, ignore_index=True)
    yr.to_csv(out / "noarb_by_year.csv", index=False)
    sens.to_csv(out / "noarb_sensitivity.csv", index=False)
    pct_cols = ("share_quotes_a_monotonic", "share_quotes_b_butterfly", "share_quotes_c_calendar", "share_quotes_any", "share_surfaces_a",
                "share_surfaces_b", "share_surfaces_c", "share_surfaces_any")
    pct2 = ("share_pairs_a", "share_pairs_a_strict_zero_tolerance", "share_triples_b", "share_triples_b_strict_zero_tolerance", "share_calendar_c")
    text = f"""# No-arbitrage diagnostics (not filters)

Generated by `python3 -m dh2.noarb`. Definitions and tolerances: the module docstring of `dh2/noarb.py`. The dataset is not filtered
for any of this (DECISIONS.md, decision 4). Two tolerances are shown: **1 tick** (Rs 0.05 per comparison, the brief's) and
**1 tick + 0.25 vol pt** (each quote may also be off by 0.25 volatility points, because NSE's end-of-day option prices are
closing prints, not simultaneous quotes, and illiquid strikes may last have traded hours earlier). A quote counts as violating if it takes part in a violating pair, triple
or calendar comparison.

"""
    for label in TOLERANCES:
        y = yr[(yr.tolerance == label) & (yr.year != "ALL")].copy()
        y["year"] = y.year.astype(str)
        allr = yr[(yr.tolerance == label) & (yr.year == "ALL")].iloc[0]
        tp = sens[(sens.tolerance == label) & (sens.period == "2022-2026")]
        text += f"""## Tolerance: {label}

Share of quotes taking part in at least one violation, and share of daily surfaces with at least one violation.

{_md(y, ["year", "days", "quotes", "share_quotes_a_monotonic", "share_quotes_b_butterfly", "share_quotes_c_calendar", "share_quotes_any", "share_surfaces_a", "share_surfaces_b", "share_surfaces_c", "share_surfaces_any"], pct=pct_cols)}
Share of the individual tests that fail (pairs for (a), triples for (b), calendar comparisons for (c)); the zero-tolerance columns apply no tolerance at all:

{_md(y, ["year", "pairs_tested", "share_pairs_a", "share_pairs_a_strict_zero_tolerance", "triples_tested", "share_triples_b", "share_triples_b_strict_zero_tolerance", "calendar_comparisons", "share_calendar_c"], pct=pct2)}
Whole history ({int(allr.days)} days, {int(allr.quotes):,} quotes): {100 * allr.share_quotes_any:.1f}% of quotes and {100 * allr.share_surfaces_any:.1f}% of surfaces
have at least one violation; (a) {100 * allr.share_quotes_a_monotonic:.2f}% of quotes, (b) {100 * allr.share_quotes_b_butterfly:.2f}%, (c) {100 * allr.share_quotes_c_calendar:.2f}%.
{int(allr.ab_violations_at_call_put_junction):,} of the (a)+(b) violations straddle the call/put junction. Median size of a violation beyond
the tolerance: (a) {allr.median_excess_a_ticks:.1f} ticks, (b) {allr.median_excess_b_ticks:.1f} ticks, (c) {allr.median_excess_c_volpts:.2f} vol points (95th percentile {allr.p95_excess_c_volpts:.2f}).

Sensitivity, test period 2022-2026: next-day IV RMSE (vol points, mean over days) scored with the target day's violating quotes excluded.
Predictions and calibrations unchanged. "Change" = with exclusion minus on all quotes, same days.

{_md(tp, ["excluded", "days_scored", "mean_share_of_quotes_excluded", "smile_scaled_rmse", "jump_carry_res_rmse", "diff_model_minus_benchmark", "change_smile_scaled", "change_jump_carry_res", "change_in_diff", "model_better_days_share"], pct=("mean_share_of_quotes_excluded", "model_better_days_share"))}
"""
    text += "\nPer year: `noarb_by_year.csv`, `noarb_sensitivity.csv`.\n"
    (out / "NOARB.md").write_text(text)
    print(yr[["tolerance", "year", "days", "quotes", "share_quotes_a_monotonic", "share_quotes_b_butterfly", "share_quotes_c_calendar", "share_quotes_any", "share_surfaces_any"]].to_string())
    print(sens[sens.period == "2022-2026"].to_string())


if __name__ == "__main__":
    main()
