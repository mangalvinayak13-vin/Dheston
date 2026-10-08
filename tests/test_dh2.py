"""Tests for Double Heston 2.0. Run from the repository root:  python3 -m pytest tests -q   (add -m slow for the long ones)

What they pin down:
  - the project's pricer is the one in use, unchanged, and still gives the site's reference price
  - the models behave in their limits (Double Heston -> Black-Scholes; jumps off -> Double Heston)
  - the data pipeline recovers a known forward and rate from synthetic put-call-parity prices
  - nothing looks ahead: the rate series, the pattern matcher and the walk-forward predictions at day t do not depend on later days
  - the calibrator recovers PRICES from a synthetic surface (the parameters themselves are not asserted: the project's finding is
    that they are weakly identified)
"""
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import calib, data, memory, models as M, struct as ST

PRICER = Path(__file__).resolve().parents[1] / "pricer" / "models.py"
SITE = dict(S=23140.5, K=23150.0, T=32 / 365, r=0.053199742777582895, q=-0.007317007626223504)
DEFAULT = [0.02, 0.5, 0.02, 0.3, -0.7, 0.02, 5.0, 0.02, 0.5, -0.7]


def synthetic_day(day=date(2025, 3, 20), F0=23000.0, r=0.06, params=DEFAULT, jumps=None, n_expiries=4):
    """A noise-free option file for one day, priced by the model with a known forward and rate: returns the DataFrame the loader reads."""
    rows = []
    for k in range(n_expiries):
        xp = day + timedelta(days=14 * (k + 2))
        T = (xp - day).days / 365
        F = F0 * math.exp(0.02 * T)
        q = r - math.log(F / F0) / T
        strikes = np.arange(F0 * 0.9 // 50 * 50, F0 * 1.1, 50.0)
        call = np.atleast_1d(M.P.double_heston_price(F0, strikes, T, r, *params, kind="call", q=q))
        put = call - np.exp(-r * T) * (F - strikes)
        for K, c, p in zip(strikes, call, put):
            for typ, px in (("CE", c), ("PE", p)):
                rows.append({"TradDt": day.isoformat(), "FinInstrmTp": "IDO", "TckrSymb": "NIFTY", "XpryDt": xp.isoformat(), "StrkPric": f"{K:.2f}",
                             "OptnTp": typ, "ClsPric": f"{max(px, 0.05):.2f}", "UndrlygPric": f"{F0:.2f}", "TtlTradgVol": "100", "OpnIntrst": "1000"})
    return pd.DataFrame(rows), F0, r


# ------------------------------------------------------------------ the pricer
def test_pricer_is_the_projects_own_and_gives_the_sites_reference_price():
    assert Path(M.P.__file__) == PRICER
    price = M.P.double_heston_price(SITE["S"], SITE["K"], SITE["T"], SITE["r"], *DEFAULT, kind="call", q=SITE["q"])
    assert abs(price - 600.40) < 0.01   # the number the live sites show for NIFTY 23150 on 25 Sep 2026 at the default settings


def test_double_heston_collapses_to_black_scholes_without_vol_of_vol():
    S, _, _ = synthetic_day()
    sig = 0.15
    x = [sig**2 / 2, 1.0, sig**2 / 2, 1e-3, 0.0, sig**2 / 2, 5.0, sig**2 / 2, 1e-3, 0.0]
    surf = _surface_from(S)
    dh = M.price("dh", x, surf)
    bs = M.price("bs", [sig], surf)
    near = np.abs(np.log(surf.K / surf.F[surf.e])) < 0.1
    assert np.max(np.abs(dh[near] - bs[near])) < 0.01


def test_jumps_switched_off_give_double_heston():
    surf = _surface_from(synthetic_day()[0])
    base = M.price("dh", DEFAULT, surf)
    assert np.max(np.abs(M.price("dhj", DEFAULT + [1e-9, -0.05, 0.05], surf) - base)) < 1e-4


def test_jumps_make_the_left_wing_dearer():
    surf = _surface_from(synthetic_day()[0])
    iv0 = M.model_iv(M.price("dhj", DEFAULT + [1e-9, -0.05, 0.05], surf), surf)
    iv1 = M.model_iv(M.price("dhj", DEFAULT + [1.0, -0.10, 0.10], surf), surf)
    puts = ~surf.is_call
    assert iv1[puts].mean() > iv0[puts].mean()


# ------------------------------------------------------------------ the data pipeline
def _read(df):
    o = df[(df.TckrSymb == "NIFTY") & (df.FinInstrmTp == "IDO")].copy()
    o["K"] = pd.to_numeric(o.StrkPric); o["px"] = pd.to_numeric(o.ClsPric); o["vol"] = pd.to_numeric(o.TtlTradgVol)
    o["oi"] = pd.to_numeric(o.OpnIntrst); o["und"] = pd.to_numeric(o.UndrlygPric)
    o["xp"] = pd.to_datetime(o.XpryDt).dt.date; o["call"] = o.OptnTp == "CE"
    return o


def _surface_from(df, r=0.06):
    return data._surface(date(2025, 3, 20), _read(df), r)


def test_parity_recovers_the_forward_and_the_rate():
    df, F0, r = synthetic_day()
    o = _read(df)
    got_r = data.implied_rate(o, date(2025, 3, 20))
    assert abs(got_r - r) < 0.002
    surf = data._surface(date(2025, 3, 20), o, got_r)
    for e in range(len(surf.T)):
        assert abs(surf.F[e] / (F0 * math.exp(0.02 * surf.T[e])) - 1) < 2e-4


def test_implied_rate_series_uses_only_the_past():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(40)]
    raw = np.random.default_rng(0).uniform(0.05, 0.12, 40)
    a = data.smooth_rates(days, raw)
    raw2 = raw.copy(); raw2[25:] = 0.5       # change the future
    b = data.smooth_rates(days, raw2)
    assert np.allclose(a[:25], b[:25])


def test_only_traded_out_of_the_money_quotes_are_kept():
    df, _, _ = synthetic_day()
    df.loc[df.index[:40], "TtlTradgVol"] = "0"      # untraded contracts carry stale closes
    surf = _surface_from(df)
    otm = (surf.is_call & (surf.K >= surf.F[surf.e])) | (~surf.is_call & (surf.K < surf.F[surf.e]))
    assert otm.all() and (surf.volume > 0).all()


# ------------------------------------------------------------------ nothing looks ahead
def test_pattern_matcher_ignores_the_future():
    rng = np.random.default_rng(1)
    feat = rng.normal(size=(500, len(memory.FEATURES)))
    t = 400
    a = memory.Memory(feat, k=8, q=0.1).match(t)
    feat2 = feat.copy(); feat2[t + 1:] = 1e6
    b = memory.Memory(feat2, k=8, q=0.1).match(t)
    assert (a is None) == (b is None)
    if a is not None:
        assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


def test_matched_stretches_have_a_known_aftermath():
    rng = np.random.default_rng(2)
    feat = rng.normal(size=(500, len(memory.FEATURES)))
    res = memory.Memory(feat, k=8, q=0.5).match(300)
    assert res is not None and (res[0] + 1 <= 300).all()   # the day after each voting stretch happened by day t


# ------------------------------------------------------------------ calibration
@pytest.mark.slow
def test_calibration_recovers_prices_from_a_synthetic_surface():
    df, _, _ = synthetic_day(params=[0.012, 0.8, 0.02, 0.4, -0.6, 0.01, 6.0, 0.015, 0.7, -0.4])
    surf = _surface_from(df)
    f = calib.fit_day("dh", surf, n_starts=3, max_nfev=80)
    assert f["rmse"] < 0.15        # volatility points: the surface is reproduced (the parameters themselves are not asserted)


def test_expected_state_reverts_toward_the_long_run_level():
    eta = np.array([2.0, 0.04, 0.3, -0.5, 5.0, 0.02, 0.5, -0.5])
    v = np.array([0.08, 0.05])
    nxt = ST.expected_state("dh", eta, v, 1 / 252)
    assert 0.04 < nxt[0] < 0.08 and 0.02 < nxt[1] < 0.05


@pytest.mark.slow
def test_walk_forward_refits_on_schedule_and_uses_only_the_past():
    """The driver must refit the structural parameters every `refit` days (a bug once froze them for ~200 days), and a day's recorded
    forecast must not change if later surfaces are replaced."""
    import run_walk
    if not list((Path(__file__).resolve().parents[1] / "data" / "nse_index").glob("2025-0[234]-*.csv")):
        pytest.skip("needs the NSE files for Feb-Apr 2025 (python3 -m dh2.fetch_nse)")
    S = data.load_all("2025-02-03", "2025-04-30", workers=2, cache=False)
    kw = dict(W=10, R=5, prior_lam=0.0, nfev_cold=15, nfev_warm=4, seed=1, per_expiry=6, burn=5)
    a, b = 20, 45
    out = run_walk.worker("heston", S, a, b, **kw)
    etas = {tuple(np.round(r["eta"], 8)) for r in out}
    assert len(etas) >= (b - a) // 5 - 1, f"only {len(etas)} distinct structural vectors in {b - a} days with a refit every 5"
    S2 = list(S)
    for i in range(31, len(S2)):          # replace every surface after day index 30
        S2[i] = S[29]
    out2 = run_walk.worker("heston", S2, a, 31, **kw)
    first = {r["t"]: r for r in out}
    for r in out2:
        if r["t"] <= 29:                  # a forecast made on day t uses days <= t and predicts t+1 <= 30, which is untouched
            assert np.allclose(r["eta"], first[r["t"]]["eta"]) and np.allclose(r["state"], first[r["t"]]["state"])
            assert np.allclose(r["pred_carry"], first[r["t"]]["pred_carry"])
