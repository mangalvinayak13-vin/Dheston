"""Unit tests for dh2/stats.py on synthetic data (no market data needed). Run:  python3 -m pytest tests/test_stats.py -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import stats as ST


def ar1(T, phi, rng, sigma=1.0):
    e = rng.normal(0, sigma, T + 200)
    x = np.zeros(T + 200)
    for t in range(1, T + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def test_dm_lag0_hln_equals_one_sample_t_test():
    """h=1, no HAC lags, HLN correction == the ordinary one-sample t-test on the differences (statistic and p-value)."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 0.3, 80), rng.normal(1.1, 0.3, 80)
    r = ST.diebold_mariano(a, b, h=1, lags=0, hln=True)
    t = sst.ttest_1samp(a - b, 0.0)
    assert r["dm_stat"] == pytest.approx(t.statistic, rel=1e-10)
    assert r["pvalue"] == pytest.approx(t.pvalue, rel=1e-10)
    assert r["mean_diff"] == pytest.approx(np.mean(a - b))


def test_hln_correction_factor_for_longer_horizon():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 60), rng.normal(0.2, 1, 60)
    h, T = 3, 60
    raw = ST.diebold_mariano(a, b, h=h, lags=h - 1, hln=False)
    cor = ST.diebold_mariano(a, b, h=h, lags=h - 1, hln=True)
    assert cor["dm_stat"] == pytest.approx(raw["dm_stat"] * math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T), rel=1e-12)


def test_newey_west_matches_hand_computation():
    d = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.1, 0.4, -0.7])
    x = d - d.mean()
    T = d.size
    g = lambda j: float(x[j:] @ x[:T - j]) / T
    expect = g(0) + 2 * (1 - 1 / 3) * g(1) + 2 * (1 - 2 / 3) * g(2)
    assert ST.newey_west_lrv(d, 2) == pytest.approx(expect, rel=1e-12)
    assert ST.newey_west_lrv(d, 0) == pytest.approx(g(0), rel=1e-12)


def test_long_run_variance_of_ar1():
    """AR(1) with phi=0.5, unit innovations: long-run variance 1/(1-phi)^2 = 4 (the plain variance is only 1.33)."""
    rng = np.random.default_rng(3)
    x = ar1(200_000, 0.5, rng)
    assert ST.newey_west_lrv(x, 40) == pytest.approx(4.0, rel=0.06)
    assert ST.newey_west_lrv(x, 0) == pytest.approx(4.0 / 3.0, rel=0.05)


def test_dm_size_under_the_null_iid():
    rng = np.random.default_rng(4)
    rej = np.mean([ST.diebold_mariano(rng.normal(0, 1, 250), rng.normal(0, 1, 250))["pvalue"] < 0.05 for _ in range(600)])
    assert 0.025 < rej < 0.085


def test_hac_controls_size_when_losses_are_autocorrelated():
    """Persistent daily losses: the naive (lags=0) test rejects a true null far too often, the Newey-West version much less."""
    rng = np.random.default_rng(5)
    naive, hac = [], []
    for _ in range(400):
        d = ar1(500, 0.6, rng)                     # the loss differential under the null (mean 0)
        naive.append(ST.diebold_mariano(d, np.zeros_like(d), lags=0)["pvalue"] < 0.05)
        hac.append(ST.diebold_mariano(d, np.zeros_like(d))["pvalue"] < 0.05)
    naive, hac = np.mean(naive), np.mean(hac)
    assert naive > 0.15                            # about 0.25 expected
    assert hac < naive - 0.08 and hac < 0.12


def test_power_against_a_real_difference():
    rng = np.random.default_rng(6)
    hits_dm, hits_w = [], []
    for _ in range(100):
        base = rng.normal(1.0, 0.3, 300)
        better = base - 0.12 + rng.normal(0, 0.15, 300)
        hits_dm.append(ST.diebold_mariano(better, base)["pvalue"] < 0.05)
        hits_w.append(ST.wilcoxon_signed_rank(better, base)["pvalue"] < 0.05)
    assert np.mean(hits_dm) > 0.95 and np.mean(hits_w) > 0.95


def test_antisymmetry_and_one_sided_p_values():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 100), rng.normal(0.3, 1, 100)
    ab, ba = ST.diebold_mariano(a, b), ST.diebold_mariano(b, a)
    assert ab["dm_stat"] == pytest.approx(-ba["dm_stat"]) and ab["pvalue"] == pytest.approx(ba["pvalue"])
    lo, hi = ST.diebold_mariano(a, b, alternative="less")["pvalue"], ST.diebold_mariano(a, b, alternative="greater")["pvalue"]
    assert lo + hi == pytest.approx(1.0)
    assert lo < 0.05 < hi                                 # a has the lower mean
    assert ST.diebold_mariano(a, b)["pvalue"] == pytest.approx(2 * min(lo, hi))
    wa, wb = ST.wilcoxon_signed_rank(a, b), ST.wilcoxon_signed_rank(b, a)
    assert wa["pvalue"] == pytest.approx(wb["pvalue"])


def test_wilcoxon_wrapper_matches_scipy_and_handles_nan_and_ties():
    rng = np.random.default_rng(8)
    a, b = rng.normal(0, 1, 120), rng.normal(0.25, 1, 120)
    ref = sst.wilcoxon(a, b)
    r = ST.wilcoxon_signed_rank(a, b)
    assert r["statistic"] == pytest.approx(ref.statistic) and r["pvalue"] == pytest.approx(ref.pvalue)
    assert r["share_a_lower"] == pytest.approx(np.mean(a < b))
    a2, b2 = a.copy(), b.copy()
    a2[3] = np.nan; b2[10] = np.nan
    assert ST.wilcoxon_signed_rank(a2, b2)["n"] == 118
    same = ST.wilcoxon_signed_rank(a, a)                  # identical series: no evidence, no crash
    assert same["pvalue"] == 1.0 and same["n_nonzero"] == 0
    assert ST.diebold_mariano(a2, b2)["n"] == 118


def test_dm_on_identical_losses_is_not_significant_and_does_not_crash():
    x = np.linspace(1, 2, 50)
    r = ST.diebold_mariano(x, x)
    assert math.isnan(r["dm_stat"]) or r["pvalue"] > 0.99 or r["lrv"] == 0


def test_holm_adjustment():
    adj = ST.holm([0.01, 0.04, 0.03, 0.5])
    # sorted p: 0.01 (x4)=0.04, 0.03 (x3)=0.09, 0.04 (x2)=0.08 -> raised to 0.09 (monotone), 0.5 (x1)=0.5
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert np.all(ST.holm([0.2, 0.9]) >= np.array([0.2, 0.9]))


def test_nw_auto_lags():
    assert ST.nw_auto_lags(100) == 4
    assert ST.nw_auto_lags(1173) == int(math.floor(4 * (11.73) ** (2 / 9))) == 6      # 6.9 rounded down
