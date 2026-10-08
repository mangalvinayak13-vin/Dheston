"""Tests for the PINN-accelerated calibration (dh2/calib_pinn.py).  Run from the repository root:  python3 -m pytest tests/test_calib_pinn.py -q

What they pin down:
  - the surrogate objective is the exact objective with the exact price replaced by the surrogate price (residuals agree to the surrogate's
    accuracy), its autograd Jacobian is right, and the cost formula is scipy's own soft-L1 cost
  - the global stage works only inside the surrogate's box (starts and candidates), the final solution inside the calibrator's bounds
  - quotes the surrogate cannot price (tau > 0.6 y) are left out of the global stage only
  - a synthetic surface priced by the EXACT pricer from known parameters is recovered in PRICES (implied-vol points), not in parameters (the project's
    finding is that the parameters are weakly identified, so the tolerance is on prices, as in tests/test_dh2.py)
  - the polish never worsens the exact cost; the accelerated fit is never worse than the exact cost of its own starting candidates
  - the instrumented baseline is dh2.calib.fit_day (same solution) and counts pricer calls
"""
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import least_squares

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import calib, calib_pinn as CP, models as M
from dh2.data import Surface, black_iv, black_vega
from dh2.pinn.surrogate import Surrogate

U_TRUE = np.array([0.012, 0.8, 0.02, 0.4, -0.6, 0.01, 5.2, 0.015, 1.8, -0.4])      # v0_1, kappa1, theta1, xi1, rho1, v0_2, dk, theta2, xi2, rho2: inside the box
FAST = CP.PinnConfig(n_starts=32, iters=(4, 8), keep=(8,), n_polish=2, polish_nfev=12, rerank=True)


def synthetic_surface(u=U_TRUE, T_days=(21, 49, 91, 182), spot=23000.0, r=0.06) -> Surface:
    """A noise-free surface priced by the EXACT pricer (dh2.models.price) from the internal vector u: OTM quotes of four expiries."""
    T = np.array(T_days, float) / 365.0
    q = np.full(len(T), 0.012)
    F = spot * np.exp((r - q) * T)
    K, e = [], []
    for k in range(len(T)):
        ks = np.round(F[k] * np.exp(np.linspace(-0.10, 0.10, 17)) / 50.0) * 50.0
        K += list(ks); e += [k] * len(ks)
    K, e = np.array(K), np.array(e)
    is_call = K >= F[e]
    n = len(K)
    S0 = Surface(date(2025, 3, 20), spot, r, T, F, q, [date(2025, 3, 20)] * len(T), K, e, is_call, np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n), np.ones(n), "file")
    px = M.price("dh", calib.to_phys("dh", u), S0)
    iv = black_iv(px, F[e], K, T[e], r, is_call)
    vega = black_vega(F[e], K, T[e], r, iv) / 100.0
    return replace(S0, price=px, iv=iv, vega=vega)


@pytest.fixture(scope="module")
def sur():
    return Surrogate.load(device="cpu")


@pytest.fixture(scope="module")
def surf():
    return synthetic_surface()


@pytest.fixture(scope="module")
def idx(surf):
    return M.thin(surf, 14)


# ------------------------------------------------------------------ the objective
def test_cost_formula_is_scipys_soft_l1():
    r = np.array([0.1, -0.7, 2.5, -6.0, 0.0])
    rr = least_squares(lambda x: r + 0.0 * x[0], [0.0], loss="soft_l1", f_scale=calib.F_SCALE, max_nfev=1)
    assert abs(rr.cost - CP.soft_l1_cost(r)) < 1e-12


def test_surrogate_residuals_match_the_exact_residuals(sur, surf, idx):
    obj = CP.SurrogateObjective(sur, surf, idx)
    assert obj.n == len(idx) and obj.n_excluded == 0
    lo, hi = CP.box_bounds(sur)
    U = CP.sobol_starts(surf, lo, hi, 8, 0, "box")
    r, J = obj.eval(U, jac=False)
    for b in range(4):
        d = r[b] - CP.exact_residual(U[b], surf, idx)
        assert np.sqrt(np.mean(d ** 2)) < 0.5      # vol points: the surrogate's accuracy away from the typical region, not the 0.05 of typical days
    r0, _ = obj.eval(U_TRUE[None], jac=False)
    assert np.sqrt(np.mean(r0[0] ** 2)) < 0.3      # the exact residual there is 0, so this is the surrogate's own error at the true parameters


def test_autograd_jacobian_is_the_derivative_of_the_surrogate_residual(sur, surf, idx):
    obj = CP.SurrogateObjective(sur, surf, idx)
    u = U_TRUE.copy()
    _, J = obj.eval(u[None])
    rng = np.random.default_rng(3)
    for _ in range(3):
        d = rng.normal(size=10) * (np.array(calib.BOUNDS["dh"][1]) - np.array(calib.BOUNDS["dh"][0])) * 0.01
        rp, _ = obj.eval((u + d)[None], jac=False)
        rm, _ = obj.eval((u - d)[None], jac=False)
        fd = (rp[0] - rm[0]) / 2.0
        ad = J[0] @ d
        assert np.linalg.norm(ad - fd) < 0.08 * np.linalg.norm(fd) + 1e-3, (np.linalg.norm(ad - fd), np.linalg.norm(fd))


def test_batched_lm_lowers_the_surrogate_cost_and_respects_bounds(sur, surf, idx):
    obj = CP.SurrogateObjective(sur, surf, idx)
    lo, hi = CP.box_bounds(sur)
    U0 = CP.sobol_starts(surf, lo, hi, 16, 1, "data")
    c0 = CP._robust(obj.eval(U0, jac=False)[0])[0]
    res = CP.batched_lm(obj, U0, lo, hi, 15)
    assert np.all(res["cost"] <= c0 + 1e-9) and np.mean(res["cost"] < 0.5 * c0) > 0.5
    assert np.all(res["U"] >= lo - 1e-12) and np.all(res["U"] <= hi + 1e-12)


# ------------------------------------------------------------------ bounds and the box
def test_starts_and_candidates_stay_inside_the_surrogate_box(sur, surf, idx):
    lo, hi = CP.box_bounds(sur)
    clo, chi = (np.array(b) for b in calib.BOUNDS["dh"])
    assert np.all(lo >= clo - 1e-12) and np.all(hi <= chi + 1e-12)                  # the box is inside the calibrator's bounds
    for mode in ("data", "box"):
        U = CP.sobol_starts(surf, lo, hi, 64, 5, mode)
        assert np.all(U >= lo - 1e-12) and np.all(U <= hi + 1e-12) and CP.in_box(U, sur).all()
    p = CP.propose(sur, surf, idx, FAST)
    assert np.all(p["U"] >= lo - 1e-9) and np.all(p["U"] <= hi + 1e-9) and CP.in_box(p["U"], sur).all()
    assert CP.in_box(p["u_sur"], sur).all()


def test_final_solution_is_inside_the_calibrators_bounds(sur, surf):
    f = CP.fit_day_pinn(sur, surf, FAST)
    clo, chi = (np.array(b) for b in calib.BOUNDS["dh"])
    assert np.all(f["u"] >= clo - 1e-12) and np.all(f["u"] <= chi + 1e-12)
    assert f["u"][6] >= 0.2 - 1e-12                                              # kappa2 = kappa1 + dk with dk >= 0.2: the factors cannot swap


def test_quotes_beyond_the_surrogates_range_are_left_out_of_the_global_stage_only(sur):
    s = synthetic_surface(T_days=(21, 91, 330))                                    # 330 days = 0.904 y > 0.6 y
    ix = M.thin(s, 14)
    obj = CP.SurrogateObjective(sur, s, ix)
    far = s.T[s.e[ix]] > sur.domain["tau_hi"]
    assert far.any() and obj.n_excluded == int(far.sum()) and obj.n == int((~far).sum())
    assert np.array_equal(obj.ok, ~far)
    # the exact objective (polish, scoring) still uses every quote of idx
    assert CP.exact_residual(U_TRUE, s, ix).shape == (len(ix),)


# ------------------------------------------------------------------ recovery, polish
def test_recovers_prices_of_a_known_surface_priced_by_the_exact_pricer(sur, surf, idx):
    assert np.max(np.abs(CP.exact_residual(U_TRUE, surf, idx))) < 1e-6              # the truth fits perfectly
    f = CP.fit_day_pinn(sur, surf, FAST)
    sc = CP.score(f["u"], surf, idx)
    assert sc["rmse"] < 0.15                       # implied-vol points over every quote (same tolerance as tests/test_dh2.py); parameters are NOT asserted
    err = M.iv_error("dh", calib.to_phys("dh", f["u"]), surf)
    assert np.max(np.abs(err)) < 0.6
    price_err = M.price("dh", calib.to_phys("dh", f["u"]), surf) - surf.price
    assert np.max(np.abs(price_err) / np.maximum(surf.vega, calib.VEGA_FLOOR)) < 0.6   # vega-weighted price error = implied-vol error to first order
    assert f["cost"] < 0.05 and f["n_exact_calls"] > 0 and f["t_total"] > 0


def test_polish_never_worsens_the_exact_cost(surf, idx):
    rng = np.random.default_rng(11)
    lo, hi = (np.array(b) for b in calib.BOUNDS["dh"])
    for _ in range(4):
        u0 = calib.clip("dh", lo + rng.uniform(0.02, 0.9, 10) * (hi - lo))
        c0 = CP.exact_cost(u0, surf, idx)
        p = CP.polish(surf, idx, u0, 5)
        assert p["cost"] <= c0 + 1e-9 and p["n_calls"] >= 1
        assert abs(p["cost"] - CP.exact_cost(p["u"], surf, idx)) < 1e-9             # the reported cost is the exact cost of the reported point


def test_accelerated_fit_is_never_worse_than_its_starting_candidates(sur, surf, idx):
    f = CP.fit_day_pinn(sur, surf, FAST)
    start_costs = [CP.exact_cost(u, surf, idx) for u in f["starts"]]
    assert f["cost"] <= min(start_costs) + 1e-9
    assert abs(f["cost"] - CP.exact_cost(f["u"], surf, idx)) < 1e-9


def test_defect_correction_never_returns_a_point_worse_than_its_start(sur, surf, idx):
    cfg = replace(FAST, n_correct=2)
    g = CP.global_search(sur, surf, idx, cfg)
    c_before = np.array([CP.exact_cost(u, surf, idx) for u in g["U"]])
    cr = CP.correct(sur, surf, idx, g["obj"], g["U"], cfg)
    assert cr["n_calls"] >= len(g["U"]) and np.all(np.diff(cr["cost"]) >= -1e-12)     # sorted by exact cost
    assert cr["cost"][0] <= c_before.min() + 1e-9
    lo, hi = CP.box_bounds(sur)
    assert np.all(cr["U"] >= lo - 1e-9) and np.all(cr["U"] <= hi + 1e-9)


# ------------------------------------------------------------------ the baseline
def test_instrumented_baseline_equals_calib_fit_day(surf):
    a = CP.fit_day_exact(surf, n_starts=2, seed=4, max_nfev=15)
    b = calib.fit_day("dh", surf, n_starts=2, seed=4, max_nfev=15)
    assert abs(a["cost"] - b["cost"]) < 1e-9 and np.allclose(a["u"], b["u"])
    assert a["n_exact_calls"] >= sum(r["nfev"] for r in a["runs"]) and a["t_total"] > 0
