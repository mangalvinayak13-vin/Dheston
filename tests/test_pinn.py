"""Tests for the Double Heston surrogate (dh2/pinn). Run from the repository root:  python3 -m pytest tests/test_pinn.py -q   (add -m slow for the long one)

What they pin down:
  - the normalised PDE is satisfied by EXACT prices of the project's pricer (finite differences), and the check can fail: with the cross-term sign
    flipped the same check is off by a large factor
  - the normalised price u = C/(K DF) does not depend on r and q (the pricer works from the forward)
  - the network's initial condition holds exactly, its Black part is the Black price, shapes and gradients flow
  - the labels' independent COS pricer agrees with the project pricer; the domain is built from the development period only
  - the inference API (price_normalized in numpy and torch, price_surface, greeks) works on the shipped checkpoint
  - (slow) held-out accuracy of the shipped checkpoint against the exact pricer on real held-out chains
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2.pinn import data as D, model as MD, pde as PDE
from dh2.pinn.surrogate import Surrogate

CK = Path(__file__).resolve().parents[1] / "results" / "pinn" / "checkpoints" / "final.pt"
THETA = np.array([0.8, 0.03, 0.4, -0.6, 6.0, 0.01, 1.8, -0.4])      # kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2
V1, V2 = 0.012, 0.015


@pytest.fixture(scope="module")
def dom():
    if not (Path(__file__).resolve().parents[1] / "data" / "results" / "walk_dh_main.pkl").exists():
        pytest.skip("needs data/results/walk_dh_main.pkl (not stored in git; see results/PIPELINE.md)")
    return D.build_domain()


def _net(dom, **kw):
    torch.manual_seed(0)
    return MD.DHNet(D.net_domain(dom), width=32, depth=3, **kw)


# ------------------------------------------------------------------ the PDE
@pytest.mark.parametrize("tau,xr", [(0.05, 0.5), (0.2, -0.8), (0.4, 1.0)])
def test_pde_holds_for_exact_prices_and_cross_term_sign_matters(tau, xr):
    sd = float(np.sqrt(D.total_variance(THETA, V1, V2, tau)))
    f = 0.125
    r = PDE.verify_pde_fd(THETA, V1, V2, tau, xr * sd, steps=dict(hx=f * sd, h1=f * V1, h2=f * V2, ht=0.4 * f * tau))
    rel = abs(r["residual"]) / r["scale"]
    t = r["terms"]
    flipped = abs(r["ut"] - (t["diffusion"] + t["drift1"] + t["drift2"] + t["volvol1"] + t["volvol2"] - t["cross1"] - t["cross2"])) / r["scale"]
    assert rel < 1e-3, rel                         # exact prices satisfy the PDE (finite-difference error only)
    assert flipped > 100 * rel and flipped > 0.02  # a sign error in the mixed term would be seen


def test_pde_residual_shrinks_with_the_step():
    sd = float(np.sqrt(D.total_variance(THETA, V1, V2, 0.3)))
    rel = []
    for f in (0.25, 0.125):
        r = PDE.verify_pde_fd(THETA, V1, V2, 0.3, 0.3 * sd, steps=dict(hx=f * sd, h1=f * V1, h2=f * V2, ht=0.4 * f * 0.3))
        rel.append(abs(r["residual"]) / r["scale"])
    assert rel[1] < rel[0] / 4                     # high-order stencil: the residual is discretisation error, not a model error


def test_normalised_price_is_independent_of_r_and_q():
    P = D._project_pricer()
    S, T = 100.0, 0.3
    args = D.theta_to_pricer(THETA, V1, V2)
    out = []
    for r, q in ((0.0, 0.0), (0.07, 0.01), (0.11, -0.02)):
        F = S * np.exp((r - q) * T)
        K = F * np.exp(np.array([-0.1, 0.0, 0.08]))                # same x = ln(F/K) every time
        c = np.atleast_1d(P.double_heston_price(S, K, T, r, *args, kind="call", q=q))
        out.append(c / (K * np.exp(-r * T)))
    assert np.max(np.abs(out[1] - out[0])) < 1e-8 and np.max(np.abs(out[2] - out[0])) < 1e-8


# ------------------------------------------------------------------ the network
def test_initial_condition_is_exact_and_black_limit(dom):
    m = _net(dom)
    x = torch.linspace(-0.3, 0.3, 41)
    v = torch.full_like(x, 0.02); th = torch.tensor(THETA, dtype=torch.float32).expand(41, 8)
    u0 = m(x, v, v, torch.zeros_like(x), th)
    assert torch.equal(u0, torch.relu(torch.exp(x) - 1.0))          # not approximately: exactly the payoff
    # with the network switched off the output is the Black price at the expected total variance
    m0 = _net(dom, out_scale=0.0)
    tau = torch.full_like(x, 0.2)
    w = MD.total_variance(x, v, v, tau, th)
    ref = D.bs_unit(x.numpy().astype(np.float64), w.numpy().astype(np.float64))
    assert np.max(np.abs(m0(x, v, v, tau, th).detach().numpy() - ref)) < 1e-6


def test_black_limit_of_the_exact_model():
    """Vanishing vol-of-vol and flat variance: the exact pricer collapses to Black at w = sum theta tau (project pricer, normalised)."""
    th = np.array([1.0, 0.02, 1e-3, 0.0, 5.0, 0.02, 1e-3, 0.0])
    x = np.array([[-0.1, 0.0, 0.1]])
    u = D.pricer_call(th[None], np.array([0.02]), np.array([0.02]), np.array([0.25]), x)
    assert np.max(np.abs(u - D.bs_unit(x, 0.04 * 0.25 * np.ones_like(x)))) < 1e-5


def test_shapes_and_gradient_flow(dom):
    m = _net(dom, g="vegat")
    rng = np.random.default_rng(0)
    a = D.sample_inputs(64, dom, rng)
    b = {k: torch.tensor(v, dtype=torch.float32) for k, v in a.items()}
    nr, res, d = PDE.normalised_residual(m, b["x"], b["v1"], b["v2"], b["tau"], b["theta"])
    assert nr.shape == (64,) and all(v.shape == (64,) for v in d.values())
    assert all(torch.isfinite(v).all() for v in d.values())
    (nr ** 2).mean().backward()
    g = [p.grad for p in m.parameters()]
    assert all(t is not None and torch.isfinite(t).all() for t in g) and sum(float(t.abs().sum()) for t in g) > 0


# ------------------------------------------------------------------ labels and domain
def test_cos_labels_match_the_project_pricer(dom):
    rng = np.random.default_rng(3)
    th, a, b = D.sample_params(12, dom, rng, p_emp=0.5)
    ta = D.sample_tau(12, rng)
    x = np.stack([D.sample_x(th, a, b, ta, rng) for _ in range(5)], 1)
    ref = D.pricer_call(th, a, b, ta, x)
    assert np.max(np.abs(D.cos_call(th, a, b, ta, x, N=2048, L=12.0) - ref)) < 1e-6


def test_domain_uses_the_development_period_only(dom):
    assert max(dom["train_days"] + dom["ho_days"]) < "2022-01-01" and min(dom["test_days"]) >= "2022-01-01"
    assert max(dom["train_days"]) < min(dom["ho_days"])                 # the held-out block is the LAST 20% of the development days
    assert np.all(D.in_domain(dom, np.vstack([dom["train_eta"], dom["ho_eta"]]), np.vstack([dom["train_state"], dom["ho_state"]])))   # every dev fit lies in the box
    rng = np.random.default_rng(1)
    for src in ("train", "holdout"):
        th, a, b = D.sample_params(500, dom, rng, p_emp=0.5, source=src)
        lo, hi = D.domain_theta_bounds(dom)
        assert np.all(th >= lo - 1e-9) and np.all(th <= hi + 1e-9) and a.min() >= D.V_MIN and b.max() <= D.V_MAX
    inp = D.sample_inputs(2000, dom, rng)
    assert np.abs(inp["x"]).max() <= D.X_MAX and inp["tau"].min() >= D.TAU_MIN and inp["tau"].max() <= D.TAU_MAX


# ------------------------------------------------------------------ inference API (needs the shipped checkpoint)
@pytest.mark.skipif(not CK.exists(), reason="checkpoint not trained yet")
def test_surrogate_api_numpy_torch_surface_and_greeks():
    s = Surrogate.load(CK, "cpu")
    x = np.linspace(-0.15, 0.15, 9)
    u = s.price_normalized(x, V1, V2, 0.1, THETA)
    assert u.shape == (9,) and u.dtype == np.float64 and np.all(u >= np.maximum(np.exp(x) - 1, 0) - 1e-3)
    ut = s.price_normalized(torch.tensor(x, requires_grad=True), V1, V2, 0.1, THETA)
    assert isinstance(ut, torch.Tensor) and np.allclose(ut.detach().numpy(), u, atol=1e-6)
    ut.sum().backward()
    F, K = 25000.0, 25000.0 * np.exp(-x)
    out = s.price_surface(F, K, 0.1, np.exp(-0.06 * 0.1), THETA, V1, V2)
    assert out["call"].shape == (9,) and out["in_domain"].all() and np.isfinite(out["iv"]).all()
    assert np.allclose(out["put"], out["call"] - np.exp(-0.006) * (F - K))
    g = s.greeks(np.full(9, F), K, np.full(9, 0.1), np.full(9, np.exp(-0.006)), THETA, V1, V2, S=F)
    assert np.all((g["delta_S"] > 0) & (g["delta_S"] < 1.01)) and np.all(g["gamma_S"] > -1e-3 * g["gamma_S"].max())


@pytest.mark.skipif(not CK.exists(), reason="checkpoint not trained yet")
def test_surrogate_is_close_to_exact_on_a_few_held_out_points(dom):
    rng = np.random.default_rng(99)
    th, a, b = D.sample_params(60, dom, rng, p_emp=0.0)
    ta = D.sample_tau(60, rng)
    x = D.sample_x(th, a, b, ta, rng)
    ref = D.pricer_call(th, a, b, ta, x.reshape(-1, 1))[:, 0]
    s = Surrogate.load(CK, "cpu")
    u = s.price_normalized(x, a, b, ta, th)
    otm_ref = D.exact_otm(ref, x)
    keep = otm_ref >= D.P_MIN
    iv_ref = D.implied_vol_unit(np.maximum(otm_ref, 1e-14), x, ta)
    iv = D.implied_vol_unit(np.maximum(D.exact_otm(u, x), 1e-14), x, ta)
    assert np.nanmean(np.abs(100 * (iv - iv_ref))[keep]) < 0.5     # vol points; the measured mean on 10k held-out points is 0.08


@pytest.mark.slow
@pytest.mark.skipif(not CK.exists(), reason="checkpoint not trained yet")
def test_heldout_real_chain_error_of_the_shipped_model():
    from dh2.pinn import validate as V
    from dh2.pinn import train as TR
    cfg = TR.load_cfg(); d = TR.get_domain(cfg)
    ref = V._real_set(d, "ho", stride=10)
    ev = V.evaluate_model(CK, {"real": ref}, "cpu", names=("real",))["real"]["summary"]
    assert ev["mean"] < 0.1 and ev["p95"] < 0.3
