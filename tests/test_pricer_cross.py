"""Independent cross-checks of the Double Heston pricers (work package A of the 2.0 upgrade).

    python3 -m pytest tests/test_pricer_cross.py -q            # fast ones, well under a minute
    python3 -m pytest tests/test_pricer_cross.py -q -m slow    # Monte Carlo (about a minute)

Four pricers meet here:
  dh2/cos.py     COS (Fang-Oosterlee), adaptive terms, cumulant truncation        <- the fast independent ground truth
  dh2/lewis.py   Lewis single integral with adaptive quadrature                   <- slow second method
  dh2/mc.py      Monte Carlo, Andersen QE                                         <- no characteristic function at all
  pricer/models.py  the project's pricer (midpoint-rule Gil-Pelaez), loaded the way dh2/models.py loads it   <- the one being checked
The characteristic function in dh2/cos.py is written independently of pricer/models.py and is itself checked against mpmath (test a).

Tolerances were fixed before the project pricer was compared (they are what a calibration that fits to ~0.3-1 vol points can tolerate):
  COS vs Lewis                  |du| <= 1e-6 u + 1e-11   (u = price / (K DF))
  project pricer vs COS         implied vol within 0.01 vol point and price within 1e-6 relative, for the quotes the calibration uses
                                (>= 3 days, |ln K/F| <= 0.22, option worth >= Rs 1 on NIFTY ~ 25000)
No QuantLib comparison: QuantLib is not installed in this environment.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dh2 import cos, data, lewis, mc                                     # noqa: E402
from dh2 import models as M                                               # noqa: E402  (only for the jump log-CF convention)

# the project's pricer, loaded exactly the way dh2/models.py loads it
_spec = importlib.util.spec_from_file_location("dh_pricer_cross", ROOT / "pricer" / "models.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

# ------------------------------------------------------------------------------------------------ parameter sets
# (v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2); the dated ones are calibrated NIFTY days (outputs/parameters_dhj_main.csv)
DEFAULT = [0.02, 0.5, 0.02, 0.3, -0.7, 0.02, 5.0, 0.02, 0.5, -0.7]                                      # the website's default settings
CALM_2017 = [0.004075, 0.136359, 0.209799, 0.417901, -0.619337, 0.006622, 15.1364, 0.002153, 0.166807, -0.91106]      # 2017-06-15
EARLY_2016 = [0.018931, 3.99925, 0.0282674, 0.504482, -0.619937, 0.014555, 18.9992, 0.002, 1.93704, -0.491318]        # 2016-12-02, non-Feller fast factor
XIFAST_2018 = [0.0161152, 0.0594736, 0.159467, 0.202257, -0.861733, 0.00511242, 15.0593, 0.002, 2.0, -0.408186]       # 2018-02-20, largest calibrated xi
COVID_2020 = [0.3, 0.797678, 0.030252, 1.49814, -0.98, 0.3, 15.5728, 0.011484, 0.315499, -0.744318]                  # 2020-03-20, at the bounds
MID_2022 = [0.033831, 0.111816, 0.092847, 0.291609, -0.495659, 0.031851, 15.1118, 0.004871, 1.1687, -0.978877]        # 2022-06-16
LOWV_2024 = [0.002, 2.85115, 0.002, 0.195099, -0.978005, 0.121298, 16.8125, 0.013225, 0.53663, -0.722181]            # 2024-06-04
Y2025 = [0.013281, 1.25795, 0.021863, 0.287753, -0.555737, 0.072053, 16.0676, 0.003159, 1.64579, -0.426974]           # 2025-04-07
RECENT_2026 = [0.006413, 0.05091, 0.159305, 0.091678, -0.71116, 0.006823, 3.93975, 0.006055, 1.27728, -0.349895]     # 2026-09-25
PARAMS = {"default": DEFAULT, "calm2017": CALM_2017, "early2016": EARLY_2016, "xifast2018": XIFAST_2018, "covid2020": COVID_2020,
          "mid2022": MID_2022, "lowv2024": LOWV_2024, "y2025": Y2025, "recent2026": RECENT_2026}
JUMPS = {"default": (0.10, -0.06, 0.20), "calm2017": (0.152074, -0.066616, 0.133221), "covid2020": (0.213038, -0.027559, 0.156264),
         "recent2026": (0.048315, -0.099255, 0.235995), "xifast2018": (0.337128, -0.0395734, 0.11678)}

# Monte Carlo cases (name, parameters, maturity in years): >= 6 sets, short to one year
MC_SEED = 20261008
MC_CASES = [("default_3m", DEFAULT, 0.25), ("default_1y", DEFAULT, 1.0), ("calm2017_1m", CALM_2017, 30 / 365), ("xifast2018_2w", XIFAST_2018, 14 / 365),
            ("covid2020_3m", COVID_2020, 0.25), ("mid2022_6m", MID_2022, 0.5), ("lowv2024_1m", LOWV_2024, 30 / 365), ("y2025_3m", Y2025, 0.25),
            ("recent2026_1y", RECENT_2026, 1.0), ("early2016_1w", EARLY_2016, 7 / 365)]

S0, R0, Q0 = 25000.0, 0.065, 0.0                       # a NIFTY-like level and financing rate (only F and DF matter)
TENORS = [1 / 365, 3 / 365, 7 / 365, 14 / 365, 30 / 365, 91 / 365, 182 / 365, 1.0]


def fwd(T, S=S0, r=R0, q=Q0):
    return S * np.exp((r - q) * T), np.exp(-r * T)


def iv(price, F, K, T, r, is_call):
    with np.errstate(all="ignore"):
        return data.black_iv(np.maximum(price, 1e-12), F, K, T, r, is_call)


# ================================================================================================ (a) characteristic function vs mpmath
def _mp_ode_logcf(u, tau, v0, kappa, theta, xi, rho):
    """log E exp(iu z) for one Heston factor from the Riccati ODE itself, solved by mpmath's Taylor-series ODE solver with 30 digits.
    Continuous in tau by construction, so it tests the complex-logarithm branch of the closed form as well as its algebra."""
    import mpmath as mp
    mp.mp.dps = 30
    u = mp.mpc(u)
    kappa, theta, xi, rho, v0, tau = (mp.mpf(v) for v in (kappa, theta, xi, rho, v0, tau))
    f = lambda t, y: [0.5 * xi**2 * y[0] ** 2 - (kappa - rho * xi * 1j * u) * y[0] - 0.5 * (u * u + 1j * u), kappa * theta * y[0]]
    D, C = mp.odefun(f, 0, [mp.mpc(0), mp.mpc(0)])(tau)
    return complex(C + D * v0)


CF_POINTS = [(5.0, 0.1, DEFAULT[:5]), (50.0, 0.05, DEFAULT[5:]), (3.0 - 0.5j, 1.0, (0.04, 2.0, 0.04, 0.8, -0.8)),
             (200.0, 1 / 365, EARLY_2016[5:]), (30.0 - 1j, 1.0, COVID_2020[:5]), (1000.0 - 0.5j, 0.5, XIFAST_2018[5:]),
             (12.0 - 1j, 0.25, (0.04, 1.0, 0.04, 1.0, 0.9)), (0.3, 1.0, XIFAST_2018[:5]),
             (20.0, 0.5, (0.03, 2.5, 0.05, 1e-4, -0.5))]


@pytest.mark.parametrize("u,tau,par", CF_POINTS)
def test_cf_matches_mpmath_riccati_solution(u, tau, par):
    ref = _mp_ode_logcf(u, tau, *par)
    mine = complex(cos.log_cf_heston(u, tau, *par))
    assert abs(mine - ref) < 1e-11 * max(1.0, abs(ref))          # the log itself, not just exp of it: no 2 pi i branch jumps


def test_cf_is_exact_for_deterministic_variance_and_martingale():
    # xi -> 0: the variance is deterministic, z is normal with variance W = theta tau + (v0 - theta)(1 - e^{-kappa tau})/kappa
    v0, kappa, theta, tau = 0.03, 2.5, 0.05, 0.7
    W = theta * tau + (v0 - theta) * (1 - np.exp(-kappa * tau)) / kappa
    u = np.array([0.0, 0.5, 3.0, 40.0, 3.0 - 0.5j])
    exact = -0.5 * (u * u + 1j * u) * W
    assert np.max(np.abs(cos.log_cf_heston(u, tau, v0, kappa, theta, 0.0, -0.5) - exact)) < 1e-11      # xi = 0 exactly: any rho
    for xi in (1e-8, 1e-5, 1e-3):                                            # rho = 0: the correction is O(xi^2) (with rho != 0 it is O(rho xi u^3), a real skew)
        assert np.max(np.abs(cos.log_cf_heston(u, tau, v0, kappa, theta, xi, 0.0) - exact)) < 1e-9 + 1e-3 * xi**2 * 40**4
    # E exp(z) = 1 (z is measured from the forward): phi(-i) = 1 for the diffusion and the compensated jumps
    for name, p in PARAMS.items():
        assert abs(cos.log_cf_double(-1j, 0.4, p, JUMPS.get(name))) < 1e-9
    # the jump term is the one dh2/models.py uses
    u = np.array([0.3, 5.0, 20.0 - 0.5j])
    assert np.allclose(cos.log_cf_jump(u, 0.3, 0.2, -0.05, 0.15), M._jump_log_cf(u, 0.3, 0.2, -0.05, 0.15), rtol=1e-14, atol=1e-14)


def test_cumulants_match_mpmath_differentiation():
    import mpmath as mp
    mp.mp.dps = 40

    def K_factor(s, v0, kappa, theta, xi, rho, tau):          # cumulant generating function C(s) + D(s) v0 of one factor (closed form, in mpmath)
        c = kappa - rho * xi * s
        d = mp.sqrt(c * c + xi * xi * (s - s * s))
        g = (c - d) / (c + d)
        e = mp.exp(-d * tau)
        D = (c - d) / xi**2 * (1 - e) / (1 - g * e)
        C = kappa * theta / xi**2 * ((c - d) * tau - 2 * mp.log((1 - g * e) / (1 - g)))
        return C + D * v0

    cases = [(DEFAULT, 0.6, None), (EARLY_2016, 0.2, None), (COVID_2020, 1.0, (0.213038, -0.027559, 0.156264))]
    for p, tau, jump in cases:
        def K(s):
            out = K_factor(s, *p[:5], tau) + K_factor(s, *p[5:], tau)
            if jump:
                lam, mu, dl = jump
                out += lam * tau * (mp.exp(s * mu + s * s * dl * dl / 2) - 1 - s * (mp.exp(mu + dl * dl / 2) - 1))
            return out
        ref = [float(mp.diff(K, 0, n)) for n in (1, 2, 3, 4)]
        mine = [float(c[0]) for c in cos.cumulants(tau, p, jump)]
        assert np.allclose(mine, ref, rtol=1e-7, atol=1e-12), (mine, ref)


# ================================================================================================ (b) COS converges
@pytest.mark.parametrize("name,T", [("default", 1.0), ("default", 7 / 365), ("early2016", 30 / 365), ("covid2020", 0.5), ("xifast2018", 2 / 365)])
def test_cos_converges_as_terms_grow(name, T):
    p = PARAMS[name]
    x = np.linspace(-0.25, 0.25, 11)
    ref = cos.normalized_price(x, T, p, N=8192)
    errs = [np.max(np.abs(cos.normalized_price(x, T, p, N=n) - ref)) for n in (16, 32, 64, 128, 256, 512, 1024)]
    # exponential convergence: the error never grows by more than noise and ends at round-off level
    assert all(b <= 2 * a + 1e-13 for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-11, errs
    assert errs[0] > 10 * errs[-1] or errs[0] < 1e-11, errs
    # the adaptive choice stops at a converged answer and says so
    u, info = cos.normalized_price(x, T, p, return_info=True)
    assert info["converged"].all()
    assert np.max(np.abs(u - ref)) < 1e-11, (info["N"], np.max(np.abs(u - ref)))


def test_cos_truncation_width_is_checked_and_sufficient():
    """Fang & Oosterlee's L = 10 is not enough for the heavy-tailed calibrated sets; the default widens L until the price stops changing."""
    x = np.linspace(-0.25, 0.25, 11)
    for name, p in PARAMS.items():
        for T in (3 / 365, 32 / 365, 1.0):
            u, info = cos.normalized_price(x, T, p, return_info=True)
            ref = cos.normalized_price(x, T, p, L=info["L_used"].max() * 1.5)           # a still wider interval
            assert np.max(np.abs(u - ref)) < 1e-10, (name, T, np.max(np.abs(u - ref)), info["L_used"].max())
            assert info["converged"].all() and info["trunc_diff"] < 1e-9
    # the old fixed L = 10 is visibly worse on the non-Feller fast factor: this is why the width is checked
    T = 32 / 365
    u10 = cos.normalized_price(x, T, EARLY_2016, L=10.0)
    assert np.max(np.abs(u10 - cos.normalized_price(x, T, EARLY_2016))) > 1e-10


def test_cos_api_shapes_and_batches():
    F, DF, T = 100.0, 0.98, 0.3
    K = np.array([80.0, 95.0, 100.0, 110.0, 130.0])
    call, put = cos.cos_price(DEFAULT, F, K, T, DF, "call"), cos.cos_price(DEFAULT, F, K, T, DF, "put")
    assert call.shape == K.shape
    assert np.allclose(call - put, DF * (F - K), atol=1e-9)
    assert np.ndim(cos.cos_price(DEFAULT, F, 100.0, T, DF)) == 0
    # many parameter sets == a loop over single ones; normalized_price == price / (K DF)
    sets = np.array([DEFAULT, CALM_2017, COVID_2020])
    many = cos.cos_price_many(sets, F, K, np.array([0.3, 0.1, 1.0]), DF)
    for i, (p, t) in enumerate(zip(sets, [0.3, 0.1, 1.0])):
        assert np.allclose(many[i], cos.cos_price(p, F, K, t, DF), rtol=1e-11, atol=1e-11)
    u = cos.normalized_price(np.log(F / K), T, DEFAULT)
    assert np.allclose(u * K * DF, call, rtol=1e-12)
    # per-row parameters and maturities in one call
    xs = np.array([-0.1, 0.0, 0.1])
    ps = np.array([DEFAULT, EARLY_2016, XIFAST_2018])
    ts = np.array([0.5, 0.02, 1.0])
    rows = cos.normalized_price(xs, ts, ps)
    for i in range(3):
        assert abs(rows[i] - cos.normalized_price(xs[i], ts[i], ps[i])) < 1e-12


# ================================================================================================ (c) COS vs Lewis
def _agree(u1, u2):
    return np.abs(u1 - u2) <= np.maximum(1e-6 * np.abs(u2), 1e-11)


@pytest.mark.parametrize("name", list(PARAMS))
def test_cos_agrees_with_lewis_on_the_grid(name):
    """9 strikes (|ln K/F| <= 0.25) x 8 maturities (1 day .. 1 year), calls and puts, diffusion model: 1e-6 relative (1e-11 absolute floor)."""
    p = PARAMS[name]
    x = np.linspace(-0.25, 0.25, 9)
    for T in TENORS:
        F, DF = fwd(T)
        K = F * np.exp(-x)
        for kind in ("call", "put"):
            c = cos.cos_price(p, F, K, T, DF, kind) / (K * DF)
            lw = lewis.lewis_price(p, F, K, T, DF, kind) / (K * DF)
            ok = _agree(c, lw)
            assert ok.all(), (name, T, kind, np.c_[x, c, lw][~ok])


@pytest.mark.parametrize("name", list(JUMPS))
def test_cos_agrees_with_lewis_with_jumps(name):
    p, jump = PARAMS[name], JUMPS[name]
    x = np.linspace(-0.25, 0.25, 9)
    for T in (3 / 365, 30 / 365, 1.0):
        F, DF = fwd(T)
        K = F * np.exp(-x)
        for kind in ("call", "put"):
            c = cos.cos_price(p, F, K, T, DF, kind, jump=jump) / (K * DF)
            lw = lewis.lewis_price(p, F, K, T, DF, kind, jump=jump) / (K * DF)
            assert _agree(c, lw).all(), (name, T, kind, np.c_[x, c, lw][~_agree(c, lw)])


def test_lewis_reproduces_black_scholes():
    from scipy.special import ndtr
    sig, T = 0.2, 0.5
    p = [sig**2 / 2, 1.0, sig**2 / 2, 0.0, 0.0, sig**2 / 2, 5.0, sig**2 / 2, 0.0, 0.0]       # xi = 0: deterministic flat variance
    F, DF = 100.0, 0.97
    K = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    w = (p[0] + p[5]) * T                                                                      # = sig^2 T with v1 + v2 = sig^2
    d1 = (np.log(F / K) + 0.5 * w) / np.sqrt(w)
    bs = DF * (F * ndtr(d1) - K * ndtr(d1 - np.sqrt(w)))
    assert np.allclose(lewis.lewis_price(p, F, K, T, DF), bs, rtol=1e-10, atol=1e-10)
    assert np.allclose(cos.cos_price(p, F, K, T, DF), bs, rtol=1e-10, atol=1e-10)


# ================================================================================================ (d) the project pricer against COS and Lewis
def _project_call(p, S, K, T, r, q, jump=None):
    if jump is None:
        return np.atleast_1d(P.double_heston_price(S, K, T, r, *p, kind="call", q=q))
    log_cf = lambda u: (P._log_cf_factor(u, *p[:5], T) + P._log_cf_factor(u, *p[5:], T) + M._jump_log_cf(u, T, *jump))
    return np.atleast_1d(P._price_from_log_cf(log_cf, S, K, T, r, q, "call"))


def _compare_with_project(p, jump, days, x):
    """OTM prices (calls above the forward, puts below, as in the data) from COS, Lewis and the project pricer, plus implied vols."""
    T = days / 365.0
    F, DF = fwd(T)
    K = F * np.exp(-x)
    is_call = x <= 0
    call = _project_call(p, S0, K, T, R0, Q0, jump)
    proj = np.where(is_call, call, call - DF * (F - K))              # what dh2.models.price does for puts
    c = np.empty(K.size)
    c[is_call] = cos.cos_price(p, F, K[is_call], T, DF, "call", jump=jump)
    c[~is_call] = cos.cos_price(p, F, K[~is_call], T, DF, "put", jump=jump)
    return T, F, DF, K, is_call, proj, c


# Known finding (see results/pricer_validation/README.md): the project pricer's fixed integration spacing (0.5) aliases the density at +-12.57 in
# ln(K/F). For the extreme calibrated set of 2020-03-20 (v0 = 0.3, xi = 1.5, rho = -0.98) the 1-year log-return has c2 = 0.41, c4 = 2.8, i.e. real
# probability beyond 12, and the price is off by up to Rs 0.007 (7.7e-6 relative, 9e-5 vol point) at 1 year. Halving the spacing removes it.
# The relative-price criterion is NOT loosened for that set: the test is a strict xfail, so it turns red the day the pricer is fixed.
_ALIASING = pytest.mark.xfail(strict=True, reason="project pricer: midpoint spacing 0.5 aliases the heavy-tailed 1-year density (rel. error 7.7e-6 > 1e-6)")


def _domain_errors(p, jump, days_list):
    x = np.linspace(-0.22, 0.22, 23)
    worst_iv, worst_rel, n = 0.0, 0.0, 0
    for days in days_list:
        T, F, DF, K, is_call, proj, c = _compare_with_project(p, jump, days, x)
        sel = c >= 1.0
        if not sel.any():
            continue
        d_iv = 100 * np.abs(iv(proj, F, K, T, R0, is_call) - iv(c, F, K, T, R0, is_call))[sel]
        rel = (np.abs(proj - c) / c)[sel]
        worst_iv, worst_rel, n = max(worst_iv, np.nanmax(d_iv)), max(worst_rel, rel.max()), n + int(sel.sum())
    return worst_iv, worst_rel, n


@pytest.mark.parametrize("name", list(PARAMS))
def test_project_pricer_vs_cos_implied_vol_in_the_calibration_domain(name):
    """The quotes the calibration uses: >= 3 days, |ln K/F| <= 0.22, option worth >= Rs 1. Implied vol within 0.01 vol point."""
    worst_iv, _, n = _domain_errors(PARAMS[name], None, (3, 5, 7, 10, 14, 21, 30, 60, 91, 182, 365))
    assert n > 20
    assert worst_iv < 0.01, (name, worst_iv)


@pytest.mark.parametrize("name", [pytest.param(n, marks=_ALIASING) if n == "covid2020" else n for n in PARAMS])
def test_project_pricer_vs_cos_relative_price_in_the_calibration_domain(name):
    _, worst_rel, _ = _domain_errors(PARAMS[name], None, (3, 5, 7, 10, 14, 21, 30, 60, 91, 182, 365))
    assert worst_rel < 1e-6, (name, worst_rel)


@pytest.mark.parametrize("name", list(JUMPS))
def test_project_pricer_vs_cos_with_jumps_implied_vol(name):
    worst_iv, _, n = _domain_errors(PARAMS[name], JUMPS[name], (3, 7, 30, 91, 365))
    assert n > 10 and worst_iv < 0.01, (name, worst_iv)


@pytest.mark.parametrize("name", [pytest.param(n, marks=_ALIASING) if n == "covid2020" else n for n in JUMPS])
def test_project_pricer_vs_cos_with_jumps_relative_price(name):
    _, worst_rel, _ = _domain_errors(PARAMS[name], JUMPS[name], (3, 7, 30, 91, 365))
    assert worst_rel < 1e-6, (name, worst_rel)


def test_project_pricer_vs_lewis_agrees_like_vs_cos():
    p = MID_2022
    for days in (7, 91):
        T, F, DF, K, is_call, proj, c = _compare_with_project(p, None, days, np.linspace(-0.2, 0.2, 9))
        lw = np.where(is_call, lewis.lewis_price(p, F, K, T, DF, "call"), lewis.lewis_price(p, F, K, T, DF, "put"))
        sel = lw >= 1.0
        assert (np.abs(proj - lw) / lw)[sel].max() < 1e-6


@pytest.mark.parametrize("name", ["default", "early2016", "covid2020", "recent2026"])
def test_project_pricer_outside_the_calibration_domain(name):
    """Beyond what the calibration uses (1-2 day options, |ln K/F| up to 0.25, tiny prices): tradeable prices (>= one tick, Rs 0.05) still within 0.1 vol point."""
    p = PARAMS[name]
    x = np.linspace(-0.25, 0.25, 26)
    for days in (1, 2):
        T, F, DF, K, is_call, proj, c = _compare_with_project(p, None, days, x)
        sel = c >= 0.05
        if sel.any():
            d_iv = 100 * np.abs(iv(proj, F, K, T, R0, is_call) - iv(c, F, K, T, R0, is_call))[sel]
            assert np.nanmax(d_iv) < 0.1, (name, days, np.nanmax(d_iv))


# ================================================================================================ (e) limits
def test_single_heston_limit_factor_two_switched_off():
    """Factor 2 off (v0 = theta = 0): Double Heston is the single Heston, which the project prices with heston_price."""
    p1 = [0.04, 1.5, 0.05, 0.6, -0.7]
    off = [0.0, 1.0, 0.0, 0.1, 0.0]
    for T in (7 / 365, 0.25, 1.0):
        F, DF = fwd(T)
        K = F * np.exp(-np.linspace(-0.2, 0.2, 9))
        proj = np.atleast_1d(P.heston_price(S0, K, T, R0, *p1, kind="call", q=Q0))
        c = cos.cos_price(p1 + off, F, K, T, DF, "call")
        lw = lewis.lewis_price(p1 + off, F, K, T, DF, "call")
        assert np.allclose(c, proj, rtol=1e-8, atol=1e-6), np.max(np.abs(c - proj))
        assert np.allclose(lw, proj, rtol=1e-8, atol=1e-6)


def test_black_scholes_limit_vol_of_vol_to_zero():
    """xi = 0 is exactly Black-Scholes. For small xi > 0 the deviation is FIRST order in xi (a real skew, rho xi): bounded by 0.1 xi F, and COS, Lewis
    and (at xi = 1e-3, below which the project pricer clamps) the project pricer all see the same smile."""
    sig = 0.18
    for xi in (0.0, 1e-4, 1e-3):
        p = [sig**2 / 2, 1.0, sig**2 / 2, xi, -0.5, sig**2 / 2, 5.0, sig**2 / 2, xi, -0.5]
        for T in (7 / 365, 0.5):
            F, DF = fwd(T)
            K = F * np.exp(-np.linspace(-0.2, 0.2, 9))
            bs = data.black_price(F, K, T, R0, sig, True)
            c = cos.cos_price(p, F, K, T, DF, "call")
            if xi == 0.0:
                assert np.allclose(c, bs, rtol=1e-9, atol=1e-7), (T, np.max(np.abs(c - bs)))
            else:
                assert np.max(np.abs(c - bs)) < 0.1 * xi * F, (xi, T, np.max(np.abs(c - bs)))
                assert np.allclose(c, lewis.lewis_price(p, F, K, T, DF, "call"), rtol=1e-9, atol=1e-7)
            if xi == 1e-3:
                proj = np.atleast_1d(P.double_heston_price(S0, K, T, R0, *p, kind="call", q=Q0))
                assert np.allclose(c, proj, rtol=1e-8, atol=1e-6), (xi, T, np.max(np.abs(c - proj)))


# ================================================================================================ (f) arbitrage-freeness and stability
@pytest.mark.parametrize("name", ["default", "early2016", "covid2020", "xifast2018", "recent2026"])
@pytest.mark.parametrize("T", [1 / 365, 2 / 365, 14 / 365, 1.0])
def test_cos_no_arbitrage_parity_monotone_convex_bounds(name, T):
    p = PARAMS[name]
    F, DF = fwd(T)
    m = 0.25 if T > 0.01 else 0.06
    K = F * np.exp(np.linspace(-m, m, 161))
    call, put = cos.cos_price(p, F, K, T, DF, "call"), cos.cos_price(p, F, K, T, DF, "put")
    assert np.all(np.isfinite(call)) and np.all(np.isfinite(put))
    tol = 1e-12 * F
    # put-call parity, with both legs integrated directly (no parity inside the pricer): checks the payoff coefficients on both legs
    direct_put = cos.cos_price(p, F, K, T, DF, "put", leg="put")
    assert np.max(np.abs(direct_put - put)) < 1e-10 * F
    _, info = cos.cos_price(p, F, K, T, DF, "call", return_info=True)
    if info["b"].max() < 5.0:             # the direct call payoff e^z amplifies round-off by e^b: only meaningful while b is small (else the pricer itself uses the put leg)
        direct_call = cos.cos_price(p, F, K, T, DF, "call", leg="call")
        assert np.max(np.abs(direct_call - direct_put - DF * (F - K))) < 1e-10 * F
        assert np.max(np.abs(direct_call - call)) < 1e-10 * F
    assert np.max(np.abs(call - put - DF * (F - K))) < 1e-9
    # bounds
    assert np.all(call >= np.maximum(DF * (F - K), 0) - tol) and np.all(call <= DF * F + tol)
    assert np.all(put >= np.maximum(DF * (K - F), 0) - tol) and np.all(put <= DF * K + tol)
    # monotone and convex in strike (butterflies on the uniform-in-log grid are scaled by the strike spacing)
    assert np.all(np.diff(call) <= tol) and np.all(np.diff(put) >= -tol)
    Ku = np.linspace(K[0], K[-1], 161)
    cu = cos.cos_price(p, F, Ku, T, DF, "call")
    assert np.all(cu[:-2] - 2 * cu[1:-1] + cu[2:] >= -tol)


def test_cos_stable_at_one_and_two_days_against_lewis():
    for name in ("default", "early2016", "covid2020"):
        for days in (1, 2):
            T = days / 365
            F, DF = fwd(T)
            K = F * np.exp(np.linspace(-0.06, 0.06, 13))
            for kind in ("call", "put"):
                c, info = cos.cos_price(PARAMS[name], F, K, T, DF, kind, return_info=True)
                assert info["converged"].all() and np.all(np.isfinite(c)) and np.all(c >= 0)
                lw = lewis.lewis_price(PARAMS[name], F, K, T, DF, kind)
                assert _agree(c / (K * DF), lw / (K * DF)).all()
    # and an absurdly short expiry (one hour)
    T = 1 / (365 * 24)
    F, DF = fwd(T)
    K = F * np.exp(np.linspace(-0.01, 0.01, 9))
    c = cos.cos_price(DEFAULT, F, K, T, DF, "call")
    assert np.all(np.isfinite(c)) and np.all(np.diff(c) <= 0) and abs(c[4] - 0.4 * np.sqrt(0.04 * T) * F * DF) < 0.05 * c[4]


# ================================================================================================ (g) Monte Carlo (slow)
@pytest.mark.slow
@pytest.mark.parametrize("name,p,T", MC_CASES)
def test_monte_carlo_agrees_with_cos_within_the_99_percent_interval(name, p, T):
    """100k antithetic pairs (200k QE paths). The at-the-money call must lie inside the 99% interval (2.576 se) of COS; all seven strikes within 4 se."""
    F, DF = 100.0, np.exp(-0.06 * T)
    xs = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
    K = F * np.exp(-xs)
    price, se, diag = mc.mc_price(p, F, K, T, DF, "call", n_pairs=100_000, seed=MC_SEED)
    exact = cos.cos_price(p, F, K, T, DF, "call")
    z = (price - exact) / se
    assert abs(z[3]) < 2.576, (name, z)
    assert np.all(np.abs(z) < 4.0), (name, z)
    assert abs(diag["mean_exp_z"] - 1.0) < 6e-3                       # the simulated forward is a martingale
    assert diag["martingale_correction_failures"] == 0


@pytest.mark.slow
def test_monte_carlo_puts_jumps_and_control_variate():
    p, jump, T = CALM_2017, JUMPS["calm2017"], 0.25
    F, DF = 100.0, np.exp(-0.06 * T)
    K = F * np.exp(-np.array([-0.08, 0.0, 0.08]))
    exact_put = cos.cos_price(p, F, K, T, DF, "put", jump=jump)
    exact_call = cos.cos_price(p, F, K, T, DF, "call", jump=jump)
    price, se, _ = mc.mc_price(p, F, K, T, DF, "put", n_pairs=100_000, seed=MC_SEED, jump=jump)
    assert abs(price[1] - exact_put[1]) < 2.576 * se[1] and np.all(np.abs((price - exact_put) / se) < 4)
    pc, sc, _ = mc.mc_price(p, F, K, T, DF, "call", n_pairs=100_000, seed=MC_SEED, jump=jump, control_variate=True)
    p0, s0, _ = mc.mc_price(p, F, K, T, DF, "call", n_pairs=100_000, seed=MC_SEED, jump=jump)
    assert np.all(sc <= s0 * 1.0001)                                  # the control variate never makes the interval wider
    assert abs(pc[1] - exact_call[1]) < 2.576 * sc[1] and np.all(np.abs((pc - exact_call) / sc) < 4)


def test_monte_carlo_is_reproducible_and_pairs_are_antithetic():
    a = mc.mc_price(DEFAULT, 100.0, [95.0, 105.0], 0.1, 0.99, n_pairs=2000, seed=3)
    b = mc.mc_price(DEFAULT, 100.0, [95.0, 105.0], 0.1, 0.99, n_pairs=2000, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    z, _ = mc.mc_terminal(DEFAULT, 0.1, n_pairs=2000, seed=3)
    assert z.size == 4000
