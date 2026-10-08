"""Domain, sampling and exact labels for the Double Heston surrogate.

Coordinates (they remove S, K, r and q from the problem):  x = ln(F/K),  u = C/(K*DF),  tau = time to expiry in years,
Theta = (kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2)  (the pricer's xi is sigma),  state = (v1, v2).

Exact labels come from a vectorised COS (Fourier-cosine) pricer written here, which prices thousands of (parameter, state, tau)
combinations and several strikes each in one call.  It is checked against the project's own Gil-Pelaez pricer (pricer/models.py)
in tests/test_pinn.py and in results/pinn/tables/cos_vs_pricer.csv; held-out validation uses the project's pricer itself.

The DOMAIN comes from the calibrated DEVELOPMENT-period parameters only (walk_dh_main.pkl, days before 2022-01-01) plus a margin.
Test-period parameters are only compared against it afterwards, to flag the ones that fall outside; nothing is tuned on them.
"""
from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WALK = ROOT / "data" / "results" / "walk_dh_main.pkl"
CUTOFF = date(2022, 1, 1)

ETA_NAMES = ["kappa1", "theta1", "sigma1", "rho1", "dk", "theta2", "sigma2", "rho2"]   # walk-forward 'eta' order (kappa2 = kappa1 + dk)
THETA_NAMES = ["kappa1", "theta1", "sigma1", "rho1", "kappa2", "theta2", "sigma2", "rho2"]
X_MAX = 0.30
TAU_MIN = 1.0 / 365.0
TAU_MAX = 0.60
V_MIN, V_MAX = 0.0016, 0.36           # calibrator state bounds are [0.002, 0.30]; a margin on both sides
VEGA_FLOOR = 2.0e-6                   # normalised price per volatility point; Rs 0.05 per point on a NIFTY strike (calibrator's VEGA_FLOOR)
P_MIN = 4.0e-5                        # normalised OTM price below which a quote would not be kept (Rs 1 on a ~25,000 index)
_POS = np.array([1, 1, 1, 0, 1, 1, 1, 0], bool)   # eta entries sampled log-uniformly (rho entries linearly)


# ------------------------------------------------------------------ parameter conversions
def eta_to_theta(eta) -> np.ndarray:
    """Walk-forward structural vector (kappa1, theta1, xi1, rho1, dk, theta2, xi2, rho2) -> Theta with kappa2 = kappa1 + dk."""
    e = np.asarray(eta, float)
    th = e.copy()
    th[..., 4] = e[..., 0] + e[..., 4]
    return th


def theta_to_pricer(theta, v1, v2) -> list:
    """Arguments of pricer.double_heston_price after (S, K, T, r): v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2."""
    t = np.asarray(theta, float)
    return [float(v1), t[0], t[1], t[2], t[3], float(v2), t[4], t[5], t[6], t[7]]


# ------------------------------------------------------------------ domain
def build_domain(margin: float = 1.25, rho_margin: float = 0.05, holdout_frac: float = 0.2) -> dict:
    """Parameter box and empirical development-period sets, from the walk-forward fits made before 2022-01-01.

    Box: for each structural parameter [min/margin, max*margin] over the development days (log scale for positive ones, +-rho_margin
    for correlations), clipped to the calibrator's own bounds.  The last `holdout_frac` of the development days (by date) are kept
    out of training: their (eta, state) pairs are the 'real held-out parameters'.  Returns plain arrays so it can be saved in a checkpoint.
    """
    from dh2 import calib                                      # only needed here, not for inference
    recs = pickle.load(open(WALK, "rb"))
    dev = sorted([r for r in recs if r["day"] < CUTOFF], key=lambda r: r["day"])
    tst = sorted([r for r in recs if r["day"] >= CUTOFF], key=lambda r: r["day"])
    eta = np.array([r["eta"] for r in dev], float)
    st = np.array([r["state"] for r in dev], float)
    lo_b, hi_b = (np.array(b, float) for b in calib.BOUNDS["dh"])
    idx = [1, 2, 3, 4, 6, 7, 8, 9]                                # structural entries in the calibrator's internal vector
    lo_b, hi_b = lo_b[idx], hi_b[idx]
    lo, hi = eta.min(0), eta.max(0)
    elo = np.where(_POS, lo / margin, lo - rho_margin)
    ehi = np.where(_POS, hi * margin, hi + rho_margin)
    elo = np.maximum(elo, lo_b); ehi = np.minimum(ehi, hi_b)
    elo[3] = max(elo[3], -0.99); ehi[3] = min(ehi[3], 0.99); elo[7] = max(elo[7], -0.99); ehi[7] = min(ehi[7], 0.99)
    n_tr = int(round(len(dev) * (1 - holdout_frac)))
    dom = {
        "eta_lo": elo, "eta_hi": ehi, "v_lo": np.array([V_MIN, V_MIN]), "v_hi": np.array([V_MAX, V_MAX]),
        "tau_lo": TAU_MIN, "tau_hi": TAU_MAX, "x_max": X_MAX,
        "train_eta": eta[:n_tr], "train_state": st[:n_tr], "ho_eta": eta[n_tr:], "ho_state": st[n_tr:],
        "ho_days": [r["day"].isoformat() for r in dev[n_tr:]], "train_days": [r["day"].isoformat() for r in dev[:n_tr]],
        "dev_eta_min": lo, "dev_eta_max": hi,
        "dev_state_min": st.min(0), "dev_state_max": st.max(0),
        "test_eta": np.array([r["eta"] for r in tst], float), "test_state": np.array([r["state"] for r in tst], float),
        "test_days": [r["day"].isoformat() for r in tst],
    }
    return dom


def domain_theta_bounds(dom: dict) -> tuple[np.ndarray, np.ndarray]:
    """Box for Theta (8,): kappa2 runs from kappa1_lo + dk_lo to kappa1_hi + dk_hi."""
    lo = dom["eta_lo"].copy(); hi = dom["eta_hi"].copy()
    lo[4] = dom["eta_lo"][0] + dom["eta_lo"][4]; hi[4] = dom["eta_hi"][0] + dom["eta_hi"][4]
    return lo, hi


def net_domain(dom: dict) -> dict:
    """The part of the domain the network and the surrogate API need (small, saved inside every checkpoint)."""
    lo, hi = domain_theta_bounds(dom)
    return {"theta_lo": lo.tolist(), "theta_hi": hi.tolist(), "v_lo": np.asarray(dom["v_lo"]).tolist(), "v_hi": np.asarray(dom["v_hi"]).tolist(),
            "x_max": float(dom["x_max"]), "tau_lo": float(dom["tau_lo"]), "tau_hi": float(dom["tau_hi"])}


def in_domain(dom: dict, eta, state) -> np.ndarray:
    """True where a structural vector (walk-forward 'eta' order) and state lie inside the box."""
    eta = np.atleast_2d(eta); state = np.atleast_2d(state)
    ok = np.all((eta >= dom["eta_lo"] - 1e-12) & (eta <= dom["eta_hi"] + 1e-12), 1)
    return ok & np.all((state >= dom["v_lo"]) & (state <= dom["v_hi"]), 1)


def save_domain(dom: dict, path) -> None:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dom.items()}
    Path(path).write_text(json.dumps(out))


# ------------------------------------------------------------------ expected total variance
def total_variance(theta, v1, v2, tau):
    """w = integral over [0, tau] of E[v1 + v2] (arrays broadcast): the variance of the BS comparison price."""
    th = np.asarray(theta, float)
    out = 0.0
    for v, k, m in ((v1, th[..., 0], th[..., 1]), (v2, th[..., 4], th[..., 5])):
        out = out + m * tau + (v - m) * (-np.expm1(-k * tau)) / k
    return out


# ------------------------------------------------------------------ sampling
def sample_params(n: int, dom: dict, rng: np.random.Generator, p_emp: float = 0.5, source: str = "train") -> tuple:
    """(theta (n,8), v1 (n,), v2 (n,)): a mixture of (a) development-day parameter sets with small jitter and (b) the box,
    log-uniform in positive parameters and states.  source='train' uses the training days, 'holdout' the held-out days (no jitter)."""
    elo, ehi = dom["eta_lo"], dom["eta_hi"]
    eta = np.empty((n, 8))
    for j in range(8):
        eta[:, j] = np.exp(rng.uniform(np.log(elo[j]), np.log(ehi[j]), n)) if _POS[j] else rng.uniform(elo[j], ehi[j], n)
    st = np.exp(rng.uniform(np.log(dom["v_lo"]), np.log(dom["v_hi"]), (n, 2)))
    if p_emp > 0:
        pick = rng.random(n) < p_emp
        m = int(pick.sum())
        if m:
            E = dom["train_eta"] if source == "train" else dom["ho_eta"]
            S = dom["train_state"] if source == "train" else dom["ho_state"]
            d = rng.integers(0, len(E), m)
            e = E[d].copy(); s = S[d].copy()
            if source == "train":                              # jitter so the network sees a neighbourhood, not 120 points
                e = np.where(_POS, e * np.exp(rng.normal(0, 0.10, e.shape)), e + rng.normal(0, 0.03, e.shape))
                s = s * np.exp(rng.normal(0, 0.15, s.shape))
            eta[pick] = np.clip(e, elo, ehi)
            st[pick] = np.clip(s, dom["v_lo"], dom["v_hi"])
    return eta_to_theta(eta), st[:, 0], st[:, 1]


def sample_tau(n: int, rng: np.random.Generator, lo: float = TAU_MIN, hi: float = TAU_MAX) -> np.ndarray:
    """Maturities: half log-uniform (many short ones), half uniform."""
    a = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    b = rng.uniform(lo, hi, n)
    return np.where(rng.random(n) < 0.5, a, b)


def sample_x(theta, v1, v2, tau, rng: np.random.Generator, x_max: float = X_MAX) -> np.ndarray:
    """Log-moneyness: dense near the money.  45% uniform in +-4.5 BS standard deviations of the comparison variance (so a 1-day
    option is sampled where it has any value), 35% uniform on [-x_max, x_max], 20% normal with sd 0.04.  Out-of-range draws are redrawn."""
    n = len(tau)
    sd = np.sqrt(total_variance(theta, v1, v2, tau))
    u = rng.random(n)
    x = np.where(u < 0.45, sd * rng.uniform(-4.5, 4.5, n), np.where(u < 0.80, rng.uniform(-x_max, x_max, n), rng.normal(0, 0.04, n)))
    bad = np.abs(x) > x_max
    x[bad] = rng.uniform(-x_max, x_max, int(bad.sum()))
    return x


def sample_inputs(n: int, dom: dict, rng: np.random.Generator, p_emp: float = 0.5, source: str = "train") -> dict:
    theta, v1, v2 = sample_params(n, dom, rng, p_emp, source)
    tau = sample_tau(n, rng)
    x = sample_x(theta, v1, v2, tau, rng)
    return {"x": x, "v1": v1, "v2": v2, "tau": tau, "theta": theta}


# ------------------------------------------------------------------ exact labels: vectorised COS
def _log_cf_factor(u, v0, kappa, theta, xi, rho, tau):
    """Log characteristic function of one variance factor for ln(S_T/F): the pricer's 'little trap' form, with array parameters."""
    xi = np.maximum(xi, 1e-3)
    c = kappa - rho * xi * 1j * u
    d = np.sqrt(c * c + xi * xi * (1j * u + u * u))
    g = (c - d) / (c + d)
    e = np.exp(-d * tau)
    C = (kappa * theta / xi**2) * ((c - d) * tau - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (c - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    return C + D * v0


def log_cf(u, theta, v1, v2, tau):
    """Double Heston log characteristic function (the two independent factors add).  theta (n,8); v1, v2, tau (n,1); u (n,k)."""
    return (_log_cf_factor(u, v1, theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4], tau)
            + _log_cf_factor(u, v2, theta[:, 4:5], theta[:, 5:6], theta[:, 6:7], theta[:, 7:8], tau))


def cumulants(theta, v1, v2, tau):
    """c1, c2, c4 of Y = ln(S_T/F).  c1 = -w/2 exactly; c2, c4 from the real part of the log CF at two small frequencies
    (Re log phi(h) = -c2 h^2/2 + c4 h^4/24 - ...), which is cheap and vectorised."""
    w = total_variance(theta, v1, v2, tau)
    h = (0.02 / np.sqrt(w + 1e-14))[:, None]
    t = tau[:, None]
    A = log_cf(h + 0j, theta, v1[:, None], v2[:, None], t).real
    B = log_cf(2 * h + 0j, theta, v1[:, None], v2[:, None], t).real
    c4 = 2.0 * (B - 4.0 * A) / h**4
    c2 = -2.0 * (A - c4 * h**4 / 24.0) / h**2
    return -0.5 * w, c2[:, 0], c4[:, 0]


def cos_call(theta, v1, v2, tau, x, N: int = 2048, L: float = 12.0, chunk: int = 600) -> np.ndarray:
    """Normalised call u = C/(K DF) at log-moneyness x, by the COS method (Fang & Oosterlee 2008), via the put and parity.

    theta (n,8), v1, v2, tau (n,), x (n,m) or (n,).  Truncation [a,b] = c1 +- L sqrt(c2 + sqrt|c4|); N cosine terms.
    Independent of the project's pricer.  Agreement with it is ~1e-7 (N=2048, L=12) and ~4e-8 with N=4096, L=16 on the sampled
    domain, including the heavy-tailed corner (sigma ~ 2, kappa2 ~ 15); see tests/test_pinn.py and results/pinn/tables/cos_vs_pricer.csv."""
    theta = np.atleast_2d(np.asarray(theta, float)); n = theta.shape[0]
    v1 = np.broadcast_to(np.asarray(v1, float), (n,)); v2 = np.broadcast_to(np.asarray(v2, float), (n,)); tau = np.broadcast_to(np.asarray(tau, float), (n,))
    x = np.asarray(x, float); squeeze = x.ndim == 1
    x = x.reshape(n, -1) if x.ndim == 1 else x
    out = np.empty(x.shape)
    k = np.arange(N)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        th, a1, a2, ta, xx = theta[sl], v1[sl], v2[sl], tau[sl], x[sl]
        c1, c2, c4 = cumulants(th, a1, a2, ta)
        half = L * np.sqrt(np.maximum(c2, 1e-16) + np.sqrt(np.abs(c4)))
        a, b = c1 - half, c1 + half
        ba = (b - a)[:, None]
        om = k[None, :] * np.pi / ba                                          # (n, N)
        phi = np.exp(log_cf(om + 0j, th, a1[:, None], a2[:, None], ta[:, None]) - 1j * om * a[:, None])
        Fk = (2.0 / ba) * phi.real
        Fk[:, 0] *= 0.5
        # put payoff 1 - e^{x+Y} on Y < -x
        d0 = np.minimum(b[:, None], -xx)                                      # (n, m)
        live = d0 > a[:, None]
        dd = np.where(live, d0 - a[:, None], 0.0)[:, :, None]                 # (n, m, 1)
        w3 = om[:, None, :]                                                   # (n, 1, N)
        sn, cs = np.sin(w3 * dd), np.cos(w3 * dd)
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = np.where(w3 > 0, sn / np.where(w3 > 0, w3, 1.0), dd)
        chi = (cs * np.exp(d0)[:, :, None] - np.exp(a)[:, None, None] + w3 * sn * np.exp(d0)[:, :, None]) / (1.0 + w3**2)
        V = psi - np.exp(xx)[:, :, None] * chi
        put = np.einsum("nk,nmk->nm", Fk, V)
        put = np.where(live, put, 0.0)
        call = put + np.exp(xx) - 1.0
        out[sl] = np.clip(call, np.maximum(np.exp(xx) - 1.0, 0.0), np.exp(xx))
    return out[:, 0] if squeeze else out


def _project_pricer():
    from dh2 import models as M                                  # the project's own pricer, imported unchanged
    return M.P


def pricer_call(theta, v1, v2, tau, x) -> np.ndarray:
    """Exact normalised call from the PROJECT's pricer (pricer/models.py, Gil-Pelaez), one parameter set per row, several strikes each.
    S = F = 1, r = q = 0, K = exp(-x): the pricer works from the forward, so r and q only enter through F and DF (tested)."""
    P = _project_pricer()
    theta = np.atleast_2d(np.asarray(theta, float)); n = theta.shape[0]
    x = np.asarray(x, float).reshape(n, -1)
    out = np.empty(x.shape)
    for i in range(n):
        K = np.exp(-x[i])
        out[i] = np.atleast_1d(P.double_heston_price(1.0, K, float(tau[i]), 0.0, *theta_to_pricer(theta[i], v1[i], v2[i]), kind="call", q=0.0)) / K
    return out


def exact_otm(u, x):
    """The out-of-the-money option value (call for x <= 0, put for x > 0) from the normalised call u."""
    return np.where(x <= 0, u, u - (np.exp(x) - 1.0))


# ------------------------------------------------------------------ Black quantities in normalised coordinates
def _ndtr(z):
    from scipy.special import ndtr
    return ndtr(z)


def bs_unit(x, w):
    """Black-76 call in normalised form: e^x N(d1) - N(d2) with d1 = (x + w/2)/sqrt(w), total variance w."""
    sw = np.sqrt(np.maximum(w, 1e-14))
    d1 = (x + 0.5 * w) / sw
    return np.exp(x) * _ndtr(d1) - _ndtr(d1 - sw)


def implied_vol_unit(u_otm, x, tau, lo: float = 1e-3, hi: float = 5.0, iters: int = 60):
    """Black implied vol (annualised) of an out-of-the-money normalised price, by vectorised bisection; NaN outside the no-arb range."""
    u_otm = np.asarray(u_otm, float); x = np.asarray(x, float); tau = np.asarray(tau, float)
    u_otm, x, tau = np.broadcast_arrays(u_otm, x, tau)
    call_form = np.where(x <= 0, u_otm, u_otm + np.exp(x) - 1.0)            # back to a call price: intrinsic is represented exactly
    a = np.full(u_otm.shape, lo); b = np.full(u_otm.shape, hi)
    st = np.sqrt(tau)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = bs_unit(x, m * m * tau) < call_form
        a = np.where(up, m, a); b = np.where(up, b, m)
    iv = 0.5 * (a + b)
    ok = (bs_unit(x, lo * lo * tau) <= call_form) & (bs_unit(x, hi * hi * tau) >= call_form) & np.isfinite(u_otm)
    return np.where(ok, iv, np.nan)


def vega_per_point(x, tau, iv):
    """d(normalised price)/d(volatility) per volatility POINT at implied vol iv (x, tau, iv broadcast)."""
    sd = iv * np.sqrt(tau)
    d1 = (x + 0.5 * sd**2) / sd
    return np.exp(x) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(tau) / 100.0


def make_labels(inp: dict, n_x_per: int = 8, N: int = 2048, backend: str = "pricer") -> dict:
    """Label a batch of sampled inputs with exact prices.  inp holds n_x_per*k rows in which consecutive blocks of n_x_per rows
    share (theta, v1, v2, tau); returns the same dict plus 'u' (call), 'u_otm', 'iv', 'vega' (per point, floored).
    backend 'pricer' = the project's Gil-Pelaez pricer (default; ~0.3 ms per parameter set with 8 strikes), 'cos' = the vectorised COS above."""
    n = len(inp["x"])
    k = n // n_x_per
    sl = slice(0, k * n_x_per)
    th = inp["theta"][sl].reshape(k, n_x_per, 8)[:, 0]
    v1 = inp["v1"][sl].reshape(k, n_x_per)[:, 0]; v2 = inp["v2"][sl].reshape(k, n_x_per)[:, 0]; tau = inp["tau"][sl].reshape(k, n_x_per)[:, 0]
    x = inp["x"][sl].reshape(k, n_x_per)
    u = pricer_call(th, v1, v2, tau, x) if backend == "pricer" else cos_call(th, v1, v2, tau, x, N=N, L=12.0)
    out = {kk: v[sl] for kk, v in inp.items()}
    out["u"] = u.reshape(-1)
    xf = out["x"]
    otm = exact_otm(out["u"], xf)
    out["u_otm"] = otm
    iv = implied_vol_unit(np.maximum(otm, 1e-14), xf, out["tau"])
    out["iv"] = iv
    vg = vega_per_point(xf, out["tau"], np.where(np.isfinite(iv), iv, 0.15))
    out["vega"] = np.maximum(vg, VEGA_FLOOR)
    return out


def make_dataset(n: int, dom: dict, seed: int, n_x_per: int = 8, p_emp: float = 0.5, backend: str = "pricer") -> dict:
    """n labelled samples (rounded down to a multiple of n_x_per): n/n_x_per distinct (theta, state, tau) combinations, n_x_per strikes each."""
    rng = np.random.default_rng(seed)
    k = n // n_x_per
    theta, v1, v2 = sample_params(k, dom, rng, p_emp)
    tau = sample_tau(k, rng)
    th_r = np.repeat(theta, n_x_per, 0); v1_r = np.repeat(v1, n_x_per); v2_r = np.repeat(v2, n_x_per); tau_r = np.repeat(tau, n_x_per)
    x = sample_x(th_r, v1_r, v2_r, tau_r, rng)
    return make_labels({"x": x, "v1": v1_r, "v2": v2_r, "tau": tau_r, "theta": th_r}, n_x_per, backend=backend)
