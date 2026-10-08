"""Independent exact pricer for the Double Heston model: the COS method (Fang & Oosterlee 2008), written from the model's own equations.

It does NOT use pricer/models.py (no _log_cf_factor, no Gil-Pelaez grid). Everything here is derived and coded separately, so that a
disagreement with the project pricer means something. dh2/lewis.py (single integral) and dh2/mc.py (simulation) are the other two
independent checks and reuse the characteristic function defined here.

Model (risk neutral, x = ln(F/K), z = ln(S_T/F), forward F = S exp((r-q)T)):
    dS/S = (r-q) dt + sqrt(v1) dW1 + sqrt(v2) dW2
    dv_j = kappa_j (theta_j - v_j) dt + xi_j sqrt(v_j) dZ_j,   corr(dW_j, dZ_j) = rho_j, every other pair independent
    optional Merton jumps (lam, mu_j, delta_j), compensated so that the forward is unchanged (same convention as dh2/models.py).
Parameter order (v0_1, kappa1, theta1, xi1, rho1, v0_2, kappa2, theta2, xi2, rho2).

Characteristic function (derived here, not copied). Under the forward measure z has drift -v/2, and
phi_j(u) = E exp(iu z_j) = exp(C_j + D_j v0_j) solves the Riccati system
    D' = (xi^2/2) D^2 - (kappa - rho xi i u) D - (u^2 + i u)/2,    C' = kappa theta D,    D(0) = C(0) = 0.
With c = kappa - rho xi i u, d = sqrt(c^2 + xi^2 (u^2 + i u)) (principal root), g = (c-d)/(c+d), e = exp(-d tau) the "little trap" solution
(Albrecher et al. 2007) is
    D = (c-d)/xi^2 (1-e)/(1-g e),    C = kappa theta/xi^2 [ (c-d) tau - 2 ln((1-g e)/(1-g)) ].
The code evaluates the algebraically identical form (c-d) = -xi^2 (u^2+iu)/(c+d), so nothing cancels when xi -> 0 and xi = 0 is allowed.
tests/test_pricer_cross.py checks D and C against mpmath (D against the Riccati ODE itself, C as the integral of kappa theta D, which tests
the complex-log branch too). The factors are independent, so the logs add (plus the jump term).

Truncation range. COS needs an interval [a, b] holding the density of z. Following Fang & Oosterlee, [a, b] = c1 +- L sqrt(c2 + sqrt(c4)) (L widened until the price stops changing, see _DEFAULTS),
c_n the cumulants of z. The cumulants are computed ANALYTICALLY, not by differencing: with s = -i u the cumulant generating function is
K(s) = C(s) + D(s) v0, D(s) = sum d_n s^n, and matching powers of s in the Riccati equation gives a closed cascade
    d1' = -kappa d1 - 1/2
    d2' = -kappa d2 + rho xi d1 + xi^2/2 d1^2 + 1/2
    d3' = -kappa d3 + rho xi d2 + xi^2 d1 d2
    d4' = -kappa d4 + rho xi d3 + xi^2 d1 d3 + xi^2/2 d2^2,         C_n' = kappa theta d_n,
    cumulant_n = n! (C_n + v0 d_n).
The products d1^2, d1 d2, ... also satisfy linear equations (they close), so the whole thing is one 16-dimensional LINEAR system solved
exactly with a matrix exponential. Jump cumulants are the compound-Poisson ones. The tests compare all four cumulants with mpmath
high-precision differentiation of the characteristic function.

Pricing. For the out-of-the-money leg (call if K > F, put if K <= F) the normalised price u = price/(K DF) is a cosine series
    u = sum'_k Re{ phi(k pi/(b-a)) exp(-i k pi a/(b-a)) } G_k(x),
with the payoff coefficients G_k in closed form; the in-the-money option follows from parity (call - put = DF (F - K)), so no digits are
lost to cancellation on the expensive side. The number of terms is not fixed: terms are added in blocks (32, 32, 64, 128, ...) and a
parameter set stops when the last block changes every one of its prices by less than tol_abs + tol_rel * price. Everything is NumPy
float64, vectorised over strikes and over parameter sets.

Public API
    cos_price(params, F, K, T, DF, kind="call", jump=None, ...)          K array, one parameter set -> price in the units of F
    cos_price_many(params_matrix, F, K, T, DF, kind="call", jump=None)   many parameter sets (rows) -> (M, nK)
    normalized_price(x, tau, params, jump=None, ...)                     u = C/(K DF) for calls, x = ln(F/K), broadcast over x, tau, params
    log_cf_double(u, tau, params, jump=None), cumulants(tau, params, jump=None)   building blocks (also used by dh2/lewis.py, dh2/mc.py)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

PARAM_NAMES = ("v0_1", "kappa1", "theta1", "xi1", "rho1", "v0_2", "kappa2", "theta2", "xi2", "rho2")


# ------------------------------------------------------------------------------------------------ characteristic function
def _log1p_over(w):
    """log(1+w)/w, continuous at w = 0 (complex w)."""
    w = np.asarray(w, dtype=complex)
    small = np.abs(w) < 1e-6
    ws = np.where(small, 1.0, w)
    return np.where(small, 1.0 - w / 2.0 + w * w / 3.0, np.log1p(ws) / ws)


def log_cf_heston(u, tau, v0, kappa, theta, xi, rho):
    """log E exp(i u z) for ONE Heston variance factor, z = ln(S_T/F) contribution of that factor. u may be complex; every argument broadcasts.
    Little-trap form (see module docstring), written without the cancellation c - d."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    q = u * u + iu                              # u^2 + i u
    c = kappa - rho * xi * iu
    d = np.sqrt(c * c + xi * xi * q)            # principal root, Re d >= 0
    cd = c + d
    e = np.exp(-d * tau)
    g_over_xi2 = -q / (cd * cd)                 # = (c-d)/(c+d) / xi^2
    g = xi * xi * g_over_xi2                    # = (c-d)/(c+d)
    one_minus_g = 1.0 - g
    D = -q / cd * (1.0 - e) / (1.0 - g * e)     # = (c-d)/xi^2 (1-e)/(1-g e)
    w = g * (1.0 - e) / one_minus_g             # (1-g e)/(1-g) = 1 + w
    C = kappa * theta * (-q * tau / cd - 2.0 * g_over_xi2 * (1.0 - e) / one_minus_g * _log1p_over(w))
    return C + D * v0


def log_cf_jump(u, tau, lam, mu, delta):
    """Merton jump term, compensated (E exp(jump part) = 1): same convention as dh2/models.py _jump_log_cf."""
    u = np.asarray(u, dtype=complex)
    kbar = np.exp(mu + 0.5 * delta**2) - 1.0
    return lam * tau * (np.exp(1j * u * mu - 0.5 * delta**2 * u * u) - 1.0 - 1j * u * kbar)


def _split(params):
    p = np.asarray(params, dtype=float)
    if p.shape[-1] != 10:
        raise ValueError("params must end in a length-10 axis (Double Heston); give jumps through the jump argument")
    return p


def log_cf_double(u, tau, params, jump=None):
    """log E exp(i u ln(S_T/F)) for Double Heston (+ optional Merton jumps). params: length-10 sequence of scalars or arrays (one per
    parameter, mutually broadcastable with u and tau); jump: None or (lam, mu_j, delta_j)."""
    v1, k1, t1, x1, r1, v2, k2, t2, x2, r2 = params
    out = log_cf_heston(u, tau, v1, k1, t1, x1, r1) + log_cf_heston(u, tau, v2, k2, t2, x2, r2)
    if jump is not None:
        out = out + log_cf_jump(u, tau, *jump)
    return out


# ------------------------------------------------------------------------------------------------ cumulants (analytic)
# Linear system Y' = M Y for Y = [1, d1, d1^2, d1^3, d1^4, d2, d1 d2, d1^2 d2, d2^2, d3, d1 d3, d4, C1, C2, C3, C4].
_I = {n: i for i, n in enumerate(["ONE", "D1", "D1_2", "D1_3", "D1_4", "D2", "D1D2", "D1_2D2", "D2_2", "D3", "D1D3", "D4", "C1", "C2", "C3", "C4"])}


def _cumulant_matrix(kappa, theta, xi, rho):
    a, b, w = kappa, rho * xi, xi * xi
    n = a.shape[0]
    M = np.zeros((n, 16, 16))
    I = _I

    def row(name, **coef):
        for col, val in coef.items():
            M[:, I[name], I[col]] = val

    row("D1", D1=-a, ONE=-0.5)
    row("D1_2", D1_2=-2 * a, D1=-1.0)
    row("D1_3", D1_3=-3 * a, D1_2=-1.5)
    row("D1_4", D1_4=-4 * a, D1_3=-2.0)
    row("D2", D2=-a, D1=b, D1_2=w / 2, ONE=0.5)
    row("D1D2", D1D2=-2 * a, D2=-0.5, D1_2=b, D1_3=w / 2, D1=0.5)
    row("D1_2D2", D1_2D2=-3 * a, D1D2=-1.0, D1_3=b, D1_4=w / 2, D1_2=0.5)
    row("D2_2", D2_2=-2 * a, D1D2=2 * b, D1_2D2=w, D2=1.0)
    row("D3", D3=-a, D2=b, D1D2=w)
    row("D1D3", D1D3=-2 * a, D3=-0.5, D1D2=b, D1_2D2=w)
    row("D4", D4=-a, D3=b, D1D3=w, D2_2=w / 2)
    for n_, name in enumerate(["C1", "C2", "C3", "C4"], start=1):
        M[:, I[name], I["D%d" % n_]] = a * theta
    return M


def _heston_cumulants(tau, v0, kappa, theta, xi, rho):
    """c1..c4 of one factor's z contribution, arrays of shape (n,)."""
    M = _cumulant_matrix(kappa, theta, xi, rho) * tau[:, None, None]
    Y = expm(M)[:, :, 0]                       # state at tau, started from [1, 0, ..., 0]
    fact = (1.0, 2.0, 6.0, 24.0)
    return [fact[n - 1] * (Y[:, _I["C%d" % n]] + v0 * Y[:, _I["D%d" % n]]) for n in (1, 2, 3, 4)]


def _jump_cumulants(tau, lam, mu, delta):
    kbar = np.exp(mu + 0.5 * delta**2) - 1.0
    m2 = mu**2 + delta**2
    m3 = mu**3 + 3 * mu * delta**2
    m4 = mu**4 + 6 * mu**2 * delta**2 + 3 * delta**4
    return lam * tau * (mu - kbar), lam * tau * m2, lam * tau * m3, lam * tau * m4


def cumulants(tau, params, jump=None):
    """(c1, c2, c3, c4): the first four cumulants of z = ln(S_T/F) under Double Heston (+ jumps), analytic (module docstring).
    tau: scalar or (n,); params: (10,) or (n, 10); jump: None, (3,) or (n, 3). Returns arrays of shape (n,) (or 0-d for scalar input)."""
    p = np.atleast_2d(_split(params)).astype(float)
    n = max(p.shape[0], np.size(tau))
    tau = np.broadcast_to(np.asarray(tau, float), (n,)).copy()
    p = np.broadcast_to(p, (n, 10))
    out = [np.zeros(n) for _ in range(4)]
    for j in (0, 5):
        cj = _heston_cumulants(tau, p[:, j], p[:, j + 1], p[:, j + 2], p[:, j + 3], p[:, j + 4])
        out = [o + c for o, c in zip(out, cj)]
    if jump is not None:
        jp = np.broadcast_to(np.atleast_2d(np.asarray(jump, float)), (n, 3))
        out = [o + c for o, c in zip(out, _jump_cumulants(tau, jp[:, 0], jp[:, 1], jp[:, 2]))]
    return tuple(out)


# ------------------------------------------------------------------------------------------------ COS core
XB_MAX = 6.0     # a call leg is integrated directly only while e^(x+b) <= e^6; beyond, the out-of-the-money call comes from the put leg by parity
def _cos_core(x, gid, tau, P, J, L, tol_abs, tol_rel, N0, Nmax, N_fixed, force_leg=None, max_cells=3_000_000):
    """Normalised price of the out-of-the-money leg for every row.
    x (n,) = ln(F/K); gid (n,) group index of each row; tau (G,), P (G, 10), J (G, 3) or None per group.
    Returns (u_leg (n,), leg_sign (n,) [+1 call leg, -1 put leg], info dict)."""
    n, G = x.size, tau.size
    c1, c2, _, c4 = cumulants(tau, P, J)
    w = L * np.sqrt(np.maximum(c2, 0.0) + np.sqrt(np.maximum(c4, 0.0)))
    a, b = c1 - w, c1 + w
    ba = b - a
    s = np.where(x < 0.0, 1.0, -1.0)           # +1: out-of-the-money call (K > F), -1: out-of-the-money put
    if force_leg is not None:                  # testing hook: price the other (in-the-money) leg directly instead of by parity
        s = np.full(n, 1.0 if force_leg == "call" else -1.0)
    ag, bg = a[gid], b[gid]
    if force_leg is None:                      # the call payoff e^z amplifies round-off by e^(x+b) when the interval reaches far right: use the put leg there
        s = np.where((s > 0) & (x + bg > XB_MAX), -1.0, s)
    cl = np.where(s > 0, np.maximum(ag, -x), ag)
    dl = np.where(s > 0, bg, np.minimum(bg, -x))
    ex_c, ex_d = np.exp(x + cl), np.exp(x + dl)  # e^x e^c and e^x e^d, both of moderate size
    u = np.zeros(n)
    N_used = np.zeros(G, dtype=int)
    converged = np.ones(G, dtype=bool)
    act = np.flatnonzero(cl < dl)               # rows with a non-empty payoff region
    k0 = 0
    k1 = N_fixed if N_fixed else N0
    first = True
    while act.size:
        grp, inv = np.unique(gid[act], return_inverse=True)
        k = np.arange(k0, k1, dtype=float)
        om_g = k[None, :] * np.pi / ba[grp][:, None]                       # (Ga, nk): u_k = k pi / (b-a)
        phi = np.exp(log_cf_double(om_g, tau[grp][:, None], [P[grp, i][:, None] for i in range(10)],
                                   None if J is None else [J[grp, i][:, None] for i in range(3)]))
        Fk = (phi * np.exp(-1j * om_g * a[grp][:, None])).real
        if k0 == 0:
            Fk[:, 0] *= 0.5
        block = np.zeros(act.size)
        cos_k_pi = np.where(k % 2 == 0, 1.0, -1.0)[None, :]
        step = max(1, max_cells // k.size)
        for i0 in range(0, act.size, step):
            sel = slice(i0, i0 + step)
            r = act[sel]
            om = om_g[inv[sel]]
            Fr, sr, a_r, c_r, d_r = Fk[inv[sel]], s[r], ag[r], cl[r], dl[r]
            acc = np.zeros(r.size)
            for leg in (1.0, -1.0):
                # Exactly one end of the payoff region [c, d] is interior; the other is a boundary of [a, b] where cos and sin are known
                # (call leg: d = b, cos(k pi) = (-1)^k, sin = 0.   put leg: c = a, cos = 1, sin = 0), so only one cos and one sin are needed.
                m = sr == leg
                if not m.any():
                    continue
                o, exc, exd = om[m], ex_c[r][m][:, None], ex_d[r][m][:, None]
                den = 1.0 + o * o
                if leg > 0:
                    th = o * (c_r[m] - a_r[m])[:, None]
                    ci, si = np.cos(th), np.sin(th)
                    chi_x = (cos_k_pi * exd - ci * exc - o * si * exc) / den          # e^x chi_k(c, b)
                    sgn_psi = -1.0
                else:
                    th = o * (d_r[m] - a_r[m])[:, None]
                    ci, si = np.cos(th), np.sin(th)
                    chi_x = (ci * exd - exc + o * si * exd) / den                     # e^x chi_k(a, d)
                    sgn_psi = 1.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    psi = np.where(o > 0, sgn_psi * si / np.where(o > 0, o, 1.0), (d_r[m] - c_r[m])[:, None])
                Gk = (2.0 / ba[gid[r]][m] * leg)[:, None] * (chi_x - psi)
                acc[m] = (Fr[m] * Gk).sum(axis=1)
            block[sel] = acc
        u[act] += block
        N_used[grp] = k1
        if N_fixed:
            break
        if not first:
            bad = np.abs(block) > np.maximum(tol_abs, tol_rel * np.abs(u[act]))
            bad_groups = np.zeros(G, dtype=bool)
            bad_groups[gid[act][bad]] = True
            if k1 >= Nmax:
                converged[bad_groups] = False
                break
            act = act[bad_groups[gid[act]]]
        first = False
        k0, k1 = k1, 2 * k1
    if N_fixed:
        converged[:] = True
    return np.maximum(u, 0.0), s, dict(a=a, b=b, N=N_used, converged=converged, c1=c1, c2=c2, c4=c4)


def _finish(u_leg, s, x, kind):
    """From the out-of-the-money leg to the requested option, by put-call parity: call - put = e^x - 1 (normalised)."""
    par = np.expm1(x)
    if kind == "call":
        return np.where(s > 0, u_leg, u_leg + par)
    if kind == "put":
        return np.where(s > 0, u_leg - par, u_leg)
    raise ValueError("kind must be 'call' or 'put'")


# Truncation width. Fang & Oosterlee's L = 10 is not enough here: with the fast variance factor at vol-of-vol ~2 (non-Feller, as calibrated
# on NIFTY) or with jumps, the density has tails far heavier than the cumulants alone suggest (at L = 10 the put leg lost ~3e-9 of probability
# mass, and L = 14 is still 1e-6 relative off for some far out-of-the-money options). So the width is CHECKED, not assumed: start at L0 = 14,
# reprice with 1.5 L, and keep widening (14, 21, 31.5, 47, 71) until the price no longer changes (|du| <= 2e-13 + 1e-9 u); every row
# reports the L it needed (info["L_used"]) and its last change (info["trunc_diff"]). Pass a number for L to switch the check off (faster: ~2x).
_DEFAULTS = dict(L="auto", L0=14.0, L_steps=4, trunc_abs=2e-13, trunc_rel=1e-9, tol_abs=1e-14, tol_rel=1e-10, N0=32, Nmax=32768)


def _build_groups(x, tau, params, jump):
    """Rows (flattened) -> groups of identical (tau, params, jump): returns xf, gid, gtau, P, J, shape."""
    x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
    shape = x.shape
    p = _split(params)
    if p.ndim > 1:
        p = np.broadcast_to(p, shape + (10,)).reshape(-1, 10)
    jarr = None if jump is None else np.asarray(jump, float)
    if jarr is not None and jarr.ndim > 1:
        jarr = np.broadcast_to(jarr, shape + (3,)).reshape(-1, 3)
    xf, tf = x.reshape(-1), tau.reshape(-1)
    if np.any(tf <= 0):
        raise ValueError("tau must be positive")
    if p.ndim == 1 and (jarr is None or jarr.ndim == 1):
        gtau = np.unique(tf)
        gid = np.searchsorted(gtau, tf)
        P = np.broadcast_to(p, (gtau.size, 10)).copy()
        J = None if jarr is None else np.broadcast_to(jarr, (gtau.size, 3)).copy()
    else:
        p2 = np.broadcast_to(p, (tf.size, 10)) if p.ndim == 1 else p
        cols = [tf[:, None], p2] + ([] if jarr is None else [np.broadcast_to(jarr, (tf.size, 3))])
        uniq, gid = np.unique(np.hstack(cols), axis=0, return_inverse=True)
        gid = gid.reshape(-1)
        gtau, P = uniq[:, 0].copy(), uniq[:, 1:11].copy()
        J = None if jarr is None else uniq[:, 11:14].copy()
    return xf, gid, gtau, P, J, shape


def normalized_price(x, tau, params, kind="call", jump=None, N=None, return_info=False, **kw):
    """u = price / (K DF) (calls by default) for x = ln(F/K), tau = time to expiry in years, params = the 10 Double Heston parameters.
    x, tau broadcast together; params is (10,) (shared) or (..., 10) matching the broadcast shape; jump None, (3,) or (..., 3).
    Rows with identical (tau, params, jump) share one characteristic-function evaluation. Pass N for a fixed number of terms.
    Keywords: L ("auto" = widen until converged, see _DEFAULTS; or a fixed number), tol_abs (1e-14) and tol_rel (1e-10) for the number of
    terms, N0 (32), Nmax (32768), leg (None, "call" or "put": force which payoff is integrated; the default prices the out-of-the-money
    one and gets the other by parity; used by the tests to check parity).
    return_info=True also gives a dict: N (terms per group), converged (terms, and width in auto mode), a, b, L_used, trunc_diff."""
    opt = dict(_DEFAULTS); opt.update(kw)
    leg = opt.get("leg")
    xf, gid, gtau, P, J, shape = _build_groups(x, tau, params, jump)
    n = xf.size

    def core(rows, L):
        u_leg, s, info = _cos_core(xf[rows], gid[rows], gtau, P, J, L, opt["tol_abs"], opt["tol_rel"], opt["N0"], opt["Nmax"], N, leg)
        return _finish(u_leg, s, xf[rows], kind), info

    rows = np.arange(n)
    if opt["L"] != "auto":
        out, info = core(rows, float(opt["L"]))
        info.update(L_used=np.full(n, float(opt["L"])), trunc_diff=np.full(n, np.nan))
    else:
        L = opt["L0"]
        out, info = core(rows, L)
        L_used, tdiff = np.full(n, L), np.full(n, np.inf)
        pending = rows
        for _ in range(opt["L_steps"]):
            L2 = 1.5 * L
            o2, i2 = core(pending, L2)
            d = np.abs(o2 - out[pending])
            out[pending] = o2
            L_used[pending], tdiff[pending] = L2, d
            info["N"] = np.maximum(info["N"], i2["N"])
            info["converged"] &= i2["converged"]
            info["a"], info["b"] = i2["a"], i2["b"]
            pending = pending[d > opt["trunc_abs"] + opt["trunc_rel"] * np.abs(o2)]
            L = L2
            if not pending.size:
                break
        if pending.size:                         # still moving after the last widening: say so
            info["converged"] = info["converged"] & ~np.isin(np.arange(gtau.size), gid[pending])
        info.update(L_used=L_used, trunc_diff=tdiff)
    out = out.reshape(shape)
    if return_info:
        info["row_group"] = gid.reshape(shape)
        info["trunc_diff"] = float(np.max(info["trunc_diff"])) if np.size(info["trunc_diff"]) else 0.0
        info["L_used"] = info["L_used"].reshape(shape)
        return out, info
    return out


def cos_price(params, F, K, T, DF, kind="call", jump=None, N=None, return_info=False, **kw):
    """COS price of European calls/puts on the forward F with strikes K (array or scalar), maturity T (years), discount factor DF, for ONE
    parameter set (10 Double Heston parameters). Same units as F. K <= 0 is invalid. Keywords as in normalized_price."""
    Ka = np.asarray(K, float)
    x = np.log(F / Ka)
    out = normalized_price(x, T, np.asarray(params, float), kind=kind, jump=jump, N=N, return_info=return_info, **kw)
    if return_info:
        u, info = out
        return DF * Ka * u, info
    return DF * Ka * out


def cos_price_many(params_matrix, F, K, T, DF, kind="call", jump=None, N=None, **kw):
    """COS prices for M parameter sets: params_matrix (M, 10); F, T, DF scalars or length M; K (nK,) shared or (M, nK); jump None, (3,) or
    (M, 3). Returns (M, nK) prices in the units of F. Each parameter set gets its own truncation range and its own number of terms."""
    P = np.atleast_2d(np.asarray(params_matrix, float))
    M = P.shape[0]
    Fv, Tv, Dv = (np.broadcast_to(np.asarray(v, float), (M,)) for v in (F, T, DF))
    Ka = np.asarray(K, float)
    Ka = np.broadcast_to(Ka, (M, Ka.shape[-1])) if Ka.ndim == 1 else Ka
    x = np.log(Fv[:, None] / Ka)
    tau = np.broadcast_to(Tv[:, None], x.shape)
    pp = np.broadcast_to(P[:, None, :], x.shape + (10,))
    jj = None
    if jump is not None:
        jj = np.broadcast_to(np.atleast_2d(np.asarray(jump, float))[:, None, :], x.shape + (3,)) if np.ndim(jump) > 1 else np.asarray(jump, float)
    u = normalized_price(x, tau, pp, kind=kind, jump=jj, N=N, **kw)
    return Dv[:, None] * Ka * u
