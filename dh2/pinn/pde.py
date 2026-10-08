"""The pricing PDE in normalised coordinates, its residual for a network, and an independent numerical verification of the PDE itself.

With x = ln(F/K), u = C/(K DF), F = S e^{(r-q) tau}, DF = e^{-r tau}, the Double Heston PDE (two independent variance factors, each
correlated with the index) is

  u_tau = 1/2 (v1+v2) (u_xx - u_x) + sum_j [ kappa_j (theta_j - v_j) u_vj + 1/2 sigma_j^2 v_j u_vjvj + rho_j sigma_j v_j u_xvj ],
  u(x, v1, v2, 0) = max(e^x - 1, 0).

I derived it from  C_t + (r-q) S C_S + 1/2 (v1+v2) S^2 C_SS + sum_j [kappa_j(theta_j - v_j) C_vj + 1/2 sigma_j^2 v_j C_vjvj
+ rho_j sigma_j v_j S C_Svj] - r C = 0  with C = K e^{-r tau} u(x, v, tau): C_t = K DF [r u - (r-q) u_x - u_tau], S C_S = K DF u_x,
S^2 C_SS = K DF (u_xx - u_x), S C_Svj = K DF u_xvj, so r and q cancel.  `verify_pde_fd` checks it numerically against the PROJECT's
exact pricer (finite differences of exact prices), so the network is never trained on an unchecked equation.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .model import total_variance


# ------------------------------------------------------------------ residual for a network (autograd)
def derivatives(model, x, v1, v2, tau, th, create_graph: bool = True):
    """u and the nine derivatives the PDE needs, by autograd.  x, v1, v2, tau are (B,) tensors, th (B,8); derivatives of u are with
    respect to the physical variables (the model normalises internally)."""
    x = x.detach().clone().requires_grad_(True)
    v1 = v1.detach().clone().requires_grad_(True)
    v2 = v2.detach().clone().requires_grad_(True)
    tau = tau.detach().clone().requires_grad_(True)
    u = model(x, v1, v2, tau, th)
    ux, u1, u2, ut = torch.autograd.grad(u.sum(), (x, v1, v2, tau), create_graph=True)
    uxx, ux1, ux2 = torch.autograd.grad(ux.sum(), (x, v1, v2), create_graph=True)
    u11 = torch.autograd.grad(u1.sum(), v1, create_graph=True)[0]
    u22 = torch.autograd.grad(u2.sum(), v2, create_graph=True)[0]
    d = dict(u=u, ux=ux, u1=u1, u2=u2, ut=ut, uxx=uxx, ux1=ux1, ux2=ux2, u11=u11, u22=u22)
    if not create_graph:
        d = {k: v.detach() for k, v in d.items()}
    return d


def rhs_terms(d, v1, v2, th):
    """The right-hand side of the PDE, term by term (a dict), so tests can show the size of each."""
    k1, m1, s1, r1, k2, m2, s2, r2 = (th[:, i] for i in range(8))
    return {
        "diffusion": 0.5 * (v1 + v2) * (d["uxx"] - d["ux"]),
        "drift1": k1 * (m1 - v1) * d["u1"], "drift2": k2 * (m2 - v2) * d["u2"],
        "volvol1": 0.5 * s1**2 * v1 * d["u11"], "volvol2": 0.5 * s2**2 * v2 * d["u22"],
        "cross1": r1 * s1 * v1 * d["ux1"], "cross2": r2 * s2 * v2 * d["ux2"],
    }


def pde_residual(model, x, v1, v2, tau, th, create_graph: bool = True):
    """u_tau - RHS, and the derivative dict."""
    d = derivatives(model, x, v1, v2, tau, th, create_graph)
    t = rhs_terms(d, v1, v2, th)
    res = d["ut"] - sum(t.values())
    return res, d


def vega_scale(x, v1, v2, tau, th, floor: float):
    """Black vega per volatility point at the model's expected volatility sqrt(w/tau) (detached): turns a price-like quantity into
    'volatility points'.  The residual is also multiplied by tau (the PDE is a rate), so  tau*R/vega  is roughly a local IV error in points."""
    with torch.no_grad():
        w = total_variance(x, v1, v2, tau, th)
        sw = torch.sqrt(w.clamp_min(1e-10))
        d1 = (x + 0.5 * w) / sw
        vega = torch.exp(x) * torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * torch.sqrt(tau) / 100.0
    return vega.clamp_min(floor)


def normalised_residual(model, x, v1, v2, tau, th, floor: float = 1.0e-5, create_graph: bool = True):
    res, d = pde_residual(model, x, v1, v2, tau, th, create_graph)
    return tau * res / vega_scale(x, v1, v2, tau, th, floor), res, d


# ------------------------------------------------------------------ independent verification of the PDE (finite differences of exact prices)
_FD1 = {5: (np.array([1, -8, 0, 8, -1]) / 12.0, 5), 7: (np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0, 7)}
_FD2 = {5: np.array([-1, 16, -30, 16, -1]) / 12.0, 7: np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0}


def _stencil(n: int):
    """Offsets -m..m for an n-point stencil, with weights for the first and second derivative (unit step)."""
    m = n // 2
    off = np.arange(-m, m + 1)
    w1 = _FD1[n][0]; w2 = _FD2[n]
    return off, w1, w2


def verify_pde_fd(theta, v1, v2, tau, x, steps=None, n_pts: int = 7, price_fn=None) -> dict:
    """Evaluate every PDE term at (x, v1, v2, tau) by high-order finite differences of EXACT prices and return the residual.

    price_fn(theta, v1s, v2s, taus, xs) -> normalised call u for arrays of equal length (default: the project's pricer).
    Steps: hx, h_v1, h_v2, h_tau (defaults scale with the local width).  The mixed derivative uses the tensor product of the
    first-derivative stencil in x and in v_j.  Returns the terms, the residual, and the scale (sum of absolute terms)."""
    from .data import pricer_call
    if price_fn is None:
        def price_fn(th, a, b, t, xs):
            return pricer_call(np.repeat(th[None], len(a), 0), a, b, t, np.asarray(xs).reshape(len(a), -1))
    th = np.asarray(theta, float)
    from .data import total_variance as tv
    w = float(tv(th, v1, v2, tau))
    sd = math.sqrt(w)
    hx = steps.get("hx", 0.25 * sd) if steps else 0.25 * sd
    h1 = steps.get("h1", 0.25 * v1) if steps else 0.25 * v1
    h2 = steps.get("h2", 0.25 * v2) if steps else 0.25 * v2
    ht = steps.get("ht", 0.10 * tau) if steps else 0.10 * tau
    off, w1, w2 = _stencil(n_pts)

    def u_at(dx, d1, d2, dt):
        """prices on a set of displaced points; each displaced point gets its own call (vectorised over rows)."""
        dx = np.atleast_1d(dx).astype(float)
        m = len(dx)
        a = np.full(m, v1) + d1; b = np.full(m, v2) + d2; t = np.full(m, tau) + dt
        out = price_fn(th, a, b, t, (x + dx).reshape(m, 1))
        return out.reshape(m)

    z = np.zeros(len(off))
    ux = np.dot(w1, u_at(off * hx, z, z, z)) / hx
    uxx = np.dot(w2, u_at(off * hx, z, z, z)) / hx**2
    u0 = u_at([0.0], [0.0], [0.0], [0.0])[0]
    u1 = np.dot(w1, u_at(z, off * h1, z, z)) / h1
    u11 = np.dot(w2, u_at(z, off * h1, z, z)) / h1**2
    u2 = np.dot(w1, u_at(z, z, off * h2, z)) / h2
    u22 = np.dot(w2, u_at(z, z, off * h2, z)) / h2**2
    ut = np.dot(w1, u_at(z, z, z, off * ht)) / ht
    # mixed: tensor product of first-derivative stencils
    O1, O2 = np.meshgrid(off, off, indexing="ij")
    W = np.outer(w1, w1)
    ux1 = np.sum(W * u_at((O1 * hx).ravel(), (O2 * h1).ravel(), np.zeros(O1.size), np.zeros(O1.size)).reshape(O1.shape)) / (hx * h1)
    ux2 = np.sum(W * u_at((O1 * hx).ravel(), np.zeros(O1.size), (O2 * h2).ravel(), np.zeros(O1.size)).reshape(O1.shape)) / (hx * h2)
    k1, m1, s1, r1, k2, m2, s2, r2 = th
    terms = {
        "diffusion": 0.5 * (v1 + v2) * (uxx - ux),
        "drift1": k1 * (m1 - v1) * u1, "drift2": k2 * (m2 - v2) * u2,
        "volvol1": 0.5 * s1**2 * v1 * u11, "volvol2": 0.5 * s2**2 * v2 * u22,
        "cross1": r1 * s1 * v1 * ux1, "cross2": r2 * s2 * v2 * ux2,
    }
    rhs = sum(terms.values())
    return {"u": u0, "ut": ut, "rhs": rhs, "residual": ut - rhs, "terms": terms,
            "scale": abs(ut) + sum(abs(t) for t in terms.values()), "derivs": dict(ux=ux, uxx=uxx, u1=u1, u2=u2, u11=u11, u22=u22, ux1=ux1, ux2=ux2)}
