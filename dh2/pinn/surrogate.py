"""Inference API of the Double Heston surrogate (for calibration and Greeks).  It approximates the exact Fourier pricer
(pricer/models.py) inside the box it was trained on; outside the box nothing is promised, so every call can return a domain flag.

    s = Surrogate.load("results/pinn/checkpoints/hyb_s0.pt")                 # or dat_s0.pt, ...
    u  = s.price_normalized(x, v1, v2, tau, theta)        # numpy in -> numpy out; torch in -> torch out (autograd works)
    iv = s.implied_vol(x, v1, v2, tau, theta)             # Black implied vol of the out-of-the-money option
    out = s.price_surface(F, K, T, DF, theta, v1, v2)     # a whole chain: rupee call/put prices, IV, in-domain flag
    g = s.greeks(F, K, T, DF, theta, v1, v2, S=spot)      # delta, gamma, dC/dv1, dC/dv2 by autograd

Conventions.  x = ln(F/K), u = C/(K*DF) (a call; a put is u - (e^x - 1)), theta = (kappa1, theta1, sigma1, rho1, kappa2, theta2,
sigma2, rho2) with sigma the pricer's xi, state (v1, v2).  From the walk-forward fits use dh2.pinn.data.eta_to_theta(eta)
(kappa2 = kappa1 + dk).  Accuracy numbers and the trained domain are in results/pinn/README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import model as MD
from . import pde as PDE

DEFAULT = Path(__file__).resolve().parents[2] / "results" / "pinn" / "checkpoints" / "final.pt"


class Surrogate:
    def __init__(self, model: MD.DHNet, domain: dict, info: dict | None = None, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.domain = domain
        self.info = info or {}
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------ construction
    @classmethod
    def load(cls, path=None, device: str = "cpu") -> "Surrogate":
        ck = torch.load(Path(path or DEFAULT), map_location="cpu", weights_only=False)
        m = MD.DHNet(ck["domain"], **ck["net_cfg"])
        m.load_state_dict(ck["state"])
        return cls(m, ck["domain"], ck.get("info"), device)

    # ------------------------------------------------------------ helpers
    def _prep(self, x, v1, v2, tau, theta):
        is_t = any(isinstance(a, torch.Tensor) for a in (x, v1, v2, tau, theta))
        ref = next(a for a in (x, v1, v2, tau, theta) if isinstance(a, torch.Tensor)) if is_t else None
        def t(a, grad=False):
            if isinstance(a, torch.Tensor):
                return a.to(self.device, torch.float32)
            return torch.as_tensor(np.asarray(a, np.float64), dtype=torch.float32, device=self.device)
        x, v1, v2, tau = (t(a) for a in (x, v1, v2, tau))
        th = t(theta)
        n = max(x.shape[0] if x.ndim else 1, th.shape[0] if th.ndim == 2 else 1, v1.numel(), v2.numel(), tau.numel())
        x, v1, v2, tau = (a.reshape(-1).expand(n) if a.numel() in (1, n) else a for a in (x, v1, v2, tau))
        th = th.reshape(1, 8).expand(n, 8) if th.ndim == 1 else th
        return is_t, x, v1, v2, tau, th

    # ------------------------------------------------------------ prices
    def price_normalized(self, x, v1, v2, tau, theta, batch: int = 262144):
        """u = C/(K*DF) for a call.  Shapes: x, v1, v2, tau scalars or (n,), theta (8,) or (n, 8).  numpy in -> numpy out (float64),
        torch in -> torch out (float32, differentiable with respect to the inputs)."""
        is_t, x, v1, v2, tau, th = self._prep(x, v1, v2, tau, theta)
        if is_t:
            return self.model(x, v1, v2, tau, th)
        with torch.no_grad():
            out = [self.model(x[s:s + batch], v1[s:s + batch], v2[s:s + batch], tau[s:s + batch], th[s:s + batch]) for s in range(0, len(x), batch)]
        return torch.cat(out).cpu().numpy().astype(np.float64)

    def otm_price(self, x, v1, v2, tau, theta):
        x = np.asarray(x, float)
        return D.exact_otm(self.price_normalized(x, v1, v2, tau, theta), x)

    def implied_vol(self, x, v1, v2, tau, theta):
        """Black implied volatility (annualised) of the surrogate's out-of-the-money price; NaN when that price is not invertible."""
        x = np.atleast_1d(np.asarray(x, float)); tau_a = np.broadcast_to(np.asarray(tau, float), x.shape)
        return D.implied_vol_unit(np.maximum(self.otm_price(x, v1, v2, tau_a, theta), 1e-14), x, tau_a)

    def in_domain(self, x, v1, v2, tau, theta):
        """True where the point lies inside the box the network was trained on."""
        d = self.domain
        x, v1, v2, tau = (np.atleast_1d(np.asarray(a, float)) for a in (x, v1, v2, tau))
        th = np.atleast_2d(np.asarray(theta, float))
        ok = (np.abs(x) <= d["x_max"]) & (tau >= d["tau_lo"] - 1e-12) & (tau <= d["tau_hi"]) & (v1 >= d["v_lo"][0]) & (v1 <= d["v_hi"][0]) & (v2 >= d["v_lo"][1]) & (v2 <= d["v_hi"][1])
        return ok & np.all((th >= np.array(d["theta_lo"]) - 1e-12) & (th <= np.array(d["theta_hi"]) + 1e-12), 1)

    def price_surface(self, F, K, T, DF, theta, v1, v2, is_call=None) -> dict:
        """Price a chain: one parameter set (theta, v1, v2) and quotes with their own F, K, T, DF (arrays of equal length).
        Returns rupee call and put prices, the price of the requested side, Black IV of the OTM side, and the in-domain flag."""
        F, K, T, DF = (np.atleast_1d(np.asarray(a, float)) for a in (F, K, T, DF))
        x = np.log(F / K)
        u = self.price_normalized(x, v1, v2, T, theta)
        call = K * DF * u
        put = call - DF * (F - K)
        iv = D.implied_vol_unit(np.maximum(D.exact_otm(u, x), 1e-14), x, T)
        out = {"call": call, "put": put, "iv": iv, "u": u, "in_domain": self.in_domain(x, v1, v2, T, theta)}
        if is_call is not None:
            out["price"] = np.where(np.asarray(is_call, bool), call, put)
        return out

    # ------------------------------------------------------------ Greeks (autograd)
    def greeks(self, F, K, T, DF, theta, v1, v2, S=None) -> dict:
        """Call Greeks by automatic differentiation of the network.  With C = K*DF*u(x, v1, v2, tau), x = ln(F/K):
        delta_F = K DF u_x / F, gamma_F = K DF (u_xx - u_x)/F^2 (derivatives with respect to the forward F);
        if the spot S is given, F = S exp((r-q) T) gives delta_S = (F/S) delta_F and gamma_S = (F/S)^2 gamma_F.
        dC_dv1, dC_dv2 = K DF u_v1, K DF u_v2 (per unit of variance); dC_dtau at fixed x, v is also returned.  Put delta = call delta - DF F/S."""
        F, K, T, DF = (np.atleast_1d(np.asarray(a, float)) for a in (F, K, T, DF))
        x = np.log(F / K)
        _, xt, a1, a2, tt, th = self._prep(x, v1, v2, T, theta)
        with torch.enable_grad():
            d = PDE.derivatives(self.model, xt, a1, a2, tt, th, create_graph=False)
        d = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in d.items()}
        kd = K * DF
        out = {"u": d["u"], "call": kd * d["u"], "delta_F": kd * d["ux"] / F, "gamma_F": kd * (d["uxx"] - d["ux"]) / F**2,
               "dC_dv1": kd * d["u1"], "dC_dv2": kd * d["u2"]}
        if S is not None:
            f = F / float(S)
            out["delta_S"] = f * out["delta_F"]; out["gamma_S"] = f * f * out["gamma_F"]
        return out
