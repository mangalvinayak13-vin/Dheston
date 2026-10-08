"""The surrogate network:  u = u_BS(x, w) + g(tau) * NN(x, v1, v2, tau, Theta)   in normalised coordinates x = ln(F/K), u = C/(K DF).

u_BS = e^x N(d1) - N(d2), d1 = (x + w/2)/sqrt(w), d2 = d1 - sqrt(w), with the model's EXPECTED total variance
    w = sum_j [theta_j tau + (v_j - theta_j)(1 - exp(-kappa_j tau))/kappa_j].
At tau = 0 the Black part is the payoff max(e^x - 1, 0) and g(0) = 0 removes the network, so the initial condition holds EXACTLY
(by construction, not by a loss term).  The network sees normalised inputs and some engineered features (d1, log w); activations are
smooth (SiLU or tanh) because the PDE loss needs second derivatives.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

SQRT1_2 = 1.0 / math.sqrt(2.0)
_POS = [1, 1, 1, 0, 1, 1, 1, 0]          # Theta entries normalised on a log scale (rho entries linearly)


def ndtr(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-z * SQRT1_2)


def total_variance(x, v1, v2, tau, th):
    """Expected integrated variance w (torch); th (B, 8) = (k1, th1, s1, r1, k2, th2, s2, r2)."""
    k1, m1, k2, m2 = th[:, 0], th[:, 1], th[:, 4], th[:, 5]
    return (m1 * tau + (v1 - m1) * (-torch.expm1(-k1 * tau)) / k1) + (m2 * tau + (v2 - m2) * (-torch.expm1(-k2 * tau)) / k2)


def bs_unit(x, w, tau=None):
    """Black-76 call in normalised form.  Exactly the payoff max(e^x - 1, 0) where tau == 0 (if tau is given)."""
    sw = torch.sqrt(w.clamp_min(1e-10))
    d1 = (x + 0.5 * w) / sw
    u = torch.exp(x) * ndtr(d1) - ndtr(d1 - sw)
    if tau is not None:
        u = torch.where(tau <= 0, torch.relu(torch.exp(x) - 1.0), u)
    return u


class DHNet(nn.Module):
    """u = u_BS + g(tau) NN.  domain: dict of numpy bounds (stored in the module so a checkpoint is self-contained)."""

    def __init__(self, domain: dict, width: int = 256, depth: int = 5, act: str = "silu", g: str = "sqrt", out_scale: float = 0.05):
        super().__init__()
        self.cfg = dict(width=width, depth=depth, act=act, g=g, out_scale=out_scale)
        self.out_scale = out_scale
        self.g_kind = g
        th_lo = np.array(domain["theta_lo"], float); th_hi = np.array(domain["theta_hi"], float)
        pos = np.array(_POS, bool)
        lo = np.where(pos, np.log(np.where(pos, th_lo, 1.0)), th_lo); hi = np.where(pos, np.log(np.where(pos, th_hi, 1.0)), th_hi)
        self.register_buffer("th_mid", torch.tensor(0.5 * (lo + hi), dtype=torch.float32))
        self.register_buffer("th_half", torch.tensor(0.5 * (hi - lo), dtype=torch.float32))
        self.register_buffer("th_pos", torch.tensor(pos))
        v_lo, v_hi = math.log(float(np.min(domain["v_lo"]))), math.log(float(np.max(domain["v_hi"])))
        self.register_buffer("v_mid", torch.tensor(0.5 * (v_lo + v_hi), dtype=torch.float32))
        self.register_buffer("v_half", torch.tensor(0.5 * (v_hi - v_lo), dtype=torch.float32))
        self.x_max = float(domain["x_max"]); self.tau_max = float(domain["tau_hi"]); self.tau_min = float(domain["tau_lo"])
        self.n_in = 15
        a = {"silu": nn.SiLU, "tanh": nn.Tanh}[act]
        layers = [nn.Linear(self.n_in, width), a()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), a()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1); self.net[-1].bias.zero_()

    # ---- pieces
    def features(self, x, v1, v2, tau, th, w=None):
        if w is None:
            w = total_variance(x, v1, v2, tau, th)
        tc = tau.clamp_min(1e-8)
        z = (x + 0.5 * w) / torch.sqrt(w.clamp_min(1e-10))
        thn = torch.where(self.th_pos, torch.log(th), th)
        thn = (thn - self.th_mid) / self.th_half
        f = torch.stack([
            x / self.x_max,
            torch.tanh(z / 3.0),
            2.0 * torch.sqrt(tc / self.tau_max) - 1.0,
            (torch.log(tc) - 0.5 * math.log(self.tau_min * self.tau_max)) / (0.5 * math.log(self.tau_max / self.tau_min)),
            (torch.log(v1) - self.v_mid) / self.v_half,
            (torch.log(v2) - self.v_mid) / self.v_half,
            (torch.log(w.clamp_min(1e-10)) + 6.0) / 6.0,
        ], -1)
        return torch.cat([f, thn], -1)

    def g(self, tau, x=None, w=None):
        """Gate on the network output; g(0) = 0 for every kind, so the initial condition is exact.
        'sqrt': g = sqrt(tau) (the spec).  'vega': g = Black vega at the comparison volatility, e^x phi(d1) sqrt(tau), so that
        g*NN = vega * (implied-vol shift) and the network learns the smile as a volatility correction (a variant, see ablation)."""
        if self.g_kind == "sqrt":
            return torch.sqrt(tau.clamp_min(0.0) + 1e-30) * (tau > 0)
        if self.g_kind == "tau":
            return tau
        if self.g_kind in ("vega", "vegat"):
            sw = torch.sqrt(w.clamp_min(1e-10))
            d1 = (x + 0.5 * w) / sw
            gate = torch.exp(x) * torch.exp(-0.5 * d1 ** 2) * 0.3989422804014327
            if self.g_kind == "vegat":
                # plus a fat (exponential) tail: the Black vega dies like exp(-d1^2/2), but the model's wing prices (fat tails) do not,
                # so a pure vega gate cannot represent them (measured: all the worst errors were 6+ Black standard deviations out)
                gate = gate + 0.02 * torch.exp(-(torch.sqrt(1.0 + d1 ** 2) - 1.0) / 4.0)
            return gate * torch.sqrt(tau.clamp_min(0.0) + 1e-30) * (tau > 0)
        raise ValueError(self.g_kind)

    def correction(self, x, v1, v2, tau, th):
        w = total_variance(x, v1, v2, tau, th)
        return self.g(tau, x, w) * self.out_scale * self.net(self.features(x, v1, v2, tau, th, w)).squeeze(-1)

    def forward(self, x, v1, v2, tau, th):
        w = total_variance(x, v1, v2, tau, th)
        ub = bs_unit(x, w, tau)
        nn_out = self.net(self.features(x, v1, v2, tau, th, w)).squeeze(-1)
        return ub + self.g(tau, x, w) * self.out_scale * nn_out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
