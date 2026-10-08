"""The training runs of the ablation, one at a time:  python3 -m dh2.pinn.experiments <run name> [<run name> ...]

A run whose checkpoint already exists is skipped (nothing is retrained).  Runs:
  hyb_s{0,1,2}   hybrid (data + PDE + asymptotics), the three seeds of the final model
  dat_s{0,1,2}   data-only network, same architecture and steps (lambda_pde = 0)
  phy_s0         physics-only (no labels): PDE + asymptotics; the initial condition is built in
  spec_dat_s0    data-only with the spec's gate g = sqrt(tau) instead of the vega gate (architecture ablation)
  de_dat_<n>, de_hyb_<n>   data-efficiency: n supervised samples (1e3, 1e4, 1e5; 1e6 is hyb_s0 / dat_s0)
"""
from __future__ import annotations

import sys

from . import train as TR


def spec(name: str, cfg: dict) -> dict:
    a = cfg["ablation"]
    if name.startswith(("hyb_s", "dat_s")):
        seed = int(name[-1])
        return dict(mode="hybrid" if name.startswith("hyb") else "data", seed=seed, n_data=None, steps=None, overrides={})
    if name == "phy_s0":
        return dict(mode="physics", seed=0, n_data=None, steps=a["physics_steps"], overrides={"pde_every": 1, "lambda_pde": 1.0})
    if name == "spec_dat_s0":
        return dict(mode="data", seed=0, n_data=None, steps=None, overrides={"g": "sqrt"})
    if name.startswith(("de_dat_", "de_hyb_")):
        n = int(name.split("_")[-1])
        return dict(mode="hybrid" if name.startswith("de_hyb") else "data", seed=0, n_data=n, steps=a["data_eff_steps"][n], overrides={})
    raise ValueError(name)


def run(name: str, cfg=None, dom=None):
    cfg = cfg or TR.load_cfg()
    f = TR.CKPT / f"{name}.pt"
    if f.exists():
        print(f"{name}: checkpoint exists, skipped", flush=True)
        return None
    dom = dom or TR.get_domain(cfg)
    s = spec(name, cfg)
    r = TR.train(name, s["mode"], cfg, dom, seed=s["seed"], n_data=s["n_data"], steps=s["steps"], overrides=s["overrides"], verbose=True)
    print(f"{name}: done in {r['info']['wall_s']:.0f}s, val IV {r['info']['val_iv']}", flush=True)
    return r


if __name__ == "__main__":
    cfg = TR.load_cfg(); dom = TR.get_domain(cfg)
    for n in sys.argv[1:]:
        run(n, cfg, dom)
