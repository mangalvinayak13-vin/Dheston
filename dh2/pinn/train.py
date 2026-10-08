"""Training of the Double Heston surrogate: data-only, physics-only and hybrid variants share one loop.

Loss = lambda_data * data + lambda_pde * pde + lambda_bc * bc, all expressed in (volatility points)^2 so the terms are comparable:
  data  = mean(((u_net - u_exact)/vega_pt)^2)      supervised error vs exact prices, vega-weighted = implied-vol error to first order
  pde   = mean((tau * R / vega_pt)^2)              PDE residual R at collocation points, same scaling (R is a rate, so times tau)
  bc    = asymptotics: the network correction g*NN must vanish where the price is known exactly -- far in the wings (|x| = 8..12
          Black standard deviations: u -> 0 / e^x - 1) and at huge variance (u -> e^x) -- penalised relative to a small price scale.
The initial condition u(x, v, 0) = max(e^x - 1, 0) is built into the architecture (g(0) = 0), so it needs no loss term.

Adam with a cosine schedule, optionally Wang-Teng-Perdikaris gradient-statistics weights, optionally residual-based adaptive
resampling of the collocation points, optionally L-BFGS at the end.  Progress is written to logs/progress/pinn_<name>.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import data as D
from . import model as MD
from . import pde as PDE

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "pinn"
CKPT = ROOT / "results" / "pinn" / "checkpoints"
TABLES = ROOT / "results" / "pinn" / "tables"
PROG = ROOT / "logs" / "progress"
CFG = ROOT / "configs" / "pinn.yaml"


def load_cfg(path=None) -> dict:
    cfg = yaml.safe_load(open(path or CFG))
    d = cfg["domain"]
    assert abs(d["x_max"] - D.X_MAX) < 1e-12 and abs(d["tau_max"] - D.TAU_MAX) < 1e-12 and abs(d["tau_min_days"] / 365 - D.TAU_MIN) < 1e-12
    assert abs(d["v_range"][0] - D.V_MIN) < 1e-12 and abs(d["v_range"][1] - D.V_MAX) < 1e-12
    return cfg


def get_domain(cfg: dict) -> dict:
    return D.build_domain(cfg["domain"]["margin"], 0.05, cfg["domain"]["holdout_frac"])


def progress(name: str, done: int, total: int) -> None:
    PROG.mkdir(parents=True, exist_ok=True)
    tmp = PROG / f".pinn_{name}.tmp"
    tmp.write_text(json.dumps({"done": int(done), "total": int(total)}))
    tmp.replace(PROG / f"pinn_{name}.json")


# ------------------------------------------------------------------ cached labelled data
_KEYS = ["x", "v1", "v2", "tau", "theta", "u", "u_otm", "iv", "vega"]


def labelled(cfg: dict, dom: dict, kind: str, n: int | None = None) -> dict:
    """kind 'train' (n samples, the large set; smaller sizes are its first n rows), 'val' (in-distribution monitor) or 'box'
    (held-out parameter sets drawn from the box only).  Cached under data/cache/pinn/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    if kind == "train":
        n, seed, p = d["n_train"], d["seed_train"], cfg["domain"]["p_emp"]
    elif kind == "val":
        n, seed, p = d["n_val"], d["seed_val"], cfg["domain"]["p_emp"]
    elif kind == "box":
        n, seed, p = cfg["validate"]["n_box"] * d["n_x_per"], d["seed_val"] + 1, 0.0
    else:
        raise ValueError(kind)
    f = CACHE / f"{kind}_{n}_{seed}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k].astype(np.float64) for k in _KEYS}
    out = D.make_dataset(n, dom, seed, d["n_x_per"], p)
    np.savez(f, **{k: out[k].astype(np.float32 if k != "u" else np.float64) for k in _KEYS})
    return out


def to_t(a: dict, keys, device, n=None):
    return {k: torch.tensor(np.asarray(a[k] if n is None else a[k][:n]), dtype=torch.float32, device=device) for k in keys}


# ------------------------------------------------------------------ evaluation helpers
@torch.no_grad()
def predict(model, inp: dict, device, batch: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    n = len(inp["x"])
    for s in range(0, n, batch):
        b = {k: torch.tensor(np.asarray(inp[k][s:s + batch]), dtype=torch.float32, device=device) for k in ("x", "v1", "v2", "tau", "theta")}
        out.append(model(b["x"], b["v1"], b["v2"], b["tau"], b["theta"]).cpu().numpy().astype(np.float64))
    model.train()
    return np.concatenate(out)


def iv_errors(u_pred, ref: dict, p_min: float = D.P_MIN):
    """Implied-vol error in volatility points for every point whose exact OTM price is at least p_min; returns (err, mask)."""
    x, tau = ref["x"], ref["tau"]
    otm = D.exact_otm(np.asarray(u_pred, float), x)
    iv = D.implied_vol_unit(np.maximum(otm, 1e-14), x, tau)
    mask = (ref["u_otm"] >= p_min) & np.isfinite(ref["iv"])
    err = 100.0 * (iv - ref["iv"])
    err = np.where(np.isfinite(err), err, np.nan)
    return err, mask


def summarise(err, mask) -> dict:
    e = np.abs(err[mask]); nan = int(np.isnan(e).sum()); e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {"n": int(mask.sum()), "nan": nan, "mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "p99": float(np.percentile(e, 99)), "max": float(e.max()), "rmse": float(np.sqrt((e ** 2).mean()))}


# ------------------------------------------------------------------ the training loop
def _bc_inputs(n: int, dom: dict, rng) -> dict:
    """Asymptotic points: half far in the wings (8..12 Black sd, with x beyond the data box allowed), half at huge variance."""
    a = D.sample_inputs(n, dom, rng)
    sd = np.sqrt(D.total_variance(a["theta"], a["v1"], a["v2"], a["tau"]))
    half = n // 2
    a["x"][:half] = np.where(rng.random(half) < 0.5, -1.0, 1.0) * sd[:half] * rng.uniform(8.0, 12.0, half)
    a["x"][:half] = np.clip(a["x"][:half], -4.0, 4.0)
    a["v1"][half:] = np.exp(rng.uniform(np.log(50.0), np.log(500.0), n - half))
    a["v2"][half:] = np.exp(rng.uniform(np.log(50.0), np.log(500.0), n - half))
    a["tau"][half:] = rng.uniform(0.1, D.TAU_MAX, n - half)
    return a


def _grad_stats(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([t.reshape(-1).abs() for t in g if t is not None])
    return flat.max(), flat.mean()


def train(name: str, mode: str, cfg: dict, dom: dict, seed: int = 0, n_data: int | None = None, steps: int | None = None,
          device: str | None = None, val: dict | None = None, overrides: dict | None = None, verbose: bool = True, save: bool = True) -> dict:
    """mode: 'hybrid' (data + pde + bc), 'data' (data only, same architecture), 'physics' (pde + bc, no labels).
    overrides: dict of train/net settings that replace the config (used by the ablation)."""
    t_cfg = dict(cfg["train"]); n_cfg = dict(cfg["net"])
    for k, v in (overrides or {}).items():
        (n_cfg if k in n_cfg else t_cfg)[k] = v
    device = device or cfg["device"]
    torch.set_num_threads(cfg.get("threads", 2))
    torch.manual_seed(seed); rng = np.random.default_rng(10_000 + seed)
    steps = steps or t_cfg["steps"]
    use_data, use_pde = mode in ("hybrid", "data"), mode in ("hybrid", "physics")
    netdom = D.net_domain(dom)
    model = MD.DHNet(netdom, **n_cfg).to(device)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=t_cfg["lr"])
    lam = {"data": t_cfg["lambda_data"] if use_data else 0.0, "pde": t_cfg["lambda_pde"] if use_pde else 0.0, "bc": t_cfg["lambda_bc"]}
    if use_data:
        full = labelled(cfg, dom, "train")
        n_data = n_data or len(full["x"])
        dd = to_t(full, ["x", "v1", "v2", "tau", "theta", "u", "vega"], device, n_data)
        n_data = len(dd["x"])
        B_d = min(t_cfg["batch_data"], n_data)
    if use_pde:
        pool = D.sample_inputs(t_cfg["rar_pool"] if t_cfg["rar"] else 200_000, dom, rng)
        pp = to_t(pool, ["x", "v1", "v2", "tau", "theta"], device)
        n_pool = len(pp["x"])
    bc = to_t(_bc_inputs(50_000, dom, rng), ["x", "v1", "v2", "tau", "theta"], device)
    if val is None:
        val = labelled(cfg, dom, "val")
    ref = "data" if use_data else "pde"
    hist = []
    last = {}
    t0 = time.time()

    def lr_at(s):
        c = 0.5 * (1 + math.cos(math.pi * s / steps))
        return t_cfg["lr_end"] + (t_cfg["lr"] - t_cfg["lr_end"]) * c

    def compute_losses(idx_d=None, idx_p=None, idx_b=None):
        L = {}
        if use_data:
            b = {k: v[idx_d] for k, v in dd.items()}
            u = model(b["x"], b["v1"], b["v2"], b["tau"], b["theta"])
            L["data"] = (((u - b["u"]) / b["vega"]) ** 2).mean()
        if use_pde and idx_p is not None:
            b = {k: v[idx_p] for k, v in pp.items()}
            nr, _, _ = PDE.normalised_residual(model, b["x"], b["v1"], b["v2"], b["tau"], b["theta"], t_cfg["pde_floor"])
            L["pde"] = (nr ** 2).mean()
        b = {k: v[idx_b] for k, v in bc.items()}
        c = model.correction(b["x"], b["v1"], b["v2"], b["tau"], b["theta"])
        sc = torch.where(b["v1"] > 10.0, torch.full_like(c, 1e-3), torch.full_like(c, 2e-5))
        L["bc"] = ((c / sc) ** 2).mean()
        return L

    def monitor(step):
        u = predict(model, val, device)
        err, mask = iv_errors(u, val)
        s = summarise(err, mask)
        return s

    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx_d = torch.randint(0, n_data, (B_d,), device=device) if use_data else None
        do_pde = use_pde and (step % t_cfg.get("pde_every", 1) == 0)             # the PDE term costs ~10x a data term: evaluated every k-th step
        idx_p = torch.randint(0, n_pool, (t_cfg["batch_pde"],), device=device) if do_pde else None
        idx_b = torch.randint(0, len(bc["x"]), (t_cfg["batch_bc"],), device=device)
        L = compute_losses(idx_d, idx_p, idx_b)
        if t_cfg["adaptive_weights"] and step > 0 and step % t_cfg["adapt_every"] == 0 and len(L) > 1:
            mx_ref, _ = _grad_stats(L[ref], params)
            for k in L:
                if k == ref:
                    continue
                _, mean_k = _grad_stats(L[k], params)
                hat = float(mx_ref / (mean_k + 1e-12))
                lam[k] = float(np.clip(t_cfg["adapt_alpha"] * lam[k] + (1 - t_cfg["adapt_alpha"]) * hat, 1e-3, 1e3))
        total = sum(lam[k] * L[k] for k in L)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if t_cfg["rar"] and use_pde and step > 0 and step % t_cfg["rar_every"] == 0:
            cand = D.sample_inputs(t_cfg["rar_cand"], dom, rng)
            ct = to_t(cand, ["x", "v1", "v2", "tau", "theta"], device)
            rs = []
            for s0 in range(0, len(ct["x"]), 4096):
                sl = {k: v[s0:s0 + 4096] for k, v in ct.items()}
                nr, _, _ = PDE.normalised_residual(model, sl["x"], sl["v1"], sl["v2"], sl["tau"], sl["theta"], t_cfg["pde_floor"], create_graph=False)
                rs.append(nr.detach().abs())
            r = torch.cat(rs)
            p = r / r.mean() + 1.0
            n_new = min(n_pool // 4, r.numel() // 2)
            pick = torch.multinomial(p / p.sum(), n_new, replacement=False)
            where = torch.randperm(n_pool, device=device)[:n_new]
            for k in pp:
                pp[k][where] = ct[k][pick]
        last.update({k: v.detach() for k, v in L.items()})
        if (step + 1) % 25 == 0 or step == steps - 1:
            progress(name, step + 1, steps)
        if (step + 1) % t_cfg["eval_every"] == 0 or step == steps - 1:
            s = monitor(step)
            row = {"step": step + 1, "time_s": time.time() - t0, **{f"L_{k}": float(v) for k, v in last.items()}, **{f"lam_{k}": lam[k] for k in lam}}
            row.update({f"val_iv_{k}": v for k, v in s.items()})
            hist.append(row)
            if verbose:
                print(f"[{name}] step {step + 1}/{steps} {time.time() - t0:.0f}s " + " ".join(f"{k}={float(v):.3e}" for k, v in last.items())
                      + f" | val IV mean {s.get('mean', float('nan')):.4f} p95 {s.get('p95', float('nan')):.4f} max {s.get('max', float('nan')):.3f}", flush=True)
    # ---- optional L-BFGS
    if t_cfg["lbfgs_iters"] > 0:
        s_before = monitor(steps)
        state_before = {k: v.clone() for k, v in model.state_dict().items()}
        nb = min(30000, n_data) if use_data else 0
        ix_d = torch.randint(0, n_data, (nb,), device=device) if use_data else None
        ix_p = torch.randint(0, n_pool, (6000,), device=device) if use_pde else None
        ix_b = torch.arange(0, 2000, device=device)
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=t_cfg["lbfgs_iters"], history_size=50, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            lb.zero_grad()
            Lc = compute_losses(ix_d, ix_p, ix_b)
            tot = sum(lam[k] * Lc[k] for k in Lc)
            tot.backward()
            return tot
        try:
            lb.step(closure)
        except Exception as e:                                                  # noqa: BLE001
            if verbose:
                print("L-BFGS failed:", e)
        s_after = monitor(steps)
        if not (s_after.get("mean", 1e9) < s_before.get("mean", 1e9)):
            model.load_state_dict(state_before)
            s_after = dict(s_before, reverted=True)
        hist.append({"step": steps + 1, "time_s": time.time() - t0, "lbfgs": True, **{f"val_iv_{k}": v for k, v in s_after.items()}})
        if verbose:
            print(f"[{name}] L-BFGS: val IV mean {s_before.get('mean'):.4f} -> {s_after.get('mean'):.4f}", flush=True)
    wall = time.time() - t0
    final = monitor(steps)
    info = {"name": name, "mode": mode, "seed": seed, "n_data": int(n_data) if use_data else 0, "steps": steps, "wall_s": wall, "device": device,
            "val_iv": final, "params": MD.count_params(model), "train_cfg": t_cfg, "net_cfg": n_cfg, "lam_final": lam}
    if save:
        CKPT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "net_cfg": n_cfg, "domain": netdom, "info": info}, CKPT / f"{name}.pt")
        import csv
        keys = sorted({k for r in hist for k in r})
        with open(TABLES / f"train_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in hist]
    progress(name, steps, steps)
    return {"model": model, "info": info, "hist": hist}
