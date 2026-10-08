import sys, time, importlib.util
from pathlib import Path
import numpy as np
import mpmath as mp
ROOT = Path("/Users/vinayakmangal/Everything related to my projects/Dheston")
sys.path.insert(0, str(ROOT))
from dh2 import cos
spec = importlib.util.spec_from_file_location("dh_pricer", ROOT / "pricer" / "models.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
mp.mp.dps = 30

def ode_cf(u, tau, v0, kappa, theta, xi, rho):
    u = mp.mpc(u)
    kappa, theta, xi, rho, v0, tau = (mp.mpf(v) for v in (kappa, theta, xi, rho, v0, tau))
    f = lambda t, y: [0.5*xi**2*y[0]**2 - (kappa - rho*xi*1j*u)*y[0] - 0.5*(u*u + 1j*u), kappa*theta*y[0]]
    sol = mp.odefun(f, 0, [mp.mpc(0), mp.mpc(0)])
    D, C = sol(tau)
    return complex(C + D*v0)

sets = {
 "covid": (0.3, 0.8, 0.03, 1.5, -0.98),
 "fastmax": (0.005, 15.0, 0.002, 2.0, -0.41),
 "slowlow": (0.016, 0.0595, 0.159, 0.2, -0.86),
 "pos_rho": (0.04, 1.0, 0.04, 1.0, 0.9),
 "smallkappa": (0.04, 0.05, 0.04, 2.0, -0.9),
}
worst = {}
t0 = time.time()
for name, par in sets.items():
    for tau in (1/365, 0.05, 0.5, 1.0):
        for im in (0.0, -0.5, -1.0):
            for ur in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
                u = ur + 1j*im
                try:
                    ref = ode_cf(u, tau, *par)
                except Exception as e:
                    print("ode fail", name, tau, u, e); continue
                mine = complex(cos.log_cf_heston(u, tau, *par))
                proj = complex(P._log_cf_factor(np.array([u]), par[0], par[1], par[2], par[3], par[4], tau)[0])
                em, ep = abs(mine - ref), abs(proj - ref)
                # compare via exp as well (branch of log): CF values themselves
                cm, cp, cr = np.exp(mine), np.exp(proj), np.exp(ref)
                key = (name, tau, im)
                w = worst.setdefault(key, [0, 0, 0, 0])
                w[0] = max(w[0], em); w[1] = max(w[1], ep); w[2] = max(w[2], abs(cm - cr)); w[3] = max(w[3], abs(cp - cr))
for k, w in worst.items():
    flag = "  <<<" if (w[0] > 1e-9 or w[1] > 1e-9) else ""
    print(k, "mine logerr %.1e proj logerr %.1e | cf err mine %.1e proj %.1e" % tuple(w), flag)
print("time", time.time() - t0)
