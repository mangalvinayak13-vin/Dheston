"""Physics-informed surrogate pricer for the diffusion Double Heston model (no jumps), in normalised coordinates.

x = ln(F/K), u = C/(K*DF), Theta = (kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2); the state is (v1, v2) and the
maturity is tau.  The network approximates the exact Fourier pricer (pricer/models.py); it is a surrogate, not a replacement.
See results/pinn/README.md for what it achieves and what it does not.
"""
