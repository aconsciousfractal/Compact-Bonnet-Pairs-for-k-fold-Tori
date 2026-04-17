"""Compute spectral coefficients for paper Appendix A."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src import theta_functions as TF
import numpy as np

tau = 0.5 + 0.3205128205j
omega = TF.find_critical_omega(tau)

th2_0 = float(np.real(TF.theta2(0, tau)))
th3_0 = float(np.real(TF.theta3(0, tau)))
th4_0 = float(np.real(TF.theta4(0, tau)))
th1p_0 = float(np.real(TF.theta1_prime_zero(tau)))

R = TF.R_omega(omega, tau)
U = TF.U_omega(omega, tau)
Up = TF.U_prime_omega(omega, tau)
U1p = TF.U1_prime_omega(omega, tau)
U2 = TF.U2_omega(omega, tau)

print(f"omega         = {float(np.real(omega)):.15f}")
print(f"R(omega)      = {float(np.real(R)):.15f}")
print(f"U(omega)      = {float(np.real(U)):.15f}")
print(f"U'(omega)     = {float(np.real(Up)):.15f}")
print(f"U1'(omega)    = {float(np.real(U1p)):.15f}")
print(f"U2(omega)     = {float(np.real(U2)):.15f}")
print(f"theta2(0)     = {th2_0:.15f}")
print(f"theta3(0)     = {th3_0:.15f}")
print(f"theta4(0)     = {th4_0:.15f}")
print(f"theta1'(0)    = {th1p_0:.15f}")
print(f"R*U check     = {float(np.real(R*U)):.15f} (should be -1)")
