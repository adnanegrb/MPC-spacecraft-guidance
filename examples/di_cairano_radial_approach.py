import numpy as np
import sys
sys.path.append('..')
from algos.linear_mpc.solver import linear_mpc_solve
from utils.dynamics import get_CW_matrices

# Di Cairano, Park & Kolmanovsky (2012), Section 5.1, radial approach.
# Reproduces their closed-loop rendezvous scenario using the validated
# ADMM + Riccati Linear MPC solver from algos/linear_mpc (no
# optimization library).
n    = 1.107e-3   # orbital rate [rad/s], 500 km altitude
Ts   = 0.5        # sampling period [s]
N    = 40         # prediction horizon
umax = 0.2        # thrust bound [m/s^2]

Q = 3e3 * np.block([[1e2*np.eye(2), np.zeros((2,2))],
                    [np.zeros((2,2)), np.eye(2)]])
P = Q
R = 1e2 * np.eye(2)

A, B = get_CW_matrices(n, Ts)
nx, nu = 4, 2

x0 = np.array([100., -10., 0., 0.])  # radial approach initial condition

x_hist = [x0.copy()]
u_hist = []
iters_hist = []
x = x0.copy()

for k in range(300):
    u_traj, n_iter = linear_mpc_solve(A, B, Q, P, R, umax, x, N,
                                       max_iter=100, tol=1e-7)
    u_k = u_traj[0]
    u_hist.append(u_k)
    iters_hist.append(n_iter)
    x = A @ x + B @ u_k
    x_hist.append(x.copy())
    if np.linalg.norm(x[:2]) < 2.6:
        break

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

print(f"Done in {len(u_hist)} steps ({len(u_hist)*Ts:.1f} s)")
print(f"Final position : ({x_hist[-1,0]:.2f}, {x_hist[-1,1]:.2f}) m")
print(f"Final velocity : ({x_hist[-1,2]:.4f}, {x_hist[-1,3]:.4f}) m/s")
print(f"Avg ADMM iterations per step : {np.mean(iters_hist):.1f}")
