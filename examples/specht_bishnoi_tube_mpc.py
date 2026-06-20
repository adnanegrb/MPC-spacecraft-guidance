import numpy as np
import sys
sys.path.append('..')
from algos.tube_mpc.solver import tube_mpc_step
from utils.dynamics import get_CW_matrices

# Specht, Bishnoi & Lampariello (2023), "Autonomous Spacecraft
# Rendezvous Using Tube-Based Model Predictive Control: Design and
# Application". Same radial-approach scenario as Di Cairano, but with
# bounded random disturbances and the Tube MPC robust tracking layer
# (nominal trajectory + pre-computed LQR feedback on the tracking
# error), built on the validated algos/tube_mpc solver.
n     = 1.107e-3
Ts    = 0.5
N     = 40
umax  = 0.2
w_max = 0.005   # bounded disturbance amplitude [m/s^2]

Q = 3e3 * np.block([[1e2*np.eye(2), np.zeros((2,2))],
                    [np.zeros((2,2)), np.eye(2)]])
P = Q
R = 1e2 * np.eye(2)

A, B = get_CW_matrices(n, Ts)
nx, nu = 4, 2

x0 = np.array([100., -10., 0., 0.])

np.random.seed(0)
x_real, z_nom = x0.copy(), x0.copy()
x_hist = [x_real.copy()]
u_hist = []
iters_hist = []

for k in range(300):
    u, v0, n_iter = tube_mpc_step(A, B, Q, P, R, umax, x_real, z_nom, N,
                                   w_max, max_iter=100, tol=1e-7)
    u_hist.append(u)
    iters_hist.append(n_iter)

    w = np.random.uniform(-w_max, w_max, nx)
    x_real = A @ x_real + B @ u + w
    z_nom = A @ z_nom + B @ v0
    x_hist.append(x_real.copy())

    if np.linalg.norm(x_real[:2]) < 2.6:
        break

x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

print(f"Done in {len(u_hist)} steps ({len(u_hist)*Ts:.1f} s)")
print(f"Final position : ({x_hist[-1,0]:.2f}, {x_hist[-1,1]:.2f}) m")
print(f"Final velocity : ({x_hist[-1,2]:.4f}, {x_hist[-1,3]:.4f}) m/s")
print(f"Avg ADMM iterations per step (nominal QP) : {np.mean(iters_hist):.1f}")
