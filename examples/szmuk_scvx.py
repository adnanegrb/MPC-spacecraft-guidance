import numpy as np
import sys
sys.path.append('..')
from algos.scvx.solver import scvx_solve

# Szmuk, Acikmese & Reynolds (2020), "Successive Convexification for
# Passively-Safe Spacecraft Rendezvous on Near Rectilinear Halo
# Orbit". Their full 6-DoF problem is reduced here to a minimal
# illustrative nonlinear example carrying the same structural idea:
# nonlinear dynamics, iterative linearization around the current
# trajectory, and a convex trust-region subproblem solved at each
# iteration, with no optimization library used anywhere.
#
# We use a simple pendulum-like nonlinear discrete dynamics as a
# compact, easily checked stand-in for the rotational/attitude
# coupling present in the original 6-DoF problem, since the full
# orbital + attitude SCvx setup from the paper requires substantially
# more bookkeeping (quaternions, thrust pointing cones) that is out
# of scope for this training example.
nx, nu, N = 2, 1, 15
dt = 0.1

def f(x, u):
    theta, omega = x
    domega = -np.sin(theta) + u[0]
    return np.array([theta + dt*omega, omega + dt*domega])

Q = np.diag([10.0, 1.0])
R = 0.1 * np.eye(nu)
P = Q
umax = 2.0

x0 = np.array([np.pi - 0.3, 0.0])  # start near the unstable equilibrium

x_init = np.tile(x0, (N + 1, 1))
u_init = np.zeros((N, nu))

x_traj, u_traj, cost_hist, n_iter = scvx_solve(
    f, Q, R, P, x0, N, nu, umax, x_init, u_init,
    trust_region=1.0, max_outer_iter=15
)

print(f"SCvx outer iterations: {n_iter}")
print("Cost per iteration:", [f"{c:.2f}" for c in cost_hist])
print(f"Final state: {x_traj[-1]}")
print(f"Final control sequence (first 5): {u_traj[:5].flatten()}")
