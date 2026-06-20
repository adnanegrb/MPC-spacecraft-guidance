import numpy as np
import sys
sys.path.append('..')
from algos.fast_mpc.qp_builder import build_condensed_qp, qp_gradient
from algos.fast_mpc.solver import fast_gradient_qp

# Richter, Jones & Morari (2012), "Computational Complexity
# Certification for Real-Time MPC With Input Constraints Based on the
# Fast Gradient Method", Section V (illustrative example).
#
# Quote: "We consider the four-state/two-input system given in [2],
# restricting the initial state to ||x||_inf <= 10 and input to
# ||u||_inf <= 1. The weighting matrices are chosen as Q = I and
# R = 0.1 I."
nx, nu = 4, 2
N = 10
umax = 1.0

A = np.array([
    [0.7, 0.2, 0.0, 0.0],
    [0.0, 0.5, 0.1, 0.0],
    [0.0, 0.0, 0.6, 0.3],
    [0.1, 0.0, 0.0, 0.4]
])
B = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [0.0, 0.5]
])

Q = np.eye(nx)
R = 0.1 * np.eye(nu)
P = Q

x0 = np.array([7.0, -8.0, 5.0, -3.0])  # satisfies ||x0||_inf <= 10

A_bar, B_bar, Q_bar, H = build_condensed_qp(A, B, Q, P, R, N)
f = qp_gradient(x0, A_bar, B_bar, Q_bar, H)

eigs = np.linalg.eigvalsh(H)
L, mu = eigs[-1], eigs[0]
print(f"Hessian condition number L/mu = {L/mu:.2f}")

v, n_iter = fast_gradient_qp(H, f, umax, max_iter=500, tol=0.1)
print(f"Converged in {n_iter} iterations at tolerance eps=0.1")
print(f"(the paper reports a few tens of iterations for N=10 at this accuracy)")
print(f"u0 = {v[:nu]}")

print("\n--- Closed-loop simulation ---")
x = x0.copy()
x_hist = [x.copy()]
iters_hist = []
for k in range(30):
    f = qp_gradient(x, A_bar, B_bar, Q_bar, H)
    v, n_iter = fast_gradient_qp(H, f, umax, max_iter=500, tol=0.1)
    u0 = v[:nu]
    iters_hist.append(n_iter)
    x = A @ x + B @ u0
    x_hist.append(x.copy())

x_hist = np.array(x_hist)
print(f"Final state after 30 steps: {x_hist[-1]}")
print(f"Average fast-gradient iterations per step: {np.mean(iters_hist):.1f}")
