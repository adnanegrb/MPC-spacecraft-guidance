import numpy as np
from scipy.linalg import solve_discrete_are
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'linear_mpc'))
from solver import linear_mpc_solve


def compute_lqr_gain(A, B, Q, R):
    """
    Pre-computed stabilizing LQR gain K used as the inner tube
    feedback, obtained from the discrete algebraic Riccati equation.
    """
    P_lqr = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P_lqr @ B) @ (B.T @ P_lqr @ A)
    return K, P_lqr


def tube_mpc_solve(A, B, Q, P, R, umax, x0, N, w_max,
                    tube_margin_factor=3.0, max_iter=300, tol=1e-8):
    """
    Solve one step of Tube MPC (Mayne, Seron & Rakovic, 2005; applied
    to spacecraft rendezvous by Specht, Bishnoi & Lampariello, 2023,
    and Oestreich, Linares & Gondhalekar, 2023).

    The real dynamics are subject to bounded additive disturbances:

        x_{k+1} = A x_k + B u_k + w_k,  w_k in W = [-w_max, w_max]^nx

    The control is split as u_k = v_k + K (x_k - z_k), where z_k is a
    nominal trajectory and K is a pre-computed LQR gain. The nominal
    problem is the SAME box-constrained QP as Linear MPC, but solved
    on tightened input bounds (umax - tube_margin) to leave room for
    the disturbance rejection term K(x_k - z_k). The nominal QP is
    solved with the validated ADMM + Riccati solver from algos/linear_mpc,
    so no new optimization machinery is introduced for the nominal part.

    Returns: applied control u0, the nominal state update z1, and the
    number of ADMM iterations used by the nominal solve.
    """
    nx, nu = B.shape
    K, _ = compute_lqr_gain(A, B, Q, R)

    tube_margin = w_max * tube_margin_factor
    u_tight = max(umax - tube_margin, 0.0)

    v_traj, n_iter = linear_mpc_solve(A, B, Q, P, R, u_tight, x0, N,
                                       max_iter=max_iter, tol=tol)
    v0 = v_traj[0]

    # In closed-loop use, x_curr and z_curr start equal; here we expose
    # the generic step assuming the caller tracks both states.
    return v0, K, n_iter


def tube_mpc_step(A, B, Q, P, R, umax, x_real, z_nom, N, w_max,
                   tube_margin_factor=3.0, max_iter=300, tol=1e-8):
    """
    One closed-loop Tube MPC step given the current real state x_real
    and nominal state z_nom (both maintained by the caller across
    steps). Returns the saturated control actually applied to the
    real system, and the nominal control used to propagate z_nom.
    """
    nx, nu = B.shape
    K, _ = compute_lqr_gain(A, B, Q, R)

    tube_margin = w_max * tube_margin_factor
    u_tight = max(umax - tube_margin, 0.0)

    v_traj, n_iter = linear_mpc_solve(A, B, Q, P, R, u_tight, z_nom, N,
                                       max_iter=max_iter, tol=tol)
    v0 = v_traj[0]

    e = x_real - z_nom
    u_applied = np.clip(v0 - K @ e, -umax, umax)

    return u_applied, v0, n_iter
