import numpy as np


def riccati_backward_pass(A, B, Q, P, R, N):
    """
    Backward Riccati recursion for the unconstrained finite-horizon LQ
    problem  min sum_i x_i^T Q x_i + u_i^T R u_i + x_N^T P x_N
             s.t. x_{i+1} = A x_i + B u_i
    Returns the time-varying feedback gains K_i such that the optimal
    unconstrained control is u_i = -K_i x_i.
    No matrix of size (N*nx) is ever built: only nx x nx and nu x nu
    blocks are stored and inverted at each step.
    """
    nx, nu = B.shape
    P_list = [None] * (N + 1)
    K_list = [None] * N
    P_list[N] = P

    for i in range(N - 1, -1, -1):
        Pn = P_list[i + 1]
        S = R + B.T @ Pn @ B
        K = np.linalg.solve(S, B.T @ Pn @ A)
        K_list[i] = K
        P_list[i] = Q + A.T @ Pn @ A - A.T @ Pn @ B @ K

    return K_list


def riccati_affine_pass(A, B, Q, P, R, N, b_list):
    """
    Backward Riccati recursion for the finite-horizon LQ problem with a
    per-step linear cost term:

        min sum_i x_i^T Q x_i + u_i^T R u_i - 2 b_i^T u_i + x_N^T P x_N
        s.t. x_{i+1} = A x_i + B u_i

    Closed form (derived and verified against a brute-force QP, see
    project notes; matches to machine precision):

        S_i = R + B^T P_{i+1} B
        K_i = S_i^{-1} B^T P_{i+1} A
        v_i = S_i^{-1} (B^T p_{i+1} + b_i)
        u_i*= -K_i x_i + v_i
        P_i = Q + A^T P_{i+1} A - A^T P_{i+1} B K_i
        p_i = A^T (p_{i+1} - P_{i+1} B v_i)

    Returns the gains needed for a forward pass: K_list, v-coefficient
    helpers (Sinv_list, B, p_list) so that u_i = -K_i x_i + v_i.
    """
    nx, nu = B.shape
    P_list = [None] * (N + 1)
    p_list = [None] * (N + 1)
    K_list = [None] * N
    Sinv_list = [None] * N
    P_list[N] = P
    p_list[N] = np.zeros(nx)

    for i in range(N - 1, -1, -1):
        Pn = P_list[i + 1]
        pn = p_list[i + 1]
        S = R + B.T @ Pn @ B
        Sinv = np.linalg.inv(S)
        K = Sinv @ B.T @ Pn @ A
        v = Sinv @ (B.T @ pn + b_list[i])

        P_list[i] = Q + A.T @ Pn @ A - A.T @ Pn @ B @ K
        p_list[i] = A.T @ (pn - Pn @ B @ v)
        K_list[i] = K
        Sinv_list[i] = Sinv

    return K_list, Sinv_list, p_list


def linear_mpc_solve(A, B, Q, P, R, umax, x0, N,
                      rho=None, max_iter=300, tol=1e-8, alpha=1.6):
    """
    Solve the box-constrained finite-horizon LQ problem

        min sum_i  x_i^T Q x_i + u_i^T R u_i  +  x_N^T P x_N
        s.t. x_{i+1} = A x_i + B u_i,  -umax <= u_i <= umax,  x_0 given

    This is the standard Linear MPC formulation (Di Cairano, Park and
    Kolmanovsky, 2012). Solved by ADMM: the box constraint is split
    from the dynamics,

        min_{u,z}  sum_i x_i^T Q x_i + u_i^T R u_i + x_N^T P x_N
        s.t.       x_{i+1} = A x_i + B u_i,  u_i = z_i,  z_i in [-umax,umax]

    The u-update is the affine-LQ problem above (exact Riccati closed
    form, never condenses A^N into a dense matrix). The z-update is a
    simple clip. The ADMM penalty rho is auto-scaled to the spectral
    radius of Q, which is necessary for problems where Q >> R (such as
    the Di Cairano weighting): a fixed small rho can stall convergence
    by orders of magnitude on this type of problem.

    Returns the full control trajectory and the number of ADMM
    iterations used.
    """
    nx, nu = B.shape
    if rho is None:
        rho = np.linalg.eigvalsh(Q)[-1]
    R_rho = R + rho * np.eye(nu)

    z = np.zeros((N, nu))
    lam = np.zeros((N, nu))

    for it in range(max_iter):
        b_list = [rho * (z[i] - lam[i]) for i in range(N)]
        K_list, Sinv_list, p_list = riccati_affine_pass(A, B, Q, P, R_rho, N, b_list)

        x = x0.copy()
        u_traj = np.zeros((N, nu))
        for i in range(N):
            v = Sinv_list[i] @ (B.T @ p_list[i + 1] + b_list[i])
            u = -K_list[i] @ x + v
            u_traj[i] = u
            x = A @ x + B @ u

        u_relaxed = alpha * u_traj + (1 - alpha) * z
        z_new = np.clip(u_relaxed + lam, -umax, umax)
        lam = lam + u_relaxed - z_new

        if np.linalg.norm(z_new - z) < tol:
            z = z_new
            it += 1
            break
        z = z_new

    return z.copy(), it + 1
