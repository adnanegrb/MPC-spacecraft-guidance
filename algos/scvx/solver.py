import numpy as np


def linearize_dynamics(f, x_ref, u_ref, eps=1e-6):
    """
    Finite-difference linearization of a nonlinear discrete dynamics
    function f(x, u) -> x_next, around a reference point (x_ref, u_ref).
    Returns matrices A, B and the affine residual c such that

        f(x, u) ~= A (x - x_ref) + B (u - u_ref) + f(x_ref, u_ref)
                 = A x + B u + c
    """
    nx = x_ref.shape[0]
    nu = u_ref.shape[0]
    f0 = f(x_ref, u_ref)

    A = np.zeros((nx, nx))
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        A[:, i] = (f(x_ref + dx, u_ref) - f0) / eps

    B = np.zeros((nx, nu))
    for j in range(nu):
        du = np.zeros(nu)
        du[j] = eps
        B[:, j] = (f(x_ref, u_ref + du) - f0) / eps

    c = f0 - A @ x_ref - B @ u_ref
    return A, B, c


def scvx_solve(f, Q, R, P, x0, N, nu, umax,
               x_traj_init, u_traj_init,
               trust_region=0.5, max_outer_iter=20, tol=1e-4,
               qp_solver=None):
    """
    Successive Convexification (Szmuk, Acikmese & Reynolds, 2020) for
    the nonlinear, discrete-time finite-horizon problem

        min sum_i x_i^T Q x_i + u_i^T R u_i + x_N^T P x_N
        s.t. x_{i+1} = f(x_i, u_i),  -umax <= u_i <= umax,  x_0 given

    At each outer iteration, f is linearized around the current
    trajectory, and the resulting time-varying linear-quadratic
    problem is solved by a simple projected-gradient QP routine
    (qp_solver), giving an exact convex subproblem with a trust
    region penalty that keeps consecutive iterates close, as
    required for convergence guarantees of SCvx.

    qp_solver must be a callable qp_solver(A_list, B_list, c_list, Q,
    R, P, umax, x0, N, nu) -> u_traj solving the LTV box-constrained
    LQ problem; if None, a simple internal solver is used.
    """
    nx = x0.shape[0]
    x_traj = x_traj_init.copy()
    u_traj = u_traj_init.copy()

    cost_hist = []

    for outer in range(max_outer_iter):
        A_list, B_list, c_list = [], [], []
        for i in range(N):
            A_i, B_i, c_i = linearize_dynamics(f, x_traj[i], u_traj[i])
            A_list.append(A_i)
            B_list.append(B_i)
            c_list.append(c_i)

        if qp_solver is None:
            u_new = _default_ltv_qp(A_list, B_list, c_list, Q, R, P,
                                     umax, x0, N, nu, u_traj, trust_region)
        else:
            u_new = qp_solver(A_list, B_list, c_list, Q, R, P, umax,
                               x0, N, nu, u_traj, trust_region)

        # propagate the TRUE nonlinear dynamics with the new controls
        x_new = np.zeros((N + 1, nx))
        x_new[0] = x0
        for i in range(N):
            x_new[i + 1] = f(x_new[i], u_new[i])

        cost = sum(x_new[i] @ Q @ x_new[i] + u_new[i] @ R @ u_new[i] for i in range(N))
        cost += x_new[N] @ P @ x_new[N]
        cost_hist.append(cost)

        delta = np.linalg.norm(u_new - u_traj)
        u_traj = u_new
        x_traj = x_new

        if delta < tol:
            break

    return x_traj, u_traj, cost_hist, outer + 1


def _default_ltv_qp(A_list, B_list, c_list, Q, R, P, umax, x0, N, nu,
                     u_ref, trust_region):
    """
    Simple projected-gradient solver for the LTV box-constrained LQ
    subproblem of SCvx, including a quadratic trust-region penalty
    around u_ref. No external optimization library is used: only
    matrix-vector products and a clip for the box and trust-region
    projection.
    """
    nx = A_list[0].shape[0]
    n = N * nu

    def rollout(u_flat):
        u = u_flat.reshape(N, nu)
        x = np.zeros((N + 1, nx))
        x[0] = x0
        for i in range(N):
            x[i + 1] = A_list[i] @ x[i] + B_list[i] @ u[i] + c_list[i]
        return x, u

    def cost_and_grad(u_flat):
        x, u = rollout(u_flat)
        cost = sum(x[i] @ Q @ x[i] + u[i] @ R @ u[i] for i in range(N))
        cost += x[N] @ P @ x[N]
        cost += trust_region * np.sum((u - u_ref)**2)

        # backward adjoint pass for the gradient (linear dynamics)
        lam = 2 * P @ x[N]
        grad = np.zeros((N, nu))
        for i in range(N - 1, -1, -1):
            grad[i] = 2 * R @ u[i] + B_list[i].T @ lam + 2*trust_region*(u[i]-u_ref[i])
            lam = 2 * Q @ x[i] + A_list[i].T @ lam
        return cost, grad.flatten()

    u_flat = u_ref.flatten().copy()
    L_est = 2 * (np.max(np.linalg.eigvalsh(R)) + trust_region) + 10.0
    step = 1.0 / L_est

    for _ in range(200):
        cost, grad = cost_and_grad(u_flat)
        u_flat = np.clip(u_flat - step * grad, -umax, umax)

    return u_flat.reshape(N, nu)
