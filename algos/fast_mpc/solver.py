import numpy as np


def fast_gradient_qp(H, f, umax, max_iter=500, tol=1e-6, v_init=None):
    """
    Nesterov's fast gradient method (Algorithm 1 of Richter, Jones &
    Morari, 2012, "Computational Complexity Certification for Real-
    Time MPC With Input Constraints Based on the Fast Gradient
    Method") applied to:

        min_v  0.5 v^T H v + f^T v
        s.t.   -umax <= v_i <= umax   (box constraint, componentwise)

    The box projection is a simple clip, so each iteration costs only
    one matrix-vector product. The step size is fixed to 1/L, where L
    is the largest eigenvalue of H (the Lipschitz constant of the
    gradient), as in the paper.
    """
    L = np.linalg.eigvalsh(H)[-1]
    mu = np.linalg.eigvalsh(H)[0]
    step = 1.0 / L

    n = H.shape[0]
    v = np.zeros(n) if v_init is None else v_init.copy()
    y = v.copy()
    t = 1.0

    lb = -umax * np.ones(n)
    ub = umax * np.ones(n)

    for it in range(max_iter):
        grad = H @ y + f
        v_new = np.clip(y - step * grad, lb, ub)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = v_new + ((t - 1) / t_new) * (v_new - v)

        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            it += 1
            break
        v, t = v_new, t_new

    return v, it + 1
