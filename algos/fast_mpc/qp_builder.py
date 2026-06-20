import numpy as np

def build_condensed_qp(A, B, Q, P, R, N):
    """
    Condense the finite-horizon LQ problem into a dense QP in the
    stacked control vector v = [u_0; ...; u_{N-1}]:

        min 0.5 v^T H v + f^T v,  -umax <= v <= umax  (handled by caller)

    Used specifically by the fast gradient method, which needs the
    Hessian's largest and smallest eigenvalues (Lipschitz constant and
    strong convexity parameter) -- see Richter, Jones & Morari (2012).
    """
    nx, nu = B.shape
    Apow = [np.eye(nx)]
    for i in range(N):
        Apow.append(Apow[-1] @ A)
    A_bar = np.vstack(Apow[1:])

    B_bar = np.zeros((N * nx, N * nu))
    for i in range(N):
        for j in range(i + 1):
            B_bar[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Apow[i - j] @ B

    Q_bar = np.zeros((N * nx, N * nx))
    for i in range(N):
        Q_bar[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = P if i == N - 1 else Q

    R_bar = np.kron(np.eye(N), R)

    H = B_bar.T @ Q_bar @ B_bar + R_bar
    H = 0.5 * (H + H.T)
    return A_bar, B_bar, Q_bar, H


def qp_gradient(x0, A_bar, B_bar, Q_bar, H):
    f = B_bar.T @ Q_bar @ (A_bar @ x0)
    return f
