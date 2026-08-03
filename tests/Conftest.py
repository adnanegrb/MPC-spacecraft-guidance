import numpy as np
from scipy.linalg import expm, solve_discrete_are
import pytest


MU_EARTH = 3.986e14
ALTITUDE = 500e3
R_EARTH = 6371e3


def cwh_matrices(Ts: float):
    n = np.sqrt(MU_EARTH / (R_EARTH + ALTITUDE) ** 3)
    Ac = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [3 * n ** 2, 0, 0, 2 * n],
        [0, 0, -2 * n, 0],
    ])
    Bc = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ])
    n_x, n_u = 4, 2
    M = np.zeros((n_x + n_u, n_x + n_u))
    M[:n_x, :n_x] = Ac
    M[:n_x, n_x:] = Bc
    Md = expm(M * Ts)
    Ad = Md[:n_x, :n_x]
    Bd = Md[:n_x, n_x:]
    return Ad, Bd, n


def condensed_qp(Ad, Bd, Q, R, P, N):
    n_x, n_u = Bd.shape
    A_list = [np.linalg.matrix_power(Ad, i) for i in range(1, N + 1)]
    A_cond = np.vstack(A_list)

    B_cond = np.zeros((n_x * N, n_u * N))
    for i in range(N):
        for j in range(i + 1):
            B_cond[n_x * i:n_x * (i + 1), n_u * j:n_u * (j + 1)] = (
                np.linalg.matrix_power(Ad, i - j) @ Bd
            )

    Qbar = np.kron(np.eye(N), Q)
    Qbar[-n_x:, -n_x:] = P
    Rbar = np.kron(np.eye(N), R)

    H = B_cond.T @ Qbar @ B_cond + Rbar
    H = 0.5 * (H + H.T)
    return A_cond, B_cond, Qbar, Rbar, H


@pytest.fixture
def cwh_setup():
    Ts = 100.0
    Ad, Bd, n = cwh_matrices(Ts)
    Q = 3e3 * np.diag([100, 100, 1, 1])
    R = 100 * np.eye(2)
    return {"Ad": Ad, "Bd": Bd, "n": n, "Ts": Ts, "Q": Q, "R": R}


@pytest.fixture
def lqr_gain(cwh_setup):
    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    Q, R = cwh_setup["Q"], cwh_setup["R"]
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P
