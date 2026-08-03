from __future__ import annotations

import numpy as np


class ADMMSolver:

    def __init__(self, H: np.ndarray, u_max: float,
                 rho: float | None = None, n_iter: int = 50) -> None:
        n = H.shape[0]
        if rho is None:
            rho = float(np.max(np.linalg.eigvalsh(H)))
        self.rho = rho
        self.u_max = u_max
        self.n_iter = n_iter
        self.n = n

        H_rho = H + rho * np.eye(n)
        H_rho = 0.5 * (H_rho + H_rho.T)
        self.L = np.linalg.cholesky(H_rho)

    def _v_update(self, f: np.ndarray, z: np.ndarray, d: np.ndarray) -> np.ndarray:
        rhs = -f + self.rho * (z - d)
        y = np.linalg.solve(self.L, rhs)
        v = np.linalg.solve(self.L.T, y)
        return v

    def solve(self, f: np.ndarray,
              v_init: np.ndarray | None = None) -> np.ndarray:
        n = self.n
        v = np.zeros(n) if v_init is None else np.asarray(v_init, dtype=float).copy()
        z = np.clip(v, -self.u_max, self.u_max)
        d = np.zeros(n)

        for _ in range(self.n_iter):
            v = self._v_update(f, z, d)
            z = np.clip(v + d, -self.u_max, self.u_max)
            d = d + v - z

        return z


def riccati_recursion(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray,
                       R: np.ndarray, P_N: np.ndarray, N: int,
                       rho: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_u = Bd.shape[1]
    R_rho = R + rho * np.eye(n_u)

    Ps = [None] * (N + 1)
    Ks = [None] * N
    Ps[N] = P_N

    for i in range(N - 1, -1, -1):
        P_next = Ps[i + 1]
        S_i = R_rho + Bd.T @ P_next @ Bd
        K_i = np.linalg.solve(S_i, Bd.T @ P_next @ Ad)
        Ps[i] = Q + Ad.T @ P_next @ Ad - Ad.T @ P_next @ Bd @ K_i
        Ks[i] = K_i

    return Ks, Ps
