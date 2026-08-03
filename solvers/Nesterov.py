from __future__ import annotations

import numpy as np


class NesterovSolver:

    def __init__(self, H: np.ndarray, u_max: float,
                 n_iter: int = 100) -> None:
        self.H = H
        self.u_max = u_max
        self.n_iter = n_iter
        self.L = float(np.max(np.linalg.eigvalsh(H)))

    def solve(self, f: np.ndarray,
              v_init: np.ndarray | None = None) -> np.ndarray:
        n = self.H.shape[0]
        v_prev = np.zeros(n) if v_init is None else np.asarray(v_init, dtype=float).copy()
        v = v_prev.copy()

        for j in range(1, self.n_iter + 1):
            beta_j = (j - 1) / (j + 2)
            y = v + beta_j * (v - v_prev)
            grad = self.H @ y + f
            v_new = np.clip(y - grad / self.L, -self.u_max, self.u_max)
            v_prev = v
            v = v_new

        return v

    def condition_number(self) -> float:
        eigvals = np.linalg.eigvalsh(self.H)
        return float(eigvals[-1] / eigvals[0])
