import numpy as np
import pytest

from solvers.admm import ADMMSolver
from solvers.nesterov import NesterovSolver
from tests.conftest import condensed_qp


def _random_spd(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_admm_nesterov_agree_on_random_qp(seed):
    n = 8
    H = _random_spd(n, seed)
    rng = np.random.default_rng(seed + 100)
    f = rng.standard_normal(n)
    u_max = 0.3

    v_admm = ADMMSolver(H, u_max, n_iter=300).solve(f)
    v_nest = NesterovSolver(H, u_max, n_iter=800).solve(f)

    assert np.linalg.norm(v_admm - v_nest) < 1e-4


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_admm_respects_box_constraint(seed):
    n = 8
    H = _random_spd(n, seed)
    rng = np.random.default_rng(seed + 200)
    f = 10 * rng.standard_normal(n)
    u_max = 0.2

    v = ADMMSolver(H, u_max, n_iter=200).solve(f)
    assert np.all(np.abs(v) <= u_max + 1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_nesterov_respects_box_constraint(seed):
    n = 8
    H = _random_spd(n, seed)
    rng = np.random.default_rng(seed + 200)
    f = 10 * rng.standard_normal(n)
    u_max = 0.2

    v = NesterovSolver(H, u_max, n_iter=500).solve(f)
    assert np.all(np.abs(v) <= u_max + 1e-9)


def test_admm_matches_unconstrained_optimum_when_box_inactive():
    n = 6
    H = _random_spd(n, 42)
    rng = np.random.default_rng(43)
    f = 0.01 * rng.standard_normal(n)
    u_max = 1000.0

    v = ADMMSolver(H, u_max, n_iter=200).solve(f)
    v_star = -np.linalg.solve(H, f)

    assert np.linalg.norm(v - v_star) < 1e-3


def test_cwh_condensed_qp_admm_nesterov_agree(cwh_setup):
    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    Q, R = cwh_setup["Q"], cwh_setup["R"]
    N = 5
    _, _, _, _, H = condensed_qp(Ad, Bd, Q, R, Q, N)

    rng = np.random.default_rng(7)
    f = rng.standard_normal(H.shape[0]) * 1e3
    u_max = 0.2

    v_admm = ADMMSolver(H, u_max, n_iter=300).solve(f)
    v_nest = NesterovSolver(H, u_max, n_iter=2000).solve(f)

    assert np.linalg.norm(v_admm - v_nest) < 1e-2
