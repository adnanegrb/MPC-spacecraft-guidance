import numpy as np
import pytest

from tests.conftest import cwh_matrices


def test_closed_loop_matrix_is_stable(lqr_gain, cwh_setup):
    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    K, _ = lqr_gain

    Acl = Ad - Bd @ K
    eigvals = np.linalg.eigvals(Acl)

    assert np.all(np.abs(eigvals) < 1.0)


def test_dare_solution_is_positive_semidefinite(lqr_gain):
    _, P = lqr_gain
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals >= -1e-8)


def test_lqr_gain_sign_convention_matches_report(lqr_gain, cwh_setup):
    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    K, _ = lqr_gain

    Acl_correct = Ad - Bd @ K
    Acl_wrong = Ad + Bd @ K

    rho_correct = np.max(np.abs(np.linalg.eigvals(Acl_correct)))
    rho_wrong = np.max(np.abs(np.linalg.eigvals(Acl_wrong)))

    assert rho_correct < 1.0
    assert rho_wrong > rho_correct


@pytest.mark.parametrize("seed", range(20))
def test_tube_mpc_error_stays_bounded_under_disturbance(lqr_gain, cwh_setup, seed):
    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    K, _ = lqr_gain
    Acl = Ad - Bd @ K

    rng = np.random.default_rng(seed)
    w_bar = np.array([5.0, 5.0, 1e-2, 1e-2])

    e = np.zeros(4)
    max_norm = 0.0
    for _ in range(200):
        xi = rng.uniform(-w_bar, w_bar)
        e = Acl @ e + xi
        max_norm = max(max_norm, np.linalg.norm(e))

    assert np.isfinite(max_norm)
    assert max_norm < 1e4


def test_condition_number_grows_with_horizon(cwh_setup):
    from tests.conftest import condensed_qp

    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    Q, R = cwh_setup["Q"], cwh_setup["R"]

    kappas = []
    for N in [5, 20, 40]:
        _, _, _, _, H = condensed_qp(Ad, Bd, Q, R, Q, N)
        eigvals = np.linalg.eigvalsh(H)
        kappas.append(eigvals[-1] / eigvals[0])

    assert kappas[0] < kappas[1] < kappas[2]


def test_condition_number_matches_report_order_of_magnitude(cwh_setup):
    from tests.conftest import condensed_qp

    Ad, Bd = cwh_setup["Ad"], cwh_setup["Bd"]
    Q, R = cwh_setup["Q"], cwh_setup["R"]

    _, _, _, _, H = condensed_qp(Ad, Bd, Q, R, Q, N=5)
    eigvals = np.linalg.eigvalsh(H)
    kappa = eigvals[-1] / eigvals[0]

    assert 1e3 < kappa < 1e5
