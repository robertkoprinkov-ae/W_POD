"""End-to-end integration tests (math tier) for Formulation 3.5 (UNREGULARIZED)
only: run the full L2+W2 optimization on a small synthetic problem, with
theta ~ 0 / lambda_reg = 0 (see integration_config_unregularized.py), and check
the result against the acceptance criteria agreed on:

  1. relative reconstruction error < a tolerance
     (integration_config_unregularized.py:IntegrationCase.max_recon_error)
  2. recovered transport (gamma) trajectories match the ground truth within a
     tolerance (gamma_tol); skipped if not set
  3. beats plain (uncentered, truncated-SVD) POD at the same number of modes

This does NOT exercise the RKHS-regularized Formulation 3.8 -- see the module
docstring of integration_config_unregularized.py. A separate
test_integration_regularized.py is planned for that, with its own
interpolation-based acceptance criteria.

Loss need not be monotonically non-increasing across iterations -- not checked.

This module is written entirely against the IntegrationCase contract in
integration_config_unregularized.py. If make_integration_case() ever raises
NotImplementedError again, this whole module SKIPS rather than erroring.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from integration_config_unregularized import make_integration_case


@pytest.fixture(scope="module")
def solved_case():
    try:
        case = make_integration_case()
    except NotImplementedError as e:
        pytest.skip(f"integration config not provided yet: {e}")
    case.solver.solve_scipy(case.initial)
    return case


# ---------------------------------------------------------------------------
# shared metrics
# ---------------------------------------------------------------------------

def _reconstruct_all(ansatz, params, data):
    """Stack ansatz.get_values_grid(z_j, params) over all snapshots -> (n_z, Npts),
    matching the shape/order of data.data_exact."""
    rows = [ansatz.get_values_grid(z, params) for z in data.data_z]
    return jnp.stack(rows)


def _relative_frobenius_error(U_hat, X):
    U_hat, X = np.asarray(U_hat), np.asarray(X)
    return float(np.linalg.norm(U_hat - X) / np.linalg.norm(X))


def _plain_pod_reconstruction_error(X, n_modes):
    """Relative Frobenius error of the best rank-n_modes reconstruction of X
    (uncentered truncated SVD -- matches the paper's Eq 1.1 linear POD, no mean
    subtraction)."""
    X = np.asarray(X)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n_modes = min(n_modes, S.shape[0])
    X_hat = (U[:, :n_modes] * S[:n_modes]) @ Vt[:n_modes, :]
    return float(np.linalg.norm(X_hat - X) / np.linalg.norm(X))


# ---------------------------------------------------------------------------
# 1. reconstruction error
# ---------------------------------------------------------------------------

def test_reconstruction_error_below_tolerance(solved_case):
    case = solved_case
    params = case.solver.params_result

    U_hat = _reconstruct_all(case.ansatz, params, case.data)
    err = _relative_frobenius_error(U_hat, case.data.data_exact)

    assert err < case.max_recon_error, (
        f"relative reconstruction error {err:.4f} >= tolerance {case.max_recon_error}"
    )


# ---------------------------------------------------------------------------
# 2. recovered transport (gamma) trajectories vs. ground truth
# ---------------------------------------------------------------------------

def test_gamma_recovery(solved_case):
    case = solved_case
    if case.gamma_tol is None:
        pytest.skip("gamma_tol not set on the integration case")

    params_per_mode = case.ansatz.arr_to_dict(case.solver.params_result)

    checked_any = False
    for mode_i, mode in enumerate(case.ansatz.modes):
        for coeff_name in mode.get_coeff_names():
            if "gamma" not in coeff_name:
                continue
            value_name = coeff_name[: coeff_name.rfind("_")] + "_value"
            true_fn = case.data.ansatz_values[mode_i].get(value_name)
            if true_fn is None:
                continue  # this mode's ground truth doesn't define this trajectory

            true_vals = np.asarray(true_fn(case.data.data_z))
            recovered = np.asarray(
                mode.calc_coeff(coeff_name, case.data.data_z,
                                params_per_mode[mode_i][coeff_name])
            )
            np.testing.assert_allclose(
                recovered, true_vals, atol=case.gamma_tol,
                err_msg=f"mode {mode_i} coeff {coeff_name!r}",
            )
            checked_any = True

    assert checked_any, "no gamma-like coefficient found on the ansatz to check"


# ---------------------------------------------------------------------------
# 3. beats plain POD
# ---------------------------------------------------------------------------

def test_beats_plain_pod(solved_case):
    case = solved_case
    params = case.solver.params_result

    U_hat = _reconstruct_all(case.ansatz, params, case.data)
    ours = _relative_frobenius_error(U_hat, case.data.data_exact)

    n_modes = case.n_pod_modes_for_baseline or len(case.ansatz.modes)
    pod_err = _plain_pod_reconstruction_error(case.data.data_exact, n_modes)

    assert ours < pod_err, (
        f"our error {ours:.4f} did not beat plain POD "
        f"({n_modes} modes) error {pod_err:.4f}"
    )
