"""Math-tier accuracy tests for RBF_interpol (group M of the "Math tier" list in
unit_tests.md).

tests/test_util.py already covers interpolation accuracy and GP-variance
behaviour for the squared-exponential kernel without a bias term, in 1D. This
module extends that in three directions, all calibrated against measured
numbers (not guessed-then-loosened tolerances):

  1. bias=True and the matern32/matern52 kernels, still in 1D (mirrors
     test_RBF_predict / test_RBF_predict_vars).
  2. dim_z=2 interpolation accuracy -- untested anywhere before this (the
     plumbing tier's test_G41_dim_z_inference only checks shapes). This is a
     real code path: the paper's 2D wildland fire case has z = [Arrhenius
     coefficient, time].
  3. predict()'s gradient w.r.t. coeff against central finite differences, in
     1D, 2D, and through a vmap'd batch-z call (the composition
     calc_l2_training/calc_w1_training actually use). This is the gradient the
     whole optimizer depends on being correct.

Precision policy (revised -- an earlier version of this file gated the WHOLE
module behind JAX_ENABLE_X64=1, which meant it got ZERO coverage on a default
`pytest` run, accuracy checks included, even though those don't need fp64 at
all):

  - Accuracy / variance tests (1D and 2D) run always, at whatever precision is
    active. Confirmed these numbers don't meaningfully change with precision
    (fp64 calibration matched the fp32 numbers to 4-5 significant figures) --
    not assumed.
  - The independent analytic cross-check
    (test_M_predict_gradient_matches_independent_analytic_reimplementation)
    also runs always: it compares two autodiff traces of algebraically
    equivalent formulas, not finite differences, so there is no FD noise floor
    to worry about -- confirmed exact (0.0) agreement at both fp32 and fp64.
  - The finite-difference gradient checks are the one place precision matters:
    at fp32, central-difference noise floors around 1e-5 to 1e-4 relative,
    which could mask a real error of similar size. Each of these therefore
    exists TWICE:
      * a "_smoke" version with a loose, fp32-safe tolerance that always runs
        (so the default test suite has *some* real gradient coverage rather
        than none -- verified this has teeth: it previously caught a
        deliberately-injected 1.5x gradient-scaling bug, atol=1e-4).
      * the original, precision-labelled version, gated behind
        `@pytest.mark.skipif(not jax.config.jax_enable_x64, ...)`, with a
        much tighter tolerance (atol=1e-9) -- calibrated at fp64, where
        agreement measured 1e-13 to 1e-16 (machine precision) everywhere.
        Run explicitly with:

            JAX_ENABLE_X64=1 python -m pytest tests/test_rbf_accuracy.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from W_POD.util import RBF_interpol


_REQUIRE_X64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="precision-labelled gradient check -- run with JAX_ENABLE_X64=1 for "
           "the tight (atol=1e-9) tolerance; see the _smoke sibling test for "
           "the always-on, fp32-safe version of the same check",
)


_DATA_Z = jnp.array([0.1, 0.25, 0.7, 1.0, 1.5, 2.8])
_THETA = 1.0
_SIGMA = 1.0


def _true_fn(z):
    return 1.0 + 2.0 * jnp.cos(z)


_Z_TEST = jnp.linspace(float(_DATA_Z.min()), float(_DATA_Z.max()), 100)

# (kernel, bias, between-points abs tolerance) -- measured max error / headroom:
#   SE      bias=True : measured 0.083 -> 0.12
#   matern32 bias=False: measured 0.026 -> 0.05
#   matern52 bias=False: measured 0.026 -> 0.05
_CASES = [
    ("SE", True, 0.12),
    ("matern32", False, 0.05),
    ("matern52", False, 0.05),
]
_IDS = [f"{k}{'-bias' if b else ''}" for k, b, _ in _CASES]


@pytest.mark.parametrize("kernel,bias,between_tol", _CASES, ids=_IDS)
def test_M_predict_accuracy(kernel, bias, between_tol):
    """Extends test_util.py::test_RBF_predict beyond SE/no-bias: fit
    1 + 2*cos(z), check the interpolant is close to the true function over the
    whole training range."""
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, bias=bias, kernel=kernel)
    coeff = rbf.get_coeff(_true_fn(_DATA_Z))

    y_pred = np.asarray(rbf.predict(_Z_TEST, coeff))
    y_true = np.asarray(_true_fn(_Z_TEST))
    np.testing.assert_allclose(y_pred, y_true, atol=between_tol)


@pytest.mark.parametrize("kernel,bias,between_tol", _CASES, ids=_IDS)
def test_M_predict_variance_behaviour(kernel, bias, between_tol):
    """Extends test_util.py::test_RBF_predict_vars beyond SE/no-bias:
    - prediction is close to the true function (small nugget -> tighter fit
      than test_M_predict_accuracy, so the same tolerance is safe)
    - variance is non-negative everywhere
    - variance is ~0 at training points
    - the +/- 2 sigma band brackets the true function everywhere on the test
      grid (calibration measured 100% coverage for all three cases)."""
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, bias=bias, kernel=kernel, nugget=1e-6)
    coeff = rbf.get_coeff(_true_fn(_DATA_Z))

    y_pred, var_pred = rbf.predict(_Z_TEST, coeff, variances=True)
    y_pred, var_pred = np.asarray(y_pred), np.asarray(var_pred)
    y_true = np.asarray(_true_fn(_Z_TEST))

    np.testing.assert_allclose(y_pred, y_true, atol=between_tol)
    assert np.all(var_pred >= 0.0)

    lower = y_pred - 2.0 * np.sqrt(var_pred)
    upper = y_pred + 2.0 * np.sqrt(var_pred)
    assert np.all(y_true >= lower) and np.all(y_true <= upper)

    _, var_train = rbf.predict(_DATA_Z, coeff, variances=True)
    np.testing.assert_allclose(np.asarray(var_train), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# dim_z = 2 interpolation accuracy
# ---------------------------------------------------------------------------

# 9 scattered points in [0,1]^2 (not a grid, so support isn't accidentally
# separable/axis-aligned); theta chosen by the same measure-first approach as
# the 1D cases (0.3 and 0.5 were both meaningfully worse -- 0.24/0.35 and
# 0.06/0.17 max error respectively -- across all three kernels).
_DATA_Z_2D = jnp.array([
    [0.10, 0.20], [0.30, 0.80], [0.60, 0.10], [0.80, 0.60], [0.45, 0.45],
    [0.05, 0.90], [0.90, 0.90], [0.50, 0.05], [0.95, 0.30],
])
_THETA_2D = 0.8


def _true_fn_2d(z):
    z1, z2 = z[..., 0], z[..., 1]
    return jnp.sin(z1 + 0.5 * z2) + 0.3 * jnp.cos(2.0 * z2)


_g = np.linspace(0.05, 0.9, 12)
_Z1, _Z2 = np.meshgrid(_g, _g)
_Z_TEST_2D = jnp.array(np.stack([_Z1.ravel(), _Z2.ravel()], axis=-1))


@pytest.mark.parametrize("bias,between_tol", [(False, 0.05), (True, 0.07)],
                         ids=["nobias", "bias"])
def test_M_predict_accuracy_2d(bias, between_tol):
    """dim_z=2 accuracy, SE kernel (the best-conditioned of the three at this
    point density -- matern32/52 measured 0.08-0.17 max error at the same
    theta, notably worse; not investigated further, flagging rather than
    hiding it). Measured max error: 0.023 (no-bias) / 0.042 (bias); tolerances
    keep ~1.7x headroom."""
    rbf = RBF_interpol(_THETA_2D, _SIGMA, _DATA_Z_2D, bias=bias, kernel="SE")
    coeff = rbf.get_coeff(_true_fn_2d(_DATA_Z_2D))

    y_pred = np.asarray(rbf.predict(_Z_TEST_2D, coeff))
    y_true = np.asarray(_true_fn_2d(_Z_TEST_2D))
    np.testing.assert_allclose(y_pred, y_true, atol=between_tol)

    # single-point (non-batched) predict returns a scalar, matches the batch
    single = rbf.predict(_DATA_Z_2D[3], coeff)
    assert jnp.asarray(single).shape == ()
    np.testing.assert_allclose(float(single), float(_true_fn_2d(_DATA_Z_2D)[3]),
                               atol=0.05)

    _, var = rbf.predict(_Z_TEST_2D, coeff, variances=True)
    assert np.all(np.asarray(var) >= 0.0)


# ---------------------------------------------------------------------------
# predict() gradient w.r.t. coeff vs. central finite differences
# ---------------------------------------------------------------------------

# fp64 ("precise"): h swept from 1e-1 to 3e-4 at fp64; error grew monotonically
# as h shrank (roundoff ~eps/h dominates before truncation error becomes
# visible for this smooth a function), bottoming out near machine precision at
# h=1e-1. h=1e-3 is a deliberately conservative middle ground, not the
# empirical optimum, and still measured a worst case of 4.6e-13 across every
# case below (1D x 3 kernels x bias, 2D, batched-z) -- atol/rtol keep >100x
# headroom over that.
_FD_H_FP64 = 1e-3
_FD_ATOL_FP64 = 1e-9
_FD_RTOL_FP64 = 1e-9

# fp32 ("smoke"): h=1e-2 measured to beat h=1e-3 in every case (float32
# central-difference roundoff dominates below the theoretical optimum
# h ~ eps^(1/3) ~ 5e-3 for eps_float32 ~ 1.2e-7) -- the textbook fp32 FD
# tradeoff, not a fluke. Worst measured relative error was 1.25e-4 (SE+bias);
# atol/rtol keep some headroom over that while staying loose enough not to be
# fp32-noise-flaky. Verified this still has real detection power: a
# deliberately wrong gradient (correct value, tangent scaled 1.5x via
# jax.custom_jvp) was caught, max|diff|=0.49.
_FD_H_SMOKE = 1e-2
_FD_ATOL_SMOKE = 1e-4
_FD_RTOL_SMOKE = 1e-4


def _central_fd_grad(f, coeff, h):
    """Elementwise central-difference gradient of a scalar-valued f(coeff)."""
    c = np.asarray(coeff, dtype=np.float64)
    grad = np.zeros_like(c)
    for i in range(c.shape[0]):
        plus, minus = c.copy(), c.copy()
        plus[i] += h
        minus[i] -= h
        f_plus = float(f(jnp.asarray(plus, dtype=coeff.dtype)))
        f_minus = float(f(jnp.asarray(minus, dtype=coeff.dtype)))
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    return grad


@pytest.mark.parametrize("kernel,bias", [
    ("SE", False), ("SE", True),
    ("matern32", False), ("matern32", True),
    ("matern52", False), ("matern52", True),
])
def test_M_predict_gradient_matches_finite_differences_1d_smoke(kernel, bias):
    """Always-on, fp32-safe version -- see module docstring. Runs regardless
    of JAX_ENABLE_X64 so the default test suite has real (if coarser)
    gradient coverage."""
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, bias=bias, kernel=kernel)
    rng = np.random.default_rng(hash((kernel, bias)) % (2**32))
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))

    for z0 in (0.4, 1.2, 2.0):
        f = lambda c, z0=z0: rbf.predict(z0, c)
        g_auto = np.asarray(jax.grad(f)(coeff))
        g_fd = _central_fd_grad(f, coeff, _FD_H_SMOKE)
        np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("kernel,bias", [
    ("SE", False), ("SE", True),
    ("matern32", False), ("matern32", True),
    ("matern52", False), ("matern52", True),
])
def test_M_predict_gradient_matches_finite_differences_1d(kernel, bias):
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, bias=bias, kernel=kernel)
    rng = np.random.default_rng(hash((kernel, bias)) % (2**32))
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))

    for z0 in (0.4, 1.2, 2.0):
        f = lambda c, z0=z0: rbf.predict(z0, c)
        g_auto = np.asarray(jax.grad(f)(coeff))
        g_fd = _central_fd_grad(f, coeff, _FD_H_FP64)
        np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


@pytest.mark.parametrize("bias", [False, True])
def test_M_predict_gradient_matches_finite_differences_2d_smoke(bias):
    rbf = RBF_interpol(_THETA_2D, _SIGMA, _DATA_Z_2D, bias=bias, kernel="SE")
    rng = np.random.default_rng(2)
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))

    z0 = jnp.array([0.55, 0.35])
    f = lambda c: rbf.predict(z0, c)
    g_auto = np.asarray(jax.grad(f)(coeff))
    g_fd = _central_fd_grad(f, coeff, _FD_H_SMOKE)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
@pytest.mark.parametrize("bias", [False, True])
def test_M_predict_gradient_matches_finite_differences_2d(bias):
    rbf = RBF_interpol(_THETA_2D, _SIGMA, _DATA_Z_2D, bias=bias, kernel="SE")
    rng = np.random.default_rng(2)
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))

    z0 = jnp.array([0.55, 0.35])
    f = lambda c: rbf.predict(z0, c)
    g_auto = np.asarray(jax.grad(f)(coeff))
    g_fd = _central_fd_grad(f, coeff, _FD_H_FP64)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


def test_M_predict_gradient_through_batched_z_matches_finite_differences_smoke():
    """calc_l2_training / calc_w1_training vmap predict() over multiple z and
    differentiate the result -- check that composition, not just a single
    scalar-z call. Always-on, fp32-safe version."""
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, kernel="SE")
    rng = np.random.default_rng(3)
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))
    z_batch = jnp.linspace(0.2, 2.5, 7)
    weights = jnp.asarray(rng.standard_normal(z_batch.shape[0]))

    f = lambda c: jnp.sum(rbf.predict(z_batch, c) * weights)
    g_auto = np.asarray(jax.grad(f)(coeff))
    g_fd = _central_fd_grad(f, coeff, _FD_H_SMOKE)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_SMOKE, rtol=_FD_RTOL_SMOKE)


@_REQUIRE_X64
def test_M_predict_gradient_through_batched_z_matches_finite_differences():
    """calc_l2_training / calc_w1_training vmap predict() over multiple z and
    differentiate the result -- check that composition, not just a single
    scalar-z call."""
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, kernel="SE")
    rng = np.random.default_rng(3)
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))
    z_batch = jnp.linspace(0.2, 2.5, 7)
    weights = jnp.asarray(rng.standard_normal(z_batch.shape[0]))

    f = lambda c: jnp.sum(rbf.predict(z_batch, c) * weights)
    g_auto = np.asarray(jax.grad(f)(coeff))
    g_fd = _central_fd_grad(f, coeff, _FD_H_FP64)
    np.testing.assert_allclose(g_auto, g_fd, atol=_FD_ATOL_FP64, rtol=_FD_RTOL_FP64)


def test_M_predict_gradient_matches_independent_analytic_reimplementation():
    """A second, tolerance-free-ish oracle: grad(predict(z, c)) should equal
    grad of a hand-written re-implementation of the same formula using
    rbf_kernel directly, computed through an entirely separate autodiff trace.
    Not finite-difference-based, so no fp32 noise floor -- runs at any
    precision. Calibration measured exact (0.0) agreement, value and gradient
    both, at both fp32 and fp64."""
    rbf = RBF_interpol(_THETA, _SIGMA, _DATA_Z, bias=True, kernel="SE")
    rng = np.random.default_rng(4)
    coeff = jnp.asarray(rng.standard_normal(rbf.n_param()))
    z0 = 1.3

    def f_lib(c):
        return rbf.predict(z0, c)

    def f_manual(c):
        k_vals = rbf.rbf_kernel(_DATA_Z, jnp.asarray(z0))
        return c[-1] + jnp.sum(c[:-1] * k_vals)

    np.testing.assert_allclose(float(f_lib(coeff)), float(f_manual(coeff)), atol=1e-10)
    g_lib = np.asarray(jax.grad(f_lib)(coeff))
    g_man = np.asarray(jax.grad(f_manual)(coeff))
    np.testing.assert_allclose(g_lib, g_man, atol=1e-10)
