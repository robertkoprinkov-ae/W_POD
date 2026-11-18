"""Pure-plumbing unit tests for ``W_POD.util.RBF_interpol``.

Covers groups G (structural) and G2 (definitional ``predict`` properties) of
``unit_tests.md``.  No interpolation-*accuracy* assertions -- those are the math
tier (some already in ``tests/test_util.py`` for SE / no-bias).

``predict`` and the kernel-scalar functions validate argument shapes with bare
``assert`` statements, so the ``pytest.raises(AssertionError)`` cases here require
pytest to run WITHOUT ``python -O`` (the default).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from W_POD.util import RBF_interpol


# Well-separated, distinct 1-D sample points; theta chosen so the Gram matrix is
# comfortably positive-definite without a nugget (min gap / theta = 1).
_DATA_Z = jnp.array([0.0, 0.5, 1.1, 1.8, 2.6, 3.5])
_THETA = 0.5
_SIGMA = 1.0

_KERNELS = ["SE", "matern32", "matern52"]


def _make(kernel="SE", bias=False, data_z=_DATA_Z, theta=_THETA, sigma=_SIGMA):
    return RBF_interpol(theta, sigma, data_z, bias=bias, kernel=kernel)


# ---------------------------------------------------------------------------
# G34 -- n_param()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_G34_n_param(kernel, bias):
    rbf = _make(kernel=kernel, bias=bias)
    assert rbf.n_param() == _DATA_Z.shape[0] + (1 if bias else 0)


# ---------------------------------------------------------------------------
# G35 -- _K: shape, symmetry, positive (semi-)definiteness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", _KERNELS)
def test_G35_K_shape_symmetric_psd(kernel):
    n = _DATA_Z.shape[0]
    K = np.asarray(_make(kernel=kernel)._K)

    assert K.shape == (n, n)
    np.testing.assert_allclose(K, K.T, rtol=0, atol=1e-6)          # symmetric

    evals = np.linalg.eigvalsh(K)
    assert evals.min() >= -1e-5                                     # PSD (tol)
    assert evals.min() > 0                                          # strictly PD here


# ---------------------------------------------------------------------------
# G36 -- get_linear_constraint_matrix()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_G36_linear_constraint_matrix(kernel, bias):
    n = _DATA_Z.shape[0]
    rbf = _make(kernel=kernel, bias=bias)
    A = np.asarray(rbf.get_linear_constraint_matrix())

    assert A.shape == (n, rbf.n_param())
    np.testing.assert_array_equal(A[:, :n], np.asarray(rbf._K))
    if bias:
        assert A.shape[1] == n + 1
        np.testing.assert_array_equal(A[:, n], np.ones(n))
    else:
        assert A.shape[1] == n


# ---------------------------------------------------------------------------
# G37 -- get_monotonic_linear_constraints_matrix()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_G37_monotonic_constraint_matrix(kernel, bias):
    n = _DATA_Z.shape[0]
    rbf = _make(kernel=kernel, bias=bias)
    A = np.asarray(rbf.get_monotonic_linear_constraints_matrix())
    K = np.asarray(rbf._K)

    assert A.shape == (n - 1, rbf.n_param())
    np.testing.assert_array_equal(A[:, :n], K[1:, :] - K[:-1, :])
    if bias:
        np.testing.assert_array_equal(A[:, n:], np.zeros((n - 1, 1)))


# ---------------------------------------------------------------------------
# G38 -- get_coeff(): shape and bias entry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_G38_get_coeff_shape_and_bias_entry(kernel, bias):
    n = _DATA_Z.shape[0]
    rbf = _make(kernel=kernel, bias=bias)
    values = jnp.asarray(np.random.default_rng(38).standard_normal(n))

    coeff = rbf.get_coeff(values)
    assert coeff.shape == (rbf.n_param(),)
    if bias:
        np.testing.assert_allclose(
            float(np.asarray(coeff)[-1]),
            float(np.mean(np.asarray(values))),
            rtol=1e-5, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# G39 -- calc_RKHS_reg()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_G39_calc_RKHS_reg(kernel, bias):
    n = _DATA_Z.shape[0]
    rbf = _make(kernel=kernel, bias=bias)
    coeff = jnp.asarray(np.random.default_rng(39).standard_normal(rbf.n_param()))

    reg = rbf.calc_RKHS_reg(coeff)
    assert np.asarray(reg).shape == ()

    c = np.asarray(coeff)
    K = np.asarray(rbf._K)
    expected = c[:n] @ K @ c[:n]          # bias entry excluded
    np.testing.assert_allclose(float(np.asarray(reg)), float(expected),
                               rtol=1e-4, atol=1e-6)

    if bias:  # result must not depend on the bias coefficient
        bumped = coeff.at[-1].add(123.0)
        np.testing.assert_allclose(
            float(np.asarray(rbf.calc_RKHS_reg(bumped))),
            float(np.asarray(reg)), rtol=1e-5, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# G40 -- kernel selection
# ---------------------------------------------------------------------------

def test_G40_unknown_kernel_raises():
    with pytest.raises(TypeError):
        RBF_interpol(_THETA, _SIGMA, _DATA_Z, kernel="not_a_kernel")


def test_G40_known_kernels_are_distinct():
    rbfs = {k: _make(kernel=k) for k in _KERNELS}

    assert rbfs["SE"].rbf_kernel_scalars.__func__ is \
        RBF_interpol._rbf_kernel_scalars_SE
    assert rbfs["matern32"].rbf_kernel_scalars.__func__ is \
        RBF_interpol._rbf_kernel_scalars_matern32
    assert rbfs["matern52"].rbf_kernel_scalars.__func__ is \
        RBF_interpol._rbf_kernel_scalars_matern52

    Ks = [np.asarray(r._K) for r in rbfs.values()]
    assert not np.allclose(Ks[0], Ks[1])
    assert not np.allclose(Ks[0], Ks[2])
    assert not np.allclose(Ks[1], Ks[2])


# ---------------------------------------------------------------------------
# G41 -- dim_z inference
# ---------------------------------------------------------------------------

def test_G41_dim_z_inference():
    r1 = RBF_interpol(_THETA, _SIGMA, jnp.array([0.0, 1.0, 2.0]))
    assert r1.dim_z == 1
    assert r1._K.shape == (3, 3)

    dz2 = jnp.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [0.5, 2.0]])
    r2 = RBF_interpol(_THETA, _SIGMA, dz2)
    assert r2.dim_z == 2
    assert r2._K.shape == (4, 4)


# ---------------------------------------------------------------------------
# G42 -- predict() rejects a wrong-length coeff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bias", [False, True])
def test_G42_predict_wrong_coeff_length_raises(bias):
    rbf = _make(bias=bias)
    good = rbf.n_param()
    with pytest.raises(AssertionError):
        rbf.predict(0.5, jnp.ones(good + 1))
    with pytest.raises(AssertionError):
        rbf.predict(0.5, jnp.ones(good - 1))


# ===========================================================================
# G2 -- definitional properties of predict()
# ===========================================================================

def _coeff(rbf, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(rbf.n_param()))


@pytest.mark.parametrize("bias", [False, True])
def test_G43_predict_output_shapes(bias):
    rbf = _make(bias=bias)
    coeff = _coeff(rbf, 43)

    val = rbf.predict(0.5, coeff)
    assert np.asarray(val).shape == ()
    v, var = rbf.predict(0.5, coeff, variances=True)
    assert np.asarray(v).shape == ()
    assert np.asarray(var).shape == ()

    zz = jnp.linspace(0.0, 3.0, 7)
    val_v = rbf.predict(zz, coeff)
    assert val_v.shape == (7,)
    v_v, var_v = rbf.predict(zz, coeff, variances=True)
    assert v_v.shape == (7,)
    assert var_v.shape == (7,)


@pytest.mark.parametrize("bias", [False, True])
def test_G44_predict_matches_definition(bias):
    n = _DATA_Z.shape[0]
    rbf = _make(bias=bias)
    coeff = _coeff(rbf, 44)
    c = np.asarray(coeff)

    for z in (0.0, 0.5, 1.3, 2.9, -0.5, 4.0):
        got = float(np.asarray(rbf.predict(float(z), coeff)))
        # k(data_z, z) via rbf_kernel (vectorised in the first argument only)
        k_vec = np.asarray(rbf.rbf_kernel(_DATA_Z, jnp.asarray(float(z))))
        expected = (c[-1] if bias else 0.0) + float(c[:n] @ k_vec)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


def test_G45_predict_linear_in_coeff_no_bias():
    n = _DATA_Z.shape[0]
    rbf = _make(bias=False)
    rng = np.random.default_rng(45)
    a = jnp.asarray(rng.standard_normal(n))
    b = jnp.asarray(rng.standard_normal(n))
    zz = jnp.linspace(-0.5, 3.5, 9)

    pa = np.asarray(rbf.predict(zz, a))
    pb = np.asarray(rbf.predict(zz, b))
    np.testing.assert_allclose(np.asarray(rbf.predict(zz, a + b)), pa + pb,
                               rtol=1e-4, atol=1e-6)

    s = 2.5
    np.testing.assert_allclose(np.asarray(rbf.predict(zz, s * a)), s * pa,
                               rtol=1e-4, atol=1e-6)


def test_G46_predict_zero_coeff():
    n = _DATA_Z.shape[0]
    zz = jnp.linspace(0.0, 3.0, 5)

    rbf = _make(bias=False)
    np.testing.assert_array_equal(np.asarray(rbf.predict(0.7, jnp.zeros(n))), 0.0)
    np.testing.assert_array_equal(
        np.asarray(rbf.predict(zz, jnp.zeros(n))), np.zeros(5)
    )

    rbf_b = _make(bias=True)
    coeff = jnp.zeros(n + 1).at[-1].set(3.5)     # kernel weights 0, bias = 3.5
    np.testing.assert_allclose(np.asarray(rbf_b.predict(0.7, coeff)), 3.5,
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(rbf_b.predict(zz, coeff)),
                               np.full(5, 3.5), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bias", [False, True])
def test_G47_predict_vectorized_matches_scalar(bias):
    rbf = _make(bias=bias)
    coeff = _coeff(rbf, 47)
    zz = jnp.array([-0.3, 0.0, 0.6, 1.9, 3.2])

    vec = np.asarray(rbf.predict(zz, coeff))
    vec_v, vec_var = rbf.predict(zz, coeff, variances=True)
    vec_v, vec_var = np.asarray(vec_v), np.asarray(vec_var)

    for k in range(zz.shape[0]):
        zk = float(zz[k])
        np.testing.assert_allclose(
            vec[k], float(np.asarray(rbf.predict(zk, coeff))),
            rtol=1e-5, atol=1e-6,
        )
        sv, svar = rbf.predict(zk, coeff, variances=True)
        np.testing.assert_allclose(vec_v[k], float(np.asarray(sv)),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(vec_var[k], float(np.asarray(svar)),
                                   rtol=1e-4, atol=1e-6)
