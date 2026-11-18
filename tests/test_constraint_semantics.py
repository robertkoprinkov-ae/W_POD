"""Math-tier tests for constraint SEMANTICS (group O in unit_tests.md).

Group C (tests/test_modes_plumbing.py) only checks get_constraints()'s
STRUCTURE: matrix shape, column offsets, zero regions, bound signs, row
counts. It never checks what the constraint actually MEANS. This file closes
that gap: for a known coefficient vector `c`, the amplitude constraint block
should satisfy

    A_amplitude @ full_params == RBF.get_linear_constraint_matrix() @ c
                              == RBF.predict(RBF.data_z, c)

i.e. the constraint's bracket really constrains the amplitude interpolant's
own predicted values at the snapshot z's -- not an arbitrary linear
combination that merely happens to have the right shape. Similarly, the
monotonic-gamma block should satisfy

    A_monotonic @ full_params == RBF_gamma.get_monotonic_linear_constraints_matrix() @ c
                              == diff(RBF_gamma.predict(RBF_gamma.data_z, c))

Everything here is exact linear algebra (predict() is linear in coeff, see
util.py) -- no RBF-fitting noise, no interpolation, so a tight,
precision-independent tolerance is used throughout (measured fp64 agreement
~1e-15/1e-16, fp32 ~1e-6/1e-8; atol=1e-4 gives large headroom at both
without needing to branch on precision).

No DataCreation needed, extends G36/G37 (which already established the
underlying algebraic facts about get_linear_constraint_matrix() and
get_monotonic_linear_constraints_matrix() in isolation, from RBF_interpol
alone) up to the mode level.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from W_POD.modes import GlobalMode, TransportedMode, StretchedMode
from W_POD.util import RBF_interpol

_ATOL = 1e-4


def _rbf(n_z, bias=False):
    return RBF_interpol(1.0, 1.0, jnp.linspace(0.0, 1.0, n_z), bias=bias)


# ---------------------------------------------------------------------------
# Fixtures: one case per mode type, each carrying the pieces needed to build
# a random parameter vector and identify which RBF is amplitude-constrained.
# ---------------------------------------------------------------------------

class _Case:
    def __init__(self, mode, layout, amplitude_rbf, amplitude_coeff_name):
        self.mode = mode
        self.layout = layout  # [(name, length), ...] in dict_to_arr order
        self.amplitude_rbf = amplitude_rbf
        self.amplitude_coeff_name = amplitude_coeff_name


def _random_params(case, seed):
    rng = np.random.default_rng(seed)
    d = {name: jnp.asarray(rng.standard_normal(length)) for name, length in case.layout}
    return case.mode.dict_to_arr(d), d


def _build_cases():
    x = jnp.linspace(0.0, 1.0, 11)
    cases = {}

    alpha_rbf = _rbf(5)
    cases["GlobalMode"] = _Case(
        mode=GlobalMode(x, alpha_rbf),
        layout=[("pod_global", 11), ("alpha_coeff", 5)],
        amplitude_rbf=alpha_rbf,
        amplitude_coeff_name="alpha_coeff",
    )

    beta_rbf_t = _rbf(5)
    cases["TransportedMode"] = _Case(
        mode=TransportedMode(x, beta_rbf_t, _rbf(7)),
        layout=[("pod_local", 11), ("beta_coeff", 5), ("gamma_coeff", 7)],
        amplitude_rbf=beta_rbf_t,
        amplitude_coeff_name="beta_coeff",
    )

    beta_rbf_s = _rbf(4)
    cases["StretchedMode"] = _Case(
        mode=StretchedMode(x, beta_rbf_s, _rbf(6), _rbf(3)),
        layout=[("pod_local", 11), ("beta_coeff", 4),
                ("gamma_shift_coeff", 6), ("gamma_stretch_coeff", 3)],
        amplitude_rbf=beta_rbf_s,
        amplitude_coeff_name="beta_coeff",
    )
    return cases


_CASES = _build_cases()


@pytest.fixture(params=list(_CASES), ids=list(_CASES))
def case(request):
    return _CASES[request.param]


# ---------------------------------------------------------------------------
# Amplitude constraint: A @ full_params == RBF.predict(RBF.data_z, coeff)
# ---------------------------------------------------------------------------

def test_O_amplitude_constraint_matrix_reproduces_predict_at_data_z(case):
    """The amplitude constraint row values are exactly the RBF's own
    predicted amplitude at each training z -- not just a same-shaped matrix
    product. Measured max|diff|: ~1e-16 (fp64) / ~6e-8 (fp32) -- pure
    floating-point roundoff, since predict() is linear in coeff and this is
    literally the same computation via two paths."""
    params, d = _random_params(case, seed=1)
    A, lb, ub = case.mode.get_constraints()

    lhs = np.asarray(A @ params)[:case.mode.n_z]
    rhs = np.asarray(case.amplitude_rbf.predict(
        case.amplitude_rbf.data_z, d[case.amplitude_coeff_name]))

    np.testing.assert_allclose(lhs, rhs, atol=_ATOL)


# Live source values, captured at implementation time -- see modes.py's
# get_constraints() for each mode. Deliberately NOT unified into one number:
# GlobalMode and StretchedMode both cap at 100 but use different lower
# bounds (0.1 / 0.5), while TransportedMode's upper bound is +inf, not 100
# (an easy thing to assume is uniform across modes and be wrong about).
_AMPLITUDE_BOUNDS = {
    "GlobalMode": (0.1, 100.0),
    "TransportedMode": (0.1, np.inf),
    "StretchedMode": (0.5, 100.0),
}


def test_O_amplitude_constraint_bounds_match_source(case):
    """Pins the CURRENT literal amplitude-constraint bounds per mode type --
    a regression tripwire, since these directly govern optimizer
    feasibility and are easy to change silently in a refactor."""
    _, lb, ub = case.mode.get_constraints()
    lb_expected, ub_expected = _AMPLITUDE_BOUNDS[case.mode.get_mode_name()]

    np.testing.assert_allclose(np.asarray(lb)[:case.mode.n_z], lb_expected)
    np.testing.assert_allclose(np.asarray(ub)[:case.mode.n_z], ub_expected)


def test_O_amplitude_constraint_correctly_flags_out_of_range_prediction(case):
    """Semantic check: pin the amplitude RBF (via bias=True + a constant
    target -- the exact-coefficient trick, see unit_tests.md group N) to a
    value strictly BELOW the mode's own lower bound, and confirm the
    constraint row correctly reports that violation; pin it to a value
    comfortably inside the bracket and confirm it's reported as satisfied.
    Demonstrates the bound is a real, meaningful check on the amplitude
    interpolant, not a vacuous shape-only formality."""
    x = jnp.linspace(0.0, 1.0, 11)
    lb_val, ub_val = _AMPLITUDE_BOUNDS[case.mode.get_mode_name()]
    below = lb_val - 0.02

    amp_rbf = RBF_interpol(1.0, 1.0, case.amplitude_rbf.data_z, bias=True)
    mode_name = case.mode.get_mode_name()
    pod_len = 11

    if mode_name == "GlobalMode":
        mode = GlobalMode(x, amp_rbf)
        build = lambda coeff: mode.dict_to_arr({"pod_global": jnp.zeros(pod_len), "alpha_coeff": coeff})
    elif mode_name == "TransportedMode":
        gamma_rbf = _rbf(7)
        mode = TransportedMode(x, amp_rbf, gamma_rbf)
        gamma_coeff = gamma_rbf.get_coeff(jnp.zeros(7))
        build = lambda coeff: mode.dict_to_arr({
            "pod_local": jnp.zeros(pod_len), "beta_coeff": coeff, "gamma_coeff": gamma_coeff,
        })
    else:  # StretchedMode
        shift_rbf, stretch_rbf = _rbf(6), _rbf(3)
        mode = StretchedMode(x, amp_rbf, shift_rbf, stretch_rbf)
        shift_coeff = shift_rbf.get_coeff(jnp.zeros(6))
        stretch_coeff = stretch_rbf.get_coeff(jnp.ones(3))
        build = lambda coeff: mode.dict_to_arr({
            "pod_local": jnp.zeros(pod_len), "beta_coeff": coeff,
            "gamma_shift_coeff": shift_coeff, "gamma_stretch_coeff": stretch_coeff,
        })

    A, lb, ub = mode.get_constraints()
    lb_np, ub_np = np.asarray(lb)[:mode.n_z], np.asarray(ub)[:mode.n_z]

    coeff_below = amp_rbf.get_coeff(below * jnp.ones(amp_rbf.data_z.shape[0]))
    row_below = np.asarray(A @ build(coeff_below))[:mode.n_z]
    assert not (np.all(row_below >= lb_np) and np.all(row_below <= ub_np))
    np.testing.assert_allclose(row_below, below, atol=_ATOL)

    within = (lb_val + 1.0) if np.isfinite(ub_val) else (lb_val + 1.0)
    coeff_within = amp_rbf.get_coeff(within * jnp.ones(amp_rbf.data_z.shape[0]))
    row_within = np.asarray(A @ build(coeff_within))[:mode.n_z]
    assert np.all(row_within >= lb_np) and np.all(row_within <= ub_np)


# ---------------------------------------------------------------------------
# Monotonic-gamma constraint (TransportedMode only):
# A_monotonic @ full_params == diff(RBF_gamma.predict(RBF_gamma.data_z, coeff))
# ---------------------------------------------------------------------------

def _transported_equal_rbf(n_pts, **kwargs):
    x = jnp.linspace(0.0, 1.0, 9)
    return TransportedMode(x, _rbf(n_pts), _rbf(n_pts), **kwargs)


@pytest.mark.parametrize("flag", ["gamma_increasing", "gamma_decreasing"])
def test_O_monotonic_gamma_constraint_reproduces_diff_predict(flag):
    """The appended monotonic-constraint rows are exactly the successive
    differences of the gamma RBF's own predicted values at the training z's
    -- not just a same-shaped matrix product."""
    n = 6
    mode = _transported_equal_rbf(n, **{flag: True})
    gamma_rbf = mode._RBF_gamma

    rng = np.random.default_rng(3)
    gamma_coeff = jnp.asarray(rng.standard_normal(n))
    beta_coeff = jnp.asarray(rng.standard_normal(n))
    pod_local = jnp.asarray(rng.standard_normal(9))
    params = mode.dict_to_arr({
        "pod_local": pod_local, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
    })

    A, _, _ = mode.get_constraints()
    lhs = np.asarray(A @ params)[n:]
    predict_vals = np.asarray(gamma_rbf.predict(gamma_rbf.data_z, gamma_coeff))
    rhs = np.diff(predict_vals)

    np.testing.assert_allclose(lhs, rhs, atol=_ATOL)


def test_O_monotonic_gamma_constraint_correctly_flags_wrong_direction():
    """Semantic check: a genuinely increasing gamma(z) trajectory satisfies
    gamma_increasing's bounds and violates gamma_decreasing's; a genuinely
    decreasing trajectory does the reverse. Demonstrates the monotonic
    constraint enforces the actual DIRECTION of gamma(z), not just a
    same-shaped, direction-agnostic formality."""
    n = 6
    gamma_rbf = _rbf(n)
    beta_coeff = jnp.ones(n)
    pod_local = jnp.zeros(9)

    increasing_target = jnp.linspace(0.0, 1.0, n)
    decreasing_target = jnp.linspace(1.0, 0.0, n)
    gamma_coeff_inc = gamma_rbf.get_coeff(increasing_target)
    gamma_coeff_dec = gamma_rbf.get_coeff(decreasing_target)

    def satisfies(mode, gamma_coeff):
        A, lb, ub = mode.get_constraints()
        params = mode.dict_to_arr({
            "pod_local": pod_local, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
        })
        row = np.asarray(A @ params)[n:]
        lb_row, ub_row = np.asarray(lb)[n:], np.asarray(ub)[n:]
        return bool(np.all(row >= lb_row) and np.all(row <= ub_row))

    # _rbf() is deterministic (same theta/sigma/data_z every call), so these
    # modes' own gamma RBF is mathematically identical to the standalone
    # gamma_rbf used to fit the coefficients above.
    mode_inc = _transported_equal_rbf(n, gamma_increasing=True)
    mode_dec = _transported_equal_rbf(n, gamma_decreasing=True)

    assert satisfies(mode_inc, gamma_coeff_inc)
    assert not satisfies(mode_inc, gamma_coeff_dec)
    assert satisfies(mode_dec, gamma_coeff_dec)
    assert not satisfies(mode_dec, gamma_coeff_inc)
